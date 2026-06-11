//! seeds.rs — per-row `rev_swhid` -> root_dir node + rel_path filename-ids (Seeds).
//!
//! Reads the cands parquet (cols `repo`, `commit_id`, `rel_path`, `rev_swhid` (or `id`
//! fallback), `bin`) in FILE ORDER and produces [`Seeds`] in that same row order
//! (the emit pass re-reads in the same order), then:
//!   uniqueness: log how often (repo,commit_id,rel_path) maps to >1 distinct rev_swhid
//!       (expected ~0). The input is winner-selected per row, so each row carries its own
//!       rev_swhid -> resolve per row; this is a sanity log, not a gate.
//!   root: distinct rev_swhid -> node_id(swh:1:rev:<hex>) -> first Directory successor =
//!        root_dir node (memoized). ROOT_MISSING if the rev doesn't resolve or has no root dir.
//!   names: split each row's normalized rel_path -> components -> component_fid lookup ->
//!        path_fids (FID_MISSING for components absent from the archive); path_off via
//!        running offsets (len n_rows + 1).
//!   bin: parse the `bin` string as u16, used for output partitioning.
//!
//! Output: `Seeds { n_rows, root, path_off, path_fids, bin }` in input row order.

use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::RandomState;
use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, LargeStringArray, StringArray};
use dashmap::DashMap;
use rayon::prelude::*;

use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use swh_graph::SWHID;
use swh_graph::graph::{SwhForwardGraph, SwhGraphWithProperties};

use crate::ntf::common::{
    self, FID_MISSING, Names, ROOT_MISSING, Seeds, TY_DIRECTORY,
};

/// Rows per streaming parquet batch (bounded memory).
const BATCH_ROWS: usize = 300_000;

/// Per-row decoded result for one batch (preserves index order via the slot index).
struct RowOut {
    root: u64,
    /// filename_ids for this row's rel_path components (in order).
    fids: Vec<u64>,
    bin: u16,
}

/// Resolve `rev_swhid` string -> root_dir node id, memoized.
///
/// `rev_str` is the raw column value (`swh:1:rev:<hex>` from `rev_swhid`, or a bare
/// `<hex>` from the `id` fallback). Returns `ROOT_MISSING` when the SWHID is malformed,
/// the rev node is absent from the graph, or the rev has no Directory successor.
fn resolve_root<G>(graph: &G, rev_str: &str, memo: &DashMap<String, u64, RandomState>) -> u64
where
    G: SwhForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let rid = common::normalize_rev_id(rev_str);
    // Memoize on the normalized SWHID string (1:1 with the rev).
    if let Some(v) = memo.get(rid.as_ref()) {
        return *v.value();
    }
    let root = (|| {
        let swh = SWHID::try_from(rid.as_ref()).ok()?;
        let rev_node = graph.properties().node_id(swh).ok()?;
        // first Directory successor = the revision's root tree.
        for succ in graph.successors(rev_node) {
            if graph.properties().node_type(succ) as u8 == TY_DIRECTORY {
                return Some(succ as u64);
            }
        }
        None
    })()
    .unwrap_or(ROOT_MISSING);
    memo.insert(rid.into_owned(), root);
    root
}

/// Parse the `bin` string into a u16. Valid bins are 48..=127; strays (2,3,47) keep
/// their parsed value but bump the `strays` counter for logging. Unparseable -> 0
/// (logged via the `bad` counter).
#[inline]
fn parse_bin(s: &str, strays: &AtomicU64, bad: &AtomicU64) -> u16 {
    match s.trim().parse::<u16>() {
        Ok(v) => {
            if !(48..=127).contains(&v) {
                strays.fetch_add(1, Ordering::Relaxed);
            }
            v
        }
        Err(_) => {
            bad.fetch_add(1, Ordering::Relaxed);
            0
        }
    }
}

fn column_as_str_array(arr: &dyn Array) -> Result<&LargeStringArray> {
    if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(s)
    } else if arr.as_any().downcast_ref::<StringArray>().is_some() {
        Err(anyhow!("expected LargeUtf8; please cast input columns to LargeUtf8"))
    } else {
        Err(anyhow!("expected Utf8/LargeUtf8 column"))
    }
}

/// Build per-row [`Seeds`] from `all_cands` (in file/row order).
pub fn build_seeds<G>(
    graph: &G,
    all_cands: &Path,
    names: &Names,
    threads: usize,
) -> Result<Seeds>
where
    G: SwhForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let _ = threads; // rayon global pool is configured by the caller (main).

    // ---- open parquet, discover column indices ----------------------------
    let file = File::open(all_cands).context("open all_cands parquet")?;
    let mut builder =
        ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(BATCH_ROWS);

    let schema = builder.metadata().file_metadata().schema_descr();
    let (mut i_repo, mut i_commit, mut i_rel, mut i_rev, mut i_id, mut i_bin) =
        (None, None, None, None, None, None);
    for (i, col) in schema.columns().iter().enumerate() {
        match col.path().string().as_str() {
            "repo" => i_repo = Some(i),
            "commit_id" => i_commit = Some(i),
            "rel_path" => i_rel = Some(i),
            "rev_swhid" => i_rev = Some(i),
            "id" => i_id = Some(i),
            "bin" => i_bin = Some(i),
            _ => {}
        }
    }
    let i_repo = i_repo.context("all_cands missing `repo` column")?;
    let i_commit = i_commit.context("all_cands missing `commit_id` column")?;
    let i_rel = i_rel.context("all_cands missing `rel_path` column")?;
    let i_bin = i_bin.context("all_cands missing `bin` column")?;
    // rev_swhid is primary; `id` is the fallback for the rev SWHID. Need at least one.
    if i_rev.is_none() && i_id.is_none() {
        return Err(anyhow!("all_cands missing both `rev_swhid` and `id` columns"));
    }

    // Project only the columns we need; we look columns up by NAME on the
    // projected batch so we are robust to ProjectionMask's leaf ordering.
    let mut leaves = vec![i_repo, i_commit, i_rel, i_bin];
    if let Some(r) = i_rev {
        leaves.push(r);
    }
    if let Some(d) = i_id {
        leaves.push(d);
    }
    let projection = ProjectionMask::leaves(builder.parquet_schema(), leaves);
    builder = builder.with_projection(projection);
    let mut reader = builder.build()?;

    // ---- accumulators (final, in row order) -------------------------------
    let mut root: Vec<u64> = Vec::new();
    let mut path_off: Vec<u64> = vec![0]; // len will be n_rows + 1
    let mut path_fids: Vec<u64> = Vec::new();
    let mut bin: Vec<u16> = Vec::new();

    // Memo: normalized rev SWHID string -> root_dir node.
    let root_memo: DashMap<String, u64, RandomState> =
        DashMap::with_hasher(RandomState::new());

    // Uniqueness sanity log (cheap, per-batch only): count (repo,commit_id,rel_path) triples
    // observed with >1 distinct rev_swhid within a batch. Not a gate.
    let a0_multi = AtomicU64::new(0);
    let bin_strays = AtomicU64::new(0);
    let bin_bad = AtomicU64::new(0);
    let mut total_rows: u64 = 0;

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let n = batch.num_rows();
        if n == 0 {
            continue;
        }

        let col = |name: &str| -> Result<&LargeStringArray> {
            let idx = batch
                .schema()
                .index_of(name)
                .with_context(|| format!("projected batch missing `{name}`"))?;
            column_as_str_array(batch.column(idx).as_ref())
        };
        let repo_arr = col("repo")?;
        let commit_arr = col("commit_id")?;
        let rel_arr = col("rel_path")?;
        let bin_arr = col("bin")?;
        // Prefer rev_swhid; fall back to id for the rev SWHID.
        let rev_arr = match (i_rev, i_id) {
            (Some(_), _) => col("rev_swhid")?,
            (None, Some(_)) => col("id")?,
            _ => unreachable!(),
        };

        // ---- uniqueness sanity: per-batch triple -> distinct rev_swhid -----------
        // Cheap probe: a HashMap of triple -> first-seen rev; bump a0_multi on
        // mismatch. Per-batch scope keeps RAM bounded; this is a log, not a gate.
        {
            let mut seen: hashbrown::HashMap<(&str, &str, &str), &str, RandomState> =
                hashbrown::HashMap::with_hasher(RandomState::new());
            for i in 0..n {
                let key = (
                    repo_arr.value(i),
                    commit_arr.value(i),
                    rel_arr.value(i),
                );
                let rev = rev_arr.value(i);
                match seen.entry(key) {
                    hashbrown::hash_map::Entry::Occupied(e) => {
                        if *e.get() != rev {
                            a0_multi.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    hashbrown::hash_map::Entry::Vacant(v) => {
                        v.insert(rev);
                    }
                }
            }
        }

        // ---- per-row decode in parallel into index-ordered slots ----------
        let rows: Vec<RowOut> = (0..n)
            .into_par_iter()
            .map(|i| {
                let rev_str = rev_arr.value(i);
                let root = resolve_root(graph, rev_str, &root_memo);

                // path components -> filename_ids (FID_MISSING when absent).
                let comps = common::path_components(rel_arr.value(i));
                let mut fids = Vec::with_capacity(comps.len());
                for c in &comps {
                    let fid = names
                        .component_fid
                        .get(c.as_slice())
                        .copied()
                        .unwrap_or(FID_MISSING);
                    fids.push(fid);
                }

                let bin = parse_bin(bin_arr.value(i), &bin_strays, &bin_bad);
                RowOut { root, fids, bin }
            })
            .collect();

        // ---- append in order; running prefix-sum for path_off -------------
        root.reserve(n);
        bin.reserve(n);
        for r in rows {
            root.push(r.root);
            bin.push(r.bin);
            path_fids.extend_from_slice(&r.fids);
            path_off.push(path_fids.len() as u64);
        }

        total_rows += n as u64;
    }

    let n_rows = total_rows as usize;
    debug_assert_eq!(root.len(), n_rows);
    debug_assert_eq!(bin.len(), n_rows);
    debug_assert_eq!(path_off.len(), n_rows + 1);

    // ---- diagnostics ------------------------------------------------------
    let missing_roots = root.iter().filter(|&&r| r == ROOT_MISSING).count();
    let n_distinct_revs = root_memo.len();
    eprintln!(
        "seeds: n_rows={n_rows} distinct_revs={n_distinct_revs} root_missing={missing_roots} \
         a0_multi_per_batch={} bin_strays={} bin_unparseable={}",
        a0_multi.load(Ordering::Relaxed),
        bin_strays.load(Ordering::Relaxed),
        bin_bad.load(Ordering::Relaxed),
    );

    Ok(Seeds {
        n_rows,
        root,
        path_off,
        path_fids,
        bin,
    })
}
