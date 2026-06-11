//! scratch.rs — from-scratch candidate resolution: a rev-prefix index (B0) + backward
//! provenance winner selection. Produces a winner-selected cands parquet (same schema as
//! `all_cands`) which the SHARED CORE (names/seeds/reach/hydrate/traverse/emit) then runs on.
//!
//! Needs the BIDIRECTIONAL graph (transposed for provenance via `predecessors`).
//!
//! B0 (rev-prefix index, build once IF `commit_id` is a prefix < 40 hex): scan all nodes in
//!   parallel, filter `node_type == Revision`, key = first `prefix_nibbles` hex chars of
//!   `swhid.hash` ([u8;20]); sharded map prefix-key -> Vec<rev_node>. If `commit_id` is a full
//!   40-hex hash, B0 is skipped and `node_id(swh:1:rev:<hex>)` is used directly.
//! B1 (per (repo, commit_prefix)): origin_node = node_id(swh:1:ori:<sha1(canonical url)>);
//!   skip group on failure. candidate revs = prefix-index lookup (or the single full-hash rev).
//!   Provenance: keep candidates whose backward walk (`predecessors`) reaches a Snapshot in the
//!   origin's snapshot set. Tiebreak Branch-Head > Max-Hex: among provenance-passing
//!   candidates keep branch-heads if any, then max by rev hex string. Winner rev -> first
//!   Directory successor = root_dir (group dropped if absent).
//! Emit cands parquet rows (all_cands schema, LargeStringArray cols): repo, commit_id, rel_path,
//!   id(=rev_swhid), origin_url, rev_swhid, file_path(=rel_path), bin. One row per input row
//!   whose group resolved to a winner (unresolved groups dropped).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;

use arrow::array::{Array, LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::file::properties::WriterProperties;

use swh_graph::graph::{
    NodeId, SwhBackwardGraph, SwhForwardGraph, SwhGraphWithProperties, SwhLabeledForwardGraph,
};
use swh_graph::{NodeType, SWHID};

use crate::ntf::common::{
    AMap, amap, canonical_github_url, normalize_rev_id, origin_swhid_from_url,
};

// ---------------------------------------------------------------------------
// B0 — rev-prefix index
// ---------------------------------------------------------------------------

/// Number of shards for the prefix index. Sharding by the first prefix byte keeps the
/// parallel fold/reduce merge cheap and contention-free.
const NUM_SHARDS: usize = 256;

/// Sharded map: `prefix-hex-string -> Vec<rev_node>`. The key is the first `prefix_nibbles`
/// lowercase hex chars of the revision hash, so it can be looked up directly with a raw
/// `commit_id` prefix string (no odd/even-nibble packing arithmetic).
type Shard = AMap<Box<[u8]>, Vec<NodeId>>;

fn empty_shards() -> Vec<Shard> {
    let mut v = Vec::with_capacity(NUM_SHARDS);
    v.resize_with(NUM_SHARDS, amap);
    v
}

/// Lowercase hex of the first `nibbles` nibbles of a 20-byte hash (clamped to 40).
#[inline]
fn hash_prefix_hex(hash: &[u8; 20], nibbles: usize) -> Box<[u8]> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let n = nibbles.min(40);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let byte = hash[i / 2];
        let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0x0f };
        out.push(HEX[nib as usize]);
    }
    out.into_boxed_slice()
}

/// Shard selector: FNV-1a hash of the FULL prefix key, spread across all NUM_SHARDS.
/// (Keying on the first byte alone would use only 16 of 256 shards, since hex keys
/// start with one of 16 ASCII chars.) Deterministic -> insert and lookup agree.
#[inline]
fn shard_of(key: &[u8]) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h as usize) & (NUM_SHARDS - 1)
}

/// Build the B0 rev-prefix index by scanning ALL nodes in parallel, keeping only Revisions.
/// Keys on the first `prefix_nibbles` hex chars of each revision's hash.
fn build_rev_prefix_index<G>(graph: &G, prefix_nibbles: usize) -> Vec<Shard>
where
    G: SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let num_nodes = graph.num_nodes();
    (0..num_nodes)
        .into_par_iter()
        .fold(empty_shards, |mut shards: Vec<Shard>, node| {
            if graph.properties().node_type(node) == NodeType::Revision {
                let swhid = graph.properties().swhid(node);
                let key = hash_prefix_hex(&swhid.hash, prefix_nibbles);
                let si = shard_of(&key);
                shards[si].entry(key).or_default().push(node);
            }
            shards
        })
        .reduce(empty_shards, |mut a, b| {
            for (sa, sb) in a.iter_mut().zip(b.into_iter()) {
                for (k, mut v) in sb {
                    sa.entry(k).or_default().append(&mut v);
                }
            }
            a
        })
}

/// Look up candidate revision nodes for a commit-prefix string (already lowercase hex).
#[inline]
fn lookup_prefix<'a>(shards: &'a [Shard], prefix: &[u8]) -> Option<&'a [NodeId]> {
    shards[shard_of(prefix)].get(prefix).map(|v| v.as_slice())
}

// ---------------------------------------------------------------------------
// B1 — origin / provenance / tie-break helpers
// ---------------------------------------------------------------------------

/// Snapshot membership set = direct Snapshot successors of the origin node.
fn build_snapshot_set<G>(graph: &G, origin: NodeId) -> std::collections::HashSet<NodeId>
where
    G: SwhForwardGraph + SwhGraphWithProperties,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let mut set = std::collections::HashSet::new();
    for succ in graph.successors(origin) {
        if graph.properties().node_type(succ) == NodeType::Snapshot {
            set.insert(succ);
        }
    }
    set
}

/// Backward BFS rev -> snapshot-in-origin-set. Returns true iff `rev`'s predecessors reach a
/// Snapshot that belongs to this origin (provenance proof).
fn reaches_origin_snapshot<G>(
    graph: &G,
    rev: NodeId,
    snapshot_set: &std::collections::HashSet<NodeId>,
) -> bool
where
    G: SwhBackwardGraph + SwhGraphWithProperties,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    use std::collections::VecDeque;
    let mut q = VecDeque::new();
    let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    q.push_back(rev);
    seen.insert(rev);
    while let Some(n) = q.pop_front() {
        for p in graph.predecessors(n) {
            if graph.properties().node_type(p) == NodeType::Snapshot && snapshot_set.contains(&p) {
                return true;
            }
            if seen.insert(p) {
                q.push_back(p);
            }
        }
    }
    false
}

/// Is `rev` a branch head of `snapshot`? Accepts a Release -> Revision indirection.
fn is_branch_head<G>(graph: &G, snapshot: NodeId, rev: NodeId) -> bool
where
    G: SwhLabeledForwardGraph + SwhForwardGraph + SwhGraphWithProperties,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    for succ in graph.successors(snapshot) {
        match graph.properties().node_type(succ) {
            NodeType::Revision => {
                if succ == rev {
                    return true;
                }
            }
            NodeType::Release => {
                for r in graph.successors(succ) {
                    if graph.properties().node_type(r) == NodeType::Revision && r == rev {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// First Directory successor of a revision (its root tree), if any.
fn root_dir_of_rev<G>(graph: &G, rev: NodeId) -> Option<NodeId>
where
    G: SwhForwardGraph + SwhGraphWithProperties,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    for succ in graph.successors(rev) {
        if graph.properties().node_type(succ) == NodeType::Directory {
            return Some(succ);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Input / output
// ---------------------------------------------------------------------------

/// Loaded input columns (owned strings, parallel vectors of length `n_rows`).
struct Input {
    repo: Vec<String>,
    commit_id: Vec<String>,
    rel_path: Vec<String>,
    bin: Vec<String>,
}

fn read_input(input: &Path) -> Result<Input> {
    let file = File::open(input).context("open input parquet")?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?.with_batch_size(300_000);

    let schema = builder.metadata().file_metadata().schema_descr();
    let mut idx_repo = None;
    let mut idx_commit = None;
    let mut idx_rel = None;
    let mut idx_bin = None;
    let mut proj = Vec::new();
    for (i, col) in schema.columns().iter().enumerate() {
        match col.path().string().as_str() {
            "repo" => {
                idx_repo = Some(i);
                proj.push(i);
            }
            "commit_id" => {
                idx_commit = Some(i);
                proj.push(i);
            }
            "rel_path" => {
                idx_rel = Some(i);
                proj.push(i);
            }
            "bin" => {
                idx_bin = Some(i);
                proj.push(i);
            }
            _ => {}
        }
    }
    let idx_repo = idx_repo.context("missing repo column")?;
    let idx_commit = idx_commit.context("missing commit_id column")?;
    let idx_rel = idx_rel.context("missing rel_path column")?;

    let projection = ProjectionMask::leaves(builder.parquet_schema(), proj);
    let mut reader = builder.with_projection(projection).build()?;

    let mut out = Input {
        repo: Vec::new(),
        commit_id: Vec::new(),
        rel_path: Vec::new(),
        bin: Vec::new(),
    };

    // After projection, columns are re-indexed densely in original column order. Recompute
    // the post-projection position of each requested leaf.
    let mut ordered: Vec<(usize, u8)> = Vec::new(); // (orig_idx, tag) tag: 0 repo 1 commit 2 rel 3 bin
    ordered.push((idx_repo, 0));
    ordered.push((idx_commit, 1));
    ordered.push((idx_rel, 2));
    if let Some(b) = idx_bin {
        ordered.push((b, 3));
    }
    ordered.sort_by_key(|&(i, _)| i);
    let mut pos_repo = 0usize;
    let mut pos_commit = 0usize;
    let mut pos_rel = 0usize;
    let mut pos_bin: Option<usize> = None;
    for (pos, &(_, tag)) in ordered.iter().enumerate() {
        match tag {
            0 => pos_repo = pos,
            1 => pos_commit = pos,
            2 => pos_rel = pos,
            _ => pos_bin = Some(pos),
        }
    }

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let repo_arr = column_as_str_array(batch.column(pos_repo).as_ref())?;
        let commit_arr = column_as_str_array(batch.column(pos_commit).as_ref())?;
        let rel_arr = column_as_str_array(batch.column(pos_rel).as_ref())?;
        let bin_arr = match pos_bin {
            Some(p) => Some(column_as_str_array(batch.column(p).as_ref())?),
            None => None,
        };
        for i in 0..batch.num_rows() {
            out.repo.push(repo_arr.value(i).to_string());
            out.commit_id.push(commit_arr.value(i).to_string());
            out.rel_path.push(rel_arr.value(i).to_string());
            out.bin.push(match bin_arr {
                Some(a) => a.value(i).to_string(),
                None => String::new(),
            });
        }
    }
    Ok(out)
}

fn column_as_str_array(arr: &dyn Array) -> Result<&LargeStringArray> {
    if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(s)
    } else if arr.as_any().downcast_ref::<StringArray>().is_some() {
        Err(anyhow!(
            "expected LargeUtf8; please cast input columns to LargeUtf8"
        ))
    } else {
        Err(anyhow!("expected Utf8/LargeUtf8 column"))
    }
}

/// One resolved output row (all_cands schema).
struct OutRow {
    repo: String,
    commit_id: String,
    rel_path: String,
    rev_swhid: String,
    origin_url: String,
    bin: String,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Resolve raw `(repo, commit_prefix, rel_path)` metadata at `input` into a winner-selected
/// cands parquet at `cands_out` (all_cands schema). `prefix_nibbles` = commit-prefix hex
/// length (e.g. 7). Uses backward provenance on the bidirectional graph.
pub fn resolve_candidates<G>(
    graph: &G,
    input: &Path,
    cands_out: &Path,
    prefix_nibbles: usize,
    threads: usize,
) -> Result<()>
where
    G: SwhForwardGraph
        + SwhBackwardGraph
        + SwhLabeledForwardGraph
        + SwhGraphWithProperties
        + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
    <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
{
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let inp = read_input(input).context("read input parquet")?;
    let n_rows = inp.repo.len();

    // Decide whether commit_id is a full 40-hex hash (skip B0) or a prefix (build B0).
    let full_hash = !inp.commit_id.is_empty()
        && inp
            .commit_id
            .iter()
            .all(|c| c.len() == 40 && c.bytes().all(|b| b.is_ascii_hexdigit()));

    // B0: build the rev-prefix index only when commit_id is a prefix shorter than full.
    let index: Option<Vec<Shard>> = if full_hash {
        None
    } else {
        Some(build_rev_prefix_index(graph, prefix_nibbles))
    };

    // Group input row indices by (repo, commit_id).
    let mut groups: AMap<(&str, &str), Vec<usize>> = amap();
    for i in 0..n_rows {
        groups
            .entry((inp.repo[i].as_str(), inp.commit_id[i].as_str()))
            .or_default()
            .push(i);
    }

    // B1: resolve each group in parallel into a winner; emit rows for resolved groups.
    let groups_vec: Vec<((&str, &str), Vec<usize>)> = groups.into_iter().collect();
    let resolved: Vec<Vec<OutRow>> = groups_vec
        .into_par_iter()
        .filter_map(|((repo_key, commit_key), idxs)| {
            resolve_group(
                graph,
                index.as_deref(),
                full_hash,
                prefix_nibbles,
                repo_key,
                commit_key,
                &idxs,
                &inp,
            )
        })
        .collect();

    write_cands(cands_out, resolved)
}

/// Resolve one (repo, commit_id) group to a winner revision and emit its rows.
#[allow(clippy::too_many_arguments)]
fn resolve_group<G>(
    graph: &G,
    index: Option<&[Shard]>,
    full_hash: bool,
    prefix_nibbles: usize,
    repo_key: &str,
    commit_key: &str,
    idxs: &[usize],
    inp: &Input,
) -> Option<Vec<OutRow>>
where
    G: SwhForwardGraph
        + SwhBackwardGraph
        + SwhLabeledForwardGraph
        + SwhGraphWithProperties
        + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
    <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
{
    // Origin node from the canonical GitHub URL; skip group on failure.
    let canonical_url = canonical_github_url(repo_key);
    let origin_swhid = origin_swhid_from_url(&canonical_url);
    let origin_node = graph.properties().node_id(origin_swhid).ok()?;
    let snapshot_set = build_snapshot_set(graph, origin_node);
    if snapshot_set.is_empty() {
        return None;
    }

    // Candidate revision nodes.
    let mut cand_nodes: Vec<NodeId> = Vec::new();
    if full_hash {
        // Direct: node_id(swh:1:rev:<hex>) for the full hash.
        let rid = normalize_rev_id(commit_key);
        if let Ok(swhid) = SWHID::try_from(rid.as_ref()) {
            if let Ok(node) = graph.properties().node_id(swhid) {
                cand_nodes.push(node);
            }
        }
    } else {
        // Prefix index lookup. Lowercase the commit prefix and clamp to prefix_nibbles.
        let key: Vec<u8> = commit_key
            .bytes()
            .take(prefix_nibbles)
            .map(|b| b.to_ascii_lowercase())
            .collect();
        if let Some(idx) = index {
            if let Some(nodes) = lookup_prefix(idx, &key) {
                cand_nodes.extend_from_slice(nodes);
            }
        }
    }
    if cand_nodes.is_empty() {
        return None;
    }
    cand_nodes.sort_unstable();
    cand_nodes.dedup();

    // Provenance filter: keep candidates whose backward walk reaches an origin snapshot.
    let mut passing: Vec<NodeId> = cand_nodes
        .into_iter()
        .filter(|&rev| reaches_origin_snapshot(graph, rev, &snapshot_set))
        .collect();
    if passing.is_empty() {
        return None;
    }

    // Tie-break: Branch-Head > Max-Hex. Among provenance-passing candidates keep branch-heads
    // if any exist, then pick the lexicographically-max rev hex string.
    passing.sort_unstable();
    let winner_rev = if passing.len() == 1 {
        passing[0]
    } else {
        let mut heads: Vec<NodeId> = Vec::new();
        for &snp in &snapshot_set {
            for &rev in &passing {
                if is_branch_head(graph, snp, rev) {
                    heads.push(rev);
                }
            }
        }
        heads.sort_unstable();
        heads.dedup();
        let pool: &[NodeId] = if heads.is_empty() { &passing } else { &heads };
        // Max by revision hex string.
        *pool
            .iter()
            .max_by(|&&a, &&b| {
                let ha = graph.properties().swhid(a).to_string();
                let hb = graph.properties().swhid(b).to_string();
                ha.cmp(&hb)
            })
            .unwrap()
    };

    // Winner must have a root directory successor.
    root_dir_of_rev(graph, winner_rev)?;

    let rev_swhid = graph.properties().swhid(winner_rev).to_string();

    // Emit one row per input row in this group.
    let mut rows = Vec::with_capacity(idxs.len());
    for &i in idxs {
        rows.push(OutRow {
            repo: inp.repo[i].clone(),
            commit_id: inp.commit_id[i].clone(),
            rel_path: inp.rel_path[i].clone(),
            rev_swhid: rev_swhid.clone(),
            origin_url: canonical_url.clone(),
            bin: inp.bin[i].clone(),
        });
    }
    Some(rows)
}

/// Write the resolved rows as a single all_cands-schema parquet with LargeStringArray columns
/// (so the shared core's LargeStringArray readers work). Columns: repo, commit_id, rel_path,
/// id(=rev_swhid), origin_url, rev_swhid, file_path(=rel_path), bin.
fn write_cands(cands_out: &Path, resolved: Vec<Vec<OutRow>>) -> Result<()> {
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("repo", DataType::LargeUtf8, false),
        Field::new("commit_id", DataType::LargeUtf8, false),
        Field::new("rel_path", DataType::LargeUtf8, false),
        Field::new("id", DataType::LargeUtf8, false),
        Field::new("origin_url", DataType::LargeUtf8, false),
        Field::new("rev_swhid", DataType::LargeUtf8, false),
        Field::new("file_path", DataType::LargeUtf8, false),
        Field::new("bin", DataType::LargeUtf8, false),
    ]));

    let mut repo: Vec<String> = Vec::new();
    let mut commit_id: Vec<String> = Vec::new();
    let mut rel_path: Vec<String> = Vec::new();
    let mut id: Vec<String> = Vec::new();
    let mut origin_url: Vec<String> = Vec::new();
    let mut rev_swhid: Vec<String> = Vec::new();
    let mut file_path: Vec<String> = Vec::new();
    let mut bin: Vec<String> = Vec::new();

    for group in resolved {
        for r in group {
            repo.push(r.repo);
            commit_id.push(r.commit_id);
            rel_path.push(r.rel_path.clone());
            id.push(r.rev_swhid.clone());
            origin_url.push(r.origin_url);
            rev_swhid.push(r.rev_swhid);
            file_path.push(r.rel_path);
            bin.push(r.bin);
        }
    }

    let props = WriterProperties::builder().build();
    let f = File::create(cands_out).context("create cands_out parquet")?;
    let mut writer = ArrowWriter::try_new(f, out_schema.clone(), Some(props))?;

    // Chunk the write to bound peak memory and avoid oversized single batches.
    const CHUNK: usize = 1_000_000;
    let total = repo.len();
    let mut start = 0usize;
    while start < total {
        let end = (start + CHUNK).min(total);
        let batch = RecordBatch::try_new(
            out_schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(repo[start..end].to_vec())),
                Arc::new(LargeStringArray::from(commit_id[start..end].to_vec())),
                Arc::new(LargeStringArray::from(rel_path[start..end].to_vec())),
                Arc::new(LargeStringArray::from(id[start..end].to_vec())),
                Arc::new(LargeStringArray::from(origin_url[start..end].to_vec())),
                Arc::new(LargeStringArray::from(rev_swhid[start..end].to_vec())),
                Arc::new(LargeStringArray::from(file_path[start..end].to_vec())),
                Arc::new(LargeStringArray::from(bin[start..end].to_vec())),
            ],
        )?;
        writer.write(&batch)?;
        start = end;
    }
    // Ensure an (empty) file is still produced when there are zero resolved rows.
    if total == 0 {
        let empty: Vec<String> = Vec::new();
        let batch = RecordBatch::try_new(
            out_schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty.clone())),
                Arc::new(LargeStringArray::from(empty)),
            ],
        )?;
        writer.write(&batch)?;
    }
    writer.close()?;
    Ok(())
}
