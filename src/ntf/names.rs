//! names.rs — build requested_fid + component_fid from rel_path components.
//!
//! Two phases:
//!   STEP 1 (distinct components): stream the `all_cands` parquet `rel_path` column
//!     (a `LargeStringArray`, projected via a `ProjectionMask` onto just that leaf),
//!     split each value into path components (`common::path_components`), and collect
//!     the DISTINCT component byte-strings into one set. Per-batch work is sharded
//!     across rayon workers (thread-local sets merged into the global set). The raw
//!     component count is ~163M but the distinct set is far smaller (mostly short
//!     names like `src`, `main.rs`, `README.md`).
//!   STEP 2 (FCL reverse map): there is NO supported name->id MPH in swh-graph 8.0.4,
//!     so we do ONE forward pass over the label FCL: for every id in
//!     `0..N_LABELS`, fetch `label_name(LabelNameId(id))` and, if those bytes are in
//!     the distinct-component set, record `component_fid[bytes] = id` and mark
//!     `requested.set(id)`. The id range is chunked across rayon workers; the FCL
//!     (`graph.labels.fcl.*`) is staged in /dev/shm so the scan is RAM-speed.

use std::fs::File;
use std::path::Path;

use ahash::RandomState;
use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, LargeStringArray, StringArray};
use hashbrown::HashSet;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;

use swh_graph::graph::SwhGraphWithProperties;
use swh_graph::labels::LabelNameId;

use crate::ntf::common::{self, AMap, FID_MISSING, N_LABELS, Names, RequestedFid, amap};

/// ahash-backed byte-string set used for the distinct-component collection / merge.
type ByteSet = HashSet<Vec<u8>, RandomState>;
fn byte_set() -> ByteSet {
    HashSet::with_hasher(RandomState::new())
}

/// Downcast an arrow column to `LargeStringArray` (all_cands cols are large_string).
fn column_as_str_array(arr: &dyn Array) -> Result<&LargeStringArray> {
    if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(s)
    } else if arr.as_any().downcast_ref::<StringArray>().is_some() {
        Err(anyhow!("expected LargeUtf8 rel_path; please cast input to LargeUtf8"))
    } else {
        Err(anyhow!("expected Utf8/LargeUtf8 rel_path column"))
    }
}

/// STEP 1 — stream `all_cands.rel_path` and collect the DISTINCT path-component bytes.
fn collect_distinct_components(all_cands: &Path) -> Result<ByteSet> {
    let file = File::open(all_cands)
        .with_context(|| format!("open all_cands parquet {}", all_cands.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("open all_cands parquet reader")?
        .with_batch_size(300_000);

    // Project onto just the `rel_path` leaf column.
    let schema = builder.metadata().file_metadata().schema_descr();
    let idx_rel = schema
        .columns()
        .iter()
        .position(|c| c.path().string().as_str() == "rel_path")
        .ok_or_else(|| anyhow!("all_cands parquet has no `rel_path` column"))?;
    let projection = ProjectionMask::leaves(builder.parquet_schema(), vec![idx_rel]);
    let mut reader = builder.with_projection(projection).build()?;

    let mut distinct = byte_set();
    while let Some(batch) = reader.next() {
        let batch = batch?;
        // After projection the (single) projected column lands at index 0.
        let rel = column_as_str_array(batch.column(0).as_ref())?;
        let n = rel.len();

        // Shard the batch across rayon workers; each builds a thread-local set, then
        // we reduce the per-shard sets into one and fold it into the global `distinct`.
        let local: ByteSet = (0..n)
            .into_par_iter()
            .fold(byte_set, |mut acc, i| {
                if rel.is_valid(i) {
                    for comp in common::path_components(rel.value(i)) {
                        acc.insert(comp);
                    }
                }
                acc
            })
            .reduce(byte_set, |mut a, mut b| {
                // Extend the larger set with the smaller to minimise rehash work.
                if a.len() < b.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                a.extend(b);
                a
            });
        if distinct.is_empty() {
            distinct = local;
        } else {
            distinct.extend(local);
        }
    }
    Ok(distinct)
}

/// STEP 2 — one forward FCL pass over `0..N_LABELS`, mapping requested component bytes
/// to their `LabelNameId`. The id range is chunked across rayon workers; each chunk
/// produces a thread-local `(name -> id)` map, which is merged into `component_fid`,
/// and marks `requested` (atomic, so concurrent `.set` is fine).
fn build_reverse_map<G>(
    graph: &G,
    distinct: &ByteSet,
    requested: &RequestedFid,
) -> AMap<Vec<u8>, u64>
where
    G: SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
{
    const CHUNK: u64 = 1 << 20;
    let n_chunks = N_LABELS.div_ceil(CHUNK);

    (0..n_chunks)
        .into_par_iter()
        .fold(amap::<Vec<u8>, u64>, |mut acc, c| {
            let start = c * CHUNK;
            let end = (start + CHUNK).min(N_LABELS);
            for id in start..end {
                let bytes = graph.properties().label_name(LabelNameId(id));
                if distinct.contains(&bytes) {
                    requested.set(id);
                    acc.insert(bytes, id);
                }
            }
            acc
        })
        .reduce(amap::<Vec<u8>, u64>, |mut a, mut b| {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            a.extend(b);
            a
        })
}

/// Build the name-pipeline outputs (`requested` membership + `component_fid` reverse
/// map) from `all_cands.rel_path`. `threads`, when non-zero, sizes the rayon pool used
/// for both phases (best-effort; a global pool may already be installed by the caller).
pub fn build_names<G>(graph: &G, all_cands: &Path, threads: usize) -> Result<Names>
where
    G: SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
{
    let run = || -> Result<Names> {
        // STEP 1: distinct rel_path components.
        let distinct = collect_distinct_components(all_cands)?;

        // STEP 2: filtered forward FCL pass -> reverse map + requested bitvector.
        let requested = RequestedFid::new();
        let component_fid = build_reverse_map(graph, &distinct, &requested);

        // Sanity: any distinct component NOT found in the FCL is permanently
        // unresolvable; it simply has no entry in `component_fid` (seeds maps it to
        // FID_MISSING). Touching the constant keeps the intent documented & the import
        // used regardless of build features.
        debug_assert_ne!(FID_MISSING, 0);

        Ok(Names { requested, component_fid })
    };

    if threads > 0 {
        // Best-effort: run on a sized pool. If a global pool is already installed this
        // builds a nested local pool; either way correctness is unaffected.
        match rayon::ThreadPoolBuilder::new().num_threads(threads).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        }
    } else {
        run()
    }
}
