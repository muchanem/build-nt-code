//! hydrate.rs — the one heavy scan: a K-way EF-split sequential scan of the
//! 4.346 TB `graph-labelled.labels` -> in-RAM `DirMap`.
//!
//! ## Design
//!
//! Goal: produce the CSR map `M: dir_node -> [(filename_id, child_node, type)]`
//! containing exactly `{ (d,fid,child,ty) : reached[d] && requested[fid] }`,
//! with each dir's entries sorted ascending by `fid` so `DirMap::lookup` can
//! binary-search.
//!
//! * Load `graph-labelled.ef` independently: `ef.get(n)` = start
//!   BIT offset of node `n`'s labels in `.labels`; `ef.len()-1 = num_nodes`.
//! * Split `[0, num_nodes)` into `K` contiguous node bands of ~equal LABEL BYTES
//!   by binary-searching the monotonic `ef.get` for the `i*total_bits/K`
//!   boundaries. Workers seek via these boundaries; they NEVER rescan from 0.
//! * `K` workers (rayon scope). Each worker streams its band front-to-back via
//!   `graph.labeled_successors(n)` (random-access per node, but in increasing
//!   `n` it walks `.labels` sequentially). For each reachable dir `n`:
//!     for (child, labels): for EdgeLabel::DirEntry(de):
//!         fid = de.label_name_id().0; if !requested.get(fid) { continue }
//!         ty = graph.properties().node_type(child) as u8;
//!         collect (fid, child, ty) under dir n.
//! * KEY INSIGHT: within a band, nodes are processed in increasing id, so a
//!   dir's entries are emitted CONSECUTIVELY, and bands are in increasing order
//!   -> the global stream is already grouped/sorted by dir node. Each worker
//!   sorts each dir's own entries by fid into a FLAT per-worker shard (no
//!   per-dir heap Vec). Concatenating worker shards in band order yields the
//!   global CSR in dir order — NO global sort needed.
//!
//! ## Memory
//! Worker shards are flat 17 B/entry arrays (`ent_fid`8 + `ent_child`8 + `ent_ty`1),
//! NOT `Vec<(u64,u64,u8)>` (24 B, 8-byte padded) and NOT per-dir Vecs (which would
//! be ~1 B small allocations). The merge pre-reserves the global arrays' address
//! space (virtual; physical pages fault in on write) and copies each worker shard
//! in band order, dropping each worker after copy so its large mmap-backed Vecs
//! return to the OS -> physical peak ~1.06x of the final 17 B/entry CSR (not 2.4x).
//! At a ~0.05 reachable fraction this is ~10 B entries, ~170 GB final / ~180 GB
//! transient; fits a 2 TB node with the ~690 GB resident shm graph.
//! `k` defaults to 16 (mount sweet spot; >16 regresses on a single TCP conn).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};

use swh_graph::NodeType;
use swh_graph::graph::{SwhForwardGraph, SwhGraphWithProperties, SwhLabeledForwardGraph};
use swh_graph::labels::EdgeLabel;

use epserde::deser::{Deserialize, Flags};
use sux::traits::IndexedSeq;
use webgraph::prelude::EF;

use crate::ntf::common::{AMap, DirMap, NodeBitset, RequestedFid, amap};

/// Default number of parallel range-readers (mount sweet spot).
pub const DEFAULT_K: usize = 16;

/// Reachable-dir count above which a full SEQUENTIAL label scan beats per-dir
/// seeking. Below it, seeking to the (few, sparse) reachable dirs is faster; above
/// it the per-dir NFS seek latency dominates and reading the 4.3 TB front-to-back at
/// ~284 MB/s wins (full-scan wins above ~2.4 M dirs).
pub const SEQ_THRESHOLD: u64 = 2_400_000;

/// One worker's flat CSR shard: its reachable dirs (increasing id), each with a
/// per-dir entry `count`, plus the flat fid/child/ty entry arrays (already
/// fid-sorted within each dir). 17 B/entry; no per-dir heap allocation.
struct WorkerShard {
    dirs: Vec<u64>,
    counts: Vec<u32>,
    fid: Vec<u64>,
    child: Vec<u64>,
    ty: Vec<u8>,
}

/// Run the hydrate scan. `k` = number of parallel range-readers
/// ([`DEFAULT_K`] = 16 recommended). Returns the in-RAM [`DirMap`].
pub fn hydrate<G>(
    graph: &G,
    graph_dir: &Path,
    reached: &NodeBitset,
    requested: &RequestedFid,
    k: usize,
) -> Result<DirMap>
where
    G: SwhForwardGraph + SwhLabeledForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let k = k.max(1);

    // --- Load the label bit-offset index ---------------------
    // ef.get(n) = start bit offset of node n's labels in graph-labelled.labels.
    // ef.len()-1 = num_nodes. Monotonic non-decreasing -> binary-searchable.
    // `graph_dir` is the swh-graph BASEPATH (e.g. .../default/graph), so the label
    // offset index is the SIBLING file `<basepath>-labelled.ef` (swh-graph's
    // suffix_path convention) — NOT `graph_dir.join(...)`, which would wrongly treat
    // the basepath's last component ("graph") as a directory.
    let ef_path = {
        let mut s = graph_dir.as_os_str().to_os_string();
        s.push("-labelled.ef");
        std::path::PathBuf::from(s)
    };
    let ef = EF::mmap(&ef_path, Flags::empty())
        .with_context(|| format!("mmap label offset index {}", ef_path.display()))?;
    let ef = &*ef; // MemCase<EF> -> &EF (impls IndexedSeq<Output=usize> + Sync)
    let num_nodes = ef.len() - 1;
    let total_bits = ef.get(num_nodes);
    let total_bytes = (total_bits / 8) as u64;

    // Consistency: the reachability bitset must cover the same node space as the
    // label index, else `reached.is_set(n)` would index out of bounds mid-scan.
    assert_eq!(
        num_nodes,
        reached.len(),
        "label-ef num_nodes ({}) != reachability bitset size ({}); graph/bitset mismatch",
        num_nodes,
        reached.len()
    );
    eprintln!(
        "[hydrate] graph-labelled.ef: num_nodes={num_nodes} total_bits={total_bits} (~{} GiB labels), K={k}",
        total_bytes / (1 << 30)
    );

    // --- Compute K contiguous node-band boundaries of ~equal label BYTES ---
    let mut bounds = vec![0usize; k + 1];
    bounds[k] = num_nodes;
    for i in 1..k {
        let target = (total_bits as u128 * i as u128 / k as u128) as usize;
        bounds[i] = first_node_at_bit(ef, num_nodes, target);
    }
    for i in 1..=k {
        if bounds[i] < bounds[i - 1] {
            bounds[i] = bounds[i - 1];
        }
    }

    // --- Choose scan mode by reachable-dir density -------------------------
    let n_reachable = reached.count_set();
    let sequential = n_reachable > SEQ_THRESHOLD;
    eprintln!(
        "[hydrate] reachable_dirs={n_reachable} -> {} scan",
        if sequential {
            "SEQUENTIAL (read full 4.3 TB front-to-back; immune to seek latency)"
        } else {
            "SEEK (sparse reachable set; per-dir random reads)"
        }
    );

    // --- Run K decoder workers (+ K prefetcher threads in sequential mode) --
    // Prefetchers do large buffered read_at() AHEAD of each decoder, warming the
    // shared page cache so the swh-graph mmap decode hits RAM instead of small NFS
    // page faults (the ~3x mmap penalty: ~94 MB/s mmap vs ~284 MB/s O_DIRECT). They
    // only warm cache -> cannot change correctness, only throughput.
    let labels_path = {
        let mut s = graph_dir.as_os_str().to_os_string();
        s.push("-labelled.labels");
        std::path::PathBuf::from(s)
    };
    let worker_pos: Vec<AtomicU64> = (0..k)
        .map(|w| AtomicU64::new((ef.get(bounds[w]) / 8) as u64))
        .collect();
    let done: Vec<AtomicBool> = (0..k).map(|_| AtomicBool::new(false)).collect();

    let worker_outs: Vec<WorkerShard> = std::thread::scope(|s| {
        if sequential {
            for w in 0..k {
                let blo = (ef.get(bounds[w]) / 8) as u64;
                let bhi = (ef.get(bounds[w + 1]) / 8) as u64;
                let pos = &worker_pos[w];
                let dn = &done[w];
                let lp = labels_path.as_path();
                std::thread::Builder::new()
                    .name(format!("prefetch{w}"))
                    .spawn_scoped(s, move || prefetch_band(lp, blo, bhi, pos, dn))
                    .expect("spawn prefetcher");
            }
        }
        let handles: Vec<_> = (0..k)
            .map(|w| {
                let lo = bounds[w];
                let hi = bounds[w + 1];
                let efr = &ef;
                let pos = &worker_pos[w];
                let dn = &done[w];
                std::thread::Builder::new()
                    .name(format!("scan{w}"))
                    .spawn_scoped(s, move || {
                        scan_band(graph, reached, requested, efr, w, lo, hi, sequential, pos, dn)
                    })
                    .expect("spawn decoder")
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("decoder thread panicked"))
            .collect()
    });

    // --- Concatenate worker shards in band order -> global CSR -------------
    // Bands are in increasing node order and each worker's dirs are in
    // increasing dir-id order, so the concatenation is globally dir-sorted with
    // no overlap.
    let num_dirs: usize = worker_outs.iter().map(|o| o.dirs.len()).sum();
    let num_entries: usize = worker_outs.iter().map(|o| o.fid.len()).sum();
    assert!(
        num_dirs <= u32::MAX as usize,
        "reachable_frac too high: num_dirs {} exceeds u32 rank space ({}); DirMap.index value is u32",
        num_dirs,
        u32::MAX
    );
    eprintln!(
        "[hydrate] scan complete: num_dirs={num_dirs} num_entries={num_entries} (~{} GiB CSR entries)",
        (num_entries as u64 * 17) / (1 << 30)
    );

    // Pass 1: index (dir node -> rank) + CSR offsets (small: ~12 B/dir).
    let mut index: AMap<u64, u32> = amap();
    index.reserve(num_dirs);
    let mut offsets: Vec<u64> = Vec::with_capacity(num_dirs + 1);
    offsets.push(0);
    let mut rank: u32 = 0;
    let mut running: u64 = 0;
    for w in &worker_outs {
        for d in 0..w.dirs.len() {
            index.insert(w.dirs[d], rank);
            rank += 1;
            running += w.counts[d] as u64;
            offsets.push(running);
        }
    }
    debug_assert_eq!(offsets.len(), num_dirs + 1);
    debug_assert_eq!(running as usize, num_entries);

    // Pass 2: flat entry arrays. Pre-reserve the full capacity (virtual address
    // space; pages fault in on write), then copy each worker shard in band order
    // and DROP it so its large mmap-backed Vecs return to the OS -> ~1.06x peak.
    let mut ent_fid: Vec<u64> = Vec::with_capacity(num_entries);
    let mut ent_child: Vec<u64> = Vec::with_capacity(num_entries);
    let mut ent_ty: Vec<u8> = Vec::with_capacity(num_entries);
    for w in worker_outs {
        ent_fid.extend_from_slice(&w.fid);
        ent_child.extend_from_slice(&w.child);
        ent_ty.extend_from_slice(&w.ty);
        // `w` dropped here -> frees its fid/child/ty Vecs back to the OS.
    }
    debug_assert_eq!(ent_fid.len(), num_entries);

    Ok(DirMap {
        index,
        offsets,
        ent_fid,
        ent_child,
        ent_ty,
    })
}

/// Scan one contiguous node band `[lo, hi)` front-to-back, collecting reachable
/// dirs' requested entries into a flat per-worker shard. Returns the band's dirs
/// in increasing-id order, each dir's entries sorted ascending by fid (and
/// deduped by fid). Logs this worker's MB/s.
fn scan_band<G, E>(
    graph: &G,
    reached: &NodeBitset,
    requested: &RequestedFid,
    ef: &E,
    worker: usize,
    lo: usize,
    hi: usize,
    sequential: bool,
    decoder_pos: &AtomicU64,
    done: &AtomicBool,
) -> WorkerShard
where
    G: SwhForwardGraph + SwhLabeledForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
    E: IndexedSeq<Output = usize> + Sync,
{
    let t0 = Instant::now();
    let band_bytes = (ef.get(hi).saturating_sub(ef.get(lo)) / 8) as u64;

    let mut dirs: Vec<u64> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();
    let mut fid: Vec<u64> = Vec::new();
    let mut child: Vec<u64> = Vec::new();
    let mut ty: Vec<u8> = Vec::new();
    // Reused scratch buffer for one dir's entries (cleared per dir; NOT taken).
    let mut buf: Vec<(u64, u64, u8)> = Vec::new();

    // Intra-band progress: log ~64 times across the band (every ~few min on the
    // ~4 h sequential scan) so a stuck/slow worker is visible without waiting hours.
    let span = hi.saturating_sub(lo).max(1);
    let report_step = (span / 64).max(1);
    let mut next_report = lo + report_step;
    let lo_bit = ef.get(lo);

    for n in lo..hi {
        // Publish our byte position so the prefetcher stays just ahead (every 64k nodes).
        if (n & 0xFFFF) == 0 {
            decoder_pos.store((ef.get(n) / 8) as u64, Ordering::Relaxed);
        }
        if n >= next_report {
            let done_bytes = (ef.get(n).saturating_sub(lo_bit) / 8) as u64;
            let secs = t0.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "[hydrate][w{worker}] {:.0}% nodes, {:.1} GiB read, {:.0} MB/s, {} dirs",
                100.0 * (n - lo) as f64 / span as f64,
                done_bytes as f64 / (1u64 << 30) as f64,
                (done_bytes as f64 / (1u64 << 20) as f64) / secs,
                dirs.len()
            );
            next_report += report_step;
        }
        let reachable = if sequential {
            // SEQUENTIAL: visit every dir in order so the label read stays front-to-back
            // (non-dirs carry zero labels, so skipping them leaves no byte gap). Process
            // only reachable dirs, but still CONSUME non-reachable dirs' labels below to
            // keep the bitstream read sequential.
            if graph.properties().node_type(n) != NodeType::Directory {
                continue;
            }
            reached.is_set(n)
        } else {
            // SEEK: only touch reachable dirs (fast when the reachable set is sparse).
            if !reached.is_set(n) {
                continue;
            }
            true
        };
        buf.clear();
        for (c, labels) in graph.labeled_successors(n) {
            for l in labels {
                if let EdgeLabel::DirEntry(de) = l {
                    if !reachable {
                        continue; // consume (advance the sequential reader) but skip
                    }
                    let f = de.label_name_id().0;
                    if !requested.get(f) {
                        continue;
                    }
                    let t = graph.properties().node_type(c) as u8;
                    buf.push((f, c as u64, t));
                }
            }
        }
        if !reachable || buf.is_empty() {
            continue;
        }
        // Sort this dir's entries by fid so DirMap::lookup can binary-search.
        buf.sort_by_key(|e| e.0);
        // Strict-unique fids (valid git trees already have unique names; this is
        // multigraph paranoia + shaves a hair of RAM).
        buf.dedup_by_key(|e| e.0);

        dirs.push(n as u64);
        counts.push(buf.len() as u32);
        for &(f, c, t) in &buf {
            fid.push(f);
            child.push(c);
            ty.push(t);
        }
    }

    // Signal our prefetcher to stop (band fully decoded).
    decoder_pos.store((ef.get(hi) / 8) as u64, Ordering::Relaxed);
    done.store(true, Ordering::Relaxed);

    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    let mb_s = (band_bytes as f64 / (1u64 << 20) as f64) / secs;
    eprintln!(
        "[hydrate][w{worker}] band [{lo},{hi}) {} dirs, {} entries, {:.1} GiB in {:.1}s -> {mb_s:.0} MB/s",
        dirs.len(),
        fid.len(),
        band_bytes as f64 / (1u64 << 30) as f64,
        secs
    );

    WorkerShard {
        dirs,
        counts,
        fid,
        child,
        ty,
    }
}

/// Prefetcher: sequentially read `[byte_lo, byte_hi)` of the labels file in large
/// buffered chunks AHEAD of the decoder, warming the shared page cache so the
/// swh-graph mmap decode hits RAM instead of small NFS faults. Throttled to stay
/// within `WINDOW` bytes of `decoder_pos` so the cache footprint is bounded and pages
/// aren't evicted before the decoder reads them. Pure cache warming -> cannot affect
/// correctness, only throughput. Exits on `done` or end-of-band.
fn prefetch_band(
    path: &Path,
    byte_lo: u64,
    byte_hi: u64,
    decoder_pos: &AtomicU64,
    done: &AtomicBool,
) {
    const CHUNK: usize = 16 << 20; // 16 MB buffered reads (good NFS read size)
    const WINDOW: u64 = 256 << 20; // stay <= 256 MB ahead of the decoder
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[hydrate][prefetch] open {} failed: {e}", path.display());
            return;
        }
    };
    let mut buf = vec![0u8; CHUNK];
    let mut pos = byte_lo;
    while pos < byte_hi {
        if done.load(Ordering::Relaxed) {
            break;
        }
        let dp = decoder_pos.load(Ordering::Relaxed);
        if pos > dp.saturating_add(WINDOW) {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        let want = ((byte_hi - pos) as usize).min(CHUNK);
        match file.read_at(&mut buf[..want], pos) {
            Ok(0) => break,
            Ok(n) => pos += n as u64,
            Err(_) => break,
        }
    }
}

/// Smallest node `m in [0, num_nodes]` with `ef.get(m) >= target_bit`, via
/// binary search over the monotonic non-decreasing `ef.get`.
#[inline]
fn first_node_at_bit<E>(ef: &E, num_nodes: usize, target_bit: usize) -> usize
where
    E: IndexedSeq<Output = usize>,
{
    let (mut lo, mut hi) = (0usize, num_nodes);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if ef.get(mid) >= target_bit {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
