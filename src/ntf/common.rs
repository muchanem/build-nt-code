//! common.rs — shared types, loaders, and utilities for the resolver pipeline.
//!
//! Every other `ntf::*` module is implemented against the types and signatures
//! declared here. Do NOT change public type shapes without updating all modules.
//!
//! ## Graph generics — copy this bound block into every fn that takes the forward graph
//! ```ignore
//! fn foo<G>(graph: &G, ...) -> ...
//! where
//!     G: SwhForwardGraph + SwhLabeledForwardGraph + SwhGraphWithProperties + Sync,
//!     <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
//!     <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
//! ```
//! The concrete type returned by [`load_forward`] satisfies all of these.
//!
//! ## Pinned graph facts (export 2025-05-18)
//!   nodes            = 49,903,891,086  (36-bit -> node ids are u64/usize, NOT u32)
//!   labels/filenames =  6,191,838,569  (33-bit -> filename_id is u64,    NOT u32)
//!   dirs 19.21e9 / contents 24.78e9 / revisions 5.09e9
//!   labeled edge label = DirEntry; de.label_name_id() -> LabelNameId(u64); label_name(id)->Vec<u8>
//!   NodeType repr(u8): Content=0, Directory=1, Origin=2, Release=3, Revision=4, Snapshot=5

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use hashbrown::HashMap as HbHashMap;

use swh_graph::graph::{
    SwhForwardGraph, SwhGraphWithProperties, SwhLabeledForwardGraph, SwhUnidirectionalGraph,
    load_full,
};
use swh_graph::mph::DynMphf;

// ---------------------------------------------------------------------------
// Pinned constants
// ---------------------------------------------------------------------------

/// Total node-id space (largest node id + 1). Export 2025-05-18.
pub const N_NODES: usize = 49_903_891_086;
/// Total distinct label-name ids (filename + branch/tag names). Export 2025-05-18.
pub const N_LABELS: u64 = 6_191_838_569;

/// ahash-backed hashbrown map alias used throughout (matches the existing bins).
pub type AMap<K, V> = HbHashMap<K, V, ahash::RandomState>;
pub fn amap<K, V>() -> AMap<K, V> {
    HbHashMap::with_hasher(ahash::RandomState::new())
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

/// Load the **forward-only labelled** graph + maps (node_id/swhid/node_type) +
/// label names. NO transposed graph, NO timestamp/person/content/string props.
/// Forward-only loader (no backward provenance).
///
/// Returned value supports `successors`, `labeled_successors`,
/// `properties().{node_id,swhid,node_type,label_name}`. It is the concrete type
/// the generic module fns are monomorphised over.
pub fn load_forward(
    graph_dir: &Path,
) -> Result<
    impl SwhForwardGraph
    + SwhLabeledForwardGraph
    + SwhGraphWithProperties<
        Maps: swh_graph::properties::Maps,
        LabelNames: swh_graph::properties::LabelNames,
    > + Send
    + Sync
    + 'static,
> {
    let g = SwhUnidirectionalGraph::new(graph_dir)
        .context("open forward graph (SwhUnidirectionalGraph::new)")?
        .init_properties()
        .load_properties(|p| p.load_maps::<DynMphf>())
        .context("load maps (node_id/swhid/node_type)")?
        .load_properties(|p| p.load_label_names())
        .context("load label names")?
        .load_labels()
        .context("load forward labels (graph-labelled.labels + .ef)")?;
    Ok(g)
}

/// Load the FULL bidirectional graph (forward + transposed + all props + labels),
/// for the from-scratch path which needs backward provenance (`predecessors`).
/// (uses swh-graph's `load_full`).
pub fn load_bidirectional(
    graph_dir: &Path,
) -> Result<
    impl swh_graph::graph::SwhForwardGraph
    + swh_graph::graph::SwhBackwardGraph
    + SwhLabeledForwardGraph
    + swh_graph::graph::SwhLabeledBackwardGraph
    + SwhGraphWithProperties<
        Maps: swh_graph::properties::Maps,
        LabelNames: swh_graph::properties::LabelNames,
    > + Send
    + Sync
    + 'static,
> {
    load_full::<DynMphf>(graph_dir.to_path_buf()).context("load_full graph")
}

// ---------------------------------------------------------------------------
// NodeBitset — lock-free node-id bitvector (reachable dirs; also generic use)
// ---------------------------------------------------------------------------

/// One bit per node id, backed by AtomicU64 words so many rayon workers can mark
/// concurrently. Used by `reach` (reachable dirs) and `hydrate` (membership test).
pub struct NodeBitset {
    words: Vec<AtomicU64>,
    n_bits: usize,
}

impl NodeBitset {
    pub fn new(num_nodes: usize) -> Self {
        let n_words = num_nodes.div_ceil(64);
        let mut words = Vec::with_capacity(n_words);
        words.resize_with(n_words, || AtomicU64::new(0));
        Self { words, n_bits: num_nodes }
    }
    /// Set bit; returns true iff THIS call flipped 0->1 (first visitor wins).
    #[inline]
    pub fn test_and_set(&self, node: usize) -> bool {
        let w = node >> 6;
        let mask = 1u64 << (node & 63);
        (self.words[w].fetch_or(mask, Ordering::Relaxed) & mask) == 0
    }
    #[inline]
    pub fn is_set(&self, node: usize) -> bool {
        let w = node >> 6;
        let mask = 1u64 << (node & 63);
        (self.words[w].load(Ordering::Relaxed) & mask) != 0
    }
    pub fn len(&self) -> usize {
        self.n_bits
    }
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }
    /// Count set bits (parallel-friendly popcount over words).
    pub fn count_set(&self) -> u64 {
        use rayon::prelude::*;
        self.words
            .par_iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// RequestedFid — membership bitvector over the filename-id (LabelNameId) space
// ---------------------------------------------------------------------------

/// Bitvector over [0, N_LABELS) marking which filename_ids appear in some
/// requested rel_path component. ~774 MB. Built (parallel) in `names`, read in
/// `hydrate` to drop dir entries whose name is never requested.
pub struct RequestedFid {
    words: Vec<AtomicU64>,
}
impl RequestedFid {
    pub fn new() -> Self {
        let n_words = (N_LABELS as usize).div_ceil(64);
        let mut words = Vec::with_capacity(n_words);
        words.resize_with(n_words, || AtomicU64::new(0));
        Self { words }
    }
    #[inline]
    pub fn set(&self, fid: u64) {
        let i = fid as usize;
        self.words[i >> 6].fetch_or(1u64 << (i & 63), Ordering::Relaxed);
    }
    #[inline]
    pub fn get(&self, fid: u64) -> bool {
        let i = fid as usize;
        (self.words[i >> 6].load(Ordering::Relaxed) & (1u64 << (i & 63))) != 0
    }
    pub fn count_set(&self) -> u64 {
        use rayon::prelude::*;
        self.words
            .par_iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
            .sum()
    }
}
impl Default for RequestedFid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Names — output of the name pipeline
// ---------------------------------------------------------------------------

/// Sentinel filename_id meaning "this rel_path component is not present in the
/// archive's label table" -> the row can never resolve -> traversal yields NotFound.
pub const FID_MISSING: u64 = u64::MAX;

pub struct Names {
    /// Membership over filename-id space (for the hydrate name filter).
    pub requested: RequestedFid,
    /// distinct rel_path component bytes -> filename_id (only for components that
    /// exist in the archive). Used by `seeds` to turn each row's rel_path into fids.
    pub component_fid: AMap<Vec<u8>, u64>,
}

// ---------------------------------------------------------------------------
// Seeds — per-row data the SHARED CORE consumes (output of `seeds`)
// ---------------------------------------------------------------------------

/// Sentinel root meaning "rev_swhid did not resolve to a node / no root dir".
pub const ROOT_MISSING: u64 = u64::MAX;

/// Compact per-row seeds. `root[i]`, the fids `path_fids[path_off[i]..path_off[i+1]]`,
/// and `bin[i]` fully describe row i's traversal. Pass-through output strings
/// (repo/commit_id/rel_path/id/origin_url/rev_swhid) are re-read from the cands
/// parquet at emit time to keep RAM small.
pub struct Seeds {
    pub n_rows: usize,
    /// root_dir node per row (ROOT_MISSING if unresolved).
    pub root: Vec<u64>,
    /// offsets into `path_fids`, len n_rows + 1.
    pub path_off: Vec<u64>,
    /// concatenated filename_ids for each row's rel_path components (FID_MISSING for
    /// components absent from the archive).
    pub path_fids: Vec<u64>,
    /// bin per row, as a small int, for output partitioning.
    pub bin: Vec<u16>,
}
impl Seeds {
    #[inline]
    pub fn path_of(&self, row: usize) -> &[u64] {
        let s = self.path_off[row] as usize;
        let e = self.path_off[row + 1] as usize;
        &self.path_fids[s..e]
    }
    /// Distinct, non-missing root_dir nodes (BFS seeds for `reach`).
    pub fn distinct_roots(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .root
            .iter()
            .filter(|&&r| r != ROOT_MISSING)
            .map(|&r| r as usize)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

// ---------------------------------------------------------------------------
// DirMap — the hydrated in-RAM map (output of hydrate, consumed by traverse)
// ---------------------------------------------------------------------------

/// CSR-like map `dir_node -> [(filename_id, child_node, node_type)]`, containing
/// exactly the entries `{ (d,fid,child,ty) : bitvector[d] && requested_fid[fid] }`.
/// Entries within each dir are sorted by `filename_id` so `lookup` can binary-search.
///
/// `hydrate` constructs this; `index` maps a dir node id to its dense rank, and the
/// three parallel `ent_*` vectors hold the entries grouped by dir in rank order.
pub struct DirMap {
    /// dir node id -> dense rank in [0, num_dirs).
    pub index: AMap<u64, u32>,
    /// CSR row offsets into the `ent_*` arrays, len num_dirs + 1.
    pub offsets: Vec<u64>,
    /// filename_id of each entry (sorted ascending within each dir's slice).
    pub ent_fid: Vec<u64>,
    /// child node id of each entry.
    pub ent_child: Vec<u64>,
    /// NodeType discriminant (u8) of each entry's child (Directory=1, Content=0, Revision=4, ...).
    pub ent_ty: Vec<u8>,
}
impl DirMap {
    /// Resolve one path step: in directory `dir`, find entry named `fid`.
    /// Returns (child_node, node_type_u8) or None if absent.
    #[inline]
    pub fn lookup(&self, dir: u64, fid: u64) -> Option<(u64, u8)> {
        let r = *self.index.get(&dir)? as usize;
        let s = self.offsets[r] as usize;
        let e = self.offsets[r + 1] as usize;
        let slice = &self.ent_fid[s..e];
        match slice.binary_search(&fid) {
            Ok(j) => Some((self.ent_child[s + j], self.ent_ty[s + j])),
            Err(_) => None,
        }
    }
    pub fn num_dirs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn num_entries(&self) -> usize {
        self.ent_fid.len()
    }
}

// ---------------------------------------------------------------------------
// Status — per-row resolution outcome
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Status {
    Ok = 0,
    NotFound = 1,   // a path component was missing in some directory
    Submodule = 2,  // hit a gitlink (Revision target) mid/at path
    DirTarget = 3,  // final component resolved to a Directory, not a Content
    BadRev = 4,     // rev_swhid did not resolve to a node
    NoRoot = 5,     // rev had no root directory successor
}
impl Status {
    /// Map to the 2 output strings (`ok` / `path_not_found`).
    /// Finer statuses are kept internally for run-metadata diagnostics only.
    pub fn output_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            _ => "path_not_found",
        }
    }
    pub fn from_u8(v: u8) -> Status {
        match v {
            0 => Status::Ok,
            1 => Status::NotFound,
            2 => Status::Submodule,
            3 => Status::DirTarget,
            4 => Status::BadRev,
            _ => Status::NoRoot,
        }
    }
}

// NodeType discriminants (mirror swh_graph::NodeType repr(u8)) for entry typing.
pub const TY_CONTENT: u8 = 0;
pub const TY_DIRECTORY: u8 = 1;
pub const TY_REVISION: u8 = 4;

// ---------------------------------------------------------------------------
// String / SWHID utilities
// ---------------------------------------------------------------------------

/// Canonicalize a repo id ("owner/name", optionally with host/scheme/.git) to the
/// origin URL used for `swh:1:ori:` hashing.
pub fn canonical_github_url(repo: &str) -> String {
    let mut r = repo.trim().trim_matches('/').to_string();
    if let Some(rem) = r.strip_prefix("https://github.com/") {
        r = rem.to_string();
    }
    if let Some(rem) = r.strip_prefix("github.com/") {
        r = rem.to_string();
    }
    if let Some(rem) = r.strip_suffix(".git") {
        r = rem.to_string();
    }
    format!("https://github.com/{}", r)
}

/// `swh:1:ori:<sha1(url)>` SWHID for an origin URL.
pub fn origin_swhid_from_url(url: &str) -> swh_graph::SWHID {
    use sha1::{Digest, Sha1};
    let mut h = Sha1::new();
    h.update(url.as_bytes());
    let hex = hex::encode(h.finalize());
    swh_graph::SWHID::try_from(format!("swh:1:ori:{hex}").as_str()).unwrap()
}

/// Normalize a rel_path: trim, strip `./` and leading `/`,
/// convert `\` to `/`. Used for the `path=/<...>` qualifier and component splitting.
pub fn normalize_path(p: &str) -> String {
    let p = p.trim().trim_start_matches("./").trim_start_matches('/');
    p.replace('\\', "/")
}

/// Ensure a rev id string is a full `swh:1:rev:<40hex>` SWHID string.
pub fn normalize_rev_id(id: &str) -> std::borrow::Cow<'_, str> {
    if id.starts_with("swh:1:rev:") {
        std::borrow::Cow::Borrowed(id)
    } else {
        std::borrow::Cow::Owned(format!("swh:1:rev:{}", id))
    }
}

/// Build the `qualified` SWHID string:
/// `<cnt>;origin=<url>;visit=<snp>;anchor=<rev>;path=/<normalized_rel_path>`.
/// `snapshot_swhid` may be empty (when not tracked) -> visit= empty.
pub fn build_qualified(
    cnt_swhid: &str,
    origin_url: &str,
    snapshot_swhid: &str,
    rev_swhid: &str,
    rel_path: &str,
) -> String {
    format!(
        "{};origin={};visit={};anchor={};path=/{}",
        cnt_swhid,
        origin_url,
        snapshot_swhid,
        rev_swhid,
        normalize_path(rel_path)
    )
}

/// Split a (already repo-relative) rel_path into non-empty path components (bytes),
/// after `normalize_path`.
pub fn path_components(rel_path: &str) -> Vec<Vec<u8>> {
    normalize_path(rel_path)
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.as_bytes().to_vec())
        .collect()
}
