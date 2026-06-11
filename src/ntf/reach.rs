//! reach.rs — label-free forward BFS dir->dir reachability bitvector.
//!
//! Seeds = the distinct winner root_dir nodes. Working in node-space (no labels) keeps
//! this scan off the multi-TB label file.
//!
//! Algorithm: for each root, descend `successors`, follow only those whose
//! `node_type == Directory`, marking a shared lock-free `NodeBitset`. `test_and_set`
//! gates the stack so each dir is explored once across parallel root walks.

use rayon::prelude::*;
use swh_graph::NodeType;
use swh_graph::graph::{SwhForwardGraph, SwhGraphWithProperties};

use crate::ntf::common::{NodeBitset, N_NODES};

/// Mark every Directory reachable (dir->dir) from any `roots` node into `reached`.
///
/// Safe to run the per-root walks in parallel: the shared bitset both records the
/// answer AND de-dups work — the `test_and_set` that gates the stack guarantees each
/// directory is explored by exactly one worker (first writer of a dir owns its subtree).
pub fn mark_reachable_dirs<G>(graph: &G, roots: &[usize], reached: &NodeBitset)
where
    G: SwhForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    roots.par_iter().for_each(|&root| {
        // Claim root; if another worker already owns it, its subtree is theirs.
        if !reached.test_and_set(root) {
            return;
        }
        let mut stack = vec![root];
        while let Some(u) = stack.pop() {
            for v in graph.successors(u) {
                // dir->dir only: descend solely into Directory successors.
                if graph.properties().node_type(v) == NodeType::Directory && reached.test_and_set(v)
                {
                    stack.push(v);
                }
            }
        }
    });
}

/// Convenience: allocate a node bitset and run the BFS over `roots`.
pub fn build_reachable<G>(graph: &G, roots: &[usize]) -> NodeBitset
where
    G: SwhForwardGraph + SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    let bs = NodeBitset::new(N_NODES);
    mark_reachable_dirs(graph, roots, &bs);
    bs
}
