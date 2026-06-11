//! nt_scratch — the from-scratch entry point (the reusable tool).
//!
//! Resolves brand-new metadata `(repo, commit_prefix, rel_path)` to content SWHIDs:
//! a rev-prefix index + backward-provenance winner selection produce a winner cands
//! parquet, then the shared core (names/seeds/reach/hydrate/traverse/emit) runs on it.
//!
//! Example:
//!   target/release/nt_scratch \
//!     --graph-dir /dev/shm/swh-graph/default/graph \
//!     --input    new_metadata.parquet \
//!     --cands-out scratch_cands.parquet \
//!     --out-dir   scratch_resolved \
//!     --prefix-nibbles 7 --threads 0 --hydrate-k 16

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

use swh_resolver::ntf::common::load_bidirectional;
use swh_resolver::ntf::{hydrate, names, reach, scratch, seeds, traverse};

#[derive(Parser, Debug, Clone)]
#[command(name = "nt_scratch")]
struct Opts {
    #[arg(long)]
    graph_dir: PathBuf,
    /// new metadata parquet (cols repo, commit_id/commit_prefix, rel_path[, bin])
    #[arg(long)]
    input: PathBuf,
    /// where to write the winner-selected cands parquet (all_cands schema)
    #[arg(long)]
    cands_out: PathBuf,
    /// final resolved output dir
    #[arg(long)]
    out_dir: PathBuf,
    /// commit-prefix hex length
    #[arg(long, default_value_t = 7)]
    prefix_nibbles: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 16)]
    hydrate_k: usize,
    /// stop after producing the cands parquet (B0/B1 only)
    #[arg(long, default_value_t = false)]
    cands_only: bool,
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    if opts.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(opts.threads)
            .build_global()
            .ok();
    }
    std::fs::create_dir_all(&opts.out_dir).ok();
    let t_all = Instant::now();

    eprintln!("[nt_scratch] loading bidirectional graph (provenance) from {:?}", opts.graph_dir);
    let graph = load_bidirectional(&opts.graph_dir).context("load_bidirectional")?;

    // B0 + B1 — candidates + provenance winner ----------------------------
    let t = Instant::now();
    scratch::resolve_candidates(&graph, &opts.input, &opts.cands_out, opts.prefix_nibbles, opts.threads)
        .context("resolve_candidates")?;
    eprintln!("[nt_scratch] B0/B1 cands -> {:?} in {:.1}s", opts.cands_out, t.elapsed().as_secs_f64());
    if opts.cands_only {
        return Ok(());
    }

    // SHARED CORE on the produced cands parquet ---------------------------
    let names = names::build_names(&graph, &opts.cands_out, opts.threads).context("build_names")?;
    let seeds = seeds::build_seeds(&graph, &opts.cands_out, &names, opts.threads).context("build_seeds")?;
    let roots = seeds.distinct_roots();
    let reached = reach::build_reachable(&graph, &roots);
    let map = hydrate::hydrate(&graph, &opts.graph_dir, &reached, &names.requested, opts.hydrate_k)
        .context("hydrate")?;
    drop(reached);
    let (leaf, status) = traverse::traverse(&map, &seeds);
    drop(map);
    traverse::emit(&graph, &opts.cands_out, &opts.out_dir, &seeds, &leaf, &status).context("emit")?;

    eprintln!("[nt_scratch] DONE in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}
