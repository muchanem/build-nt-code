//! `swh_resolver` — resolve code metadata `(repo, commit_prefix, rel_path)` to
//! Software Heritage content SWHIDs (`swh:1:cnt:<sha1_git>`) using the compressed
//! SWH graph.
//!
//! Module map (see `ntf::*`):
//!   common   - shared types, graph loaders, and string/SWHID utilities
//!   names    - rel_path components -> requested filename-id set + component->id map
//!   reach    - label-free forward BFS dir->dir -> reachable-dir bitvector
//!   seeds    - per-row rev_swhid -> root_dir node + path filename-ids (Seeds)
//!   hydrate  - K-way sequential scan of graph-labelled.labels -> in-RAM DirMap
//!   traverse - pointer-chase root->...->leaf, leaf->cnt_swhid, write parquet
//!   scratch  - rev-prefix index + backward-provenance winner selection
pub mod ntf;
