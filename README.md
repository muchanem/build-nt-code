# swh_resolver

Resolve code metadata `(repo, commit_prefix, rel_path)` to **Software Heritage content
SWHIDs** (`swh:1:cnt:<sha1_git>`) using the compressed SWH graph. Given e.g. `("torvalds/linux", "a1b2c3d", "kernel/sched/core.c")`
it returns the SWHID of that exact file blob.

Two binaries:
- **`nt_scratch`** — the resolver. Takes a parquet of metadata rows and writes resolved rows.
- **`nt_verify`** — independent check: re-resolves a sample of the output by walking the graph
  directly (a different code path) and diffs the result.

Build: `cargo build --release` → `target/release/{nt_scratch,nt_verify}`. Needs a local copy
of the SWH graph (see *Staging*) and a big-RAM machine (~2 TB; the label scan dominates).

## Pipeline (`nt_scratch`)

```
input parquet (repo, commit_id, rel_path)
 B0  rev-prefix index   revision SWHIDs -> {prefix : [rev nodes]}   (skipped if commit_id is full 40-hex)
 B1  provenance winner  per (repo,commit): origin -> snapshot set; keep candidate revs whose
                        backward walk reaches it; tie-break Branch-Head > Max-Hex -> one winner rev
                        -> winner cands parquet
 names    rel_path components -> requested filename-id set + component->id map
 seeds    winner rev -> root_dir node; per-row path filename-ids
 reach    forward BFS dir->dir from the root dirs -> reachable-dir bitvector
 hydrate  ONE K-way sequential scan of graph-labelled.labels -> in-RAM map dir -> [(name,child,type)]
 traverse pointer-chase each row root -> ... -> leaf; leaf node -> cnt_swhid
 emit     write 10-col parquet, partitioned by bin
```

`commit_id` may be a short hex prefix (e.g. 7 chars, set `--prefix-nibbles`) or a full 40-hex
commit hash (then B0 is skipped). The graph is the *complete* set of revisions, so prefix
lookup finds the commit whenever SWH has it; provenance picks the one that belongs to the repo.

## Run

```
target/release/nt_scratch \
  --graph-dir /dev/shm/swh-graph/default/graph \   # swh-graph basename (the loader opens siblings)
  --input    new_metadata.parquet \                # cols: repo, commit_id, rel_path [, bin]
  --cands-out scratch_cands.parquet \              # intermediate winner-cands (B0/B1 output)
  --out-dir   scratch_resolved \                   # final: scratch_resolved/bin=<b>/part-*.parquet
  --prefix-nibbles 7 --threads 0 --hydrate-k 16

target/release/nt_verify --graph-dir … --resolved scratch_resolved --n 5000
```

Input columns are read as `LargeUtf8`. An optional `bin` column only controls output
partitioning (defaults to one partition).

## Output (10 cols, arrow Utf8, not-null)

`repo, commit_id, rel_path, id, origin_url, snapshot_swhid, rev_swhid, cnt_swhid, qualified, status`
- `id` = `rev_swhid` = full `swh:1:rev:<40hex>`; `cnt_swhid` = `swh:1:cnt:<40hex>` (`""` if not found).
- `qualified` = `<cnt>;origin=<url>;visit=<snp>;anchor=<rev>;path=/<rel_path>` (visit empty).
- `status ∈ {ok, path_not_found}`.

## The SWH graph & where to stage each file

Point `--graph-dir` at a directory holding the swh-graph export (basename `…/graph`). The loader
opens many sibling files; performance depends entirely on putting the hot ones in RAM and keeping
the one enormous file off random access. For a ~2 TB-RAM node (sizes from the 2025-05-18 export):

| File(s) | Size | Stage on | Access |
|---|---|---|---|
| `graph-labelled.labels` | **4.3 TB** | **`bulk/net storage` (symlink, don't copy)** | one sequential scan (hydrate) |
| `node2swhid.bin` | 1.1 TB | local SSD | random (leaf → cnt_swhid) |
| `*.pthash.order` | 399 GB | local SSD (+ `vmtouch`) | random (SWHID→node MPH) |
| `graph.graph` (+`.ef`) | ~335 GB | RAM (`/dev/shm`) | topology |
| `graph-transposed.graph` (+`.ef`) | ~238 GB | RAM (`/dev/shm`) | backward provenance (B1) |
| `graph.labels.fcl.*` | ~207 GB | RAM (`/dev/shm`) | filename-id → bytes |
| `graph-labelled.ef` / `.labeloffsets` | ~116 GB | RAM (`/dev/shm`) | label scan boundaries |
| `*.pthash` | 15 GB | RAM (`/dev/shm`) | MPH |
| `graph.node2type.bin` | 19 GB | RAM (`/dev/shm`) | node-type checks |
| `*.properties`, small meta | tiny | symlink `/net` | loader |
