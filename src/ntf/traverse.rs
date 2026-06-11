//! traverse.rs — pointer-chase root->...->leaf, leaf->cnt_swhid, and parquet emit.
//!
//! `traverse` (pure, no graph): for each row, walk root -> rel_path components via
//! `map.lookup(cur, fid)`:
//!   - empty fids / fid == FID_MISSING -> Status::NotFound
//!   - lookup miss -> NotFound
//!   - last component: Content -> Ok (leaf=child); Directory -> DirTarget; Revision -> Submodule
//!   - intermediate component: must be Directory to continue; Revision -> Submodule; else NotFound
//!   - root == ROOT_MISSING -> NoRoot
//! Returns (leaf_node per row [0 if not Ok], status_u8 per row).
//!
//! `emit`: stream the cands parquet in FILE ORDER (cols repo, commit_id,
//! rel_path, id, origin_url, rev_swhid as LargeStringArray); row i aligns with
//! leaf[i]/status[i]/seeds.bin[i]. Build the 10-col output
//! (repo, commit_id, rel_path, id, origin_url, snapshot_swhid="", rev_swhid,
//! cnt_swhid, qualified, status) where cnt_swhid = swhid(leaf) for Ok rows else "",
//! qualified via common::build_qualified, status via Status::output_str. Partition output
//! by `bin` into out_dir/bin=<b>/part-<NNNNN>.parquet (zstd). 10 cols, arrow Utf8, not-null.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;

use arrow::array::{Array, LargeStringArray, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use swh_graph::graph::SwhGraphWithProperties;

use crate::ntf::common::{
    DirMap, FID_MISSING, ROOT_MISSING, Seeds, Status, TY_CONTENT, TY_DIRECTORY, TY_REVISION,
    build_qualified, normalize_rev_id,
};

/// Flush a per-bin builder set into a RecordBatch every ~100k rows to avoid the
/// i32 StringBuilder 2GB offset overflow.
const OUTPUT_BATCH_SIZE: usize = 100_000;

/// Resolve every row's leaf content node + status. Returns (leaf, status) per row.
///
/// Pure pointer-chase over the hydrated `DirMap`; no graph access. Parallel over rows.
/// `leaf[i]` is the resolved Content node iff `status[i] == Ok`, else 0.
pub fn traverse(map: &DirMap, seeds: &Seeds) -> (Vec<u64>, Vec<u8>) {
    let n = seeds.n_rows;
    let mut leaf = vec![0u64; n];
    let mut status = vec![0u8; n];

    // Walk each row independently. ~1.2B lookups total; per-row is fine (minutes).
    leaf.par_iter_mut()
        .zip(status.par_iter_mut())
        .enumerate()
        .for_each(|(r, (leaf_out, status_out))| {
            let (l, s) = walk_row(map, seeds, r);
            *leaf_out = l;
            *status_out = s as u8;
        });

    (leaf, status)
}

/// Resolve a single row: returns (leaf_node, status). leaf is 0 unless status==Ok.
#[inline]
fn walk_row(map: &DirMap, seeds: &Seeds, row: usize) -> (u64, Status) {
    let root = seeds.root[row];
    if root == ROOT_MISSING {
        return (0, Status::NoRoot);
    }
    let fids = seeds.path_of(row);
    if fids.is_empty() {
        return (0, Status::NotFound);
    }

    let last = fids.len() - 1;
    let mut cur = root;
    for (i, &fid) in fids.iter().enumerate() {
        if fid == FID_MISSING {
            return (0, Status::NotFound);
        }
        let (child, ty) = match map.lookup(cur, fid) {
            Some(c) => c,
            None => return (0, Status::NotFound),
        };
        if i == last {
            return match ty {
                TY_CONTENT => (child, Status::Ok),
                TY_DIRECTORY => (0, Status::DirTarget),
                TY_REVISION => (0, Status::Submodule),
                _ => (0, Status::NotFound),
            };
        }
        // Intermediate component: must be a directory to keep descending.
        match ty {
            TY_DIRECTORY => cur = child,
            TY_REVISION => return (0, Status::Submodule),
            _ => return (0, Status::NotFound),
        }
    }
    // Unreachable (non-empty fids always returns on the last iteration).
    (0, Status::NotFound)
}

/// Write the 10-col parquet partitioned by bin, re-reading the cands parquet for
/// pass-through strings and resolving cnt_swhid via graph.properties().swhid(leaf).
///
/// Streams the cands parquet in FILE ORDER; row i aligns with `leaf[i]` / `status[i]` /
/// `seeds.bin[i]`. snapshot_swhid is "" (not present in the cands parquet).
pub fn emit<G>(
    graph: &G,
    all_cands: &Path,
    out_dir: &Path,
    seeds: &Seeds,
    leaf: &[u64],
    status: &[u8],
) -> Result<()>
where
    G: SwhGraphWithProperties + Sync,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
{
    assert_eq!(seeds.n_rows, leaf.len(), "leaf len != seeds.n_rows");
    assert_eq!(seeds.n_rows, status.len(), "status len != seeds.n_rows");

    // Output schema: 10 cols, all Utf8, not-null, in the recon-R4 order.
    let out_schema = Arc::new(Schema::new(vec![
        Field::new("repo", DataType::Utf8, false),
        Field::new("commit_id", DataType::Utf8, false),
        Field::new("rel_path", DataType::Utf8, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("origin_url", DataType::Utf8, false),
        Field::new("snapshot_swhid", DataType::Utf8, false),
        Field::new("rev_swhid", DataType::Utf8, false),
        Field::new("cnt_swhid", DataType::Utf8, false),
        Field::new("qualified", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();

    // Open the all_cands reader, projecting the 6 pass-through columns.
    let file = File::open(all_cands)
        .with_context(|| format!("open all_cands parquet {}", all_cands.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("ParquetRecordBatchReaderBuilder::try_new(all_cands)")?
        .with_batch_size(OUTPUT_BATCH_SIZE);

    // Resolve column indices by name (schema order is fixed but we map robustly).
    let schema = builder.metadata().file_metadata().schema_descr();
    let mut idx_repo = None;
    let mut idx_commit = None;
    let mut idx_rel = None;
    let mut idx_id = None;
    let mut idx_origin = None;
    let mut idx_rev = None;
    for (i, col) in schema.columns().iter().enumerate() {
        match col.path().string().as_str() {
            "repo" => idx_repo = Some(i),
            "commit_id" => idx_commit = Some(i),
            "rel_path" => idx_rel = Some(i),
            "id" => idx_id = Some(i),
            "origin_url" => idx_origin = Some(i),
            "rev_swhid" => idx_rev = Some(i),
            _ => {}
        }
    }
    let idx_repo = idx_repo.context("all_cands missing column repo")?;
    let idx_commit = idx_commit.context("all_cands missing column commit_id")?;
    let idx_rel = idx_rel.context("all_cands missing column rel_path")?;
    let idx_id = idx_id.context("all_cands missing column id")?;
    let idx_origin = idx_origin.context("all_cands missing column origin_url")?;
    let idx_rev = idx_rev.context("all_cands missing column rev_swhid")?;

    // After ProjectionMask::leaves, batch columns appear in ORIGINAL schema order
    // (ascending leaf index), NOT in the order passed to the projection vector.
    // Compute each column's batch position as its rank among the projected leaves.
    let mut projected = [idx_repo, idx_commit, idx_rel, idx_id, idx_origin, idx_rev];
    let projection = ProjectionMask::leaves(builder.parquet_schema(), projected.to_vec());
    builder = builder.with_projection(projection);

    projected.sort_unstable();
    let pos_of = |orig: usize| -> usize {
        projected.iter().position(|&x| x == orig).unwrap()
    };
    let pos_repo = pos_of(idx_repo);
    let pos_commit = pos_of(idx_commit);
    let pos_rel = pos_of(idx_rel);
    let pos_id = pos_of(idx_id);
    let pos_origin = pos_of(idx_origin);
    let pos_rev = pos_of(idx_rev);

    let mut reader = builder.build().context("build all_cands parquet reader")?;

    // One BinWriter per bin, created lazily.
    let mut writers: HashMap<u16, BinWriter> = HashMap::new();

    let props = Arc::new(props);
    let mut row_global: usize = 0;

    while let Some(batch) = reader.next() {
        let batch = batch.context("read all_cands batch")?;
        let nrows = batch.num_rows();
        // Guard: a clean error (not an OOB panic) if all_cands has more rows than seeds.
        if row_global + nrows > seeds.n_rows {
            return Err(anyhow!(
                "all_cands yielded more rows ({}) than seeds.n_rows ({}); alignment broken",
                row_global + nrows,
                seeds.n_rows
            ));
        }

        let repo_arr = column_as_str_array(batch.column(pos_repo).as_ref())?;
        let commit_arr = column_as_str_array(batch.column(pos_commit).as_ref())?;
        let rel_arr = column_as_str_array(batch.column(pos_rel).as_ref())?;
        let id_arr = column_as_str_array(batch.column(pos_id).as_ref())?;
        let origin_arr = column_as_str_array(batch.column(pos_origin).as_ref())?;
        let rev_arr = column_as_str_array(batch.column(pos_rev).as_ref())?;

        for i in 0..nrows {
            let r = row_global + i;
            let st = Status::from_u8(status[r]);
            let origin_url = origin_arr.value(i);
            let rel_path = rel_arr.value(i);
            let rev_swhid = rev_arr.value(i);

            let (cnt_swhid, qualified) = if st == Status::Ok {
                let cnt = graph
                    .properties()
                    .swhid(leaf[r] as usize)
                    .to_string();
                let q = build_qualified(&cnt, origin_url, "", rev_swhid, rel_path);
                (cnt, q)
            } else {
                (String::new(), String::new())
            };
            let status_str = st.output_str();

            // The `id` column is the FULL rev SWHID (swh:1:rev:<40hex>), not the bare
            // 40-hex `id` stored in the cands parquet. Normalize it.
            let id_full = normalize_rev_id(id_arr.value(i));

            let bin = seeds.bin[r];
            let w = writers
                .entry(bin)
                .or_insert_with(|| BinWriter::new(out_dir, bin, out_schema.clone(), props.clone()));
            w.push(
                repo_arr.value(i),
                commit_arr.value(i),
                rel_path,
                id_full.as_ref(),
                origin_url,
                "", // snapshot_swhid — not present in the cands parquet
                rev_swhid,
                &cnt_swhid,
                &qualified,
                status_str,
            )?;
        }

        row_global += nrows;
    }

    if row_global != seeds.n_rows {
        return Err(anyhow!(
            "all_cands row count {} != seeds.n_rows {}; leaf/status alignment broken",
            row_global,
            seeds.n_rows
        ));
    }

    // Flush remainders and close every bin writer.
    for (_bin, mut w) in writers.into_iter() {
        w.finish()?;
    }

    Ok(())
}

/// Per-bin output writer: owns 10 StringBuilders + a parquet writer for
/// out_dir/bin=<b>/part-<NNNNN>.parquet, flushing every OUTPUT_BATCH_SIZE rows.
struct BinWriter {
    schema: Arc<Schema>,
    props: Arc<WriterProperties>,
    dir: std::path::PathBuf,
    part_idx: u64,
    writer: Option<ArrowWriter<File>>,
    builders: [StringBuilder; 10],
    pending: usize,
}

impl BinWriter {
    fn new(out_dir: &Path, bin: u16, schema: Arc<Schema>, props: Arc<WriterProperties>) -> Self {
        let dir = out_dir.join(format!("bin={}", bin));
        Self {
            schema,
            props,
            dir,
            part_idx: 0,
            writer: None,
            builders: std::array::from_fn(|_| StringBuilder::new()),
            pending: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        repo: &str,
        commit_id: &str,
        rel_path: &str,
        id: &str,
        origin_url: &str,
        snapshot_swhid: &str,
        rev_swhid: &str,
        cnt_swhid: &str,
        qualified: &str,
        status: &str,
    ) -> Result<()> {
        self.builders[0].append_value(repo);
        self.builders[1].append_value(commit_id);
        self.builders[2].append_value(rel_path);
        self.builders[3].append_value(id);
        self.builders[4].append_value(origin_url);
        self.builders[5].append_value(snapshot_swhid);
        self.builders[6].append_value(rev_swhid);
        self.builders[7].append_value(cnt_swhid);
        self.builders[8].append_value(qualified);
        self.builders[9].append_value(status);
        self.pending += 1;
        if self.pending >= OUTPUT_BATCH_SIZE {
            self.flush()?;
        }
        Ok(())
    }

    /// Build a RecordBatch from the current builders (resetting them) and write it.
    fn flush(&mut self) -> Result<()> {
        if self.pending == 0 {
            return Ok(());
        }
        let cols: Vec<Arc<dyn Array>> = self
            .builders
            .iter_mut()
            .map(|b| Arc::new(b.finish()) as Arc<dyn Array>)
            .collect();
        let batch = RecordBatch::try_new(self.schema.clone(), cols)
            .context("build output RecordBatch")?;

        // Open the part file lazily on first flush.
        if self.writer.is_none() {
            fs::create_dir_all(&self.dir)
                .with_context(|| format!("create out dir {}", self.dir.display()))?;
            self.part_idx += 1;
            let path = self.dir.join(format!("part-{:05}.parquet", self.part_idx));
            let f = File::create(&path)
                .with_context(|| format!("create part file {}", path.display()))?;
            let w = ArrowWriter::try_new(
                f,
                self.schema.clone(),
                Some((*self.props).clone()),
            )
            .context("ArrowWriter::try_new")?;
            self.writer = Some(w);
        }
        self.writer
            .as_mut()
            .unwrap()
            .write(&batch)
            .context("write output batch")?;
        self.pending = 0;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(w) = self.writer.take() {
            w.close().context("close output parquet writer")?;
        }
        Ok(())
    }
}

/// Downcast a parquet-read column to LargeStringArray (all_cands cols are large_string).
fn column_as_str_array(arr: &dyn Array) -> Result<&LargeStringArray> {
    if let Some(s) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(s)
    } else if arr.as_any().downcast_ref::<StringArray>().is_some() {
        Err(anyhow!(
            "expected LargeUtf8 in all_cands; got Utf8 — cast input columns to LargeUtf8"
        ))
    } else {
        Err(anyhow!("expected Utf8/LargeUtf8 column in all_cands"))
    }
}
