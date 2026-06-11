//! nt_verify — independent correctness check for resolver output.
//!
//! Re-resolves a random sample of OUTPUT rows by walking the graph DIRECTLY (forward
//! labeled walk per row, NO hydrate map), and diffs the directly-resolved cnt_swhid
//! against the pipeline's. Also tallies the status distribution. Two independent code
//! paths producing the same cnt_swhid is strong evidence the hydrate/traverse is correct.
//!
//! Example:
//!   target/release/nt_verify \
//!     --graph-dir /dev/shm/swh-graph/default/graph \
//!     --resolved scratch_resolved \
//!     --n 2000

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, LargeStringArray, StringArray};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use swh_graph::SWHID;
use swh_graph::graph::{SwhForwardGraph, SwhGraphWithProperties, SwhLabeledForwardGraph};
use swh_graph::labels::EdgeLabel;

use swh_resolver::ntf::common::{load_forward, normalize_path};

#[derive(Parser, Debug)]
#[command(name = "nt_verify")]
struct Opts {
    #[arg(long)]
    graph_dir: PathBuf,
    /// resolved output dir (bin=*/part-*.parquet) or a single parquet file
    #[arg(long)]
    resolved: PathBuf,
    /// number of OK rows to re-resolve
    #[arg(long, default_value_t = 2000)]
    n: usize,
}

/// One sampled output row.
struct Row {
    rev_swhid: String,
    rel_path: String,
    cnt_swhid: String,
}

fn main() -> Result<()> {
    let opts = Opts::parse();
    eprintln!("[nt_verify] loading forward graph from {:?}", opts.graph_dir);
    let graph = load_forward(&opts.graph_dir).context("load_forward")?;

    // Collect parquet files.
    let files = collect_parquets(&opts.resolved)?;
    if files.is_empty() {
        return Err(anyhow!("no parquet files under {:?}", opts.resolved));
    }
    eprintln!("[nt_verify] {} output parquet file(s)", files.len());

    // Sample rows: read files until we have ~n OK rows (take from the front of each).
    let mut sample: Vec<Row> = Vec::new();
    let mut n_ok: u64 = 0;
    let mut n_total: u64 = 0;
    let mut status_hist: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    'outer: for f in &files {
        let file = std::fs::File::open(f)?;
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(8192)
            .build()?;
        while let Some(batch) = reader.next() {
            let batch = batch?;
            let cols = batch.schema();
            let get = |name: &str| -> Option<usize> { cols.index_of(name).ok() };
            let (i_rev, i_rel, i_cnt, i_st) = (
                get("rev_swhid").context("no rev_swhid col")?,
                get("rel_path").context("no rel_path col")?,
                get("cnt_swhid").context("no cnt_swhid col")?,
                get("status").context("no status col")?,
            );
            let rev = str_col(batch.column(i_rev).as_ref())?;
            let rel = str_col(batch.column(i_rel).as_ref())?;
            let cnt = str_col(batch.column(i_cnt).as_ref())?;
            let st = str_col(batch.column(i_st).as_ref())?;
            for i in 0..batch.num_rows() {
                n_total += 1;
                let status = st.get(i);
                *status_hist.entry(status.clone()).or_insert(0) += 1;
                if status == "ok" {
                    n_ok += 1;
                    if sample.len() < opts.n {
                        sample.push(Row {
                            rev_swhid: rev.get(i),
                            rel_path: rel.get(i),
                            cnt_swhid: cnt.get(i),
                        });
                    }
                }
            }
            if sample.len() >= opts.n {
                break 'outer;
            }
        }
    }
    eprintln!(
        "[nt_verify] scanned {} rows ({} ok); re-resolving {} sampled ok rows",
        n_total,
        n_ok,
        sample.len()
    );
    eprintln!("[nt_verify] status histogram (scanned prefix): {:?}", status_hist);

    // Re-resolve each directly and diff.
    let mut matched = 0u64;
    let mut mism = 0u64;
    let mut unresolved = 0u64;
    let mut examples: Vec<String> = Vec::new();
    for row in &sample {
        match direct_resolve(&graph, &row.rev_swhid, &row.rel_path) {
            Some(cnt) => {
                if cnt == row.cnt_swhid {
                    matched += 1;
                } else {
                    mism += 1;
                    if examples.len() < 10 {
                        examples.push(format!(
                            "MISMATCH rev={} path={} pipeline={} direct={}",
                            row.rev_swhid, row.rel_path, row.cnt_swhid, cnt
                        ));
                    }
                }
            }
            None => {
                unresolved += 1;
                if examples.len() < 10 {
                    examples.push(format!(
                        "DIRECT-UNRESOLVED rev={} path={} pipeline={}",
                        row.rev_swhid, row.rel_path, row.cnt_swhid
                    ));
                }
            }
        }
    }

    eprintln!(
        "\n[nt_verify] RESULT: matched={matched} mismatched={mism} direct_unresolved={unresolved} (of {} ok rows)",
        sample.len()
    );
    for e in &examples {
        eprintln!("  {e}");
    }
    if mism == 0 && unresolved == 0 {
        eprintln!("[nt_verify] PASS — all sampled ok rows re-resolve identically via direct walk.");
    } else {
        eprintln!("[nt_verify] FAIL — discrepancies found (see above).");
    }
    Ok(())
}

/// Direct forward resolution: rev_swhid -> node -> root dir -> labeled walk -> cnt swhid.
fn direct_resolve<G>(graph: &G, rev_swhid: &str, rel_path: &str) -> Option<String>
where
    G: SwhForwardGraph + SwhLabeledForwardGraph + SwhGraphWithProperties,
    <G as SwhGraphWithProperties>::Maps: swh_graph::properties::Maps,
    <G as SwhGraphWithProperties>::LabelNames: swh_graph::properties::LabelNames,
{
    use swh_graph::NodeType;
    let swh = SWHID::try_from(rev_swhid).ok()?;
    let rev = graph.properties().node_id(swh).ok()?;
    let mut cur = graph
        .successors(rev)
        .into_iter()
        .find(|&s| graph.properties().node_type(s) == NodeType::Directory)?;

    let norm = normalize_path(rel_path);
    let parts: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
    for (i, comp) in parts.iter().enumerate() {
        let comp_b = comp.as_bytes();
        let mut found: Option<usize> = None;
        for (child, labels) in graph.labeled_successors(cur) {
            for l in labels {
                if let EdgeLabel::DirEntry(de) = l {
                    let name = graph.properties().label_name(de.label_name_id());
                    if name == comp_b {
                        found = Some(child);
                        break;
                    }
                }
            }
            if found.is_some() {
                break;
            }
        }
        let child = found?;
        let last = i + 1 == parts.len();
        let nt = graph.properties().node_type(child);
        if last {
            return if nt == NodeType::Content {
                Some(graph.properties().swhid(child).to_string())
            } else {
                None
            };
        }
        if nt != NodeType::Directory {
            return None;
        }
        cur = child;
    }
    None
}

fn collect_parquets(p: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if p.is_file() {
        out.push(p.to_path_buf());
        return Ok(out);
    }
    // walk one or two levels (out_dir/bin=*/part-*.parquet)
    for e in std::fs::read_dir(p)? {
        let e = e?;
        let path = e.path();
        if path.is_dir() {
            for e2 in std::fs::read_dir(&path)? {
                let p2 = e2?.path();
                if p2.extension().map(|x| x == "parquet").unwrap_or(false) {
                    out.push(p2);
                }
            }
        } else if path.extension().map(|x| x == "parquet").unwrap_or(false) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Read a Utf8 OR LargeUtf8 column as owned strings accessor.
enum StrCol<'a> {
    Large(&'a LargeStringArray),
    Small(&'a StringArray),
}
impl StrCol<'_> {
    fn get(&self, i: usize) -> String {
        match self {
            StrCol::Large(a) => a.value(i).to_string(),
            StrCol::Small(a) => a.value(i).to_string(),
        }
    }
}
fn str_col(arr: &dyn Array) -> Result<StrCol<'_>> {
    if let Some(a) = arr.as_any().downcast_ref::<LargeStringArray>() {
        Ok(StrCol::Large(a))
    } else if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        Ok(StrCol::Small(a))
    } else {
        Err(anyhow!("expected Utf8/LargeUtf8 column"))
    }
}
