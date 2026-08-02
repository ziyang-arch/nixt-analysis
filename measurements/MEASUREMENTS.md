# Camera-ready overhead measurements — NIXT

Trace: **Nemotron-4 340B, 2048 GPUs**
Dump path (internal):
`<data-root>/llmb_workload_inspector_enabled/pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606/…/nccl_inspector_dump/`

Host: 64-core x86 server, 187 GiB RAM, 1.8 TB local HDD
(spinning-rust local disk, not the parquet SSD tier).
Env: Miniconda `nccl_exporter`, DuckDB 1.4.3, pandas 2.3.3, pyarrow 23.0.0.
Ingest script: `nccl/ext-profiler/inspector/exporter/example/perf_summary_exporter.py`
(`ProcessPoolExecutor`, `max_workers = min(64, len(files), os.cpu_count())`).

Run stamp: `20260722T041924Z` (2026-07-21 21:19 PDT).

## Store sizes (single trace)

| Quantity                                | Bytes            | Human  |
|-----------------------------------------|------------------|--------|
| Raw NCCL Inspector logs (1669 `.log`)   | 17,662,627,253   | 17.66 GB / 16.4 GiB |
| Parquet log store (1669 `.parquet`)     |    667,471,106   |   667 MB / 636 MiB  |
| Full analysis dir (parquet + summary CSV/PNG) | 747,855,200 | 748 MB              |

Compression ratio: **26.5×** raw JSON → Parquet
(**~23.6×** if you count the entire analysis dir).

Row count in the DuckDB view over Parquet: **37,530,721**.

## Ingest wall clock (single-pass, cold-cache warmed once)

End-to-end `perf_summary_exporter.py` (parquet + summary CSVs + histogram/boxplot PNGs):
**56.51 s** (`/usr/bin/time -v`, 2969% CPU → ≈30-way effective parallelism, peak RSS 1.8 GB).

Broken down via the timestamps in the pipeline's `output.log`:

| Phase                                   | Wall clock |
|-----------------------------------------|-----------|
| Parquet stage (raw JSON → Parquet)      | **24.2 s** |
| Summary CSVs + histograms/boxplots      | 28.9 s    |
| Fixed overhead (imports, arg parsing)   | ~3 s      |

The "single-pass ingest" number the paper needs is the **parquet stage ≈ 24 s** (or the full pipeline ≈ 57 s, depending on where we draw the line — see caveat below).

## Query latencies for Fig. 3 primitives

DuckDB 1.4.3, 64 threads, page cache warm, 5 repetitions per query (1 cold + 4 warm).
Reported: min-of-4-warm.

| Fig. 3 panel          | SQL primitive           | Warm min (Python `fetchall`) | Warm min (compute only) | Result rows |
|----------------------|--------------------------|----------------------------:|------------------------:|-----------:|
| (a) Summary — histogram | `GROUP BY coll`          |   **0.225 s** |          0.225 s |          4 |
| (b) Temporal            | `WHERE msg_size≥1MB`     |     3.580 s   |      **0.247 s** |  2,955,666 |
| (c) Spatial             | `p50(bw) BY rank`        |     0.296 s   |          0.296 s |         32 |
| (d) Resource            | `GROUP BY n_ranks, coll` |     0.245 s   |          0.245 s |          9 |
| (e) Other               | `p50(bw) BY git_rev`     |     0.244 s   |          0.244 s |          1 |

Aggregate: **max single primitive < 0.30 s** for (a), (c), (d), (e).
Primitive (b) returns ≈3 M rows; the DuckDB compute stage is 0.25 s, and the
extra 3.3 s is entirely Python-side result materialization — not a fair number
to quote for the analysis engine.

**Cleanest thing to put in the paper:** *"under 0.3 s each on this store"* or
*"sub-second for each primitive"* — both statements are true for the
DuckDB compute stage.

## Caveats / honesty notes for J

1. Ingest number is **warm-cache** (script pre-cats the 17 GB dump before the
   timed run). Cold-cache from spinning disk would be dominated by ~150–200 MB/s
   sequential read, i.e. ~90–120 s just for I/O — not really the pipeline's fault.
   Warm-cache is the right number to report because it reflects the CPU-bound
   ingest cost.
2. Fig. 3 (b) exercises the "return everything ≥1 MB" firehose. The 0.25 s
   compute number matches the physics; the 3.58 s Python number does not.
   I recommend we phrase the paper as compute-time.
3. Store size is a **single trace**. The paper text says "for the largest
   trace in our dataset", so 667 MB / 17.66 GB / 26.5× is the correct triple.
4. Numbers were produced on the measurement server, not the H100 cluster where the training
   ran. This is fine — the paper text already says "single-node backend is
   sufficient at 2,048-GPU trace volume", and the measurement server (64 cores, 187 GB RAM,
   HDD) is a very unglamorous single node.

## Reproducibility

- `nixt_ingest_cameraready.sh` — driver script that ran the ingest.
- `ingest_report_20260722T041924Z.txt` — full stdout of that run (GNU `time -v` details, page fault counts, output.log tail).
- `nixt_query_bench.py` — 5-primitive query benchmark.
- `query_bench_20260722.txt` — stdout of the query benchmark.
