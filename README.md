# NIXT Artifact — Demystifying Collective Communication (IISWC 2026)

Artifact for the IISWC 2026 paper **"Demystifying Collective Communication: Observability of NCCL Collectives in Large-Scale LLM Training with NIXT"**.

NIXT (NCCL Inspector eXporter Tool) is an observability pipeline for NCCL collective communication:

```
NCCL Inspector plugin (C++)  →  per-rank JSON dumps (.log.gz)
        exporter (Python)    →  parquet + performance summaries
        analysis (Python)    →  paper figures
```

## Important note on data

The telemetry analyzed in the paper was collected from **large-scale production training runs
(up to 2048 H100 GPUs)** on infrastructure whose data is **confidential and cannot be
redistributed**. This artifact therefore contains **code only**:

- We claim **Artifacts Available** and **Artifacts Evaluated — Functional**.
- We do **not** claim **Results Reproduced**: reproducing the paper's figures requires the
  confidential telemetry (and the original clusters).
- To demonstrate functionality end-to-end, `demo/run_demo.sh` runs the full pipeline
  (plugin → dumps → parquet → analysis) on a small GPU node using `nccl-tests` as the workload.
- `expected_output/` contains the exact experiment figures from the camera-ready paper, so
  evaluators can see what the analysis stage produces when pointed at the full dataset.

## Repository layout

```
inspector/         NCCL Inspector profiler plugin (C++, standalone Makefile,
                   vendored from NCCL ext-profiler; BSD-3, NVIDIA copyright)
exporter/          perf_summary_exporter.py — parses .log.gz dumps into
                   parquet files + summary reports/plots
analysis/          Paper figure scripts (read parquet under $NIXT_ROOT/data)
demo/              End-to-end functional demo (2–8 GPUs, nccl-tests workload)
expected_output/   Experiment figures exactly as they appear in the paper
docs/figure_map.md Paper figure ↔ script ↔ input mapping
```

## Requirements

**Hardware:** Linux x86_64 node with ≥ 2 NVIDIA GPUs (any recent architecture; the demo was
developed against A100/H100 but has no arch-specific code).

**Software:**
- CUDA ≥ 12.x (plugin builds with c++14/c++17 depending on CUDA version)
- NCCL ≥ 2.28 (profiler plugin API; NCCL 2.28–2.30 tested)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) (cloned/built automatically by the demo)
- Python ≥ 3.10 with: `pandas`, `tqdm`, `duckdb`, `matplotlib`, `pyarrow`, `numpy`
  (`pip install -r exporter/requirements.txt`)

## Getting started (functional evaluation, ~15 min)

```bash
# 0) Python deps
pip install -r exporter/requirements.txt

# 1) Build the Inspector plugin
make -C inspector            # set CUDA_HOME=/path/to/cuda if not /usr/local/cuda

# 2) Run the end-to-end demo (builds nccl-tests if needed, runs 2 collectives
#    with the plugin attached, exports dumps to parquet, emits summary plots)
NGPUS=2 ./demo/run_demo.sh
```

Expected results:
- `demo/dump/<timestamp>/` contains per-rank `*.log.gz` Inspector dumps
- `data/demo-analysis/parquet_files/*.parquet` contains one parquet per rank
- `data/demo-analysis/` contains summary CSV/plots produced by the exporter

## Running the plugin on your own workload

```bash
export NCCL_PROFILER_PLUGIN=/path/to/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=/path/to/dumps        # optional
<launch your NCCL/PyTorch job as usual>
```

See `inspector/README.md` for all knobs (verbose event traces, Prometheus export, etc.).

Then export and analyze:

```bash
python3 exporter/perf_summary_exporter.py --input_dir /path/to/dumps
```

## Paper analysis scripts

The scripts in `analysis/` generated every experiment figure in the paper. They read parquet
produced by the exporter from `$NIXT_ROOT/data/*-analysis/parquet_files/*.parquet` and write
figures to `$NIXT_ROOT/figures/` (`NIXT_ROOT` defaults to this repository's root; override via
the environment). The exact figure ↔ script mapping is in `docs/figure_map.md`.

They are provided so the full methodology is inspectable and reusable on your own telemetry;
they will not reproduce the paper's figures without the confidential dataset.

## License

BSD-3-Clause (see `LICENSE`). The `inspector/` plugin is vendored from NVIDIA's NCCL
repository (`ext-profiler/inspector`) and retains its NVIDIA copyright headers.
