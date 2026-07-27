#!/usr/bin/env bash
# End-to-end functional demo for the NIXT artifact:
#   build inspector plugin -> run nccl-tests with the plugin attached ->
#   export dumps to parquet -> exporter summary outputs.
# Usage: NGPUS=2 ./demo/run_demo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGPUS="${NGPUS:-2}"
NCCL_TESTS_HOME="${NCCL_TESTS_HOME:-$ROOT/demo/nccl-tests}"

echo "==> [1/4] Building NCCL Inspector plugin"
make -C "$ROOT/inspector" -j"$(nproc)"
PLUGIN="$ROOT/inspector/libnccl-profiler-inspector.so"
[ -f "$PLUGIN" ] || { echo "ERROR: plugin build failed ($PLUGIN not found)"; exit 1; }

echo "==> [2/4] Preparing nccl-tests"
if [ ! -x "$NCCL_TESTS_HOME/build/all_reduce_perf" ]; then
    [ -d "$NCCL_TESTS_HOME" ] || git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_HOME"
    make -C "$NCCL_TESTS_HOME" -j"$(nproc)" MPI=0
fi

echo "==> [3/4] Running collectives with Inspector attached ($NGPUS GPUs)"
DUMP_DIR="$ROOT/demo/dump/$(date +%s)"
mkdir -p "$DUMP_DIR"
export NCCL_PROFILER_PLUGIN="$PLUGIN"
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR="$DUMP_DIR"

"$NCCL_TESTS_HOME/build/all_reduce_perf" -b 1K -e 256M -f 2 -g "$NGPUS"
"$NCCL_TESTS_HOME/build/all_gather_perf" -b 1K -e 256M -f 2 -g "$NGPUS"

NDUMPS=$(find "$DUMP_DIR" -name '*.log*' | wc -l)
echo "==> Inspector produced $NDUMPS dump file(s) in $DUMP_DIR"
[ "$NDUMPS" -gt 0 ] || { echo "ERROR: no Inspector dumps were produced"; exit 1; }

echo "==> [4/4] Exporting dumps to parquet + summaries"
mkdir -p "$ROOT/data"
cd "$ROOT/data"
python3 "$ROOT/exporter/perf_summary_exporter.py" --input_dir "$DUMP_DIR" --output_dir demo-analysis

echo ""
echo "Demo complete."
echo "  Raw dumps   : $DUMP_DIR"
echo "  Parquet     : $ROOT/data/demo-analysis/parquet_files/"
echo "  Summaries   : $ROOT/data/demo-analysis/"
