#!/usr/bin/env bash
# NIXT camera-ready — ingest timing for 340B/2048-GPU trace.
# Writes to <data-root>/nccl_inspector_exporter_cameraready/<fresh-dir>
set -uo pipefail

RUN_TAG="pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606"
DUMP="<data-root>/llmb_workload_inspector_enabled/${RUN_TAG}/pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512/nccl_inspector_dump"
OUT_ROOT="<data-root>/nccl_inspector_exporter_cameraready"
mkdir -p "$OUT_ROOT"
cd "$OUT_ROOT"

# Fresh dir per run — never overwrite historical <data-root>/nccl_inspector_exporter/data
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_ROOT}/${RUN_TAG}-run-${STAMP}"

REPORT="${OUT_ROOT}/report_${STAMP}.txt"

exec > >(tee -a "$REPORT") 2>&1

echo "===================================================================="
echo "NIXT camera-ready ingest run — $STAMP"
echo "===================================================================="
echo "DUMP : $DUMP"
echo "OUT  : $OUT_DIR"
date -u +"start=%Y-%m-%dT%H:%M:%S.%NZ"

# Baseline sizes
echo
echo "--- RAW LOG SIZE ---"
du -sb "$DUMP" | awk '{printf "raw_bytes=%s\nraw_files=", $1}'
find "$DUMP" -type f -name "*.log" | wc -l
du -sh "$DUMP" | awk '{print "raw_human="$1}'

# Env
source <conda-root>/etc/profile.d/conda.sh
conda activate nccl_exporter
which python
python -c "import duckdb,pandas,pyarrow;print('duckdb',duckdb.__version__,'pandas',pandas.__version__,'pyarrow',pyarrow.__version__)"

SCRIPT="<data-root>/nccl_inspector_exporter/nccl/ext-profiler/inspector/exporter/example/perf_summary_exporter.py"

echo
echo "--- WARMUP: page-cache priming pass over raw dump (sequential cat > /dev/null) ---"
/usr/bin/time -v bash -c "find '$DUMP' -type f -name '*.log' -print0 | xargs -0 -n64 cat > /dev/null" 2>&1 | tail -20 || true

echo
echo "--- INGEST + SUMMARY (full pipeline, warm cache) ---"
date -u +"ingest_start=%Y-%m-%dT%H:%M:%S.%NZ"
/usr/bin/time -v python "$SCRIPT" --input_dir "$DUMP" --output_dir "$OUT_DIR" 2>&1 | tail -60
RC=$?
date -u +"ingest_end=%Y-%m-%dT%H:%M:%S.%NZ"
echo "ingest_rc=$RC"

echo
echo "--- OUTPUT SIZES ---"
du -sh "$OUT_DIR" | awk '{print "analysis_total="$1}'
du -sb "$OUT_DIR" | awk '{print "analysis_bytes="$1}'
du -sh "$OUT_DIR/parquet_files" 2>/dev/null | awk '{print "parquet_dir="$1}'
du -sb "$OUT_DIR/parquet_files" 2>/dev/null | awk '{print "parquet_bytes="$1}'
find "$OUT_DIR/parquet_files" -type f -name "*.parquet" 2>/dev/null | wc -l | awk '{print "parquet_files="$1}'

echo
echo "--- PARQUET-ONLY WALL CLOCK (from output.log timestamps) ---"
LOG="$OUT_DIR/output.log"
if [ -f "$LOG" ]; then
  FIRST=$(grep -m1 "Created parquet file" "$LOG" | awk '{print $1" "$2}' | sed 's/,/./')
  LAST=$(grep "Created parquet file" "$LOG" | tail -1 | awk '{print $1" "$2}' | sed 's/,/./')
  SUM_START=$(grep -m1 "Generating summary for" "$LOG" | awk '{print $1" "$2}' | sed 's/,/./')
  SUM_END=$(tail -1 "$LOG" | awk '{print $1" "$2}' | sed 's/,/./')
  echo "first_parquet_ts=$FIRST"
  echo "last_parquet_ts =$LAST"
  echo "first_summary_ts=$SUM_START"
  echo "last_line_ts    =$SUM_END"
fi

echo
date -u +"done=%Y-%m-%dT%H:%M:%S.%NZ"
echo "REPORT=$REPORT"
echo "OUT_DIR=$OUT_DIR"
