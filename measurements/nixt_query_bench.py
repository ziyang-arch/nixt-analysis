#!/usr/bin/env python3
"""
Camera-ready query-latency benchmark for the 5 analysis primitives in
Fig. 3 of the NIXT IISWC paper. Runs each SQL 5 times (1 cold + 4 warm),
reports min/median warm and full-materialization time (fetchall).

Store: Parquet-backed log store for the Nemotron-4 340B / 2048-GPU trace.
"""
from __future__ import annotations
import argparse, statistics, sys, time
from pathlib import Path

import duckdb

DEFAULT_PARQUET_GLOB = (
    "<data-root>/nccl_inspector_exporter_cameraready/"
    "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606"
    "-run-20260722T041924Z/parquet_files/*.parquet"
)

RUN_NAME = "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-run-20260722T041924Z"

QUERIES = {
    "a_summary_histogram": """
        SELECT coll, COUNT(*) AS ops
        FROM logs
        WHERE run='{RUN}'
        GROUP BY coll;
    """,
    "b_temporal_correlation": """
        SELECT dump_timestamp_us, coll, coll_busbw_gbs
        FROM logs
        WHERE run='{RUN}'
          AND coll_msg_size_bytes >= 1000000;
    """,
    "c_spatial_correlation": """
        SELECT rank, quantile_cont(coll_busbw_gbs, 0.5) AS p50_bw
        FROM logs
        WHERE run='{RUN}'
          AND coll='AllGather'
        GROUP BY rank;
    """,
    "d_resource_correlation": """
        SELECT n_ranks, coll, COUNT(*) AS ops
        FROM logs
        WHERE run='{RUN}'
        GROUP BY n_ranks, coll;
    """,
    "e_other_correlation": """
        SELECT git_rev, quantile_cont(coll_busbw_gbs, 0.5) AS p50_bw
        FROM logs
        WHERE coll='AllGather'
        GROUP BY git_rev;
    """,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-glob", default=DEFAULT_PARQUET_GLOB)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = DuckDB default (cores).")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Total runs per query (first is cold, rest warm).")
    args = ap.parse_args()

    con = duckdb.connect()
    if args.threads:
        con.execute(f"PRAGMA threads={args.threads}")

    # Build the logs view exactly the way the paperplot scripts do it.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW logs AS
        SELECT
          regexp_extract(filename, '/([^/]+)/parquet_files/', 1) AS run,
          *
        FROM read_parquet('{args.parquet_glob}', filename=true);
        """
    )

    # Sanity
    nrows = con.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    nruns = con.execute("SELECT COUNT(DISTINCT run) FROM logs").fetchone()[0]
    print(f"# logs view: {nrows:,} rows across {nruns} run(s)")
    threads_val = con.execute("SELECT current_setting('threads')").fetchone()[0]
    print(f"# threads: {threads_val}")
    print(f"# duckdb: {duckdb.__version__}")
    print()

    header = f"{'query':30s}  {'runs':>4s}  {'cold_s':>7s}  {'warm_min_s':>10s}  {'warm_med_s':>10s}  {'rows':>10s}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, sql in QUERIES.items():
        sql_final = sql.format(RUN=RUN_NAME)
        times = []
        n_rows_out = None
        for i in range(args.repeats):
            t0 = time.perf_counter()
            rows = con.execute(sql_final).fetchall()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            n_rows_out = len(rows)
        cold = times[0]
        warm = times[1:]
        warm_min = min(warm) if warm else float("nan")
        warm_med = statistics.median(warm) if warm else float("nan")
        results[name] = {
            "cold_s": cold, "warm_min_s": warm_min, "warm_med_s": warm_med,
            "n_rows_out": n_rows_out, "all_times_s": times,
        }
        print(f"{name:30s}  {args.repeats:>4d}  {cold:>7.3f}  {warm_min:>10.3f}  {warm_med:>10.3f}  {n_rows_out:>10d}")

    print()
    warm_sum = sum(r["warm_min_s"] for r in results.values())
    warm_max = max(r["warm_min_s"] for r in results.values())
    print(f"# sum(warm_min) across all 5 primitives: {warm_sum:.3f} s")
    print(f"# max(warm_min) single primitive       : {warm_max:.3f} s")

if __name__ == "__main__":
    main()
