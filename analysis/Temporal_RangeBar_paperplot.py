"""
Paper-ready temporal-correlation figures.

For the 340B @ 2,048 GPUs run, draws range bars (whiskers from min to max, box
from p25 to p75, marker at p50) of two performance metrics
(bus bandwidth in GB/s, exec time in ms) against two temporal identifiers:

  - wall-clock time bin (default 5 s)
  - per-communicator collective sequence number bin (default 50 ops)

Two dominant buckets are shown side by side:

  - nic-only AllGather 32r/32n  ~13 MB
  - nvlink-only AllGather 8r/1n ~18 MB

Outputs (under figures/experiments/Measurement_vs_Counter/):

  temporal_pertimebin_rangebar_340b_gpus2048_1756587606.{pdf,png}
  temporal_percollsnbin_rangebar_340b_gpus2048_1756587606.{pdf,png}
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis"
DATA_GLOB = NIXT_ROOT + "/data/*-analysis/parquet_files/*.parquet"
FIG_DIR = Path(NIXT_ROOT + "/figures/experiments/Measurement_vs_Counter")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (label, comm_type, coll, msg_size_bytes)
BUCKETS = [
    ("nic-only AllGather 32r/32n 12.87 MiB",  "hca-only",    "AllGather", 13_492_224),
    ("nvlink-only AllGather 8r/1n 18 MiB",    "nvlink-only", "AllGather", 18_874_368),
]

# (metric_label, sql_expr, divisor_for_display, ylabel, color, normalize_to_pct)
METRICS = [
    ("Bus bandwidth",        "coll_busbw_gbs",    1.0,    "Norm BW",       True),
    ("Exec time (ms)",       "coll_exec_time_us", 1000.0, "Exec time (ms)", False),
]

SOFT = {"blue": "#8DA0CB", "orange": "#FC8D62", "teal": "#66C2A5", "gray": "#666666"}
BUCKET_COLORS = [SOFT["blue"], SOFT["teal"]]


def set_rcparams() -> None:
    mpl.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )


def open_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"""
CREATE OR REPLACE VIEW logs AS
SELECT regexp_extract(filename, '/data/([^/]+)/parquet_files/', 1) AS run, *
FROM read_parquet('{DATA_GLOB}', filename=true);
"""
    )
    return con


def per_bin_stats(
    con: duckdb.DuckDBPyConnection,
    *,
    bucket: tuple[str, str, str, int],
    bin_expr: str,
    metric_sql: str,
    divisor: float,
) -> pd.DataFrame:
    _, ct, coll, msg = bucket
    where = (
        f"run = '{RUN}' AND comm_type = '{ct}' AND coll = '{coll}' "
        f"AND coll_msg_size_bytes = {msg}"
    )
    sql = f"""
WITH base AS (
  SELECT *,
    (SELECT MIN(dump_timestamp_us) FROM logs WHERE {where}) AS t0_us
  FROM logs WHERE {where}
)
SELECT {bin_expr} AS x,
  COUNT(*) AS n,
  MIN({metric_sql}) / {divisor} AS lo,
  MAX({metric_sql}) / {divisor} AS hi,
  quantile_cont({metric_sql}, 0.25) / {divisor} AS p25,
  quantile_cont({metric_sql}, 0.50) / {divisor} AS p50,
  quantile_cont({metric_sql}, 0.75) / {divisor} AS p75
FROM base
GROUP BY x
HAVING COUNT(*) >= 50
ORDER BY x
"""
    return con.execute(sql).df()


def draw_rangebar(ax: plt.Axes, df: pd.DataFrame, color: str) -> None:
    x = df["x"].values
    ax.vlines(x, df["lo"].values, df["hi"].values, color=color, alpha=0.35, linewidth=0.6)
    ax.vlines(x, df["p25"].values, df["p75"].values, color=color, alpha=0.95, linewidth=1.6)
    ax.plot(x, df["p50"].values, "o", color="black", markersize=1.8, markerfacecolor="white", markeredgewidth=0.6)


def make_grid_fig(
    con: duckdb.DuckDBPyConnection,
    *,
    out_name: str,
    bin_expr: str,
    xlabel: str,
    title_suffix: str,
) -> list[str]:
    """Generate a 2x2 grid: rows=metrics (bus_bw, exec_time), cols=buckets."""
    set_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), constrained_layout=True, sharex="col")
    for row, (mlabel, msql, divisor, ylabel, normalize_to_pct) in enumerate(METRICS):
        for col, (bucket, color) in enumerate(zip(BUCKETS, BUCKET_COLORS)):
            ax = axes[row, col]
            df = per_bin_stats(con, bucket=bucket, bin_expr=bin_expr, metric_sql=msql, divisor=divisor)
            if df.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            if normalize_to_pct:
                denom = max(float(df["hi"].max()), 1e-12)
                for c in ("lo", "hi", "p25", "p50", "p75"):
                    if c in df.columns:
                        df[c] = df[c].astype(float) / denom * 100.0
            draw_rangebar(ax, df, color)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(bucket[0])
            if row == 1:
                ax.set_xlabel(xlabel)
            if row == 1 and ylabel.startswith("Exec"):
                ax.set_yscale("log")
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    out_pdf = FIG_DIR / f"{out_name}.pdf"
    out_png = FIG_DIR / f"{out_name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(out_pdf), str(out_png)]


def main() -> None:
    con = open_con()
    written: list[str] = []

    # 5-second wall-clock bins, x = bin start (s) measured from run start
    bin_expr_time = "FLOOR((dump_timestamp_us - t0_us) / 1e6 / 5) * 5"
    written += make_grid_fig(
        con,
        out_name="temporal_pertimebin_rangebar_340b_gpus2048_1756587606",
        bin_expr=bin_expr_time,
        xlabel="Wall-clock time (s, 5-s bins)",
        title_suffix="Per-call performance dispersion vs wall-clock time",
    )

    # 50-call coll_sn bins, x = bin start
    bin_expr_sn = "FLOOR(coll_sn / 50) * 50"
    written += make_grid_fig(
        con,
        out_name="temporal_percollsnbin_rangebar_340b_gpus2048_1756587606",
        bin_expr=bin_expr_sn,
        xlabel="Collective sequence number (50-op bins)",
        title_suffix="Per-call performance dispersion vs sequence number",
    )

    print("wrote:")
    for w in written:
        print(" -", w)


if __name__ == "__main__":
    main()
