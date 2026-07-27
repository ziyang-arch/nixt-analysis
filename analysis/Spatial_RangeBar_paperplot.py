"""
Paper-ready figures: per-communicator-id and per-host performance dispersion.

For the 340B @ 2,048 GPUs run, draws sorted "range bars" (whiskers from min to max,
box from p25 to p75, marker at p50) of per-call bus bandwidth, with one entity
(communicator id or hostname) per x position. NCCL Inspector logs rank-in-
communicator (not a global rank), so the meaningful cluster-level spatial axis
is the hostname; we use it for the per-rank-equivalent figure.

Two dominant buckets are shown:

  - hca-only AllGather 32r/32n  ~13 MB (inter-node data-parallel)
  - nvlink-only AllGather 8r/1n ~18 MB (intra-node tensor-parallel)

Outputs (under figures/experiments/Measurement_vs_Identifier/):

  spatial_perid_rangebar_340b_gpus2048_1756587606.{pdf,png}
  spatial_perhost_rangebar_340b_gpus2048_1756587606.{pdf,png}
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path
from typing import Sequence

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis"
DATA_GLOB = NIXT_ROOT + "/data/*-analysis/parquet_files/*.parquet"
FIG_DIR = Path(NIXT_ROOT + "/figures/experiments/Measurement_vs_Identifier")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (label, comm_type, coll, msg_size_bytes, min_ops_per_entity)
BUCKETS: list[tuple[str, str, str, int, int]] = [
    ("nic-only AllGather 32r/32n 12.87 MiB",  "hca-only",    "AllGather", 13_492_224, 50),
    ("nvlink-only AllGather 8r/1n 18 MiB",    "nvlink-only", "AllGather", 18_874_368, 50),
]

SOFT = {"blue": "#8DA0CB", "orange": "#FC8D62", "teal": "#66C2A5", "gray": "#666666"}


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


def per_entity_stats(
    con: duckdb.DuckDBPyConnection,
    entity_col: str,
    comm_type: str,
    coll: str,
    msg_size: int,
    min_ops: int,
) -> pd.DataFrame:
    sql = f"""
SELECT {entity_col} AS entity,
  COUNT(*) AS n,
  AVG(coll_busbw_gbs) AS mean_bw,
  MIN(coll_busbw_gbs) AS min_bw,
  MAX(coll_busbw_gbs) AS max_bw,
  quantile_cont(coll_busbw_gbs, 0.25) AS p25_bw,
  quantile_cont(coll_busbw_gbs, 0.50) AS p50_bw,
  quantile_cont(coll_busbw_gbs, 0.75) AS p75_bw,
  STDDEV_SAMP(coll_busbw_gbs) / NULLIF(AVG(coll_busbw_gbs), 0) AS cv_bw
FROM logs
WHERE run = '{RUN}'
  AND comm_type = '{comm_type}' AND coll = '{coll}' AND coll_msg_size_bytes = {msg_size}
GROUP BY {entity_col}
HAVING COUNT(*) >= {min_ops}
ORDER BY p50_bw ASC
"""
    return con.execute(sql).df().reset_index(drop=True)


def draw_rangebar(ax: plt.Axes, df: pd.DataFrame, color: str) -> None:
    x = np.arange(len(df))
    # whiskers: min..max
    ax.vlines(x, df["min_bw"].values, df["max_bw"].values, color=color, alpha=0.35, linewidth=0.6)
    # IQR box drawn as a thick vlines
    ax.vlines(x, df["p25_bw"].values, df["p75_bw"].values, color=color, alpha=0.95, linewidth=1.6)
    # p50 marker
    ax.plot(x, df["p50_bw"].values, "o", color="black", markersize=1.6, markerfacecolor="white", markeredgewidth=0.6)


def per_comm_rank_stats(
    con: duckdb.DuckDBPyConnection,
    comm_type: str,
    coll: str,
    msg_size: int,
    min_ops: int,
) -> pd.DataFrame:
    """Stats per (communicator id, rank-in-communicator) cell."""
    sql = f"""
SELECT id, rank,
  COUNT(*) AS n,
  AVG(coll_busbw_gbs) AS mean_bw,
  MIN(coll_busbw_gbs) AS min_bw,
  MAX(coll_busbw_gbs) AS max_bw,
  quantile_cont(coll_busbw_gbs, 0.25) AS p25_bw,
  quantile_cont(coll_busbw_gbs, 0.50) AS p50_bw,
  quantile_cont(coll_busbw_gbs, 0.75) AS p75_bw
FROM logs
WHERE run = '{RUN}'
  AND comm_type = '{comm_type}' AND coll = '{coll}' AND coll_msg_size_bytes = {msg_size}
GROUP BY id, rank
HAVING COUNT(*) >= {min_ops}
ORDER BY id, rank
"""
    return con.execute(sql).df().reset_index(drop=True)


def plot_per_commrank_panel(
    con: duckdb.DuckDBPyConnection,
    out_name: str,
) -> list[str]:
    """1x2 grid: for each bucket, plot a range bar at every (comm, rank) cell,
    ordered as contiguous comm blocks. Thin vertical dividers separate comms."""
    set_rcparams()
    fig, axes = plt.subplots(1, len(BUCKETS), figsize=(8.5, 3.2), constrained_layout=True)
    palette = [SOFT["blue"], SOFT["teal"]]
    for ax, (label, ct, coll, msg, minops), color in zip(axes, BUCKETS, palette):
        df = per_comm_rank_stats(con, ct, coll, msg, minops)
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        bw_cols = ["min_bw", "max_bw", "p25_bw", "p50_bw", "p75_bw", "mean_bw"]
        denom = max(float(df["max_bw"].max()), 1e-12)
        for c in bw_cols:
            if c in df.columns:
                df[c] = df[c].astype(float) / denom * 100.0

        comm_ids = list(df["id"].drop_duplicates())[:8]
        df = df[df["id"].isin(comm_ids)].reset_index(drop=True)
        comm_index = {cid: i for i, cid in enumerate(comm_ids)}
        n_ranks_per_comm = int(df.groupby("id")["rank"].nunique().max())
        x = df["id"].map(comm_index).to_numpy() * n_ranks_per_comm + df["rank"].to_numpy()

        lw_whisker = 0.3 if len(x) > 500 else 0.6
        lw_iqr = 0.6 if len(x) > 500 else 1.4
        ax.vlines(x, df["min_bw"].values, df["max_bw"].values, color=color, alpha=0.35, linewidth=lw_whisker)
        ax.vlines(x, df["p25_bw"].values, df["p75_bw"].values, color=color, alpha=0.95, linewidth=lw_iqr)
        if len(x) <= 500:
            ax.plot(x, df["p50_bw"].values, "o", color="black", markersize=1.2,
                    markerfacecolor="white", markeredgewidth=0.4)
        for i in range(1, len(comm_ids)):
            ax.axvline(i * n_ranks_per_comm - 0.5, color="gray", linewidth=0.3, alpha=0.4)

        ax.set_xlabel(f"communicator block (each block = {n_ranks_per_comm} ranks)")
        ax.set_ylabel("Norm BW")
        ax.set_title(" ".join(label.split()[:2]))
        ax.set_xlim(-0.5, len(comm_ids) * n_ranks_per_comm - 0.5)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    out_pdf = FIG_DIR / f"{out_name}.pdf"
    out_png = FIG_DIR / f"{out_name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(out_pdf), str(out_png)]


def plot_perentity_panel(
    con: duckdb.DuckDBPyConnection,
    entity_col: str,
    out_name: str,
    entity_xlabel: str,
    use_entity_value: bool = False,
) -> list[str]:
    set_rcparams()
    fig, axes = plt.subplots(1, len(BUCKETS), figsize=(8.5, 3.2), constrained_layout=True)
    palette = [SOFT["blue"], SOFT["teal"]]
    for ax, (label, ct, coll, msg, minops), color in zip(axes, BUCKETS, palette):
        df = per_entity_stats(con, entity_col, ct, coll, msg, minops)
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        # Normalize by panel max so all bw values are in % of panel peak.
        bw_cols = ["min_bw", "max_bw", "p25_bw", "p50_bw", "p75_bw", "mean_bw"]
        denom = max(float(df["max_bw"].max()), 1e-12)
        df_norm = df.copy()
        for c in bw_cols:
            if c in df_norm.columns:
                df_norm[c] = df_norm[c].astype(float) / denom * 100.0
        if use_entity_value:
            df_norm = df_norm.sort_values("entity").reset_index(drop=True)
            x = df_norm["entity"].astype(int).to_numpy()
            ax.vlines(x, df_norm["min_bw"].values, df_norm["max_bw"].values, color=color, alpha=0.35, linewidth=0.6)
            ax.vlines(x, df_norm["p25_bw"].values, df_norm["p75_bw"].values, color=color, alpha=0.95, linewidth=1.6)
            ax.plot(x, df_norm["p50_bw"].values, "o", color="black", markersize=1.6, markerfacecolor="white", markeredgewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x])
            ax.set_xlabel(entity_xlabel)
        else:
            draw_rangebar(ax, df_norm, color)
            ax.set_xlabel(f"{entity_xlabel} index")
        ax.set_ylabel("Norm BW")
        ax.set_title(label)
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
    written += plot_per_commrank_panel(
        con,
        out_name="spatial_percommrank_rangebar_340b_gpus2048_1756587606",
    )
    written += plot_perentity_panel(
        con, entity_col="hostname",
        out_name="spatial_perhost_rangebar_340b_gpus2048_1756587606",
        entity_xlabel="hostname",
    )
    print("wrote:")
    for w in written:
        print(" -", w)


if __name__ == "__main__":
    main()
