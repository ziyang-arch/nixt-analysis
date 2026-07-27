#!/usr/bin/env python3
"""
Generate all paper plots for the NCCL Inspector × Nemotron-4 case study.

This script is the "non-notebook" companion to `summary_all.ipynb`:
- Loads exported Parquet logs via DuckDB
- Computes the same aggregates
- Saves paper-styled figures into the figures directory

Default output directory:
  $NIXT_ROOT/figures

Output conventions (paper-ready):
- Figures are saved in vector PDF format and PNG by default (PNG is for quick preview).
- Plots report normalized values (percent shares or x / max), not raw bytes/GB/s.
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import duckdb  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, ScalarFormatter  # type: ignore
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "a required package")
    raise SystemExit(
        f"Missing Python dependency: {missing}\n\n"
        "Run this script inside the same environment used by the notebook.\n"
        "For this repo, that is typically:\n"
        "  source \"$(conda info --base)/etc/profile.d/conda.sh\"\n"
        "  conda activate nccl_exporter\n"
        "  python exporter/summary_scaling_paperplot.py\n\n"
        "Or install deps into your current env (not recommended for shared systems):\n"
        "  pip install duckdb pandas numpy matplotlib\n"
    )


# -----------------------------
# Styling (paper / low-contrast)
# -----------------------------

SOFT = {
    # Set2-like (lower contrast) – consistent with notebook edits
    "blue": "#8DA0CB",
    "teal": "#66C2A5",
    "orange": "#FC8D62",
    "pink": "#E78AC3",
    "gray": "#B3B3B3",
}

COLL_COLORS = {
    "AllReduce": SOFT["blue"],
    "ReduceScatter": SOFT["orange"],
    "AllGather": SOFT["teal"],
    "Broadcast": SOFT["pink"],
}

MODEL_COLORS = {
    "15b": SOFT["blue"],
    "340b": SOFT["orange"],
}

COMM_COLORS = {
    "nic-only": SOFT["blue"],
    "nvlink-only": SOFT["teal"],
    "mixed": SOFT["orange"],
    "single-rank": SOFT["gray"],
}
COMM_DISPLAY_REMAP = {"hca-only": "nic-only"}


def set_rcparams():
    mpl.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 0.8,
        }
    )


def legend_bottom(ax: plt.Axes, ncol: int = 2, y: float = -0.28, title: Optional[str] = None):
    """Put legend at the bottom of the plot (below axes), per paper_structure.md."""
    ax.legend(
        title=title,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        ncol=ncol,
        columnspacing=0.9,
        handlelength=1.2,
    )


def set_gpu_xticks_only(ax: plt.Axes, gpus: List[int] | np.ndarray | pd.Series) -> None:
    """
    For scaling plots where x is numeric GPU count: show ticks only where we have data.
    This keeps a true numeric axis (so gaps reflect missing GPU counts) while decluttering labels.
    """
    g = sorted({int(x) for x in list(gpus) if pd.notna(x)})
    if not g:
        return
    ax.set_xticks(g)
    ax.set_xlim(min(g) - 0.5, max(g) + 0.5)


def set_gpu_xaxis_log2(ax: plt.Axes, gpus: List[int] | np.ndarray | pd.Series) -> None:
    """
    For scaling plots where x is numeric GPU count: use a log2 x-axis and show ticks only
    where we have data (e.g., 16, 32, ...). This preserves numeric spacing while decluttering.
    """
    g = sorted({int(x) for x in list(gpus) if pd.notna(x) and int(x) > 0})
    if not g:
        return

    ax.set_xscale("log", base=2)
    ax.set_xticks(g)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Pad in multiplicative space (log axis cannot include 0).
    lo = float(min(g)) / 1.35
    hi = float(max(g)) * 1.35
    ax.set_xlim(lo, hi)


# -----------------------------
# DuckDB + IO helpers
# -----------------------------


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def bytes_to_hr(n: int) -> str:
    n = int(n)
    for unit, step in [("B", 1024), ("KB", 1024), ("MB", 1024), ("GB", 1024), ("TB", 1024)]:
        if n < step:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n = n / step
    return f"{n:.2f}PB"


def connect_logs(data_root: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    parquet_glob = f"{data_root}/*-analysis/parquet_files/*.parquet"
    con.execute(
        f"""
CREATE OR REPLACE VIEW logs AS
SELECT
  regexp_extract(filename, '/data/([^/]+)/parquet_files/', 1) AS run,
  *
FROM read_parquet('{parquet_glob}', filename=true);
"""
    )
    return con


def q(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def savefig(fig_dir: Path, name: str, formats: List[str], dpi: int = 300) -> List[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    wrote: List[str] = []
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip(".")
        out = fig_dir / f"{name}.{fmt}"
        if fmt in {"pdf", "svg", "eps"}:
            plt.savefig(out, bbox_inches="tight")
        else:
            plt.savefig(out, dpi=dpi, bbox_inches="tight")
        wrote.append(str(out))
    return wrote


def normalize_to_percent(x: pd.Series, denom: float) -> pd.Series:
    denom = float(denom)
    if denom <= 0:
        return x.astype(float) * 0.0
    return (x.astype(float) / denom) * 100.0


# -----------------------------
# Run metadata parsing
# -----------------------------

_run_re_gpus = re.compile(r"gpus(?P<gpus>\d+)")

# From README: these timestamps correspond to "straggler node" 15B@16 experiments.
STRAGGLER_RUN_TS = {"1757455306", "1757716170"}


def parse_run_meta(run: str) -> dict:
    r = run.lower()
    m = _run_re_gpus.search(r)
    gpus = int(m.group("gpus")) if m else None

    if "nemotron4_15b" in r:
        model = "15b"
    elif "nemotron4_340b" in r:
        model = "340b"
    else:
        model = None

    is_straggler = any(ts in run for ts in STRAGGLER_RUN_TS)
    return {"run": run, "gpus": gpus, "model": model, "is_straggler": is_straggler}


def load_meta(con: duckdb.DuckDBPyConnection, exclude_stragglers: bool = True) -> pd.DataFrame:
    runs_df = q(con, "SELECT DISTINCT run FROM logs").copy()
    runs_df["run"] = runs_df["run"].astype(str)
    meta = pd.DataFrame([parse_run_meta(r) for r in runs_df["run"].tolist()])
    meta = meta.dropna(subset=["gpus", "model"]).reset_index(drop=True)
    meta = meta[meta["run"].str.contains("pretrain_nemotron4_", case=False, na=False)].copy()
    meta.sort_values(["model", "gpus", "run"], inplace=True)

    if exclude_stragglers:
        meta = meta[~meta["is_straggler"]].copy().reset_index(drop=True)
    return meta


# -----------------------------
# Plots: single-run summaries
# -----------------------------


def plot_bytes_by_coll_commtype(con, fig_dir: Path, run: str, tag: str) -> List[str]:
    set_rcparams()
    d = q(
        con,
        f"""
SELECT coll, comm_type, SUM(coll_msg_size_bytes) AS bytes
FROM logs
WHERE run = '{run}'
GROUP BY coll, comm_type
""",
    )
    keep_colls = ["AllGather", "ReduceScatter", "AllReduce", "Broadcast"]
    d = d[d["coll"].isin(keep_colls)].copy()
    d["comm_type"] = d["comm_type"].replace(COMM_DISPLAY_REMAP)
    total_bytes = float(d["bytes"].sum())
    d["pct"] = normalize_to_percent(d["bytes"], total_bytes)
    pivot = (
        d.pivot_table(index="coll", columns="comm_type", values="pct", aggfunc="sum", fill_value=0)
        .reindex(index=[c for c in keep_colls if c in d["coll"].unique()])
        .reindex(columns=[c for c in ["nic-only", "nvlink-only", "mixed", "single-rank"] if c in d["comm_type"].unique()])
    )
    color_list = [COMM_COLORS.get(c, SOFT["gray"]) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    pivot.plot(kind="bar", ax=ax, width=0.82, color=color_list, edgecolor="none")
    ax.set_xlabel("Collective")
    ax.set_ylabel("Normalized comm. volume")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.6)
    legend_bottom(ax, ncol=2, y=-0.28)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"{slug(tag)}_bytes_by_coll_commtype", formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_top_msg_sizes_by_bytes(con, fig_dir: Path, run: str, tag: str, limit: int = 10) -> List[str]:
    set_rcparams()
    d = q(
        con,
        f"""
SELECT coll, coll_msg_size_bytes, COUNT(*) AS ops, SUM(coll_msg_size_bytes) AS bytes
FROM logs
WHERE run = '{run}'
GROUP BY coll, coll_msg_size_bytes
ORDER BY bytes DESC
LIMIT {int(limit)}
""",
    ).copy()
    d["label"] = d.apply(lambda r: f"{r['coll']} {float(r['coll_msg_size_bytes'])/1e6:.1f}MB", axis=1)
    total_bytes = float(d["bytes"].sum()) if float(d["bytes"].sum()) > 0 else 1.0
    # Normalize within the displayed top-K set (so the chart is interpretable even if tail exists).
    d["pct"] = normalize_to_percent(d["bytes"], total_bytes)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.barh(range(len(d))[::-1], d["pct"].astype(float).values[::-1], color=SOFT["blue"], edgecolor="none")
    ax.set_yticks(range(len(d))[::-1])
    ax.set_yticklabels(d["label"].values[::-1], fontsize=7)
    ax.set_xlabel("Share of top-K bytes (%)")
    ax.set_xlim(0, max(1.0, float(d["pct"].max()) * 1.1))
    ax.grid(True, which="major", axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"{slug(tag)}_top_msg_sizes_by_bytes", formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_busbw_percentiles(
    con, fig_dir: Path, run: str, tag: str, min_msg_bytes: int = 10_000_000, limit: int = 10
) -> List[str]:
    set_rcparams()
    d = q(
        con,
        f"""
SELECT
  coll,
  comm_type,
  coll_msg_size_bytes,
  COUNT(*) AS ops,
  quantile_cont(coll_busbw_gbs, 0.1) AS p10,
  quantile_cont(coll_busbw_gbs, 0.5) AS p50,
  quantile_cont(coll_busbw_gbs, 0.9) AS p90
FROM logs
WHERE run = '{run}' AND coll_msg_size_bytes >= {int(min_msg_bytes)}
GROUP BY coll, comm_type, coll_msg_size_bytes
ORDER BY ops DESC
LIMIT {int(limit)}
""",
    ).copy()

    if d.empty:
        raise RuntimeError(f"No bus bandwidth samples for run={run} (min_msg_bytes={min_msg_bytes})")

    d["comm_type"] = d["comm_type"].replace(COMM_DISPLAY_REMAP)

    def short_label(r) -> str:
        mb = float(r["coll_msg_size_bytes"]) / 1e6
        return f"{r['comm_type']} {r['coll']} {mb:.0f}MB"

    labels = [short_label(r) for _, r in d.iterrows()]
    x = np.arange(len(d))
    # Normalize to percent of max median (p50) within this figure.
    p50_max = float(d["p50"].max())
    y = (d["p50"].astype(float) / max(p50_max, 1e-12) * 100.0).values
    yerr_low = ((d["p50"] - d["p10"]).astype(float) / max(p50_max, 1e-12) * 100.0).values
    yerr_high = ((d["p90"] - d["p50"]).astype(float) / max(p50_max, 1e-12) * 100.0).values

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.errorbar(
        x,
        y,
        yerr=[yerr_low, yerr_high],
        fmt="o",
        capsize=2.5,
        color=SOFT["blue"],
        markersize=3.5,
        linewidth=1.0,
    )
    ax.set_ylabel("Norm BW")
    ax.set_xlabel("Dominant message types")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0, 110)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"{slug(tag)}_busbw_percentiles", formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_compare_15b_340b_topmsg_busbw_2x2(
    con,
    fig_dir: Path,
    run_15b: str,
    run_340b: str,
    *,
    out_name: str = "compare_15b_340b_2048_topmsg_and_busbw",
    top_limit: int = 10,
    min_msg_bytes: int = 1_000_000,
    bw_limit: int = 10,
) -> List[str]:
    """2x2 grid: rows = (top message sizes, bus BW percentiles); cols = (15B@2048, 340B@2048)."""
    set_rcparams()

    def top_msg(run: str) -> pd.DataFrame:
        return q(
            con,
            f"""
SELECT coll, coll_msg_size_bytes, COUNT(*) AS ops, SUM(coll_msg_size_bytes) AS bytes
FROM logs WHERE run = '{run}'
GROUP BY coll, coll_msg_size_bytes
ORDER BY bytes DESC LIMIT {int(top_limit)}
""",
        ).copy()

    def busbw(run: str) -> pd.DataFrame:
        d = q(
            con,
            f"""
SELECT coll, comm_type, coll_msg_size_bytes, COUNT(*) AS ops,
  quantile_cont(coll_busbw_gbs, 0.1) AS p10,
  quantile_cont(coll_busbw_gbs, 0.5) AS p50,
  quantile_cont(coll_busbw_gbs, 0.9) AS p90
FROM logs WHERE run = '{run}' AND coll_msg_size_bytes >= {int(min_msg_bytes)}
GROUP BY coll, comm_type, coll_msg_size_bytes
ORDER BY ops DESC LIMIT {int(bw_limit)}
""",
        ).copy()
        if not d.empty:
            d["comm_type"] = d["comm_type"].replace(COMM_DISPLAY_REMAP)
        return d

    panels = [
        ("Nemotron-4 15B @ 2,048 GPUs",  top_msg(run_15b),  busbw(run_15b)),
        ("Nemotron-4 340B @ 2,048 GPUs", top_msg(run_340b), busbw(run_340b)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), constrained_layout=True)

    for col, (title, top_df, bw_df) in enumerate(panels):
        # Top row: top message sizes
        ax = axes[0, col]
        d = top_df.copy()
        if d.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            d["label"] = d.apply(lambda r: f"{r['coll']} {float(r['coll_msg_size_bytes'])/1e6:.1f}MB", axis=1)
            total_bytes = float(d["bytes"].sum()) if float(d["bytes"].sum()) > 0 else 1.0
            d["pct"] = normalize_to_percent(d["bytes"], total_bytes)
            ax.barh(range(len(d))[::-1], d["pct"].astype(float).values[::-1], color=SOFT["blue"], edgecolor="none")
            ax.set_yticks(range(len(d))[::-1])
            ax.set_yticklabels(d["label"].values[::-1], fontsize=8)
            ax.set_xlim(0, max(1.0, float(d["pct"].max()) * 1.1))
            ax.grid(True, which="major", axis="x", alpha=0.25, linewidth=0.6)
        ax.set_xlabel("Share of top-K bytes (%)")
        ax.set_title(title)

        # Bottom row: bus BW percentiles
        ax = axes[1, col]
        d = bw_df.copy()
        if d.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        else:
            def _short(r):
                mb = float(r["coll_msg_size_bytes"]) / 1e6
                return f"{r['comm_type']} {r['coll']} {mb:.0f}MB"
            labels = [_short(r) for _, r in d.iterrows()]
            x = np.arange(len(d))
            p50_max = float(d["p50"].max())
            y = (d["p50"].astype(float) / max(p50_max, 1e-12) * 100.0).values
            yerr_low = ((d["p50"] - d["p10"]).astype(float) / max(p50_max, 1e-12) * 100.0).values
            yerr_high = ((d["p90"] - d["p50"]).astype(float) / max(p50_max, 1e-12) * 100.0).values
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=2.5,
                        color=SOFT["blue"], markersize=3.5, linewidth=1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_ylim(0, 110)
        ax.set_xlabel("Dominant message types")
        if col == 0:
            ax.set_ylabel("Norm BW")

    out = savefig(fig_dir, out_name, formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_scatter_nic(con, fig_dir: Path, run: str, tag: str, coll: str) -> List[str]:
    set_rcparams()
    d = q(
        con,
        f"""
SELECT
  coll_sn,
  coll_msg_size_bytes,
  AVG(coll_busbw_gbs) AS mean_coll_busbw_gbs,
  COUNT(*) AS log_count
FROM logs
WHERE run = '{run}' AND comm_type = 'hca-only' AND coll = '{coll}'
GROUP BY coll_sn, coll_msg_size_bytes
ORDER BY coll_sn
""",
    ).copy()
    if d.empty:
        raise RuntimeError(f"No nic-only data for run={run} coll={coll}")

    y_max = float(d["mean_coll_busbw_gbs"].max())
    d["bw_pct"] = d["mean_coll_busbw_gbs"].astype(float) / max(y_max, 1e-12) * 100.0

    palette = [SOFT["blue"], SOFT["teal"], SOFT["orange"], SOFT["pink"], SOFT["gray"]]
    msg_sizes = d.groupby("coll_msg_size_bytes")["log_count"].sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, ms in enumerate(msg_sizes):
        sub = d[d["coll_msg_size_bytes"] == ms]
        mean_bw = float(sub["bw_pct"].mean())
        ax.scatter(
            sub["coll_sn"].astype(int),
            sub["bw_pct"].astype(float),
            s=6,
            alpha=0.55,
            color=palette[i % len(palette)],
            edgecolors="none",
            label=f"{bytes_to_hr(ms)} (mean {mean_bw:.1f}%)",
        )

    ax.set_xlabel("Op seq. no. (coll_sn)")
    ax.set_ylabel("Norm BW")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0, 110)
    legend_bottom(ax, ncol=2, y=-0.30, title="Msg size")
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"{slug(tag)}_scatter_nic-only_{coll}", formats=["pdf", "png"])
    plt.close(fig)
    return out


# -----------------------------
# Plots: scaling (GPU count + model size)
# -----------------------------


def compute_bytes_by_run_coll(con, meta: pd.DataFrame) -> pd.DataFrame:
    agg = q(
        con,
        """
SELECT run, coll, SUM(coll_msg_size_bytes) AS bytes
FROM logs
GROUP BY run, coll
""",
    )
    agg["bytes"] = agg["bytes"].astype("int64")
    df = agg.merge(meta[["run", "gpus", "model"]], on="run", how="inner")
    df["tb"] = df["bytes"] / 1e12
    return df


def plot_scaling_15b_bytes_by_coll_vs_gpus(fig_dir: Path, df: pd.DataFrame) -> List[str]:
    set_rcparams()
    d15 = df[df["model"] == "15b"].copy()
    pivot_bytes = (
        d15.groupby(["gpus", "coll"])["bytes"].sum().reset_index().pivot_table(index="gpus", columns="coll", values="bytes", aggfunc="sum", fill_value=0).sort_index()
    )
    coll_order = [c for c in ["AllReduce", "ReduceScatter", "AllGather", "Broadcast"] if c in pivot_bytes.columns]
    pivot_bytes = pivot_bytes.reindex(columns=coll_order)
    totals = pivot_bytes.sum(axis=1).replace(0, np.nan)
    pivot_pct = pivot_bytes.div(totals, axis=0) * 100.0
    color_list = [COLL_COLORS.get(c, SOFT["gray"]) for c in pivot_bytes.columns]

    # Slightly larger figure; keep GPU tick labels horizontal for paper readability.
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    pivot_pct.plot(kind="bar", stacked=False, ax=ax, width=0.82, color=color_list, edgecolor="none")
    ax.set_xlabel("GPUs", fontsize=16)
    ax.set_ylabel("Norm Comm Vol")
    ax.tick_params(axis="x", labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.6)
    legend_bottom(ax, ncol=2, y=-0.28)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, "scaling_15b_bytes_by_coll_vs_gpus", formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_scaling_15b_total_bytes_vs_gpus(fig_dir: Path, df: pd.DataFrame) -> List[str]:
    set_rcparams()
    d15 = df[df["model"] == "15b"].copy()
    totals = d15.groupby("gpus")["bytes"].sum().reset_index().sort_values("gpus")
    # Normalize bytes per GPU to max across GPU counts (within the 15B scaling set).
    totals["bytes_per_gpu"] = totals["bytes"].astype(float) / totals["gpus"].astype(float)
    denom = float(totals["bytes_per_gpu"].max()) if len(totals) else 1.0
    totals["bytes_per_gpu_norm_pct"] = totals["bytes_per_gpu"] / max(denom, 1e-12) * 100.0

    # Larger fonts for this figure (paper readability)
    with mpl.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.plot(
            totals["gpus"].astype(int),
            totals["bytes_per_gpu_norm_pct"].astype(float),
            marker="o",
            linewidth=1.4,
            markersize=4.2,
            color=SOFT["blue"],
        )
        ax.set_xlabel("GPUs", fontsize=16)
        ax.set_ylabel("Norm Comm Vol")
        set_gpu_xaxis_log2(ax, totals["gpus"])
        ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylim(0, max(110, float(totals["bytes_per_gpu_norm_pct"].max()) * 1.1))
        fig.tight_layout(pad=0.2)
        out = savefig(fig_dir, "scaling_15b_total_bytes_vs_gpus", formats=["pdf", "png"])
        plt.close(fig)
        return out


def compute_bw_by_run_coll(con, meta: pd.DataFrame, min_msg_bytes: int = 1_000_000) -> pd.DataFrame:
    bw_agg = q(
        con,
        f"""
SELECT
  run,
  coll,
  COUNT(*) AS ops,
  SUM(coll_msg_size_bytes) AS bytes,
  SUM(coll_busbw_gbs * coll_msg_size_bytes) / NULLIF(SUM(coll_msg_size_bytes), 0) AS bw_wmean_gbs,
  quantile_cont(coll_busbw_gbs, 0.5) AS bw_p50_gbs
FROM logs
WHERE coll_msg_size_bytes >= {int(min_msg_bytes)}
GROUP BY run, coll
""",
    ).copy()
    bw_agg["bytes"] = bw_agg["bytes"].astype("int64")
    bw = bw_agg.merge(meta[["run", "gpus", "model"]], on="run", how="inner")
    return bw


def plot_scaling_15b_busbw_by_coll_vs_gpus(fig_dir: Path, bw: pd.DataFrame) -> List[str]:
    set_rcparams()
    bw15 = bw[bw["model"] == "15b"].copy()

    # combine multiple runs at same GPU count by bytes-weighted averaging
    bw15_g = (
        bw15.groupby(["gpus", "coll"])
        .apply(
            lambda x: pd.Series(
                {
                    "bw_wmean_gbs": (x["bw_wmean_gbs"] * x["bytes"]).sum() / max(x["bytes"].sum(), 1),
                    "bytes": x["bytes"].sum(),
                }
            )
        )
        .reset_index()
        .sort_values(["gpus", "coll"])
    )

    coll_order = [c for c in ["AllGather", "ReduceScatter", "AllReduce"] if c in bw15_g["coll"].unique()]
    # Normalize each collective's BW to its max across GPU counts (within the 15B scaling set).
    bw_max = bw15_g.groupby("coll")["bw_wmean_gbs"].max().to_dict()

    # Larger fonts for this figure (paper readability)
    with mpl.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for coll in coll_order:
            sub = bw15_g[bw15_g["coll"] == coll].sort_values("gpus")
            denom = float(bw_max.get(coll, 0.0))
            y_pct = sub["bw_wmean_gbs"].astype(float) / max(denom, 1e-12) * 100.0
            ax.plot(
                sub["gpus"].astype(int),
                y_pct,
                marker="o",
                linewidth=1.4,
                markersize=4.2,
                color=COLL_COLORS.get(coll, SOFT["gray"]),
                label=coll,
            )

        ax.set_xlabel("GPUs", fontsize=16)
        ax.set_ylabel("Norm BW")
        set_gpu_xaxis_log2(ax, bw15_g["gpus"])
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_ylim(0, 110)
        legend_bottom(ax, ncol=3, y=-0.28)
        fig.tight_layout(pad=0.2)
        out = savefig(fig_dir, "scaling_15b_busbw_by_coll_vs_gpus", formats=["pdf", "png"])
        plt.close(fig)
        return out


def plot_compare_modelsize_gpus2048_bytes_by_coll(fig_dir: Path, df: pd.DataFrame, g: int = 2048) -> List[str]:
    set_rcparams()
    models = ["15b", "340b"]
    dd = df[(df["gpus"] == g) & (df["model"].isin(models))].copy()
    if dd.empty:
        raise RuntimeError(f"No rows for gpus={g} in both models")

    coll_order = [c for c in ["AllGather", "ReduceScatter", "AllReduce", "Broadcast"] if c in dd["coll"].unique()]
    pivot_bytes = (
        dd.groupby(["model", "coll"])["bytes"]
        .sum()
        .reset_index()
        .pivot_table(index="coll", columns="model", values="bytes", aggfunc="sum", fill_value=0)
        .reindex(index=coll_order)
    )
    # Normalize within each model to show composition (% of model total at this GPU count).
    pivot_pct = pivot_bytes.div(pivot_bytes.sum(axis=0).replace(0, np.nan), axis=1) * 100.0
    color_list = [MODEL_COLORS.get(m, SOFT["gray"]) for m in pivot_bytes.columns]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    pivot_pct.plot(kind="bar", ax=ax, width=0.82, color=color_list, edgecolor="none")
    ax.set_xlabel("Collective", fontsize=16)
    ax.set_ylabel("Norm Comm Vol")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.6)
    legend_bottom(ax, ncol=2, y=-0.28)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"compare_modelsize_gpus{g}_bytes_by_coll", formats=["pdf", "png"])
    plt.close(fig)
    return out


def plot_compare_modelsize_gpus2048_busbw_by_coll(fig_dir: Path, bw: pd.DataFrame, g: int = 2048) -> List[str]:
    set_rcparams()
    models = ["15b", "340b"]
    bw_same = bw[(bw["gpus"] == g) & (bw["model"].isin(models))].copy()
    if bw_same.empty:
        raise RuntimeError(f"No bandwidth rows for gpus={g} in both models")

    # bytes-weighted average across reruns at same (gpus, model, coll)
    bw_same_g = (
        bw_same.groupby(["model", "coll"])
        .apply(lambda x: (x["bw_wmean_gbs"] * x["bytes"]).sum() / max(x["bytes"].sum(), 1))
        .reset_index(name="bw_wmean_gbs")
    )
    coll_order = [c for c in ["AllGather", "ReduceScatter", "AllReduce"] if c in bw_same_g["coll"].unique()]
    bw_same_g["coll"] = pd.Categorical(bw_same_g["coll"], categories=coll_order, ordered=True)
    bw_same_g.sort_values(["coll", "model"], inplace=True)
    pivot_bw = bw_same_g.pivot_table(index="coll", columns="model", values="bw_wmean_gbs", aggfunc="sum", fill_value=0)
    # Normalize per collective to % of the best model at this GPU count.
    pivot_pct = pivot_bw.div(pivot_bw.max(axis=1).replace(0, np.nan), axis=0) * 100.0
    color_list = [MODEL_COLORS.get(m, SOFT["gray"]) for m in pivot_bw.columns]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    pivot_pct.plot(kind="bar", ax=ax, width=0.82, color=color_list, edgecolor="none")
    ax.set_xlabel("Collective", fontsize=16)
    ax.set_ylabel("Norm BW")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    legend_bottom(ax, ncol=2, y=-0.28)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, f"compare_modelsize_gpus{g}_busbw_by_coll", formats=["pdf", "png"])
    plt.close(fig)
    return out


# -----------------------------
# Motif plot (time-sequence view)
# -----------------------------


@dataclass
class Motif:
    token_seq: Tuple[str, ...]
    positions: List[int]
    count: int


def load_event_stream(con, run: str, start_us: Optional[int], end_us: Optional[int], min_msg_bytes: int, keep_colls: List[str]) -> pd.DataFrame:
    where = [f"run = '{run}'"]
    if start_us is not None:
        where.append(f"dump_timestamp_us >= {int(start_us)}")
    if end_us is not None:
        where.append(f"dump_timestamp_us <= {int(end_us)}")
    if min_msg_bytes > 0:
        where.append(f"coll_msg_size_bytes >= {int(min_msg_bytes)}")
    if keep_colls:
        cols = ",".join([f"'{c}'" for c in keep_colls])
        where.append(f"coll IN ({cols})")
    where_sql = " AND ".join(where)

    df = q(
        con,
        f"""
SELECT
  dump_timestamp_us,
  coll,
  comm_type,
  coll_msg_size_bytes,
  coll_busbw_gbs,
  coll_exec_time_us,
  n_ranks,
  nnodes,
  id,
  coll_sn
FROM logs
WHERE {where_sql}
ORDER BY dump_timestamp_us ASC
""",
    )
    df["comm_type"] = df["comm_type"].replace(COMM_DISPLAY_REMAP)
    return df


def make_tokens(df: pd.DataFrame) -> List[str]:
    return (
        df["coll"].astype(str)
        + "|"
        + df["comm_type"].astype(str)
        + "|"
        + df["coll_msg_size_bytes"].astype("int64").astype(str)
    ).tolist()


def find_repeated_ngrams(tokens: List[str], n: int, min_count: int, topk: int) -> List[Motif]:
    if n <= 0 or len(tokens) < n:
        return []
    idx: Dict[Tuple[str, ...], List[int]] = {}
    for i in range(0, len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        idx.setdefault(key, []).append(i)
    motifs = [Motif(k, v, len(v)) for k, v in idx.items() if len(v) >= min_count]
    motifs.sort(key=lambda m: (m.count, -len(m.token_seq)), reverse=True)
    return motifs[:topk]


def plot_motif_occurrence(fig_dir: Path, df: pd.DataFrame, motif: Motif, occurrence: int, out_name: str) -> List[str]:
    set_rcparams()
    start_i = motif.positions[occurrence]
    end_i = start_i + len(motif.token_seq)
    sub = df.iloc[start_i:end_i].copy()
    t0 = int(sub["dump_timestamp_us"].iloc[0])
    sub["t_sec"] = (sub["dump_timestamp_us"].astype("int64") - t0) / 1e6

    cats = pd.Categorical(sub["coll"].astype(str), categories=sorted(pd.unique(sub["coll"].astype(str))))
    sub["y"] = cats.codes
    ytick_labels = list(cats.categories)

    sizes = np.sqrt(sub["coll_msg_size_bytes"].astype(float).clip(lower=1.0))
    sizes = 20.0 * (sizes / sizes.max())
    sizes = sizes.clip(lower=4.0)

    comm_cats = pd.Categorical(sub["comm_type"].astype(str))
    colors = comm_cats.codes

    fig, ax = plt.subplots(figsize=(4.8, 2.0))
    ax.scatter(sub["t_sec"], sub["y"], s=sizes, c=colors, alpha=0.75)
    ax.set_yticks(range(len(ytick_labels)))
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Time since motif start (s)")
    ax.set_ylabel("Collective")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.2)
    out = savefig(fig_dir, out_name, formats=["pdf", "png"], dpi=200)
    plt.close(fig)
    return out


def generate_motif_plot(con, fig_dir: Path, run: str) -> List[str]:
    stream = load_event_stream(
        con=con,
        run=run,
        start_us=None,
        end_us=None,
        min_msg_bytes=1_000_000,
        keep_colls=["AllGather", "ReduceScatter", "AllReduce", "Broadcast"],
    )
    tokens = make_tokens(stream)
    motifs = find_repeated_ngrams(tokens, n=30, min_count=3, topk=5)
    if not motifs:
        raise RuntimeError("No motifs found (try lowering n or min_msg_bytes).")
    # Plot the first occurrence of the top motif
    return plot_motif_occurrence(fig_dir, stream, motifs[0], occurrence=0, out_name="motif_15b_16_n30_min1000000")


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=NIXT_ROOT + "/data")
    ap.add_argument("--fig_dir", default=NIXT_ROOT + "/figures")
    ap.add_argument("--run_15b_16", default="pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1756589443-analysis")
    ap.add_argument("--run_15b_2048", default="pretrain_nemotron4_15b_fp8_gpus2048_tp2_pp1_cp1_vpNone_mbs2_gbs8192_1757872279-analysis")
    ap.add_argument("--run_340b_2048", default="pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis")
    ap.add_argument("--motif", action="store_true", help="Also generate the (optional) motif plot.")
    ap.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats (default: pdf,png). Example: pdf,png",
    )
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    con = connect_logs(args.data_root)
    formats = [f.strip() for f in str(args.formats).split(",") if f.strip()]

    meta = load_meta(con, exclude_stragglers=True)
    df_bytes = compute_bytes_by_run_coll(con, meta)
    df_bw = compute_bw_by_run_coll(con, meta, min_msg_bytes=1_000_000)

    outputs: List[str] = []

    # Single-run: 15B@16
    outputs.extend(plot_bytes_by_coll_commtype(con, fig_dir, args.run_15b_16, "15b_16"))
    outputs.extend(plot_top_msg_sizes_by_bytes(con, fig_dir, args.run_15b_16, "15b_16"))
    outputs.extend(plot_busbw_percentiles(con, fig_dir, args.run_15b_16, "15b_16"))

    # Single-run: 340B@2048
    outputs.extend(plot_bytes_by_coll_commtype(con, fig_dir, args.run_340b_2048, "340b_2048"))
    outputs.extend(plot_top_msg_sizes_by_bytes(con, fig_dir, args.run_340b_2048, "340b_2048"))
    outputs.extend(plot_busbw_percentiles(con, fig_dir, args.run_340b_2048, "340b_2048"))

    # Scatter (paper-styled)
    outputs.extend(plot_scatter_nic(con, fig_dir, args.run_340b_2048, "340b_2048", "AllGather"))
    outputs.extend(plot_scatter_nic(con, fig_dir, args.run_340b_2048, "340b_2048", "AllReduce"))

    # Scaling: 15B vs GPU count
    outputs.extend(plot_scaling_15b_bytes_by_coll_vs_gpus(fig_dir, df_bytes))
    outputs.extend(plot_scaling_15b_total_bytes_vs_gpus(fig_dir, df_bytes))
    outputs.extend(plot_scaling_15b_busbw_by_coll_vs_gpus(fig_dir, df_bw))

    # Model-size compare at 2048
    outputs.extend(plot_compare_modelsize_gpus2048_bytes_by_coll(fig_dir, df_bytes, g=2048))
    outputs.extend(plot_compare_modelsize_gpus2048_busbw_by_coll(fig_dir, df_bw, g=2048))

    # 2x2: 15B@2048 vs 340B@2048 top-msg-sizes and bus-BW percentiles
    outputs.extend(plot_compare_15b_340b_topmsg_busbw_2x2(
        con, fig_dir, args.run_15b_2048, args.run_340b_2048,
    ))

    # Motif plot (optional; disabled by default)
    if args.motif:
        outputs.extend(generate_motif_plot(con, fig_dir, args.run_15b_16))

    print("\nWrote figures:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()


