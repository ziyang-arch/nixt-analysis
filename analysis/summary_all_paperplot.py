#!/usr/bin/env python3
"""
Paper-ready summary plots (standalone).

Currently generates (for two fixed runs):
  - Bytes by (coll, comm_type) as % of total bytes (stacked bars)
    - y: % of total bytes
    - x: collective (horizontal labels)
    - legend only (no title), placed below the plot
    - annotate the total % on top of each stacked bar

Outputs:
  Writes PDF + PNG into:
    $NIXT_ROOT/figures/

Target filenames (match existing PNG names from summary_all.ipynb):
  - 15b_16_bytes_by_coll_commtype.(pdf|png)
  - 340b_2048_bytes_by_coll_commtype.(pdf|png)
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import duckdb  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "a required package")
    raise SystemExit(
        f"Missing Python dependency: {missing}\n\n"
        "Run this script inside the same environment used by the notebooks.\n"
        "For this repo, that is typically:\n"
        "  source \"$(conda info --base)/etc/profile.d/conda.sh\"\n"
        "  conda activate nccl_exporter\n"
        "  python exporter/summary_all_paperplot.py\n"
    ) from e


# ------------------------------
# DuckDB view over Parquet
# ------------------------------

DATA_ROOT = NIXT_ROOT + "/data"
PARQUET_GLOB = f"{DATA_ROOT}/*-analysis/parquet_files/*.parquet"

con = duckdb.connect()
con.execute(
    f"""
CREATE OR REPLACE VIEW logs AS
SELECT
  regexp_extract(filename, '/data/([^/]+)/parquet_files/', 1) AS run,
  *
FROM read_parquet('{PARQUET_GLOB}', filename=true);
"""
)


def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


# ------------------------------
# Styling + saving
# ------------------------------

FIG_DIR = Path(NIXT_ROOT + "/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def savefig_paper(out_base: str, *, formats: Sequence[str] = ("pdf", "png"), dpi: int = 200) -> List[str]:
    wrote: List[str] = []
    for fmt in formats:
        fmt = str(fmt).lower().strip().lstrip(".")
        out = FIG_DIR / f"{out_base}.{fmt}"
        if fmt in {"pdf", "svg", "eps"}:
            plt.savefig(out, bbox_inches="tight")
        else:
            plt.savefig(out, dpi=int(dpi), bbox_inches="tight")
        wrote.append(str(out))
    print("wrote", ", ".join(wrote))
    return wrote


COMM_ORDER = ["nic-only", "nvlink-only", "mixed", "single-rank"]
COMM_COLORS: Dict[str, str] = {
    "nic-only": "#8DA0CB",  # soft blue
    "nvlink-only": "#66C2A5",  # soft teal
    "mixed": "#FC8D62",  # soft orange
    "single-rank": "#B3B3B3",  # soft gray
}
COMM_DISPLAY_REMAP: Dict[str, str] = {"hca-only": "nic-only"}


def plot_bytes_by_coll_commtype_pct(*, run: str, tag: str, formats: Sequence[str], dpi: int) -> List[str] | None:
    d = q(
        f"""
SELECT coll, comm_type, SUM(coll_msg_size_bytes) AS bytes
FROM logs
WHERE run = '{run}'
GROUP BY coll, comm_type
"""
    ).copy()

    if d.empty:
        print(f"[skip] no rows for run={run}")
        return None

    d["comm_type"] = d["comm_type"].replace(COMM_DISPLAY_REMAP)
    d["bytes"] = pd.to_numeric(d["bytes"], errors="coerce")
    total = float(d["bytes"].sum())
    if not np.isfinite(total) or total <= 0:
        print(f"[skip] invalid total bytes for run={run}")
        return None

    pivot = d.pivot_table(index="coll", columns="comm_type", values="bytes", aggfunc="sum", fill_value=0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot.reindex(columns=[c for c in COMM_ORDER if c in pivot.columns])

    pivot_pct = pivot.astype(float) / total * 100.0
    bar_totals_pct = pivot_pct.sum(axis=1).astype(float).values

    # Single-column friendly paper size
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = [COMM_COLORS.get(c, "#B3B3B3") for c in pivot_pct.columns]
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="none", width=0.82)

    ax.set_ylabel("% of total bytes")
    ax.set_xlabel("")  # no extra label (x tick labels are the collective names)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

    # Horizontal x tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Write % inside each stacked segment (skip tiny segments to avoid clutter).
    MIN_SEG_LABEL_PCT = 3.0
    vals = pivot_pct.to_numpy(dtype=float)  # shape: (n_coll, n_comm)
    for i in range(vals.shape[0]):
        cum = 0.0
        for j, comm in enumerate(pivot_pct.columns.tolist()):
            v = float(vals[i, j]) if np.isfinite(vals[i, j]) else 0.0
            if v >= MIN_SEG_LABEL_PCT:
                # Choose white/black text depending on segment color brightness.
                rgb = mpl.colors.to_rgb(COMM_COLORS.get(str(comm), "#B3B3B3"))
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                tc = "white" if lum < 0.55 else "black"
                ax.text(i, cum + v / 2.0, f"{v:.1f}%", ha="center", va="center", fontsize=7, color=tc)
            cum += v

    # Legend only (no title), at bottom
    ax.legend(
        title=None,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.9,
        handlelength=1.2,
    )

    # Write the total % on top of each stacked bar
    xt = ax.get_xticks()
    for i, x in enumerate(xt):
        if i >= len(bar_totals_pct):
            break
        y = float(bar_totals_pct[i])
        ax.text(x, y + 0.8, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.0, max(1.0, float(np.nanmax(bar_totals_pct)) + 6.0))

    fig.tight_layout(pad=0.2)
    out = savefig_paper(f"{slug(tag)}_bytes_by_coll_commtype", formats=formats, dpi=dpi)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", default="pdf,png", help="Comma-separated output formats (default: pdf,png).")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    args = ap.parse_args()

    formats = [f.strip().lower().lstrip(".") for f in str(args.formats).split(",") if f.strip()]

    runs = {
        "15b_16": "pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1756589443-analysis",
        "340b_2048": "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis",
    }

    for tag, run in runs.items():
        _ = plot_bytes_by_coll_commtype_pct(run=run, tag=tag, formats=formats, dpi=int(args.dpi))


if __name__ == "__main__":
    main()


