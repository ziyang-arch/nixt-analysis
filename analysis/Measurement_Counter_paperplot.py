#!/usr/bin/env python3
"""
Paper-ready plots for Measurement vs Counter.

This script generates the specific paper plot requested:
  - Large-message busBW scatter over *time* (50–150s)
  - Distinguish different (n_ranks, nnodes) using marker shapes
  - Hue encodes `coll`
  - Shade encodes `coll_msg_size_bytes` (small=lighter, large=deeper/darker)
  - Y is normalized by the max busBW value in the plotted subset
  - Legends:
      - coll legend: bottom (no title)
      - topology legend (ranks,nodes markers): bottom, same line as coll legend
      - message-size legend: right side, lists every distinct size and its shade color

Outputs: PDF + PNG under:
  $NIXT_ROOT/figures/experiments/Measurement_vs_Counter/
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import duckdb  # type: ignore
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import matplotlib as mpl  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.colors as mcolors  # type: ignore
except ModuleNotFoundError as e:
    missing = getattr(e, "name", "a required package")
    raise SystemExit(
        f"Missing Python dependency: {missing}\n\n"
        "Run this script inside the same environment used by the notebooks.\n"
        "For this repo, that is typically:\n"
        "  source \"$(conda info --base)/etc/profile.d/conda.sh\"\n"
        "  conda activate nccl_exporter\n"
        "  python exporter/Measurement_Counter_paperplot.py\n"
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


def _sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


# ------------------------------
# Styling + saving
# ------------------------------

FIG_DIR = Path(NIXT_ROOT + "/figures")
CATEGORY = "Measurement_vs_Counter"

# Keep consistent with other paperplot scripts
mpl.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
    }
)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


_run_re_gpus = re.compile(r"gpus(?P<gpus>\d+)")


def run_suffix(run: str) -> str:
    r = str(run)
    rl = r.lower()
    if "nemotron4_15b" in rl:
        model = "15b"
    elif "nemotron4_340b" in rl:
        model = "340b"
    else:
        model = "unk"

    m = _run_re_gpus.search(rl)
    gpus = m.group("gpus") if m else "Unknown"

    mt = re.search(r"_(\d{9,})-analysis$", r)
    ts = mt.group(1)[-10:] if mt else "na"

    return slug(f"{model}_gpus{gpus}_{ts}")


def _category_dir(category: str = CATEGORY) -> Path:
    out = FIG_DIR / "experiments" / slug(category)
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig_paper(name: str, category: str = CATEGORY, formats: Sequence[str] = ("pdf", "png"), dpi: int = 200) -> List[str]:
    out_dir = _category_dir(category)
    wrote: List[str] = []
    for fmt in formats:
        fmt = str(fmt).lower().strip().lstrip(".")
        out = out_dir / f"{name}.{fmt}"
        if fmt in {"pdf", "svg", "eps"}:
            plt.savefig(out, bbox_inches="tight")
        else:
            plt.savefig(out, dpi=int(dpi), bbox_inches="tight")
        wrote.append(str(out))
    print("wrote", ", ".join(wrote))
    return wrote


def _bytes_si(x: float) -> str:
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    if x < 1024:
        return f"{int(x)}B"
    for u, s in [("KB", 1024.0), ("MB", 1024.0**2), ("GB", 1024.0**3)]:
        if x < 1024.0 * s:
            return f"{x / s:.0f}{u}"
    return f"{x / (1024.0**4):.1f}TB"


def _shade_by_msg(base_color: str, msg_bytes: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
    """Blend base hue with white based on message size (log scale)."""
    base = np.array(mcolors.to_rgb(base_color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)

    v = np.asarray(msg_bytes, dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, np.nan)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        t = np.full(v.shape, 1.0)
    else:
        z = (np.log2(v) - math.log2(lo)) / max(1e-12, (math.log2(hi) - math.log2(lo)))
        z = np.clip(z, 0.0, 1.0)

        # Deeper colors for paper visibility.
        shade_floor = 0.30
        shade_gamma = 0.75
        t_raw = np.power(z, shade_gamma)
        t = shade_floor + (1.0 - shade_floor) * t_raw

    # Blend with white: small msg -> more white; large msg -> closer to base hue.
    cols = (1.0 - t)[:, None] * white[None, :] + t[:, None] * base[None, :]
    return cols


def _coll_palette(colls: Sequence[str]) -> Dict[str, str]:
    """Deterministic non-gray palette for any set of collectives."""
    explicit = {
        "AllReduce": "#1f77b4",
        "ReduceScatter": "#ff7f0e",
        "AllGather": "#2ca02c",
        "Broadcast": "#d62728",
        "Reduce": "#9467bd",
        "Gather": "#8c564b",
        "Scatter": "#e377c2",
        "AllToAll": "#bcbd22",
        "AllToAllv": "#17becf",
    }
    cmap = plt.get_cmap("tab20")
    pal: Dict[str, str] = {}
    for i, c in enumerate(sorted(set(map(str, colls)))):
        pal[c] = explicit.get(c, mcolors.to_hex(cmap(i % cmap.N)))
    return pal


# ------------------------------
# Data + plot
# ------------------------------


def load_large_msg_busbw(*, run: str, msg_threshold_bytes: int) -> pd.DataFrame:
    rq = _sql_quote(run)
    d = q(
        f"""
SELECT
  dump_timestamp_us,
  coll_sn,
  coll_busbw_gbs,
  n_ranks,
  nnodes,
  coll,
  coll_msg_size_bytes
FROM logs
WHERE run = '{rq}'
  AND dump_timestamp_us IS NOT NULL
  AND coll_sn IS NOT NULL
  AND coll_busbw_gbs IS NOT NULL
  AND n_ranks IS NOT NULL
  AND nnodes IS NOT NULL
  AND coll IS NOT NULL
  AND coll_msg_size_bytes IS NOT NULL
  AND coll_msg_size_bytes >= {int(msg_threshold_bytes)}
"""
    ).copy()

    if d.empty:
        return d

    for c in ["dump_timestamp_us", "coll_sn", "coll_busbw_gbs", "n_ranks", "nnodes", "coll_msg_size_bytes"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["coll"] = d["coll"].astype(str)
    d = d.dropna(subset=["dump_timestamp_us", "coll_sn", "coll_busbw_gbs", "n_ranks", "nnodes", "coll_msg_size_bytes", "coll"])

    t0 = float(d["dump_timestamp_us"].min())
    d["t_s"] = (d["dump_timestamp_us"].astype(float) - t0) / 1e6
    return d.reset_index(drop=True)


def plot_large_msg_busbw_time_topo(
    *,
    d: pd.DataFrame,
    run: str,
    t_min_s: float,
    t_max_s: float,
    topk_cfg: int,
    max_points_per_cfg: int,
    msg_threshold_bytes: int,
    formats: Sequence[str],
    dpi: int,
) -> List[str] | None:
    if d.empty:
        print("[skip] empty dataframe")
        return None

    # Select top-K configs by ops to keep the plot readable and sampling bounded.
    cfg_cols = ["n_ranks", "nnodes", "coll"]
    counts = d.groupby(cfg_cols).size().sort_values(ascending=False)
    cfgs = list(counts.head(int(topk_cfg)).index)
    if not cfgs:
        print("[skip] no configs")
        return None

    parts = []
    for (nr, nn, coll) in cfgs:
        g = d[(d["n_ranks"] == nr) & (d["nnodes"] == nn) & (d["coll"] == coll)].copy()
        if g.empty:
            continue
        if len(g) > int(max_points_per_cfg):
            g = g.sample(n=int(max_points_per_cfg), random_state=0)
        parts.append(g)

    if not parts:
        print("[skip] empty after sampling")
        return None

    dd = pd.concat(parts, ignore_index=True)

    # Time window
    dd = dd[(dd["t_s"] >= float(t_min_s)) & (dd["t_s"] <= float(t_max_s))].copy()
    if dd.empty:
        print("[skip] empty after time window filter")
        return None

    # Normalize busBW by max within subset.
    y_raw = dd["coll_busbw_gbs"].to_numpy(dtype=float)
    yb = y_raw[np.isfinite(y_raw)]
    y_max = float(np.nanmax(yb)) if yb.size else float("nan")
    if not np.isfinite(y_max) or y_max <= 0:
        print("[skip] invalid max busBW for normalization")
        return None
    dd["_busbw_norm"] = y_raw / y_max

    # Message size range for shading
    msg_all = dd["coll_msg_size_bytes"].to_numpy(dtype=float)
    msg_all = msg_all[np.isfinite(msg_all) & (msg_all > 0)]
    msg_lo = float(np.nanmin(msg_all)) if msg_all.size else float(msg_threshold_bytes)
    msg_hi = float(np.nanmax(msg_all)) if msg_all.size else float(msg_threshold_bytes)

    pal = _coll_palette(dd["coll"].astype(str).tolist())

    # Compute shaded RGBs per coll group (fast).
    colors = np.empty((len(dd), 3), dtype=float)
    for coll, idx in dd.groupby("coll").groups.items():
        base = pal.get(str(coll), "#1f77b4")
        m = dd.loc[idx, "coll_msg_size_bytes"].to_numpy(dtype=float)
        colors[np.asarray(list(idx), dtype=int), :] = _shade_by_msg(base, m, lo=msg_lo, hi=msg_hi)

    x = dd["t_s"].to_numpy(dtype=float)
    y = dd["_busbw_norm"].to_numpy(dtype=float)

    # Marker shapes by topology (n_ranks, nnodes)
    topo = dd[["n_ranks", "nnodes"]].astype(int)
    topo_pairs = sorted({(int(r), int(n)) for r, n in topo.to_numpy().tolist()})
    markers = ["o", "^", "s", "D", "v", "P", "X", ">", "<", "h", "*"]
    topo_to_marker = {p: markers[i % len(markers)] for i, p in enumerate(topo_pairs)}

    fig, ax = plt.subplots(figsize=(8.2, 2.8))

    for (r, n) in topo_pairs:
        idx = (topo["n_ranks"] == r) & (topo["nnodes"] == n)
        if not bool(idx.any()):
            continue
        mask = idx.to_numpy()
        ax.scatter(
            x[mask],
            y[mask],
            s=3.0,
            c=colors[mask],
            alpha=0.35,
            linewidths=0,
            marker=topo_to_marker[(r, n)],
        )

    ax.grid(True, alpha=0.20, linewidth=0.6)
    ax.set_xlabel("time (s)")
    ax.set_xlim(float(t_min_s), float(t_max_s))
    ax.set_ylabel("Norm BW")
    ax.set_ylim(0.0, 1.05)

    # Coll legend (bottom, no title)
    coll_list = sorted(set(dd["coll"].astype(str).tolist()))
    coll_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markersize=5, color=pal[c], label=c)
        for c in coll_list
    ]

    # Topology legend (bottom, same line)
    topo_handles = [
        mpl.lines.Line2D([], [], marker=topo_to_marker[(r, n)], linestyle="", markersize=6, markerfacecolor="#444444", markeredgecolor="none")
        for (r, n) in topo_pairs
    ]
    topo_labels = [f"r{r},n{n}" for (r, n) in topo_pairs]

    # Layout: bottom legends + right-side msg legend
    coll_anchor_y = -0.36
    fig.subplots_adjust(left=0.09, right=0.78, top=0.92, bottom=0.50)

    coll_ncol = min(len(coll_handles), 6) if coll_handles else 1
    coll_leg = ax.legend(
        handles=coll_handles,
        loc="upper center",
        bbox_to_anchor=(0.30, coll_anchor_y),
        frameon=False,
        ncol=coll_ncol,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.35,
        borderaxespad=0.0,
    )

    topo_ncol = 4 if len(topo_handles) >= 8 else 3
    _ = ax.legend(
        topo_handles,
        topo_labels,
        loc="upper center",
        bbox_to_anchor=(0.76, coll_anchor_y),
        frameon=False,
        ncol=topo_ncol,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    ax.add_artist(coll_leg)

    # Message-size discrete legend (right side): list every distinct size and its shade.
    msg_sizes = sorted({int(v) for v in dd["coll_msg_size_bytes"].dropna().astype(int).tolist()})
    base_for_msg = pal.get("AllReduce", pal.get(coll_list[0], "#1f77b4")) if coll_list else "#1f77b4"

    msg_handles = []
    msg_labels = []
    for sz in msg_sizes:
        rgb = _shade_by_msg(str(base_for_msg), np.array([float(sz)]), lo=msg_lo, hi=msg_hi)[0]
        msg_handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker="s",
                linestyle="",
                markersize=8,
                markerfacecolor=rgb,
                markeredgecolor="none",
            )
        )
        msg_labels.append(_bytes_si(sz))

    msg_ncol = 1 if len(msg_sizes) <= 18 else (2 if len(msg_sizes) <= 36 else 3)
    msg_fs = 12 if len(msg_sizes) <= 24 else 10
    fig.legend(
        msg_handles,
        msg_labels,
        loc="upper left",
        bbox_to_anchor=(0.80, 0.92),
        frameon=False,
        title="message size",
        ncol=msg_ncol,
        fontsize=msg_fs,
        title_fontsize=msg_fs,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.30,
        borderaxespad=0.0,
    )

    out = savefig_paper(
        f"counter_large_msg_busbw_dots_all_topo_time_{int(t_min_s)}-{int(t_max_s)}s_{run_suffix(run)}",
        category=CATEGORY,
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        default="pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis",
        help="Run id (full exporter run string).",
    )
    ap.add_argument("--msg_threshold_bytes", type=int, default=1 * 1024 * 1024, help="Large-message threshold (bytes).")
    ap.add_argument("--topk_cfg", type=int, default=12, help="Top-K (n_ranks,nnodes,coll) configs by ops.")
    ap.add_argument("--max_points_per_cfg", type=int, default=12000, help="Sample cap per config.")
    ap.add_argument("--tmin_s", type=float, default=50.0, help="Time window start (seconds since run start).")
    ap.add_argument("--tmax_s", type=float, default=150.0, help="Time window end (seconds since run start).")
    ap.add_argument("--formats", default="pdf,png", help="Comma-separated formats, e.g. pdf,png")
    ap.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    args = ap.parse_args()

    fmts = [f.strip().lower().lstrip(".") for f in str(args.formats).split(",") if f.strip()]

    d = load_large_msg_busbw(run=str(args.run), msg_threshold_bytes=int(args.msg_threshold_bytes))
    print("rows=", len(d))
    _ = plot_large_msg_busbw_time_topo(
        d=d,
        run=str(args.run),
        t_min_s=float(args.tmin_s),
        t_max_s=float(args.tmax_s),
        topk_cfg=int(args.topk_cfg),
        max_points_per_cfg=int(args.max_points_per_cfg),
        msg_threshold_bytes=int(args.msg_threshold_bytes),
        formats=fmts,
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()