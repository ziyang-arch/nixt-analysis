#!/usr/bin/env python3
"""
Paper-ready plots for Measurement vs Configuration.

Currently generates:

Situation × message-size heatmaps (log-count):
  0) count heatmap over [ranks,nodes,coll] × message size  [log(count), B/W]

CDF plots (one ECDF line per config: [n_ranks, nnodes, coll, coll_msg_size_bytes]):
  3) exec-time ECDF for small messages (<= threshold)
  4) busBW ECDF for large messages (>= threshold)

Outputs: PDF + PNG under:
  $NIXT_ROOT/figures/experiments/Measurement_vs_Configuration/

Runs:
  - Situation×msg-size heatmap: 340B@2048 and 15B@2048 (fixed run ids in this script)
  - CDFs: generated for both 340B@2048 and 15B@2048
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path
import colorsys
import re

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: duckdb. Activate your nccl_exporter environment and retry.\n"
        "Example:\n"
        "  conda activate nccl_exporter"
    ) from e

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


def list_runs() -> list[str]:
    return q("SELECT DISTINCT run FROM logs ORDER BY run")["run"].astype(str).tolist()


def _sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


# ------------------------------
# Plot styling + saving
# ------------------------------

FIG_DIR = Path(NIXT_ROOT + "/figures")
CATEGORY = "Measurement_vs_Configuration"

# Fixed run id (edit here if you want to switch)
RUN_FIXED = "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis"
RUN_FIXED_15B_2048 = "pretrain_nemotron4_15b_fp8_gpus2048_tp2_pp1_cp1_vpNone_mbs2_gbs8192_1757872279-analysis"

# CDF thresholds (edit if needed)
SMALL_MSG_THRESHOLD_BYTES = 64 * 1024        # 64 KiB
LARGE_MSG_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MiB

mpl.rcParams.update(
    {
        # Larger, paper-friendly fonts
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
    }
)


def trunc_cmap(cmap_name: str = "Greys", *, minval: float = 0.0, maxval: float = 0.80, n: int = 256):
    """Truncate a colormap so the top end doesn't reach pure black (print-friendly)."""
    # matplotlib>=3.7: prefer new API to avoid deprecation warnings
    try:
        base = mpl.colormaps.get_cmap(cmap_name).resampled(n)
    except Exception:
        base = mpl.cm.get_cmap(cmap_name, n)  # fallback for older matplotlib
    colors = base(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", colors)


BW_CMAP = trunc_cmap("Greys", maxval=0.80)

#
# ------------------------------
# Topology categories (used by both heatmap and CDF plots)
# ------------------------------
#
# Topology category rules (per request):
#   (1) single-rank: n_ranks = 1
#   (2) nvlink-only: nnodes = 1 (all ranks within one node)
#   (3) nic-only: n_ranks = nnodes (one rank per node; inter-node dominated), for multi-node
#   (4) mixed: multi-node with multiple ranks per node (everything else)
#


def topology_category(n_ranks: int, nnodes: int) -> str:
    try:
        r = int(n_ranks)
        n = int(nnodes)
    except Exception:
        return "mixed"

    if r == 1:
        return "single-rank"
    if n == 1:
        return "nvlink-only"
    if r == n:
        return "nic-only"
    return "mixed"


TOPO_COLORS = {
    # Non-gray, paper-friendly base hues
    "single-rank": "#9467BD",
    "nvlink-only": "#2CA02C",
    "nic-only": "#FF7F0E",
    "mixed": "#1F77B4",
}


def topo_color_for_line(
    topo: str,
    coll: str,
    size_idx: int = 0,
    size_total: int = 1,
) -> tuple[float, float, float]:
    """
    Pick a line color such that configs in the same topology category have similar color.
    We use the topology base hue and apply a *more apparent* lightness/saturation variation
    by collective type (still keeping a consistent hue family per topology). Within a
    (topology, collective) pair, distinct message sizes are spread along a wider
    lightness ramp so the lines remain visually distinguishable.
    """
    base = np.asarray(mcolors.to_rgb(TOPO_COLORS.get(str(topo), TOPO_COLORS["mixed"])), dtype=float)
    coll = str(coll)

    # Deterministic variations within a topology.
    # (keeps "similar color" while avoiding complete overlap when multiple colls exist)
    coll_order = ["AllReduce", "ReduceScatter", "AllGather", "Broadcast"]
    try:
        k = coll_order.index(coll)
    except ValueError:
        k = (abs(hash(coll)) % 4)

    # Convert to HSV and vary V/S more strongly for clearer separation.
    h, s, v = colorsys.rgb_to_hsv(float(base[0]), float(base[1]), float(base[2]))
    # Make it a bit more saturated overall (bounded).
    s = min(1.0, max(0.0, s * 1.15))

    # Per-coll adjustments (bigger spread than before):
    #   - some are brighter, some darker, some more saturated
    dv = [0.18, 0.05, -0.10, -0.24][k]   # value shift
    ds = [0.10, 0.00, 0.05, 0.12][k]     # saturation shift

    # Per-message-size lightness ramp within a (topo, coll) family.
    # Range of +-0.22 around the coll-level value gives clearly distinguishable
    # lines for typical 2-4 sizes per coll, while preserving the family hue.
    if size_total > 1:
        size_span = 0.44
        size_shift = (size_idx / (size_total - 1) - 0.5) * size_span
    else:
        size_shift = 0.0

    s2 = min(1.0, max(0.0, s + ds))
    v2 = min(1.0, max(0.0, v + dv + size_shift))
    r, g, b = colorsys.hsv_to_rgb(h, s2, v2)
    return (float(r), float(g), float(b))


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


# From README: these two 15B@16GPU runs are "straggler node" experiments.
STRAGGLER_RUN_TS = {"1757455306", "1757716170"}
_run_re_gpus = re.compile(r"gpus(?P<gpus>\d+)")


def parse_run_meta(run: str) -> dict:
    r = str(run)
    rl = r.lower()

    if "nemotron4_15b" in rl:
        model = "15b"
    elif "nemotron4_340b" in rl:
        model = "340b"
    else:
        model = "unk"

    m = _run_re_gpus.search(rl)
    gpus = int(m.group("gpus")) if m else None

    is_straggler = any(ts in r for ts in STRAGGLER_RUN_TS)
    health = "straggler" if is_straggler else "healthy"

    mt = re.search(r"_(\d{9,})-analysis$", r)
    ts = mt.group(1)[-10:] if mt else "na"

    return {"model": model, "gpus": gpus, "health": health, "ts": ts}


def run_suffix(run: str) -> str:
    meta = parse_run_meta(run)
    g = meta["gpus"] if meta["gpus"] is not None else "Unknown"
    tag = f"{meta['model']}_gpus{g}_{meta['health']}_{meta['ts']}"
    return slug(tag)


def _category_dir(category: str = CATEGORY) -> Path:
    out = FIG_DIR / "experiments" / slug(category)
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig_paper(name: str, category: str = CATEGORY, formats=("pdf", "png"), dpi=200):
    out_dir = _category_dir(category)
    wrote = []
    for fmt in formats:
        fmt = str(fmt).lower().strip().lstrip(".")
        out = out_dir / f"{name}.{fmt}"
        if fmt in {"pdf", "svg", "eps"}:
            plt.savefig(out, bbox_inches="tight")
        else:
            plt.savefig(out, dpi=dpi, bbox_inches="tight")
        wrote.append(str(out))
    print("wrote", ", ".join(wrote))
    return wrote


# ------------------------------
# Plot 0: (ranks,nodes,coll) × message size heatmap (log-count)
# ------------------------------

def _bytes_si(x: float) -> str:
    """Format bytes using SI units: B, KB, MB, GB."""
    try:
        b = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(b) or b < 0:
        return str(x)
    if b < 1000:
        return f"{int(round(b))}B"
    if b < 1000**2:
        return f"{b/1000:.3g}KB"
    if b < 1000**3:
        return f"{b/1000**2:.3g}MB"
    return f"{b/1000**3:.3g}GB"


def plot_situation_msgsize_count_heatmap(
    *,
    run: str,
    out_name: str,
    category: str = CATEGORY,
    max_xticks: int = 18,
    max_yticks: int = 40,
    figsize: tuple[float, float] = (6.5, 2.6),
):
    """
    Heatmap of op frequency:
      y = [ranks,nodes,coll], x = message size, value = count (log scale).

    Styling matches the notebook: white→gray (no pure black), no titles/captions.
    """
    rq = _sql_quote(run)
    d = q(
        f"""
SELECT
  n_ranks::INT AS n_ranks,
  nnodes::INT AS nnodes,
  coll::VARCHAR AS coll,
  coll_msg_size_bytes::BIGINT AS coll_msg_size_bytes,
  COUNT(*)::BIGINT AS ops
FROM logs
WHERE run = '{rq}'
  AND n_ranks IS NOT NULL
  AND nnodes IS NOT NULL
  AND coll IS NOT NULL
  AND coll_msg_size_bytes IS NOT NULL
GROUP BY n_ranks, nnodes, coll, coll_msg_size_bytes
ORDER BY n_ranks, nnodes, coll, coll_msg_size_bytes
"""
    ).copy()
    if d.empty:
        print(f"[skip] situation×msgsize heatmap: no rows for run={run}")
        return None

    d["y"] = d.apply(lambda r: f"{int(r['n_ranks'])},{int(r['nnodes'])},{str(r['coll'])}", axis=1)
    d["x"] = d["coll_msg_size_bytes"].astype(int).astype(str)

    # Reverse y-axis ordering: larger (n_ranks, nnodes) should appear towards the top.
    ylabels = (
        d[["n_ranks", "nnodes", "coll", "y"]]
        .drop_duplicates()
        .sort_values(["n_ranks", "nnodes", "coll"], ascending=[False, False, True])
        ["y"]
        .astype(str)
        .tolist()
    )
    xlabels = sorted(d["coll_msg_size_bytes"].astype(int).unique().tolist())
    xlabels_str = [str(int(x)) for x in xlabels]

    piv = (
        d.pivot_table(index="y", columns="x", values="ops", aggfunc="sum")
        .reindex(index=ylabels, columns=xlabels_str)
        .fillna(0.0)
    )

    vals = piv.values.astype(float)
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    data = np.ma.masked_where(vals <= 0, vals)

    cmap_use = BW_CMAP
    try:
        cmap_use = BW_CMAP.copy()
    except Exception:
        pass
    try:
        cmap_use.set_bad("white")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=figsize)
    norm = mcolors.LogNorm(vmin=1.0, vmax=max(1.0, vmax))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap_use, norm=norm)

    # Sparse ticks for compact, near-square figures
    n_x = len(xlabels_str)
    n_y = len(ylabels)
    x_step = max(1, int(np.ceil(n_x / max(1, int(max_xticks)))))
    y_step = max(1, int(np.ceil(n_y / max(1, int(max_yticks)))))
    x_idx = np.arange(0, n_x, x_step, dtype=int)
    y_idx = np.arange(0, n_y, y_step, dtype=int)

    ax.set_xticks(x_idx)
    ax.set_yticks(y_idx)
    ax.set_xticklabels(
        [_bytes_si(int(xlabels_str[j])) for j in x_idx.tolist()],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_yticklabels([ylabels[i] for i in y_idx.tolist()], fontsize=9)
    # Color y tick labels by topology category (same palette used for CDF line colors).
    for tick in ax.get_yticklabels():
        txt = tick.get_text()
        try:
            r_s, n_s, _ = txt.split(",", 2)
            topo = topology_category(int(r_s), int(n_s))
        except Exception:
            topo = "mixed"
        tick.set_color(TOPO_COLORS.get(topo, TOPO_COLORS["mixed"]))

    ax.set_xlabel("message size")
    ax.set_ylabel("[ranks, nodes, coll]")

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("count (log scale)", fontsize=11)

    # powers-of-10 ticks when applicable
    try:
        import matplotlib.ticker as mticker

        ticks = [1]
        while ticks[-1] * 10 <= vmax:
            ticks.append(ticks[-1] * 10)
        if len(ticks) >= 2:
            cbar.set_ticks(ticks)
        # Use exponent formatting (e.g., 10^5) instead of raw integers like 100000.
        cbar.formatter = mticker.LogFormatterMathtext(base=10)
        cbar.update_ticks()
    except Exception:
        pass

    fig.tight_layout(pad=0.2)
    # Save both vector (PDF) + raster (PNG) for convenience.
    savefig_paper(out_name, category=category, formats=("pdf", "png"))
    plt.close(fig)
    return piv


# ------------------------------
# Plot 3+4: CDFs by configuration
# ------------------------------

CONFIG_COLS = ["n_ranks", "nnodes", "coll", "coll_msg_size_bytes"]


def ecdf(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / float(x.size)
    return x, y


def _cfg_label(cfg: tuple) -> str:
    n_ranks, nnodes, coll, msgb = cfg
    # Shorten the label a bit to reduce legend width/overlap.
    topo = topology_category(int(n_ranks), int(nnodes))
    return f"{topo} r{int(n_ranks)},n{int(nnodes)},{str(coll)},{_bytes_si(int(msgb))}"


def load_metrics_for_run(run: str) -> pd.DataFrame:
    rq = _sql_quote(run)
    d = q(
        f"""
SELECT
  coll_exec_time_us,
  coll_busbw_gbs,
  n_ranks,
  nnodes,
  coll,
  coll_msg_size_bytes
FROM logs
WHERE run = '{rq}'
"""
    ).copy()
    if d.empty:
        return d
    # Keep typing stable for grouping/labeling
    for c in ["n_ranks", "nnodes", "coll_msg_size_bytes"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def plot_cdf_by_config(
    *,
    metrics_df: pd.DataFrame,
    out_name: str,
    metric_col: str,
    msg_filter: str,  # 'le' or 'ge'
    msg_threshold_bytes: int,
    xlabel: str,
    category: str = CATEGORY,
    max_legend_items: int = 40,
    normalize: bool = True,
    normalize_by: str = "max",  # 'max' | 'median' | 'p50' | 'mean'
    cdf_label_fontsize: int = 13,
    cdf_tick_fontsize: int = 12,
    cdf_legend_fontsize: int = 10,
    cdf_legend_title_fontsize: int = 10,
):
    if msg_filter not in {"le", "ge"}:
        raise ValueError("msg_filter must be 'le' or 'ge'")

    d = metrics_df.copy()
    d = d.dropna(subset=CONFIG_COLS + [metric_col])

    if msg_filter == "le":
        d = d[d["coll_msg_size_bytes"] <= msg_threshold_bytes]
    else:
        d = d[d["coll_msg_size_bytes"] >= msg_threshold_bytes]

    if d.empty:
        print(
            f"[skip] empty after filter for {metric_col}, msg_filter={msg_filter}, thr={msg_threshold_bytes}"
        )
        return None

    groups = list(d.groupby(CONFIG_COLS, sort=True))
    n_groups = len(groups)
    print(f"plotting {n_groups} ECDF lines for metric={metric_col} ({out_name})")

    # Use a dedicated legend panel below the plot so legend never overlaps axes text.
    # The figure height scales with the number of legend rows.
    n_show = min(n_groups, int(max_legend_items))
    if n_show <= 16:
        ncol = 2
    elif n_show <= 30:
        ncol = 3
    elif n_show <= 50:
        ncol = 4
    else:
        ncol = 5

    import math

    n_rows = int(math.ceil(n_show / float(ncol))) if n_show > 0 else 0
    base_h_in = 2
    # Empirically: ~0.28in per row + a bit for the legend title.
    leg_h_in = (0.30 * n_rows) + 0.55 if n_rows > 0 else 0.0
    fig_h_in = base_h_in + leg_h_in

    fig = plt.figure(figsize=(8, fig_h_in))
    if leg_h_in > 0:
        # Increase vertical spacing so the x-axis label never feels cramped against the legend title.
        gs = fig.add_gridspec(2, 1, height_ratios=[base_h_in, leg_h_in], hspace=0.16)
        ax = fig.add_subplot(gs[0])
        legax = fig.add_subplot(gs[1])
        legax.axis("off")
    else:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        legax = None

    # Pre-compute, for each (topology, collective) family, the sorted list of
    # message sizes that appear in this figure. This lets us spread lines for
    # different sizes within the same coll along a lightness ramp.
    size_index_by_cfg: dict[tuple, tuple[int, int]] = {}
    sizes_by_family: dict[tuple[str, str], list[int]] = {}
    for cfg, _g in groups:
        n_ranks, nnodes, coll, msgb = cfg
        topo = topology_category(int(n_ranks), int(nnodes))
        sizes_by_family.setdefault((topo, str(coll)), []).append(int(msgb))
    sizes_by_family = {k: sorted(set(v)) for k, v in sizes_by_family.items()}
    for cfg, _g in groups:
        n_ranks, nnodes, coll, msgb = cfg
        topo = topology_category(int(n_ranks), int(nnodes))
        ordered = sizes_by_family[(topo, str(coll))]
        size_index_by_cfg[cfg] = (ordered.index(int(msgb)), len(ordered))

    handles = []
    labels = []
    for i, (cfg, g) in enumerate(groups):
        v = g[metric_col].to_numpy()
        if normalize:
            vb = v[np.isfinite(v)]
            if vb.size == 0:
                continue
            nb = str(normalize_by).lower().strip()
            if nb == "max":
                denom = float(np.nanmax(vb))
            elif nb in {"median", "p50"}:
                denom = float(np.nanmedian(vb))
            elif nb in {"mean", "avg"}:
                denom = float(np.nanmean(vb))
            else:
                raise ValueError("normalize_by must be one of: 'max', 'median', 'p50', 'mean'")
            # Avoid divide-by-zero / pathological cases.
            if not np.isfinite(denom) or denom <= 0:
                continue
            v = v / denom

        x, y = ecdf(v)
        if x.size == 0:
            continue
        # Color by topology category (configs in same topology share similar color).
        n_ranks, nnodes, coll, _msgb = cfg
        topo = topology_category(int(n_ranks), int(nnodes))
        size_idx, size_total = size_index_by_cfg[cfg]
        color = topo_color_for_line(topo, str(coll), size_idx, size_total)
        (ln,) = ax.plot(x, y, lw=1.6, color=color, alpha=0.95)
        # For huge config counts, avoid an unreadable legend.
        if len(handles) < max_legend_items:
            handles.append(ln)
            labels.append(_cfg_label(cfg))

    ax.set_xlabel(xlabel, fontsize=cdf_label_fontsize)
    ax.set_ylabel("CDF", fontsize=cdf_label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=cdf_tick_fontsize)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)

    if handles and legax is not None:
        suffix = "" if n_groups <= max_legend_items else f" (showing {len(handles)}/{n_groups})"
        legax.legend(
            handles,
            labels,
            # Anchor legend lower in its panel to leave extra whitespace above it.
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=ncol,
            fontsize=cdf_legend_fontsize,
            frameon=False,
            title=("config [ranks,nodes,coll,msg]" + suffix),
            title_fontsize=cdf_legend_title_fontsize,
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=0.9,
            labelspacing=0.35,
            borderaxespad=0.0,
        )

    # Tighten only the main axes area; legend is in its own panel.
    try:
        ax.margins(x=0.02)
    except Exception:
        pass

    savefig_paper(out_name, category=category)
    plt.close(fig)
    return ax


def main():
    # Added: two situation×msg-size heatmaps (340B@2048 and 15B@2048), PDF-only
    plot_situation_msgsize_count_heatmap(
        run=RUN_FIXED,
        out_name=f"exp_situation_msgsize_count_heatmap_{run_suffix(RUN_FIXED)}",
    )
    plot_situation_msgsize_count_heatmap(
        run=RUN_FIXED_15B_2048,
        out_name=f"exp_situation_msgsize_count_heatmap_{run_suffix(RUN_FIXED_15B_2048)}",
    )

    # CDF plots: generate for both 340B@2048 and 15B@2048.
    for run in (RUN_FIXED, RUN_FIXED_15B_2048):
        metrics = load_metrics_for_run(run)
        if metrics.empty:
            print(f"[skip] no metrics rows for run={run}")
            continue

        _ = plot_cdf_by_config(
            metrics_df=metrics,
            out_name=f"exp_cdf_exec_small_by_config_{run_suffix(run)}",
            metric_col="coll_exec_time_us",
            msg_filter="le",
            msg_threshold_bytes=SMALL_MSG_THRESHOLD_BYTES,
            xlabel=f"Normalized exec time, msg ≤ {SMALL_MSG_THRESHOLD_BYTES / 1024:.0f} KiB",
        )
        _ = plot_cdf_by_config(
            metrics_df=metrics,
            out_name=f"exp_cdf_busbw_large_by_config_{run_suffix(run)}",
            metric_col="coll_busbw_gbs",
            msg_filter="ge",
            msg_threshold_bytes=LARGE_MSG_THRESHOLD_BYTES,
            xlabel=f"Norm BW, msg ≥ {LARGE_MSG_THRESHOLD_BYTES / (1024**2):.1f} MiB",
        )


if __name__ == "__main__":
    main()


