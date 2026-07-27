"""
Paper-ready figures: Measurement vs Identifier (paper_structure.md §1)

This script generates paper-ready plots that *localize bottlenecks* by tying performance
measurements to physical entities (hostnames) and by visualizing per-rank activity over time.

Figures produced (saved under `figures/experiments/Measurement_vs_Identifier/`):

1) **GPU Straggler Detection (hostname)**:
   - For a chosen collective (default: AllGather) and large messages, compute per-host
     median exec time (p50 `coll_exec_time_us`) and normalize it by the *run-wide* median
     across hosts (100%).
   - Hosts above 100% are slower-than-typical (potential stragglers); a long tail suggests
     node-level imbalance/jitter.

2) **Topology Performance Mapping (hostname)**:
   - For the same collective and message filter, compute per-host median bus bandwidth
     (p50 `coll_busbw_gbs`) and normalize it by the *run-wide* max across hosts (100%).
   - Hosts with low % of max are likely bottlenecks (e.g., noisy neighbor, NIC/NVLink/HCA issues).

3) **Exact per-message timeline (rank vs time), hue=coll, shade=message size**:
   - Each collective op is drawn as a short horizontal segment spanning its start/end time.
   - **Hue** encodes `coll` (no grey "Other" bucket); **shade** encodes `coll_msg_size_bytes`
     (small=light, large=deep), with robust normalization within the selected window.
   - Y-axis is ranks grouped by host index (8 neighboring ranks per host).

4) **Exact per-message timeline (rank vs time), hue=coll, shade=bandwidth**:
   - Same as (3), but shade encodes `coll_busbw_gbs` (low=light, high=deep), normalized
     robustly within the selected window.
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from pathlib import Path
import re
import os

import pandas as pd
import numpy as np

try:
    import duckdb  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "duckdb is required. Activate your environment (e.g., `conda activate nccl_exporter`) "
        "and re-run."
    ) from e

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

# Larger fonts (paper readability)
import matplotlib as mpl

# -----------------------
# Fixed run (paper plot)
# -----------------------

RUN_FIXED = "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis"

# -----------------------
# Data access
# -----------------------

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


def _sql_quote(s: str) -> str:
    return str(s).replace("'", "''")


def q(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


# -----------------------
# Styling + saving
# -----------------------

FIG_DIR = Path(NIXT_ROOT + "/figures")
CATEGORY = "Measurement_vs_Identifier"

# Hard cap for paperplot outputs (to avoid huge vector PDFs on dense timeline plots)
MAX_OUT_BYTES = 5 * 1024 * 1024  # 5 MiB

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

SOFT = {
    # soft Set2-ish colors (less contrast)
    "blue": "#8DA0CB",
    "orange": "#FC8D62",
    "teal": "#66C2A5",
    "gray": "#666666",
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _category_dir(category: str = CATEGORY) -> Path:
    out = FIG_DIR / "experiments" / _slug(category)
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    health = "straggler" if any(ts in r for ts in STRAGGLER_RUN_TS) else "healthy"
    mt = re.search(r"_(\d{9,})-analysis$", r)
    ts = mt.group(1)[-10:] if mt else "na"
    return {"model": model, "gpus": gpus, "health": health, "ts": ts}


def run_suffix(run: str) -> str:
    meta = parse_run_meta(run)
    g = meta["gpus"] if meta["gpus"] is not None else "Unknown"
    return _slug(f"{meta['model']}_gpus{g}_{meta['health']}_{meta['ts']}")


def savefig_paper(
    name: str,
    *,
    category: str = CATEGORY,
    formats: tuple[str, ...] = ("pdf",),
    dpi: int = 200,
    max_bytes: int = MAX_OUT_BYTES,
) -> list[str]:
    out_dir = _category_dir(category)
    wrote: list[str] = []

    def _sizeof(p: Path) -> int:
        try:
            return int(p.stat().st_size)
        except Exception:
            return 0

    def _save_one(out: Path, fmt: str) -> bool:
        """
        Save one figure with a file-size cap.

        Notes:
        - Vector formats can explode in size for dense timeline plots. We rely on rasterizing
          the heavy artists (LineCollection) and then use dpi retries to control embed size.
        - If we still exceed the cap, delete the oversized file and report a warning.
        """
        fmt = fmt.lower()
        is_vector = fmt in {"pdf", "svg", "eps"}

        # Aggressive compression for PDF (helps when we embed rasterized artists).
        if fmt == "pdf":
            try:
                plt.rcParams["pdf.compression"] = 9
            except Exception:
                pass

        if is_vector:
            dpi_try = [int(dpi), 150, 120, 100, 80, 60]
            dpi_try = [d for d in dpi_try if d > 0]
            for d in dpi_try:
                plt.savefig(out, bbox_inches="tight", dpi=int(d))
                if _sizeof(out) <= int(max_bytes):
                    return True
        else:
            dpi_try = [int(dpi), 180, 150, 120, 100, 80]
            dpi_try = [d for d in dpi_try if d > 0]
            for d in dpi_try:
                save_kwargs = {"dpi": int(d), "bbox_inches": "tight"}
                # Pillow kwargs are supported in newer matplotlib; fall back if unavailable.
                try:
                    plt.savefig(out, **save_kwargs, pil_kwargs={"optimize": True, "compress_level": 9})
                except TypeError:
                    plt.savefig(out, **save_kwargs)
                if _sizeof(out) <= int(max_bytes):
                    return True

        sz = _sizeof(out)
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
        print(
            f"[warn] skipped oversized output: {out} ({sz/1024/1024:.1f} MiB) "
            f"> {int(max_bytes)/1024/1024:.1f} MiB"
        )
        return False

    for fmt in formats:
        fmt = str(fmt).lower().strip().lstrip(".")
        out = out_dir / f"{name}.{fmt}"
        if _save_one(out, fmt):
            wrote.append(str(out))
    if wrote:
        print("wrote", ", ".join(wrote))
    return wrote


def pct(x: pd.Series, denom: float) -> pd.Series:
    d = float(denom) if float(denom) != 0.0 else 1.0
    return (x.astype(float) / d) * 100.0


# -----------------------
# Exact-timeline helpers (rank grouped by host index)
# -----------------------


def _is_grey(rgb: tuple[float, float, float], eps: float = 0.08) -> bool:
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return (max(r, g, b) - min(r, g, b)) < eps


def _run_tmin_us(run: str) -> int | None:
    rq = _sql_quote(run)
    df = q(f"SELECT min(dump_timestamp_us)::BIGINT AS t0 FROM logs WHERE run = '{rq}' AND dump_timestamp_us IS NOT NULL")
    if df.empty:
        return None
    t0 = df["t0"].iloc[0]
    if t0 is None:
        return None
    try:
        return int(t0)
    except Exception:
        return None


def _build_rank_layout_by_host_index(run: str, ranks_per_host: int = 8) -> tuple[dict[int, int], list[int]]:
    """Return (rank_to_row, host_bounds) where y=row and 8 neighboring ranks belong to one host index.

    - Hostnames are mapped to host indices by lexicographic order (stable across runs).
    - For each host, ranks are sorted ascending and assigned contiguous rows.
    """
    rq = _sql_quote(run)
    df = q(
        f"""
SELECT DISTINCT
  hostname::VARCHAR AS hostname,
  "rank"::INT AS rank
FROM logs
WHERE run = '{rq}'
  AND hostname IS NOT NULL
  AND "rank" IS NOT NULL
ORDER BY hostname ASC, rank ASC
"""
    ).copy()
    if df.empty:
        return {}, []

    host_names = sorted(df["hostname"].astype(str).unique().tolist())
    host_to_idx = {h: i for i, h in enumerate(host_names)}

    rank_to_row: dict[int, int] = {}
    for h in host_names:
        hi = host_to_idx[h]
        ranks = sorted(df.loc[df["hostname"].astype(str) == h, "rank"].astype(int).unique().tolist())
        base = hi * int(ranks_per_host)
        for j, r in enumerate(ranks):
            rank_to_row[int(r)] = base + j

    host_bounds = [i * int(ranks_per_host) for i in range(len(host_names) + 1)]
    return rank_to_row, host_bounds


def _coll_palette_no_grey(colls_present: list[str]) -> tuple[list[str], dict[str, str]]:
    """Deterministic palette for any set of collectives, avoiding grey-ish colors."""
    COLL_HUE = {
        "AllReduce": "#FC8D62",  # soft orange
        "AllGather": "#8DA0CB",  # soft blue
        "ReduceScatter": "#66C2A5",  # soft teal
        "Broadcast": "#A6D854",  # soft green
    }

    ordered = [c for c in COLL_HUE.keys() if c in colls_present] + sorted([c for c in colls_present if c not in COLL_HUE])

    extra: list[str] = []
    for cmap in ("tab20", "tab20b", "tab20c"):
        extra.extend([mcolors.to_hex(c) for c in plt.get_cmap(cmap).colors])
    extra = [c for c in extra if not _is_grey(mcolors.to_rgb(c))]

    coll_to_color: dict[str, str] = {c: COLL_HUE[c] for c in ordered if c in COLL_HUE}
    j = 0
    for c in ordered:
        if c in coll_to_color:
            continue
        coll_to_color[c] = extra[j % len(extra)] if extra else "#9467BD"
        j += 1

    return ordered, coll_to_color


def plot_exact_timeline_color_by_coll_shade_by_msg_size(
    *,
    run: str,
    window_start_s: float,
    window_dur_s: float,
    max_msg: int | None = None,
    keep_colls: list[str] | None = None,
    min_exec_us: int = 0,
    category: str = CATEGORY,
) -> list[str] | None:
    t_min = _run_tmin_us(run)
    if t_min is None:
        print("[skip] exact timeline (msg size): could not resolve t_min_us for run", run)
        return None

    rq = _sql_quote(run)
    start_us = int(t_min + float(window_start_s) * 1e6)
    end_us = int(start_us + float(window_dur_s) * 1e6)

    limit_sql = f"LIMIT {int(max_msg)}" if max_msg else ""
    if keep_colls is None:
        coll_filter_sql = ""
    else:
        keep_sql = ",".join(["'" + _sql_quote(c) + "'" for c in keep_colls])
        coll_filter_sql = f"AND coll IN ({keep_sql})"

    msgs = q(
        f"""
SELECT
  "rank"::INT AS rank,
  dump_timestamp_us::BIGINT AS end_us,
  coll_exec_time_us::DOUBLE AS exec_us,
  coll_msg_size_bytes::DOUBLE AS msg_bytes,
  coll::VARCHAR AS coll
FROM logs
WHERE run = '{rq}'
  AND "rank" IS NOT NULL
  AND dump_timestamp_us IS NOT NULL
  AND coll_exec_time_us IS NOT NULL
  AND coll_msg_size_bytes IS NOT NULL
  AND coll IS NOT NULL
  AND dump_timestamp_us BETWEEN {start_us} AND {end_us}
  {coll_filter_sql}
  AND coll_exec_time_us >= {int(min_exec_us)}
ORDER BY dump_timestamp_us ASC
{limit_sql}
"""
    ).copy()

    print("selected msgs:", len(msgs), {"window_s": window_dur_s, "max": max_msg})
    if msgs.empty:
        print("[skip] exact timeline (msg size): no messages in selected window")
        return None

    rank_to_row, host_bounds = _build_rank_layout_by_host_index(run, ranks_per_host=8)
    if rank_to_row:
        y = msgs["rank"].astype(int).map(lambda r: rank_to_row.get(int(r), None))
        ok = y.notna()
        msgs = msgs.loc[ok].copy()
        msgs["y"] = y.loc[ok].astype(int)
    else:
        uniq = sorted(msgs["rank"].astype(int).unique().tolist())
        r2i = {r: i for i, r in enumerate(uniq)}
        msgs["y"] = msgs["rank"].astype(int).map(r2i)

    msgs["x1"] = (msgs["end_us"].astype(float) - float(t_min)) / 1e6
    msgs["x0"] = (msgs["end_us"].astype(float) - msgs["exec_us"].astype(float) - float(t_min)) / 1e6

    segs = np.stack(
        [
            np.column_stack([msgs["x0"].to_numpy(), msgs["y"].to_numpy()]),
            np.column_stack([msgs["x1"].to_numpy(), msgs["y"].to_numpy()]),
        ],
        axis=1,
    )

    colls_present = msgs["coll"].astype(str).unique().tolist()
    ordered_colls, coll_to_color = _coll_palette_no_grey(colls_present)

    base_hex = msgs["coll"].astype(str).map(coll_to_color)
    base_rgb = np.array([mcolors.to_rgb(h) for h in base_hex.to_list()], dtype=float)

    # Shade by per-message msg size: robust log-scale normalization within this window
    msgb = pd.to_numeric(msgs["msg_bytes"], errors="coerce").to_numpy(dtype=float)
    v = np.log10(np.maximum(msgb, 1.0))
    t_raw = np.zeros_like(v, dtype=float)
    okv = np.isfinite(v)
    if okv.any():
        vv = v[okv]
        vmin = float(np.nanpercentile(vv, 5.0))
        vmax = float(np.nanpercentile(vv, 95.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vv))
            vmax = float(np.nanmax(vv))
        if vmax > vmin:
            t_raw[okv] = (vv - vmin) / (vmax - vmin)
        else:
            t_raw[okv] = 1.0
    # Make colors deeper: boost mid/high values and reduce white mixing.
    # - Higher floor increases saturation even for small values.
    # - Gamma < 1 boosts mid-tones so the plot looks less washed out.
    t_raw = np.clip(t_raw, 0.0, 1.0)
    shade_floor = 0.35
    shade_gamma = 0.70
    t_boost = np.power(t_raw, shade_gamma)
    t_vis = shade_floor + (1.0 - shade_floor) * t_boost

    white = np.ones_like(base_rgb)
    colors = white * (1.0 - t_vis[:, None]) + base_rgb * t_vis[:, None]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    # Rasterize dense segments so PDF output doesn't store millions of vector primitives.
    ax.set_rasterization_zorder(0)
    lc = LineCollection(segs, colors=colors, linewidths=0.3, alpha=0.9, rasterized=True, zorder=0)
    ax.add_collection(lc)

    ax.set_xlim(float(window_start_s), float(window_start_s) + float(window_dur_s))
    ax.set_xlabel("time (s since run start)")
    ax.set_ylabel("rank (grouped by host index)")
    ax.set_ylim(-1, int(msgs["y"].max()) + 1)

    # Avoid strong black separators in paper PDFs; use a very light gray guide instead.
    for b in host_bounds[1:-1]:
        ax.axhline(b - 0.5, color="#D0D0D0", lw=0.2, alpha=0.25)

    import matplotlib.patches as mpatches

    patches = [mpatches.Patch(color=coll_to_color[c], label=c) for c in ordered_colls]
    fig.legend(
        handles=patches,
        ncol=min(6, max(1, len(patches))),
        fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    # Shade legend (outside): msg size absolute labels
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    shade_cmap = LinearSegmentedColormap.from_list("shade", ["#FFFFFF", "#000000"])
    sm = cm.ScalarMappable(cmap=shade_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.55])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("msg size", fontsize=10)

    if np.isfinite(msgb).any():
        p5, p50, p95 = np.nanpercentile(msgb[np.isfinite(msgb)], [5, 50, 95])

        def _fmt_mib(x: float) -> str:
            return f"{x / (1024**2):.2g} MiB"

        v5, v50, v95 = np.log10(max(p5, 1.0)), np.log10(max(p50, 1.0)), np.log10(max(p95, 1.0))
        if vmax > vmin:
            ticks_raw = np.clip([(v5 - vmin) / (vmax - vmin), (v50 - vmin) / (vmax - vmin), (v95 - vmin) / (vmax - vmin)], 0.0, 1.0)
            ticks = 0.15 + 0.85 * np.array(ticks_raw)
            cb.set_ticks(ticks.tolist())
            cb.set_ticklabels([_fmt_mib(float(p5)), _fmt_mib(float(p50)), _fmt_mib(float(p95))])
    cb.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.14)
    out = savefig_paper(
        f"identifier_exact_timeline_rank_time_{window_start_s:g}-{(window_start_s + window_dur_s):g}s_{run_suffix(run)}",
        category=category,
        formats=("pdf", "png"),
    )
    plt.close(fig)
    return out


def plot_exact_timeline_color_by_coll_shade_by_busbw(
    *,
    run: str,
    window_start_s: float,
    window_dur_s: float,
    keep_colls: list[str] | None = None,
    min_exec_us: int = 0,
    category: str = CATEGORY,
) -> list[str] | None:
    t_min = _run_tmin_us(run)
    if t_min is None:
        print("[skip] exact timeline (busbw): could not resolve t_min_us for run", run)
        return None

    rq = _sql_quote(run)
    start_us = int(t_min + float(window_start_s) * 1e6)
    end_us = int(start_us + float(window_dur_s) * 1e6)

    if keep_colls is None:
        coll_filter_sql = ""
    else:
        keep_sql = ",".join(["'" + _sql_quote(c) + "'" for c in keep_colls])
        coll_filter_sql = f"AND coll IN ({keep_sql})"

    msgs = q(
        f"""
SELECT
  "rank"::INT AS rank,
  dump_timestamp_us::BIGINT AS end_us,
  coll_exec_time_us::DOUBLE AS exec_us,
  coll_busbw_gbs::DOUBLE AS busbw_gbs,
  coll::VARCHAR AS coll
FROM logs
WHERE run = '{rq}'
  AND "rank" IS NOT NULL
  AND dump_timestamp_us IS NOT NULL
  AND coll_exec_time_us IS NOT NULL
  AND coll_busbw_gbs IS NOT NULL
  AND coll IS NOT NULL
  AND dump_timestamp_us BETWEEN {start_us} AND {end_us}
  {coll_filter_sql}
  AND coll_exec_time_us >= {int(min_exec_us)}
ORDER BY dump_timestamp_us ASC
"""
    ).copy()

    print("selected msgs:", len(msgs), {"window_s": window_dur_s})
    if msgs.empty:
        print("[skip] exact timeline (busbw): no messages in selected window")
        return None

    rank_to_row, host_bounds = _build_rank_layout_by_host_index(run, ranks_per_host=8)
    if rank_to_row:
        y = msgs["rank"].astype(int).map(lambda r: rank_to_row.get(int(r), None))
        ok = y.notna()
        msgs = msgs.loc[ok].copy()
        msgs["y"] = y.loc[ok].astype(int)
    else:
        uniq = sorted(msgs["rank"].astype(int).unique().tolist())
        r2i = {r: i for i, r in enumerate(uniq)}
        msgs["y"] = msgs["rank"].astype(int).map(r2i)

    msgs["x1"] = (msgs["end_us"].astype(float) - float(t_min)) / 1e6
    msgs["x0"] = (msgs["end_us"].astype(float) - msgs["exec_us"].astype(float) - float(t_min)) / 1e6

    segs = np.stack(
        [
            np.column_stack([msgs["x0"].to_numpy(), msgs["y"].to_numpy()]),
            np.column_stack([msgs["x1"].to_numpy(), msgs["y"].to_numpy()]),
        ],
        axis=1,
    )

    colls_present = msgs["coll"].astype(str).unique().tolist()
    ordered_colls, coll_to_color = _coll_palette_no_grey(colls_present)

    base_hex = msgs["coll"].astype(str).map(coll_to_color)
    base_rgb = np.array([mcolors.to_rgb(h) for h in base_hex.to_list()], dtype=float)

    # Shade by per-message bandwidth: robust in-window normalization
    bw = pd.to_numeric(msgs["busbw_gbs"], errors="coerce").to_numpy(dtype=float)
    t = np.zeros_like(bw, dtype=float)
    ok = np.isfinite(bw)
    if ok.any():
        bw_ok = bw[ok]
        vmin = float(np.nanpercentile(bw_ok, 5.0))
        vmax = float(np.nanpercentile(bw_ok, 95.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(bw_ok))
            vmax = float(np.nanmax(bw_ok))
        if vmax > vmin:
            t[ok] = (bw_ok - vmin) / (vmax - vmin)
        else:
            t[ok] = 1.0

    # Make colors deeper (same mapping as msg-size plot).
    t_raw = np.clip(t, 0.0, 1.0)
    shade_floor = 0.35
    shade_gamma = 0.70
    t_boost = np.power(t_raw, shade_gamma)
    t_vis = shade_floor + (1.0 - shade_floor) * t_boost

    white = np.ones_like(base_rgb)
    colors = white * (1.0 - t_vis[:, None]) + base_rgb * t_vis[:, None]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    # Rasterize dense segments so PDF output doesn't store millions of vector primitives.
    ax.set_rasterization_zorder(0)
    lc = LineCollection(segs, colors=colors, linewidths=0.3, alpha=0.9, rasterized=True, zorder=0)
    ax.add_collection(lc)

    ax.set_xlim(float(window_start_s), float(window_start_s) + float(window_dur_s))
    ax.set_xlabel("time (s since run start)")
    ax.set_ylabel("rank (grouped by host index)")
    ax.set_ylim(-1, int(msgs["y"].max()) + 1)

    # Avoid strong black separators in paper PDFs; use a very light gray guide instead.
    for b in host_bounds[1:-1]:
        ax.axhline(b - 0.5, color="#D0D0D0", lw=0.2, alpha=0.25)

    import matplotlib.patches as mpatches

    patches = [mpatches.Patch(color=coll_to_color[c], label=c) for c in ordered_colls]
    fig.legend(
        handles=patches,
        ncol=min(6, max(1, len(patches))),
        fontsize=10,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )

    # Shade legend (outside): bandwidth normalized 0..1 within window
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    shade_cmap = LinearSegmentedColormap.from_list("shade", ["#FFFFFF", "#000000"])
    sm = cm.ScalarMappable(cmap=shade_cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.55])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("bandwidth (norm)", fontsize=10)
    cb.set_ticks([0.15, 0.15 + 0.85 * 0.5, 1.0])
    cb.set_ticklabels(["0.0", "0.5", "1.0"])
    cb.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.14)
    out = savefig_paper(
        f"identifier_exact_timeline_rank_time_coll_hue_busbw_shade_{window_start_s:g}-{(window_start_s + window_dur_s):g}s_{run_suffix(run)}",
        category=category,
        formats=("pdf", "png"),
    )
    plt.close(fig)
    return out


# -----------------------
# Plots (two paper figures)
# -----------------------


def plot_straggler_hostname_execpct(
    *,
    run: str,
    coll_focus: str = "AllGather",
    min_msg_bytes: int = 1_000_000,
    ops_thresh: int = 100,
    topn: int = 20,
    category: str = CATEGORY,
) -> list[str] | None:
    """Hostname median exec time, normalized to % of run median (100%)."""
    rq = _sql_quote(run)
    strag = q(
        f"""
SELECT
  hostname,
  COUNT(*) AS ops,
  quantile_cont(coll_exec_time_us, 0.5) AS p50_exec_us
FROM logs
WHERE run = '{rq}'
  AND coll = '{_sql_quote(coll_focus)}'
  AND coll_msg_size_bytes >= {int(min_msg_bytes)}
GROUP BY hostname
HAVING ops >= {int(ops_thresh)}
ORDER BY p50_exec_us DESC
"""
    ).copy()

    if strag.empty:
        print(f"[skip] straggler plot: run={run} coll={coll_focus} (no hostnames above threshold)")
        return None

    med = float(strag["p50_exec_us"].median())
    strag["p50_exec_pct_of_median"] = pct(strag["p50_exec_us"], med)

    top = strag.head(int(topn)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.barh(top["hostname"], top["p50_exec_pct_of_median"], color=SOFT["orange"], edgecolor="none")
    ax.axvline(100, color=SOFT["gray"], linewidth=1.0)
    ax.set_xlabel(f"Median exec time (% of run median) — {coll_focus}")
    ax.set_ylabel("Hostname")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.2)

    out = savefig_paper(
        f"exp_straggler_hostname_execpct_{_slug(coll_focus)}_{run_suffix(run)}",
        category=category,
    )
    plt.close(fig)
    return out


def plot_topology_hostname_busbwpct(
    *,
    run: str,
    coll_focus: str = "AllGather",
    min_msg_bytes: int = 1_000_000,
    ops_thresh: int = 100,
    topn: int = 20,
    category: str = CATEGORY,
) -> list[str] | None:
    """Hostname median bus BW, normalized to % of run max (100%)."""
    rq = _sql_quote(run)
    bw_host = q(
        f"""
SELECT
  hostname,
  COUNT(*) AS ops,
  quantile_cont(coll_busbw_gbs, 0.5) AS p50_busbw_gbs
FROM logs
WHERE run = '{rq}'
  AND coll = '{_sql_quote(coll_focus)}'
  AND coll_msg_size_bytes >= {int(min_msg_bytes)}
GROUP BY hostname
HAVING ops >= {int(ops_thresh)}
ORDER BY p50_busbw_gbs ASC
"""
    ).copy()

    if bw_host.empty:
        print(f"[skip] topology plot: run={run} coll={coll_focus} (no hostnames above threshold)")
        return None

    mx = float(bw_host["p50_busbw_gbs"].max())
    bw_host["p50_busbw_pct_of_max"] = pct(bw_host["p50_busbw_gbs"], mx)

    worst = bw_host.head(int(topn)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.barh(worst["hostname"], worst["p50_busbw_pct_of_max"], color=SOFT["blue"], edgecolor="none")
    ax.set_xlabel(f"Median bus BW (% of run max) — {coll_focus}")
    ax.set_ylabel("Hostname")
    ax.set_xlim(0, 110)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.2)

    out = savefig_paper(
        f"exp_topology_hostname_busbwpct_{_slug(coll_focus)}_{run_suffix(run)}",
        category=category,
    )
    plt.close(fig)
    return out


def main() -> None:
    # Knobs: keep aligned with the notebook defaults
    run = RUN_FIXED
    coll_focus = "AllGather"
    min_msg_bytes = 1_000_000
    ops_thresh = 100

    plot_straggler_hostname_execpct(
        run=run,
        coll_focus=coll_focus,
        min_msg_bytes=min_msg_bytes,
        ops_thresh=ops_thresh,
    )
    plot_topology_hostname_busbwpct(
        run=run,
        coll_focus=coll_focus,
        min_msg_bytes=min_msg_bytes,
        ops_thresh=ops_thresh,
    )

    # Exact-timeline plots (as requested): two windows × two shading choices.
    # Note: the 10–110s window can be very large (many millions of segments) and may be slow.
    plot_exact_timeline_color_by_coll_shade_by_msg_size(
        run=run,
        window_start_s=0,
        window_dur_s=3,
        max_msg=None,
        keep_colls=None,
        min_exec_us=0,
    )
    plot_exact_timeline_color_by_coll_shade_by_msg_size(
        run=run,
        window_start_s=10,
        window_dur_s=100,
        max_msg=None,
        keep_colls=None,
        min_exec_us=0,
    )
    plot_exact_timeline_color_by_coll_shade_by_busbw(
        run=run,
        window_start_s=0,
        window_dur_s=3,
        keep_colls=None,
        min_exec_us=0,
    )
    plot_exact_timeline_color_by_coll_shade_by_busbw(
        run=run,
        window_start_s=10,
        window_dur_s=100,
        keep_colls=None,
        min_exec_us=0,
    )


if __name__ == "__main__":
    main()


