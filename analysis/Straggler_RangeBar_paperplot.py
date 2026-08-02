"""
Paper-ready straggler comparison figures: side-by-side range bars (healthy vs
straggler) for the dominant Nemotron-4 15B @ 16 GPUs bucket
(mixed AllGather [8 ranks, 2 nodes], ~24 MiB).

Four figures are produced (each a 1x2 row, healthy on the left, straggler on
the right; bus bandwidth on the y-axis; range bar = min/max whisker, p25-p75
box, p50 marker):

  straggler_perid_rangebar_15b_mixedAG24MB.{pdf,png}
  straggler_perhost_rangebar_15b_mixedAG24MB.{pdf,png}
  straggler_pertimebin_rangebar_15b_mixedAG24MB.{pdf,png}
  straggler_percollsnbin_rangebar_15b_mixedAG24MB.{pdf,png}

The healthy reference and straggler-affected runs are referenced via the
HEALTHY_RUN / STRAGGLER_RUN constants below; their full Inspector log paths
contain dataset-specific timestamps and hostnames that the paper does not
disclose.
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

DATA_GLOB = NIXT_ROOT + "/data/*-analysis/parquet_files/*.parquet"
FIG_DIR = Path(NIXT_ROOT + "/figures/experiments/Straggler")
FIG_DIR.mkdir(parents=True, exist_ok=True)

HEALTHY_RUN = "pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1756589443-analysis"
STRAGGLER_A_RUN = "pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1757455306-analysis"
STRAGGLER_RUN = "pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1757716170-analysis"
STRAGGLER_B_RUN = STRAGGLER_RUN

# Dominant bucket present in both runs: mixed AllGather [8r,2n] 24.4 MiB
BUCKET_FILTER = "comm_type='mixed' AND coll='AllGather' AND coll_msg_size_bytes=24379392"
BUCKET_LABEL = "mixed AllGather [8r,2n] 24 MiB"

SOFT = {"blue": "#8DA0CB", "orange": "#FC8D62", "teal": "#66C2A5", "gray": "#666666", "pink": "#E78AC3"}
COLOR_HEALTHY = SOFT["teal"]
COLOR_STRAGGLER_A = SOFT["pink"]
COLOR_STRAGGLER = SOFT["orange"]
COLOR_STRAGGLER_B = COLOR_STRAGGLER


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
    *,
    run: str,
    entity_expr: str,
    min_ops: int = 20,
    extra_with: str = "",
    time_limit_s: float | None = None,
    t_origin_us: int | None = None,
) -> pd.DataFrame:
    where = f"run='{run}' AND {BUCKET_FILTER}"
    if extra_with:
        if t_origin_us is not None:
            t0_def = f"{int(t_origin_us)} AS t0_us"
            t_filter = f"AND dump_timestamp_us >= {int(t_origin_us)}"
        else:
            t0_def = f"(SELECT MIN(dump_timestamp_us) FROM logs WHERE {where}) AS t0_us"
            t_filter = ""
        time_clip = f"AND (dump_timestamp_us - t0_us) / 1e6 <= {time_limit_s}" if time_limit_s else ""
        sql = f"""
WITH base AS (
  SELECT *, {t0_def}
  FROM logs WHERE {where} {t_filter}
)
SELECT {entity_expr} AS x,
  COUNT(*) AS n,
  MIN(coll_busbw_gbs) AS lo, MAX(coll_busbw_gbs) AS hi,
  quantile_cont(coll_busbw_gbs, 0.25) AS p25,
  quantile_cont(coll_busbw_gbs, 0.50) AS p50,
  quantile_cont(coll_busbw_gbs, 0.75) AS p75,
  STDDEV_SAMP(coll_busbw_gbs)/NULLIF(AVG(coll_busbw_gbs),0) AS cv
FROM base
WHERE 1=1 {time_clip}
GROUP BY x
HAVING COUNT(*) >= {min_ops}
ORDER BY p50 ASC
"""
    else:
        sql = f"""
SELECT {entity_expr} AS x,
  COUNT(*) AS n,
  MIN(coll_busbw_gbs) AS lo, MAX(coll_busbw_gbs) AS hi,
  quantile_cont(coll_busbw_gbs, 0.25) AS p25,
  quantile_cont(coll_busbw_gbs, 0.50) AS p50,
  quantile_cont(coll_busbw_gbs, 0.75) AS p75,
  STDDEV_SAMP(coll_busbw_gbs)/NULLIF(AVG(coll_busbw_gbs),0) AS cv
FROM logs WHERE {where}
GROUP BY x
HAVING COUNT(*) >= {min_ops}
ORDER BY p50 ASC
"""
    return con.execute(sql).df().reset_index(drop=True)


def draw_rangebar(ax: plt.Axes, df: pd.DataFrame, color: str, x_use_value: bool) -> None:
    x = np.atleast_1d(df["x"].values if x_use_value else np.arange(len(df)))
    lo = np.atleast_1d(df["lo"].values)
    hi = np.atleast_1d(df["hi"].values)
    p25 = np.atleast_1d(df["p25"].values)
    p50 = np.atleast_1d(df["p50"].values)
    p75 = np.atleast_1d(df["p75"].values)
    if x_use_value:
        ax.vlines(x, lo, hi, color=color, alpha=0.35, linewidth=0.6)
        ax.vlines(x, p25, p75, color=color, alpha=0.95, linewidth=1.6)
        ax.plot(x, p50, "o", color="black", markersize=2.0,
                markerfacecolor="white", markeredgewidth=0.6)
        return
    # Index-mode panels (per-ID / per-host): bars are 1 unit apart, so we can
    # draw real IQR rectangles instead of hairline vlines that get lost in a
    # 4-inch panel with only 1-3 entries.
    box_w = 0.8
    cap_w = box_w * 0.6
    ax.vlines(x, lo, hi, color=color, alpha=0.55, linewidth=1.0, zorder=2)
    ax.hlines(lo, x - cap_w / 2, x + cap_w / 2, color=color, alpha=0.55, linewidth=1.0, zorder=2)
    ax.hlines(hi, x - cap_w / 2, x + cap_w / 2, color=color, alpha=0.55, linewidth=1.0, zorder=2)
    for xi, q1, q3 in zip(x, p25, p75):
        ax.add_patch(plt.Rectangle(
            (xi - box_w / 2, q1), box_w, q3 - q1,
            facecolor=color, edgecolor=color, alpha=0.85,
            linewidth=0.8, zorder=3,
        ))
    ax.hlines(p50, x - box_w / 2, x + box_w / 2, color="black", linewidth=1.3, zorder=4)


def make_compare_fig(
    con: duckdb.DuckDBPyConnection,
    *,
    out_name: str,
    entity_expr: str,
    xlabel: str,
    extra_with: str = "",
    x_use_value: bool = False,
    y_share: bool = True,
    time_limit_s: float | None = None,
    straggler_t_origin_us: int | None = None,
    xticks_from_data: bool = False,
) -> list[str]:
    set_rcparams()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), sharey=y_share, constrained_layout=True)
    panel_inputs = [
        (HEALTHY_RUN,    "Healthy",     COLOR_HEALTHY,    None),
        (STRAGGLER_A_RUN, "Straggler-A", COLOR_STRAGGLER_A, None),
        (STRAGGLER_B_RUN, "Straggler-B", COLOR_STRAGGLER_B, straggler_t_origin_us),
    ]
    # First pass: fetch all panels and compute a shared normalization base
    # (the healthy panel's max) so the straggler panel reads as a fraction of
    # the healthy peak rather than rescaling away the slowdown.
    panel_dfs: list[pd.DataFrame] = []
    for run, _label, _color, t_origin in panel_inputs:
        df = per_entity_stats(
            con, run=run, entity_expr=entity_expr,
            extra_with=extra_with, time_limit_s=time_limit_s,
            t_origin_us=t_origin,
        )
        panel_dfs.append(df)
    healthy_df = panel_dfs[0]
    denom = max(float(healthy_df["hi"].max()) if not healthy_df.empty else 0.0, 1e-12)
    for ax, (_run, _label, color, _t_origin), df in zip(axes, panel_inputs, panel_dfs):
        ax.set_title(_label)
        if df.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        for c in ("lo", "hi", "p25", "p50", "p75"):
            if c in df.columns:
                df[c] = df[c].astype(float) / denom * 100.0
        draw_rangebar(ax, df, color, x_use_value=x_use_value)
        ax.set_xlabel(xlabel)
        if xticks_from_data:
            ticks = (df["x"].values if x_use_value else np.arange(len(df))).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
        if not x_use_value:
            n = len(df)
            pad = 0.6
            ax.set_xlim(-pad, max(pad, (n - 1) + pad))
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Norm BW")
    out_pdf = FIG_DIR / f"{out_name}.pdf"
    out_png = FIG_DIR / f"{out_name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return [str(out_pdf), str(out_png)]


def find_straggler_window_origin_us(con: duckdb.DuckDBPyConnection) -> int:
    """First timestamp (us) at which the affected hosts (those *other* than the
    initial host) appear in the dominant bucket of the straggler run."""
    sql = f"""
WITH first_host AS (
  SELECT hostname FROM logs
  WHERE run='{STRAGGLER_RUN}' AND {BUCKET_FILTER}
  GROUP BY hostname ORDER BY MIN(dump_timestamp_us) LIMIT 1
)
SELECT MIN(dump_timestamp_us)::BIGINT FROM logs
WHERE run='{STRAGGLER_RUN}' AND {BUCKET_FILTER}
  AND hostname NOT IN (SELECT hostname FROM first_host)
"""
    return int(con.execute(sql).fetchone()[0])


def main() -> None:
    con = open_con()
    written: list[str] = []
    straggler_origin = find_straggler_window_origin_us(con)
    print(f"straggler window starts at us = {straggler_origin}")

    # 1) per communicator id
    written += make_compare_fig(
        con,
        out_name="straggler_perid_rangebar_15b_mixedAG24MB",
        entity_expr="id",
        xlabel="communicator ID",
        xticks_from_data=True,
    )
    # 2) per hostname
    written += make_compare_fig(
        con,
        out_name="straggler_perhost_rangebar_15b_mixedAG24MB",
        entity_expr="hostname",
        xlabel="hostname index",
    )
    # 3) per wall-clock time bin (5s). For the straggler run we shift t=0 to the
    #    moment the affected hosts join the dominant bucket (the 30 h "healthy
    #    warm-up" by the original host is excluded so both panels show the
    #    straggler-impact period).
    written += make_compare_fig(
        con,
        out_name="straggler_pertimebin_rangebar_15b_mixedAG24MB",
        entity_expr="FLOOR((dump_timestamp_us - t0_us) / 1e6 / 5) * 5",
        xlabel="Wall-clock time (s, 5-s bins)",
        extra_with="t0",
        x_use_value=True,
        y_share=True,
        straggler_t_origin_us=straggler_origin,
    )
    # 4) per coll_sn bin (50 ops)
    written += make_compare_fig(
        con,
        out_name="straggler_percollsnbin_rangebar_15b_mixedAG24MB",
        entity_expr="FLOOR(coll_sn / 50) * 50",
        xlabel="Collective sequence number (50-op bins)",
        x_use_value=True,
        y_share=True,
    )

    print("wrote:")
    for w in written:
        print(" -", w)


if __name__ == "__main__":
    main()
