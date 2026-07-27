#!/usr/bin/env python3
"""
Paper-ready paired-ECDF figure for Section 3.7 (Comparison to nccl-tests Baseline).

For each dominant 340B@2048 communicator topology, overlays the per-call
bus-bandwidth ECDF from production training (solid) against the corresponding
nccl-tests run in isolation (dashed). The mean is roughly preserved across
the two sources; what changes is the dispersion, which is the point.

Note: nccl-tests was run at smaller message sizes than the production picks
described in the table (paper Section 3.7). We label each curve with its
actual (n_ranks, nnodes, coll, msg) so the figure stays honest. The CV
reported on each curve is what the figure is really comparing.

Output:
  figures/exp_ecdf_busbw_training_vs_ncclperf_340b_2048.{pdf,png}
"""

from __future__ import annotations
import os as _os
NIXT_ROOT = _os.environ.get("NIXT_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: duckdb. Activate your nccl_exporter environment and retry.\n"
        "  conda activate nccl_exporter"
    ) from e

import matplotlib as mpl
import matplotlib.pyplot as plt


# ------------------------------
# Paths and styling
# ------------------------------

DATA_ROOT = NIXT_ROOT + "/data"
SUPDATA_ROOT = NIXT_ROOT + "/sup_data/NCCL2.30.4"
FIG_DIR = Path(NIXT_ROOT + "/figures")

PROD_RUN_340B = (
    "pretrain_nemotron4_340b_fp8_gpus2048_tp8_pp8_cp1_vp12_mbs1_gbs512_1756587606-analysis"
)

PARQUET_GLOB = f"{DATA_ROOT}/{PROD_RUN_340B}/parquet_files/*.parquet"

mpl.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
    }
)


# ------------------------------
# Helpers
# ------------------------------


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / float(x.size)
    return x, y


def _bytes_si(b: int) -> str:
    b = float(b)
    if b < 1000:
        return f"{int(round(b))}B"
    if b < 1000**2:
        return f"{b/1000:.3g}KB"
    if b < 1000**3:
        return f"{b/1000**2:.3g}MB"
    return f"{b/1000**3:.3g}GB"


def cv(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    m = arr.mean()
    if m == 0:
        return float("nan")
    return float(arr.std(ddof=0) / m)


# ------------------------------
# Data loaders
# ------------------------------


def load_nccl_tests_busbw(dir_glob: str) -> pd.DataFrame:
    """Stream a directory of nccl-tests inspector JSONL logs.

    Returns a DataFrame with: n_ranks, nnodes, coll, coll_msg_size_bytes,
    coll_busbw_gbs, coll_exec_time_us.
    """
    rows: list[dict] = []
    for fp in sorted(glob.glob(dir_glob)):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    h = rec.get("header", {})
                    m = rec.get("coll_perf", {})
                    if "coll_busbw_gbs" not in m:
                        continue
                    rows.append(
                        {
                            "n_ranks": h.get("n_ranks"),
                            "nnodes": h.get("nnodes"),
                            "coll": m.get("coll"),
                            "coll_msg_size_bytes": m.get("coll_msg_size_bytes"),
                            "coll_busbw_gbs": m.get("coll_busbw_gbs"),
                            "coll_exec_time_us": m.get("coll_exec_time_us"),
                        }
                    )
        except (json.JSONDecodeError, OSError) as e:
            print(f"  warn: skipped {fp}: {e}")
    return pd.DataFrame(rows)


_con: Optional[duckdb.DuckDBPyConnection] = None


def _duck() -> duckdb.DuckDBPyConnection:
    global _con
    if _con is None:
        _con = duckdb.connect()
        _con.execute(
            f"""
CREATE OR REPLACE VIEW logs AS
SELECT * FROM read_parquet('{PARQUET_GLOB}', filename=true);
"""
        )
    return _con


def load_production_busbw(
    *, n_ranks: int, nnodes: int, coll: str, msg_bytes: int
) -> pd.DataFrame:
    """Pull per-call bus bandwidth from production parquet for one pick."""
    con = _duck()
    return con.execute(
        """
SELECT coll_busbw_gbs, coll_msg_size_bytes
FROM logs
WHERE n_ranks = ? AND nnodes = ? AND coll = ? AND coll_msg_size_bytes = ?
"""
    , [n_ranks, nnodes, coll, msg_bytes]).df()


# ------------------------------
# Picks
# ------------------------------


@dataclass
class Pick:
    title: str
    prod_n_ranks: int
    prod_nnodes: int
    prod_coll: str
    prod_msg_bytes: int
    nccl_tests_dir: str  # glob pattern for *.log files


PICKS: List[Pick] = [
    Pick(
        title="Intra-node AllGather",
        prod_n_ranks=8,
        prod_nnodes=1,
        prod_coll="AllGather",
        prod_msg_bytes=18874368,  # 18 MiB (production-hot bucket)
        nccl_tests_dir=f"{SUPDATA_ROOT}/nccl_out_var_allgather_intranode_11729991_1/"
        f"inspector_allgather_intranode/*.log",
    ),
    Pick(
        title="Inter-node AllGather",
        prod_n_ranks=32,
        prod_nnodes=32,
        prod_coll="AllGather",
        prod_msg_bytes=13492224,  # 13.49 MiB (production-hot bucket)
        nccl_tests_dir=f"{SUPDATA_ROOT}/nccl_out_var_allgather_internode_11729992_32/"
        f"inspector_allgather_split_mask_0x1/*.log",
    ),
]


# ------------------------------
# Plot
# ------------------------------


def plot_paired_ecdfs(picks: List[Pick], out_path: Path) -> List[Path]:
    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.4))
    if n == 1:
        axes = [axes]

    for ax, pick in zip(axes, picks):
        prod = load_production_busbw(
            n_ranks=pick.prod_n_ranks,
            nnodes=pick.prod_nnodes,
            coll=pick.prod_coll,
            msg_bytes=pick.prod_msg_bytes,
        )
        nctest = load_nccl_tests_busbw(pick.nccl_tests_dir)

        # Normalize bw to % of panel peak so both curves share a unit-free axis.
        prod_arr = prod["coll_busbw_gbs"].to_numpy(dtype=float)
        nctest_arr = nctest["coll_busbw_gbs"].to_numpy(dtype=float) if not nctest.empty else None
        max_candidates = [prod_arr.max() if prod_arr.size else 0.0]
        if nctest_arr is not None and nctest_arr.size:
            max_candidates.append(nctest_arr.max())
        denom = max(max(max_candidates), 1e-12)
        norm = lambda a: a / denom * 100.0

        # Curve 1: production
        x_p, y_p = ecdf(norm(prod_arr))
        prod_label = (
            f"training: r{pick.prod_n_ranks},n{pick.prod_nnodes},"
            f"{pick.prod_coll}\n"
            f"$n$={x_p.size:,}, CV={cv(prod_arr):.3f}"
        )
        ax.plot(x_p, y_p, color="C0", linestyle="-", label=prod_label)

        # Curve 2: nccl-tests (one or more configurations may exist in dir;
        # plot each (n_ranks, nnodes, coll, msg) separately as a dashed line)
        if not nctest.empty:
            grp_cols = ["n_ranks", "nnodes", "coll", "coll_msg_size_bytes"]
            for keys, g in nctest.groupby(grp_cols, sort=False):
                nr, nn, c, mb = keys
                g_arr = g["coll_busbw_gbs"].to_numpy(dtype=float)
                x_n, y_n = ecdf(norm(g_arr))
                nctest_label = (
                    f"nccl-tests: r{int(nr)},n{int(nn)},{c}\n"
                    f"$n$={x_n.size:,}, CV={cv(g_arr):.3f}"
                )
                ax.plot(x_n, y_n, color="C3", linestyle="--", label=nctest_label)
        else:
            print(f"  warn: no nccl-tests data for {pick.title}")

        ax.set_xlabel("Norm BW")
        ax.set_ylabel("ECDF")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=True, fontsize=11)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for fmt in ("pdf", "png"):
        out = out_path.with_suffix(f".{fmt}")
        if fmt == "pdf":
            fig.savefig(out, bbox_inches="tight")
        else:
            fig.savefig(out, dpi=200, bbox_inches="tight")
        written.append(out)
        print(f"wrote {out}")
    plt.close(fig)
    return written


# ------------------------------
# Main
# ------------------------------


def main() -> None:
    out_path = FIG_DIR / "exp_ecdf_busbw_training_vs_ncclperf_340b_2048.pdf"
    plot_paired_ecdfs(PICKS, out_path)


if __name__ == "__main__":
    main()
