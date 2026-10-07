"""High-level :func:`select_pool` pipeline for Shenwan industry indexes.

Ported from ``etf_daily/pool/builder/selector.py``. The ETF version reads a
fund list and applies type / inception / whitelist filters; the Shenwan
universe is just the second-level industry info CSV, so the only exclusion
reasons are ``missing_from_qlib_data`` and ``insufficient_history_for_clustering``.

Writes under ``paths.POOL_DIR``:

- ``sw_cluster_mapping.csv`` (every industry: cluster, selected flag, reason)
- ``sw_cluster_mapping_selected.csv`` (selected rows only)
- ``sw_cluster_mapping_selected_metadata.json``
- ``dendrogram_selected_reps.png`` / ``.svg``
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from sw_daily.paths import POOL_DIR, SECOND_INFO_CSV

from . import clustering as C
from .clustering import (
    cluster_and_select,
    prepare_recent_window,
    summarize_overlap_counts,
)
from .data_loading import load_close_volume
from .dendrogram import plot_selected_dendrogram
from .export import export_mapping, write_json

CSV_OUT = POOL_DIR / "sw_cluster_mapping.csv"
CSV_SELECTED_OUT = POOL_DIR / "sw_cluster_mapping_selected.csv"
METADATA_OUT = POOL_DIR / "sw_cluster_mapping_selected_metadata.json"
IMG_OUT = POOL_DIR / "dendrogram_selected_reps.png"
SVG_OUT = POOL_DIR / "dendrogram_selected_reps.svg"


def load_sw_universe() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the second-level industry universe from the ETL info CSV."""
    if not SECOND_INFO_CSV.is_file():
        raise SystemExit(f"missing {SECOND_INFO_CSV}; run: sw-daily etl info")
    frame = pd.read_csv(SECOND_INFO_CSV, dtype=str).fillna("")
    if "行业代码" not in frame.columns or "行业名称" not in frame.columns:
        raise SystemExit(f"{SECOND_INFO_CSV} needs 行业代码 and 行业名称 columns; got {list(frame.columns)}")
    universe = pd.DataFrame(
        {
            "code": frame["行业代码"].astype(str).str.strip().str[:6],
            "name": frame["行业名称"].astype(str).str.strip(),
        }
    )
    if "上级行业" in frame.columns:
        universe["parent_industry"] = frame["上级行业"].astype(str).str.strip()
    universe = universe[universe["code"] != ""].drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    code_name_map = dict(zip(universe["code"], universe["name"]))
    return universe, code_name_map


def _build_diagnostic_view(close: pd.DataFrame, vol: pd.DataFrame | None) -> dict[str, object]:
    """Build row-level diagnostics and clustering inputs for the universe."""
    _, recent_vol, returns_raw = prepare_recent_window(close, vol, C.CLUSTER_LOOKBACK_DAYS)
    recent_valid_days = returns_raw.notna().sum().astype(int)
    available_history_days = close.notna().sum().astype(int)
    eligible_for_cluster = recent_valid_days >= C.MIN_CLUSTER_HISTORY_DAYS
    eligible_for_selection = recent_valid_days >= C.MIN_SELECTION_HISTORY_DAYS
    diagnostics_df = pd.DataFrame(
        {
            "code": close.columns,
            "available_history_days": close.columns.map(lambda c: int(available_history_days.get(c, 0))),
            "recent_valid_days": close.columns.map(lambda c: int(recent_valid_days.get(c, 0))),
            "eligible_for_cluster": close.columns.map(lambda c: bool(eligible_for_cluster.get(c, False))),
            "eligible_for_selection": close.columns.map(lambda c: bool(eligible_for_selection.get(c, False))),
        }
    )
    eligible_codes = diagnostics_df.loc[diagnostics_df["eligible_for_cluster"], "code"].tolist()
    overlap_columns = ["overlap_min_days", "overlap_median_days", "overlap_max_days"]

    if not eligible_codes:
        for col in overlap_columns:
            diagnostics_df[col] = 0
        return {
            "diagnostics_df": diagnostics_df,
            "eligible_codes": eligible_codes,
            "returns": returns_raw.iloc[:, 0:0],
            "cluster_vol": recent_vol.iloc[:, 0:0] if recent_vol is not None else None,
            "recent_valid_days": recent_valid_days,
            "corr": None,
            "overlap": None,
        }

    returns = returns_raw[eligible_codes]
    cluster_vol = recent_vol[eligible_codes] if recent_vol is not None else None
    corr, overlap, _ = C.pairwise_overlap_matrices(returns, C.MIN_PAIR_OVERLAP_DAYS, C.LOW_OVERLAP_DISTANCE)
    overlap_summary = summarize_overlap_counts(overlap)
    diagnostics_df = diagnostics_df.merge(overlap_summary, on="code", how="left")
    diagnostics_df[overlap_columns] = diagnostics_df[overlap_columns].fillna(0)
    return {
        "diagnostics_df": diagnostics_df,
        "eligible_codes": eligible_codes,
        "returns": returns,
        "cluster_vol": cluster_vol,
        "recent_valid_days": recent_valid_days,
        "corr": corr,
        "overlap": overlap,
    }


def select_pool(
    *,
    test_period: tuple[str, str] = ("2020-01-01", "2026-08-31"),
) -> dict:
    """Run the full selection pipeline and return a metadata dict."""
    universe, code_name_map = load_sw_universe()
    codes = universe["code"].tolist()
    close, vol = load_close_volume(codes, test_period=test_period)

    loaded_codes = set(close.columns.tolist())
    missing_provider_codes = sorted(set(codes) - loaded_codes)

    diag_view = _build_diagnostic_view(close, vol)
    diagnostics_df = diag_view["diagnostics_df"]
    eligible_codes = diag_view["eligible_codes"]
    if not eligible_codes:
        raise RuntimeError("No index meets the minimum recent-history requirement for clustering")
    returns = diag_view["returns"]
    cluster_vol = diag_view["cluster_vol"]
    recent_valid_days = diag_view["recent_valid_days"]
    corr = diag_view["corr"]
    overlap = diag_view["overlap"]

    reasons: dict[str, str] = {
        code: "missing_from_qlib_data" for code in missing_provider_codes
    }
    for code in diagnostics_df.loc[~diagnostics_df["eligible_for_cluster"], "code"].tolist():
        reasons.setdefault(code, "insufficient_history_for_clustering")
    print("Returns shape", returns.shape)

    Z, clusters, cluster_df, selected, cluster_reasons = cluster_and_select(
        returns,
        cluster_vol,
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        target_count=C.TARGET_SELECTED_COUNT,
        history_days=recent_valid_days.reindex(eligible_codes).fillna(0),
        corr=corr,
        overlap=overlap,
    )
    reasons.update(cluster_reasons)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    export_mapping(
        universe["code"].tolist(),
        clusters,
        selected,
        reasons,
        cluster_df,
        code_name_map,
        universe=universe,
        out_csv=CSV_OUT,
        selected_csv=CSV_SELECTED_OUT,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "production_rule": C.RULE,
        "dist_t": float(C.DIST_T),
        "per_cluster": int(C.PER_CLUSTER),
        "target_selected_count": int(C.TARGET_SELECTED_COUNT),
        "cluster_lookback_days": int(C.CLUSTER_LOOKBACK_DAYS),
        "min_cluster_history_days": int(C.MIN_CLUSTER_HISTORY_DAYS),
        "min_selection_history_days": int(C.MIN_SELECTION_HISTORY_DAYS),
        "min_pair_overlap_days": int(C.MIN_PAIR_OVERLAP_DAYS),
        "test_period": [test_period[0], test_period[1]],
        "selected_csv": str(CSV_SELECTED_OUT),
        "selected_count": int(len(selected)),
    }
    write_json(METADATA_OUT, metadata)
    print("Wrote generation metadata to", METADATA_OUT)

    codes_in_cluster = list(clusters.index)
    plot_selected_dendrogram(Z, codes_in_cluster, set(selected), code_name_map, str(IMG_OUT), str(SVG_OUT))
    return metadata
