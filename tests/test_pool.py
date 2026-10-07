from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sw_daily.pool.clustering import (
    _build_multi_rep_candidate_caps,
    _return_scores,
    cluster_and_select,
    pairwise_overlap_matrices,
    prepare_recent_window,
)
from sw_daily.pool.export import export_mapping


def _returns_panel(codes: list[str], days: int = 260, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    base = rng.normal(0.0, 0.01, size=(days, len(codes)))
    return pd.DataFrame(base, index=dates, columns=codes)


def test_return_scores_orders_by_sharpe_and_momentum() -> None:
    days = 260
    dates = pd.bdate_range("2025-01-01", periods=days)
    strong = pd.Series(0.002, index=dates)
    weak = pd.Series(-0.002, index=dates)
    flat = pd.Series(0.0, index=dates)
    returns = pd.DataFrame({"A": strong, "B": weak, "C": flat})
    scores = _return_scores(["A", "B", "C"], returns)
    assert scores.index[0] == "A"
    assert scores.index[-1] == "B"


def test_prepare_recent_window_tails_and_returns() -> None:
    codes = ["A", "B"]
    close = (_returns_panel(codes, days=300) + 0.001).cumsum() + 100
    recent_close, recent_vol, returns = prepare_recent_window(close, None, lookback_days=252)
    assert len(recent_close) == 253
    assert recent_vol is None
    assert len(returns) == 253
    assert returns.iloc[0].isna().all()


def test_pairwise_overlap_identical_series_corr_one() -> None:
    returns = _returns_panel(["A", "B"])
    returns["C"] = returns["A"]
    corr, overlap, dist = pairwise_overlap_matrices(returns, min_overlap_days=10)
    assert corr.loc["A", "C"] == pytest.approx(1.0)
    assert dist.loc["A", "C"] == pytest.approx(0.0)
    assert overlap.loc["A", "B"] == len(returns)


def test_cluster_and_select_groups_clones_and_covers_clusters() -> None:
    returns = _returns_panel(["A", "B", "C", "D"], days=260)
    returns["B"] = returns["A"] * 1.01  # near-clone of A
    returns["D"] = returns["C"] * 0.99  # near-clone of C
    Z, clusters, cluster_df, selected, reasons = cluster_and_select(
        returns,
        None,
        t=0.40,
        target_count=4,
        history_days=pd.Series(260.0, index=returns.columns),
    )
    assert clusters.loc["A"] == clusters.loc["B"]
    assert clusters.loc["C"] == clusters.loc["D"]
    assert len(clusters.unique()) == 2
    # near-clone dedup keeps at most one of each clone pair
    assert not ({"A", "B"} <= set(selected))
    assert not ({"C", "D"} <= set(selected))
    # every cluster keeps a representative
    represented = {clusters.loc[code] for code in selected}
    assert represented == set(clusters.unique())
    assert set(selected) == set(reasons)


def test_candidate_caps_cover_every_cluster() -> None:
    cluster_df = pd.DataFrame(
        {
            "cluster": [1, 2, 3],
            "candidate_members": [["A"], ["B", "C"], ["D", "E", "F"]],
            "candidate_top_return": [0.1, 0.2, 0.3],
        }
    )
    caps = _build_multi_rep_candidate_caps(cluster_df, target_count=3)
    cap_map = dict(zip(caps["cluster"], caps["candidate_cap"]))
    assert cap_map[1] == 1
    assert cap_map[2] >= 1
    assert cap_map[3] >= 1


def test_export_mapping_writes_selected_csv(tmp_path: Path) -> None:
    universe = pd.DataFrame(
        {
            "code": ["801010", "801011"],
            "name": ["农林牧渔", "种植业"],
            "parent_industry": ["农林牧渔", "农林牧渔"],
        }
    )
    clusters = pd.Series({"801010": 1, "801011": 1}, name="cluster")
    cluster_df = pd.DataFrame({"cluster": [1], "n": [2]})
    out_csv = tmp_path / "mapping.csv"
    selected_csv = tmp_path / "selected.csv"
    export_mapping(
        ["801010", "801011"],
        clusters,
        ["801011"],
        {"801011": "multi_rep_equal_cluster_1_rank_1_coverage"},
        cluster_df,
        {"801010": "农林牧渔", "801011": "种植业"},
        universe=universe,
        out_csv=out_csv,
        selected_csv=selected_csv,
    )
    mapping = pd.read_csv(out_csv, dtype=str)
    assert list(mapping["code"]) == ["801010", "801011"]
    assert mapping.loc[mapping["code"] == "801011", "selected"].iloc[0] == "True"
    assert "parent_industry" in mapping.columns
    selected = pd.read_csv(selected_csv, dtype=str)
    assert list(selected["code"]) == ["801011"]
