from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sw_daily.pool.review import (
    build_missed_clusters,
    build_rep_lag,
    compute_trailing_return_metrics,
    load_mapping_csv,
    parse_windows,
    window_bounds,
)


def _mapping() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["甲", "乙", "丙", "丁"],
            "code": ["801010", "801011", "801020", "801021"],
            "cluster": [1, 1, 2, 2],
            "selected": [True, False, False, False],
            "reason": ["multi_rep_equal_cluster_1_rank_1_coverage", "", "", ""],
            "n": [2, 2, 2, 2],
        }
    )


def _future_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["801010", "801011", "801020", "801021"],
            "fwd_ret_20d": [0.01, 0.05, 0.08, 0.02],
        }
    )


def test_parse_windows_sorts_and_dedups() -> None:
    assert parse_windows("20,5,60,20") == [5, 20, 60]
    with pytest.raises(ValueError):
        parse_windows(" , ")


def test_window_bounds_is_business_days_before_anchor() -> None:
    bound = window_bounds("2026-10-07", [5, 20, 60])
    assert pd.Timestamp(bound) < pd.Timestamp("2026-10-07")
    gap_days = (pd.Timestamp("2026-10-07") - pd.Timestamp(bound)).days
    assert gap_days >= 100  # 60 windows + 40 buffer, business days


def test_load_mapping_csv_normalizes_selected(tmp_path) -> None:
    path = tmp_path / "mapping.csv"
    _mapping().to_csv(path, index=False)
    df = load_mapping_csv(path)
    assert df["selected"].tolist() == [True, False, False, False]
    assert df["cluster"].tolist() == [1, 1, 2, 2]


def test_build_missed_clusters_finds_cluster_without_rep() -> None:
    missed = build_missed_clusters(
        _mapping(),
        ["801010"],
        _future_df(),
        min_cluster_n=2,
        focus_window=20,
    )
    assert len(missed) == 1
    row = missed.iloc[0]
    assert row["cluster"] == 2
    assert row["best_code"] == "801020"
    assert row["best_20d_ret"] == pytest.approx(0.08)
    assert row["reason_kind"] == "target_count_not_selected"


def test_build_missed_clusters_skips_small_clusters() -> None:
    mapping = _mapping()
    mapping["n"] = [2, 2, 1, 1]
    missed = build_missed_clusters(mapping, ["801010"], _future_df(), min_cluster_n=2)
    assert missed.empty


def test_build_rep_lag_flags_lagging_representative() -> None:
    lag = build_rep_lag(
        _mapping(),
        ["801010"],
        _future_df(),
        pd.DataFrame(),
        min_regret=0.01,
        focus_window=20,
    )
    assert len(lag) == 1
    row = lag.iloc[0]
    assert row["selected_code"] == "801010"
    assert row["best_code"] == "801011"
    assert row["regret_20d"] == pytest.approx(0.04)
    assert row["rank_20d"] == 2


def test_build_rep_lag_quiet_when_rep_is_best() -> None:
    future = _future_df()
    future.loc[future["code"] == "801010", "fwd_ret_20d"] = 0.09
    lag = build_rep_lag(_mapping(), ["801010"], future, pd.DataFrame(), min_regret=0.01)
    assert lag.empty


def test_compute_trailing_return_metrics() -> None:
    dates = pd.bdate_range("2026-09-01", periods=30)
    close = pd.DataFrame(
        {
            "A": np.linspace(100, 110, 30),
            "B": np.linspace(100, 90, 30),
        },
        index=dates,
    )
    metrics = compute_trailing_return_metrics(close, "2026-10-07", [5, 20])
    a_row = metrics.loc[metrics["code"] == "A"].iloc[0]
    b_row = metrics.loc[metrics["code"] == "B"].iloc[0]
    assert a_row["fwd_ret_20d"] > 0
    assert b_row["fwd_ret_20d"] < 0
    assert a_row["market_ret_rank_20d"] == 1.0
