from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sw_daily.hrp.dendrogram import (
    compute_cluster_representatives,
    corr_medoid,
    prepare_returns,
    run_hrp,
    split_by_min_corr,
    ward_labels,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2025-01-02", periods=80)
    shared = rng.normal(0.0, 0.01, len(index))
    return pd.DataFrame(
        {
            "A": shared,
            "B": shared * 0.9 + rng.normal(0, 0.001, len(index)),
            "C": rng.normal(0.0, 0.02, len(index)),
        },
        index=index,
    )


def test_prepare_returns_drops_a_flat_series() -> None:
    close = pd.DataFrame({"A": np.linspace(100, 120, 30), "B": np.full(30, 50.0)})
    returns, used, dropped = prepare_returns(close, ["A", "B", "missing"])
    assert "A" in used
    assert "B" in dropped
    assert "missing" in dropped
    assert "B" not in returns.columns


def test_ward_labels_group_clones_apart_from_noise() -> None:
    labels = ward_labels(_returns(), dist_t=0.40)
    assert labels.loc["A"] == labels.loc["B"]
    assert labels.loc["C"] != labels.loc["A"]


def test_split_by_min_corr_isolates_a_weak_pair() -> None:
    labels = pd.Series({"A": 1, "B": 1}, name="cluster")
    corr = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=["A", "B"], columns=["A", "B"])
    split = split_by_min_corr(labels, corr, min_corr=0.55)
    assert split.loc["A"] != split.loc["B"]


def test_corr_medoid_picks_the_most_central_code() -> None:
    corr = pd.DataFrame(
        [[1, 0.9, 0.8], [0.9, 1, 0.85], [0.8, 0.85, 1]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    assert corr_medoid(["A", "B", "C"], corr) == "B"


def test_each_cluster_has_one_medoid_and_a_directional_mover() -> None:
    returns = _returns()
    close = (1 + returns).cumprod() * 100
    close.iloc[-1, close.columns.get_loc("A")] = close.iloc[-21, close.columns.get_loc("A")] * 1.12
    reps = compute_cluster_representatives(returns, close, name_map={"A": "甲", "B": "乙", "C": "丙"}, min_rep_move=1.0)
    medoids = reps[reps["is_representative"]]
    assert medoids.groupby("cluster")["code"].nunique().eq(1).all()
    assert reps["is_extreme_mover"].any()


def test_run_hrp_writes_html_and_csv(tmp_path: Path) -> None:
    returns = _returns()
    close = (1 + returns).cumprod() * 100
    html_path = tmp_path / "hrp.html"
    universe = run_hrp(close, ("A", "B", "C"), {"A": "甲", "B": "乙", "C": "丙"}, asof_date="2025-04-30", output=html_path, min_rep_move=0.0)
    assert universe.error is None
    assert html_path.is_file() and html_path.stat().st_size > 0
    csv_path = tmp_path / "cluster_representatives_20250430.csv"
    assert csv_path.is_file()
    assert "code" in pd.read_csv(csv_path).columns
