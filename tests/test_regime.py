from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sw_daily.regime.backtest import run_validation
from sw_daily.regime.board import compute_frozen_transition_step, fit_frozen_regime_model
from sw_daily.regime.hmm import HmmConfig, fit_hmm_kstate, forward_filter_step, replay_forward_filter
from sw_daily.regime.validation import (
    cluster_positive_events,
    compute_trend_up_label,
    match_alerts_to_events,
    select_fdr_threshold,
)


def _two_regime_returns(n: int = 400, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0002, 0.004, n // 2)
    wild = rng.normal(-0.001, 0.02, n - n // 2)
    return np.concatenate([calm, wild])


def test_hmm_fit_separates_two_volatility_regimes() -> None:
    returns = _two_regime_returns()
    fit = fit_hmm_kstate(returns, HmmConfig(n_states=2, min_samples=60, max_em_iter=40))
    assert fit.reliable
    assert fit.sigma_ratio is not None and fit.sigma_ratio > 1.15
    assert np.isclose(fit.pi_filt.sum(), 1.0)
    assert np.allclose(fit.trans.sum(axis=1), 1.0)


def test_forward_filter_probabilities_sum_to_one() -> None:
    returns = _two_regime_returns()
    fit = fit_hmm_kstate(returns, HmmConfig(n_states=2, min_samples=60, max_em_iter=30))
    alpha = replay_forward_filter(fit, returns[:80])
    step = forward_filter_step(fit, alpha, float(returns[80]))
    assert np.isclose(step.alpha.sum(), 1.0)
    assert np.isclose(step.xi.sum(), 1.0)


def test_large_up_day_is_an_up_quantile_switch() -> None:
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.008, 220)
    prices = 100.0 * np.exp(np.cumsum(rets))
    prices = np.append(prices, prices[-1] * 1.12)
    index = pd.bdate_range("2024-01-02", periods=len(prices))
    close = pd.Series(prices, index=index)
    model = fit_frozen_regime_model(close, index[-2])
    metrics, alpha, _step = compute_frozen_transition_step(close, index[-1], model)
    assert metrics.reliable
    assert alpha is not None
    assert metrics.quantile_switch
    assert metrics.switch_side == "up"


def test_trend_up_label_accepts_a_clean_rally_and_rejects_noise() -> None:
    rally = pd.Series(np.linspace(100, 140, 80), index=pd.bdate_range("2025-01-02", periods=80))
    assert compute_trend_up_label(rally, rally.index[40], horizon=10) is True
    rng = np.random.default_rng(9)
    noise_px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, 80)))
    noise = pd.Series(noise_px, index=rally.index)
    assert compute_trend_up_label(noise, noise.index[40], horizon=10) is False


def test_fdr_threshold_picks_the_selective_score() -> None:
    rows = []
    for i in range(60):
        high = i < 40
        rows.append(
            {
                "p_trend_cal": 0.9 if high else 0.2,
                "label_trend_up": high and i >= 5,
                "month": "2025-01" if i < 30 else "2025-02",
            }
        )
    thr = select_fdr_threshold(pd.DataFrame(rows), target_fdr=0.35, min_confirm=20, n_bootstrap=40)
    assert thr["q_confirm"] == pytest.approx(0.9)
    assert thr["n_confirm_support"] >= 20


def test_events_merge_small_gaps_and_mark_unmatched_alerts_as_false_positives() -> None:
    dates = pd.bdate_range("2025-03-03", periods=12)
    labels = pd.DataFrame(
        {
            "code": ["A"] * 12,
            "as_of": dates,
            "label_trend_up": [True, True, False, False, True, True, False, False, False, False, False, False],
        }
    )
    events = cluster_positive_events(labels, max_gap=5)
    assert len(events) == 1
    alerts = pd.DataFrame(
        {
            "code": ["A", "A"],
            "as_of": [dates[1], dates[10]],
            "p_trend_cal": [0.8, 0.4],
        }
    )
    matched = match_alerts_to_events(alerts, events, lead=3, horizon=5)
    assert pd.notna(matched.iloc[0]["matched_event_onset"])
    assert bool(matched.iloc[1]["is_fp"])


def test_run_validation_writes_a_decision_pack(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    codes = [f"80{i:04d}" for i in range(6)]
    index = pd.bdate_range("2024-01-02", periods=260)
    panel = pd.DataFrame(100.0, index=index, columns=codes)
    for code in codes:
        panel[code] = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(index))))
    result = run_validation(
        panel,
        codes,
        name_map={code: code for code in codes},
        eval_start="2024-11-01",
        eval_end="2024-12-31",
        horizon=5,
        jobs=1,
        output_dir=tmp_path,
        use_cache=False,
        resume=False,
    )
    signals = pd.read_csv(tmp_path / "signals_oos.csv")
    assert not signals.empty
    assert {"as_of", "code", "signal_level", "quantile_switch", "label_trend_up"} <= set(signals.columns)
    assert (tmp_path / "go_nogo.json").is_file()
    assert (tmp_path / "REGIME_TRANSITION_VALIDATION.md").is_file()
    assert result["go_no_go"] in {"PASS", "BLOCKED"}
