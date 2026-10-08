from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sw_daily.adaptive.methods import METHODS, build_px, retrospective_truth
from sw_daily.adaptive.run import run_adaptive
from sw_daily.adaptive.stage import (
    DEGENERATE_FALLBACK_METHOD,
    prepare_features,
    select_regime_method_legacy,
    simulate_hold_up,
)


def _ohlcv(close: np.ndarray, start: str = "2020-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "datetime": index,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
        }
    )


def _trend_then_drop(n: int = 400) -> np.ndarray:
    up = np.linspace(100, 220, n // 2)
    down = np.linspace(220, 120, n - n // 2)
    return np.concatenate([up, down])


def test_build_px_has_the_method_columns() -> None:
    px = build_px(_ohlcv(np.linspace(100, 140, 80)))
    for column in ("MA20", "MA60", "prior60", "band60", "rv20", "adx14", "plus_di", "minus_di"):
        assert column in px.columns


def test_retrospective_truth_marks_a_rally_and_a_decline() -> None:
    truth = retrospective_truth(_trend_then_drop(), prom=0.08, mindist=10, up_ret=0.1, down_ret=-0.1, min_seg=20)
    assert "up" in set(truth)
    assert "down" in set(truth)


def test_legacy_selection_prefers_a_directional_method_on_a_trend() -> None:
    px = prepare_features(_ohlcv(_trend_then_drop(500)))
    method, _params, meta = select_regime_method_legacy(px)
    assert method in METHODS
    assert meta.get("degenerate_truth_fallback") is not True


def test_legacy_selection_falls_back_when_truth_is_all_range() -> None:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, 300)))
    px = prepare_features(_ohlcv(close))
    method, _params, meta = select_regime_method_legacy(px)
    if meta.get("degenerate_truth_fallback"):
        assert method == DEGENERATE_FALLBACK_METHOD


def test_hold_up_buys_the_up_run_and_keeps_an_open_position() -> None:
    index = pd.bdate_range("2025-01-02", periods=6)
    frame = pd.DataFrame(
        {
            "as_of": index,
            "$close": [10, 11, 12, 13, 14, 15],
            "regime": ["range", "up", "up", "up", "up", "up"],
        }
    )
    events, stats, open_mtm = simulate_hold_up(frame)
    assert [event["side"] for event in events] == ["BUY"]
    assert stats["open_pos"] is True
    assert open_mtm is not None
    assert stats["n"] == 0


def test_hold_up_sells_when_the_up_regime_ends() -> None:
    index = pd.bdate_range("2025-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "as_of": index,
            "$close": [10.0, 12.0, 11.0, 11.0],
            "regime": ["up", "up", "down", "down"],
        }
    )
    events, stats, open_mtm = simulate_hold_up(frame)
    assert [event["side"] for event in events] == ["BUY", "SELL"]
    assert events[1]["why"] == "离上涨态"
    assert open_mtm is None
    assert stats["n"] == 1


def test_run_adaptive_writes_summary_and_html(tmp_path: Path) -> None:
    frames = {}
    names = {}
    for i, slope in enumerate((0.0015, -0.001, 0.0002)):
        code = f"80101{i}"
        rng = np.random.default_rng(i)
        close = 100.0 * np.exp(np.cumsum(np.full(360, slope) + rng.normal(0, 0.004, 360)))
        frames[code] = _ohlcv(close, start="2024-01-02")
        names[code] = code
    summary = run_adaptive(
        frames,
        names,
        as_of="2025-06-30",
        start_date="2025-01-02",
        train_cutoff="2025-01-02",
        out_dir=tmp_path,
        jobs=1,
    )
    assert not summary.empty
    assert {"code", "method", "edge", "edge_vs_fixed"} <= set(summary.columns)
    assert (tmp_path / "batch_summary.csv").is_file()
    assert list(tmp_path.glob("*_adaptive.html"))
    assert list((tmp_path / "configs").glob("*.json"))


def _three_frames() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames = {}
    names = {}
    for i, slope in enumerate((0.0015, -0.001, 0.0002)):
        code = f"80101{i}"
        rng = np.random.default_rng(i)
        close = 100.0 * np.exp(np.cumsum(np.full(360, slope) + rng.normal(0, 0.004, 360)))
        frames[code] = _ohlcv(close, start="2024-01-02")
        names[code] = code
    return frames, names


def _methods(summary: pd.DataFrame) -> dict[str, str]:
    return dict(zip(summary["code"].astype(str), summary["method"].astype(str)))


def test_second_day_reuses_the_shared_train_cache(tmp_path: Path, monkeypatch) -> None:
    frames, names = _three_frames()
    cache = tmp_path / "_train_cache"
    kwargs = dict(
        frames=frames,
        names=names,
        start_date="2025-01-02",
        train_cutoff="2025-01-02",
        jobs=1,
        train_cache_dir=cache,
    )
    first = run_adaptive(**kwargs, as_of="2025-06-30", out_dir=tmp_path / "day1")
    assert list((cache.iterdir()))

    def _refuse_retrain(*_args, **_kwargs):
        raise AssertionError("retrained")

    monkeypatch.setattr("sw_daily.adaptive.run.select_regime_method_legacy", _refuse_retrain)
    second = run_adaptive(**kwargs, as_of="2025-07-31", out_dir=tmp_path / "day2")
    assert _methods(second) == _methods(first)


def test_same_window_does_not_redraw_html(tmp_path: Path, monkeypatch) -> None:
    frames, names = _three_frames()
    out = tmp_path / "adaptive"
    kwargs = dict(
        frames=frames,
        names=names,
        as_of="2025-06-30",
        start_date="2025-01-02",
        train_cutoff="2025-01-02",
        out_dir=out,
        jobs=1,
    )
    run_adaptive(**kwargs)

    def _refuse_html(*_args, **_kwargs):
        raise AssertionError("redrawn")

    monkeypatch.setattr("sw_daily.adaptive.etf_page.write_etf_html", _refuse_html)
    again = run_adaptive(**kwargs)
    assert set(again["code"].astype(str)) == set(frames)


def test_new_as_of_seeds_from_the_previous_day(tmp_path: Path, monkeypatch) -> None:
    frames, names = _three_frames()
    parent = tmp_path / "adaptive"
    kwargs = dict(
        frames=frames,
        names=names,
        start_date="2025-01-02",
        train_cutoff="2025-01-02",
        jobs=1,
    )
    first = run_adaptive(**kwargs, as_of="2026-09-30", out_dir=parent / "20260930all_adaptive")

    def _refuse_retrain(*_args, **_kwargs):
        raise AssertionError("retrained")

    monkeypatch.setattr("sw_daily.adaptive.run.select_regime_method_legacy", _refuse_retrain)
    second = run_adaptive(**kwargs, as_of="2026-10-08", out_dir=parent / "20261008all_adaptive")
    assert _methods(second) == _methods(first)


def test_series_ending_before_the_window_is_still_plotted(tmp_path: Path) -> None:
    close = 100.0 * np.exp(np.cumsum(np.full(180, 0.001)))
    summary = run_adaptive(
        {"801011": _ohlcv(close, start="2023-09-07")},
        {"801011": "林业Ⅱ"},
        as_of="2026-09-30",
        start_date="2026-04-01",
        train_cutoff="2026-04-01",
        out_dir=tmp_path,
        jobs=1,
    )
    assert list(summary["code"]) == ["801011"]
    assert bool(summary.iloc[0]["clipped_to_history"])
    assert summary.iloc[0]["plotted_end"] < "2026-04-01"
