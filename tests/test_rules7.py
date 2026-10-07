"""Tests for the rules7 state machine (synthetic series, no qlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sw_daily.rules7.features import build_feature_frame_from_series, prepare_fig9_touch_frame
from sw_daily.rules7.rules7 import (
    TYPE_CYCLE,
    TYPE_HIVOL,
    TYPE_SHORT,
    TYPE_SMOOTH,
    TYPE_TREND,
    TYPE_PARAMS,
    RuleSpec,
    buy_sell_masks,
    classify,
    flat_buy_quote,
    indicator_snapshot,
    run_state_machine,
    triggered,
)
from sw_daily.rules7.scan import _md_table, attach_cluster_recommendations


def _series(values, dtype=float) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=dtype), index=pd.bdate_range("2024-01-01", periods=len(values)))


def _mask(n: int, on: list[int]) -> pd.Series:
    a = np.zeros(n, bool)
    a[on] = True
    return _series(a, bool)


def test_trail_peak_starts_at_fill_bar():
    close = _series([10.0, 8.0, 7.6, 9.0, 8.0, 8.0])
    res = run_state_machine(close, _mask(6, [0]), _mask(6, []), trail=0.10)
    assert len(res["trades"]) == 1
    trade = res["trades"][0]
    assert trade["sell"] == close.index[4] and trade["why"] == "回落"
    assert np.isclose(trade["ret"], 8.0 / 8.0 - 1)


def test_g1_suppresses_rail_sell():
    close = _series(np.linspace(10, 12, 8))
    buy, sell = _mask(8, [0]), _mask(8, [5])
    plain = run_state_machine(close, buy, sell)
    assert plain["trades"][0]["sell"] == close.index[5]
    held = run_state_machine(close, buy, sell, g1=_mask(8, [3]))
    assert held["trades"] == []
    assert held["state"]["holding"] and held["state"]["g1_seen"]
    assert held["state"]["action"] == "持有"


def test_buy_on_last_bar_is_today_action():
    close = _series([10.0, 10.1, 10.2, 10.3])
    res = run_state_machine(close, _mask(4, [3]), _mask(4, []), trail=0.15)
    st = res["state"]
    assert st["action"] == "买"
    assert not st["holding"]
    assert st["fill_px"] is None
    assert res["exposure"] == 0.0


def test_fast_pattern_blocks_buy():
    n = 5
    masks = {
        "ae_buy": _mask(n, [1, 3]),
        "b1": _mask(n, [4]),
        "fast": _mask(n, [1, 4]),
        "ae_sell": _mask(n, []),
        "fig8_sell": _mask(n, []),
    }
    blocked, _ = buy_sell_masks(masks, TYPE_PARAMS[TYPE_HIVOL])
    assert list(np.flatnonzero(blocked.to_numpy())) == [3]
    open_buy, _ = buy_sell_masks(masks, TYPE_PARAMS[TYPE_TREND])
    assert list(np.flatnonzero(open_buy.to_numpy())) == [1, 3, 4]


def test_classify_order():
    base = dict(bars=1000, vol=0.20, bh_cagr=0.12, bh_mdd=-0.25, cross_per_year=8)
    assert classify({**base, "bars": 100}) == TYPE_SHORT
    assert classify({**base, "vol": 0.35, "bh_mdd": -0.47}) == TYPE_HIVOL
    assert classify({**base, "bh_mdd": -0.48}) == TYPE_CYCLE
    assert classify({**base, "bh_cagr": 0.02}) == TYPE_CYCLE
    assert classify({**base, "vol": 0.16, "cross_per_year": 10}) == TYPE_SMOOTH
    assert classify(base) == TYPE_TREND


def _snap_frame(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    px = np.linspace(10.0, 12.6, n)
    px[-1] = 9.9
    frame = pd.DataFrame(index=idx)
    frame["px"] = px
    frame["fig9_lower"], frame["fig9_upper"], frame["fig8_upper"] = 10.0, 13.0, 14.0
    frame["ma5"], frame["ma20"] = 10.5, 11.0
    frame["gap_5d"], frame["gap_20d"] = -0.01, -0.02
    frame["fig7_gap"], frame["fig8_gap"] = -0.10, -0.05
    frame["fig9_pos"], frame["fig10_pos"] = 0.5, 0.5
    for k in (7, 8, 9, 10):
        frame[f"fig{k}_g"] = 0.1
    frame["vol5_pct_120d"] = 0.5
    for col in ("fig9_touch_hi", "fig10_touch_hi", "locmax"):
        frame[col] = False
    frame["aux_buy"] = (frame["px"] < frame["ma20"]) & (frame["ma5"] < frame["ma20"])
    frame["aux_sell"] = False
    frame["fig9_mid_stretch_sell"] = False
    return frame


def test_indicator_snapshot_today_flags():
    frame = _snap_frame()
    n = len(frame)
    near_lo = frame["px"] <= frame["fig9_lower"] * 1.01
    edge = near_lo & ~near_lo.shift(fill_value=False)
    masks = {
        "ae_buy": edge & frame["aux_buy"],
        "ae_sell": _mask(n, [n - 1]),
        "b1": near_lo & (frame["fig7_gap"] <= -0.08),
        "fast": _mask(n, []),
        "fig8_sell": _mask(n, []),
        "g1": _mask(n, []),
    }
    spec = RuleSpec(b1=True, fast_block=True, sell="fig9", g1_hold=True, trail=0.15)
    state = {"trail_hit": False, "sell_masked_by_g1": True, "stop": 9.0}
    snap = indicator_snapshot(frame, masks, spec, state)
    assert snap["near_lo"] and snap["near_lo_edge"] and not snap["near_lo_prev"]
    assert snap["aux_buy"] and snap["ae_buy"] and snap["b1"] and snap["disc_ok"]
    assert snap["buy_signal"] and not snap["fast"]
    assert not snap["near_hi"] and not snap["aux_sell"]
    assert np.isclose(snap["dist_lo9"], 9.9 / 10.0 - 1)
    assert np.isclose(snap["dist_stop"], 9.9 / 9.0 - 1)
    labels = set(triggered(snap))
    assert {"近下轨上跳沿", "辅助买像", "aux_edge买点", "B1", "买入信号", "卖点被G1屏蔽"} <= labels
    assert "快形态" not in labels


def _gold_snap(**over) -> dict:
    snap = {
        "lo9": 7.930,
        "px": 8.633,
        "near_lo": False,
        "aux_buy": True,
        "fast": False,
        "disc_ok": False,
        "buy_signal": False,
    }
    snap.update(over)
    return snap


def test_flat_buy_quote_still_short_of_the_rail():
    price, status = flat_buy_quote(_gold_snap(), TYPE_PARAMS[TYPE_HIVOL])
    assert np.isclose(price, 7.930 * 1.01)
    assert status == "还差这个价"


def test_flat_buy_quote_already_inside_band():
    _, status = flat_buy_quote(_gold_snap(near_lo=True), TYPE_PARAMS[TYPE_HIVOL])
    assert status == "已在带内，明天同价不算"


def test_flat_buy_quote_fast_blocks_before_buy_signal():
    _, status = flat_buy_quote(_gold_snap(fast=True, buy_signal=True, near_lo=True), TYPE_PARAMS[TYPE_HIVOL])
    assert status == "快形态挡住"


def test_flat_buy_quote_b1_needs_only_the_rail():
    _, status = flat_buy_quote(_gold_snap(disc_ok=True, aux_buy=False), TYPE_PARAMS[TYPE_TREND])
    assert status == "贴轨即可"


def test_flat_buy_quote_aux_buy_missing():
    _, status = flat_buy_quote(_gold_snap(aux_buy=False), TYPE_PARAMS[TYPE_TREND])
    assert status == "辅助买像未齐"


def test_flat_buy_quote_buy_signal_today():
    _, status = flat_buy_quote(_gold_snap(buy_signal=True, near_lo=True), TYPE_PARAMS[TYPE_TREND])
    assert status == "今天已是买点"


def test_flat_buy_quote_not_applicable():
    price, status = flat_buy_quote(_gold_snap(buy_signal=True), TYPE_PARAMS[TYPE_TREND], applicable=False)
    assert np.isclose(price, 7.930 * 1.01)
    assert status == "规则不适用"


def test_flat_buy_quote_missing_lower_rail():
    price, status = flat_buy_quote(_gold_snap(lo9=float("nan")), TYPE_PARAMS[TYPE_TREND])
    assert price is None
    assert status == "无下轨"


def test_md_table_missing_column_does_not_raise():
    frame = pd.DataFrame([{"代码": "801010", "名称": "农林牧渔"}])
    lines = _md_table(frame, ["代码", "快形态"])
    assert "801010" in lines[2]
    assert "—" in lines[2]


def test_feature_frame_has_gap_and_fig9_rails():
    idx = pd.bdate_range("2020-01-02", periods=400)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, len(idx)))), index=idx)
    anchor = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx)))), index=idx)
    ohlcv = pd.DataFrame({"$high": close * 1.01, "$low": close * 0.99, "$close": close}, index=idx)
    frame = prepare_fig9_touch_frame(build_feature_frame_from_series(close, ohlcv, anchor, lookback=10))
    assert {"gap_5d", "gap_20d", "fig9_lower", "fig9_upper", "aux_buy", "fig9_pos"} <= set(frame.columns)
    assert frame["fig9_lower"].notna().sum() > 100


def test_one_recommended_buy_per_cluster():
    checklist = pd.DataFrame([
        {"代码": "801010", "名称": "甲", "今日动作": "买", "_dist_lo9": -0.01},
        {"代码": "801020", "名称": "乙", "今日动作": "买", "_dist_lo9": -0.02},
        {"代码": "801030", "名称": "丙", "今日动作": "观望", "_dist_lo9": 0.0},
    ])
    cluster = pd.DataFrame([
        {"code": "801010", "cluster": 1, "is_representative": True, "ret20_pct": 3.0, "n_members": 2, "group": "up"},
        {"code": "801020", "cluster": 1, "is_representative": False, "ret20_pct": 9.0, "n_members": 2, "group": "up"},
        {"code": "801030", "cluster": 1, "is_representative": False, "ret20_pct": 1.0, "n_members": 2, "group": "up"},
    ])
    out = attach_cluster_recommendations(checklist, cluster)
    recommended = out.loc[out["推荐买入"] == "是", "代码"].tolist()
    assert recommended == ["801010"]
    assert out.loc[out["代码"] == "801030", "推荐买入"].iloc[0] == ""
