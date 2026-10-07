"""Train-time method selection and hold-up execution.

Ported from the B0 path of ``etf_daily.lib.adaptive_stage_common``. Each
symbol picks one causal method on history before the cutoff. Live trading is
long-only: buy when the up regime starts, sell when it ends.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sw_daily.adaptive.methods import METHODS, build_px, retrospective_truth, run_method

PRIOR_K = 5.0
MIN_TRAIN_BARS = 120
COST_PER_SIDE = 0.001
NO_TRADE_BUY = "NO_TRADE"
NO_TRADE_EXIT = "none"
TRUTH_VERSION = "retro_full+causal_confirmed_v1"
METHOD_IMPL_VERSION = "strict_vrev_nearstack_v1"
REGIME_SELECT_LEGACY = "legacy_score_method"
TRADE_MODE_HOLD_UP = "hold_up"
DEGENERATE_TRUTH_RANGE = 0.90
DEGENERATE_FALLBACK_METHOD = "ma_stack_hyst"
HOLD_UP_METHOD_EXCLUDE = frozenset()

STRICT_P60_THR_GRID = (0.03, 0.06, 0.10)
STRICT_P20_VREV_GRID = (0.03, 0.06)
CONFIRM_GRID = (1, 3)
SMOOTH_GRID_B0 = (3, 5)
MIN_SEG_GRID_B0 = (5, 10)

REGIME_HOLD_UP_RULES: dict[str, dict[str, str]] = {
    "up": {"buy": "进入上涨态", "exit": "离上涨态"},
    "down": {"buy": NO_TRADE_BUY, "exit": NO_TRADE_EXIT},
    "range": {"buy": NO_TRADE_BUY, "exit": NO_TRADE_EXIT},
}
FIXED_HYBRID_METHOD = "hybrid_ma_adx"


def prepare_features(ohlcv: pd.DataFrame, *, method: str | None = None, method_params: dict | None = None) -> pd.DataFrame:
    px = build_px(ohlcv)
    px["as_of"] = pd.to_datetime(px["as_of"]).dt.normalize()
    close = px["$close"].astype(float)
    high = px["$high"].astype(float)
    low = px["$low"].astype(float)
    open_ = px["$open"].astype(float)
    px["MA5"] = close.rolling(5, min_periods=5).mean()
    px["is_green"] = close > open_
    px["is_red"] = close < open_
    px["ma_bull"] = px["MA20"] > px["MA60"]
    px["above_ma10"] = close > px["MA10"]
    px["above_ma20"] = close > px["MA20"]
    px["golden"] = (px["MA5"] > px["MA20"]) & (px["MA5"].shift(1) <= px["MA20"].shift(1))
    px["death"] = (px["MA5"] < px["MA20"]) & (px["MA5"].shift(1) >= px["MA20"].shift(1))
    px["ret1"] = close.pct_change(fill_method=None)
    px["close_eq_low"] = np.isclose(close, low, rtol=0, atol=1e-4)
    roll_lo = low.rolling(20, min_periods=5).min()
    roll_hi = high.rolling(20, min_periods=5).max()
    px["range_pos"] = (close - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
    if method is not None:
        px["regime"] = label_regime(px, method, method_params)
    return px


def label_regime(px: pd.DataFrame, method: str, method_params: dict | None = None, *, sticky_guard: bool = False) -> np.ndarray:
    if method not in METHODS:
        raise KeyError(f"unknown method {method}")
    labs = run_method(method, px, params=dict(method_params or {}))
    if sticky_guard:
        labs = apply_sticky_up_guard(labs, px)
    return np.asarray(labs, dtype=object)


def apply_sticky_up_guard(
    labs: np.ndarray, px: pd.DataFrame, *, below20_days: int = 3, drawdown: float = 0.18, reenter_guard: int = 2
) -> np.ndarray:
    out = np.asarray(labs, dtype=object).copy()
    close = px["$close"].to_numpy(float)
    ma20 = px["MA20"].to_numpy(float)
    below = 0
    peak = np.nan
    guard = 0
    for i in range(len(out)):
        c, m = close[i], ma20[i]
        above = np.isfinite(c) and np.isfinite(m) and c >= m
        if guard > 0:
            if out[i] == "up":
                out[i] = "range"
                if above:
                    guard -= 1
                below = 0
                peak = np.nan
                continue
            guard = 0
        if out[i] != "up":
            below = 0
            peak = np.nan
            continue
        if np.isfinite(c):
            peak = c if (not np.isfinite(peak) or c > peak) else peak
        if np.isfinite(c) and np.isfinite(m) and c < m:
            below += 1
        else:
            below = 0
        dd = (c / peak - 1.0) if (np.isfinite(c) and np.isfinite(peak) and peak > 0) else 0.0
        if below >= below20_days or (below >= 1 and dd <= -drawdown):
            out[i] = "range"
            below = 0
            peak = np.nan
            guard = max(int(reenter_guard), 0)
    return out


def n_switches(labs: np.ndarray) -> int:
    return int(sum(1 for i in range(1, len(labs)) if labs[i] != labs[i - 1]))


def hold_up_compound(labs: np.ndarray, close: np.ndarray) -> tuple[float, int]:
    equity = 1.0
    pos = False
    entry = None
    n = 0
    for i in range(len(labs)):
        if (not pos) and labs[i] == "up":
            pos = True
            entry = i
        elif pos and labs[i] != "up":
            equity *= close[i] / close[entry]
            n += 1
            pos = False
    if pos:
        equity *= close[-1] / close[entry]
        n += 1
    return float(equity - 1), n


def score_method(labs: np.ndarray, close: np.ndarray, truth: np.ndarray) -> float:
    hold_up, _ = hold_up_compound(labs, close)
    agree = float((labs == truth).mean()) if len(labs) else 0.0
    up_frac = float((labs == "up").mean()) if len(labs) else 0.0
    return 0.45 * hold_up + 0.25 * agree + 0.15 * up_frac - 0.015 * n_switches(labs)


def enter_up_param_grid(method: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if method == "ma_stack_strict":
        for p60 in STRICT_P60_THR_GRID:
            for p20 in STRICT_P20_VREV_GRID:
                for conf in CONFIRM_GRID:
                    for ms in MIN_SEG_GRID_B0:
                        out.append({"p60_thr": float(p60), "p20_vrev": float(p20), "confirm": int(conf), "min_seg": int(ms)})
    elif method in ("ma_stack_hyst", "hybrid_ma_adx"):
        for conf in CONFIRM_GRID:
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append({"confirm": int(conf), "smooth": int(sm), "min_seg": int(ms)})
    elif method in ("ma_stack_struct", "dual_ma_cross"):
        for conf in CONFIRM_GRID:
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append({
                        "confirm": int(conf),
                        "confirm_enter": int(conf),
                        "confirm_leave": int(conf),
                        "smooth": int(sm),
                        "min_seg": int(ms),
                    })
    elif method == "adx_di":
        for thr in (18.0, 22.0):
            for sm in SMOOTH_GRID_B0:
                for ms in MIN_SEG_GRID_B0:
                    out.append({"adx_thr": float(thr), "smooth": int(sm), "min_seg": int(ms)})
    elif method == "prior60_band":
        for ms in MIN_SEG_GRID_B0:
            out.append({"min_seg": int(ms)})
    else:
        out.append({})
    if {} not in out:
        out.insert(0, {})
    return out


def select_regime_method_legacy(px_train: pd.DataFrame) -> tuple[str, dict, dict]:
    """Pick the method and params with the best train score. Degenerate truth falls back."""
    close = px_train["$close"].to_numpy(float)
    truth = retrospective_truth(close)
    truth_range = float((truth == "range").mean()) if len(truth) else 0.0
    truth_dir = float(((truth == "up") | (truth == "down")).mean()) if len(truth) else 0.0
    scores: dict[str, float] = {}
    for name in METHODS:
        if name in HOLD_UP_METHOD_EXCLUDE:
            continue
        scores[name] = float(score_method(run_method(name, px_train), close, truth))

    if truth_range >= DEGENERATE_TRUTH_RANGE or truth_dir < 0.05:
        fallback = DEGENERATE_FALLBACK_METHOD
        return fallback, {}, {
            "scores": scores,
            "score": float(scores.get(fallback, 0.0)),
            "degenerate_truth_fallback": True,
            "truth_range": truth_range,
            "truth_dir": truth_dir,
        }

    best_m: str | None = None
    best_p: dict[str, Any] = {}
    best_s = -1e18
    vetoed: list[str] = []
    for name in METHODS:
        if name in HOLD_UP_METHOD_EXCLUDE:
            continue
        for params in enter_up_param_grid(name):
            labs = run_method(name, px_train, params=params)
            score = float(score_method(labs, close, truth))
            range_frac = float((labs == "range").mean()) if len(labs) else 1.0
            dir_frac = float(((labs == "up") | (labs == "down")).mean()) if len(labs) else 0.0
            collapsed = range_frac >= 0.85 or (range_frac >= 0.75 and range_frac > truth_range + 0.35 and truth_range >= 0.30)
            starved = truth_dir >= 0.20 and dir_frac < truth_dir * 0.25 and range_frac >= 0.65
            if collapsed or starved:
                if name not in vetoed:
                    vetoed.append(name)
                continue
            if score > best_s:
                best_s = score
                best_m = name
                best_p = dict(params)
            scores[name] = max(scores.get(name, -1e18), score)

    if best_m is None:
        best_m = max(scores, key=scores.get) if scores else "ma_stack_strict"
        best_s = float(scores.get(best_m, -1e18))
        best_p = {}
    return best_m, dict(best_p), {
        "scores": scores,
        "score": best_s,
        "vetoed_range_collapse": vetoed,
        "truth_range": truth_range,
        "truth_dir": truth_dir,
        "method_params": dict(best_p),
    }


def segments(regime: np.ndarray, dates: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    labs = list(regime)
    rows: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[start]:
            rows.append({
                "regime": labs[start],
                "start": str(dates[start]),
                "end": str(dates[i - 1]),
                "n_bars": i - start,
                "ret": float(close[i - 1] / close[start] - 1),
            })
            start = i
    return pd.DataFrame(rows)


def simulate_hold_up(df: pd.DataFrame) -> tuple[list[dict], dict, dict | None]:
    """Buy on the first up bar of a run, sell on the first bar that leaves up."""
    close = df["$close"].to_numpy(float)
    dates = df["as_of"].dt.strftime("%Y-%m-%d").to_numpy()
    regime = df["regime"].to_numpy()
    bh = float(close[-1] / close[0] - 1) if len(close) > 1 else 0.0
    cash = 1.0
    pos = 0.0
    entry = None
    entry_i = None
    events: list[dict] = []
    prev = None
    for i in range(len(df)):
        rg = str(regime[i])
        c = close[i]
        if pos == 0:
            if rg == "up" and prev != "up":
                entry = c
                entry_i = i
                pos = cash / c
                cash = 0.0
                events.append({"side": "BUY", "date": dates[i], "px": c, "why": "进入上涨态", "regime": rg})
            prev = rg
            continue
        if rg != "up":
            cash = pos * c
            events.append({
                "side": "SELL",
                "date": dates[i],
                "px": c,
                "why": "离上涨态",
                "entry_date": dates[entry_i],
                "fwd_ret": c / entry - 1,
                "regime": "up",
            })
            pos = 0.0
            entry = entry_i = None
        prev = rg

    open_mtm = None
    if pos > 0:
        open_mtm = {
            "entry_date": dates[entry_i],
            "entry_px": float(entry),
            "last_date": dates[-1],
            "last_px": float(close[-1]),
            "unrealized": float(close[-1] / entry - 1),
        }
        equity = cash + pos * close[-1]
    else:
        equity = cash
    sells = [event for event in events if event["side"] == "SELL"]
    # Round-trip cost: buy and sell each pay COST_PER_SIDE, including an open position.
    n_sides = len(sells) * 2 + (1 if open_mtm else 0)
    equity *= (1.0 - COST_PER_SIDE) ** n_sides
    stats = {
        "compound": float(equity - 1),
        "bh": bh,
        "edge": float(equity - 1) - bh,
        "n": len(sells),
        "win": float(np.mean([event["fwd_ret"] > 0 for event in sells])) if sells else float("nan"),
        "open_pos": bool(open_mtm),
        "open_unrealized": open_mtm["unrealized"] if open_mtm else None,
    }
    return events, stats, open_mtm
