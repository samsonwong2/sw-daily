"""Causal up/down/range methods and look-ahead swing truth.

Ported from ``etf_daily.scripts.eval_causal_regime_switch_pool``. The seven
methods use only the current bar and history. ``retrospective_truth`` looks
ahead and is used only to score a method on the training window.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

VREV_WASHOUT_STACK_GAP = -0.05


def _rolling_hist_percentile(series: pd.Series, *, window: int = 252, min_periods: int = 60) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        window_vals = arr[lo : i + 1]
        window_vals = window_vals[np.isfinite(window_vals)]
        if window_vals.size < min_periods or not np.isfinite(arr[i]):
            continue
        out[i] = float(np.mean(window_vals <= arr[i]))
    return pd.Series(out, index=series.index)


def build_px(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature frame. Expects ``$open/$high/$low/$close`` and ``datetime``."""
    df = ohlcv.copy()
    df["as_of"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.sort_values("as_of")
    df = df[~df["as_of"].duplicated(keep="last")].reset_index(drop=True)
    for column in ("$open", "$high", "$low", "$close"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    close = df["$close"]
    for window in (5, 10, 20, 60):
        df[f"MA{window}"] = close.rolling(window, min_periods=max(3, window // 2)).mean()
    df["prior20"] = close / close.shift(20) - 1.0
    df["prior60"] = close / close.shift(60) - 1.0
    df["dist_ma20"] = close / df["MA20"] - 1.0
    df["ma20_ma60"] = df["MA20"] / df["MA60"] - 1.0
    df["roll_hi60"] = df["$high"].rolling(60, min_periods=20).max()
    df["roll_lo60"] = df["$low"].rolling(60, min_periods=20).min()
    df["band60"] = (df["roll_hi60"] - df["roll_lo60"]) / close
    true_range = pd.concat(
        [
            df["$high"] - df["$low"],
            (df["$high"] - close.shift(1)).abs(),
            (df["$low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr20"] = true_range.rolling(20, min_periods=10).mean() / close
    df["rv20"] = close.pct_change(fill_method=None).rolling(20, min_periods=10).std() * np.sqrt(252.0)
    df["rv20_pct_252"] = _rolling_hist_percentile(df["rv20"], window=252, min_periods=60)
    up_move = df["$high"].diff()
    down_move = -df["$low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = true_range.rolling(14, min_periods=7).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14, min_periods=7).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14, min_periods=7).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    df["adx14"] = dx.rolling(14, min_periods=7).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def retrospective_truth(
    close: np.ndarray,
    *,
    prom: float = 0.18,
    mindist: int = 25,
    up_ret: float = 0.22,
    down_ret: float = -0.22,
    min_seg: int = 30,
) -> np.ndarray:
    """Look-ahead major swing labels. Training score only."""
    n = len(close)
    if n < 60:
        return np.array(["range"] * n, dtype=object)
    smooth = pd.Series(close).rolling(8, min_periods=1).mean().to_numpy()
    pivots: list[tuple[int, str, float]] = []
    last_i, last_px, last_t = 0, float(smooth[0]), None
    for i in range(1, n - 1):
        is_high = smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]
        is_low = smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]
        if not (is_high or is_low):
            continue
        kind = "H" if is_high else "L"
        if last_t is None:
            last_t, last_i, last_px = kind, i, float(smooth[i])
            continue
        if kind == last_t:
            if (kind == "H" and smooth[i] >= last_px) or (kind == "L" and smooth[i] <= last_px):
                last_i, last_px = i, float(smooth[i])
            continue
        move = abs(smooth[i] - last_px) / max(last_px, 1e-8)
        if move >= prom and (i - last_i) >= mindist:
            pivots.append((last_i, last_t, float(close[last_i])))
            last_t, last_i, last_px = kind, i, float(smooth[i])
    pivots.append((last_i, last_t or "L", float(close[last_i])))
    if pivots[0][0] > 0:
        pivots.insert(0, (0, "L" if pivots[0][1] == "H" else "H", float(close[0])))
    if pivots[-1][0] < n - 1:
        pivots.append((n - 1, "H" if pivots[-1][1] == "L" else "L", float(close[-1])))

    legs: list[list[Any]] = []
    for left, right in zip(pivots[:-1], pivots[1:]):
        i0, t0, p0 = left
        i1, t1, p1 = right
        if i1 <= i0:
            continue
        ret = p1 / p0 - 1.0
        if t0 == "L" and t1 == "H" and ret >= up_ret:
            lab = "up"
        elif t0 == "H" and t1 == "L" and ret <= down_ret:
            lab = "down"
        else:
            lab = "range"
        legs.append([i0, i1, lab])

    merged: list[list[Any]] = []
    for leg in legs:
        if merged and merged[-1][2] == leg[2]:
            merged[-1][1] = leg[1]
        else:
            merged.append(list(leg))
    out_legs: list[list[Any]] = []
    for i0, i1, lab in merged:
        if out_legs and (i1 - i0 + 1) < min_seg:
            out_legs[-1][1] = i1
        else:
            out_legs.append([i0, i1, lab])
    merged2: list[list[Any]] = []
    for leg in out_legs:
        if merged2 and merged2[-1][2] == leg[2]:
            merged2[-1][1] = leg[1]
        else:
            merged2.append(list(leg))
    truth = np.array(["range"] * n, dtype=object)
    for i0, i1, lab in merged2:
        truth[i0 : i1 + 1] = lab
    return truth


def retrospective_truth_full_slice(
    close_full: np.ndarray,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    truth_full = retrospective_truth(close_full, **kwargs)
    if end_idx is None:
        end_idx = len(truth_full) - 1
    start_idx = max(0, int(start_idx))
    end_idx = min(len(truth_full) - 1, int(end_idx))
    if end_idx < start_idx:
        return np.array([], dtype=object)
    return truth_full[start_idx : end_idx + 1].copy()


def segment_bounds(labels: np.ndarray) -> list[tuple[int, int, str]]:
    segs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append((start, i - 1, str(labels[start])))
            start = i
    return segs


def merge_short_segments(labels: np.ndarray, *, min_bars: int = 10) -> np.ndarray:
    labs = np.asarray(labels, dtype=object).copy()
    n = len(labs)
    if n == 0 or min_bars <= 1:
        return labs
    for _ in range(2):
        segs = segment_bounds(labs)
        out = labs.copy()
        changed = False
        for start, end, lab in segs:
            length = end - start + 1
            if lab == "range" or length >= min_bars:
                continue
            if start > 0:
                fill = out[start - 1]
            elif end + 1 < n:
                fill = out[end + 1]
            else:
                fill = "range"
            out[start : end + 1] = fill
            changed = True
        labs = out
        if not changed:
            break
    return np.asarray(labs, dtype=object)


def apply_lowvol_range_bias(
    labels: np.ndarray,
    vol_pct: np.ndarray,
    prior60: np.ndarray,
    prior20: np.ndarray,
    *,
    low_pct: float = 0.35,
    look: int = 15,
    persist: int = 10,
    escape_p60: float = 0.12,
    escape_p20: float = 0.08,
) -> np.ndarray:
    labs = np.asarray(labels, dtype=object).copy()
    n = len(labs)
    if n == 0:
        return labs
    vol = np.asarray(vol_pct, dtype=float)
    p60 = np.asarray(prior60, dtype=float)
    p20 = np.asarray(prior20, dtype=float)
    for i in range(n):
        window = vol[max(0, i - look + 1) : i + 1]
        n_low = int(np.sum(np.isfinite(window) & (window < low_pct)))
        if n_low < persist:
            continue
        a60 = abs(p60[i]) if np.isfinite(p60[i]) else 0.0
        a20 = abs(p20[i]) if np.isfinite(p20[i]) else 0.0
        if a60 >= escape_p60 or a20 >= escape_p20:
            continue
        labs[i] = "range"
    return labs


def finalize_regime_labels(labels: np.ndarray, px: pd.DataFrame, *, min_seg: int = 10) -> np.ndarray:
    labs = np.asarray(labels, dtype=object)
    if "rv20_pct_252" in px.columns:
        labs = apply_lowvol_range_bias(
            labs,
            px["rv20_pct_252"].to_numpy(float),
            px["prior60"].to_numpy(float),
            px["prior20"].to_numpy(float),
        )
    if int(min_seg or 0) > 1:
        labs = merge_short_segments(labs, min_bars=int(min_seg))
    return np.asarray(labs, dtype=object)


def causal_smooth(labs: list[str], window: int = 5) -> list[str]:
    out = []
    for i in range(len(labs)):
        sl = labs[max(0, i - window + 1) : i + 1]
        vals, counts = np.unique(sl, return_counts=True)
        out.append(str(vals[np.argmax(counts)]))
    return out


def apply_hysteresis(
    raw: list[str],
    *,
    above20: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    close: np.ndarray | None = None,
    confirm: int = 2,
    soft_leave_above20: int = 2,
    soft_leave_bounce: float = 0.04,
    soft_leave_below20: int = 2,
    soft_leave_drawdown: float = 0.06,
    reenter_down_guard: int = 3,
) -> list[str]:
    state = "range"
    below = above = 0
    trough = np.nan
    peak = np.nan
    soft_guard = 0
    if close is None:
        close = np.full(len(raw), np.nan)
    out = []
    for i, lab in enumerate(raw):
        if above20[i]:
            above += 1
            below = 0
        else:
            below += 1
            above = 0
        m20, m60 = ma20[i], ma60[i]
        c = float(close[i]) if np.isfinite(close[i]) else np.nan
        bull_stack = np.isfinite(m20) and np.isfinite(m60) and m20 > m60
        bear_stack = np.isfinite(m20) and np.isfinite(m60) and m20 < m60
        if state == "up":
            if np.isfinite(c):
                peak = c if (not np.isfinite(peak) or c > peak) else peak
            dd = (c / peak - 1.0) if (np.isfinite(c) and np.isfinite(peak) and peak > 0) else 0.0
            soft_up = (soft_leave_below20 > 0 and below >= soft_leave_below20) or (
                soft_leave_drawdown > 0 and dd <= -soft_leave_drawdown
            )
            if lab == "down" and (below >= confirm or bear_stack):
                state = "down"
                trough = c if np.isfinite(c) else np.nan
                peak = np.nan
                soft_guard = 0
            elif soft_up or (lab == "range" and below >= confirm + 1):
                state = "range"
                trough = np.nan
                peak = np.nan
        elif state == "down":
            if np.isfinite(c):
                trough = c if (not np.isfinite(trough) or c < trough) else trough
            bounce = (c / trough - 1.0) if (np.isfinite(c) and np.isfinite(trough) and trough > 0) else 0.0
            soft_leave = (soft_leave_above20 > 0 and above >= soft_leave_above20) or (
                soft_leave_bounce > 0 and bounce >= soft_leave_bounce
            )
            if lab == "up" and (above >= confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif soft_leave or (lab == "range" and above >= confirm):
                state = "range"
                soft_guard = max(reenter_down_guard, 0)
                peak = np.nan
        else:
            if soft_guard > 0:
                soft_guard -= 1
            if lab == "up" and (above >= confirm or bull_stack):
                state = "up"
                trough = np.nan
                peak = c if np.isfinite(c) else np.nan
                soft_guard = 0
            elif lab == "down" and (below >= confirm or bear_stack):
                higher_low = np.isfinite(trough) and np.isfinite(c) and c > trough
                if higher_low:
                    pass
                elif soft_guard > 0 and below < max(confirm, reenter_down_guard):
                    pass
                else:
                    state = "down"
                    trough = c if np.isfinite(c) else np.nan
                    peak = np.nan
                    soft_guard = 0
        out.append(state)
    return out


def apply_hysteresis_ma60(
    raw: list[str],
    *,
    close: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    confirm_leave: int = 3,
    confirm_enter: int = 2,
    soft_leave_below20: int = 3,
    soft_leave_drawdown: float = 0.12,
    soft_leave_bounce: float = 0.04,
) -> list[str]:
    state = "range"
    bull_n = bear_n = 0
    below20_n = above20_n = 0
    peak = np.nan
    trough = np.nan
    out: list[str] = []
    for i, lab in enumerate(raw):
        c, m20, m60 = close[i], ma20[i], ma60[i]
        ok = np.isfinite(m20) and np.isfinite(m60)
        if ok and m20 > m60:
            bull_n += 1
            bear_n = 0
        elif ok and m20 < m60:
            bear_n += 1
            bull_n = 0
        else:
            bull_n = bear_n = 0
        above20 = np.isfinite(c) and np.isfinite(m20) and c > m20
        if above20:
            above20_n += 1
            below20_n = 0
        else:
            below20_n += 1
            above20_n = 0
        if state == "up":
            if np.isfinite(c):
                peak = c if (not np.isfinite(peak) or c > peak) else peak
            dd = (c / peak - 1.0) if (np.isfinite(c) and np.isfinite(peak) and peak > 0) else 0.0
            soft_leave = (soft_leave_below20 > 0 and below20_n >= soft_leave_below20) or (
                soft_leave_drawdown > 0 and dd <= -soft_leave_drawdown
            )
            if bear_n >= confirm_leave:
                state = "down"
                peak = np.nan
                trough = c if np.isfinite(c) else np.nan
            elif soft_leave or (lab == "range" and bear_n >= confirm_enter):
                state = "range"
                peak = np.nan
                trough = np.nan
        elif state == "down":
            if np.isfinite(c):
                trough = c if (not np.isfinite(trough) or c < trough) else trough
            bounce = (c / trough - 1.0) if (np.isfinite(c) and np.isfinite(trough) and trough > 0) else 0.0
            soft_leave_down = (soft_leave_below20 > 0 and above20_n >= soft_leave_below20) or (
                soft_leave_bounce > 0 and bounce >= soft_leave_bounce
            )
            if bull_n >= confirm_leave and above20_n >= confirm_enter:
                state = "up"
                peak = c if np.isfinite(c) else np.nan
                trough = np.nan
            elif soft_leave_down or (lab == "range" and bull_n >= confirm_enter):
                state = "range"
                peak = np.nan
                trough = np.nan
        else:
            if lab == "up" and bull_n >= confirm_enter and above20_n >= confirm_enter:
                state = "up"
                peak = c if np.isfinite(c) else np.nan
                trough = np.nan
            elif lab == "down" and bear_n >= confirm_enter:
                state = "down"
                peak = np.nan
                trough = c if np.isfinite(c) else np.nan
        out.append(state)
    return out


def _ma_stack_raw_labels(px: pd.DataFrame, *, soft_hold60: bool) -> list[str]:
    raw: list[str] = []
    for _, row in px.iterrows():
        if pd.isna(row["MA20"]) or pd.isna(row["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(row["$close"]), float(row["MA20"]), float(row["MA60"])
        p20 = float(row["prior20"]) if pd.notna(row["prior20"]) else 0.0
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        band = float(row["band60"]) if pd.notna(row["band60"]) else np.nan
        hug = abs(m20 - m60) / c < 0.02
        chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.10) or (hug and abs(p60) < 0.12)
        if soft_hold60:
            up = m20 > m60 and c > m20 and c >= m60 and (p60 >= 0.02 or p20 >= 0.02)
            down = m20 < m60 and c < m20 and c <= m60 and (p60 <= -0.02 or p20 <= -0.02)
        else:
            up = (m20 > m60 and c > m20 and p60 >= 0.03) or (m20 > m60 and c > m20 and p20 >= 0.03)
            if c > m20 and p20 >= 0.06 and (m20 > m60 or p60 <= -0.08 or p60 >= 0.08):
                up = True
            down = (m20 < m60 and c < m20 and p60 <= -0.03) or (m20 < m60 and c < m20 and p20 <= -0.03)
        if up and not down:
            lab = "up"
        elif down and not up:
            lab = "down"
        elif chop:
            lab = "range"
        else:
            lab = "range"
        raw.append(lab)
    return raw


def method_ma_stack_params(px: pd.DataFrame, *, confirm: int = 3, smooth: int = 3) -> np.ndarray:
    raw = _ma_stack_raw_labels(px, soft_hold60=False)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=int(confirm),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack(px: pd.DataFrame) -> np.ndarray:
    return method_ma_stack_params(px)


def method_ma_stack_struct_params(
    px: pd.DataFrame, *, confirm_enter: int = 3, confirm_leave: int = 3, smooth: int = 3
) -> np.ndarray:
    raw = _ma_stack_raw_labels(px, soft_hold60=True)
    labs = apply_hysteresis_ma60(
        raw,
        close=px["$close"].to_numpy(float),
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        confirm_leave=int(confirm_leave),
        confirm_enter=int(confirm_enter),
        soft_leave_below20=3,
        soft_leave_drawdown=0.12,
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack_struct(px: pd.DataFrame) -> np.ndarray:
    return method_ma_stack_struct_params(px)


def method_ma_stack_strict_params(
    px: pd.DataFrame,
    *,
    p60_thr: float = 0.06,
    band_chop: float = 0.26,
    p60_chop: float = 0.12,
    p20_vrev: float = 0.06,
    confirm: int = 3,
    smooth: int = 3,
) -> np.ndarray:
    raw = []
    p20_vrev = float(p20_vrev)
    close_arr = pd.to_numeric(px["$close"], errors="coerce").to_numpy(float)
    for i, (_, row) in enumerate(px.iterrows()):
        if pd.isna(row["MA20"]) or pd.isna(row["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(close_arr[i]), float(row["MA20"]), float(row["MA60"])
        p20 = float(row["prior20"]) if pd.notna(row["prior20"]) else 0.0
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        band = float(row["band60"]) if pd.notna(row["band60"]) else np.nan
        chop = pd.notna(band) and band < band_chop and abs(p60) < p60_chop
        prev_c = close_arr[i - 1] if i > 0 else np.nan
        recovering = (not np.isfinite(prev_c)) or (c >= prev_c)
        near_stack = m60 > 0 and (m20 / m60 - 1.0) > VREV_WASHOUT_STACK_GAP
        washout = p60 <= -0.08 and near_stack and recovering
        up = (m20 > m60 and c > m20 and p60 >= p60_thr and p20 >= 0.0) or (
            c > m20 and p20 >= p20_vrev and (m20 > m60 or washout or p60 >= 0.08)
        ) or (m20 > m60 and c > m20 and p60 >= max(0.18, p60_thr * 3) and p20 >= -0.08)
        down = m20 < m60 and c < m20 and p60 <= -p60_thr and p20 <= 0.0
        if up:
            lab = "up"
        elif down:
            lab = "down"
        elif chop:
            lab = "range"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=int(confirm),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_ma_stack_strict(px: pd.DataFrame) -> np.ndarray:
    return method_ma_stack_strict_params(px)


def method_dual_ma_cross_params(
    px: pd.DataFrame, *, confirm_enter: int = 3, confirm_leave: int = 3, smooth: int = 3
) -> np.ndarray:
    raw: list[str] = []
    for _, row in px.iterrows():
        if pd.isna(row["MA20"]) or pd.isna(row["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(row["$close"]), float(row["MA20"]), float(row["MA60"])
        p20 = float(row["prior20"]) if pd.notna(row["prior20"]) else 0.0
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        hug = abs(m20 - m60) / c < 0.025
        strong_up = c > m20 and (p20 >= 0.04 or p60 >= 0.06)
        strong_dn = c < m20 and (p20 <= -0.04 or p60 <= -0.06)
        if hug and not strong_up and not strong_dn:
            lab = "range"
        elif (m20 > m60 and c > m20 and c >= m60) or strong_up:
            lab = "up"
        elif (m20 < m60 and c < m20 and c <= m60) or strong_dn:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    labs = apply_hysteresis_ma60(
        raw,
        close=px["$close"].to_numpy(float),
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        confirm_leave=int(confirm_leave),
        confirm_enter=int(confirm_enter),
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_dual_ma_cross(px: pd.DataFrame) -> np.ndarray:
    return method_dual_ma_cross_params(px)


def method_adx_di(px: pd.DataFrame) -> np.ndarray:
    raw = []
    for _, row in px.iterrows():
        adx = float(row["adx14"]) if pd.notna(row["adx14"]) else 0.0
        pdi = float(row["plus_di"]) if pd.notna(row["plus_di"]) else 0.0
        mdi = float(row["minus_di"]) if pd.notna(row["minus_di"]) else 0.0
        c = float(row["$close"])
        m20 = float(row["MA20"]) if pd.notna(row["MA20"]) else np.nan
        m60 = float(row["MA60"]) if pd.notna(row["MA60"]) else np.nan
        p20 = float(row["prior20"]) if pd.notna(row["prior20"]) else 0.0
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        if adx < 18:
            if np.isfinite(m20) and np.isfinite(m60) and m20 > m60 and c >= m20 and (p20 >= 0.02 or p60 >= 0.03):
                lab = "up"
            elif np.isfinite(m20) and np.isfinite(m60) and m20 < m60 and c <= m20 and (p20 <= -0.02 or p60 <= -0.03):
                lab = "down"
            else:
                lab = "range"
        elif pdi > mdi:
            lab = "up"
        elif mdi > pdi:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=2,
    )
    return np.array(causal_smooth(labs, 3), dtype=object)


def method_adx_di_params(px: pd.DataFrame, *, adx_thr: float = 18, smooth: int = 5) -> np.ndarray:
    raw = []
    for _, row in px.iterrows():
        adx = float(row["adx14"]) if pd.notna(row["adx14"]) else 0.0
        pdi = float(row["plus_di"]) if pd.notna(row["plus_di"]) else 0.0
        mdi = float(row["minus_di"]) if pd.notna(row["minus_di"]) else 0.0
        if adx < adx_thr:
            lab = "range"
        elif pdi > mdi:
            lab = "up"
        elif mdi > pdi:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    above20 = (px["$close"] > px["MA20"]).fillna(False).to_numpy(bool)
    labs = apply_hysteresis(
        raw,
        above20=above20,
        ma20=px["MA20"].to_numpy(float),
        ma60=px["MA60"].to_numpy(float),
        close=px["$close"].to_numpy(float),
        confirm=2,
    )
    return np.array(causal_smooth(labs, max(1, int(smooth))), dtype=object)


def method_prior60_band(px: pd.DataFrame) -> np.ndarray:
    raw = []
    for _, row in px.iterrows():
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        band = float(row["band60"]) if pd.notna(row["band60"]) else np.nan
        c_above = bool(row["$close"] > row["MA20"]) if pd.notna(row["MA20"]) else False
        if pd.notna(band) and band < 0.22 and abs(p60) < 0.10:
            lab = "range"
        elif p60 >= 0.10 and c_above:
            lab = "up"
        elif p60 <= -0.10 and not c_above:
            lab = "down"
        elif p60 >= 0.05 and c_above:
            lab = "up"
        elif p60 <= -0.05 and not c_above:
            lab = "down"
        else:
            lab = "range"
        raw.append(lab)
    return np.array(causal_smooth(raw, 5), dtype=object)


def method_hybrid_v1_params(px: pd.DataFrame, *, confirm: int = 3, smooth: int = 3) -> np.ndarray:
    base = method_ma_stack_params(px, confirm=confirm, smooth=smooth)
    out = []
    for i, lab in enumerate(base):
        row = px.iloc[i]
        adx = float(row["adx14"]) if pd.notna(row["adx14"]) else 0.0
        p20 = float(row["prior20"]) if pd.notna(row["prior20"]) else 0.0
        p60 = float(row["prior60"]) if pd.notna(row["prior60"]) else 0.0
        c = float(row["$close"])
        m20 = float(row["MA20"]) if pd.notna(row["MA20"]) else np.nan
        mom = abs(p60) >= 0.06 or abs(p20) >= 0.04
        above = np.isfinite(m20) and c > m20
        if lab in ("up", "down") and adx < 15 and not mom and not (lab == "up" and above and p20 >= 0.02):
            out.append("range")
        else:
            out.append(lab)
    return np.asarray(out, dtype=object)


def method_hybrid_v1(px: pd.DataFrame) -> np.ndarray:
    return method_hybrid_v1_params(px)


METHODS = {
    "ma_stack_hyst": method_ma_stack,
    "ma_stack_struct": method_ma_stack_struct,
    "ma_stack_strict": method_ma_stack_strict,
    "dual_ma_cross": method_dual_ma_cross,
    "adx_di": method_adx_di,
    "prior60_band": method_prior60_band,
    "hybrid_ma_adx": method_hybrid_v1,
}


def run_method(name: str, px: pd.DataFrame, *, params: dict[str, Any] | None = None, min_seg: int = 10) -> np.ndarray:
    params = dict(params or {})
    label_keys = {
        "p60_thr",
        "band_chop",
        "p60_chop",
        "p20_vrev",
        "confirm",
        "confirm_enter",
        "confirm_leave",
        "smooth",
        "adx_thr",
    }
    use_params = bool(label_keys.intersection(params))
    if name == "adx_di" and use_params:
        labs = method_adx_di_params(px, adx_thr=float(params.get("adx_thr", 18)), smooth=int(params.get("smooth", 5)))
    elif name == "ma_stack_strict" and use_params:
        labs = method_ma_stack_strict_params(
            px,
            p60_thr=float(params.get("p60_thr", 0.06)),
            band_chop=float(params.get("band_chop", 0.26)),
            p60_chop=float(params.get("p60_chop", 0.12)),
            p20_vrev=float(params.get("p20_vrev", 0.06)),
            confirm=int(params.get("confirm", 3)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "ma_stack_hyst" and use_params:
        labs = method_ma_stack_params(px, confirm=int(params.get("confirm", 3)), smooth=int(params.get("smooth", 3)))
    elif name == "ma_stack_struct" and use_params:
        conf = int(params.get("confirm", params.get("confirm_enter", 3)))
        labs = method_ma_stack_struct_params(
            px,
            confirm_enter=int(params.get("confirm_enter", conf)),
            confirm_leave=int(params.get("confirm_leave", conf)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "dual_ma_cross" and use_params:
        conf = int(params.get("confirm", params.get("confirm_enter", 3)))
        labs = method_dual_ma_cross_params(
            px,
            confirm_enter=int(params.get("confirm_enter", conf)),
            confirm_leave=int(params.get("confirm_leave", conf)),
            smooth=int(params.get("smooth", 3)),
        )
    elif name == "hybrid_ma_adx" and use_params:
        labs = method_hybrid_v1_params(px, confirm=int(params.get("confirm", 3)), smooth=int(params.get("smooth", 3)))
    else:
        if name not in METHODS:
            raise KeyError(name)
        labs = METHODS[name](px)
    return finalize_regime_labels(labs, px, min_seg=int(params.get("min_seg", min_seg) or 0))
