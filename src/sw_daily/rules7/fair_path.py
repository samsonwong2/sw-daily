"""Causal fair-path, realized-vol, and gap helpers used by rules7.

Ported from etf-daily's plot and vol-board modules. Every series uses only
bars through the current date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
ANCHOR_ERP_ANN = 0.043
FAIR_PATH_LOG_TREND_MIN_BARS = 63
FAIR_PATH_LOG_TREND_TRAIL_YEARS = 2
DEFAULT_RV_WINDOW = 5
DEFAULT_RV20_WINDOW = 20
DEFAULT_VOL_PCT_LOOKBACK = 120
DEFAULT_VOL_PCT_MIN_HISTORY = 5
HORIZON_5 = 5
HORIZON_20 = 20

FIG_PANELS: tuple[tuple[str, float, str], ...] = (
    ("fig7", 2.0, "2年"),
    ("fig8", 1.0, "1年"),
    ("fig9", 0.5, "6个月"),
    ("fig10", 0.25, "3个月"),
    ("fig11", 1.0 / 12.0, "1个月"),
)


def _slice_own_cagr(
    y_arr: np.ndarray,
    start: int,
    end: int,
    *,
    trading_days_per_year: float,
) -> float:
    """CAGR on ``y_arr[start:end+1]`` using first/last positive finite closes."""
    if end < start:
        return float("nan")
    sl = y_arr[start : end + 1]
    ok = np.isfinite(sl) & (sl > 0)
    if int(ok.sum()) < 2:
        return float("nan")
    idx = np.flatnonzero(ok)
    p0 = float(sl[int(idx[0])])
    pt = float(sl[int(idx[-1])])
    n_steps = int(idx[-1] - idx[0])
    tdy = float(trading_days_per_year)
    if n_steps <= 0 or p0 <= 0 or not np.isfinite(pt) or not np.isfinite(tdy) or tdy <= 0:
        return float("nan")
    return float((pt / p0) ** (tdy / float(n_steps)) - 1.0)


def _trail_fit_min_bars(
    trail_years: float,
    *,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
    floor: int | None = None,
) -> int:
    """OLS min bars: never demand more points than the trail window holds."""
    tdy = float(trading_days_per_year)
    y = float(trail_years)
    trail_bars = int(round(y * tdy)) if np.isfinite(y) and y > 0 and tdy > 0 else 0
    base = int(FAIR_PATH_LOG_TREND_MIN_BARS) if floor is None else max(2, int(floor))
    if trail_bars <= 0:
        return max(2, base)
    return max(2, min(base, trail_bars))


def trailing_log_trend_price_path(
    close: pd.Series,
    *,
    trail_years: float | None = None,
    min_bars: int | None = None,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
    fallback_erp: float = ANCHOR_ERP_ANN,
) -> tuple[pd.Series, pd.Series]:
    """Causal trailing OLS on ``log(close)``. Returns ``(path, g)``."""
    y = pd.to_numeric(close, errors="coerce")
    out = pd.Series(np.nan, index=y.index, dtype=float)
    g_out = pd.Series(np.nan, index=y.index, dtype=float)
    tdy = float(trading_days_per_year)
    if y.empty or not np.isfinite(tdy) or tdy <= 0:
        return out, g_out
    trail_y = float(FAIR_PATH_LOG_TREND_TRAIL_YEARS) if trail_years is None else float(trail_years)
    min_b = int(FAIR_PATH_LOG_TREND_MIN_BARS) if min_bars is None else max(2, int(min_bars))
    trail_bars = int(round(trail_y * tdy)) if np.isfinite(trail_y) and trail_y > 0 else 0
    y_arr = y.to_numpy(dtype=float)
    n = len(y_arr)
    path_arr = np.full(n, np.nan, dtype=float)
    g_arr = np.full(n, np.nan, dtype=float)
    fb = float(fallback_erp) if np.isfinite(fallback_erp) else float("nan")
    for t in range(n):
        start = 0
        if trail_bars > 0 and (t + 1) >= trail_bars:
            start = t + 1 - trail_bars
        sl = y_arr[start : t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        n_ok = int(ok.sum())
        if n_ok < min_b:
            continue
        jj = np.flatnonzero(ok).astype(float)
        log_y = np.log(sl[ok])
        b, a = np.polyfit(jj, log_y, 1)
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        path_arr[t] = float(np.exp(a + b * float(jj[-1])))
        g_arr[t] = float(np.exp(b * tdy) - 1.0)
    first_fit = int(np.flatnonzero(np.isfinite(path_arr))[0]) if np.isfinite(path_arr).any() else n
    for t in range(first_fit):
        sl = y_arr[: t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) < 2:
            continue
        g = _slice_own_cagr(y_arr, 0, t, trading_days_per_year=tdy)
        if not np.isfinite(g):
            g = fb
        if not np.isfinite(g):
            continue
        idx = np.flatnonzero(ok)
        p0 = float(sl[int(idx[0])])
        n_steps = int(idx[-1] - idx[0])
        path_arr[t] = p0 * ((1.0 + float(g)) ** (n_steps / tdy)) if n_steps > 0 else p0
        g_arr[t] = float(g)
    out.iloc[:] = path_arr
    g_out.iloc[:] = g_arr
    return out, g_out


def erp_path_symmetric_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = 0.95,
    min_bars: int | None = None,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal expanding P95 bands parallel to ``path``."""
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = int(FAIR_PATH_LOG_TREND_MIN_BARS) if min_bars is None else max(2, int(min_bars))
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    path_arr = p.to_numpy(dtype=float)
    close_arr = c.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_hist: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        ct = float(close_arr[t])
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
        if len(abs_hist) < min_b:
            continue
        if not (np.isfinite(pt) and pt > 0):
            continue
        delta_t = float(np.quantile(np.asarray(abs_hist, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        upper_arr[t] = pt * float(np.exp(delta_t))
        lower_arr[t] = pt * float(np.exp(-delta_t))
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def compute_realized_vol(
    close: pd.Series,
    *,
    window: int = DEFAULT_RV_WINDOW,
    ann_factor: float = TRADING_DAYS_PER_YEAR,
    ddof: int = 0,
) -> pd.Series:
    """Annualized rolling realized vol from log returns (causal through T)."""
    px = pd.to_numeric(close, errors="coerce")
    log_ret = np.log(px / px.shift(1))
    rv = log_ret.rolling(int(window), min_periods=int(window)).std(ddof=int(ddof))
    return rv * float(np.sqrt(ann_factor))


def compute_vol5_pct_120d(
    rv: pd.Series,
    *,
    lookback: int = DEFAULT_VOL_PCT_LOOKBACK,
    min_history: int = DEFAULT_VOL_PCT_MIN_HISTORY,
) -> pd.Series:
    """Percentile of current rv inside the trailing lookback window."""
    values = pd.to_numeric(rv, errors="coerce").to_numpy(dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    lb = int(lookback)
    min_hist = int(min_history)
    for i in range(values.shape[0]):
        cur = values[i]
        if not np.isfinite(cur):
            continue
        start = max(0, i + 1 - lb)
        window = values[start : i + 1]
        finite = window[np.isfinite(window)]
        if finite.size < min_hist:
            continue
        out[i] = float(np.sum(finite <= cur)) / float(finite.size)
    return pd.Series(out, index=rv.index, name="vol5_pct_120d")


def compute_volatility_regime_frame(
    close: pd.Series,
    *,
    rv_window: int = DEFAULT_RV_WINDOW,
    rv20_window: int = DEFAULT_RV20_WINDOW,
    lookback: int = DEFAULT_VOL_PCT_LOOKBACK,
    min_history: int = DEFAULT_VOL_PCT_MIN_HISTORY,
) -> pd.DataFrame:
    """Causal rv5 / rv20 / vol5 percentile. Columns: as_of, rv5, rv20, vol5_pct_120d."""
    if close is None:
        raise ValueError("close series is required")
    series = pd.Series(pd.to_numeric(close, errors="coerce"), copy=False)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series.index = pd.DatetimeIndex(series.index).normalize()
    rv5 = compute_realized_vol(series, window=rv_window, ddof=0)
    rv20 = compute_realized_vol(series, window=rv20_window, ddof=0)
    rv5_for_pct = compute_realized_vol(series, window=rv_window, ddof=1)
    vol_pct = compute_vol5_pct_120d(rv5_for_pct, lookback=lookback, min_history=min_history)
    return pd.DataFrame(
        {
            "as_of": series.index,
            "rv5": rv5.to_numpy(dtype=float),
            "rv20": rv20.to_numpy(dtype=float),
            "vol5_pct_120d": vol_pct.to_numpy(dtype=float),
        }
    )


def fair_need_ann_series(
    rv20: pd.Series,
    rv20_anchor: pd.Series,
    *,
    erp_ann: float = ANCHOR_ERP_ANN,
) -> pd.Series:
    """Fair annualized need: ``erp_ann × rv / rv_anchor``."""
    left = pd.to_numeric(rv20, errors="coerce")
    right = pd.to_numeric(rv20_anchor, errors="coerce")
    aligned = pd.concat([left.rename("rv"), right.rename("anchor")], axis=1, join="inner")
    if aligned.empty:
        return pd.Series(dtype=float, name="r_need_ann")
    prem = float(erp_ann)
    out = np.full(len(aligned), np.nan, dtype=float)
    if np.isfinite(prem):
        ok = (
            np.isfinite(aligned["rv"].to_numpy(dtype=float))
            & np.isfinite(aligned["anchor"].to_numpy(dtype=float))
            & (aligned["anchor"].to_numpy(dtype=float) > 0.0)
            & (aligned["rv"].to_numpy(dtype=float) >= 0.0)
        )
        out[ok] = prem * aligned["rv"].to_numpy(dtype=float)[ok] / aligned["anchor"].to_numpy(dtype=float)[ok]
    return pd.Series(out, index=aligned.index, name="r_need_ann")


def realized_simple_return(close: pd.Series, *, bars: int = HORIZON_20) -> pd.Series:
    """Causal N-bar simple return: ``close[t] / close[t-N] − 1``."""
    c = pd.to_numeric(close, errors="coerce")
    c = c[~c.index.duplicated(keep="last")].sort_index()
    n = int(bars)
    name = f"r_{n}d"
    if n <= 0 or c.empty:
        return pd.Series(dtype=float, name=name)
    return c.pct_change(n, fill_method=None).rename(name)


def period_return_from_ann(r_need_ann: pd.Series, *, bars: int = HORIZON_20) -> pd.Series:
    """``(1 + r_need_ann)^(N/252) − 1``. Values ``<= -1`` become NaN."""
    r = pd.to_numeric(r_need_ann, errors="coerce")
    n = int(bars)
    name = f"r_need_{n}d"
    out = np.full(len(r), np.nan, dtype=float)
    if n <= 0 or r.empty:
        return pd.Series(out, index=r.index, name=name)
    vals = r.to_numpy(dtype=float)
    ok = np.isfinite(vals) & (vals > -1.0)
    out[ok] = np.power(1.0 + vals[ok], float(n) / float(TRADING_DAYS_PER_YEAR)) - 1.0
    return pd.Series(out, index=r.index, name=name)


def gap_realized_minus_need(
    close: pd.Series,
    r_need_ann: pd.Series,
    *,
    bars: int = HORIZON_20,
) -> pd.DataFrame:
    """Daily gap: realized N-bar return minus fair period need."""
    need_ann = pd.to_numeric(r_need_ann, errors="coerce").rename("r_need_ann")
    need_n = period_return_from_ann(need_ann, bars=bars)
    r_n = realized_simple_return(close, bars=bars)
    aligned = pd.concat(
        [need_ann, need_n.rename("r_need_period"), r_n.rename("r_realized")],
        axis=1,
        join="outer",
    )
    aligned["gap"] = aligned["r_realized"] - aligned["r_need_period"]
    return aligned


def atr20(ohlcv: pd.DataFrame) -> pd.Series:
    h = ohlcv["$high"].astype(float)
    l = ohlcv["$low"].astype(float)
    c = ohlcv["$close"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(20).mean()


def compute_multi_scale_frame(close: pd.Series) -> pd.DataFrame:
    """One row per date; path/g/upper/lower/gap/touch columns per fig panel."""
    idx = close.index
    out: dict[str, pd.Series] = {"px": close.astype(float)}
    for key, years, _ in FIG_PANELS:
        path, g = trailing_log_trend_price_path(
            close,
            trail_years=float(years),
            min_bars=_trail_fit_min_bars(float(years)),
        )
        upper, lower, _delta = erp_path_symmetric_envelope(path, close)
        out[f"{key}_path"] = path
        out[f"{key}_g"] = g
        out[f"{key}_upper"] = upper
        out[f"{key}_lower"] = lower
        gap = close / path - 1.0
        out[f"{key}_gap"] = gap.where(path > 0)
        out[f"{key}_touch_lo"] = close <= lower
        out[f"{key}_touch_hi"] = close >= upper
        out[f"{key}_above_path"] = close >= path
    df = pd.DataFrame(out, index=idx)
    df.index.name = "date"
    return df
