"""Feature frame for rules7: MA/gap, fig7–11 rails, and aux_edge masks.

Hard-buy and second-knife columns from the ETF extrema script are not kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sw_daily.rules7.fair_path import (
    ANCHOR_ERP_ANN,
    atr20,
    compute_multi_scale_frame,
    compute_volatility_regime_frame,
    fair_need_ann_series,
    gap_realized_minus_need,
)

DEFAULT_NEAR_PCT = 0.01
DEFAULT_POS_MIN = 0.80
DEFAULT_FIG10_POS_MIN = 0.90
DEFAULT_FIG7_PREM = 0.05
DEFAULT_VOL_CLUSTER = 0.70


def causal_local_extrema(close: pd.Series, lookback: int) -> tuple[pd.Series, pd.Series]:
    """True when today's close is min/max of [t-lookback, t] (no future bars)."""
    px = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    n = len(px)
    is_min = np.zeros(n, dtype=bool)
    is_max = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        w = px[i - lookback : i + 1]
        if not np.isfinite(px[i]):
            continue
        if px[i] == np.nanmin(w):
            is_min[i] = True
        if px[i] == np.nanmax(w):
            is_max[i] = True
    idx = close.index
    return pd.Series(is_min, index=idx), pd.Series(is_max, index=idx)


def _band_pos(px: pd.Series, lower: pd.Series, upper: pd.Series) -> pd.Series:
    lo = np.log(pd.to_numeric(lower, errors="coerce"))
    hi = np.log(pd.to_numeric(upper, errors="coerce"))
    return (np.log(px) - lo) / (hi - lo)


def build_feature_frame_from_series(
    close: pd.Series,
    ohlcv: pd.DataFrame,
    anchor_close: pd.Series,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Feature frame from a close series. ``ohlcv`` is indexed by date."""
    vf_a = compute_volatility_regime_frame(anchor_close)
    vf_a["as_of"] = pd.to_datetime(vf_a["as_of"]).dt.normalize()
    vf_a = vf_a.set_index("as_of")
    frame = compute_multi_scale_frame(close)
    vol = compute_volatility_regime_frame(close)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    vol = vol.set_index("as_of").reindex(close.index)
    need20 = fair_need_ann_series(
        pd.to_numeric(vol["rv20"], errors="coerce"),
        pd.to_numeric(vf_a["rv20"], errors="coerce").reindex(close.index),
        erp_ann=ANCHOR_ERP_ANN,
    )
    g20 = gap_realized_minus_need(close, need20, bars=20)
    frame["gap_20d"] = pd.to_numeric(g20["gap"], errors="coerce").reindex(close.index)
    need5 = fair_need_ann_series(
        pd.to_numeric(vol["rv5"], errors="coerce"),
        pd.to_numeric(vf_a["rv5"], errors="coerce").reindex(close.index),
        erp_ann=ANCHOR_ERP_ANN,
    )
    g5 = gap_realized_minus_need(close, need5, bars=5)
    frame["gap_5d"] = pd.to_numeric(g5["gap"], errors="coerce").reindex(close.index)
    frame["ma5"] = close.rolling(5).mean()
    frame["ma20"] = close.rolling(20).mean()
    frame["s1"] = frame["fig9_touch_hi"].fillna(False) | frame["fig10_touch_hi"].fillna(False)
    frame["s_lo"] = frame["fig9_touch_lo"].fillna(False) | frame["fig10_touch_lo"].fillna(False)
    frame["disc5"] = frame["fig7_gap"] <= -0.05
    locmin, locmax = causal_local_extrema(close, lookback)
    frame["locmin"] = locmin
    frame["locmax"] = locmax
    frame["vol5_pct_120d"] = pd.to_numeric(vol["vol5_pct_120d"], errors="coerce")
    frame["atr20"] = atr20(ohlcv).reindex(close.index)
    return frame


def prepare_fig9_touch_frame(
    frame: pd.DataFrame,
    *,
    near_pct: float = DEFAULT_NEAR_PCT,
    pos_min: float = DEFAULT_POS_MIN,
    fig10_pos_min: float = DEFAULT_FIG10_POS_MIN,
    fig7_prem: float = DEFAULT_FIG7_PREM,
    vol_cluster: float = DEFAULT_VOL_CLUSTER,
    mid_stretch: bool = True,
) -> pd.DataFrame:
    """Add near-rail, aux, band-pos, and mid-stretch sell columns (causal)."""
    out = frame.copy()
    close = out["px"].astype(float)
    near = float(near_pct)
    lower = pd.to_numeric(out["fig9_lower"], errors="coerce")
    upper = pd.to_numeric(out["fig9_upper"], errors="coerce")
    out["fig9_near_lo"] = (close <= lower * (1.0 + near)).fillna(False)
    out["fig9_near_hi"] = (close >= upper * (1.0 - near)).fillna(False)
    out["aux_buy"] = (
        (close < out["ma20"])
        & (out["ma5"] < out["ma20"])
        & (out["gap_5d"] <= 0)
        & (out["gap_20d"] <= 0)
        & (~out["fig10_touch_hi"].fillna(False))
    )
    out["aux_sell"] = (
        (close > out["ma20"])
        & (out["ma5"] > out["ma20"])
        & (out["gap_5d"] >= 0)
        & (out["gap_20d"] >= 0)
    )
    out["fig9_pos"] = _band_pos(close, lower, upper)
    out["fig10_pos"] = _band_pos(
        close,
        pd.to_numeric(out["fig10_lower"], errors="coerce"),
        pd.to_numeric(out["fig10_upper"], errors="coerce"),
    )
    s1 = out["fig9_touch_hi"].fillna(False) | out["fig10_touch_hi"].fillna(False)
    if not mid_stretch:
        out["fig9_mid_stretch_sell"] = pd.Series(False, index=out.index)
    else:
        out["fig9_mid_stretch_sell"] = (
            out["aux_sell"].fillna(False)
            & out["locmax"].fillna(False)
            & (pd.to_numeric(out["vol5_pct_120d"], errors="coerce") >= float(vol_cluster))
            & (pd.to_numeric(out["fig9_g"], errors="coerce") > 0)
            & (~s1)
            & (out["fig9_pos"] >= float(pos_min))
            & (
                (pd.to_numeric(out["fig7_gap"], errors="coerce") >= float(fig7_prem))
                | (out["fig10_pos"] >= float(fig10_pos_min))
            )
        ).fillna(False)
    return out


def rising_edge(mask: pd.Series) -> pd.Series:
    m = np.asarray(mask.fillna(False), dtype=bool)
    out = np.zeros(len(m), dtype=bool)
    if len(m):
        out[0] = bool(m[0])
        if len(m) > 1:
            out[1:] = m[1:] & ~m[:-1]
    return pd.Series(out, index=mask.index)


def falling_edge(mask: pd.Series) -> pd.Series:
    m = np.asarray(mask.fillna(False), dtype=bool)
    out = np.zeros(len(m), dtype=bool)
    if len(m) > 1:
        out[1:] = (~m[1:]) & m[:-1]
    return pd.Series(out, index=mask.index)


def variant_masks(frame: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series, str]:
    lo = frame["fig9_near_lo"].fillna(False).astype(bool)
    hi = frame["fig9_near_hi"].fillna(False).astype(bool)
    extra = frame["fig9_mid_stretch_sell"].fillna(False).astype(bool)
    aux_b = frame["aux_buy"].fillna(False).astype(bool)
    aux_s = frame["aux_sell"].fillna(False).astype(bool)
    locmin = frame["locmin"].fillna(False).astype(bool)
    locmax = frame["locmax"].fillna(False).astype(bool)
    notes = {
        "raw": "图9近下买 / 近上轨∨带内分位拉伸（状态日）",
        "edge": "图9近下/近上上跳沿 ∨ 带内分位拉伸",
        "aux": "图9近轨 ∧ 均线/gap；卖再∨带内分位拉伸",
        "aux_edge": "图9近轨上跳沿 ∧ 均线/gap；卖再∨带内分位拉伸",
        "aux_extrema": "图9近轨 ∧ 均线/gap ∧ 局部极值；卖再∨带内分位拉伸",
    }
    if name == "raw":
        return lo, hi | extra, notes[name]
    if name == "edge":
        return rising_edge(lo), rising_edge(hi) | extra, notes[name]
    if name == "aux":
        return lo & aux_b, (hi & aux_s) | extra, notes[name]
    if name == "aux_edge":
        return rising_edge(lo) & aux_b, (rising_edge(hi) & aux_s) | extra, notes[name]
    if name == "aux_extrema":
        return lo & aux_b & locmin, (hi & aux_s & locmax) | extra, notes[name]
    raise ValueError(name)
