"""Causal six-state labels and the fig12 candlestick.

Priority: 过热, 急跌刀锋, 趋势下行, 趋势上行, 底部修复, 震荡.
"""
from __future__ import annotations

from typing import Mapping

import pandas as pd
import plotly.graph_objects as go

STATE_WARMUP = "数据不足"
STATE_OVERHEAT = "①过热"
STATE_KNIFE = "②急跌刀锋"
STATE_DOWN = "③趋势下行"
STATE_REPAIR = "④底部修复"
STATE_UP = "⑤趋势上行"
STATE_CHOP = "⑥震荡"

STATE_ORDER: tuple[str, ...] = (
    STATE_OVERHEAT,
    STATE_KNIFE,
    STATE_DOWN,
    STATE_REPAIR,
    STATE_UP,
    STATE_CHOP,
)

STATE_COLORS: dict[str, str] = {
    STATE_OVERHEAT: "rgba(214,39,40,0.32)",
    STATE_KNIFE: "rgba(0,100,0,0.30)",
    STATE_DOWN: "rgba(60,179,113,0.16)",
    STATE_REPAIR: "rgba(100,149,237,0.22)",
    STATE_UP: "rgba(255,120,120,0.20)",
    STATE_CHOP: "rgba(150,150,150,0.13)",
}

STATE_LEGEND_COLORS: dict[str, str] = {
    STATE_OVERHEAT: "rgba(214,39,40,0.85)",
    STATE_KNIFE: "rgba(0,100,0,0.85)",
    STATE_DOWN: "rgba(60,179,113,0.85)",
    STATE_REPAIR: "rgba(100,149,237,0.85)",
    STATE_UP: "rgba(255,120,120,0.85)",
    STATE_CHOP: "rgba(150,150,150,0.85)",
}

_KNIFE_BARS = 10


def classify_six_states(ms: pd.DataFrame) -> pd.Series:
    """One label per row of a multi-scale frame."""
    required = ("px", "fig8_g", "fig9_g", "fig10_g", "fig10_path", "fig9_touch_hi", "fig10_touch_hi")
    missing = [col for col in required if col not in ms.columns]
    if missing:
        raise KeyError(f"classify_six_states missing columns: {missing}")
    px = pd.to_numeric(ms["px"], errors="coerce")
    p10 = pd.to_numeric(ms["fig10_path"], errors="coerce")
    g8 = pd.to_numeric(ms["fig8_g"], errors="coerce")
    g9 = pd.to_numeric(ms["fig9_g"], errors="coerce")
    g10 = pd.to_numeric(ms["fig10_g"], errors="coerce")
    touch_hi = ms["fig9_touch_hi"].eq(True) | ms["fig10_touch_hi"].eq(True)
    below = (px < p10).fillna(False)
    below10_run = below.astype(int).groupby((~below).cumsum()).cumsum()
    warmup = ~(px.notna() & p10.notna())
    out = pd.Series(STATE_CHOP, index=ms.index, dtype=object)
    out = out.mask(px.ge(p10), STATE_REPAIR)
    out = out.mask(g8.gt(0) & g9.gt(0), STATE_UP)
    out = out.mask(g10.le(0) & g9.le(0), STATE_DOWN)
    out = out.mask(below10_run.ge(_KNIFE_BARS), STATE_KNIFE)
    out = out.mask(touch_hi.fillna(False), STATE_OVERHEAT)
    out = out.mask(warmup, STATE_WARMUP)
    out.name = "six_state"
    return out


def current_state(states: pd.Series) -> str:
    if states.empty:
        return STATE_WARMUP
    return str(states.iloc[-1])


def build_fig12_figure(
    ohlcv: pd.DataFrame,
    states: pd.Series,
    code_name: str,
    *,
    colors: Mapping[str, str] | None = None,
) -> go.Figure:
    """Clean candlesticks plus consecutive-state background. No rangeslider."""
    fill = dict(STATE_COLORS if colors is None else colors)
    idx = pd.DatetimeIndex(pd.to_datetime(ohlcv.index)).normalize()
    frame = pd.DataFrame(
        {
            "open": pd.to_numeric(ohlcv["$open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(ohlcv["$high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(ohlcv["$low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(ohlcv["$close"], errors="coerce").to_numpy(),
        },
        index=idx,
    )
    st = states.copy()
    st.index = pd.DatetimeIndex(pd.to_datetime(st.index)).normalize()
    st = st.reindex(frame.index).fillna(STATE_WARMUP)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
            name="K线",
            showlegend=False,
        )
    )
    runs = (st != st.shift()).cumsum()
    for _, seg in st.groupby(runs, sort=False):
        label = str(seg.iloc[0])
        color = fill.get(label)
        if color is None:
            continue
        fig.add_vrect(
            x0=pd.Timestamp(seg.index[0]),
            x1=pd.Timestamp(seg.index[-1]) + pd.Timedelta(days=1),
            fillcolor=color,
            line_width=0,
            layer="below",
        )
    for label in STATE_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=STATE_LEGEND_COLORS.get(label, fill[label])),
                name=label,
            )
        )
    fig.update_layout(
        title=f"图12 六状态背景色 · {code_name}" if code_name else "图12 六状态背景色",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.08),
    )
    return fig
