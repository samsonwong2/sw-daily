"""Twelve-panel adaptive HTML: price through fig11, plus fig12 six states.

Panel order matches etf-daily from_listing pages. Fig7–11 are causal
trailing-log paths with expanding P95 rails. Fig12 is appended under the
main figure.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sw_daily.adaptive.stage import COST_PER_SIDE
from sw_daily.rules7.fair_path import (
    ANCHOR_ERP_ANN,
    FIG_PANELS,
    compute_multi_scale_frame,
    compute_volatility_regime_frame,
    fair_need_ann_series,
    gap_realized_minus_need,
)

_UNSAFE_NAME_RE = re.compile(r'[\\/:*?"<>|\s]+')
FIG12_START = "<!-- FIG12_SIX_STATES_START -->"
FIG12_END = "<!-- FIG12_SIX_STATES_END -->"

# fig7 is the 2-year panel inside FIG_PANELS; the rest are the extra trails.
_TRAIL_TITLES = {
    "fig7": "图7 K线 vs 公平路径（自有log趋势·滚动2年·因果P95）",
    "fig8": "图8 K线 vs 公平路径（自有log趋势·滚动1年·因果P95）",
    "fig9": "图9 K线 vs 公平路径（自有log趋势·滚动6个月·因果P95）",
    "fig10": "图10 K线 vs 公平路径（自有log趋势·滚动3个月·因果P95）",
    "fig11": "图11 K线 vs 公平路径（自有log趋势·滚动1个月·因果P95）",
}


def sanitize_name_for_filename(name: str) -> str:
    cleaned = _UNSAFE_NAME_RE.sub("_", str(name).strip()).strip("._")
    return cleaned or "unnamed"


def html_filename(code: str, name: str, start: str, end: str) -> str:
    """``regime_transition_{code}_{中文名}_{start}_{end}_adaptive.html``."""
    safe = sanitize_name_for_filename(name)
    start_tag = pd.Timestamp(start).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end).strftime("%Y%m%d")
    return f"regime_transition_{code}_{safe}_{start_tag}_{end_tag}_adaptive.html"


def _dated_close(frame: pd.DataFrame, date_col: str) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(frame[date_col])).normalize()
    series = pd.Series(pd.to_numeric(frame["$close"], errors="coerce").to_numpy(float), index=index)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _hold_up_nav(window: pd.DataFrame) -> pd.Series:
    """Mark-to-market equity of the hold-up rule, cost charged on each fill."""
    close = window["$close"].astype(float).to_numpy()
    regime = window["regime"].astype(str).to_numpy()
    cash, shares = 1.0, 0.0
    prev = None
    nav = np.empty(len(close))
    for i, (px, rg) in enumerate(zip(close, regime)):
        if shares == 0 and rg == "up" and prev != "up" and px > 0:
            shares = cash * (1.0 - COST_PER_SIDE) / px
            cash = 0.0
        elif shares > 0 and rg != "up":
            cash = shares * px * (1.0 - COST_PER_SIDE)
            shares = 0.0
        nav[i] = cash + shares * px
        prev = rg
    return pd.Series(nav, index=pd.DatetimeIndex(pd.to_datetime(window["as_of"])).normalize())


def _ohlc(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(frame[date_col]),
            "open": pd.to_numeric(frame["$open"], errors="coerce"),
            "high": pd.to_numeric(frame["$high"], errors="coerce"),
            "low": pd.to_numeric(frame["$low"], errors="coerce"),
            "close": pd.to_numeric(frame["$close"], errors="coerce"),
        }
    )
    if "$volume" in frame.columns:
        out["volume"] = pd.to_numeric(frame["$volume"], errors="coerce")
    return out


def build_adaptive_figure(  # noqa: PLR0913
    window: pd.DataFrame,
    history: pd.DataFrame,
    *,
    code: str,
    name: str,
    method: str,
    start: str,
    end: str,
    seg_df: pd.DataFrame,
    events: list[dict[str, Any]],
    open_mtm: dict[str, Any] | None,
    anchor_close: pd.Series | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    """Shared-x figure: price, volume, vol, two gap rows, NAV, fig7–fig11."""
    plot = _ohlc(window, "as_of")
    hist_close = _dated_close(history, "datetime" if "datetime" in history.columns else "as_of")
    plot_close = _dated_close(window, "as_of")
    ms = compute_multi_scale_frame(hist_close).reindex(plot_close.index)
    hv = compute_volatility_regime_frame(hist_close)
    hv["as_of"] = pd.to_datetime(hv["as_of"]).dt.normalize()
    hv = hv.set_index("as_of")
    vol = hv.reindex(plot_close.index)

    need20 = gap20 = need5 = gap5 = None
    if anchor_close is not None and not anchor_close.empty:
        anchor = anchor_close.copy()
        anchor.index = pd.DatetimeIndex(pd.to_datetime(anchor.index)).normalize()
        anchor = anchor[~anchor.index.duplicated(keep="last")].sort_index()
        anchor_vol = compute_volatility_regime_frame(anchor)
        anchor_vol["as_of"] = pd.to_datetime(anchor_vol["as_of"]).dt.normalize()
        anchor_vol = anchor_vol.set_index("as_of")
        n20 = fair_need_ann_series(hv["rv20"], anchor_vol["rv20"].reindex(hv.index), erp_ann=ANCHOR_ERP_ANN)
        g20 = gap_realized_minus_need(hist_close, n20, bars=20)
        n5 = fair_need_ann_series(hv["rv5"], anchor_vol["rv5"].reindex(hv.index), erp_ann=ANCHOR_ERP_ANN)
        g5 = gap_realized_minus_need(hist_close, n5, bars=5)
        need20 = n20.reindex(plot_close.index)
        gap20 = g20["gap"].reindex(plot_close.index)
        need5 = n5.reindex(plot_close.index)
        gap5 = g5["gap"].reindex(plot_close.index)

    titles = [
        "价格",
        "成交量",
        "波动率",
        "公平年化与20日差额",
        "公平年化与5日差额",
        "自训练策略净值",
        *[_TRAIL_TITLES[key] for key, _, _ in FIG_PANELS],
    ]
    n_trail = len(FIG_PANELS)
    row_heights = [0.16, 0.05, 0.07, 0.07, 0.07, 0.08] + [0.50 / n_trail] * n_trail
    specs: list[list[dict]] = [
        [{}],
        [{}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{}],
    ] + [[{}] for _ in range(n_trail)]
    fig = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.012,
        subplot_titles=tuple(titles),
        row_heights=row_heights,
        specs=specs,
    )
    x = plot["datetime"]
    fig.add_trace(
        go.Candlestick(x=x, open=plot["open"], high=plot["high"], low=plot["low"], close=plot["close"], name="K线", showlegend=False),
        row=1, col=1,
    )
    for ma, color in (("MA5", "#1f77b4"), ("MA20", "#ff7f0e")):
        if ma in window.columns:
            fig.add_trace(
                go.Scatter(x=window["as_of"], y=window[ma], mode="lines", name=ma, line=dict(color=color, width=1.2)),
                row=1, col=1,
            )
    colors = {"up": "rgba(46,160,67,0.15)", "down": "rgba(214,39,40,0.15)", "range": "rgba(120,120,120,0.08)"}
    for row in seg_df.itertuples(index=False):
        fig.add_vrect(x0=row.start, x1=row.end, fillcolor=colors.get(row.regime, "rgba(0,0,0,0)"), line_width=0, layer="below", row=1, col=1)
    buys = [event for event in events if event["side"] == "BUY"]
    sells = [event for event in events if event["side"] == "SELL"]
    if buys:
        fig.add_trace(go.Scatter(x=[e["date"] for e in buys], y=[e["px"] for e in buys], mode="markers", name="买入", marker=dict(symbol="triangle-up", color="#2ca02c", size=9)), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(x=[e["date"] for e in sells], y=[e["px"] for e in sells], mode="markers", name="卖出", marker=dict(symbol="triangle-down", color="#d62728", size=9)), row=1, col=1)
    if open_mtm:
        fig.add_trace(go.Scatter(x=[open_mtm["last_date"]], y=[open_mtm["last_px"]], mode="markers", name="持仓", marker=dict(symbol="diamond", color="#1f77b4", size=10)), row=1, col=1)

    if "volume" in plot.columns:
        fig.add_trace(go.Bar(x=x, y=plot["volume"], name="成交量", marker_color="#9ecae1", showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=plot_close.index, y=vol["rv5"], name="rv5", line=dict(width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_close.index, y=vol["rv20"], name="rv20", line=dict(width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_close.index, y=vol["vol5_pct_120d"], name="vol5分位", line=dict(width=1, dash="dot")), row=3, col=1, secondary_y=True)

    if need20 is not None:
        fig.add_trace(go.Scatter(x=need20.index, y=need20, name="公平年化20", line=dict(width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=gap20.index, y=gap20, name="20日差额", line=dict(width=1)), row=4, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=need5.index, y=need5, name="公平年化5", line=dict(width=1.2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=gap5.index, y=gap5, name="5日差额", line=dict(width=1)), row=5, col=1, secondary_y=True)

    nav = _hold_up_nav(window)
    bh = plot_close / float(plot_close.iloc[0])
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="hold_up", line=dict(width=1.4)), row=6, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="买入持有", line=dict(width=1, dash="dot")), row=6, col=1)

    for offset, (key, _years, _label) in enumerate(FIG_PANELS):
        row = 7 + offset
        fig.add_trace(
            go.Candlestick(
                x=x, open=plot["open"], high=plot["high"], low=plot["low"], close=plot["close"],
                name=key, showlegend=False,
            ),
            row=row, col=1,
        )
        fig.add_trace(go.Scatter(x=ms.index, y=ms[f"{key}_path"], name=f"{key}路径", line=dict(width=1.3, color="#2ca02c"), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=ms.index, y=ms[f"{key}_upper"], name=f"{key}上轨", line=dict(width=1, color="#98df8a"), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=ms.index, y=ms[f"{key}_lower"], name=f"{key}下轨", line=dict(width=1, color="#dbdb8d"), showlegend=False), row=row, col=1)

    fig.update_layout(
        title=f"{code} {name} · {method} · hold_up · {start} → {end}",
        height=3400,
        template="plotly_white",
        legend=dict(orientation="h", y=1.01),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig, ms


def render_html(
    window: pd.DataFrame,
    history: pd.DataFrame,
    *,
    code: str,
    name: str,
    method: str,
    start: str,
    end: str,
    seg_df: pd.DataFrame,
    events: list[dict[str, Any]],
    open_mtm: dict[str, Any] | None,
    anchor_close: pd.Series | None = None,
) -> str:
    fig, ms = build_adaptive_figure(
        window, history, code=code, name=name, method=method, start=start, end=end,
        seg_df=seg_df, events=events, open_mtm=open_mtm, anchor_close=anchor_close,
    )
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    fig12 = _fig12_block(window, ms, f"{code} {name}")
    body = html.lower().rfind("</body>")
    if body < 0:
        return html + fig12
    return html[:body] + fig12 + html[body:]


def _fig12_block(window: pd.DataFrame, ms: pd.DataFrame, code_name: str) -> str:
    from sw_daily.rules7.six_states import build_fig12_figure, classify_six_states, current_state

    states = classify_six_states(ms)
    ohlcv = window.set_index(pd.DatetimeIndex(pd.to_datetime(window["as_of"])).normalize())
    fig = build_fig12_figure(ohlcv, states, code_name)
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig12_six_states")
    state = current_state(states)
    heading = (
        f'<h3 style="font-family:sans-serif;padding-left:12px">图12 六状态背景色'
        f'<span style="font-size:13px;color:#666">（当前状态：{state}）</span></h3>'
    )
    return (
        f"\n{FIG12_START}\n<hr style=\"margin:24px 0\">\n{heading}\n{inner}\n{FIG12_END}\n"
    )
