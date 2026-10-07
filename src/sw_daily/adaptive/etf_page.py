"""Twelve-panel listing page, same layout as the 801951 coal example.

Uses etf-daily's figure (price through fig11, EWMA rails, legend below)
and appends fig12. The caller passes bars that already went through
``prepare_ohlcv``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from sw_daily.adaptive.chart import html_filename

EXTRA_TRAILS = "1,0.5,0.25,1/12"
_DROP_TITLES = {
    "价格",
    "成交量",
    "波动率",
    "公平年化与20日差额",
    "公平年化与5日差额",
    "自训练策略净值",
}


def _imports():
    from etf_daily.lib.adaptive_stage_common import annotate_figure, _apply_title_top_margin
    from etf_daily.lib.plotly_html_autoscale import write_adaptive_html
    from etf_daily.lib.regime_transition_plot import compute_volatility_regime_frame
    from etf_daily.lib.self_train_policy import (
        config_block,
        evaluate_self_train_policy,
        policy_frame_from_eval,
    )
    from etf_daily.lib.six_states import build_fig12_figure, classify_six_states, current_state
    from etf_daily.lib.vol_need_return_board import ANCHOR_ERP_ANN
    from etf_daily.plots.add_fig12_six_states import _inject_before_body_end, build_fig12_block
    from etf_daily.plots.plot_adaptive_stage_pool import _need_frame_vs_anchor
    from etf_daily.plots.plot_regime_transition_example import build_candle_volume_vol_figure
    from sw_daily.rules7.fair_path import compute_multi_scale_frame

    return locals()


def _as_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "datetime" not in out.columns and "as_of" in out.columns:
        out = out.rename(columns={"as_of": "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    if "$volume" not in out.columns:
        out["$volume"] = 0.0
    return out


def _anchor_rv(anchor_close: pd.Series | None, vol_fn) -> tuple[pd.Series | None, pd.Series | None]:
    if anchor_close is None or anchor_close.empty:
        return None, None
    series = anchor_close.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    vol = vol_fn(series)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    indexed = vol.set_index("as_of")
    rv5 = indexed["rv5"] if "rv5" in indexed.columns else None
    return indexed["rv20"], rv5


def _layout_like_etf(fig) -> None:
    kept = []
    for ann in list(fig.layout.annotations or []):
        text = str(getattr(ann, "text", "") or "")
        if text in _DROP_TITLES:
            continue
        if text.startswith("图") and "K线" in text:
            ann.font = dict(size=12)
        kept.append(ann)
    fig.layout.annotations = tuple(kept)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        ),
        margin=dict(t=160, b=160, l=60, r=80),
    )


def _reflow_title(fig, apply_margin) -> None:
    import re

    text = str(fig.layout.title.text or "")

    def _wrap(match: re.Match) -> str:
        parts = [part.strip() for part in match.group(2).split("|") if part.strip() and "nan%" not in part]
        lines = [" | ".join(parts[i : i + 5]) for i in range(0, len(parts), 5)]
        return match.group(1) + "<br>".join(lines) + match.group(3)

    text = re.sub(
        r"(<span style='font-size:11px;color:#555'>)(.*?)(</span>)",
        _wrap,
        text,
        count=1,
        flags=re.DOTALL,
    )
    fig.update_layout(
        title=dict(text=text, x=0.01, xanchor="left", y=0.99, yanchor="top"),
        margin=dict(l=60, r=80, b=160),
    )
    apply_margin(fig, min_t=160)


def _append_fig12(path: Path, ohlcv: pd.DataFrame, code: str, name: str, deps: dict) -> None:
    indexed = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize()).sort_index()
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    states = deps["classify_six_states"](deps["compute_multi_scale_frame"](indexed["$close"].astype(float)))
    fig = deps["build_fig12_figure"](indexed, states, f"{code} {name}")
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig12_six_states")
    block = deps["build_fig12_block"](inner, state=deps["current_state"](states))
    raw = path.read_text(encoding="utf-8")
    path.write_text(deps["_inject_before_body_end"](raw, "<!-- ETF_STYLE_V1 -->\n" + block), encoding="utf-8")


def write_etf_html(
    out_dir: Path,
    code: str,
    name: str,
    history: pd.DataFrame,
    *,
    method: str,
    method_params: dict | None,
    train_cutoff: str,
    anchor_close: pd.Series | None,
    start: str,
    end: str,
    seg_df: pd.DataFrame,
    events: list[dict[str, Any]],
    open_mtm: dict | None,
    stats: dict,
) -> str:
    """Write ``regime_transition_{code}_{中文名}_{start}_{end}_adaptive.html``."""
    deps = _imports()
    history = _as_ohlcv(history)
    plot = history[
        (history["datetime"] >= pd.Timestamp(start)) & (history["datetime"] <= pd.Timestamp(end))
    ].copy()
    if open_mtm is not None and "regime" not in open_mtm:
        open_mtm = dict(open_mtm)
        open_mtm["regime"] = "up"
    close = plot.set_index(pd.to_datetime(plot["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")]
    from sw_daily.adaptive.stage import prepare_features

    labeled = prepare_features(history, method=method, method_params=method_params or {})
    policy = deps["evaluate_self_train_policy"](
        labeled["as_of"],
        labeled["$close"].to_numpy(float),
        labeled["regime"].to_numpy(object),
        train_cutoff=train_cutoff,
    )
    policy_frame = deps["policy_frame_from_eval"](policy)
    policy_frame = policy_frame[
        (pd.to_datetime(policy_frame["as_of"]) >= pd.Timestamp(start))
        & (pd.to_datetime(policy_frame["as_of"]) <= pd.Timestamp(end))
    ]
    rv20, rv5 = _anchor_rv(anchor_close, deps["compute_volatility_regime_frame"])
    need = None
    if rv20 is not None:
        need = deps["_need_frame_vs_anchor"](history, rv20, rv5_anchor=rv5, erp_ann=deps["ANCHOR_ERP_ANN"])
    fig = deps["build_candle_volume_vol_figure"](
        plot,
        deps["compute_volatility_regime_frame"](close),
        need_frame=need,
        policy_frame=policy_frame,
        policy_meta=deps["config_block"](policy),
        train_cutoff=train_cutoff,
        fair_path_ohlcv=history,
        fair_path_extra_trail_years=EXTRA_TRAILS,
        hover_asset_name=name,
        equity_anchor_label="沪深300",
    )
    fig.update_layout(title=f"{name}（{code}）")
    _layout_like_etf(fig)
    fig = deps["annotate_figure"](
        fig=fig,
        ohlcv=plot,
        seg_df=seg_df,
        aev=events,
        open_mtm=open_mtm,
        stats=stats,
        start_date=start,
        end_date=end,
        title_note=f"{method} · 只做多（进上涨买 / 离上涨卖）（{start}→{end}）",
    )
    _reflow_title(fig, deps["_apply_title_top_margin"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = html_filename(code, name, start, end)
    for old in out_dir.glob(f"regime_transition_{code}_*_adaptive.html"):
        if old.name != filename:
            old.unlink()
    path = out_dir / filename
    deps["write_adaptive_html"](fig, path)
    _append_fig12(path, plot, code, name, deps)
    return filename
