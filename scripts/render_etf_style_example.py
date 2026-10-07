"""Render one listing HTML with etf-daily's twelve-panel figure.

Example:
    python scripts/render_etf_style_example.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SW = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
_ETF = Path.home() / "etf-daily"
for root in (_ETF / "src", _SW / "src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from etf_daily.lib.adaptive_stage_common import annotate_figure, _apply_title_top_margin  # noqa: E402
from etf_daily.lib.plotly_html_autoscale import write_adaptive_html  # noqa: E402
from etf_daily.lib.regime_transition_plot import compute_volatility_regime_frame  # noqa: E402
from etf_daily.lib.self_train_policy import (  # noqa: E402
    config_block,
    evaluate_self_train_policy,
    policy_frame_from_eval,
)
from etf_daily.lib.six_states import (  # noqa: E402
    build_fig12_figure,
    classify_six_states,
    current_state,
)
from etf_daily.lib.vol_need_return_board import ANCHOR_ERP_ANN  # noqa: E402
from etf_daily.plots.add_fig12_six_states import (  # noqa: E402
    _inject_before_body_end,
    build_fig12_block,
)
from etf_daily.plots.plot_adaptive_stage_pool import _need_frame_vs_anchor  # noqa: E402
from etf_daily.plots.plot_regime_transition_example import (  # noqa: E402
    build_candle_volume_vol_figure,
)

from sw_daily.adaptive.stage import prepare_features, segments, simulate_hold_up  # noqa: E402
from sw_daily.paths import ANCHOR_CODE, FUND_QLIB_DIR  # noqa: E402
from sw_daily.pool.data_loading import load_ohlcv  # noqa: E402
from sw_daily.rules7.fair_path import compute_multi_scale_frame  # noqa: E402

OUT_DIR = Path.home() / "temp/sw/adaptive/20260930_from_listing"
CODE = "801951"
NAME = "煤炭开采"
END = "2026-09-30"
EXTRA = "1,0.5,0.25,1/12"
# A hole longer than this (the 2017-01-20 → 2021-12-13 break) starts a new chart.
GAP_BREAK_DAYS = 30


def _anchor_vol(end: str) -> tuple[pd.Series, pd.Series]:
    frame = load_ohlcv(ANCHOR_CODE, "2005-01-01", end, provider_uri=str(FUND_QLIB_DIR))
    close = frame.set_index(pd.to_datetime(frame["datetime"]).dt.normalize())["$close"]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    vol = compute_volatility_regime_frame(close)
    vol["as_of"] = pd.to_datetime(vol["as_of"]).dt.normalize()
    indexed = vol.set_index("as_of")
    return indexed["rv20"], indexed["rv5"]


def _fig12(path: Path, ohlcv: pd.DataFrame) -> None:
    indexed = ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize()).sort_index()
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    close = indexed["$close"].astype(float)
    states = classify_six_states(compute_multi_scale_frame(close))
    fig = build_fig12_figure(indexed, states, f"{CODE} {NAME}")
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id="fig12_six_states")
    block = build_fig12_block(inner, state=current_state(states))
    raw = path.read_text(encoding="utf-8")
    path.write_text(_inject_before_body_end(raw, block), encoding="utf-8")


def _resume_after_gap(frame: pd.DataFrame) -> pd.Timestamp:
    """First bar of the latest stretch. A long hole drops everything before it."""
    dates = pd.to_datetime(frame["datetime"]).sort_values().reset_index(drop=True)
    gap = dates.diff().dt.days
    breaks = gap[gap > GAP_BREAK_DAYS]
    if breaks.empty:
        return pd.Timestamp(dates.iloc[0])
    return pd.Timestamp(dates.iloc[int(breaks.index[-1])])


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep real sessions. Qlib inserts calendar days with no bar as NaN, which breaks the rails."""
    out = frame.copy()
    for col in ("$open", "$high", "$low", "$close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["$close", "$open", "$high", "$low"])
    out = out[out["$close"] > 0]
    return out.sort_values("datetime").reset_index(drop=True)


_DROP_TITLES = {
    "价格",
    "成交量",
    "波动率",
    "公平年化与20日差额",
    "公平年化与5日差额",
    "自训练策略净值",
}


def _layout_like_etf(fig) -> None:
    """Match the gold-ETF page: no upper subplot titles, legend under the figure."""
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


def _reflow_title(fig) -> None:
    """Wrap the segment strip so it stays in the top margin instead of one endless line."""
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
    _apply_title_top_margin(fig, min_t=160)


def main() -> None:
    cfg_path = Path.home() / "temp/sw/adaptive/20260930all_adaptive/configs/801951.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rv20, rv5 = _anchor_vol(END)
    ohlcv = load_ohlcv(CODE, "2005-01-01", END)
    ohlcv = ohlcv[pd.to_datetime(ohlcv["datetime"]) <= pd.Timestamp(END)].copy()
    ohlcv = _clean_ohlcv(ohlcv)
    start_ts = _resume_after_gap(ohlcv)
    ohlcv = ohlcv[pd.to_datetime(ohlcv["datetime"]) >= start_ts].reset_index(drop=True)
    start = start_ts.strftime("%Y-%m-%d")
    print(f"[plot] {CODE} {start} → {END} bars={len(ohlcv)}")
    px = prepare_features(ohlcv, method=cfg["method"], method_params=cfg.get("method_params") or {})
    window = px[(px.as_of >= start) & (px.as_of <= END)].reset_index(drop=True)
    close = window["$close"].to_numpy(float)
    dates = window.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = segments(window.regime.to_numpy(), dates, close)
    events, stats, open_mtm = simulate_hold_up(window)
    if open_mtm is not None:
        open_mtm = dict(open_mtm)
        open_mtm["regime"] = "up"
    policy = evaluate_self_train_policy(
        px["as_of"],
        px["$close"].to_numpy(float),
        px["regime"].to_numpy(object),
        train_cutoff=str(cfg.get("train_cutoff") or "2026-04-01"),
    )
    policy_frame = policy_frame_from_eval(policy)
    policy_frame = policy_frame[
        (pd.to_datetime(policy_frame["as_of"]) >= pd.Timestamp(start))
        & (pd.to_datetime(policy_frame["as_of"]) <= pd.Timestamp(END))
    ]
    need = _need_frame_vs_anchor(ohlcv, rv20, rv5_anchor=rv5, erp_ann=ANCHOR_ERP_ANN)
    fig = build_candle_volume_vol_figure(
        ohlcv,
        compute_volatility_regime_frame(
            ohlcv.set_index(pd.to_datetime(ohlcv["datetime"]).dt.normalize())["$close"]
        ),
        need_frame=need,
        policy_frame=policy_frame,
        policy_meta=config_block(policy),
        train_cutoff=str(cfg.get("train_cutoff") or "2026-04-01"),
        fair_path_ohlcv=ohlcv,
        fair_path_extra_trail_years=EXTRA,
        hover_asset_name=NAME,
        equity_anchor_label="沪深300",
    )
    fig.update_layout(title=f"{NAME}（{CODE}）")
    _layout_like_etf(fig)
    fig = annotate_figure(
        fig=fig,
        ohlcv=ohlcv,
        seg_df=seg_df,
        aev=events,
        open_mtm=open_mtm,
        stats=stats,
        start_date=start,
        end_date=END,
        title_note=(
            f"{cfg['method']} · 只做多（进上涨买 / 离上涨卖）（{start}→{END}）"
        ),
    )
    _reflow_title(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(END).strftime('%Y%m%d')}"
    out = OUT_DIR / f"regime_transition_{CODE}_{NAME}_{tag}_adaptive.html"
    for stale in OUT_DIR.glob(f"regime_transition_{CODE}_{NAME}_*_adaptive.html"):
        if stale != out:
            stale.unlink()
    write_adaptive_html(fig, out)
    _fig12(out, ohlcv)
    print(f"[OK] {out} bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
