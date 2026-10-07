"""rules7 checklist for one from_listing batch of Shenwan industry HTML.

Bars come from the sw qlib dataset. The SH510300 anchor comes from the fund
qlib dataset. HRP cluster membership comes from ``HRP_DIR``.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sw_daily.paths import ANCHOR_CODE, FUND_QLIB_DIR, HRP_DIR
from sw_daily.rules7 import rules7
from sw_daily.rules7.features import build_feature_frame_from_series, prepare_fig9_touch_frame

HTML_RE = re.compile(r"regime_transition_(\d{6})_")
HISTORY_START = "1990-01-01"
TYPE_ORDER = [rules7.TYPE_SMOOTH, rules7.TYPE_TREND, rules7.TYPE_HIVOL, rules7.TYPE_CYCLE, rules7.TYPE_SHORT]
ACTION_ORDER = {"买": 0, "卖": 1, "持有": 2, "观望": 3}
NEAR_WARN = 0.03
CLUSTER_COLS = ("聚类", "簇人数", "是否簇代表", "簇组", "ret20", "推荐买入")


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pct(v: Any, digits: int = 1) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.{digits}%}"


def _px(v: Any) -> str:
    x = _f(v)
    return "—" if x is None else f"{x:.4f}"


def _yn(b: Any) -> str:
    return "是" if bool(b) else ""


def _day(ts: Any) -> str:
    return "" if ts is None else pd.Timestamp(ts).strftime("%Y-%m-%d")


def _p(key: str, digits: int = 1) -> Callable[[dict], str]:
    return lambda s: _pct(s[key], digits)


def _x(key: str) -> Callable[[dict], str]:
    return lambda s: _px(s[key])


def _b(key: str) -> Callable[[dict], str]:
    return lambda s: _yn(s[key])


SNAP_COLS: list[tuple[str, Callable[[dict], str]]] = [
    ("图9下轨", _x("lo9")), ("距图9下轨", _p("dist_lo9")),
    ("图9近下轨(≤下轨×1.01)", _b("near_lo")), ("昨日近下轨", _b("near_lo_prev")),
    ("近下轨上跳沿", _b("near_lo_edge")),
    ("MA5", _x("ma5")), ("MA20", _x("ma20")),
    ("收盘<MA20", _b("below_ma20")), ("MA5<MA20", _b("ma5_below")),
    ("gap5", _p("gap5", 2)), ("gap5≤0", _b("gap5_le0")),
    ("gap20", _p("gap20", 2)), ("gap20≤0", _b("gap20_le0")),
    ("图10未触上轨", _b("fig10_not_hi")), ("辅助买像", _b("aux_buy")),
    ("aux_edge买点", _b("ae_buy")),
    ("图7折价", _p("gap7")), ("图8折价", _p("gap8")),
    ("图7或图8折价≤-8%", _b("disc_ok")), ("B1", _b("b1")),
    ("近60日最大40日涨幅", _p("run40")), ("快形态(≥20%)", _b("fast")),
    ("图9上轨", _x("hi9")), ("距图9上轨", _p("dist_hi9")),
    ("图9近上轨(≥上轨×0.99)", _b("near_hi")), ("近上轨上跳沿", _b("near_hi_edge")),
    ("收盘>MA20", _b("above_ma20")), ("MA5>MA20", _b("ma5_above")),
    ("gap5≥0", _b("gap5_ge0")), ("gap20≥0", _b("gap20_ge0")),
    ("辅助卖像", _b("aux_sell")), ("卖点A", _b("sell_a")),
    ("10日局部高点", _b("locmax")),
    ("图9带内位置", _p("pos9", 0)), ("图9位置≥80%", _b("pos9_ok")),
    ("图9斜率", _p("g9")), ("图9斜率>0", _b("g9_pos")),
    ("图9图10未触上轨", _b("no_touch_hi")),
    ("vol5分位", _p("vol5", 0)), ("vol5分位≥70%", _b("vol5_ok")),
    ("图7溢价≥5%", _b("prem7_ok")),
    ("图10带内位置", _p("pos10", 0)), ("图10位置≥90%", _b("pos10_ok")),
    ("带内拉伸B", _b("stretch")), ("图9卖点", _b("fig9_sell")),
    ("图8上轨", _x("hi8")), ("距图8上轨", _p("dist_hi8")),
    ("图8近上轨(≥上轨×0.99)", _b("near_hi8")), ("图8卖点", _b("fig8_sell")),
    ("图7斜率", _p("g7")), ("图7斜率>0", _b("g7_pos")),
    ("图8斜率", _p("g8")), ("图8斜率>0", _b("g8_pos")),
    ("图9位置10日变化", _p("pos9_10", 0)), ("10日变化≥30pp", _b("pos9_10_ok")),
    ("未触图9上轨", _b("not_touch9hi")),
    ("图10斜率", _p("g10")), ("图10斜率>图9", _b("g10_gt_g9")),
    ("G1今日", _b("g1")),
]
HEAD_COLS = (
    "代码", "名称", "T收盘", "类型", "类型来源", "买规则", "卖规则", "G1后不卖", "回落幅度",
    "今日状态", "今日动作", "卖出原因", "触发指标", "买入信号", "买点价", "买点状态", "卖出信号",
    *CLUSTER_COLS,
)
POS_COLS = (
    "买入日", "买入价", "持仓最高收盘", "持仓浮盈", "回落止损价", "距止损",
    "回落触发", "本笔G1已触发", "卖点被G1屏蔽",
)
COLS = (*HEAD_COLS, *(c for c, _ in SNAP_COLS), *POS_COLS)

THRESHOLDS = [
    ("图9近下轨", "收盘 ≤ 图9下轨 × 1.01", "aux_edge买点、B1"),
    ("近下轨上跳沿", "今天近下轨、昨天不是", "aux_edge买点"),
    ("辅助买像", "收盘 < MA20、MA5 < MA20、gap5 ≤ 0、gap20 ≤ 0、图10 未触上轨", "aux_edge买点"),
    ("B1", "图7 或图8 折价 ≤ −8%，且图9近下轨", "买（规则含 B1 时）"),
    ("快形态", "近 60 日任一 40 日涨幅 ≥ 20%", "规则含“快形态不买”时拦买"),
    ("图9近上轨", "收盘 ≥ 图9上轨 × 0.99", "卖点A"),
    ("辅助卖像", "收盘 > MA20、MA5 > MA20、gap5 ≥ 0、gap20 ≥ 0", "卖点A、图8卖点、带内拉伸B"),
    ("卖点A", "图9近上轨上跳沿，且辅助卖像", "图9卖点"),
    ("带内拉伸B", "辅助卖像、10日局部高点、图9位置 ≥ 80%、图9斜率 > 0、图9图10未触上轨、"
                 "vol5分位 ≥ 70%、图7溢价 ≥ 5% 或图10位置 ≥ 90%", "图9卖点、图8卖点"),
    ("图8卖点", "收盘 ≥ 图8上轨 × 0.99 且辅助卖像，或带内拉伸B", "卖（规则为图8卖时）"),
    ("G1", "图7、图8 斜率 > 0，图9位置 10 日升 ≥ 30pp 且 ≥ 80%，未触图9上轨，图10斜率 > 图9斜率",
     "持仓中触发后本笔不再按图9/图8 卖"),
    ("回落触发", "收盘 ≤ 持仓最高收盘 × (1 − 回落幅度)", "卖（规则设回落时）"),
    ("买点价", "收盘 ≤ 图9下轨 × 1.01", "空仓行的买点价列，不是新的成交规则"),
]


def as_of_from_listing(path: Path) -> str | None:
    digits = ""
    for char in path.name:
        if char.isdigit():
            digits += char
            if len(digits) == 8:
                break
        else:
            digits = ""
    if len(digits) != 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listing-dir", type=Path, required=True)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: date in the directory name)")
    p.add_argument("--anchor-code", default=ANCHOR_CODE)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--cluster-csv", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--rewrite-from-csv", action="store_true")
    return p.parse_args(argv)


def default_cluster_csv(as_of: pd.Timestamp | str) -> Path | None:
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    pack = HRP_DIR / tag
    for name in (
        f"cluster_representatives_{tag}.csv",
        f"cluster_representatives_{tag}_d080.csv",
    ):
        cand = pack / name
        if cand.is_file():
            return cand
    return None


def load_cluster_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).expanduser().is_file():
        return None
    df = pd.read_csv(Path(path).expanduser())
    if df.empty or "code" not in df.columns:
        return None
    out = df.copy()
    out["code"] = out["code"].astype(str).str.upper()
    if "is_representative" in out.columns:
        out["is_representative"] = out["is_representative"].map(
            lambda x: x is True or str(x).strip().lower() in {"true", "1", "yes"}
        )
    return out


def _parse_pct_cell(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    s = str(v).strip()
    if not s or s in {"—", "-", "nan", "None"}:
        return None
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def attach_cluster_recommendations(df: pd.DataFrame, cluster: pd.DataFrame | None) -> pd.DataFrame:
    """Add HRP cluster columns; mark one 推荐买入 per cluster among today's buys."""
    out = df.copy()
    for col in CLUSTER_COLS:
        out[col] = ""
    out["_cluster"] = pd.NA
    out["_is_rep"] = False
    out["_ret20"] = np.nan
    out["_recommend"] = False
    if out.empty or cluster is None or cluster.empty:
        return out

    mem = cluster.copy()
    mem["code"] = mem["code"].astype(str).str.upper()
    mem["_abs"] = pd.to_numeric(mem.get("ret20_pct"), errors="coerce").abs()
    if "is_representative" not in mem.columns:
        mem["is_representative"] = False
    mem = mem.sort_values(
        ["code", "is_representative", "_abs"], ascending=[True, False, False], kind="mergesort"
    ).drop_duplicates("code", keep="first")
    by_code = mem.set_index("code")
    codes = out["代码"].astype(str).str.upper()
    out["_cluster"] = codes.map(by_code["cluster"] if "cluster" in by_code.columns else {})
    out["_is_rep"] = codes.map(by_code["is_representative"]).fillna(False).astype(bool)
    out["_ret20"] = codes.map(by_code["ret20_pct"] if "ret20_pct" in by_code.columns else {})
    n_map = by_code["n_members"] if "n_members" in by_code.columns else None
    g_map = by_code["group"] if "group" in by_code.columns else None
    out["聚类"] = out["_cluster"].map(lambda x: "" if pd.isna(x) else str(int(x)))
    out["簇人数"] = (
        codes.map(n_map).map(lambda x: "" if pd.isna(x) else str(int(x))) if n_map is not None else ""
    )
    out["是否簇代表"] = out["_is_rep"].map(lambda b: "是" if b else "")
    if g_map is not None:
        out["簇组"] = codes.map(g_map).map(
            lambda g: {"up": "涨", "down": "跌"}.get(str(g).lower(), str(g) if pd.notna(g) else "")
        )
    out["ret20"] = out["_ret20"].map(lambda x: "" if pd.isna(x) else f"{float(x):+.1f}%")

    buys = out[out["今日动作"] == "买"].copy()
    if buys.empty:
        out["推荐买入"] = ""
        return out
    picks: list[Any] = []
    clustered = buys[buys["_cluster"].notna()]
    for _, sub in clustered.groupby("_cluster", sort=True):
        sub = sub.copy()
        sub["_abs"] = sub["_ret20"].abs()
        sub["_dlo"] = pd.to_numeric(sub.get("_dist_lo9"), errors="coerce")
        sub = sub.sort_values(["_is_rep", "_abs", "_dlo"], ascending=[False, False, True], kind="mergesort")
        picks.append(sub.index[0])
    picks.extend(buys[buys["_cluster"].isna()].index.tolist())
    out.loc[picks, "_recommend"] = True
    out["推荐买入"] = out["_recommend"].map(lambda b: "是" if b else "")
    out.loc[out["今日动作"] != "买", "推荐买入"] = ""
    out.loc[out["今日动作"] != "买", "_recommend"] = False
    return out


def bars_from_qlib(code: str, start: str, end: str, provider_uri: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    from sw_daily.pool.data_loading import load_ohlcv

    raw = load_ohlcv(code, start, end, provider_uri=provider_uri)
    if raw.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    raw = raw.copy()
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    raw = raw.drop_duplicates("datetime", keep="last").set_index("datetime").sort_index()
    close = raw["$close"].astype(float)
    close.index.name = "date"
    return close, raw


def row_from_frame(code: str, name: str, frame: pd.DataFrame, html_name: str) -> dict[str, Any]:
    res = rules7.evaluate(code, frame)
    spec, st, snap = res["spec"], res["state"], res["snapshot"]
    px = snap["px"]
    fill_px = st["fill_px"]
    trig = rules7.triggered(snap)
    row: dict[str, Any] = {
        "代码": code,
        "名称": name,
        "T收盘": _px(px),
        "类型": res["type"],
        "类型来源": res["source"],
        "买规则": spec.buy_text(),
        "卖规则": spec.sell_text(),
        "G1后不卖": _yn(spec.g1_hold),
        "回落幅度": spec.trail_text(),
        "今日状态": "持仓" if st["holding"] else "空仓",
        "今日动作": st["action"],
        "卖出原因": st["why"],
        "触发指标": "；".join(trig),
        "买入信号": _yn(snap["buy_signal"]),
        "卖出信号": _yn(snap["sell_signal"]),
    }
    quote, quote_status = (None, "")
    if not st["holding"]:
        quote, quote_status = rules7.flat_buy_quote(snap, spec)
    row["买点价"] = "" if st["holding"] or quote is None else _px(quote)
    row["买点状态"] = "" if st["holding"] else quote_status
    row["_html"] = html_name
    row.update({col: fmt(snap) for col, fmt in SNAP_COLS})
    row.update({
        "买入日": _day(st["buy_date"]),
        "买入价": _px(fill_px),
        "持仓最高收盘": _px(st["peak"]),
        "持仓浮盈": _pct(px / fill_px - 1 if fill_px else None),
        "回落止损价": _px(st["stop"]),
        "距止损": _pct(snap["dist_stop"]),
        "回落触发": _yn(snap["trail_hit"]),
        "本笔G1已触发": _yn(st["g1_seen"]),
        "卖点被G1屏蔽": _yn(snap["sell_masked_by_g1"]),
        "_action_rank": ACTION_ORDER.get(st["action"], 9),
        "_n_trig": len(trig),
        "_type": res["type"],
        "_holding": st["holding"],
        "_dist_lo9": snap["dist_lo9"],
        "_dist_stop": snap["dist_stop"],
        **{f"_t_{k}": bool(snap.get(k)) for k in rules7.TRIGGER_LABELS},
    })
    return row


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    as_of = pd.Timestamp(payload["as_of"]).normalize()
    close = payload["close"]
    frame = build_feature_frame_from_series(close, payload["ohlcv"], payload["anchor"], lookback=10)
    frame = prepare_fig9_touch_frame(frame)
    frame = frame.loc[frame.index <= as_of]
    if frame.empty:
        raise ValueError(f"no bars on/before {as_of.date()}")
    if pd.Timestamp(frame.index[-1]).normalize() != as_of:
        raise ValueError(f"last bar {pd.Timestamp(frame.index[-1]).date()} != as-of {as_of.date()}")
    return row_from_frame(payload["code"], payload["name"], frame, payload["html"])


def _md_table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    if df.empty:
        return ["（无）", ""]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) if c in r.index else "—" for c in cols) + " |")
    return lines + [""]


def _html_for(code: str, html_name: Any, listing: Path) -> str:
    name = str(html_name or "").strip()
    if name and name.lower() not in {"nan", "none"}:
        return name
    matches = sorted(listing.glob(f"regime_transition_{code}_*_adaptive.html"))
    return matches[0].name if matches else ""


def _gap_to_buy(px_cell: Any, buy_cell: Any) -> float:
    px, buy = _f(px_cell), _f(buy_cell)
    if px is None or buy is None or buy == 0:
        return float("inf")
    return px / buy - 1.0


def _format_flat_buy_section(flat: pd.DataFrame, listing: Path) -> list[str]:
    n_buy = int(flat["今日动作"].astype(str).eq("买").sum()) if "今日动作" in flat.columns else 0
    lines = [
        "## 空仓买点价",
        "",
        f"今天买入信号 **{n_buy}** 只。买入信号为 0 则今天不买。",
        "买点价 = 图9下轨 × 1.01，只对空仓行填写，不是盘中委托。成交仍是信号出现之后的下一根收盘。",
        "已在带内的下一根收盘若仍在带内，不是新的 aux_edge。",
        "",
    ]
    if flat.empty:
        return lines + ["（无）", ""]
    show = flat.copy()
    show["_gap"] = [_gap_to_buy(px, buy) for px, buy in zip(show["T收盘"], show.get("买点价", pd.Series(index=show.index)))]
    show = show.sort_values("_gap", kind="mergesort")
    links = []
    for _, r in show.iterrows():
        html_name = _html_for(str(r["代码"]), r["_html"] if "_html" in show.columns else "", listing)
        links.append(f"[{r['代码']}]({html_name})" if html_name else str(r["代码"]))
    show = show.copy()
    show["代码"] = links
    lines += _md_table(show, ["代码", "名称", "买点价", "买点状态", "T收盘", "图9下轨", "距图9下轨"])
    return lines


def _format_hrp_cluster_section(df: pd.DataFrame, *, cluster_path: Path | None) -> list[str]:
    lines = [
        "## 按 HRP 簇（相似去重）",
        "",
        "默认读 `info_dir/hrp/<YYYYMMDD>/cluster_representatives_<YYYYMMDD>.csv`。同簇标的走势接近，买入时只取一只推荐。",
        "",
    ]
    if cluster_path is None or "_cluster" not in df.columns or df["_cluster"].isna().all():
        return lines + [f"（未挂上聚类{f': {cluster_path}' if cluster_path else ''}）", ""]
    lines += [
        "| 簇 | 人数 | 代表 | ret20 | 组内清单状态 | 今日买入 | 推荐 |",
        "| ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for cid, sub in df[df["_cluster"].notna()].groupby("_cluster", sort=True):
        n_members = ""
        if "簇人数" in sub.columns:
            nm = sub["簇人数"].replace("", pd.NA).dropna()
            if not nm.empty:
                n_members = str(nm.iloc[0])
        reps = sub[sub["_is_rep"]]
        if reps.empty:
            rep_txt, ret_txt = "—", "—"
        else:
            r0 = reps.assign(_abs=reps["_ret20"].abs()).sort_values("_abs", ascending=False).iloc[0]
            rep_txt = f"{r0['代码']} {r0['名称']}"
            ret_txt = r0.get("ret20") or "—"
        members = "、".join(
            f"{r['代码']}{r['名称']}({r['今日状态']}/{r['今日动作']})"
            for _, r in sub.sort_values("代码").iterrows()
        )
        buy_codes = "、".join(sub.loc[sub["今日动作"] == "买", "代码"].astype(str))
        rec_codes = "、".join(sub.loc[sub.get("推荐买入", pd.Series(dtype=str)).fillna("").eq("是"), "代码"].astype(str))
        lines.append(
            f"| {int(cid)} | {n_members or sub['代码'].nunique()} | {rep_txt} | {ret_txt} | "
            f"{members} | {buy_codes or '—'} | {rec_codes or '—'} |"
        )
    unmatched = df[df["_cluster"].isna()]
    if not unmatched.empty:
        lines += ["", f"未进聚类（{len(unmatched)} 只）：" + "、".join(f"{r['代码']} {r['名称']}" for _, r in unmatched.iterrows()) + "。"]
    lines.append("")
    return lines


def write_markdown(
    path: Path,
    df: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    listing: Path,
    errors: list[dict[str, str]],
    cluster_path: Path | None = None,
) -> None:
    n = len(df)
    holding = df[df["_holding"]]
    lines = [
        f"# 规则当天指标清单 {as_of.date()}",
        "",
        "> **身份：研究清单，不改生产。** 规则与 etf-daily 的 7 只研究相同，申万行业全部按自身历史自动分型，没有专属覆盖。"
        "持仓状态从第一根行情推算，只写当天结果。相似标的按 HRP 聚类去重：同簇今日买入信号只标一只 **推荐买入**。",
        "",
        f"目录：`{listing}`。as-of **{as_of.date()}** 收盘出信号，次日收盘成交。"
        f"有效 **{n}** 只，其中持仓 **{len(holding)}** 只、空仓 **{n - len(holding)}** 只。",
    ]
    if cluster_path is not None:
        lines.append(f"聚类：`{cluster_path}`。")
    if errors:
        lines.append("跳过：" + "；".join(f"{e['code']} {e['name']}（{e['error']}）" for e in errors) + "。")
    lines.append("")

    buys = df[df["今日动作"] == "买"]
    rec = buys[buys.get("推荐买入", pd.Series(dtype=str)).fillna("").eq("是")] if "推荐买入" in buys.columns else buys.iloc[0:0]
    sells = df[df["今日动作"] == "卖"]
    lines += [f"## 今日推荐买入（同簇只留一只）：{len(rec)} 只", ""]
    lines += _md_table(rec, ["代码", "名称", "聚类", "簇人数", "是否簇代表", "簇组", "ret20", "类型", "买规则", "T收盘", "距图9下轨", "触发指标"])
    lines += [f"## 今日买入信号（去重前）：{len(buys)} 只", ""]
    lines += _md_table(buys, ["代码", "名称", "推荐买入", "聚类", "是否簇代表", "类型", "买规则", "T收盘", "距图9下轨", "触发指标"])
    lines += [f"## 今日卖出（次日收盘卖）：{len(sells)} 只", ""]
    lines += _md_table(sells, ["代码", "名称", "类型", "卖出原因", "买入日", "买入价", "T收盘", "持仓浮盈", "触发指标"])
    lines += _format_hrp_cluster_section(df, cluster_path=cluster_path)

    lines += ["## 关键指标触发统计", "", "| 指标 | 达标只数 | 标的 |", "| --- | ---: | --- |"]
    for key, label in rules7.TRIGGER_LABELS.items():
        hit = df[df[f"_t_{key}"]] if f"_t_{key}" in df.columns else df.iloc[0:0]
        names = "、".join(f"{r['代码']} {r['名称']}" for _, r in hit.iterrows())
        lines.append(f"| {label} | {len(hit)} | {names} |")
    lines.append("")

    hold = holding.sort_values("_dist_stop", na_position="last") if "_dist_stop" in holding.columns else holding
    lines += [f"## 持仓中：{len(hold)} 只", "", "按距回落止损从近到远排。", ""]
    lines += _md_table(hold, ["代码", "名称", "聚类", "是否簇代表", "类型", "今日动作", "买入日", "买入价", "T收盘", "持仓浮盈", "回落止损价", "距止损", "本笔G1已触发", "卖点被G1屏蔽", "触发指标"])

    flat = df[~df["_holding"]].copy()
    lines += _format_flat_buy_section(flat, listing)
    near_stop = holding[holding["_dist_stop"] <= NEAR_WARN].sort_values("_dist_stop") if "_dist_stop" in holding.columns else holding.iloc[0:0]
    lines += [f"## 接近触发（{NEAR_WARN:.0%} 以内）", "", f"### 持仓、距回落止损 ≤ {NEAR_WARN:.0%}：{len(near_stop)} 只", ""]
    lines += _md_table(near_stop, ["代码", "名称", "聚类", "类型", "T收盘", "回落止损价", "距止损", "持仓浮盈"])

    counts = df["_type"].value_counts() if "_type" in df.columns else pd.Series(dtype=int)
    lines += ["## 类型与规则", "", "| 类型 | 买 | 卖 | G1 后不卖 | 回落卖出 | 只数 |", "| --- | --- | --- | --- | --- | ---: |"]
    for t in TYPE_ORDER:
        s = rules7.TYPE_PARAMS[t]
        lines.append(
            f"| {t} | {s.buy_text()} | {s.sell_text()} | {'是' if s.g1_hold else '否'} | {s.trail_text()} | {int(counts.get(t, 0))} |"
        )
    lines += ["", "每只实际所用规则见 CSV 的“买规则 / 卖规则 / 回落幅度”。", ""]
    lines += ["## 指标门槛", "", "| 指标 | 门槛 | 用于 |", "| --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in THRESHOLDS]
    lines += [
        "",
        "## 注意",
        "",
        "- 今日动作看的是该标的所用规则：买入信号只在空仓时生效，卖出信号只在持仓且本笔未触发 G1 时生效（G1 后不卖的规则）。",
        "- 买点价 = 图9下轨 × 1.01，只对空仓行填写。不是盘中委托。成交仍是信号日之后的下一根收盘。",
        "- 已在带内的下一根收盘若仍在带内，不是新的 aux_edge。",
        "- **推荐买入**：同一 HRP 簇内若多只同时出现买入信号，只标一只（优先簇代表，其次 |ret20| 更大，再次更贴近图9下轨）。未进聚类的买入信号各自保留。",
        "- 历史不足的标的归类不可靠，信号只当提示。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_helper_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_holding"] = out["今日状态"].astype(str).eq("持仓")
    out["_action_rank"] = out["今日动作"].map(ACTION_ORDER).fillna(9).astype(int)
    out["_n_trig"] = out["触发指标"].fillna("").astype(str).map(
        lambda s: 0 if not s.strip() else len([x for x in s.split("；") if x.strip()])
    )
    if "_type" not in out.columns:
        out["_type"] = out["类型"].astype(str)
    out["_dist_lo9"] = out["距图9下轨"].map(_parse_pct_cell).map(lambda x: np.nan if x is None else x / 100.0)
    out["_dist_stop"] = out["距止损"].map(_parse_pct_cell).map(lambda x: np.nan if x is None else x / 100.0)
    for key, label in rules7.TRIGGER_LABELS.items():
        col = f"_t_{key}"
        if col not in out.columns:
            out[col] = out["触发指标"].fillna("").astype(str).map(
                lambda s, lab=label: lab in {x.strip() for x in s.split("；")}
            )
    return out


def list_listing_html(listing: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for path in sorted(listing.glob("regime_transition_*_adaptive.html")):
        match = HTML_RE.search(path.name)
        if match:
            found.append((match.group(1), path))
    return found


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    listing = Path(args.listing_dir).expanduser().resolve()
    as_of_text = args.as_of or as_of_from_listing(listing)
    if not as_of_text:
        print("pass --as-of YYYY-MM-DD")
        return 2
    as_of = pd.Timestamp(as_of_text).normalize()
    tag = as_of.strftime("%Y%m%d")
    out = args.out or (listing / f"rules7_checklist_{tag}.csv")
    out_md = out.with_suffix(".md")
    cluster_path = args.cluster_csv or default_cluster_csv(as_of)
    cluster = load_cluster_frame(cluster_path)
    if cluster_path is not None:
        print(f"[cluster] {cluster_path}" + (" (loaded)" if cluster is not None else " (missing)"), flush=True)

    if args.rewrite_from_csv:
        if not out.is_file():
            print(f"missing checklist CSV: {out}")
            return 2
        df = _ensure_helper_cols(pd.read_csv(out, dtype=str).fillna(""))
        df = df.drop(columns=[c for c in CLUSTER_COLS if c in df.columns], errors="ignore")
        df = attach_cluster_recommendations(df, cluster)
        df.loc[:, [c for c in COLS if c in df.columns]].to_csv(out, index=False, encoding="utf-8-sig")
        write_markdown(out_md, df, as_of=as_of, listing=listing, errors=[], cluster_path=cluster_path)
        print(f"[OK] {out}", flush=True)
        print(f"[OK] {out_md}", flush=True)
        return 0

    from sw_daily.pool.selector import load_sw_universe

    _table, name_map = load_sw_universe()
    htmls = list_listing_html(listing)
    print(f"[anchor] {args.anchor_code} from {FUND_QLIB_DIR}", flush=True)
    anchor_close, _anchor_ohlcv = bars_from_qlib(
        str(args.anchor_code), HISTORY_START, as_of.strftime("%Y-%m-%d"), provider_uri=str(FUND_QLIB_DIR)
    )
    if anchor_close.empty:
        print(f"anchor {args.anchor_code} has no bars", flush=True)
        return 2

    payloads: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for code, html in htmls:
        close, ohlcv = bars_from_qlib(code, HISTORY_START, as_of.strftime("%Y-%m-%d"))
        if close.empty:
            errors.append({"code": code, "name": name_map.get(code, code), "error": "no bars"})
            print(f"[error] {code}: no bars", flush=True)
            continue
        payloads.append({
            "code": code,
            "name": name_map.get(code, code),
            "html": html.name,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "close": close,
            "ohlcv": ohlcv,
            "anchor": anchor_close,
        })

    records: list[dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    print(f"[INFO] {len(payloads)} names  jobs={jobs}  as-of {as_of.date()}", flush=True)

    def _collect(payload: dict[str, Any], row_or_exc: dict[str, Any] | Exception) -> None:
        if isinstance(row_or_exc, Exception):
            errors.append({"code": payload["code"], "name": payload["name"], "error": f"{type(row_or_exc).__name__}: {row_or_exc}"})
            print(f"[error] {payload['code']} {payload['name']}: {row_or_exc}", flush=True)
            return
        records.append(row_or_exc)
        print(f"[scan] {payload['code']} {row_or_exc['今日状态']} -> {row_or_exc['今日动作']}  {row_or_exc['触发指标']}", flush=True)

    if jobs == 1:
        for payload in payloads:
            try:
                _collect(payload, _worker(payload))
            except Exception as exc:  # noqa: BLE001
                _collect(payload, exc)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, payload): payload for payload in payloads}
            for fut in as_completed(futures):
                payload = futures[fut]
                try:
                    _collect(payload, fut.result())
                except Exception as exc:  # noqa: BLE001
                    _collect(payload, exc)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["_action_rank", "_n_trig", "代码"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        df = attach_cluster_recommendations(df, cluster)
    out_df = df.loc[:, list(COLS)] if not df.empty else pd.DataFrame(columns=list(COLS))
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {out}", flush=True)
    if errors:
        err_path = out.with_name(out.stem + "_skipped.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {err_path} ({len(errors)} skipped)", flush=True)
    if not df.empty:
        write_markdown(out_md, df, as_of=as_of, listing=listing, errors=errors, cluster_path=cluster_path)
        print(f"[OK] {out_md}", flush=True)
        for act in ("买", "卖"):
            sel = df[df["今日动作"] == act]
            print(f"  今日{act}: {len(sel)}" + "".join(f"\n    {r['代码']} {r['名称']} ({r['触发指标']})" for _, r in sel.iterrows()))
        rec = df[df["推荐买入"].fillna("").eq("是")]
        print(f"  推荐买入: {len(rec)}" + "".join(f"\n    {r['代码']} {r['名称']} 簇{r.get('聚类', '')}" for _, r in rec.iterrows()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
