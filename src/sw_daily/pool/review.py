"""Quality review for ``sw_cluster_mapping_selected.csv``.

Ported from ``etf_daily/pool/builder/daily_review.py``. Checks whether each
selected industry is still the best recent representative within its cluster,
lists clusters with no selected representative, and writes a Markdown report.

The ETF version compares a hand-curated pool against a machine snapshot and
classifies whitelist/blacklist exclusion reasons; the Shenwan pool has no
hand pool, so the review is quality-only against the generated mapping CSVs.

Usage::

    sw-daily cluster-review
    sw-daily cluster-review --future-end 2026-10-07 --windows 5,20,60
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

from sw_daily.paths import POOL_DIR

from . import clustering as C
from .data_loading import load_close_volume
from .selector import CSV_OUT, CSV_SELECTED_OUT, load_sw_universe

DEFAULT_WINDOWS = "5,20,60"
DEFAULT_MIN_REGRET = 0.01
DEFAULT_MIN_CLUSTER_N = 2
DEFAULT_TOP_N = 15
DEFAULT_RELATED_SELECTED_TOP_K = 3

REASON_KIND_CN = {
    "insufficient_history": "历史数据不足",
    "missing_from_qlib_data": "qlib 无行情",
    "mixed_diagnostics": "混合剔除原因",
    "target_count_not_selected": "目标池裁剪（multi_rep 未选中该簇）",
}

MISSED_MD_COLUMN_CN = {
    "cluster": "簇编号",
    "n": "簇规模",
    "members_preview": "簇成员预览",
    "best_code": "最强代码",
    "best_name": "最强中文名",
    "best_20d_ret_pct": "最强近20日收益",
    "reason": "原因说明",
    "related_selected_clusters": "相关已选簇（选入标的/20日收益）",
}

REP_LAG_MD_COLUMN_CN = {
    "cluster": "簇编号",
    "selected_code": "入选代码",
    "selected_name": "入选中文名",
    "best_code": "最强代码",
    "best_name": "最强中文名",
    "selected_pct": "入选近20日收益",
    "best_pct": "最强近20日收益",
    "regret_pct": "落后幅度",
    "rank_20d": "簇内20日排名",
    "likely_due_to_low_volume": "可能因流动性保留",
}


def parse_windows(windows_text: str) -> list[int]:
    windows = [int(tok.strip()) for tok in str(windows_text).split(",") if tok.strip()]
    windows = sorted(set(windows))
    if not windows:
        raise ValueError("No valid windows were provided")
    return windows


def window_bounds(future_end: str, windows: list[int]) -> str:
    """Trailing-window start: enough business days before the anchor."""
    max_window = max(windows) if windows else 20
    return (pd.Timestamp(future_end) - pd.tseries.offsets.BDay(max_window + 40)).strftime("%Y-%m-%d")


def _normalize_selected(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"}).fillna(False)


def load_mapping_csv(mapping_csv: str | Path) -> pd.DataFrame:
    if not Path(mapping_csv).is_file():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}; run: sw-daily pool")
    df = pd.read_csv(mapping_csv)
    required = {"code", "cluster", "selected", "reason"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {mapping_csv}: {sorted(missing)}")
    df = df.copy()
    df["code"] = df["code"].astype(str).str.strip()
    df = df[df["code"] != ""].copy()
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
    df = df[df["cluster"].notna()].copy()
    df["cluster"] = df["cluster"].astype(int)
    df["selected"] = _normalize_selected(df["selected"])
    for col in ("name", "n"):
        if col not in df.columns:
            df[col] = "" if col == "name" else np.nan
    return df


def load_selected_codes(selected_csv: str | Path) -> list[str]:
    if not Path(selected_csv).is_file():
        raise FileNotFoundError(f"Selected CSV not found: {selected_csv}; run: sw-daily pool")
    df = pd.read_csv(selected_csv)
    col = "code" if "code" in df.columns else df.columns[0]
    codes = df[col].astype(str).str.strip().loc[lambda s: s != ""].tolist()
    return list(dict.fromkeys(codes))


def lookup_name(code: str, name_map: dict[str, str]) -> str:
    return str(name_map.get(str(code).strip(), "")).strip()


def format_code_with_cn(code: str, name_map: dict[str, str]) -> str:
    code_s = str(code).strip()
    cn = lookup_name(code_s, name_map)
    return f"{code_s} {cn}" if cn else code_s


def _split_reason_tags(reason_series: pd.Series) -> list[str]:
    tags: list[str] = []
    for raw in reason_series.fillna("").astype(str):
        for part in raw.split("|"):
            part = part.strip()
            if part:
                tags.append(part)
    return tags


def _infer_missed_cluster_reason(grp: pd.DataFrame) -> str:
    """Classify why a cluster has no selected representative."""
    tags = _split_reason_tags(grp["reason"]) if "reason" in grp.columns else []
    if any("insufficient_history" in tag for tag in tags):
        return "insufficient_history"
    if any("missing_from_qlib_data" in tag for tag in tags):
        return "missing_from_qlib_data"
    if tags:
        return "mixed_diagnostics"
    return "target_count_not_selected"


def format_missed_reason_detail(reason_kind: str) -> str:
    return REASON_KIND_CN.get(str(reason_kind), str(reason_kind))


def format_members_preview_cn(preview: str, name_map: dict[str, str]) -> str:
    if preview is None or (isinstance(preview, float) and np.isnan(preview)):
        return ""
    return "；".join(
        format_code_with_cn(part.strip(), name_map)
        for part in str(preview).split("|")
        if str(part).strip()
    )


def format_missed_df_for_markdown(missed_df: pd.DataFrame, name_map: dict[str, str], top_n: int) -> pd.DataFrame:
    if missed_df.empty:
        return missed_df
    show = missed_df.head(top_n).copy()
    if "best_20d_ret" in show.columns:
        show["best_20d_ret_pct"] = show["best_20d_ret"].map(
            lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
        )
    if "members_preview" in show.columns:
        show["members_preview"] = show["members_preview"].map(
            lambda x: format_members_preview_cn(x, name_map)
        )
    if "best_code" in show.columns:
        show["best_code"] = show["best_code"].map(lambda x: str(x).strip() if pd.notna(x) else "")
        show["best_name"] = show["best_code"].map(lambda c: lookup_name(c, name_map) if c else "")
    if "reason_kind" in show.columns:
        show["reason"] = show["reason_kind"].map(format_missed_reason_detail)
    display_cols = [
        c
        for c in [
            "cluster",
            "n",
            "members_preview",
            "best_code",
            "best_name",
            "best_20d_ret_pct",
            "reason",
            "related_selected_clusters",
        ]
        if c in show.columns
    ]
    return show[display_cols].rename(columns=MISSED_MD_COLUMN_CN)


def format_rep_lag_df_for_markdown(rep_lag_df: pd.DataFrame, name_map: dict[str, str], top_n: int) -> pd.DataFrame:
    if rep_lag_df.empty:
        return rep_lag_df
    show = rep_lag_df.head(top_n).copy()
    pct_map = {
        "selected_20d_ret": "selected_pct",
        "best_20d_ret": "best_pct",
        "regret_20d": "regret_pct",
    }
    for src, dst in pct_map.items():
        if src in show.columns:
            show[dst] = show[src].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-")
    if "selected_code" in show.columns:
        show["selected_code"] = show["selected_code"].map(lambda x: str(x).strip() if pd.notna(x) else "")
        show["selected_name"] = show["selected_code"].map(lambda c: lookup_name(c, name_map) if c else "")
    if "best_code" in show.columns:
        show["best_code"] = show["best_code"].map(lambda x: str(x).strip() if pd.notna(x) else "")
        show["best_name"] = show["best_code"].map(lambda c: lookup_name(c, name_map) if c else "")
    display_cols = [
        c
        for c in [
            "cluster",
            "selected_code",
            "selected_name",
            "best_code",
            "best_name",
            "selected_pct" if "selected_pct" in show.columns else "selected_20d_ret",
            "best_pct" if "best_pct" in show.columns else "best_20d_ret",
            "regret_pct" if "regret_pct" in show.columns else "regret_20d",
            "rank_20d",
            "likely_due_to_low_volume",
        ]
        if c in show.columns
    ]
    return show[display_cols].rename(columns=REP_LAG_MD_COLUMN_CN)


def compute_trailing_return_metrics(close: pd.DataFrame, as_of: str, windows: list[int]) -> pd.DataFrame:
    """Trailing returns over N trading days ending on or before as_of."""
    as_of_ts = pd.Timestamp(as_of)
    close = close.sort_index()
    end_candidates = np.where(close.index <= as_of_ts)[0]
    if len(end_candidates) == 0:
        raise RuntimeError(f"No price found on or before as_of={as_of}")
    end_pos = int(end_candidates[-1])
    end_prices = close.iloc[end_pos]

    rows = []
    for code in close.columns:
        row = {"code": code}
        end_price = end_prices.get(code, np.nan)
        if pd.isna(end_price) or end_price == 0:
            rows.append(row)
            continue
        series = close[code]
        for window in windows:
            start_pos = end_pos - window
            if start_pos < 0:
                row[f"fwd_ret_{window}d"] = np.nan
                continue
            trail_slice = series.iloc[start_pos : end_pos + 1].dropna()
            if trail_slice.empty:
                row[f"fwd_ret_{window}d"] = np.nan
                continue
            start_price = trail_slice.iloc[0]
            if pd.isna(start_price) or start_price == 0:
                row[f"fwd_ret_{window}d"] = np.nan
                continue
            row[f"fwd_ret_{window}d"] = end_price / start_price - 1.0
        rows.append(row)

    future_df = pd.DataFrame(rows)
    for window in windows:
        col = f"fwd_ret_{window}d"
        future_df[f"market_ret_rank_{window}d"] = future_df[col].rank(pct=True, method="average")
    return future_df


def build_build_window_metrics(
    returns: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    member_map: dict[int, list[str]],
) -> pd.DataFrame:
    """Per-code volume rank within its cluster (build window)."""
    if volume_df is None or volume_df.empty:
        return pd.DataFrame(columns=["code", "volume_rank_in_cluster"])
    avg_volume = volume_df.mean()
    rows = []
    for cluster_id, members in member_map.items():
        present = [m for m in members if m in avg_volume.index]
        if not present:
            continue
        volume_rank = avg_volume[present].rank(method="min", ascending=False)
        for code in present:
            rows.append({"code": code, "volume_rank_in_cluster": float(volume_rank.get(code, np.nan))})
    return pd.DataFrame(rows)


def _code_key(code: object) -> str:
    return str(code).strip()


def _coerce_cluster_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_signed_pct(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(numeric):
        return "-"
    return f"{numeric * 100:+.2f}%"


def _build_forward_return_map(future_df: pd.DataFrame, focus_window: int) -> dict[str, float]:
    ret_col = f"fwd_ret_{focus_window}d"
    if future_df.empty or ret_col not in future_df.columns or "code" not in future_df.columns:
        return {}
    ret_by_code: dict[str, float] = {}
    for _, future_row in future_df.iterrows():
        key = _code_key(future_row.get("code", ""))
        if not key:
            continue
        value = pd.to_numeric(future_row.get(ret_col), errors="coerce")
        ret_by_code[key] = float(value) if pd.notna(value) else np.nan
    return ret_by_code


def _format_selected_cluster_entry(
    cluster_id: int,
    selected_codes: list[str],
    name_map: dict[str, str],
    ret_by_code: dict[str, float],
) -> str:
    parts: list[str] = []
    for code in selected_codes:
        key = _code_key(code)
        label = format_code_with_cn(key, name_map)
        parts.append(f"{label} {_format_signed_pct(ret_by_code.get(key, np.nan))}")
    return f"簇 {cluster_id}：" + "；".join(parts)


def _build_cophenetic_distance_df(returns: pd.DataFrame, code_keys: list[str]) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()

    matched_keys = [key for key in dict.fromkeys(code_keys) if key in returns.columns]
    if len(matched_keys) < 2:
        return pd.DataFrame()

    panel = returns[matched_keys].copy()
    corr = panel.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dist_values = np.clip(1.0 - corr.to_numpy(dtype=float), 0.0, 2.0)
    dist_values = (dist_values + dist_values.T) / 2.0
    np.fill_diagonal(dist_values, 0.0)
    condensed = squareform(dist_values, checks=False)
    linkage_matrix = linkage(condensed, method="ward")
    cophenetic_result = cophenet(linkage_matrix, condensed)
    cophenetic_condensed = cophenetic_result[1] if isinstance(cophenetic_result, tuple) else cophenetic_result
    cophenetic_values = squareform(cophenetic_condensed)
    np.fill_diagonal(cophenetic_values, 0.0)
    return pd.DataFrame(cophenetic_values, index=matched_keys, columns=matched_keys)


def add_related_selected_clusters(
    missed_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    selected_codes: list[str],
    future_df: pd.DataFrame,
    returns: pd.DataFrame | None,
    name_map: dict[str, str],
    *,
    focus_window: int = 20,
    top_k: int = DEFAULT_RELATED_SELECTED_TOP_K,
) -> pd.DataFrame:
    out = missed_df.copy()
    out["related_selected_clusters"] = "-"
    if out.empty:
        return out

    target_mask = out["reason_kind"].astype(str).eq("target_count_not_selected") if "reason_kind" in out.columns else pd.Series(False, index=out.index)
    if not target_mask.any():
        return out

    if returns is None or returns.empty:
        out.loc[target_mask, "related_selected_clusters"] = "-"
        return out

    mapping_work = mapping_df.copy()
    mapping_work["code_key"] = mapping_work["code"].map(_code_key)
    mapping_work["cluster_id"] = pd.to_numeric(mapping_work["cluster"], errors="coerce")
    mapping_work = mapping_work[mapping_work["code_key"].ne("") & mapping_work["cluster_id"].notna()].copy()
    if mapping_work.empty:
        return out
    mapping_work["cluster_id"] = mapping_work["cluster_id"].astype(int)

    selected_set = {_code_key(code) for code in selected_codes if _code_key(code)}
    selected_rows = mapping_work[mapping_work["code_key"].isin(selected_set)].copy()
    if selected_rows.empty:
        return out

    cluster_members = mapping_work.groupby("cluster_id")["code_key"].apply(
        lambda series: list(dict.fromkeys(series.tolist()))
    ).to_dict()
    selected_by_cluster = selected_rows.groupby("cluster_id")["code_key"].apply(
        lambda series: list(dict.fromkeys(series.tolist()))
    ).to_dict()
    cophenetic_df = _build_cophenetic_distance_df(returns, mapping_work["code_key"].tolist())
    if cophenetic_df.empty:
        return out

    ret_by_code = _build_forward_return_map(future_df, focus_window)

    for missed_index, missed_row in out.loc[target_mask].iterrows():
        missed_cluster_id = _coerce_cluster_id(missed_row.get("cluster"))
        if missed_cluster_id is None:
            continue

        missed_members = [
            code for code in cluster_members.get(missed_cluster_id, [])
            if code in cophenetic_df.index
        ]
        if not missed_members:
            continue

        ranking: list[tuple[float, float, int, list[str]]] = []
        for selected_cluster_id, sel_codes in selected_by_cluster.items():
            if selected_cluster_id == missed_cluster_id:
                continue
            selected_members = [
                code for code in cluster_members.get(selected_cluster_id, [])
                if code in cophenetic_df.columns
            ]
            if not selected_members:
                continue
            distance_block = cophenetic_df.loc[missed_members, selected_members]
            min_distance = float(distance_block.min().min())
            if pd.isna(min_distance):
                continue
            selected_returns = [ret_by_code.get(_code_key(code), np.nan) for code in sel_codes]
            valid_selected_returns = [float(value) for value in selected_returns if pd.notna(value)]
            best_selected_return = max(valid_selected_returns) if valid_selected_returns else -np.inf
            ranking.append((min_distance, -best_selected_return, int(selected_cluster_id), sel_codes))

        if not ranking:
            continue

        ranking.sort(key=lambda item: (item[0], item[1], item[2]))
        entries = [
            _format_selected_cluster_entry(cluster_id, sel_codes, name_map, ret_by_code)
            for _, _, cluster_id, sel_codes in ranking[: max(1, int(top_k))]
        ]
        out.at[missed_index, "related_selected_clusters"] = "<br>".join(entries)

    return out


def build_missed_clusters(
    mapping_df: pd.DataFrame,
    selected_codes: list[str],
    future_df: pd.DataFrame,
    *,
    min_cluster_n: int,
    focus_window: int = 20,
) -> pd.DataFrame:
    selected_set = {str(c).strip() for c in selected_codes}
    merged = mapping_df.merge(future_df, on="code", how="left")
    rows: list[dict] = []
    ret_col = f"fwd_ret_{focus_window}d"

    for cluster_id, grp in merged.groupby("cluster"):
        members = grp["code"].astype(str).tolist()
        n = int(grp["n"].iloc[0]) if "n" in grp.columns and grp["n"].notna().any() else len(grp)
        if n < min_cluster_n:
            continue
        selected_in_cluster = [c for c in members if c in selected_set]
        if selected_in_cluster:
            continue

        grp_sorted = grp.sort_values(ret_col, ascending=False, na_position="last")
        best = grp_sorted.iloc[0] if not grp_sorted.empty else None
        reason_kind = _infer_missed_cluster_reason(grp)

        rows.append(
            {
                "cluster": int(cluster_id),
                "n": n,
                "members_count": len(members),
                "members_preview": "|".join(members[:6]),
                "best_code": best["code"] if best is not None else "",
                "best_name": best.get("name", "") if best is not None else "",
                f"best_{focus_window}d_ret": float(best[ret_col]) if best is not None and pd.notna(best.get(ret_col)) else np.nan,
                "reason_kind": reason_kind,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cluster",
                "n",
                "members_count",
                "members_preview",
                "best_code",
                "best_name",
                f"best_{focus_window}d_ret",
                "reason_kind",
            ]
        )
    out = pd.DataFrame(rows)
    sort_col = f"best_{focus_window}d_ret"
    return out.sort_values(sort_col, ascending=False, na_position="last")


def build_rep_lag(
    mapping_df: pd.DataFrame,
    selected_codes: list[str],
    future_df: pd.DataFrame,
    build_metrics: pd.DataFrame,
    *,
    min_regret: float,
    focus_window: int = 20,
) -> pd.DataFrame:
    selected_set = {str(c).strip() for c in selected_codes}
    merged = mapping_df.merge(future_df, on="code", how="left")
    if not build_metrics.empty:
        merged = merged.merge(build_metrics, on="code", how="left")

    ret_col = f"fwd_ret_{focus_window}d"
    rows: list[dict] = []

    for cluster_id, grp in merged.groupby("cluster"):
        members = grp.copy()
        if members.empty:
            continue
        members = members.sort_values(ret_col, ascending=False, na_position="last")
        best_row = members.iloc[0]
        best_code = str(best_row["code"])
        best_ret = float(best_row[ret_col]) if pd.notna(best_row.get(ret_col)) else np.nan

        selected_rows = members[members["code"].astype(str).isin(selected_set)]
        if selected_rows.empty:
            continue

        for _, sel in selected_rows.iterrows():
            sel_code = str(sel["code"])
            sel_ret = float(sel[ret_col]) if pd.notna(sel[ret_col]) else np.nan
            regret = best_ret - sel_ret if pd.notna(best_ret) and pd.notna(sel_ret) else np.nan
            rank = int((members[ret_col] >= sel_ret).sum()) if pd.notna(sel_ret) else len(members)

            vol_rank = sel.get("volume_rank_in_cluster", np.nan)
            likely_low_volume = bool(pd.notna(vol_rank) and float(vol_rank) > 1.0)

            if rank <= 1:
                continue
            if pd.notna(regret) and regret < min_regret:
                continue

            rows.append(
                {
                    "cluster": int(cluster_id),
                    "n": int(sel.get("n", len(members)) if pd.notna(sel.get("n")) else len(members)),
                    "selected_code": sel_code,
                    "selected_name": sel.get("name", ""),
                    "best_code": best_code,
                    "best_name": best_row.get("name", ""),
                    f"selected_{focus_window}d_ret": sel_ret,
                    f"best_{focus_window}d_ret": best_ret,
                    f"regret_{focus_window}d": regret,
                    f"rank_{focus_window}d": rank,
                    "volume_rank_in_cluster": vol_rank,
                    "likely_due_to_low_volume": likely_low_volume,
                    "selected_reason": sel.get("reason", ""),
                }
            )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(f"regret_{focus_window}d", ascending=False, na_position="last")


def _escape_md_table_cell(value: object) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text.replace("|", "&#124;")


def _df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无）"
    header = "| " + " | ".join(_escape_md_table_cell(c) for c in df.columns) + " |"
    sep_row = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(_escape_md_table_cell(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep_row] + rows)


def build_daily_markdown(
    *,
    review_date: str,
    selected_count: int,
    mapping_count: int,
    missed_df: pd.DataFrame,
    rep_lag_df: pd.DataFrame,
    top_n: int,
    name_map: dict[str, str],
    generated_at: str | None = None,
) -> str:
    lines = [
        f"# 每日聚类池审查 {review_date}",
        "",
        "## 摘要",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 审查日 | {review_date} |",
        f"| 报告生成时间 | {generated_at or '-'} |",
        f"| 入选池标的数 | {selected_count} |",
        f"| 全量 mapping 行数 | {mapping_count} |",
        f"| 漏族（簇内无人入选） | {len(missed_df)} |",
        f"| 代表落后（20d 非簇内第1） | {len(rep_lag_df)} |",
        f"| 收益锚点日 | {review_date}（回看 20 个交易日） |",
        "",
        "## 漏族（Top {})".format(top_n),
        "",
    ]
    if missed_df.empty:
        lines.append("（无 — 每个 n≥2 的簇在入选池中至少有一只代表）")
    else:
        lines.append(_df_to_md_table(format_missed_df_for_markdown(missed_df, name_map, top_n)))
    lines.append("")

    lines.append("## 代表落后（Top {}，按 regret_20d）".format(top_n))
    lines.append("")
    if rep_lag_df.empty:
        lines.append("（无 — 入选标的均为各自簇内近20日收益前列，或落后未达阈值）")
    else:
        lines.append(_df_to_md_table(format_rep_lag_df_for_markdown(rep_lag_df, name_map, top_n)))
    lines.append("")

    lines.append("## 今日建议")
    lines.append("")
    suggestions: list[str] = []
    if not missed_df.empty:
        top_miss = missed_df.iloc[0]
        best_code = str(top_miss.get("best_code", "")).strip()
        best_cn = lookup_name(best_code, name_map) if best_code else ""
        best_label = f"{best_code} {best_cn}".strip() if best_cn else best_code
        reason_detail = format_missed_reason_detail(str(top_miss.get("reason_kind", "")))
        suggestions.append(
            f"- 漏族：簇 {top_miss['cluster']}（n={top_miss['n']}）无入选代表；"
            f"近20日最强为 {best_label}；原因：{reason_detail}。"
        )
    if not rep_lag_df.empty:
        top_lag = rep_lag_df.iloc[0]
        suggestions.append(
            f"- 代表落后：簇 {top_lag['cluster']} 选 {top_lag['selected_code']}，"
            f"但 {top_lag['best_code']} 近20日更强（regret≈{top_lag.get('regret_20d', 0)*100:.2f}%）；"
            f"{'可能因流动性规则保留现代表' if top_lag.get('likely_due_to_low_volume') else '可考虑替换代表'}。"
        )
    if not suggestions:
        suggestions.append("- 今日无明显漏族或代表落后；维持现有池即可。")
    lines.extend(suggestions)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "说明：审查以 `sw_cluster_mapping_selected.csv` 为权威；"
        "全量 `sw_cluster_mapping.csv` 仅用于同簇对比。换代表需重跑 `sw-daily pool`。"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review sw_cluster_mapping_selected.csv quality against the full mapping"
    )
    parser.add_argument("--mapping-csv", default=str(CSV_OUT), help="Full mapping from the same pool run")
    parser.add_argument("--selected-csv", default=str(CSV_SELECTED_OUT), help="Pool to audit")
    parser.add_argument("--future-end", default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--min-regret", type=float, default=DEFAULT_MIN_REGRET)
    parser.add_argument("--min-cluster-n", type=int, default=DEFAULT_MIN_CLUSTER_N)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--out-dir", default=None)
    return parser


def run_review(args: argparse.Namespace) -> int:
    windows = parse_windows(args.windows)
    focus_window = 20 if 20 in windows else windows[-1]
    review_date = pd.Timestamp(args.future_end).strftime("%Y-%m-%d")
    date_tag = pd.Timestamp(args.future_end).strftime("%Y%m%d")

    out_dir = Path(args.out_dir) if args.out_dir else POOL_DIR / "review" / f"daily_{date_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_codes = load_selected_codes(args.selected_csv)
    _, name_map = load_sw_universe()
    mapping_df = load_mapping_csv(args.mapping_csv)

    codes = mapping_df["code"].tolist()
    review_start = window_bounds(args.future_end, windows)
    close, vol = load_close_volume(codes, test_period=(review_start, args.future_end))
    future_df = compute_trailing_return_metrics(close, args.future_end, windows)

    member_map = mapping_df.groupby("cluster")["code"].apply(list).to_dict()
    recent_vol = vol.tail(C.CLUSTER_LOOKBACK_DAYS + 1) if vol is not None else None
    build_metrics = build_build_window_metrics(None, recent_vol, member_map)  # type: ignore[arg-type]

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    missed_df = build_missed_clusters(
        mapping_df,
        selected_codes,
        future_df,
        min_cluster_n=int(args.min_cluster_n),
        focus_window=focus_window,
    )
    rep_lag_df = build_rep_lag(
        mapping_df,
        selected_codes,
        future_df,
        build_metrics,
        min_regret=float(args.min_regret),
        focus_window=focus_window,
    )
    missed_df = add_related_selected_clusters(
        missed_df,
        mapping_df,
        selected_codes,
        future_df,
        returns,
        name_map,
        focus_window=focus_window,
    )

    missed_df.to_csv(out_dir / "missed_clusters.csv", index=False)
    rep_lag_df.to_csv(out_dir / "rep_lag.csv", index=False)

    report_md = build_daily_markdown(
        review_date=review_date,
        selected_count=len(selected_codes),
        mapping_count=len(mapping_df),
        missed_df=missed_df,
        rep_lag_df=rep_lag_df,
        top_n=int(args.top_n),
        name_map=name_map,
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    report_path = out_dir / f"daily_cluster_review_{date_tag}.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"[INFO] Daily review report: {report_path}")
    if not missed_df.empty and "reason_kind" in missed_df.columns:
        preview = missed_df.head(3)[["cluster", "reason_kind"]]
        print("[INFO] missed_clusters reason preview (top 3):")
        print(preview.to_string(index=False))
    print(f"[INFO] missed_clusters={len(missed_df)} rep_lag={len(rep_lag_df)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_review(args)
