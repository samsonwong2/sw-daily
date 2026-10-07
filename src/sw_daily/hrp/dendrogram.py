"""Ward clustering plus an HRP-CVaR dendrogram for the selected industry pool.

Ported from ``etf_daily.hrp.dendrogram``. Holdings, bond/overseas/domestic
domain splits, and the close-price pickle cache are omitted: every Shenwan
industry is one domestic universe, and the qlib panel is small enough to reload.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from skfolio import RiskMeasure
from skfolio.optimization import HierarchicalRiskParity
from skfolio.preprocessing import prices_to_returns

from sw_daily.paths import HRP_DIR

DIST_T = 0.40
DEFAULT_MIN_CORR = 0.55
MIN_REP_MOVE_PCT = 5.0
CLUSTER_LOOKBACK_DAYS = 252
DOMAIN_INDUSTRY = "industry"


@dataclass
class UniverseResult:
    name: str
    requested_codes: tuple[str, ...]
    used_codes: tuple[str, ...] = ()
    dropped_codes: tuple[str, ...] = ()
    error: str | None = None
    figures: list[tuple[str, object]] = field(default_factory=list)
    cluster_reps: pd.DataFrame | None = None


def build_test_period(asof_date: str, lookback_days: int) -> tuple[str, str]:
    end = datetime.strptime(asof_date, "%Y-%m-%d").date()
    start = end - timedelta(days=int(lookback_days) + 30)
    return start.isoformat(), end.isoformat()


def prepare_returns(close: pd.DataFrame, requested_codes: Iterable[str]) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    requested = tuple(str(code).strip() for code in requested_codes)
    available = {str(col): col for col in close.columns}
    present = [available[code] for code in requested if code in available]
    missing = tuple(code for code in requested if code not in available)
    if not present:
        return pd.DataFrame(), tuple(), missing
    prices = close[present].copy()
    prices.columns = [str(col) for col in prices.columns]
    returns = prices_to_returns(prices).replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(returns.mean())
    var_series = returns.var(axis=0)
    zero_var = tuple(col for col, value in var_series.items() if value <= 0 or np.isnan(value))
    if zero_var:
        returns = returns.drop(columns=list(zero_var))
    used = tuple(returns.columns.tolist())
    dropped = tuple(dict.fromkeys([*missing, *zero_var]))
    return returns, used, dropped


def _aligned_corr(returns: pd.DataFrame, corr: pd.DataFrame | None) -> pd.DataFrame:
    cols = list(returns.columns)
    if corr is not None and all(col in corr.index and col in corr.columns for col in cols):
        return corr.loc[cols, cols]
    return returns.corr()


def ward_labels(
    returns: pd.DataFrame,
    *,
    dist_t: float = DIST_T,
    n_clusters: int | None = None,
    corr: pd.DataFrame | None = None,
) -> pd.Series:
    codes = list(returns.columns)
    if len(codes) == 0:
        return pd.Series(dtype=int)
    if len(codes) == 1:
        return pd.Series([1], index=codes, dtype=int)
    corr = _aligned_corr(returns, corr)
    dist = (1.0 - corr).clip(lower=0.0).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    linked = linkage(condensed, method="ward")
    if n_clusters is not None and int(n_clusters) >= 2:
        labels = fcluster(linked, t=min(int(n_clusters), len(codes)), criterion="maxclust")
    else:
        labels = fcluster(linked, t=float(dist_t), criterion="distance")
    return pd.Series(labels, index=codes, name="cluster", dtype=int)


def split_by_min_corr(labels: pd.Series, corr: pd.DataFrame, *, min_corr: float = DEFAULT_MIN_CORR) -> pd.Series:
    if float(min_corr) <= 0 or labels.empty:
        return labels.astype(int)
    out = labels.copy().astype(int)
    next_id = int(out.max()) + 1 if len(out) else 1
    for cid, members in labels.groupby(labels).groups.items():
        names = [str(member) for member in members]
        if len(names) < 2:
            continue
        sub = corr.reindex(index=names, columns=names)
        values = [float(sub.loc[a, b]) for i, a in enumerate(names) for b in names[i + 1 :] if pd.notna(sub.loc[a, b])]
        if not values or min(values) >= float(min_corr):
            continue
        adjacent: dict[str, list[str]] = {code: [] for code in names}
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                value = sub.loc[left, right]
                if pd.notna(value) and float(value) >= float(min_corr):
                    adjacent[left].append(right)
                    adjacent[right].append(left)
        seen: set[str] = set()
        components: list[list[str]] = []
        for code in names:
            if code in seen:
                continue
            stack = [code]
            seen.add(code)
            component = [code]
            while stack:
                node = stack.pop()
                for neighbor in adjacent[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
            components.append(component)
        for index, component in enumerate(components):
            new_id = int(cid) if index == 0 else next_id
            if index > 0:
                next_id += 1
            for code in component:
                out.loc[code] = new_id
    return out.astype(int)


def corr_medoid(members: list[str], corr: pd.DataFrame) -> str:
    names = [str(member) for member in members]
    if len(names) == 1:
        return names[0]
    sub = corr.reindex(index=names, columns=names)
    best_code = names[0]
    best_score = -np.inf
    for code in names:
        others = [other for other in names if other != code]
        score = float(pd.to_numeric(sub.loc[code, others], errors="coerce").mean())
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_code = code
    return best_code


def compute_cluster_representatives(
    returns: pd.DataFrame,
    close: pd.DataFrame,
    *,
    name_map: dict[str, str],
    n_clusters: int | None = None,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    dist_t: float = DIST_T,
    min_corr: float = DEFAULT_MIN_CORR,
    labels: pd.Series | None = None,
    corr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if labels is None or corr is None:
        corr = _aligned_corr(returns, corr)
        labels = split_by_min_corr(ward_labels(returns, dist_t=dist_t, n_clusters=n_clusters, corr=corr), corr, min_corr=min_corr)
    if labels.empty:
        return pd.DataFrame()
    prices = close.copy()
    prices.columns = [str(col) for col in prices.columns]
    window = max(int(rep_window), 1)
    base = prices.iloc[-(window + 1)] if len(prices) > window else prices.iloc[0]
    ret_n = (prices.iloc[-1] / base - 1.0) * 100.0
    medoids = {int(cid): corr_medoid([str(member) for member in members], corr) for cid, members in labels.groupby(labels).groups.items()}
    records: list[dict[str, object]] = []
    for cid, members_idx in labels.groupby(labels).groups.items():
        members = [str(member) for member in members_idx]
        moves = ret_n.reindex(members).astype(float)
        medoid = medoids[int(cid)]
        up_members = [code for code in members if pd.notna(moves.get(code)) and moves[code] > 0]
        down_members = [code for code in members if pd.notna(moves.get(code)) and moves[code] < 0]
        flat = [code for code in members if code not in up_members and code not in down_members]
        if flat:
            up_best = max((abs(float(moves[code])) for code in up_members), default=0.0)
            down_best = max((abs(float(moves[code])) for code in down_members), default=0.0)
            (up_members if up_best >= down_best else down_members).extend(flat)
        for group_label, group_members, direction in (("up", up_members, "涨"), ("down", down_members, "跌")):
            if not group_members:
                continue
            group = moves.reindex(group_members)
            best = float(group.abs().max())
            has_direction = float(min_rep_move) <= 0 or best >= float(min_rep_move)
            extreme = str(group.abs().idxmax()) if has_direction else None
            for code in group_members:
                records.append(
                    {
                        "cluster": int(cid),
                        "domain": DOMAIN_INDUSTRY,
                        "group": group_label,
                        "code": code,
                        "name": name_map.get(code, code),
                        f"ret{window}_pct": float(group[code]),
                        "direction": direction,
                        "is_representative": code == medoid,
                        "is_extreme_mover": code == extreme,
                        "group_status": "representative" if has_direction else "no_direction",
                        "n_members": len(members),
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    ret_col = f"ret{window}_pct"
    return frame.assign(_abs=frame[ret_col].abs()).sort_values(
        ["cluster", "group", "is_representative", "is_extreme_mover", "_abs"],
        ascending=[True, True, False, False, False],
    ).drop(columns=["_abs"]).reset_index(drop=True)


def _display_names(codes: list[str], name_map: dict[str, str]) -> list[str]:
    assigned: set[str] = set()
    labels: list[str] = []
    for code in codes:
        base = (name_map.get(code) or code).strip() or code
        label = base if base not in assigned else f"{base}({code})"
        assigned.add(label)
        labels.append(label)
    return labels


def plot_ward_dendrogram(returns, labels, name_map, *, layout_width, layout_height, corr=None):
    import plotly.graph_objects as go

    codes = [str(code) for code in returns.columns]
    title = "Ward(1-corr) dendrogram"
    if len(codes) < 2:
        fig = go.Figure()
        fig.update_layout(title=title, width=layout_width, height=layout_height)
        return fig
    display = _display_names(codes, name_map)
    corr = _aligned_corr(returns, corr)
    dist = (1.0 - corr).clip(lower=0.0).fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    linked = linkage(squareform(dist.values, checks=False), method="ward")
    dendro = scipy_dendrogram(linked, labels=display, no_plot=True, color_threshold=None)
    fig = go.Figure()
    for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#64748b", width=1), hoverinfo="skip", showlegend=False))
    lab_by_display = {display[i]: int(labels.loc[codes[i]]) for i in range(len(codes))}
    leaf_pos = {lab: i * 10 for i, lab in enumerate(dendro["ivl"])}
    fig.update_layout(
        title=title,
        width=layout_width,
        height=layout_height,
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=[leaf_pos[lab] + 5 for lab in dendro["ivl"]],
            ticktext=[f"{lab} [C{lab_by_display.get(lab, '?')}]" if lab in lab_by_display else lab for lab in dendro["ivl"]],
            tickangle=90,
        ),
        yaxis_title="Ward distance (1-corr)",
        margin=dict(b=160),
    )
    return fig


def fit_hrp_dendrograms(returns: pd.DataFrame, *, layout_width: int, layout_height: int) -> list[tuple[str, object]]:
    model = HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR, portfolio_params={"name": "HRP-CVaR-Ward-Pearson"})
    model.fit(returns)
    fig_no_heat = model.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)
    fig_heat = model.hierarchical_clustering_estimator_.plot_dendrogram()
    for title, fig in (("HRP-CVaR reference (heatmap=False)", fig_no_heat), ("HRP-CVaR reference (heatmap=True)", fig_heat)):
        fig.update_layout(title=title, width=layout_width, height=layout_height, showlegend=False)
    return [("HRP-CVaR reference (heatmap=False)", fig_no_heat), ("HRP-CVaR reference (heatmap=True)", fig_heat)]


def rename_returns_to_display_names(returns: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    renamed = returns.copy()
    renamed.columns = _display_names([str(code) for code in renamed.columns], name_map)
    return renamed


def _fmt_ret(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:+.2f}%"


def render_cluster_reps_html(frame: pd.DataFrame, rep_window: int, min_rep_move: float = MIN_REP_MOVE_PCT) -> str:
    ret_col = f"ret{rep_window}_pct"
    group_zh = {"up": "涨组", "down": "跌组"}
    rows: list[str] = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>簇</th><th>域</th><th>组</th><th>代码</th><th>名称</th><th>区间涨跌</th><th>成员数</th><th>说明</th></tr>",
    ]
    for cid, cluster in frame.groupby("cluster", sort=True):
        med = cluster[cluster["is_representative"]].drop_duplicates("code")
        n_members = int(cluster["n_members"].iloc[0])
        if not med.empty:
            row = med.iloc[0]
            med_cells = f"<td><b>{html.escape(str(row['code']))}</b></td><td>{html.escape(str(row['name']))}</td><td>{_fmt_ret(row[ret_col])}</td>"
        else:
            med_cells = "<td colspan='3'>—</td>"
        rows.append(f"<tr><td>{int(cid)}</td><td>{DOMAIN_INDUSTRY}</td><td>medoid</td>{med_cells}<td>{n_members}</td><td>簇相关中心</td></tr>")
        for label, group in cluster.groupby("group", sort=True):
            movers = group[group["is_extreme_mover"].astype(bool)]
            if not movers.empty:
                mover = movers.iloc[0]
                mover_cells = f"<td>{html.escape(str(mover['code']))}</td><td>{html.escape(str(mover['name']))}</td><td>{_fmt_ret(mover[ret_col])}</td>"
            elif str(group["group_status"].iloc[0]) == "no_direction":
                mover_cells = f"<td colspan='3'>无明确方向（组内最大|涨跌| &lt; {float(min_rep_move):.1f}%）</td>"
            else:
                mover_cells = "<td colspan='3'>—</td>"
            rows.append(f"<tr><td>{int(cid)}</td><td>{DOMAIN_INDUSTRY}</td><td>{group_zh.get(str(label), str(label))}</td>{mover_cells}<td>{n_members}</td><td>方向组极端动量</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def figure_to_html(fig: object, *, include_plotlyjs: bool | str) -> str:
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=False)  # type: ignore[attr-defined]


def render_html(*, output_path: Path, asof_date: str, test_period: tuple[str, str], universe: UniverseResult, rep_window: int, min_rep_move: float, dist_t: float, min_corr: float) -> None:
    sections = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>HRP Dendrogram</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#1f2937} .meta{color:#4b5563} .error{color:#b91c1c}</style></head><body>",
        "<h1>HRP 层次聚类 Dendrogram</h1>",
        f"<p class='meta'>生成时间: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}<br>数据区间: {html.escape(test_period[0])} ~ {html.escape(test_period[1])} (asof={html.escape(asof_date)})</p>",
        f"<h2>{html.escape(universe.name)}</h2>",
        f"<p class='meta'>请求 {len(universe.requested_codes)}；有效 {len(universe.used_codes)}；剔除 {len(universe.dropped_codes)}</p>",
    ]
    if universe.dropped_codes:
        sections.append(f"<p class='meta'>剔除: {html.escape(', '.join(universe.dropped_codes))}</p>")
    if universe.error:
        sections.append(f"<p class='error'>{html.escape(universe.error)}</p>")
    else:
        if universe.cluster_reps is not None and not universe.cluster_reps.empty:
            n_clusters = int(universe.cluster_reps["cluster"].nunique())
            sections.append(f"<h3>聚类代表（{n_clusters} 簇，dist_t={dist_t:.2f}，min_corr={min_corr:.2f}）</h3>")
            sections.append(render_cluster_reps_html(universe.cluster_reps, rep_window, min_rep_move))
        include: bool | str = "cdn"
        for subtitle, fig in universe.figures:
            sections.append(f"<h3>{html.escape(subtitle)}</h3>")
            sections.append(figure_to_html(fig, include_plotlyjs=include))
            include = False
    sections.append("</body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(sections), encoding="utf-8")


def build_universe_result(close: pd.DataFrame, codes: tuple[str, ...], name_map: dict[str, str], *, dist_t: float, min_corr: float, n_clusters: int | None, rep_window: int, min_rep_move: float) -> UniverseResult:
    result = UniverseResult(name="申万二级行业入选池", requested_codes=codes)
    if len(codes) < 2:
        result.error = "fewer than 2 industries"
        return result
    returns, used, dropped = prepare_returns(close, codes)
    result.used_codes = used
    result.dropped_codes = dropped
    if len(used) < 2:
        result.error = f"fewer than 2 industries with returns (used={len(used)})"
        return result
    corr = returns.corr()
    labels = split_by_min_corr(ward_labels(returns, dist_t=dist_t, n_clusters=n_clusters, corr=corr), corr, min_corr=min_corr)
    result.cluster_reps = compute_cluster_representatives(
        returns, close, name_map=name_map, rep_window=rep_window, min_rep_move=min_rep_move, labels=labels, corr=corr
    )
    result.figures = [
        ("Ward(1-corr) dendrogram", plot_ward_dendrogram(returns, labels, name_map, layout_width=2400, layout_height=700, corr=corr)),
        *fit_hrp_dendrograms(rename_returns_to_display_names(returns, name_map), layout_width=2400, layout_height=700),
    ]
    return result


def output_paths(asof_date: str, dist_t: float, explicit: Path | None = None) -> tuple[Path, Path]:
    tag = asof_date.replace("-", "")
    cut = "" if abs(float(dist_t) - DIST_T) <= 1e-9 else f"_d{int(round(float(dist_t) * 100)):03d}"
    html_path = Path(explicit) if explicit is not None else HRP_DIR / tag / f"hrp_dendrogram_{tag}{cut}.html"
    csv_path = html_path.parent / f"cluster_representatives_{tag}{cut}.csv"
    return html_path, csv_path


def run_hrp(
    close: pd.DataFrame,
    codes: tuple[str, ...],
    name_map: dict[str, str],
    *,
    asof_date: str,
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
    dist_t: float = DIST_T,
    min_corr: float = DEFAULT_MIN_CORR,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    n_clusters: int | None = None,
    output: Path | None = None,
) -> UniverseResult:
    test_period = build_test_period(asof_date, lookback_days)
    universe = build_universe_result(
        close, codes, name_map, dist_t=dist_t, min_corr=min_corr, n_clusters=n_clusters, rep_window=rep_window, min_rep_move=min_rep_move
    )
    html_path, csv_path = output_paths(asof_date, dist_t, output)
    render_html(
        output_path=html_path,
        asof_date=asof_date,
        test_period=test_period,
        universe=universe,
        rep_window=rep_window,
        min_rep_move=min_rep_move,
        dist_t=dist_t,
        min_corr=min_corr,
    )
    if universe.cluster_reps is not None and not universe.cluster_reps.empty:
        universe.cluster_reps.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")
    status = "ok" if not universe.error else universe.error
    print(f"requested={len(universe.requested_codes)} used={len(universe.used_codes)} dropped={len(universe.dropped_codes)} status={status}")
    return universe


def run_hrp_from_qlib(
    *,
    asof_date: str,
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
    dist_t: float = DIST_T,
    min_corr: float = DEFAULT_MIN_CORR,
    rep_window: int = 20,
    min_rep_move: float = MIN_REP_MOVE_PCT,
    n_clusters: int | None = None,
    output: Path | None = None,
) -> int:
    from sw_daily.pool.data_loading import load_close_volume
    from sw_daily.pool.selector import CSV_SELECTED_OUT, load_sw_universe

    if not CSV_SELECTED_OUT.is_file():
        raise SystemExit(f"missing {CSV_SELECTED_OUT}; run: sw-daily pool")
    selected = pd.read_csv(CSV_SELECTED_OUT, dtype=str)
    codes = tuple(selected["code"].astype(str).str.strip().tolist())
    _, name_map = load_sw_universe()
    start, end = build_test_period(asof_date, lookback_days)
    close, _volume = load_close_volume(list(codes), test_period=(start, end))
    run_hrp(
        close,
        codes,
        name_map,
        asof_date=asof_date,
        lookback_days=lookback_days,
        dist_t=dist_t,
        min_corr=min_corr,
        rep_window=rep_window,
        min_rep_move=min_rep_move,
        n_clusters=n_clusters,
        output=output,
    )
    return 0
