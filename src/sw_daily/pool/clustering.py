"""Correlation clustering and representative selection for Shenwan indexes.

Ported from ``etf_daily/pool/builder/clustering.py`` with the ETF-specific
force-keep / family-suppression branches removed. Codes here are plain
6-digit Shenwan index codes, so no exchange-prefix normalization is needed.

Pipeline:
  1. hierarchical clustering on the returns correlation matrix (Ward)
  2. inside each cluster, score members by (annualized Sharpe + 12-1 momentum)
  3. near-clone dedup with a stricter intra-cluster threshold, then a
     coverage-rescue pass so every cluster keeps one representative
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

DIST_T = float(os.environ.get("POOL_DIST_T", 0.40))
PER_CLUSTER = 1
RULE = "multi_rep_equal_mainline"
MULTI_REP_EXP_BASE = 0.65
TARGET_SELECTED_COUNT = int(os.environ.get("POOL_TARGET_COUNT", 200))
CLUSTER_LOOKBACK_DAYS = 252
MIN_CLUSTER_HISTORY_DAYS = 120
MIN_PAIR_OVERLAP_DAYS = 120
MIN_SELECTION_HISTORY_DAYS = 180
LOW_OVERLAP_DISTANCE = 0.95
FINAL_MAX_ABS_CORR_THRESHOLDS = (0.70, 0.75, 0.80)
INTRA_CLUSTER_MAX_ABS_CORR = float(os.environ.get("POOL_INTRA_CLUSTER_CORR_LIMIT", 0.60))


def _rank_score(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    series = series.astype(float)
    if series.empty:
        return series
    return series.rank(method="average", pct=True)


def _return_scores(codes: Iterable[str], returns: pd.DataFrame) -> pd.Series:
    """Score candidates by a 50/50 blend of annualized Sharpe and 12-1 momentum.

    ``POOL_SCORING_MODE=legacy_mean`` restores rank-of-mean-return scoring.
    """
    codes = list(codes)
    if not codes:
        return pd.Series(dtype=float)
    code_list = list(dict.fromkeys(codes))
    panel = returns[code_list]

    mode = os.environ.get("POOL_SCORING_MODE", "sharpe_mom").lower()
    if mode == "legacy_mean":
        mean_return = panel.mean().reindex(code_list).fillna(0.0)
        return _rank_score(mean_return).sort_values(ascending=False)

    mean_daily = panel.mean()
    std_daily = panel.std(ddof=0).replace(0.0, np.nan)
    sharpe = (mean_daily / std_daily * np.sqrt(252.0)).reindex(code_list).fillna(0.0)

    n_rows = len(panel)
    skip_recent = min(21, max(0, n_rows - 1))
    momentum_window = panel.iloc[: n_rows - skip_recent] if skip_recent > 0 else panel
    momentum = (momentum_window + 1.0).prod() - 1.0
    momentum = momentum.reindex(code_list).fillna(0.0)

    score = 0.5 * _rank_score(sharpe) + 0.5 * _rank_score(momentum)
    return score.sort_values(ascending=False)


def _multi_rep_rank_weight(rank_index: int, scheme: str, exp_base: float = MULTI_REP_EXP_BASE) -> float:
    rank_number = int(rank_index) + 1
    if scheme == "equal":
        return 1.0
    if scheme == "harmonic":
        return 1.0 / float(rank_number)
    if scheme == "exp":
        return float(exp_base) ** float(rank_index)
    raise ValueError(f"Unsupported multi-rep weight scheme: {scheme}")


def _build_multi_rep_candidate_caps(cluster_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    quota_rows = []
    for _, row in cluster_df.iterrows():
        quota_rows.append(
            {
                "cluster": row["cluster"],
                "member_count": len(row["candidate_members"]),
                "top_return": float(row["candidate_top_return"]),
            }
        )
    quota_df = pd.DataFrame(quota_rows)
    total_members = int(quota_df["member_count"].sum()) if not quota_df.empty else 0
    if total_members <= 0:
        return pd.DataFrame(columns=["cluster", "member_count", "top_return", "raw_quota", "candidate_cap"])

    quota_df["raw_quota"] = quota_df["member_count"] / total_members * target_count
    quota_df["candidate_cap"] = np.ceil(quota_df["raw_quota"]).astype(int)
    quota_df["candidate_cap"] = quota_df[["candidate_cap", "member_count"]].min(axis=1)
    quota_df["candidate_cap"] = quota_df["candidate_cap"].clip(lower=1)

    target_candidate_count = int(target_count) + min(
        max(5, int(np.ceil(target_count * 0.15))),
        max(0, total_members - int(target_count)),
    )
    current_candidates = int(quota_df["candidate_cap"].sum())
    quota_df["remainder"] = quota_df["raw_quota"] - np.floor(quota_df["raw_quota"])
    while current_candidates < target_candidate_count:
        available = quota_df.loc[quota_df["candidate_cap"] < quota_df["member_count"]].copy()
        if available.empty:
            break
        available = available.sort_values(
            ["remainder", "member_count", "top_return", "cluster"],
            ascending=[False, False, False, True],
        )
        pick_index = available.index[0]
        quota_df.loc[pick_index, "candidate_cap"] += 1
        current_candidates += 1
    return quota_df


def _build_multi_rep_cluster_pool(
    cluster_df: pd.DataFrame,
    returns: pd.DataFrame,
    history_days: pd.Series,
    target_count: int | None,
    weight_scheme: str = "equal",
    exp_base: float = MULTI_REP_EXP_BASE,
    corr: pd.DataFrame | None = None,
    near_clone_corr_limit: float | None = None,
    intra_cluster_corr_limit: float | None = None,
) -> tuple[list[str], dict[str, str]]:
    if cluster_df is None or cluster_df.empty:
        return [], {}

    target_count = int(target_count or len(returns.columns))
    history_days = history_days if history_days is not None else pd.Series(dtype=float)
    work_df = cluster_df.copy()
    work_df["candidate_members"] = work_df["members"].apply(
        lambda members: [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
        or list(members)
    )
    work_df["candidate_top_return"] = work_df["candidate_members"].apply(
        lambda members: float(_return_scores(members, returns).iloc[0]) if members else -np.inf
    )

    cap_df = _build_multi_rep_candidate_caps(work_df, target_count)
    cap_map = dict(zip(cap_df["cluster"], cap_df["candidate_cap"]))
    raw_quota_map = dict(zip(cap_df["cluster"], cap_df["raw_quota"]))
    global_return_scores = _return_scores(list(returns.columns), returns)

    selected_rows = []
    for _, row in work_df.iterrows():
        members = row["candidate_members"]
        candidate_cap = int(cap_map.get(row["cluster"], 0))
        if candidate_cap <= 0 or not members:
            continue
        member_scores = global_return_scores.reindex(members).fillna(0.0).sort_values(ascending=False)
        picks = member_scores.head(candidate_cap).index.tolist()
        for rank_index, code in enumerate(picks):
            selected_rows.append(
                {
                    "code": code,
                    "cluster": row["cluster"],
                    "cluster_quota": float(raw_quota_map.get(row["cluster"], 0.0)),
                    "within_cluster_rank": rank_index + 1,
                    "return_score": float(member_scores.get(code, 0.0)),
                    "rep_weight": _multi_rep_rank_weight(rank_index, weight_scheme, exp_base),
                    "short_history": float(history_days.get(code, 0.0)) < MIN_SELECTION_HISTORY_DAYS,
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        selected: list[str] = []
        selected_reason_map: dict[str, str] = {}
    else:
        selected_df["priority_score"] = selected_df["return_score"] * selected_df["rep_weight"]
        selected_df = selected_df.sort_values(
            [
                "priority_score",
                "return_score",
                "cluster_quota",
                "rep_weight",
                "cluster",
                "within_cluster_rank",
                "code",
            ],
            ascending=[False, False, False, False, True, True, True],
        )
        unique_selected_df = selected_df.drop_duplicates(subset=["code"], keep="first").copy()
        # Guaranteed-coverage ordering: every cluster contributes its rank-1
        # representative first, then rank-2+ seats follow in priority order.
        # POOL_PLAN_B_COVERAGE=0 restores the plain priority ordering.
        _plan_b_coverage = os.environ.get("POOL_PLAN_B_COVERAGE", "1") == "1"
        if _plan_b_coverage:
            rank1 = unique_selected_df[unique_selected_df["within_cluster_rank"] == 1].copy()
            rank1 = rank1.sort_values(
                ["return_score", "cluster_quota", "cluster", "code"],
                ascending=[False, False, True, True],
            )
            rankN = unique_selected_df[unique_selected_df["within_cluster_rank"] > 1]
            ordered_df = pd.concat([rank1, rankN], ignore_index=False)
        else:
            ordered_df = unique_selected_df
        selected = ordered_df["code"].tolist()
        selected_reason_map = {}
        for _, picked in ordered_df.iterrows():
            short_history_suffix = "_short_history" if bool(picked["short_history"]) else ""
            coverage_tag = "_coverage" if (_plan_b_coverage and int(picked["within_cluster_rank"]) == 1) else ""
            selected_reason_map[picked["code"]] = (
                f"multi_rep_{weight_scheme}_cluster_{picked['cluster']}_rank_{int(picked['within_cluster_rank'])}{coverage_tag}{short_history_suffix}"
            )

    final_selected = selected[:target_count]
    dedup_reason_overrides: dict[str, str] = {}

    # Final near-clone dedup pass: drop any code whose |corr| with an
    # already-kept code exceeds the limit (cross-cluster) or the stricter
    # intra-cluster limit. Runs after quota filling.
    if near_clone_corr_limit is not None and corr is not None and not corr.empty:
        cluster_to_candidates: dict[object, list[str]] = {}
        cluster_of_code: dict[str, object] = {}
        for _, row in work_df.iterrows():
            members = list(row["candidate_members"] or [])
            if not members:
                continue
            ordered = global_return_scores.reindex(members).fillna(0.0)
            ordered = ordered.sort_values(ascending=False).index.tolist()
            cluster_to_candidates[row["cluster"]] = ordered
            for m in ordered:
                cluster_of_code[m] = row["cluster"]

        def _passes_near_clone(code: str, kept: list[str]) -> bool:
            if code not in corr.index:
                return True
            already = [c for c in kept if c in corr.columns]
            if not already:
                return True
            cross_limit = float(near_clone_corr_limit)
            intra_limit = (
                float(intra_cluster_corr_limit)
                if intra_cluster_corr_limit is not None
                else cross_limit
            )
            code_cluster = cluster_of_code.get(code)
            for peer in already:
                peer_cluster = cluster_of_code.get(peer)
                limit = intra_limit if (
                    code_cluster is not None and peer_cluster == code_cluster
                ) else cross_limit
                val = float(abs(corr.loc[code, peer]))
                if val > limit:
                    return False
            return True

        dedup: list[str] = []
        clusters_with_rep: set[object] = set()
        for code in final_selected:
            if _passes_near_clone(code, dedup):
                dedup.append(code)
                clusters_with_rep.add(cluster_of_code.get(code))

        # Coverage rescue: a cluster that lost its rep to the trim tries
        # lower-ranked members until one passes the correlation test.
        for cluster_id, ordered_members in cluster_to_candidates.items():
            if cluster_id in clusters_with_rep:
                continue
            chosen_from_cluster = [c for c in final_selected if cluster_of_code.get(c) == cluster_id]
            chosen_set = set(chosen_from_cluster)
            for rank_index, candidate in enumerate(ordered_members):
                if candidate in chosen_set:
                    continue
                if candidate in dedup:
                    continue
                if candidate not in returns.columns:
                    continue
                if _passes_near_clone(candidate, dedup):
                    dedup.append(candidate)
                    clusters_with_rep.add(cluster_id)
                    short_history = float(history_days.get(candidate, 0.0)) < MIN_SELECTION_HISTORY_DAYS
                    short_suffix = "_short_history" if short_history else ""
                    dedup_reason_overrides[candidate] = (
                        f"multi_rep_{weight_scheme}_cluster_{cluster_id}_rank_{rank_index + 1}_coverage_rescue{short_suffix}"
                    )
                    break

        final_selected = dedup

    reasons: dict[str, str] = {}
    for code in final_selected:
        if code in dedup_reason_overrides:
            reasons[code] = dedup_reason_overrides[code]
        else:
            reasons[code] = selected_reason_map.get(code, f"multi_rep_{weight_scheme}")
    return final_selected, reasons


def prepare_recent_window(
    close: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    recent_rows = max(int(lookback_days) + 1, 2)
    recent_close = close.sort_index().tail(recent_rows)
    recent_volume = volume_df.sort_index().reindex(recent_close.index) if volume_df is not None else None
    returns = recent_close.pct_change().replace([np.inf, -np.inf], np.nan)
    return recent_close, recent_volume, returns


def pairwise_overlap_matrices(
    returns: pd.DataFrame,
    min_overlap_days: int = MIN_PAIR_OVERLAP_DAYS,
    low_overlap_distance: float = LOW_OVERLAP_DISTANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fallback_corr = 1.0 - float(low_overlap_distance)
    overlap = returns.notna().astype(int).T.dot(returns.notna().astype(int))
    corr = returns.corr(min_periods=min_overlap_days).fillna(fallback_corr)
    corr = corr.clip(lower=-1.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)

    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    return corr, overlap, dist


def subset_pairwise(
    corr: pd.DataFrame,
    overlap: pd.DataFrame,
    codes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice precomputed pair matrices. Pairwise corr/overlap do not depend on other columns."""
    ordered = [code for code in codes if code in corr.index and code in corr.columns and code in overlap.index and code in overlap.columns]
    return corr.loc[ordered, ordered].copy(), overlap.loc[ordered, ordered].copy()


def summarize_overlap_counts(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in overlap.index:
        values = overlap.loc[code].drop(index=code, errors="ignore")
        values = values[values > 0]
        rows.append(
            {
                "code": code,
                "overlap_min_days": int(values.min()) if not values.empty else 0,
                "overlap_median_days": float(values.median()) if not values.empty else 0.0,
                "overlap_max_days": int(values.max()) if not values.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _pick_cluster_representatives(
    members: list[str],
    returns: pd.DataFrame,
    per_cluster: int,
    history_days: pd.Series | None = None,
) -> list[str]:
    if not members:
        return []
    if history_days is None:
        history_days = pd.Series(dtype=float)
    eligible_members = [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
    candidate_members = eligible_members or members
    return _return_scores(candidate_members, returns).head(per_cluster).index.tolist()


def _trim_selected(
    selected: list[str],
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    target_count: int | None,
) -> list[str]:
    if target_count is None or target_count <= 0:
        return selected
    selected = list(dict.fromkeys(selected))
    if len(selected) <= target_count:
        return selected

    candidates = list(selected)
    mean_returns = returns[candidates].mean().reindex(candidates).fillna(0.0)
    final_selected: list[str] = []

    for max_abs_corr_limit in FINAL_MAX_ABS_CORR_THRESHOLDS:
        if len(final_selected) >= target_count:
            break
        while len(final_selected) < target_count:
            available_selected = [picked for picked in final_selected if picked in returns.columns]
            ranked_candidates = []
            for code in candidates:
                if code in final_selected:
                    continue
                if available_selected and code in corr.index:
                    corr_slice = corr.loc[code, available_selected]
                    if isinstance(corr_slice, pd.Series):
                        max_abs_corr = float(corr_slice.abs().max()) if not corr_slice.empty else 0.0
                    else:
                        max_abs_corr = float(abs(corr_slice))
                else:
                    max_abs_corr = 0.0

                if max_abs_corr <= max_abs_corr_limit:
                    ranked_candidates.append((code, max_abs_corr, float(mean_returns.get(code, 0.0))))

            if not ranked_candidates:
                break

            ranked_candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
            final_selected.append(ranked_candidates[0][0])

    return final_selected


def cluster_and_select(
    returns: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    t: float = DIST_T,
    per_cluster: int = PER_CLUSTER,
    rule: str = RULE,
    target_count: int | None = None,
    history_days: pd.Series | None = None,
    corr: pd.DataFrame | None = None,
    overlap: pd.DataFrame | None = None,
    linkage_method: str = "ward",
):
    """Run hierarchical clustering and selection.

    1. hierarchical clustering over all ``returns`` columns (Ward by default)
    2. inside each cluster, score by (Sharpe + 12-1 momentum), pick top-K
    3. final near-clone dedup with ``FINAL_MAX_ABS_CORR_THRESHOLDS``

    Returns ``(Z, clusters, cluster_df, selected, reasons)``.
    """
    if corr is None or overlap is None:
        corr, overlap, _ = pairwise_overlap_matrices(returns, MIN_PAIR_OVERLAP_DAYS, LOW_OVERLAP_DISTANCE)

    codes = list(returns.columns)
    dist = 1.0 - corr
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=str(linkage_method).strip().lower())
    labels = fcluster(Z, t=t, criterion="distance")
    clusters = pd.Series(labels, index=codes, name="cluster")

    rows = []
    for cid, members in clusters.groupby(clusters).groups.items():
        members = list(members)
        n = len(members)
        if n <= 1:
            mean_corr = 1.0
        else:
            subcorr = corr.loc[members, members].values
            mean_corr = (subcorr.sum() - n) / (n * (n - 1))
        mean_volatility = returns[members].std().mean()
        mean_return = returns[members].mean().mean()
        try:
            avg_volume = volume_df[members].mean().mean()
        except Exception:
            avg_volume = None
        member_history = pd.Series(dtype=float) if history_days is None else history_days.reindex(members).fillna(0.0)
        rows.append(
            {
                "cluster": cid,
                "members": members,
                "n": n,
                "mean_corr": mean_corr,
                "mean_volatility": mean_volatility,
                "mean_return": mean_return,
                "avg_volume": avg_volume,
                "cluster_min_history_days": float(member_history.min()) if not member_history.empty else 0.0,
                "cluster_median_history_days": float(member_history.median()) if not member_history.empty else 0.0,
            }
        )
    cluster_df = pd.DataFrame(rows).sort_values(["mean_corr", "n"], ascending=[False, False])

    selection_rule = str(rule or RULE).strip().lower()
    multi_rep_rule_to_scheme = {
        "multi_rep_equal_mainline": "equal",
        "multi_rep_equal": "equal",
        "equal": "equal",
        "multi_rep_exp_benchmark": "exp",
        "multi_rep_exp": "exp",
        "exp": "exp",
        "multi_rep_harmonic_deprecated": "harmonic",
        "multi_rep_harmonic": "harmonic",
        "harmonic": "harmonic",
    }

    if selection_rule in multi_rep_rule_to_scheme:
        selected, reasons = _build_multi_rep_cluster_pool(
            cluster_df=cluster_df,
            returns=returns,
            history_days=history_days.reindex(codes).fillna(0.0) if history_days is not None else pd.Series(dtype=float),
            target_count=target_count,
            weight_scheme=multi_rep_rule_to_scheme[selection_rule],
            exp_base=MULTI_REP_EXP_BASE,
            corr=corr,
            near_clone_corr_limit=(
                float(FINAL_MAX_ABS_CORR_THRESHOLDS[int(os.environ.get("POOL_NEAR_CLONE_IDX", 0))])
                if FINAL_MAX_ABS_CORR_THRESHOLDS
                else None
            ),
            intra_cluster_corr_limit=float(INTRA_CLUSTER_MAX_ABS_CORR),
        )
    else:
        selected = []
        reasons = {}
        for _, row in cluster_df.iterrows():
            members = row["members"]
            cid = row["cluster"]
            if len(members) == 1:
                pick = members[0]
                selected.append(pick)
                if history_days is not None and float(history_days.get(pick, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[pick] = f"only_member_cluster_{cid}_short_history"
                else:
                    reasons[pick] = f"only_member_cluster_{cid}"
                continue
            picks = _pick_cluster_representatives(members, returns, per_cluster, history_days=history_days)
            for p in picks:
                selected.append(p)
                if history_days is not None and float(history_days.get(p, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[p] = f"return_top_cluster_{cid}_short_history"
                else:
                    reasons[p] = f"return_top_cluster_{cid}"

        selected = list(dict.fromkeys(selected))
        selected = _trim_selected(selected, returns, corr, target_count)

    return Z, clusters, cluster_df, selected, reasons
