"""Monthly frozen walk-forward for Shenwan industry regime transitions.

Equivalent to ``etf-daily regime --skip-rebuild --skip-html``: score the
selected pool, calibrate CONFIRM/OBSERVE, apply the extreme-drop override,
and write the decision pack. Reversal-label rebuild and HTML are not run.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sw_daily.paths import REGIME_DIR
from sw_daily.regime.board import (
    FrozenRegimeTransitionModel,
    RegimeTransitionConfig,
    compute_frozen_transition_step,
    compute_predictive_moments_next,
    config_hash_for_regime,
    fit_frozen_regime_model,
)
from sw_daily.regime.validation import (
    attach_instrument_names,
    build_expanding_crossfit_predictions,
    cluster_positive_events,
    compute_trend_up_label,
    eligible_codes_on_date,
    last_trading_day_before_month,
    match_alerts_to_events,
    maybe_isotonic_calibrate,
    month_key,
    select_fdr_threshold,
    summarize_calibration,
    trading_days_in_month,
)

CACHE_SCHEMA = "regime_frozen_hmm_v3"
FEATURE_COLS = [
    "p_into_reversal_adj",
    "p_into_reversal_hmm",
    "p_state_reversal",
    "p_switch_hmm",
    "p_no_exit_given_reversal_h10",
    "emission_surprise",
    "transition_score",
    "return_z",
    "direction_up",
    "pred_cdf",
    "quantile_switch",
]
DEFAULT_EXTREME_DROP_THRESHOLD = -0.07


def load_pool_codes(selected_csv: Path | None = None) -> tuple[list[str], dict[str, str]]:
    """Selected second-level industries and their Chinese names."""
    from sw_daily.pool.selector import CSV_SELECTED_OUT, load_sw_universe

    path = Path(selected_csv) if selected_csv is not None else CSV_SELECTED_OUT
    if not path.is_file():
        raise SystemExit(f"missing {path}; run: sw-daily pool")
    frame = pd.read_csv(path, dtype=str)
    col = "code" if "code" in frame.columns else frame.columns[0]
    codes = list(dict.fromkeys(c for c in frame[col].astype(str).str.strip() if c and c != "nan"))
    _, name_map = load_sw_universe()
    return codes, name_map


def _candidate_mask(frame: pd.DataFrame) -> pd.Series:
    rz = pd.to_numeric(frame.get("return_z"), errors="coerce")
    p_adj = pd.to_numeric(frame.get("p_into_reversal_adj"), errors="coerce")
    score = pd.to_numeric(frame.get("transition_score"), errors="coerce")
    direction = frame.get("transition_direction")
    up = frame.get("direction_up")
    if up is None:
        up = direction.astype(str).eq("up") if direction is not None else False
    else:
        up = up.astype(bool)
    qsw = frame.get("quantile_switch")
    qsw_b = qsw.astype(bool) if qsw is not None else False
    return qsw_b | ((rz >= 1.5) & up) | (p_adj >= 0.20) | ((score >= 0.50) & up)


def _data_fingerprint(close: pd.Series, train_end: pd.Timestamp) -> str:
    series = pd.to_numeric(close, errors="coerce").dropna()
    hist = series.loc[series.index <= train_end]
    if hist.empty:
        return "empty"
    vals = hist.to_numpy(dtype=float)
    digest = hashlib.sha256()
    digest.update(str(len(vals)).encode())
    digest.update(str(hist.index[0]).encode())
    digest.update(str(hist.index[-1]).encode())
    digest.update(vals.tobytes())
    return digest.hexdigest()[:16]


def _cache_path(cache_dir: Path, *, code: str, train_end: str, cfg_hash: str, data_fp: str) -> Path:
    key = f"{CACHE_SCHEMA}_{code}_{train_end}_{cfg_hash}_{data_fp}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return cache_dir / f"{digest}.pkl"


def _load_cached_model(path: Path) -> FrozenRegimeTransitionModel | None:
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, FrozenRegimeTransitionModel):
            return None
        return obj
    except Exception:
        return None


def _save_cached_model(path: Path, model: FrozenRegimeTransitionModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _fit_one_symbol(
    code: str,
    close: pd.Series,
    train_end: pd.Timestamp,
    cfg: RegimeTransitionConfig,
    cache_dir: Path | None,
    use_cache: bool,
) -> tuple[str, FrozenRegimeTransitionModel | None, str]:
    cfg_hash = config_hash_for_regime(cfg)
    data_fp = _data_fingerprint(close, train_end)
    cache_hit = "miss"
    if use_cache and cache_dir is not None:
        path = _cache_path(cache_dir, code=code, train_end=train_end.strftime("%Y-%m-%d"), cfg_hash=cfg_hash, data_fp=data_fp)
        cached = _load_cached_model(path)
        if cached is not None:
            return code, cached, "hit"
    try:
        model = fit_frozen_regime_model(close, train_end, cfg=cfg)
    except Exception as exc:  # pragma: no cover
        return code, None, f"error:{exc}"
    if use_cache and cache_dir is not None:
        path = _cache_path(cache_dir, code=code, train_end=train_end.strftime("%Y-%m-%d"), cfg_hash=cfg_hash, data_fp=data_fp)
        _save_cached_model(path, model)
    return code, model, cache_hit


def run_month_symbol_rows(
    code: str,
    close: pd.Series,
    days: pd.DatetimeIndex,
    model: FrozenRegimeTransitionModel,
    cfg: RegimeTransitionConfig,
    *,
    horizon: int,
    kappa_ret: float,
    kappa_eff: float,
    kappa_mae: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alpha = None
    for ts in days:
        metrics, alpha, _step = compute_frozen_transition_step(close, ts, model, cfg=cfg, alpha_prev=alpha)
        if not metrics.reliable:
            continue
        pred_mean_next, pred_scale_next = compute_predictive_moments_next(alpha, model)
        label = compute_trend_up_label(
            close, ts, horizon=horizon, kappa_ret=kappa_ret, kappa_eff=kappa_eff, kappa_mae=kappa_mae
        )
        rows.append(
            {
                "as_of": pd.Timestamp(ts).strftime("%Y-%m-%d"),
                "month": month_key(ts),
                "code": code,
                "regime_now": metrics.regime_now,
                "prev_regime": metrics.prev_regime,
                "p_switch_hmm": metrics.p_switch_hmm,
                "p_into_reversal_hmm": metrics.p_into_reversal_hmm,
                "p_state_reversal": metrics.p_state_reversal,
                "p_enter_current_prior": metrics.p_enter_current_prior,
                "p_transition": metrics.p_transition,
                "transition_score": metrics.transition_score,
                "transition_direction": metrics.transition_direction,
                "emission_surprise": metrics.emission_surprise,
                "p_no_exit_given_reversal_h5": metrics.p_no_exit_given_reversal_h5,
                "p_no_exit_given_reversal_h10": metrics.p_no_exit_given_reversal_h10,
                "p_no_exit_given_reversal_h20": metrics.p_no_exit_given_reversal_h20,
                "p_in_reversal_and_no_exit_h5": metrics.p_in_reversal_and_no_exit_h5,
                "p_in_reversal_and_no_exit_h10": metrics.p_in_reversal_and_no_exit_h10,
                "p_in_reversal_and_no_exit_h20": metrics.p_in_reversal_and_no_exit_h20,
                "regime_separated": metrics.regime_separated,
                "label_ambiguous": metrics.label_ambiguous,
                "ambiguity_reasons": "|".join(metrics.ambiguity_reasons),
                "model_train_end": metrics.model_train_end,
                "config_hash": metrics.config_hash,
                "return_z": metrics.return_z,
                "p_into_reversal_adj": metrics.p_into_reversal_adj,
                "direction_up": bool(metrics.direction_up),
                "pred_cdf": metrics.pred_cdf,
                "quantile_switch": bool(metrics.quantile_switch),
                "switch_side": metrics.switch_side,
                "regime_hmm_raw": metrics.regime_hmm_raw,
                "pred_log_density": metrics.pred_log_density,
                "pred_mean": metrics.pred_mean,
                "pred_scale": metrics.pred_scale,
                "pred_mean_next": pred_mean_next,
                "pred_scale_next": pred_scale_next,
                "label_trend_up": label,
            }
        )
    return rows


def _build_elig_by_day(
    codes: list[str],
    days: pd.DatetimeIndex,
    panel: pd.DataFrame,
    *,
    min_history: int,
    min_valid_in_20: int,
) -> dict[pd.Timestamp, set[str]]:
    return {
        d: set(eligible_codes_on_date(codes, d, panel, min_history=min_history, min_valid_in_20=min_valid_in_20))
        for d in days
    }


def _score_month_symbol(
    code: str,
    close: pd.Series,
    eligible_day_strs: list[str],
    model: FrozenRegimeTransitionModel,
    cfg: RegimeTransitionConfig,
    *,
    horizon: int,
    kappa_ret: float,
    kappa_eff: float,
    kappa_mae: float,
) -> list[dict[str, Any]]:
    if not eligible_day_strs:
        return []
    return run_month_symbol_rows(
        code,
        close,
        pd.DatetimeIndex(eligible_day_strs),
        model,
        cfg,
        horizon=horizon,
        kappa_ret=kappa_ret,
        kappa_eff=kappa_eff,
        kappa_mae=kappa_mae,
    )


def _apply_calibration_and_signals(
    all_rows: pd.DataFrame,
    *,
    target_fdr: float,
    bootstrap_seed: int,
    q_hmm: float,
    q_observe: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {}
    if all_rows.empty:
        return all_rows, {"confirm_suppressed_reason": "no_rows"}

    mature = all_rows[all_rows["label_trend_up"].notna()].copy()
    for col in FEATURE_COLS:
        if col not in mature.columns:
            mature[col] = np.nan
    mature["direction_up"] = mature.get("direction_up", False)
    if "direction_up" in mature.columns:
        mature["direction_up"] = mature["direction_up"].astype(float)
    mature[FEATURE_COLS] = mature[FEATURE_COLS].astype(float).fillna(0.0)

    cross = build_expanding_crossfit_predictions(
        mature, feature_cols=FEATURE_COLS, label_col="label_trend_up", month_col="month", min_train_rows=80
    )
    if cross.empty:
        out = all_rows.copy()
        out["p_trend_raw"] = np.nan
        out["p_trend_cal"] = np.nan
        out["signal_level"] = "NONE"
        out["is_candidate"] = False
        meta["confirm_suppressed_reason"] = "insufficient_crossfit"
        return out, meta

    raw = cross["p_trend_raw"].to_numpy(dtype=float)
    if float(np.nanstd(raw)) < 0.02:
        cal_scores, used_iso = raw, False
    else:
        cal_scores, used_iso = maybe_isotonic_calibrate(
            raw, cross["label_trend_up"].astype(int).to_numpy(), raw, min_rows=500, min_pos=50, min_top_bin=30
        )
    cross["p_trend_cal"] = cal_scores
    cross["is_candidate"] = _candidate_mask(cross)
    meta["used_isotonic"] = used_iso
    meta["n_crossfit_months"] = int(cross["month"].nunique())
    meta["n_candidates"] = int(cross["is_candidate"].sum())

    cand = cross[cross["is_candidate"]].copy()
    if "quantile_switch" in cand.columns and "switch_side" in cand.columns:
        up_sw = cand[cand["quantile_switch"].astype(bool) & (cand["switch_side"].astype(str) == "up")].copy()
    else:
        up_sw = cand[cand["direction_up"].astype(bool)].copy()
    thr = select_fdr_threshold(
        up_sw if len(up_sw) >= 30 else cand,
        score_col="p_trend_cal",
        label_col="label_trend_up",
        target_fdr=target_fdr,
        min_confirm=30,
        seed=bootstrap_seed,
    )
    if thr.get("q_confirm") is None:
        thr = _fallback_threshold(up_sw if len(up_sw) >= 20 else cand, thr, target_fdr=target_fdr)
    else:
        thr["threshold_mode"] = "fdr"
        thr.setdefault("z_confirm", None)
        thr.setdefault("cdf_confirm", None)
    meta.update(thr)

    score_map = cross.set_index(["as_of", "code"])["p_trend_cal"]
    raw_map = cross.set_index(["as_of", "code"])["p_trend_raw"]
    out = all_rows.copy()
    keys = list(zip(out["as_of"].astype(str), out["code"].astype(str)))
    cross_index = score_map.index
    # set_index may have duplicated keys; last wins via dict
    score_lookup = dict(zip(cross_index, score_map.to_numpy()))
    raw_lookup = dict(zip(raw_map.index, raw_map.to_numpy()))
    out["p_trend_cal"] = [score_lookup.get(k, np.nan) for k in keys]
    out["p_trend_raw"] = [raw_lookup.get(k, np.nan) for k in keys]
    out["is_candidate"] = _candidate_mask(out)
    out["signal_level"] = [
        _signal_level(row, meta, q_hmm=q_hmm, q_observe=q_observe) for _, row in out.iterrows()
    ]
    return out, meta


def _fallback_threshold(pool: pd.DataFrame, thr: dict[str, Any], *, target_fdr: float) -> dict[str, Any]:
    base = float(pool["label_trend_up"].astype(bool).mean()) if len(pool) else 0.0
    rz = pd.to_numeric(pool.get("return_z"), errors="coerce")
    up = pool["direction_up"].astype(bool) if "direction_up" in pool.columns else True
    z_mask = (rz >= 2.5) & up
    z_prec = float(pool.loc[z_mask, "label_trend_up"].astype(bool).mean()) if int(z_mask.sum()) >= 10 else base
    best = None
    best_n = 0
    best_prec = -1.0
    best_mode = "none"
    best_kind = None
    best_thr_val = None

    if "pred_cdf" in pool.columns and len(pool) >= 20:
        cdf = pd.to_numeric(pool["pred_cdf"], errors="coerce")
        for q_cdf in (0.995, 0.99, 0.97, 0.95):
            m = cdf >= float(q_cdf)
            n = int(m.sum())
            if n < 15:
                continue
            prec = float(pool.loc[m, "label_trend_up"].astype(bool).mean())
            if prec >= z_prec + 0.05 and prec >= best_prec:
                best, best_n, best_prec = float(q_cdf), n, prec
                best_mode, best_kind, best_thr_val = "beat_z_baseline", "cdf", float(q_cdf)
            elif best_mode != "beat_z_baseline" and prec >= max(base + 0.05, z_prec) and prec >= best_prec:
                best, best_n, best_prec = float(q_cdf), n, prec
                best_mode, best_kind, best_thr_val = "precision_lift", "cdf", float(q_cdf)

    if best is None and len(pool) >= 30 and "p_trend_cal" in pool.columns:
        scores = np.sort(pool["p_trend_cal"].astype(float).unique())[::-1]
        for q in scores:
            m = pool["p_trend_cal"].astype(float) >= float(q)
            n = int(m.sum())
            if n < 20:
                continue
            prec = float(pool.loc[m, "label_trend_up"].astype(bool).mean())
            if prec >= z_prec + 0.05 and (best is None or prec > best_prec or (prec == best_prec and n > best_n)):
                best, best_n, best_prec = float(q), n, prec
                best_mode, best_kind, best_thr_val = "beat_z_baseline", "p_cal", float(q)
            elif best_mode != "beat_z_baseline" and prec >= max(base + 0.05, z_prec) and prec >= best_prec:
                best, best_n, best_prec = float(q), n, prec
                best_mode, best_kind, best_thr_val = "precision_lift", "p_cal", float(q)

    if best is None or best_mode == "none":
        for z_thr in (4.0, 3.5, 3.0, 2.75, 2.5):
            m = (rz >= float(z_thr)) & up
            if "label_ambiguous" in pool.columns:
                m = m & ~pool["label_ambiguous"].astype(bool)
            n = int(m.sum())
            if n < 20:
                continue
            prec = float(pool.loc[m, "label_trend_up"].astype(bool).mean())
            if prec >= z_prec + 0.05 and prec >= best_prec:
                best, best_n, best_prec = float(z_thr), n, prec
                best_mode, best_kind, best_thr_val = "z_ladder_beat", "z", float(z_thr)
            elif best_mode not in ("beat_z_baseline", "precision_lift", "z_ladder_beat") and prec >= z_prec and prec >= best_prec:
                best, best_n, best_prec = float(z_thr), n, prec
                best_mode, best_kind, best_thr_val = "z_ladder_match", "z", float(z_thr)

    common = {
        "confirm_suppressed_reason": None,
        "n_confirm_support": best_n,
        "target_fdr": target_fdr,
        "candidate_base_rate": base,
        "z_baseline_precision": z_prec,
        "precision_at_q": best_prec,
    }
    if best is not None and best_kind == "cdf":
        return {**common, "q_confirm": None, "z_confirm": None, "cdf_confirm": best_thr_val, "threshold_mode": f"pred_cdf_{best_mode}"}
    if best is not None and best_kind == "z":
        return {**common, "q_confirm": None, "z_confirm": best_thr_val, "cdf_confirm": None, "threshold_mode": best_mode}
    if best is not None:
        return {**common, "q_confirm": best_thr_val, "z_confirm": None, "cdf_confirm": None, "threshold_mode": best_mode}
    thr = dict(thr)
    thr["threshold_mode"] = "none"
    thr["z_baseline_precision"] = z_prec
    thr["candidate_base_rate"] = base
    thr.setdefault("cdf_confirm", None)
    return thr


def _signal_level(row: pd.Series, meta: dict[str, Any], *, q_hmm: float, q_observe: float) -> str:
    p = row.get("p_trend_cal")
    p_adj = row.get("p_into_reversal_adj")
    p_into = row.get("p_into_reversal_hmm")
    amb = bool(row.get("label_ambiguous"))
    score = row.get("transition_score")
    direction = str(row.get("transition_direction") or "")
    rz = row.get("return_z")
    cand = bool(row.get("is_candidate"))
    up = bool(row.get("direction_up")) or direction == "up"
    q_switch = bool(row.get("quantile_switch"))
    switch_side = str(row.get("switch_side") or "")
    pred_cdf = row.get("pred_cdf")
    q_confirm = meta.get("q_confirm")
    z_confirm = meta.get("z_confirm")
    cdf_confirm = meta.get("cdf_confirm")

    strong_jump = (
        score is not None
        and float(score) >= 0.70
        and direction == "up"
        and rz is not None
        and math.isfinite(float(rz))
        and float(rz) >= 2.0
    )
    observe_hit = q_switch or strong_jump
    if p is None or (isinstance(p, float) and not math.isfinite(float(p))):
        return "OBSERVE" if observe_hit else "NONE"
    p = float(p)
    hmm_ok = q_switch or (
        p_adj is not None and math.isfinite(float(p_adj)) and float(p_adj) >= float(q_hmm)
    ) or (
        score is not None
        and float(score) >= 0.50
        and direction == "up"
        and rz is not None
        and math.isfinite(float(rz))
        and float(rz) >= 1.5
    )
    months_ok = meta.get("n_crossfit_months", 0) >= 6
    cal_confirm = (
        q_confirm is not None and cand and up and q_switch and switch_side == "up" and p >= float(q_confirm) and hmm_ok and not amb and months_ok
    )
    cdf_gate_confirm = (
        cdf_confirm is not None
        and q_switch
        and switch_side == "up"
        and pred_cdf is not None
        and math.isfinite(float(pred_cdf))
        and float(pred_cdf) >= float(cdf_confirm)
        and months_ok
    )
    z_gate_confirm = (
        z_confirm is not None
        and cand
        and up
        and q_switch
        and rz is not None
        and math.isfinite(float(rz))
        and float(rz) >= float(z_confirm)
        and hmm_ok
        and not amb
        and months_ok
    )
    if cal_confirm or cdf_gate_confirm or z_gate_confirm:
        return "CONFIRM"
    if observe_hit or p >= float(q_observe):
        return "OBSERVE"
    if p_into is not None and float(p_into) >= float(q_hmm) and p < float(q_confirm or q_observe):
        return "REJECT_AS_ONE_DAY_JUMP"
    return "NONE"


def evaluate_go_no_go(
    signals: pd.DataFrame,
    *,
    target_fdr: float = 0.35,
    max_fp_per_code_year: float = 4.0,
) -> dict[str, Any]:
    """Go/No-Go without the ETF anchor row. Switch coverage is the primary gate."""
    out: dict[str, Any] = {"checks": {}, "pass": False}
    if signals.empty:
        out["reason"] = "empty_signals"
        return out

    labeled = signals[signals["label_trend_up"].notna()].copy()
    confirm = labeled[labeled["signal_level"] == "CONFIRM"]
    qsw_rate = float(signals["quantile_switch"].astype(bool).mean()) if "quantile_switch" in signals.columns else float("nan")
    check_qsw = bool(math.isfinite(qsw_rate) and 0.08 <= qsw_rate <= 0.35)
    out["checks"]["switch_coverage"] = {
        "pass": check_qsw,
        "quantile_switch_rate": qsw_rate,
        "expected_band": [0.08, 0.35],
        "q_star": 0.90,
        "bilateral": True,
    }

    pool = labeled
    if "is_candidate" in labeled.columns and labeled["is_candidate"].notna().any():
        cand_pool = labeled[labeled["is_candidate"].astype(bool)]
        if len(cand_pool) >= 50:
            pool = cand_pool
    prec_model = float(confirm["label_trend_up"].astype(bool).mean()) if len(confirm) >= 10 else float("nan")
    rz = pd.to_numeric(pool.get("return_z"), errors="coerce")
    if "direction_up" in pool.columns:
        z_base = pool[(rz >= 2.5) & (pool["direction_up"].astype(bool))]
    else:
        z_base = pool[rz >= 2.5]
    prec_z = float(z_base["label_trend_up"].astype(bool).mean()) if len(z_base) else float("nan")
    lift = prec_model - prec_z if math.isfinite(prec_model) and math.isfinite(prec_z) else float("nan")
    lift_ci_low = float("nan")
    if len(confirm) >= 10 and len(z_base) >= 10 and "month" in labeled.columns:
        rng = np.random.default_rng(42)
        months = labeled["month"].astype(str).unique()
        boots = []
        for _ in range(200):
            chosen = rng.choice(months, size=len(months), replace=True)
            sub = labeled[labeled["month"].astype(str).isin(chosen)]
            c = sub[sub["signal_level"] == "CONFIRM"]
            sub_pool = sub
            if "is_candidate" in sub.columns:
                sp = sub[sub["is_candidate"].astype(bool)]
                if len(sp) >= 20:
                    sub_pool = sp
            zb = sub_pool[(pd.to_numeric(sub_pool["return_z"], errors="coerce") >= 2.5) & (sub_pool["direction_up"].astype(bool))]
            if len(c) < 5 or len(zb) < 5:
                continue
            boots.append(float(c["label_trend_up"].astype(bool).mean()) - float(zb["label_trend_up"].astype(bool).mean()))
        if boots:
            lift_ci_low = float(np.quantile(boots, 0.025))

    confirm_suppressed = len(confirm) == 0
    if confirm_suppressed:
        check_precision = True
        out["checks"]["precision_note"] = "confirm_suppressed_switch_only"
    else:
        check_precision = bool(math.isfinite(lift) and lift >= 0.05 and math.isfinite(lift_ci_low) and lift_ci_low > 0)
        if not check_precision and math.isfinite(lift) and lift >= 0.05 and len(confirm) >= 15:
            check_precision = True
        if (
            not check_precision
            and math.isfinite(prec_model)
            and math.isfinite(prec_z)
            and prec_model >= prec_z
            and prec_model >= 0.30
            and len(confirm) >= 20
        ):
            check_precision = True
    out["checks"]["precision"] = {
        "pass": check_precision,
        "precision_confirm": prec_model,
        "precision_z_baseline": prec_z,
        "lift": lift,
        "lift_ci_low": lift_ci_low,
        "n_confirm": int(len(confirm)),
        "confirm_suppressed": confirm_suppressed,
    }

    cal_rows = labeled.dropna(subset=["p_trend_cal"]) if "p_trend_cal" in labeled.columns else labeled.iloc[0:0]
    if "is_candidate" in cal_rows.columns:
        cal_rows = cal_rows[cal_rows["is_candidate"].astype(bool)]
    if len(cal_rows) >= 50:
        p = cal_rows["p_trend_cal"].astype(float).to_numpy()
        y = cal_rows["label_trend_up"].astype(float).to_numpy()
        brier = float(np.mean((p - y) ** 2))
        bins = np.linspace(0, 1, 6)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if m.any():
                ece += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / p.size)
        check_cal = bool(brier <= 0.25 and float(ece) <= 0.15)
    else:
        brier, ece = float("nan"), float("nan")
        check_cal = False
    out["checks"]["calibration"] = {"pass": check_cal, "brier": brier, "ece": ece, "n": int(len(cal_rows))}

    if confirm_suppressed:
        check_fp, fp_rate = True, 0.0
    elif len(confirm):
        years = pd.to_datetime(confirm["as_of"]).dt.year.max() - pd.to_datetime(confirm["as_of"]).dt.year.min() + 1
        n_codes = max(int(signals["code"].nunique()), 1)
        fp = int((~confirm["label_trend_up"].astype(bool)).sum())
        fp_rate = fp / max(years, 1) / n_codes
        check_fp = fp_rate <= max_fp_per_code_year
    else:
        fp_rate, check_fp = 0.0, False
    out["checks"]["false_alarms"] = {"pass": check_fp, "fp_per_code_year": fp_rate, "limit": max_fp_per_code_year}

    if confirm_suppressed:
        check_thr = True
    else:
        check_thr = len(confirm) >= 10
    out["checks"]["threshold"] = {"pass": check_thr, "n_confirm": int(len(confirm)), "target_fdr": target_fdr}
    out["pass"] = bool(check_qsw and check_precision and check_cal and check_fp and check_thr)
    out["n_observe"] = int((signals["signal_level"] == "OBSERVE").sum())
    out["n_confirm"] = int((signals["signal_level"] == "CONFIRM").sum())
    out["mvp_mode"] = "quantile_switch_observe"
    return out


def day_returns_frame_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for code in panel.columns:
        series = pd.to_numeric(panel[code], errors="coerce")
        series.index = pd.to_datetime(series.index).normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        ret = series.pct_change()
        if ret.dropna().empty:
            continue
        rows.append(pd.DataFrame({"code": str(code), "as_of": ret.index, "day_ret": ret.to_numpy(dtype=float)}))
    if not rows:
        return pd.DataFrame(columns=["code", "as_of", "day_ret"])
    return pd.concat(rows, ignore_index=True)


def apply_extreme_drop_override(
    signals: pd.DataFrame,
    day_ret: pd.DataFrame,
    *,
    threshold: float = DEFAULT_EXTREME_DROP_THRESHOLD,
) -> pd.DataFrame:
    """One red triangle per day: model switch, else a day return at or below -7%."""
    if signals.empty:
        return signals
    out = signals.copy()
    out["as_of"] = pd.to_datetime(out["as_of"]).dt.normalize()
    out["code"] = out["code"].astype(str)
    out["model_quantile_switch"] = out.get("quantile_switch", False)
    out["model_quantile_switch"] = out["model_quantile_switch"].astype(bool)
    out["model_switch_side"] = out.get("switch_side")
    ret = day_ret.copy()
    ret["as_of"] = pd.to_datetime(ret["as_of"]).dt.normalize()
    ret["code"] = ret["code"].astype(str)
    ret["day_ret"] = pd.to_numeric(ret["day_ret"], errors="coerce")
    ret = ret.drop_duplicates(subset=["code", "as_of"], keep="last")
    if "day_ret" in out.columns:
        out = out.drop(columns=["day_ret"])
    out = out.merge(ret[["code", "as_of", "day_ret"]], on=["code", "as_of"], how="left")
    model_down = out["model_quantile_switch"] & out["model_switch_side"].astype(str).eq("down")
    model_up = out["model_quantile_switch"] & out["model_switch_side"].astype(str).eq("up")
    extreme = out["day_ret"].notna() & (out["day_ret"] <= float(threshold)) & ~model_down
    out["extreme_drop_switch"] = extreme
    effective = out["model_quantile_switch"] | out["extreme_drop_switch"]
    side = np.where(model_down, "down", np.where(model_up, "up", np.where(out["extreme_drop_switch"], "down", "")))
    source = np.where(
        model_down | model_up,
        "quantile_tail",
        np.where(out["extreme_drop_switch"], "extreme_drop_override", ""),
    )
    out["quantile_switch"] = effective.astype(bool)
    out["switch_side"] = pd.Series(side, index=out.index).replace({"": pd.NA})
    out["switch_source"] = pd.Series(source, index=out.index)
    out.loc[~out["quantile_switch"], "switch_source"] = ""
    return out


def coverage_split_stats(signals: pd.DataFrame) -> dict[str, Any]:
    if signals.empty:
        return {"n_rows": 0, "n_extreme_override": 0}
    n = len(signals)
    model = signals.get("model_quantile_switch", pd.Series(False, index=signals.index)).astype(bool)
    extreme = signals.get("extreme_drop_switch", pd.Series(False, index=signals.index)).astype(bool)
    effective = signals.get("quantile_switch", pd.Series(False, index=signals.index)).astype(bool)
    return {
        "n_rows": int(n),
        "model_switch_rate": float(model.mean()),
        "extreme_override_rate": float(extreme.mean()),
        "effective_switch_rate": float(effective.mean()),
        "n_extreme_override": int(extreme.sum()),
    }


def day_already_scored(shard_csv: Path, signals_oos: Path, eval_end: str) -> bool:
    end = pd.Timestamp(eval_end)
    if pd.isna(end):
        return False
    for path in (shard_csv, signals_oos):
        if not path.is_file():
            return False
        try:
            frame = pd.read_csv(path, usecols=["as_of"])
        except (ValueError, pd.errors.EmptyDataError, OSError):
            return False
        if frame.empty:
            return False
        dates = pd.to_datetime(frame["as_of"], errors="coerce").dropna()
        if dates.empty or dates.max() < end:
            return False
    return True


def run_validation(
    panel: pd.DataFrame,
    codes: list[str],
    *,
    name_map: dict[str, str] | None = None,
    cfg: RegimeTransitionConfig | None = None,
    eval_start: str = "2024-06-01",
    eval_end: str | None = None,
    horizon: int = 10,
    trim_for_labels: bool = False,
    kappa_ret: float = 0.35,
    kappa_eff: float = 0.28,
    kappa_mae: float = 1.0,
    target_fdr: float = 0.35,
    q_hmm: float = 0.20,
    q_observe: float = 0.28,
    bootstrap_seed: int = 42,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    jobs: int = 1,
    output_dir: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Score ``codes`` month by month and write the decision pack when ``output_dir`` is set."""
    cfg = cfg or RegimeTransitionConfig()
    name_map = name_map or {}
    keep = [c for c in codes if c in panel.columns]
    panel = panel[keep].copy()
    panel.index = pd.to_datetime(panel.index)

    cal = pd.DatetimeIndex(panel.index).sort_values()
    start_ts = pd.Timestamp(eval_start)
    end_ts = pd.Timestamp(eval_end) if eval_end else pd.Timestamp(cal[-1])
    if trim_for_labels and horizon > 0 and len(cal) > horizon:
        label_end = pd.Timestamp(cal[-1 - horizon])
        if end_ts > label_end:
            print(f"[INFO] trim eval_end {end_ts.date()} -> {label_end.date()} for horizon={horizon}")
            end_ts = label_end
    else:
        print(f"[INFO] eval_end={end_ts.date()} (no label trim); calendar last={cal[-1].date()}")

    months = sorted({month_key(d) for d in cal if start_ts <= pd.Timestamp(d) <= end_ts})
    shard_dir = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "universe_audit.json").write_text(
            json.dumps({"n_codes": len(keep), "codes": keep, "eval_start": eval_start, "eval_end": str(end_ts.date())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    cache_stats = {"hit": 0, "miss": 0, "error": 0}
    all_month_frames: list[pd.DataFrame] = []

    for month in months:
        train_end = last_trading_day_before_month(cal, month)
        if train_end is None:
            continue
        days = trading_days_in_month(cal, month)
        days = days[(days >= start_ts) & (days <= end_ts)]
        if days.empty:
            continue

        shard_path = shard_dir / f"{month}.signals.csv" if shard_dir else None
        done_marker = shard_dir / f"{month}.done" if shard_dir else None
        if resume and shard_path is not None and done_marker is not None and shard_path.exists() and done_marker.exists():
            all_month_frames.append(pd.read_csv(shard_path))
            continue

        existing_frame = None
        score_days = days
        if resume and shard_path is not None and shard_path.exists() and (done_marker is None or not done_marker.exists()):
            try:
                existing_frame = pd.read_csv(shard_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] month={month} failed to read shard: {exc}")
                existing_frame = None
            if existing_frame is not None and not existing_frame.empty and "as_of" in existing_frame.columns:
                last = pd.to_datetime(existing_frame["as_of"]).max()
                score_days = days[days > last]
                print(f"[INFO] month={month} day-resume after {last.date()} new_days={len(score_days)}")
                if score_days.empty:
                    if done_marker is not None:
                        done_marker.write_text("ok\n", encoding="utf-8")
                    all_month_frames.append(existing_frame)
                    continue
            else:
                existing_frame = None

        fit_jobs = []
        for code in keep:
            series = pd.to_numeric(panel[code], errors="coerce").dropna()
            if series.size < cfg.min_samples + horizon + 5:
                continue
            fit_jobs.append((code, series))

        models: dict[str, FrozenRegimeTransitionModel] = {}
        month_rows: list[dict[str, Any]] = []
        if jobs <= 1:
            for code, series in fit_jobs:
                c, model, status = _fit_one_symbol(code, series, train_end, cfg, cache_dir, use_cache)
                _count_cache(cache_stats, status)
                if model is not None:
                    models[c] = model
            elig_by_day = _build_elig_by_day(keep, score_days, panel, min_history=cfg.min_samples, min_valid_in_20=18)
            for code, model in models.items():
                series = pd.to_numeric(panel[code], errors="coerce").dropna()
                eligible_days = [d for d in score_days if code in elig_by_day[d]]
                if not eligible_days:
                    continue
                month_rows.extend(
                    run_month_symbol_rows(
                        code, series, pd.DatetimeIndex(eligible_days), model, cfg,
                        horizon=horizon, kappa_ret=kappa_ret, kappa_eff=kappa_eff, kappa_mae=kappa_mae,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futs = {
                    pool.submit(_fit_one_symbol, code, series, train_end, cfg, cache_dir, use_cache): code
                    for code, series in fit_jobs
                }
                for fut in as_completed(futs):
                    c, model, status = fut.result()
                    _count_cache(cache_stats, status)
                    if model is not None:
                        models[c] = model
                elig_by_day = _build_elig_by_day(keep, score_days, panel, min_history=cfg.min_samples, min_valid_in_20=18)
                score_futs = {}
                for code, model in models.items():
                    series = pd.to_numeric(panel[code], errors="coerce").dropna()
                    eligible_days = [d for d in score_days if code in elig_by_day[d]]
                    if not eligible_days:
                        continue
                    elig_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in eligible_days]
                    score_futs[pool.submit(
                        _score_month_symbol, code, series, elig_strs, model, cfg,
                        horizon=horizon, kappa_ret=kappa_ret, kappa_eff=kappa_eff, kappa_mae=kappa_mae,
                    )] = code
                for fut in as_completed(score_futs):
                    month_rows.extend(fut.result())

        frame = pd.DataFrame(month_rows)
        if existing_frame is not None and not existing_frame.empty:
            frame = pd.concat([existing_frame, frame], ignore_index=True)
            if not frame.empty and {"as_of", "code"}.issubset(frame.columns):
                frame = frame.sort_values(["as_of", "code"], kind="stable").drop_duplicates(subset=["as_of", "code"], keep="last")
        frame = attach_instrument_names(frame, name_map)
        if shard_path is not None:
            tmp = shard_path.with_suffix(".tmp")
            frame.to_csv(tmp, index=False, encoding="utf-8-sig")
            tmp.replace(shard_path)
            if done_marker is not None:
                done_marker.write_text("ok\n", encoding="utf-8")
        all_month_frames.append(frame)
        print(f"[INFO] month={month} train_end={train_end.date()} rows={len(frame)} models={len(models)} scored_days={len(score_days)}")

    signals = pd.concat(all_month_frames, ignore_index=True) if all_month_frames else pd.DataFrame()
    if not signals.empty:
        signals = signals.sort_values(["as_of", "code"], kind="stable").drop_duplicates(subset=["as_of", "code"], keep="last")
        signals = attach_instrument_names(signals, name_map)

    calibrated, cal_meta = _apply_calibration_and_signals(
        signals, target_fdr=target_fdr, bootstrap_seed=bootstrap_seed, q_hmm=q_hmm, q_observe=q_observe
    )
    if not calibrated.empty:
        calibrated = attach_instrument_names(calibrated, name_map)
        calibrated = apply_extreme_drop_override(calibrated, day_returns_frame_from_panel(panel))
        cal_meta["extreme_drop_override"] = coverage_split_stats(calibrated)

    event_df = pd.DataFrame()
    if not calibrated.empty and "signal_level" in calibrated.columns:
        label_frame = calibrated[["code", "as_of", "label_trend_up"]].dropna(subset=["label_trend_up"])
        events = cluster_positive_events(label_frame)
        alerts = calibrated[calibrated["signal_level"] == "CONFIRM"].copy()
        if alerts.empty:
            alerts = calibrated[calibrated["signal_level"].isin(["CONFIRM", "OBSERVE"])]
        event_df = match_alerts_to_events(alerts, events, lead=3, horizon=horizon, score_col="p_trend_cal")
        event_df = attach_instrument_names(event_df, name_map)

    labeled = calibrated.dropna(subset=["p_trend_cal", "label_trend_up"]) if not calibrated.empty and "p_trend_cal" in calibrated.columns else pd.DataFrame()
    cal_summary = summarize_calibration(labeled, score_col="p_trend_cal", label_col="label_trend_up")
    go_eval = evaluate_go_no_go(calibrated, target_fdr=target_fdr)
    go_no_go = "PASS" if go_eval.get("pass") else "BLOCKED"

    if output_dir is not None:
        _write_pack(output_dir, calibrated, event_df, cal_summary, cal_meta, go_eval, go_no_go, cfg, cache_stats, keep, eval_start, end_ts, horizon)

    return {
        "signals": calibrated,
        "event_matches": event_df,
        "calibration_summary": cal_summary,
        "cal_meta": cal_meta,
        "cache_stats": cache_stats,
        "go_no_go": go_no_go,
        "go_eval": go_eval,
        "months": months,
    }


def _count_cache(stats: dict[str, int], status: str) -> None:
    if status == "hit":
        stats["hit"] += 1
    elif status.startswith("error"):
        stats["error"] += 1
    else:
        stats["miss"] += 1


def _write_pack(output_dir, calibrated, event_df, cal_summary, cal_meta, go_eval, go_no_go, cfg, cache_stats, codes, eval_start, end_ts, horizon) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(output_dir / "signals_oos.csv", index=False, encoding="utf-8-sig")
    if not event_df.empty:
        event_df.to_csv(output_dir / "event_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cal_summary]).to_csv(output_dir / "calibration_by_fold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([cal_meta]).to_csv(output_dir / "threshold_tradeoff.csv", index=False, encoding="utf-8-sig")
    (output_dir / "go_nogo.json").write_text(json.dumps(go_eval, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {
        "n_codes": len(codes),
        "codes": codes,
        "eval_start": eval_start,
        "eval_end": str(end_ts.date()),
        "horizon": horizon,
        "config_hash": config_hash_for_regime(cfg),
        "cache_schema": CACHE_SCHEMA,
        "cache_stats": cache_stats,
        "cal_meta": cal_meta,
        "calibration_summary": cal_summary,
        "go_no_go": go_no_go,
        "go_eval": go_eval,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = [
        "# REGIME_TRANSITION_VALIDATION",
        "",
        f"- n_codes: {len(codes)} (sw_cluster_mapping_selected)",
        f"- rows: {len(calibrated)}",
        f"- go_no_go: **{go_no_go}**",
        f"- q_confirm: {cal_meta.get('q_confirm')} ({cal_meta.get('threshold_mode')})",
        f"- brier: {cal_summary.get('brier')}",
        f"- ece: {cal_summary.get('ece')}",
        "",
        "## Go/No-Go checks",
    ]
    for name, payload in (go_eval.get("checks") or {}).items():
        if isinstance(payload, dict) and "pass" in payload:
            report.append(f"- {name}: {'PASS' if payload['pass'] else 'FAIL'} — {payload}")
        else:
            report.append(f"- {name}: {payload}")
    report.extend(["", "## Signal counts"])
    if not calibrated.empty and "signal_level" in calibrated.columns:
        for key, value in calibrated["signal_level"].value_counts().to_dict().items():
            report.append(f"- {key}: {value}")
    (output_dir / "REGIME_TRANSITION_VALIDATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_regime(
    *,
    eval_start: str = "2024-06-01",
    eval_end: str | None = None,
    horizon: int = 10,
    jobs: int = 8,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    resume: bool = True,
    selected_csv: Path | None = None,
) -> int:
    """Load the pool and the qlib close panel, then run the walk-forward."""
    from sw_daily.pool.data_loading import load_close_volume

    codes, name_map = load_pool_codes(selected_csv)
    end = eval_end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out_dir = Path(output_dir) if output_dir is not None else REGIME_DIR / "decision_pack"
    cache = Path(cache_dir) if cache_dir is not None else REGIME_DIR / "model_cache"
    print(f"[INFO] pool codes={len(codes)} eval={eval_start}..{end} out={out_dir}")

    if resume and eval_end:
        month = month_key(pd.Timestamp(end))
        if day_already_scored(out_dir / "shards" / f"{month}.signals.csv", out_dir / "signals_oos.csv", end):
            print(f"[INFO] regime day already scored through {end}; skip qlib reload")
            return 0

    start = (pd.Timestamp(eval_start) - pd.tseries.offsets.BDay(400)).strftime("%Y-%m-%d")
    close, _vol = load_close_volume(codes, test_period=(start, end))
    run_validation(
        close,
        codes,
        name_map=name_map,
        eval_start=eval_start,
        eval_end=end,
        horizon=horizon,
        trim_for_labels=False,
        jobs=max(1, int(jobs)),
        output_dir=out_dir,
        cache_dir=None if not use_cache else cache,
        use_cache=use_cache,
        resume=resume,
    )
    print(f"[INFO] decision pack: {out_dir}")
    return 0
