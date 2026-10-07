"""Frozen 3-state HMM regime transition detector.

Ported from ``etf_daily.lib.regime_transition_board``. Fits a student-t HMM
through a train-end date, freezes it, then scores each later day with a
causal forward filter and a bilateral predictive-quantile switch (q*=0.90).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from sw_daily.regime.hmm import (
    ForwardFilterStep,
    HmmFitResult,
    fit_hmm_kstate,
    forward_filter_step,
    hmm_config_from_regime,
    min_pairwise_sigma_ratio,
    replay_forward_filter,
)

TransitionDirection = Literal["up", "down", "mixed"]
CANONICAL_LABELS: tuple[str, ...] = ("calm", "bear_grind", "volatile_reversal")


@dataclass(frozen=True)
class RegimeTransitionConfig:
    n_states: int = 3
    train_window: int = 252
    min_samples: int = 60
    emission: str = "student_t"
    nu_fixed: float = 5.0
    regime_filter_lookback: int = 40
    min_sigma_ratio: float = 1.15
    z_surprise_thr: float = 2.0
    rule_jump_std_mult: float = 2.5
    drop_split_returns: bool = True
    split_threshold: float = 0.25
    max_em_iter: int = 80
    em_tol: float = 1e-6
    days_in_regime_max_lookback: int = 30
    min_direction_mu_mult: float = 0.25
    min_state_mass: float = 0.02
    hard_ambiguous_require_both_separation_fails: bool = True
    switch_quantile_q: float = 0.90
    switch_quantile_bilateral: bool = True


@dataclass(frozen=True)
class RegimeTransitionMetrics:
    regime_now: str
    p_calm: float | None
    p_bear: float | None
    p_reversal: float | None
    days_in_regime: int
    p_transition: float | None
    emission_surprise: float | None
    transition_score: float
    transition_direction: TransitionDirection
    regime_separated: bool
    min_pairwise_sigma_ratio: float | None
    reliable: bool
    basis: str
    prev_regime: str | None
    n_samples: int = 0
    p_switch_hmm: float | None = None
    p_into_reversal_hmm: float | None = None
    p_state_reversal: float | None = None
    p_enter_current_prior: float | None = None
    p_no_exit_given_reversal_h5: float | None = None
    p_no_exit_given_reversal_h10: float | None = None
    p_no_exit_given_reversal_h20: float | None = None
    p_in_reversal_and_no_exit_h5: float | None = None
    p_in_reversal_and_no_exit_h10: float | None = None
    p_in_reversal_and_no_exit_h20: float | None = None
    label_ambiguous: bool = False
    ambiguity_reasons: tuple[str, ...] = ()
    model_train_end: str | None = None
    config_hash: str | None = None
    return_z: float | None = None
    p_into_reversal_adj: float | None = None
    direction_up: bool = False
    pred_cdf: float | None = None
    quantile_switch: bool = False
    switch_side: TransitionDirection | None = None
    regime_hmm_raw: str | None = None
    pred_log_density: float | None = None
    pred_mean: float | None = None
    pred_scale: float | None = None


@dataclass(frozen=True)
class FrozenRegimeTransitionModel:
    fit: HmmFitResult
    labels: tuple[str, ...]
    model_train_end: str
    label_ambiguous: bool
    ambiguity_reasons: tuple[str, ...]
    config_hash: str
    perm: tuple[int, ...]


def config_hash_for_regime(cfg: RegimeTransitionConfig) -> str:
    payload = {k: getattr(cfg, k) for k in sorted(RegimeTransitionConfig.__dataclass_fields__)}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def slice_log_returns(close: pd.Series, end: pd.Timestamp, *, lookback_days: int) -> np.ndarray:
    available = pd.to_numeric(close, errors="coerce").dropna()
    available = available.loc[available.index <= pd.Timestamp(end)]
    if len(available) < 2:
        return np.array([], dtype=float)
    if lookback_days > 0 and len(available) > lookback_days + 1:
        available = available.tail(lookback_days + 1)
    prices = available.astype(float).to_numpy()
    prices = prices[prices > 0]
    if len(prices) < 2:
        return np.array([], dtype=float)
    return np.diff(np.log(prices))


def drop_split_log_returns(
    close: pd.Series,
    *,
    end: pd.Timestamp,
    lookback_days: int,
    split_threshold: float = 0.25,
) -> np.ndarray:
    """Log returns with single-day |r| above the threshold removed.

    Indexes have no corporate splits; the cut still drops broken bars.
    """
    rets = slice_log_returns(close, end, lookback_days=lookback_days)
    if rets.size == 0:
        return rets
    keep = np.isfinite(rets) & (np.abs(rets) <= float(split_threshold))
    return rets[keep]


def _slice_returns(close: pd.Series, end: pd.Timestamp, cfg: RegimeTransitionConfig) -> np.ndarray:
    if cfg.drop_split_returns:
        return drop_split_log_returns(
            close, end=end, lookback_days=cfg.train_window, split_threshold=cfg.split_threshold
        )
    return slice_log_returns(close, end, lookback_days=cfg.train_window)


def _slice_returns_through(
    close: pd.Series,
    end: pd.Timestamp,
    cfg: RegimeTransitionConfig,
    *,
    lookback_days: int | None = None,
) -> np.ndarray:
    lb = int(lookback_days if lookback_days is not None else cfg.train_window)
    if cfg.drop_split_returns:
        return drop_split_log_returns(
            close, end=end, lookback_days=lb, split_threshold=cfg.split_threshold
        )
    return slice_log_returns(close, end, lookback_days=lb)


def relabel_hmm_states(
    mu: np.ndarray,
    sigma: np.ndarray,
    trans: np.ndarray,
    nu: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...], tuple[int, ...]]:
    """Map EM states to calm / bear_grind / volatile_reversal."""
    k = int(sigma.size)
    idxs = list(range(k))

    def _sigma_key(i: int) -> tuple:
        return (float(sigma[i]), abs(float(mu[i])), i)

    calm_i = min(idxs, key=_sigma_key)
    rest = [i for i in idxs if i != calm_i]
    if not rest:
        perm = (calm_i,)
        labels: tuple[str, ...] = ("calm",)
    elif len(rest) == 1:
        perm = (calm_i, rest[0])
        labels = ("calm", "bear_grind")
    else:
        a, b = rest[0], rest[1]
        mu_a, mu_b = float(mu[a]), float(mu[b])
        if abs(mu_a - mu_b) <= 1e-8:
            sig_a, sig_b = float(sigma[a]), float(sigma[b])
            if abs(sig_a - sig_b) <= 1e-8:
                bear_i, rev_i = (a, b) if a < b else (b, a)
            elif sig_a > sig_b:
                rev_i, bear_i = a, b
            else:
                rev_i, bear_i = b, a
        elif mu_a < mu_b:
            bear_i, rev_i = a, b
        else:
            bear_i, rev_i = b, a
        perm = (calm_i, bear_i, rev_i)
        labels = CANONICAL_LABELS
    perm_arr = np.asarray(perm, dtype=int)
    mu2 = mu[perm_arr]
    sigma2 = sigma[perm_arr]
    trans2 = trans[np.ix_(perm_arr, perm_arr)]
    nu2 = nu[perm_arr] if nu is not None else None
    return mu2, sigma2, trans2, nu2, labels, tuple(int(x) for x in perm)


def diagnose_ambiguity(
    fit: HmmFitResult,
    labels: tuple[str, ...],
    *,
    cfg: RegimeTransitionConfig,
    returns: np.ndarray,
    prev_labels: tuple[str, ...] | None = None,
    prev_mu: np.ndarray | None = None,
    prev_sigma: np.ndarray | None = None,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if not fit.reliable:
        return True, ("fit_unreliable",)

    sigma_fail = False
    sigma_ratio = min_pairwise_sigma_ratio(fit.sigma)
    if sigma_ratio is None or sigma_ratio < float(cfg.min_sigma_ratio):
        reasons.append("sigma_not_separated")
        sigma_fail = True

    mu_fail = False
    clean = returns[np.isfinite(returns)].astype(float)
    med_abs = float(np.median(np.abs(clean))) if clean.size else 0.0
    if "bear_grind" in labels and "volatile_reversal" in labels:
        bi = labels.index("bear_grind")
        ri = labels.index("volatile_reversal")
        mu_diff = abs(float(fit.mu[bi]) - float(fit.mu[ri]))
        if mu_diff < float(cfg.min_direction_mu_mult) * max(med_abs, 1e-12):
            reasons.append("direction_mu_not_separated")
            mu_fail = True

    mass = fit.state_posterior_mass
    n_effective = fit.n_states
    if mass is not None and mass.size == fit.n_states:
        n_effective = int(np.sum(np.asarray(mass, dtype=float) >= float(cfg.min_state_mass)))
        if float(np.min(mass)) < float(cfg.min_state_mass):
            reasons.append("low_effective_state_mass")
        if n_effective < 2:
            reasons.append("single_state_collapse")

    drift = False
    if prev_labels is not None and prev_mu is not None and prev_sigma is not None:
        if prev_labels != labels:
            reasons.append("semantic_label_drift")
            drift = True
        else:
            for j, lab in enumerate(labels):
                if lab not in prev_labels:
                    reasons.append("semantic_label_drift")
                    drift = True
                    break
                pj = prev_labels.index(lab)
                if lab in ("bear_grind", "volatile_reversal") and len(labels) == 3:
                    other = "volatile_reversal" if lab == "bear_grind" else "bear_grind"
                    if other in prev_labels:
                        po = prev_labels.index(other)
                        d_same = abs(float(fit.mu[j]) - float(prev_mu[pj])) + abs(
                            float(fit.sigma[j]) - float(prev_sigma[pj])
                        )
                        d_swap = abs(float(fit.mu[j]) - float(prev_mu[po])) + abs(
                            float(fit.sigma[j]) - float(prev_sigma[po])
                        )
                        if d_swap + 1e-12 < d_same:
                            reasons.append("semantic_label_drift")
                            drift = True
                            break

    uniq: list[str] = []
    for reason in reasons:
        if reason not in uniq:
            uniq.append(reason)

    both_sep_fail = sigma_fail and mu_fail
    hard = bool(
        drift
        or "single_state_collapse" in uniq
        or (both_sep_fail if cfg.hard_ambiguous_require_both_separation_fails else (sigma_fail or mu_fail))
    )
    return hard, tuple(uniq)


def jump_into_reversal_evidence(returns: np.ndarray, *, thr: float = 2.5) -> float:
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size < 21:
        return 0.0
    r_t = float(clean[-1])
    if r_t <= 0:
        return 0.0
    std_20 = float(np.std(clean[-21:-1], ddof=1))
    if std_20 <= 0 or not math.isfinite(std_20):
        return 0.0
    z = r_t / std_20
    if z < thr:
        return 0.0
    return float(min(max((z - thr) / thr, 0.0), 1.0))


def return_z_score(returns: np.ndarray) -> float | None:
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size < 21:
        return None
    r_t = float(clean[-1])
    std_20 = float(np.std(clean[-21:-1], ddof=1))
    if std_20 <= 0 or not math.isfinite(std_20):
        return None
    return r_t / std_20


def _normalize_mixture_params(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    pi_arr = np.asarray(pi, dtype=float).ravel()
    mu_arr = np.asarray(mu, dtype=float).ravel()
    sig_arr = np.asarray(sigma, dtype=float).ravel()
    nu_arr = np.asarray([] if nu is None else nu, dtype=float).ravel()
    if pi_arr.size == 0 or pi_arr.size != mu_arr.size:
        return None
    mass = float(np.sum(pi_arr))
    if mass <= 0:
        return None
    pi_n = pi_arr / mass
    if nu_arr.size != pi_n.size:
        nu_out = np.full(pi_n.size, 5.0, dtype=float)
        n_copy = min(nu_arr.size, pi_n.size)
        if n_copy > 0:
            nu_out[:n_copy] = nu_arr[:n_copy]
    else:
        nu_out = nu_arr.copy()
    nu_out = np.where(np.isfinite(nu_out) & (nu_out > 0), nu_out, 5.0)
    return pi_n, mu_arr, sig_arr, nu_out


def student_t_mixture_cdf(
    r: float, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray | None
) -> float | None:
    from scipy.stats import t as student_t

    params = _normalize_mixture_params(pi, mu, sigma, nu)
    if params is None or not math.isfinite(float(r)):
        return None
    pi_n, mu_arr, sig_arr, nu_arr = params
    cdf = 0.0
    for k in range(pi_n.size):
        sig = float(sig_arr[k])
        if sig <= 0 or not math.isfinite(sig):
            continue
        z = (float(r) - float(mu_arr[k])) / sig
        cdf += float(pi_n[k]) * float(student_t.cdf(z, df=float(nu_arr[k])))
    if not math.isfinite(cdf):
        return None
    return float(min(max(cdf, 0.0), 1.0))


def student_t_mixture_pdf(
    r: float, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray | None
) -> float | None:
    from scipy.stats import t as student_t

    params = _normalize_mixture_params(pi, mu, sigma, nu)
    if params is None or not math.isfinite(float(r)):
        return None
    pi_n, mu_arr, sig_arr, nu_arr = params
    dens = 0.0
    for k in range(pi_n.size):
        sig = float(sig_arr[k])
        if sig <= 0 or not math.isfinite(sig):
            continue
        z = (float(r) - float(mu_arr[k])) / sig
        dens += float(pi_n[k]) * float(student_t.pdf(z, df=float(nu_arr[k]))) / sig
    if not math.isfinite(dens) or dens < 0.0:
        return None
    return float(dens)


def student_t_mixture_log_density(
    r: float, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray | None
) -> float | None:
    dens = student_t_mixture_pdf(r, pi, mu, sigma, nu)
    if dens is None or dens <= 0.0:
        return None
    return float(math.log(dens))


def student_t_mixture_moments(
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray | None
) -> tuple[float | None, float | None]:
    params = _normalize_mixture_params(pi, mu, sigma, nu)
    if params is None:
        return None, None
    pi_n, mu_arr, sig_arr, nu_arr = params
    mean = float(np.sum(pi_n * mu_arr))
    if not math.isfinite(mean):
        return None, None
    var = 0.0
    used_var = False
    for k in range(pi_n.size):
        sig = float(sig_arr[k])
        if sig <= 0 or not math.isfinite(sig):
            continue
        nu_k = float(nu_arr[k])
        if nu_k > 2.0:
            comp_var = (sig**2) * (nu_k / (nu_k - 2.0))
            used_var = True
        else:
            comp_var = sig**2
        var += float(pi_n[k]) * (comp_var + (float(mu_arr[k]) - mean) ** 2)
    if not math.isfinite(var) or var < 0.0:
        return mean, None
    scale = math.sqrt(var) if used_var or var > 0 else None
    return mean, (float(scale) if scale is not None and math.isfinite(scale) else None)


def predictive_state_probs(alpha_tm1: np.ndarray, trans: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha_tm1, dtype=float).ravel()
    matrix = np.asarray(trans, dtype=float)
    total = float(np.sum(a))
    if total <= 0 or matrix.ndim != 2 or matrix.shape[0] != a.size:
        return a
    pi = (a / total) @ matrix
    mass = float(np.sum(pi))
    return pi / mass if mass > 0 else pi


def compute_predictive_moments_next(
    alpha_t: np.ndarray | None, model: FrozenRegimeTransitionModel
) -> tuple[float | None, float | None]:
    if alpha_t is None:
        return None, None
    a = np.asarray(alpha_t, dtype=float).ravel()
    if a.size == 0 or float(np.sum(a)) <= 0:
        return None, None
    pi_pred = predictive_state_probs(a, model.fit.trans)
    return student_t_mixture_moments(pi_pred, model.fit.mu, model.fit.sigma, model.fit.nu)


def quantile_switch_from_cdf(
    pred_cdf: float | None, *, q_star: float = 0.90, bilateral: bool = True
) -> tuple[bool, TransitionDirection | None]:
    if pred_cdf is None or not math.isfinite(float(pred_cdf)):
        return False, None
    q = float(q_star)
    if not (0.5 <= q < 1.0):
        return False, None
    cdf = float(pred_cdf)
    if cdf >= q:
        return True, "up"
    if bilateral and cdf <= (1.0 - q):
        return True, "down"
    return False, None


def fit_frozen_regime_model(
    close: pd.Series,
    model_train_end: pd.Timestamp | str,
    *,
    cfg: RegimeTransitionConfig | None = None,
    prev_model: FrozenRegimeTransitionModel | None = None,
) -> FrozenRegimeTransitionModel:
    cfg = cfg or RegimeTransitionConfig()
    end = pd.Timestamp(model_train_end)
    returns = _slice_returns(close, end, cfg)
    ch = config_hash_for_regime(cfg)
    fit = fit_hmm_kstate(returns, hmm_config_from_regime(cfg))
    if not fit.reliable:
        return FrozenRegimeTransitionModel(
            fit=fit,
            labels=CANONICAL_LABELS[: fit.n_states],
            model_train_end=end.strftime("%Y-%m-%d"),
            label_ambiguous=True,
            ambiguity_reasons=("fit_unreliable",),
            config_hash=ch,
            perm=tuple(range(fit.n_states)),
        )
    mu, sigma, trans, nu, labels, perm = relabel_hmm_states(fit.mu, fit.sigma, fit.trans, fit.nu)
    mass = fit.state_posterior_mass
    if mass is not None and mass.size == fit.n_states:
        mass = mass[np.asarray(perm)]
        mass = mass / max(float(mass.sum()), 1e-12)
    fit2 = replace(fit, mu=mu, sigma=sigma, trans=trans, nu=nu, state_posterior_mass=mass)
    ambiguous, reasons = diagnose_ambiguity(
        fit2,
        labels,
        cfg=cfg,
        returns=returns,
        prev_labels=prev_model.labels if prev_model else None,
        prev_mu=prev_model.fit.mu if prev_model else None,
        prev_sigma=prev_model.fit.sigma if prev_model else None,
    )
    return FrozenRegimeTransitionModel(
        fit=fit2,
        labels=labels,
        model_train_end=end.strftime("%Y-%m-%d"),
        label_ambiguous=ambiguous,
        ambiguity_reasons=reasons,
        config_hash=ch,
        perm=perm,
    )


def _persistence_fields(
    fit: HmmFitResult, labels: tuple[str, ...], p_state_rev: float | None
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "p_no_exit_given_reversal_h5": None,
        "p_no_exit_given_reversal_h10": None,
        "p_no_exit_given_reversal_h20": None,
        "p_in_reversal_and_no_exit_h5": None,
        "p_in_reversal_and_no_exit_h10": None,
        "p_in_reversal_and_no_exit_h20": None,
    }
    if "volatile_reversal" not in labels:
        return out
    k = labels.index("volatile_reversal")
    stay = float(fit.trans[k, k])
    for h in (5, 10, 20):
        p_no = stay**h
        out[f"p_no_exit_given_reversal_h{h}"] = p_no
        if p_state_rev is not None:
            out[f"p_in_reversal_and_no_exit_h{h}"] = float(p_state_rev) * p_no
    return out


def probabilities_from_xi(xi: np.ndarray, labels: tuple[str, ...]) -> dict[str, float]:
    xi = np.asarray(xi, dtype=float)
    p_switch = float(xi.sum() - np.trace(xi))
    p_state = xi.sum(axis=0)
    p_state = p_state / max(float(p_state.sum()), 1e-12)

    def _p(label: str) -> float:
        if label not in labels:
            return 0.0
        return float(p_state[labels.index(label)])

    into_rev = 0.0
    if "volatile_reversal" in labels:
        j = labels.index("volatile_reversal")
        into_rev = float(xi[:, j].sum() - xi[j, j])
    return {
        "p_switch_hmm": p_switch,
        "p_into_reversal_hmm": into_rev,
        "p_state_reversal": _p("volatile_reversal"),
        "p_state_calm": _p("calm"),
        "p_state_bear": _p("bear_grind"),
    }


def _rule_fallback_score(returns: np.ndarray, cfg: RegimeTransitionConfig) -> float:
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size < 21:
        return 0.0
    r_t = float(clean[-1])
    std_20 = float(np.std(clean[-20:], ddof=1))
    mean_5 = float(np.mean(clean[-5:]))
    if std_20 <= 0:
        return 0.0
    if abs(r_t) > cfg.rule_jump_std_mult * std_20 and ((r_t > 0) != (mean_5 > 0) or abs(mean_5) < 1e-12):
        return 0.7
    return 0.0


def _jump_transition_boost(
    returns: np.ndarray, prev_regime: str | None, *, cfg: RegimeTransitionConfig
) -> tuple[float, str | None]:
    clean = returns[np.isfinite(returns)].astype(float)
    if clean.size < 21:
        return 0.0, None
    r_t = float(clean[-1])
    std_20 = float(np.std(clean[-20:], ddof=1))
    if std_20 <= 0:
        return 0.0, None
    jump_z = abs(r_t) / std_20
    if jump_z < cfg.rule_jump_std_mult:
        return 0.0, None
    mean_5 = float(np.mean(clean[-6:-1])) if clean.size >= 6 else float(np.mean(clean[:-1]))
    scale = min(jump_z / (cfg.rule_jump_std_mult * 1.5), 1.0)
    override: str | None = None
    if prev_regime == "bear_grind" and r_t > 0:
        return 0.85 * scale, "volatile_reversal"
    if prev_regime == "volatile_reversal" and r_t < 0:
        return 0.7 * scale, "bear_grind"
    if (r_t > 0) != (mean_5 > 0) or abs(mean_5) < 1e-12:
        if r_t > 0:
            override = "volatile_reversal"
        return 0.6 * scale, override
    return 0.35 * scale, None


def compute_frozen_transition_step(
    close: pd.Series,
    as_of: pd.Timestamp | str,
    model: FrozenRegimeTransitionModel,
    *,
    cfg: RegimeTransitionConfig | None = None,
    alpha_prev: np.ndarray | None = None,
) -> tuple[RegimeTransitionMetrics, np.ndarray | None, ForwardFilterStep | None]:
    """Emit metrics for ``as_of`` under a frozen model."""
    cfg = cfg or RegimeTransitionConfig()
    end = pd.Timestamp(as_of)
    empty = RegimeTransitionMetrics(
        regime_now="—",
        p_calm=None,
        p_bear=None,
        p_reversal=None,
        days_in_regime=0,
        p_transition=None,
        emission_surprise=None,
        transition_score=0.0,
        transition_direction="mixed",
        regime_separated=False,
        min_pairwise_sigma_ratio=None,
        reliable=False,
        basis="样本不足",
        prev_regime=None,
        label_ambiguous=True,
        ambiguity_reasons=("fit_unreliable",),
        model_train_end=model.model_train_end,
        config_hash=model.config_hash,
    )
    if not model.fit.reliable:
        return empty, None, None

    returns = _slice_returns_through(
        close, end, cfg, lookback_days=max(cfg.train_window, cfg.regime_filter_lookback + 5)
    )
    if returns.size < 2:
        return replace(empty, n_samples=int(returns.size)), None, None

    r_t = float(returns[-1])
    lookback = int(cfg.regime_filter_lookback)
    try:
        if alpha_prev is None:
            warmup = returns[-(lookback + 1) : -1]
            alpha_tm1 = replay_forward_filter(model.fit, warmup)
        else:
            alpha_tm1 = np.asarray(alpha_prev, dtype=float).ravel()
        step = forward_filter_step(model.fit, alpha_tm1, r_t)
    except ValueError:
        return (
            replace(empty, n_samples=int(returns.size), basis="HMM3·filter失败"),
            None,
            None,
        )

    labels = model.labels
    pi_t = step.alpha
    pi_tm1 = alpha_tm1 / max(float(np.sum(alpha_tm1)), 1e-12)
    probs = probabilities_from_xi(step.xi, labels)
    s_t = int(np.argmax(pi_t))
    regime_hmm_raw = labels[s_t] if s_t < len(labels) else labels[-1]
    regime_now = regime_hmm_raw
    prev_regime = labels[int(np.argmax(pi_tm1))] if pi_tm1.size else None

    p_transition = float(np.sum(pi_tm1 * model.fit.trans[:, s_t]))
    mu_s = float(model.fit.mu[s_t])
    sig_s = float(model.fit.sigma[s_t])
    emission_surprise = abs(r_t - mu_s) / sig_s if sig_s > 0 and math.isfinite(sig_s) else None
    sigma_ratio = min_pairwise_sigma_ratio(model.fit.sigma)
    regime_separated = bool(sigma_ratio is not None and sigma_ratio >= float(cfg.min_sigma_ratio))
    sep_weight = 1.0 if regime_separated else 0.5
    surprise_factor = 0.0
    if emission_surprise is not None and cfg.z_surprise_thr > 0:
        surprise_factor = min(emission_surprise / float(cfg.z_surprise_thr), 3.0)
    model_score = p_transition * surprise_factor * sep_weight
    rule_score = _rule_fallback_score(returns, cfg) if not regime_separated else 0.0
    jump_score, regime_override = _jump_transition_boost(returns, prev_regime, cfg=cfg)

    pi_pred = predictive_state_probs(pi_tm1, model.fit.trans)
    pred_cdf = student_t_mixture_cdf(r_t, pi_pred, model.fit.mu, model.fit.sigma, model.fit.nu)
    pred_log_density = student_t_mixture_log_density(
        r_t, pi_pred, model.fit.mu, model.fit.sigma, model.fit.nu
    )
    pred_mean, pred_scale = student_t_mixture_moments(
        pi_pred, model.fit.mu, model.fit.sigma, model.fit.nu
    )
    q_switch, q_side = quantile_switch_from_cdf(
        pred_cdf, q_star=float(cfg.switch_quantile_q), bilateral=bool(cfg.switch_quantile_bilateral)
    )
    quantile_score = 0.0
    if q_switch and pred_cdf is not None:
        q = float(cfg.switch_quantile_q)
        if q_side == "up":
            quantile_score = float(min(max((float(pred_cdf) - q) / max(1.0 - q, 1e-12), 0.0), 1.0))
            quantile_score = 0.5 + 0.5 * quantile_score
            regime_override = "volatile_reversal"
        elif q_side == "down":
            lo = 1.0 - q
            quantile_score = float(min(max((lo - float(pred_cdf)) / max(lo, 1e-12), 0.0), 1.0))
            quantile_score = 0.5 + 0.5 * quantile_score
            regime_override = "bear_grind"

    if regime_override is not None:
        regime_now = regime_override
    transition_score = min(max(model_score, rule_score, jump_score, quantile_score), 1.0)

    if q_switch and q_side in ("up", "down"):
        direction: TransitionDirection = q_side
    elif jump_score > 0 and regime_override == "volatile_reversal":
        direction = "up"
    elif jump_score > 0 and regime_override == "bear_grind":
        direction = "down"
    elif prev_regime != regime_now:
        if regime_now == "volatile_reversal" and r_t > 0:
            direction = "up"
        elif regime_now == "bear_grind" and r_t < 0:
            direction = "down"
        else:
            direction = "mixed"
    else:
        direction = "mixed"

    def _p_at(label: str) -> float | None:
        if label not in labels:
            return None
        idx = labels.index(label)
        if idx < pi_t.size:
            return float(pi_t[idx])
        return None

    p_rev = probs["p_state_reversal"]
    persist = _persistence_fields(model.fit, labels, p_rev)
    rz = return_z_score(returns)
    jump_ev = jump_into_reversal_evidence(returns, thr=float(cfg.rule_jump_std_mult))
    p_into_adj = max(float(probs["p_into_reversal_hmm"]), jump_ev)
    if q_switch and q_side == "up":
        p_into_adj = max(p_into_adj, float(quantile_score))
    basis_parts = ["HMM3", "frozen"]
    if q_switch:
        basis_parts.append(f"q*={cfg.switch_quantile_q:g}分位跳")
    if jump_score > 0:
        basis_parts.append("jump跃迁")
    if model.label_ambiguous:
        basis_parts.append("ambiguous")
    if not regime_separated:
        basis_parts.append("塌缩fallback" if rule_score > 0 else "塌缩")

    metrics = RegimeTransitionMetrics(
        regime_now=regime_now,
        p_calm=_p_at("calm"),
        p_bear=_p_at("bear_grind"),
        p_reversal=_p_at("volatile_reversal"),
        days_in_regime=1,
        p_transition=p_transition,
        emission_surprise=emission_surprise,
        transition_score=transition_score,
        transition_direction=direction,
        regime_separated=regime_separated,
        min_pairwise_sigma_ratio=sigma_ratio,
        reliable=True,
        basis="·".join(basis_parts),
        prev_regime=prev_regime if prev_regime != regime_now else None,
        n_samples=int(returns.size),
        p_switch_hmm=probs["p_switch_hmm"],
        p_into_reversal_hmm=probs["p_into_reversal_hmm"],
        p_state_reversal=p_rev,
        p_enter_current_prior=p_transition,
        p_no_exit_given_reversal_h5=persist["p_no_exit_given_reversal_h5"],
        p_no_exit_given_reversal_h10=persist["p_no_exit_given_reversal_h10"],
        p_no_exit_given_reversal_h20=persist["p_no_exit_given_reversal_h20"],
        p_in_reversal_and_no_exit_h5=persist["p_in_reversal_and_no_exit_h5"],
        p_in_reversal_and_no_exit_h10=persist["p_in_reversal_and_no_exit_h10"],
        p_in_reversal_and_no_exit_h20=persist["p_in_reversal_and_no_exit_h20"],
        label_ambiguous=model.label_ambiguous,
        ambiguity_reasons=model.ambiguity_reasons,
        model_train_end=model.model_train_end,
        config_hash=model.config_hash,
        return_z=rz,
        p_into_reversal_adj=p_into_adj,
        direction_up=(direction == "up"),
        pred_cdf=pred_cdf,
        quantile_switch=bool(q_switch),
        switch_side=q_side,
        regime_hmm_raw=regime_hmm_raw,
        pred_log_density=pred_log_density,
        pred_mean=pred_mean,
        pred_scale=pred_scale,
    )
    return metrics, step.alpha, step
