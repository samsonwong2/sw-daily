"""Student-t HMM fit and causal forward filter.

Ported from ``etf_daily.lib.hmm_short_horizon``. Only the EM fit and the
one-step filter used by the regime-transition walk-forward are kept; the
Monte-Carlo path simulator is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class HmmConfig:
    n_states: int = 3
    train_window: int = 252
    min_samples: int = 60
    emission: str = "student_t"
    nu_fixed: float = 5.0
    min_sigma_ratio: float = 1.15
    regime_filter_lookback: int = 40
    max_em_iter: int = 80
    em_tol: float = 1e-6


@dataclass(frozen=True)
class HmmFitResult:
    n_states: int
    trans: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    nu: np.ndarray | None
    pi_filt: np.ndarray
    emission: str
    reliable: bool
    n_samples: int
    flags: tuple[str, ...] = ()
    loglik: float | None = None
    regime_separated: bool = True
    sigma_ratio: float | None = None
    state_posterior_mass: np.ndarray | None = None


@dataclass(frozen=True)
class ForwardFilterStep:
    """One causal forward-filter update under frozen HMM parameters."""

    alpha: np.ndarray
    xi: np.ndarray
    log_evidence: float


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = m_safe + np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True))
    if axis is None:
        return float(np.squeeze(out))
    return np.squeeze(out, axis=axis)


def _emission_logpdf(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    emission: str,
    nu: np.ndarray | None,
) -> np.ndarray:
    """Return (T, K) log emission densities."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(1, -1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(1, -1), 1e-8)
    if emission == "student_t" and nu is not None:
        nu_arr = np.asarray(nu, dtype=float).reshape(1, -1)
        z = (x - mu) / sigma
        return stats.t.logpdf(z, df=nu_arr) - np.log(sigma)
    return stats.norm.logpdf(x, loc=mu, scale=sigma)


def _forward_backward(
    log_emit: np.ndarray,
    log_trans: np.ndarray,
    log_start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return gamma (T, K), xi (T-1, K, K), loglik."""
    t_len, k = log_emit.shape
    log_alpha = np.full((t_len, k), -np.inf)
    log_alpha[0] = log_start + log_emit[0]
    for t in range(1, t_len):
        for j in range(k):
            log_alpha[t, j] = log_emit[t, j] + _logsumexp(log_alpha[t - 1] + log_trans[:, j])

    loglik = float(_logsumexp(log_alpha[-1]))
    log_beta = np.zeros((t_len, k))
    for t in range(t_len - 2, -1, -1):
        for i in range(k):
            log_beta[t, i] = _logsumexp(log_trans[i, :] + log_emit[t + 1] + log_beta[t + 1])

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)

    xi = np.zeros((max(t_len - 1, 0), k, k))
    for t in range(t_len - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + log_emit[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        log_xi = log_xi - _logsumexp(log_xi)
        xi[t] = np.exp(log_xi)
    return gamma, xi, loglik


def _init_params(
    returns: np.ndarray,
    k: int,
    emission: str,
    nu_fixed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    abs_r = np.abs(returns)
    med = float(np.median(abs_r)) if abs_r.size else 0.01
    mu = np.zeros(k, dtype=float)
    if k == 2:
        sigma = np.array([max(med * 0.7, 1e-4), max(med * 2.0, 2e-4)], dtype=float)
    elif k >= 3 and returns.size >= k * 5:
        sigma = np.zeros(k, dtype=float)
        edges = np.quantile(abs_r, np.linspace(0.0, 1.0, k + 1))
        for j in range(k):
            lo, hi = edges[j], edges[j + 1]
            mask = (abs_r >= lo) & (abs_r <= hi if j == k - 1 else abs_r < hi)
            if not np.any(mask):
                mask = abs_r <= hi
            chunk = returns[mask]
            if chunk.size:
                mu[j] = float(np.mean(chunk))
                sigma[j] = max(float(np.std(chunk, ddof=1)), 1e-4)
            else:
                sigma[j] = max(float(np.quantile(abs_r, (j + 1) / (k + 1))), 1e-4)
    else:
        qs = np.linspace(0.25, 0.75, k)
        sigma = np.maximum(np.quantile(abs_r, qs), 1e-4)
    trans = np.full((k, k), 0.1 / max(k - 1, 1))
    np.fill_diagonal(trans, 0.9)
    nu = np.full(k, nu_fixed, dtype=float) if emission == "student_t" else None
    start = np.full(k, 1.0 / k)
    return start, trans, mu, sigma, nu


def min_pairwise_sigma_ratio(sigma: np.ndarray) -> float | None:
    """Minimum sigma_i/sigma_j over all i<j."""
    sig = np.asarray(sigma, dtype=float).ravel()
    if sig.size < 2:
        return None
    ratios: list[float] = []
    for i in range(sig.size):
        for j in range(i + 1, sig.size):
            low = float(min(sig[i], sig[j]))
            high = float(max(sig[i], sig[j]))
            if low <= 0 or not math.isfinite(low) or not math.isfinite(high):
                continue
            ratios.append(high / low)
    return min(ratios) if ratios else None


def fit_hmm_kstate(returns: np.ndarray, cfg: HmmConfig | None = None) -> HmmFitResult:
    cfg = cfg or HmmConfig()
    clean = returns[np.isfinite(returns)].astype(float)
    flags: list[str] = []
    if clean.size < cfg.min_samples:
        return HmmFitResult(
            n_states=cfg.n_states,
            trans=np.eye(cfg.n_states),
            mu=np.zeros(cfg.n_states),
            sigma=np.ones(cfg.n_states) * 0.01,
            nu=None,
            pi_filt=np.full(cfg.n_states, 1.0 / cfg.n_states),
            emission=cfg.emission,
            reliable=False,
            n_samples=int(clean.size),
            flags=("low_n",),
        )

    k = int(cfg.n_states)
    start, trans, mu, sigma, nu = _init_params(clean, k, cfg.emission, cfg.nu_fixed)
    log_start = np.log(np.maximum(start, 1e-12))
    prev_ll = -np.inf

    for _ in range(cfg.max_em_iter):
        log_trans = np.log(np.maximum(trans, 1e-12))
        log_emit = _emission_logpdf(clean, mu, sigma, emission=cfg.emission, nu=nu)
        gamma, xi, ll = _forward_backward(log_emit, log_trans, log_start)
        start = gamma[0] / max(float(gamma[0].sum()), 1e-12)
        log_start = np.log(np.maximum(start, 1e-12))
        xi_sum = xi.sum(axis=0)
        row_sum = xi_sum.sum(axis=1, keepdims=True)
        trans = xi_sum / np.maximum(row_sum, 1e-12)
        trans = np.maximum(trans, 1e-3)
        trans = trans / trans.sum(axis=1, keepdims=True)

        for j in range(k):
            w = gamma[:, j]
            w_sum = float(np.sum(w))
            if w_sum < 1e-8:
                continue
            mu[j] = float(np.sum(w * clean) / w_sum)
            var = float(np.sum(w * (clean - mu[j]) ** 2) / w_sum)
            floor = 1e-5
            if k >= 3:
                floor = max(float(np.median(np.abs(clean))) * 0.15, 1e-4)
            sigma[j] = max(math.sqrt(var), floor)

        order = np.argsort(sigma)
        mu = mu[order]
        sigma = sigma[order]
        trans = trans[np.ix_(order, order)]
        start = start[order]
        log_start = np.log(np.maximum(start, 1e-12))
        if nu is not None:
            nu = nu[order]

        if abs(ll - prev_ll) < cfg.em_tol:
            prev_ll = ll
            break
        prev_ll = ll

    log_trans = np.log(np.maximum(trans, 1e-12))
    log_emit = _emission_logpdf(clean, mu, sigma, emission=cfg.emission, nu=nu)
    gamma, _, ll = _forward_backward(log_emit, log_trans, log_start)
    pi_filt = gamma[-1] / max(float(gamma[-1].sum()), 1e-12)
    state_mass = gamma.mean(axis=0).astype(float)
    state_mass = state_mass / max(float(state_mass.sum()), 1e-12)

    if not np.all(np.isfinite(pi_filt)) or not np.all(np.isfinite(sigma)):
        flags.append("em_failed")
        return HmmFitResult(
            n_states=k,
            trans=trans,
            mu=mu,
            sigma=sigma,
            nu=nu,
            pi_filt=np.full(k, 1.0 / k),
            emission=cfg.emission,
            reliable=False,
            n_samples=int(clean.size),
            flags=tuple(flags),
            loglik=None,
            regime_separated=False,
            sigma_ratio=None,
            state_posterior_mass=None,
        )

    sigma_ratio = min_pairwise_sigma_ratio(sigma)
    separated = bool(sigma_ratio is not None and sigma_ratio >= float(cfg.min_sigma_ratio))
    if not separated:
        flags.append("regime_collapsed")

    return HmmFitResult(
        n_states=k,
        trans=trans,
        mu=mu,
        sigma=sigma,
        nu=nu,
        pi_filt=pi_filt.astype(float),
        emission=cfg.emission,
        reliable=True,
        n_samples=int(clean.size),
        flags=tuple(flags),
        loglik=float(prev_ll),
        regime_separated=separated,
        sigma_ratio=sigma_ratio,
        state_posterior_mass=state_mass,
    )


def forward_filter_step(
    fit: HmmFitResult,
    alpha_prev: np.ndarray,
    observation: float,
) -> ForwardFilterStep:
    """One-step causal filter under frozen parameters."""
    if not fit.reliable:
        raise ValueError("forward_filter_step requires a reliable HmmFitResult")
    k = int(fit.n_states)
    alpha_prev = np.asarray(alpha_prev, dtype=float).ravel()
    if alpha_prev.size != k:
        raise ValueError(f"alpha_prev size {alpha_prev.size} != n_states {k}")
    if not np.all(np.isfinite(alpha_prev)):
        raise ValueError("alpha_prev contains non-finite values")
    obs = float(observation)
    if not math.isfinite(obs):
        raise ValueError("observation must be finite")

    alpha_prev = np.maximum(alpha_prev, 0.0)
    alpha_prev = alpha_prev / max(float(alpha_prev.sum()), 1e-12)

    log_emit = _emission_logpdf(
        np.asarray([obs], dtype=float),
        fit.mu,
        fit.sigma,
        emission=fit.emission,
        nu=fit.nu,
    )[0]
    log_trans = np.log(np.maximum(fit.trans, 1e-12))
    log_alpha_prev = np.log(np.maximum(alpha_prev, 1e-12))
    log_xi = log_alpha_prev[:, None] + log_trans + log_emit[None, :]
    log_evidence = float(_logsumexp(log_xi))
    if not math.isfinite(log_evidence):
        raise ValueError("non-finite filter evidence")
    xi = np.exp(log_xi - log_evidence)
    if not np.all(np.isfinite(xi)):
        raise ValueError("non-finite xi after normalization")
    xi_sum = float(xi.sum())
    if xi_sum <= 0.0:
        raise ValueError("zero xi mass after normalization")
    xi = xi / xi_sum
    alpha = xi.sum(axis=0)
    alpha = alpha / max(float(alpha.sum()), 1e-12)
    return ForwardFilterStep(alpha=alpha.astype(float), xi=xi.astype(float), log_evidence=log_evidence)


def replay_forward_filter(
    fit: HmmFitResult,
    observations: np.ndarray,
    *,
    alpha_start: np.ndarray | None = None,
) -> np.ndarray:
    """Replay a frozen HMM over observations; return the final alpha."""
    clean = np.asarray(observations, dtype=float).ravel()
    clean = clean[np.isfinite(clean)]
    k = int(fit.n_states)
    if alpha_start is None:
        alpha = np.full(k, 1.0 / k, dtype=float)
    else:
        alpha = np.asarray(alpha_start, dtype=float).ravel()
        if alpha.size != k:
            raise ValueError(f"alpha_start size {alpha.size} != n_states {k}")
        alpha = np.maximum(alpha, 0.0)
        alpha = alpha / max(float(alpha.sum()), 1e-12)
    if clean.size == 0:
        return alpha
    for obs in clean:
        step = forward_filter_step(fit, alpha, float(obs))
        alpha = step.alpha
    return alpha


def hmm_config_from_regime(cfg) -> HmmConfig:
    """Build an HMM config from a RegimeTransitionConfig-like object."""
    return HmmConfig(
        n_states=int(cfg.n_states),
        train_window=int(cfg.train_window),
        min_samples=int(cfg.min_samples),
        emission=str(cfg.emission),
        nu_fixed=float(cfg.nu_fixed),
        min_sigma_ratio=float(cfg.min_sigma_ratio),
        regime_filter_lookback=int(cfg.regime_filter_lookback),
        max_em_iter=int(cfg.max_em_iter),
        em_tol=float(cfg.em_tol),
    )


# keep replace import used by callers that rebuild fits
__all__ = [
    "ForwardFilterStep",
    "HmmConfig",
    "HmmFitResult",
    "fit_hmm_kstate",
    "forward_filter_step",
    "hmm_config_from_regime",
    "min_pairwise_sigma_ratio",
    "replay_forward_filter",
    "replace",
]
