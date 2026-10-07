"""Labels, eligibility, calibration, and event matching for regime validation.

Ported from ``etf_daily.lib.regime_transition_validation``. Listing-date
membership audits are dropped: a code is eligible when it has enough history
and enough finite bars in the last 20 sessions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendEvent:
    code: str
    direction: str
    onset: pd.Timestamp
    event_end: pd.Timestamp


def _trading_loc(close: pd.Series, as_of: pd.Timestamp) -> tuple[int, pd.Timestamp] | None:
    idx = close.index
    as_of = pd.Timestamp(as_of)
    if as_of not in idx:
        pos = int(idx.searchsorted(as_of, side="right")) - 1
        if pos < 0:
            return None
        as_of = pd.Timestamp(idx[pos])
    loc = idx.get_loc(as_of)
    if isinstance(loc, slice):
        loc = loc.start
    return int(loc), pd.Timestamp(as_of)


def realized_sigma_20(close: pd.Series, as_of: pd.Timestamp) -> float | None:
    loc_pair = _trading_loc(close, as_of)
    if loc_pair is None:
        return None
    loc, _ = loc_pair
    if loc < 20:
        return None
    window = close.iloc[loc - 20 : loc + 1]
    prices = pd.to_numeric(window, errors="coerce").dropna()
    if prices.size < 19:
        return None
    px = prices.to_numpy(dtype=float)
    rets = np.log(px[1:] / px[:-1])
    rets = rets[np.isfinite(rets)]
    if rets.size < 18:
        return None
    sigma = float(np.std(rets, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return None
    return sigma


def compute_trend_up_label(
    close: pd.Series,
    origin: pd.Timestamp,
    *,
    horizon: int = 10,
    kappa_ret: float = 0.35,
    kappa_eff: float = 0.28,
    kappa_mae: float = 1.0,
) -> bool | None:
    """Independent price-path trend label. None = not enough future bars."""
    loc_pair = _trading_loc(close, origin)
    if loc_pair is None:
        return None
    loc, origin = loc_pair
    end_loc = loc + int(horizon)
    if end_loc >= len(close.index):
        return None
    path = pd.to_numeric(close.iloc[loc : end_loc + 1], errors="coerce")
    if path.isna().any() or path.size != int(horizon) + 1:
        return None
    px = path.to_numpy(dtype=float)
    if np.any(px <= 0) or not np.all(np.isfinite(px)):
        return None
    r_th = float(np.log(px[-1] / px[0]))
    step = np.diff(np.log(px))
    path_len = float(np.sum(np.abs(step)))
    efficiency = abs(r_th) / path_len if path_len > 0 else 0.0
    cum = np.cumsum(step)
    mae = float(np.min(cum)) if cum.size else 0.0
    sigma = realized_sigma_20(close, origin)
    if sigma is None:
        return None
    scale = sigma * math.sqrt(float(horizon))
    return bool(
        r_th >= float(kappa_ret) * scale
        and efficiency >= float(kappa_eff)
        and mae >= -float(kappa_mae) * scale
    )


def is_eligible_on_date(
    as_of: pd.Timestamp,
    close: pd.Series | None,
    *,
    min_history: int = 60,
    min_valid_in_20: int = 18,
) -> bool:
    if close is None:
        return False
    ts = pd.Timestamp(as_of).normalize()
    series = pd.to_numeric(close, errors="coerce").dropna()
    if series.empty:
        return False
    hist = series.loc[series.index <= ts]
    if hist.size < min_history:
        return False
    tail = hist.iloc[-20:]
    return int(np.isfinite(tail.to_numpy(dtype=float)).sum()) >= min_valid_in_20


def eligible_codes_on_date(
    codes: list[str],
    as_of: pd.Timestamp,
    panel: pd.DataFrame,
    *,
    min_history: int = 60,
    min_valid_in_20: int = 18,
) -> list[str]:
    out: list[str] = []
    for code in codes:
        series = panel[code] if code in panel.columns else None
        if is_eligible_on_date(as_of, series, min_history=min_history, min_valid_in_20=min_valid_in_20):
            out.append(code)
    return out


def month_key(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    return f"{t.year:04d}-{t.month:02d}"


def last_trading_day_before_month(calendar: pd.DatetimeIndex, month: str) -> pd.Timestamp | None:
    year, mon = [int(x) for x in month.split("-")]
    start = pd.Timestamp(year=year, month=mon, day=1)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    prior = cal[cal < start]
    if prior.empty:
        return None
    return pd.Timestamp(prior[-1])


def trading_days_in_month(calendar: pd.DatetimeIndex, month: str) -> pd.DatetimeIndex:
    year, mon = [int(x) for x in month.split("-")]
    start = pd.Timestamp(year=year, month=mon, day=1)
    if mon == 12:
        end = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end = pd.Timestamp(year=year, month=mon + 1, day=1)
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    return cal[(cal >= start) & (cal < end)]


def attach_instrument_names(frame: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty or "code" not in frame.columns:
        return frame
    out = frame.copy()
    out["name"] = out["code"].astype(str).map(lambda c: name_map.get(c, ""))
    return out


def build_expanding_crossfit_predictions(
    rows: pd.DataFrame,
    *,
    feature_cols: list[str],
    label_col: str = "label_trend_up",
    month_col: str = "month",
    min_train_rows: int = 30,
) -> pd.DataFrame:
    """Predict each month with a logistic fit on earlier mature months only."""
    from sklearn.linear_model import LogisticRegression

    if rows.empty:
        return rows.copy()
    frame = rows[rows[label_col].notna()].copy()
    if frame.empty:
        return frame
    frame[label_col] = frame[label_col].astype(bool).astype(int)
    months = sorted(frame[month_col].astype(str).unique())
    preds: list[pd.DataFrame] = []
    for i, month in enumerate(months):
        train = frame[frame[month_col].astype(str) < month]
        test = frame[frame[month_col].astype(str) == month]
        if len(train) < min_train_rows or train[label_col].nunique() < 2:
            continue
        x_train = train[feature_cols].astype(float).to_numpy()
        y_train = train[label_col].to_numpy()
        x_test = test[feature_cols].astype(float).to_numpy()
        if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
            mask_tr = np.all(np.isfinite(x_train), axis=1)
            mask_te = np.all(np.isfinite(x_test), axis=1)
            if mask_tr.sum() < min_train_rows or mask_te.sum() == 0:
                continue
            x_train, y_train = x_train[mask_tr], y_train[mask_tr]
            test = test.iloc[np.where(mask_te)[0]]
            x_test = x_test[mask_te]
            if y_train.min() == y_train.max():
                continue
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=0)
        model.fit(x_train, y_train)
        score = model.predict_proba(x_test)[:, 1]
        out = test.copy()
        out["p_trend_raw"] = score
        out["crossfit_train_months"] = i
        preds.append(out)
    if not preds:
        return frame.iloc[0:0].copy()
    return pd.concat(preds, ignore_index=True)


def maybe_isotonic_calibrate(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    apply_scores: np.ndarray,
    *,
    min_rows: int = 500,
    min_pos: int = 50,
    min_top_bin: int = 30,
) -> tuple[np.ndarray, bool]:
    train_scores = np.asarray(train_scores, dtype=float)
    train_labels = np.asarray(train_labels, dtype=float)
    apply_scores = np.asarray(apply_scores, dtype=float)
    if (
        train_scores.size < min_rows
        or int(train_labels.sum()) < min_pos
        or train_labels.min() == train_labels.max()
    ):
        return apply_scores, False
    order = np.argsort(train_scores)
    sorted_s = train_scores[order]
    q80 = float(np.quantile(sorted_s, 0.8))
    q90 = float(np.quantile(sorted_s, 0.9))
    n_top = int((train_scores >= q90).sum())
    n_mid = int(((train_scores >= q80) & (train_scores < q90)).sum())
    if n_top < min_top_bin or n_mid < min_top_bin:
        return apply_scores, False
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(train_scores, train_labels)
    return iso.predict(apply_scores), True


def select_fdr_threshold(
    crossfit_rows: pd.DataFrame,
    *,
    score_col: str = "p_trend_cal",
    label_col: str = "label_trend_up",
    target_fdr: float = 0.20,
    min_confirm: int = 50,
    n_bootstrap: int = 200,
    block_col: str = "month",
    seed: int = 42,
) -> dict:
    """Choose the lowest threshold whose bootstrap FDR upper CI is within target."""
    empty = {"q_confirm": None, "confirm_suppressed_reason": "no_crossfit_rows", "n_confirm_support": 0}
    if crossfit_rows.empty or score_col not in crossfit_rows.columns:
        return empty
    frame = crossfit_rows.dropna(subset=[score_col, label_col]).copy()
    if frame.empty:
        return {**empty, "confirm_suppressed_reason": "empty_after_dropna"}
    scores = frame[score_col].astype(float).to_numpy()
    labels = frame[label_col].astype(bool).to_numpy()
    candidates = np.unique(np.round(scores, 6))[::-1]
    rng = np.random.default_rng(seed)
    blocks = frame[block_col].astype(str).to_numpy() if block_col in frame.columns else None
    unique_blocks = np.unique(blocks) if blocks is not None else None

    for thr in candidates:
        mask = scores >= float(thr)
        n = int(mask.sum())
        if n < min_confirm:
            continue
        fp = int((~labels[mask]).sum())
        fdr_point = fp / max(n, 1)
        if unique_blocks is not None and unique_blocks.size >= 2:
            boot = []
            for _ in range(n_bootstrap):
                chosen = rng.choice(unique_blocks, size=unique_blocks.size, replace=True)
                sel = np.isin(blocks, chosen) & mask
                nn = int(sel.sum())
                if nn == 0:
                    continue
                boot.append(float((~labels[sel]).sum()) / nn)
            if not boot:
                continue
            upper = float(np.quantile(boot, 0.95))
        else:
            upper = fdr_point
        if upper <= target_fdr:
            return {
                "q_confirm": float(thr),
                "confirm_suppressed_reason": None,
                "n_confirm_support": n,
                "target_fdr": target_fdr,
            }
    return {
        "q_confirm": None,
        "confirm_suppressed_reason": "no_threshold_meets_fdr",
        "n_confirm_support": 0,
        "target_fdr": target_fdr,
    }


def summarize_calibration(rows: pd.DataFrame, *, score_col: str, label_col: str) -> dict[str, float]:
    if rows.empty:
        return {"n": 0, "brier": float("nan"), "ece": float("nan")}
    p = rows[score_col].astype(float).to_numpy()
    y = rows[label_col].astype(float).to_numpy()
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size == 0:
        return {"n": 0, "brier": float("nan"), "ece": float("nan")}
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not m.any():
            continue
        ece += abs(float(y[m].mean()) - float(p[m].mean())) * (m.sum() / p.size)
    return {"n": int(p.size), "brier": brier, "ece": float(ece)}


def cluster_positive_events(
    labels: pd.DataFrame,
    *,
    max_gap: int = 5,
    code_col: str = "code",
    date_col: str = "as_of",
    label_col: str = "label_trend_up",
    direction: str = "up",
) -> list[TrendEvent]:
    if labels.empty:
        return []
    frame = labels[labels[label_col].astype(bool)].copy()
    if frame.empty:
        return []
    frame[date_col] = pd.to_datetime(frame[date_col])
    events: list[TrendEvent] = []
    for code, grp in frame.groupby(code_col, sort=False):
        dates = sorted(pd.Timestamp(d) for d in grp[date_col])
        if not dates:
            continue
        all_dates = sorted(pd.Timestamp(d) for d in labels.loc[labels[code_col] == code, date_col])
        index = {d: i for i, d in enumerate(all_dates)}
        cluster = [dates[0]]
        for day in dates[1:]:
            prev = cluster[-1]
            gap = index.get(day, 0) - index.get(prev, 0)
            if gap <= max_gap:
                cluster.append(day)
            else:
                events.append(TrendEvent(code=str(code), direction=direction, onset=cluster[0], event_end=cluster[-1]))
                cluster = [day]
        events.append(TrendEvent(code=str(code), direction=direction, onset=cluster[0], event_end=cluster[-1]))
    events.sort(key=lambda e: (e.code, e.onset))
    return events


def match_alerts_to_events(
    alerts: pd.DataFrame,
    events: list[TrendEvent],
    *,
    lead: int = 3,
    horizon: int = 10,
    code_col: str = "code",
    date_col: str = "as_of",
    score_col: str = "p_trend_cal",
    calendar_by_code: dict[str, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    if alerts.empty:
        return pd.DataFrame(
            columns=[code_col, date_col, "matched_event_onset", "detection_delay", "is_duplicate", "is_fp"]
        )
    frame = alerts.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([code_col, date_col], kind="stable").reset_index(drop=True)
    frame["matched_event_onset"] = pd.NaT
    frame["detection_delay"] = np.nan
    frame["is_duplicate"] = False
    frame["is_fp"] = False

    by_code: dict[str, pd.DataFrame] = {str(code): grp for code, grp in frame.groupby(code_col, sort=False)}
    used_alert: set[int] = set()

    for event in sorted(events, key=lambda e: (e.onset, e.code)):
        code = str(event.code)
        sub = by_code.get(code)
        if sub is None or sub.empty:
            continue
        cal = None
        if calendar_by_code and code in calendar_by_code:
            cal = pd.DatetimeIndex(calendar_by_code[code])
        if cal is not None and event.onset in cal:
            onset_loc = int(cal.get_loc(event.onset))
            lo = cal[max(0, onset_loc - lead)]
            hi = cal[min(len(cal) - 1, onset_loc + horizon)]
        else:
            lo = event.onset - pd.tseries.offsets.BDay(lead)
            hi = event.onset + pd.tseries.offsets.BDay(horizon)

        dates = sub[date_col]
        in_window = (dates >= lo) & (dates <= hi)
        cand_idx = [int(i) for i in sub.index[in_window] if int(i) not in used_alert]
        if not cand_idx:
            continue

        def _key(i: int) -> tuple:
            day = pd.Timestamp(frame.at[i, date_col])
            if cal is not None and event.onset in cal and day in cal:
                dist = abs(int(cal.get_loc(day)) - int(cal.get_loc(event.onset)))
            else:
                dist = abs((day - event.onset).days)
            score = float(frame.at[i, score_col]) if score_col in frame.columns else 0.0
            if not math.isfinite(score):
                score = 0.0
            return (dist, day, -score)

        best = sorted(cand_idx, key=_key)[0]
        d_best = pd.Timestamp(frame.at[best, date_col])
        if cal is not None and event.onset in cal and d_best in cal:
            delay = int(cal.get_loc(d_best)) - int(cal.get_loc(event.onset))
        else:
            delay = int((d_best - event.onset).days)
        frame.at[best, "matched_event_onset"] = event.onset
        frame.at[best, "detection_delay"] = delay
        used_alert.add(best)
        for i in cand_idx:
            if i == best:
                continue
            frame.at[i, "is_duplicate"] = True
            used_alert.add(i)

    unmatched = frame["matched_event_onset"].isna() & ~frame["is_duplicate"].astype(bool)
    frame.loc[unmatched, "is_fp"] = True
    return frame
