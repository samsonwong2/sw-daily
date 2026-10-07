"""Drop empty qlib calendar days and start a series after a long hole.

A break longer than ``GAP_BREAK_DAYS`` (801951 stops on 2017-01-20 and
resumes on 2021-12-13) starts a new chart. Bars before that resume are not
drawn and are not used to fit the fair-path rails.
"""
from __future__ import annotations

import pandas as pd

GAP_BREAK_DAYS = 30


def prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Real sessions only, clipped to the latest continuous stretch."""
    if frame is None or frame.empty or "datetime" not in frame.columns:
        return frame
    out = frame.copy()
    for col in ("$open", "$high", "$low", "$close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["$close", "$open", "$high", "$low"])
    out = out[out["$close"] > 0]
    if out.empty:
        return out.reset_index(drop=True)
    out = out.sort_values("datetime").reset_index(drop=True)
    start = resume_after_gap(out)
    out = out[pd.to_datetime(out["datetime"]) >= start].reset_index(drop=True)
    return out


def resume_after_gap(frame: pd.DataFrame) -> pd.Timestamp:
    """First bar of the latest stretch. No long hole means the first bar."""
    dates = pd.to_datetime(frame["datetime"]).sort_values().reset_index(drop=True)
    if dates.empty:
        raise ValueError("ohlcv has no dates")
    gap = dates.diff().dt.days
    breaks = gap[gap > GAP_BREAK_DAYS]
    if breaks.empty:
        return pd.Timestamp(dates.iloc[0])
    return pd.Timestamp(dates.iloc[int(breaks.index[-1])])


def latest_stretch_panel(close: pd.DataFrame) -> pd.DataFrame:
    """NaN every column before its own resume date. Index must be datetimes."""
    out = close.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out.index = index
    for col in out.columns:
        valid = out[col].dropna()
        if len(valid) < 2:
            continue
        gap = valid.index.to_series().diff().dt.days
        breaks = gap[gap > GAP_BREAK_DAYS]
        if breaks.empty:
            continue
        resume = pd.Timestamp(breaks.index[-1])
        out.loc[out.index < resume, col] = pd.NA
    return out
