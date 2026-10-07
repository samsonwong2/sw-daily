"""Load Shenwan index bars from the local qlib dataset.

The ETL writes one CSV per second-level industry under ``paths.CSV_DIR`` and
refreshes the qlib bin dataset under ``paths.QLIB_DIR``. The pool reads the
qlib dataset directly; index codes are plain 6-digit strings, so no
exchange-prefix normalization is needed.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from sw_daily.paths import QLIB_DIR

try:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
except Exception as exc:  # pragma: no cover
    qlib = None  # type: ignore[assignment]
    D = None  # type: ignore[assignment]
    REG_CN = None  # type: ignore[assignment]
    print("Warning: qlib import failed:", exc)

_QLIB_PROVIDER_URI: str | None = None


def _ensure_qlib(provider_uri: str) -> None:
    global _QLIB_PROVIDER_URI
    if qlib is None or D is None:
        raise RuntimeError("qlib is not importable in this interpreter")
    if _QLIB_PROVIDER_URI == str(provider_uri):
        return
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    _QLIB_PROVIDER_URI = str(provider_uri)


def filter_stockpool(instruments: Iterable[str]) -> list[str]:
    return [str(code).strip() for code in instruments if str(code).strip()]


def load_ohlcv(
    code: str,
    start: str,
    end: str,
    provider_uri: str | None = None,
) -> pd.DataFrame:
    """One symbol's open/high/low/close with a ``datetime`` column."""
    uri = str(provider_uri or QLIB_DIR)
    _ensure_qlib(uri)
    raw = D.features(
        [str(code).strip()],
        fields=["$open", "$high", "$low", "$close", "$volume"],
        start_time=start,
        end_time=end,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["datetime", "$open", "$high", "$low", "$close", "$volume"])
    frame = raw.reset_index()
    date_col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_col: "datetime"})
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    keep = ["datetime", "$open", "$high", "$low", "$close", "$volume"]
    return frame.loc[:, keep].sort_values("datetime").reset_index(drop=True)


def load_close_volume(
    instruments: Iterable[str],
    test_period: tuple[str, str],
    provider_uri: str | None = None,
    *,
    ffill: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ``$close`` and ``$volume`` panels (date index x code columns)."""
    uri = str(provider_uri or QLIB_DIR)
    _ensure_qlib(uri)
    stockpool = filter_stockpool(instruments)
    raw = D.features(
        stockpool,
        fields=["$close", "$volume"],
        start_time=test_period[0],
        end_time=test_period[1],
    )
    close = raw["$close"].unstack(level="instrument")
    vol = raw["$volume"].unstack(level="instrument")
    close.index.name = "Date"
    vol.index.name = "Date"
    close = close.sort_index()
    if ffill:
        close = close.ffill()
    vol = vol.sort_index()
    print("Loaded close shape", close.shape, "volume shape", vol.shape)
    return close, vol
