from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sw_daily.etl.sw_index import _numeric_bars, normalize_sw_code, upsert_today_bar


def _bar(day: str, close: float) -> dict[str, object]:
    return {
        "date": day,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100,
        "code": "801010",
    }


def test_normalize_sw_code_keeps_six_digits() -> None:
    assert normalize_sw_code("801010.SI") == "801010"
    assert normalize_sw_code(801010) == "801010"
    assert normalize_sw_code("SW801010") == "801010"


def test_normalize_sw_code_rejects_short_values() -> None:
    with pytest.raises(ValueError):
        normalize_sw_code("abc")
    with pytest.raises(ValueError):
        normalize_sw_code("12345")


def test_numeric_bars_drops_bad_rows_and_keeps_first_duplicate() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-02", "not-a-date", "2026-01-01", "2026-01-02"],
            "open": [1, 9, 2, 3],
            "high": [1, 9, 2, 3],
            "low": [1, 9, 2, 3],
            "close": [1, None, 2, 3],
            "volume": [10, 90, 20, 30],
            "code": ["801010", "801010", "801010", "801010"],
        }
    )
    out = _numeric_bars(frame)
    assert list(out["date"]) == ["2026-01-01", "2026-01-02"]
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "code"]
    assert out.iloc[0]["close"] == 2
    assert out.iloc[1]["close"] == 1


def test_upsert_today_bar_writes_new_file(tmp_path: Path) -> None:
    status = upsert_today_bar("801010", _bar("2026-10-07", 1.5), force_today=False, csv_dir=tmp_path)
    assert status == "written"
    saved = pd.read_csv(tmp_path / "801010.csv")
    assert len(saved) == 1
    assert saved.iloc[0]["date"] == "2026-10-07"
    assert saved.iloc[0]["close"] == 1.5


def test_upsert_today_bar_leaves_existing_date(tmp_path: Path) -> None:
    upsert_today_bar("801010", _bar("2026-10-07", 1.5), force_today=False, csv_dir=tmp_path)
    status = upsert_today_bar("801010", _bar("2026-10-07", 9.0), force_today=False, csv_dir=tmp_path)
    assert status == "exists"
    saved = pd.read_csv(tmp_path / "801010.csv")
    assert len(saved) == 1
    assert saved.iloc[0]["close"] == 1.5


def test_upsert_today_bar_force_replaces_existing_date(tmp_path: Path) -> None:
    upsert_today_bar("801010", _bar("2026-10-06", 1.0), force_today=False, csv_dir=tmp_path)
    upsert_today_bar("801010", _bar("2026-10-07", 1.5), force_today=False, csv_dir=tmp_path)
    status = upsert_today_bar("801010", _bar("2026-10-07", 9.0), force_today=True, csv_dir=tmp_path)
    assert status == "written"
    saved = pd.read_csv(tmp_path / "801010.csv")
    assert list(saved["date"]) == ["2026-10-06", "2026-10-07"]
    assert saved.iloc[-1]["close"] == 9.0
