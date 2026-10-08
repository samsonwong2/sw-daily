from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sw_daily.adaptive.run import (
    find_prev_listing_dir,
    first_bar_date_from_frame,
    load_listing_lookup,
    merge_listing_lookups,
    resolve_config_source,
    run_listing,
    write_listing_cache,
)


def _ohlcv(close: np.ndarray, start: str) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "datetime": index,
            "$open": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "$close": close,
        }
    )


def _frames() -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, dict]]:
    frames = {
        "801010": _ohlcv(np.linspace(100, 160, 160), "2024-01-02"),
        "801020": _ohlcv(np.linspace(80, 90, 80), "2025-01-02"),
    }
    names = {"801010": "农林牧渔", "801020": "煤炭"}
    configs = {
        code: {"code": code, "method": "ma_stack_hyst", "method_params": {}, "rules": {}}
        for code in frames
    }
    return frames, names, configs


def test_listing_lookup_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "listing_dates.csv"
    write_listing_cache(path, {"801020": "2025-01-02", "801010": "2024-01-02"})
    loaded = load_listing_lookup(path)
    assert loaded["801010"] == "2024-01-02"
    assert merge_listing_lookups(loaded, {"801010": "2024-06-01"})["801010"] == "2024-06-01"


def test_first_bar_date_from_frame() -> None:
    frame = _ohlcv(np.linspace(100, 110, 10), "2023-09-07")
    assert first_bar_date_from_frame(frame) == "2023-09-07"


def test_resolve_config_source_finds_the_adaptive_dir(tmp_path: Path, monkeypatch) -> None:
    import sw_daily.adaptive.run as run_mod

    adaptive = tmp_path / "20260930all_adaptive" / "configs"
    adaptive.mkdir(parents=True)
    monkeypatch.setattr(run_mod, "ADAPTIVE_DIR", tmp_path)
    assert run_mod.resolve_config_source("2026-09-30") == tmp_path / "20260930all_adaptive"


def test_run_listing_writes_html_and_listing_dates(tmp_path: Path) -> None:
    frames, names, configs = _frames()
    summary = run_listing(
        frames,
        names,
        configs,
        as_of="2025-06-30",
        out_dir=tmp_path,
        jobs=1,
        listing_lookup={"801010": "2024-01-02"},
    )
    assert set(summary["code"]) == {"801010", "801020"}
    dates = pd.read_csv(tmp_path / "listing_dates.csv", dtype=str)
    listed = dict(zip(dates["code"], dates["listing"]))
    assert listed["801010"] == "2024-01-02"
    assert listed["801020"] == "2025-01-02"
    pages = list(tmp_path.glob("regime_transition_*_adaptive.html"))
    assert any("农林牧渔" in path.name for path in pages)
    text = next(path for path in pages if "农林牧渔" in path.name).read_text(encoding="utf-8")
    assert "ETF_STYLE_V1" in text
    assert "图12" in text or "FIG12_SIX_STATES_START" in text


def test_second_run_reuses_an_unchanged_page(tmp_path: Path) -> None:
    frames, names, configs = _frames()
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_listing(frames, names, configs, as_of="2025-06-30", out_dir=first, jobs=1)
    before = (first / "listing_dates.csv").read_bytes()
    run_listing(
        frames,
        names,
        configs,
        as_of="2025-06-30",
        out_dir=second,
        jobs=1,
        prev_dir=first,
    )
    summary = pd.read_csv(second / "batch_summary.csv")
    assert summary["reused"].astype(str).str.lower().eq("true").all()
    copied = next(second.glob("regime_transition_801010_*_adaptive.html"))
    original = next(first.glob("regime_transition_801010_*_adaptive.html"))
    assert copied.read_bytes() == original.read_bytes()
    assert (first / "listing_dates.csv").read_bytes() == before


def test_one_new_bar_appends_instead_of_redrawing(tmp_path: Path, monkeypatch) -> None:
    # A one-year trail needs enough history for the fig7 candle the append path reads.
    frames = {
        "801010": _ohlcv(np.linspace(100, 180, 400), "2023-01-03"),
        "801020": _ohlcv(np.linspace(80, 120, 400), "2023-01-03"),
    }
    names = {"801010": "农林牧渔", "801020": "煤炭"}
    configs = {
        code: {"code": code, "method": "ma_stack_hyst", "method_params": {}, "rules": {}}
        for code in frames
    }
    anchor = pd.Series(np.linspace(3.0, 4.0, 400), index=pd.bdate_range("2023-01-03", periods=400))
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_listing(frames, names, configs, as_of="2025-06-30", out_dir=first, jobs=1, anchor_close=anchor)
    extended = {}
    new_end = None
    for code, frame in frames.items():
        last = frame.iloc[-1]
        nxt = pd.Timestamp(last["datetime"]) + pd.offsets.BDay(1)
        new_end = nxt.strftime("%Y%m%d")
        extra = {
            "datetime": nxt,
            "$open": float(last["$open"]) * 1.01,
            "$high": float(last["$high"]) * 1.01,
            "$low": float(last["$low"]) * 1.01,
            "$close": float(last["$close"]) * 1.01,
        }
        extended[code] = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)

    def _refuse_redraw(*_args, **_kwargs):
        raise AssertionError("redrawn")

    monkeypatch.setattr("sw_daily.adaptive.etf_page.write_etf_html", _refuse_redraw)
    anchor2 = pd.concat([
        anchor,
        pd.Series({pd.Timestamp(anchor.index[-1]) + pd.offsets.BDay(1): float(anchor.iloc[-1]) * 1.01}),
    ])
    summary = run_listing(
        extended,
        names,
        configs,
        as_of="2025-06-30",
        out_dir=second,
        jobs=1,
        prev_dir=first,
        anchor_close=anchor2,
    )
    assert summary["incremental"].astype(str).str.lower().eq("true").all()
    page = next(second.glob("regime_transition_801010_*_adaptive.html"))
    assert new_end in page.name
    text = page.read_text(encoding="utf-8")
    assert "ETF_STYLE_V1" in text
    assert "FIG12_SIX_STATES_START" in text or "图12" in text


def test_find_prev_listing_dir_picks_the_latest_older_day(tmp_path: Path) -> None:
    for tag in ("20260929", "20260928"):
        day = tmp_path / f"{tag}_from_listing"
        day.mkdir()
        pd.DataFrame({"code": ["801010"], "listing": ["2020-01-02"]}).to_csv(day / "listing_dates.csv", index=False)
    (tmp_path / "20260930_from_listing").mkdir()
    found = find_prev_listing_dir("2026-09-30", root=tmp_path)
    assert found is not None
    assert found.name == "20260929_from_listing"
