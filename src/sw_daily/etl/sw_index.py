"""Shenwan industry index ETL.

Writes industry metadata under ``paths.INFO_DIR`` and a standalone qlib dataset
under ``paths.QLIB_DIR``. It never reads or writes the ETF provider at
``all_fund_data``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import random
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from sw_daily.paths import (
    CSV_DIR,
    FIRST_INFO_CSV,
    INFO_DIR,
    LOG_DIR,
    QLIB_DIR,
    QLIB_SCRIPTS_DIR,
    SECOND_INFO_CSV,
)

BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume", "code"]
HIST_RENAME = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}
REALTIME_RENAME = {
    "今开盘": "open",
    "最高价": "high",
    "最低价": "low",
    "最新价": "close",
    "成交量": "volume",
}

logger = logging.getLogger("sw_index_etl")


def setup_logger() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_DIR / "sw_index_etl.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def normalize_sw_code(value: object) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 6:
        raise ValueError(f"cannot parse Shenwan code from {value!r}")
    return digits[:6]


def load_second_codes() -> list[str]:
    if not SECOND_INFO_CSV.is_file():
        raise SystemExit(f"missing {SECOND_INFO_CSV}; run: sw-daily etl info")
    frame = pd.read_csv(SECOND_INFO_CSV, dtype=str)
    if "行业代码" not in frame.columns:
        raise SystemExit(f"{SECOND_INFO_CSV} has no 行业代码 column")
    codes: list[str] = []
    seen: set[str] = set()
    for raw in frame["行业代码"].dropna():
        code = normalize_sw_code(raw)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _numeric_bars(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date")
    return out.loc[:, BAR_COLUMNS]


def save_history_frame(code: str, raw: pd.DataFrame, csv_dir: Path | None = None) -> Path:
    renamed = raw.rename(columns=HIST_RENAME)
    missing = [column for column in ("date", "open", "high", "low", "close", "volume") if column not in renamed.columns]
    if missing:
        raise ValueError(f"{code} history missing columns {missing}; got {list(raw.columns)}")
    renamed["code"] = code
    frame = _numeric_bars(renamed)
    if frame.empty:
        raise ValueError(f"{code} history is empty after cleaning")
    directory = CSV_DIR if csv_dir is None else csv_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{code}.csv"
    frame.to_csv(path, index=False)
    return path


def fetch_one_history(code: str, force: bool, csv_dir: Path | None = None) -> tuple[bool, bool]:
    """Return (success, requested_network)."""
    import akshare as ak

    directory = CSV_DIR if csv_dir is None else csv_dir
    path = directory / f"{code}.csv"
    if path.is_file() and not force:
        logger.info("skip existing %s", path.name)
        return True, False
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            raw = ak.index_hist_sw(symbol=code, period="day")
            path = save_history_frame(code, raw, csv_dir=directory)
            logger.info("saved %s rows=%s", path.name, len(pd.read_csv(path)))
            return True, True
        except Exception as exc:  # noqa: BLE001 - keep going through the industry list
            last_error = exc
            logger.warning("history %s attempt %s failed: %s", code, attempt, exc)
            time.sleep(2)
    logger.error("history %s failed: %s", code, last_error)
    return False, True


def run_info() -> None:
    import akshare as ak

    INFO_DIR.mkdir(parents=True, exist_ok=True)
    first = ak.sw_index_first_info()
    second = ak.sw_index_second_info()
    first.to_csv(FIRST_INFO_CSV, index=False, encoding="utf-8-sig")
    second.to_csv(SECOND_INFO_CSV, index=False, encoding="utf-8-sig")
    logger.info("wrote %s rows=%s", FIRST_INFO_CSV, len(first))
    logger.info("wrote %s rows=%s", SECOND_INFO_CSV, len(second))


def run_history(force: bool) -> None:
    codes = load_second_codes()
    logger.info("fetching history for %s second-level industries", len(codes))
    ok = 0
    for index, code in enumerate(codes, start=1):
        logger.info("[%s/%s] %s", index, len(codes), code)
        success, requested = fetch_one_history(code, force=force)
        if success:
            ok += 1
        if requested:
            time.sleep(random.randint(1, 3))
    logger.info("history done ok=%s failed=%s", ok, len(codes) - ok)
    if ok == 0:
        raise SystemExit("no history files written")
    dump_qlib()


def upsert_today_bar(
    code: str,
    bar: dict[str, object],
    force_today: bool,
    csv_dir: Path | None = None,
) -> str:
    directory = CSV_DIR if csv_dir is None else csv_dir
    path = directory / f"{code}.csv"
    row = _numeric_bars(pd.DataFrame([bar]))
    if row.empty:
        logger.warning("skip %s: realtime bar has no close", code)
        return "skip"
    today = str(row.iloc[0]["date"])
    if path.is_file():
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if today in set(existing["date"].dropna()):
            if not force_today:
                return "exists"
            existing = existing[existing["date"] != today]
        merged = pd.concat([existing, row], ignore_index=True)
    else:
        merged = row
    merged = _numeric_bars(merged)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return "written"


def run_daily(trade_date: str | None, force_today: bool) -> None:
    import akshare as ak

    realtime = ak.index_realtime_sw(symbol="二级行业")
    renamed = realtime.rename(columns=REALTIME_RENAME)
    required = ["指数代码", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise SystemExit(f"index_realtime_sw missing columns {missing}; got {list(realtime.columns)}")

    known = set(load_second_codes()) if SECOND_INFO_CSV.is_file() else None
    day = trade_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    written = exists = skipped = 0
    for record in renamed.to_dict(orient="records"):
        try:
            code = normalize_sw_code(record["指数代码"])
        except ValueError as exc:
            logger.warning("%s", exc)
            skipped += 1
            continue
        if known is not None and code not in known:
            continue
        bar = {
            "date": day,
            "open": record["open"],
            "high": record["high"],
            "low": record["low"],
            "close": record["close"],
            "volume": record["volume"],
            "code": code,
        }
        status = upsert_today_bar(code, bar, force_today=force_today)
        if status == "written":
            written += 1
        elif status == "exists":
            exists += 1
        else:
            skipped += 1
    seen_codes = {normalize_sw_code(value) for value in renamed["指数代码"].dropna()}
    if known is not None:
        missing_codes = sorted(known - seen_codes)
        if missing_codes:
            logger.warning("realtime response omitted %s codes: %s", len(missing_codes), ",".join(missing_codes))
    logger.info("daily %s written=%s already_present=%s skipped=%s", day, written, exists, skipped)
    if not any(CSV_DIR.glob("*.csv")):
        raise SystemExit(f"no csv files in {CSV_DIR}; run history first")
    dump_qlib()


def _qlib_dump_env() -> dict[str, str]:
    """Prefer the running interpreter's qlib. Fall back to the source tree.

    ``dump_bin.py`` imports ``qlib.utils``. Some interpreters already have that
    (the py312 env). The default conda env has an unrelated package also named
    ``qlib``, so the Microsoft source checkout has to go on ``PYTHONPATH``.
    """
    env = os.environ.copy()
    probe = [sys.executable, "-c", "from qlib.utils import fname_to_code"]
    if subprocess.run(probe, capture_output=True, env=env).returncode == 0:
        return env
    if QLIB_SCRIPTS_DIR is None:
        raise SystemExit("qlib.utils is not importable and the qlib source checkout is missing")
    qlib_root = str(Path(QLIB_SCRIPTS_DIR).parent)
    env["PYTHONPATH"] = qlib_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    probed = subprocess.run(probe, capture_output=True, env=env, text=True)
    if probed.returncode != 0:
        detail = (probed.stderr or probed.stdout or "").strip()
        raise SystemExit(f"cannot import qlib.utils for dump_bin: {detail}")
    logger.info("dump_bin will import qlib from %s", qlib_root)
    return env


def dump_qlib() -> None:
    if QLIB_SCRIPTS_DIR is None:
        raise SystemExit(f"dump_bin.py directory is missing: {QLIB_SCRIPTS_DIR}")
    script = Path(QLIB_SCRIPTS_DIR) / "dump_bin.py"
    if not script.is_file():
        raise SystemExit(f"dump_bin.py not found: {script}")
    if importlib.util.find_spec("fire") is None:
        raise SystemExit(f"missing dependency fire. Install with: {sys.executable} -m pip install fire")
    QLIB_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "dump_all",
        "--data_path",
        str(CSV_DIR),
        "--qlib_dir",
        str(QLIB_DIR),
        "--symbol_field_name",
        "code",
        "--date_field_name",
        "date",
        "--include_fields",
        "open,high,low,close,volume",
    ]
    env = _qlib_dump_env()
    logger.info("dump_bin %s", " ".join(command))
    subprocess.run(command, check=True, env=env)
    logger.info("qlib dataset updated at %s", QLIB_DIR)
