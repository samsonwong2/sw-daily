"""Train a regime method per industry and trade the frozen hold-up rule.

Equivalent to ``etf-daily adaptive --as-of``. HTML is the twelve-panel page
(price through fig11, plus fig12) and the filename includes the Chinese name.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from sw_daily.adaptive.stage import (
    FIXED_HYBRID_METHOD,
    METHOD_IMPL_VERSION,
    MIN_TRAIN_BARS,
    REGIME_HOLD_UP_RULES,
    TRADE_MODE_HOLD_UP,
    TRUTH_VERSION,
    prepare_features,
    segments,
    select_regime_method_legacy,
    simulate_hold_up,
)
from sw_daily.paths import ADAPTIVE_DIR


def train_one_symbol(code: str, ohlcv: pd.DataFrame, train_cutoff: str) -> dict[str, Any]:
    train_end = (pd.Timestamp(train_cutoff) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    px = prepare_features(ohlcv)
    px_train = px[px.as_of <= train_end].reset_index(drop=True)
    out: dict[str, Any] = {
        "code": code,
        "train_end": str(px_train.as_of.iloc[-1].date()) if len(px_train) else None,
        "train_cutoff": train_cutoff,
        "train_bars": len(px_train),
        "skipped_train": False,
        "trade_mode": TRADE_MODE_HOLD_UP,
        "truth_version": TRUTH_VERSION,
        "rules": REGIME_HOLD_UP_RULES,
    }
    if len(px_train) < MIN_TRAIN_BARS:
        out["skipped_train"] = True
        out["method"] = FIXED_HYBRID_METHOD
        out["method_params"] = {}
        return out
    method, params, meta = select_regime_method_legacy(px_train)
    out["method"] = method
    out["method_params"] = params
    out["method_score"] = float(meta.get("score") or 0.0)
    out["method_select_meta"] = {k: v for k, v in meta.items() if k != "scores"}
    return out


def oos_one(
    code: str,
    cfg: dict[str, Any],
    ohlcv: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    name: str,
    out_dir: Path,
    anchor_close: pd.Series | None = None,
) -> dict[str, Any]:
    method = cfg["method"]
    params = cfg.get("method_params") or {}
    px = prepare_features(ohlcv, method=method, method_params=params)
    window = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    clipped = False
    if len(window) < 5:
        available = px[px.as_of <= pd.Timestamp(end_date)].reset_index(drop=True)
        if len(available) < 5:
            raise ValueError(f"{code} has only {len(available)} bars on or before {end_date}")
        window = available
        start_date = str(window.as_of.iloc[0].date())
        end_date = str(window.as_of.iloc[-1].date())
        clipped = True
    close = window["$close"].to_numpy(float)
    dates = window.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = segments(window.regime.to_numpy(), dates, close)
    events, stats, open_mtm = simulate_hold_up(window)

    fixed = prepare_features(ohlcv, method=FIXED_HYBRID_METHOD)
    fixed_w = fixed[(fixed.as_of >= start_date) & (fixed.as_of <= end_date)].reset_index(drop=True)
    if fixed_w.empty:
        fixed_w = fixed[fixed.as_of <= pd.Timestamp(end_date)].reset_index(drop=True)
    _, stats_fixed, _ = simulate_hold_up(fixed_w)

    html_name = _write_html(
        out_dir, code, name, ohlcv, seg_df, events, open_mtm, stats,
        start_date, end_date, method, params, str(cfg.get("train_cutoff") or start_date),
        anchor_close=anchor_close,
    )
    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(trades_dir / f"{code}_trades.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(trades_dir / f"{code}_segments.csv", index=False, encoding="utf-8-sig")
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    saved = dict(cfg)
    saved["oos"] = {
        "start": start_date,
        "end": end_date,
        "compound": stats["compound"],
        "bh": stats["bh"],
        "edge": stats["edge"],
        "fixed_hybrid_edge": stats_fixed["edge"],
    }
    (cfg_dir / f"{code}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "code": code,
        "name": name,
        "out": html_name,
        "method": method,
        "train_bars": cfg.get("train_bars"),
        "skipped_train": cfg.get("skipped_train"),
        "bars": len(window),
        "compound": stats["compound"],
        "bh": stats["bh"],
        "edge": stats["edge"],
        "n_closed": stats["n"],
        "win": stats["win"],
        "open_pos": stats["open_pos"],
        "fixed_hybrid_edge": stats_fixed["edge"],
        "edge_vs_fixed": float(stats["edge"] - stats_fixed["edge"]),
        "n_seg": len(seg_df),
        "plotted_start": start_date,
        "plotted_end": end_date,
        "clipped_to_history": clipped,
    }


def _write_html(
    out_dir, code, name, history, seg_df, events, open_mtm, stats,
    start, end, method, method_params, train_cutoff, anchor_close=None,
) -> str:
    from sw_daily.adaptive.etf_page import write_etf_html

    return write_etf_html(
        Path(out_dir), code, name, history,
        method=method, method_params=method_params, train_cutoff=train_cutoff,
        anchor_close=anchor_close, start=start, end=end,
        seg_df=seg_df, events=events, open_mtm=open_mtm, stats=stats,
    )


def _cache_tag(train_cutoff: str) -> str:
    return f"{train_cutoff.replace('-', '')}_{TRADE_MODE_HOLD_UP}_{TRUTH_VERSION.replace('+', '-')}_{METHOD_IMPL_VERSION}"


def _train_worker(payload: tuple[str, pd.DataFrame, str]) -> dict[str, Any]:
    code, ohlcv, cutoff = payload
    return train_one_symbol(code, ohlcv, cutoff)


def _oos_worker(payload: tuple) -> dict[str, Any]:
    code, cfg, ohlcv, start, end, name, out_dir, anchor = payload
    try:
        return oos_one(
            code, cfg, ohlcv, start_date=start, end_date=end, name=name,
            out_dir=Path(out_dir), anchor_close=anchor,
        )
    except Exception as exc:  # noqa: BLE001 - one thin history must not stop the pool
        return {"code": code, "name": name, "error": str(exc)}


def run_adaptive(
    frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    *,
    as_of: str,
    start_date: str = "2026-04-01",
    train_cutoff: str | None = None,
    out_dir: Path | None = None,
    jobs: int = 1,
    retrain: bool = False,
    anchor_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Train, trade, and write HTML for each frame. ``frames`` maps code to OHLCV."""
    cutoff = train_cutoff or start_date
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    dest = Path(out_dir) if out_dir is not None else ADAPTIVE_DIR / f"{tag}all_adaptive"
    dest.mkdir(parents=True, exist_ok=True)
    cfg_dir = dest / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cache_meta = dest / f".train_cache_{_cache_tag(cutoff)}.json"

    configs: dict[str, dict] = {}
    if cache_meta.is_file() and not retrain:
        configs = json.loads(cache_meta.read_text(encoding="utf-8"))
    missing = [code for code in frames if code not in configs]
    if missing:
        payloads = [(code, frames[code], cutoff) for code in missing]
        trained: list[dict] = []
        if jobs <= 1 or len(payloads) == 1:
            trained = [_train_worker(item) for item in payloads]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                trained = list(pool.map(_train_worker, payloads))
        for cfg in trained:
            configs[cfg["code"]] = cfg
            (cfg_dir / f"{cfg['code']}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        cache_meta.write_text(json.dumps(configs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    oos_payloads = [
        (code, configs[code], frames[code], start_date, as_of, names.get(code, code), str(dest), anchor_close)
        for code in frames
        if code in configs and not configs[code].get("skipped_train")
    ]
    rows: list[dict] = []
    if jobs <= 1 or len(oos_payloads) <= 1:
        rows = [_oos_worker(item) for item in oos_payloads]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_oos_worker, item): item[0] for item in oos_payloads}
            for fut in as_completed(futs):
                rows.append(fut.result())
    errors = [row for row in rows if row.get("error")]
    for row in errors:
        print(f"[SKIP] {row['code']}: {row['error']}")
    rows = [row for row in rows if not row.get("error")]
    for row in rows:
        if row.get("clipped_to_history"):
            print(f"[INFO] {row['code']} 行情在请求窗口之前结束，改画 {row['plotted_start']}→{row['plotted_end']}")
    summary = pd.DataFrame(rows)
    path = dest / "batch_summary.csv"
    if path.is_file() and not summary.empty:
        prior = pd.read_csv(path)
        summary = pd.concat([prior, summary], ignore_index=True).drop_duplicates(subset=["code"], keep="last")
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    _write_readme(dest, summary, cutoff, start_date, as_of, len(frames))
    print(f"[OK] adaptive out={dest} ok={len(rows)}")
    return summary


def _write_readme(dest: Path, summary: pd.DataFrame, cutoff: str, start: str, end: str, n_codes: int) -> None:
    def _mean(col: str) -> str:
        if summary.empty or col not in summary.columns:
            return "n/a"
        return f"{float(summary[col].mean()):+.2%}"

    text = f"""# 按标的自适应趋势阶段（hold_up）

训练截止 {cutoff}，样本外图窗 {start} → {end}。成功 {len(summary)}/{n_codes}。

上涨态开始买入，上涨态结束卖出，下跌和横盘不交易。每个行业在训练期从 7 种因果分段法里选训练分最高的一种。

| 指标 | 值 |
| --- | --- |
| mean edge | {_mean("edge")} |
| mean win | {_mean("win")} |
| 固定 hybrid mean edge | {_mean("fixed_hybrid_edge")} |
| mean (adaptive − fixed) | {_mean("edge_vs_fixed")} |

输出：`*_adaptive.html`、`configs/{{code}}.json`、`batch_summary.csv`、`trades/`。
"""
    (dest / "README.md").write_text(text, encoding="utf-8")


def run_from_qlib(
    *,
    as_of: str,
    start_date: str = "2026-04-01",
    train_cutoff: str | None = None,
    jobs: int = 8,
    retrain: bool = False,
    codes: list[str] | None = None,
    out_dir: Path | None = None,
) -> int:
    from sw_daily.market_bars import prepare_ohlcv
    from sw_daily.pool.data_loading import load_ohlcv
    from sw_daily.pool.selector import CSV_SELECTED_OUT, load_sw_universe

    _, name_map = load_sw_universe()
    if codes:
        selected = list(codes)
    else:
        if not CSV_SELECTED_OUT.is_file():
            raise SystemExit(f"missing {CSV_SELECTED_OUT}; run: sw-daily pool")
        selected = pd.read_csv(CSV_SELECTED_OUT, dtype=str)["code"].astype(str).str.strip().tolist()
    names = {code: name_map.get(code, code) for code in selected}
    load_start = "2005-01-01"
    frames: dict[str, pd.DataFrame] = {}
    for code in selected:
        frame = prepare_ohlcv(load_ohlcv(code, load_start, as_of))
        if frame.empty:
            print(f"[skip] {code} empty ohlcv")
            continue
        frames[code] = frame
    anchor = _load_anchor_close(as_of)
    run_adaptive(
        frames,
        names,
        as_of=as_of,
        start_date=start_date,
        train_cutoff=train_cutoff,
        out_dir=out_dir,
        jobs=jobs,
        retrain=retrain,
        anchor_close=anchor,
    )
    return 0


def _load_anchor_close(end: str) -> pd.Series | None:
    """SH510300 close used as the fair-need anchor. Loaded before industry bars."""
    from sw_daily.paths import ANCHOR_CODE, FUND_QLIB_DIR
    from sw_daily.pool.data_loading import load_ohlcv

    frame = load_ohlcv(ANCHOR_CODE, "2005-01-01", end, provider_uri=str(FUND_QLIB_DIR))
    if frame.empty:
        print(f"[WARN] anchor {ANCHOR_CODE} has no bars; gap panels will be empty")
        return None
    series = pd.Series(frame["$close"].to_numpy(float), index=pd.to_datetime(frame["datetime"]).dt.normalize())
    return series[~series.index.duplicated(keep="last")].sort_index()


def first_bar_date_from_frame(ohlcv: pd.DataFrame) -> str:
    if ohlcv.empty or "datetime" not in ohlcv.columns:
        raise ValueError("ohlcv has no datetime column")
    return pd.Timestamp(ohlcv["datetime"].min()).strftime("%Y-%m-%d")


def load_listing_lookup(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    frame = pd.read_csv(path, dtype=str)
    if "code" not in frame.columns or "listing" not in frame.columns:
        return {}
    out: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        code = str(row.code).strip()
        listing = str(row.listing)[:10]
        if code and listing and listing.lower() not in {"nan", "none"}:
            out[code] = listing
    return out


def merge_listing_lookups(*lookups: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for lookup in lookups:
        merged.update(lookup)
    return merged


def write_listing_cache(path: Path, lookup: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"code": code, "listing": lookup[code]} for code in sorted(lookup)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def resolve_config_source(as_of: str, explicit: Path | None = None) -> Path:
    """Directory that holds frozen ``configs/*.json`` from ``sw-daily adaptive``."""
    if explicit is not None:
        source = Path(explicit)
        if not (source / "configs").is_dir():
            raise FileNotFoundError(f"config source has no configs/: {source}")
        return source
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    candidate = ADAPTIVE_DIR / f"{tag}all_adaptive"
    if (candidate / "configs").is_dir():
        return candidate
    found = sorted(ADAPTIVE_DIR.glob("*all_adaptive"), reverse=True)
    for path in found:
        if (path / "configs").is_dir():
            return path
    raise FileNotFoundError(f"no adaptive configs for {as_of}; run: sw-daily adaptive --as-of {as_of}")


def seed_out_dir(out_dir: Path, source: Path, codes: list[str]) -> None:
    """Copy frozen method configs. Source wins when both sides have a file."""
    dest = out_dir / "configs"
    dest.mkdir(parents=True, exist_ok=True)
    src_cfg = source / "configs"
    missing: list[str] = []
    for code in codes:
        src = src_cfg / f"{code}.json"
        dst = dest / f"{code}.json"
        if src.is_file():
            shutil.copy2(src, dst)
        elif not dst.is_file():
            missing.append(code)
    if missing:
        raise FileNotFoundError(f"missing frozen configs for {', '.join(missing)} in {src_cfg}")


def find_prev_listing_dir(as_of: str, root: Path | None = None) -> Path | None:
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    base = root or ADAPTIVE_DIR
    best: tuple[str, Path] | None = None
    for path in base.glob("*_from_listing"):
        if not (path / "listing_dates.csv").is_file():
            continue
        name = path.name.removesuffix("_from_listing")
        if len(name) == 8 and name.isdigit() and name < tag and (best is None or name > best[0]):
            best = (name, path)
    return None if best is None else best[1]


def _prev_html(prev_dir: Path, code: str) -> Path | None:
    matches = sorted(prev_dir.glob(f"regime_transition_{code}_*_adaptive.html"))
    return matches[-1] if matches else None


def _close_on(ohlcv: pd.DataFrame, day: str) -> float | None:
    dates = pd.to_datetime(ohlcv["datetime"]).dt.strftime("%Y-%m-%d")
    hit = ohlcv.loc[dates == day, "$close"]
    if hit.empty:
        return None
    return float(hit.iloc[-1])


def _can_reuse_prev(ohlcv: pd.DataFrame, prev_row: pd.Series, prev_html: Path, *, name: str, tol: float = 1e-6) -> bool:
    """Reuse the previous page only when the window, last close, and filename still match."""
    from sw_daily.adaptive.chart import sanitize_name_for_filename

    if not prev_html.is_file():
        return False
    if sanitize_name_for_filename(name) not in prev_html.name:
        return False
    text = prev_html.read_text(encoding="utf-8", errors="ignore")
    if "ETF_STYLE_V1" not in text:
        return False
    start_tag = pd.Timestamp(ohlcv["datetime"].min()).strftime("%Y%m%d")
    if start_tag not in prev_html.name:
        return False
    prev_end = str(prev_row.get("plot_end", ""))[:10]
    prev_close = prev_row.get("close_last")
    if not prev_end or pd.isna(prev_close):
        return False
    last = pd.Timestamp(ohlcv["datetime"].max()).strftime("%Y-%m-%d")
    if last != prev_end:
        return False
    current = _close_on(ohlcv, prev_end)
    if current is None:
        return False
    base = float(prev_close)
    if base == 0:
        return abs(current) <= tol
    return abs(current - base) / abs(base) <= tol


def run_listing(
    frames: dict[str, pd.DataFrame],
    names: dict[str, str],
    configs: dict[str, dict],
    *,
    as_of: str,
    start_override: str | None = None,
    out_dir: Path | None = None,
    jobs: int = 1,
    prev_dir: Path | None = None,
    listing_lookup: dict[str, str] | None = None,
    anchor_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Plot each industry from its first bar through ``as_of`` using frozen configs."""
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    dest = Path(out_dir) if out_dir is not None else ADAPTIVE_DIR / f"{tag}_from_listing"
    dest.mkdir(parents=True, exist_ok=True)
    known = dict(listing_lookup or {})
    listing_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    codes = list(frames)
    payloads = [
        (
            code,
            frames[code],
            names.get(code, code),
            configs.get(code),
            as_of,
            start_override,
            str(dest),
            str(prev_dir) if prev_dir else "",
            known.get(code, ""),
            anchor_close,
        )
        for code in codes
    ]
    produced: list[tuple] = []
    if jobs <= 1 or len(payloads) <= 1:
        for payload in payloads:
            try:
                produced.append(_listing_worker(*payload))
            except Exception as exc:  # noqa: BLE001
                errors.append({"code": payload[0], "error": str(exc)})
                print(f"[SKIP] {payload[0]}: {exc}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            futs = {pool.submit(_listing_worker, *payload): payload[0] for payload in payloads}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    produced.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append({"code": code, "error": str(exc)})
                    print(f"[SKIP] {code}: {exc}")
    for listing_row, summary, _err in produced:
        listing_rows.append(listing_row)
        if summary is not None:
            summary_rows.append(summary)
            if summary.get("reused"):
                print(f"[REUSE] {listing_row['code']} {listing_row['plot_start']}→{listing_row['plot_end']}")
            else:
                print(f"[OK] {listing_row['code']} {listing_row['plot_start']}→{listing_row['plot_end']}")
    listing_out = pd.DataFrame(listing_rows)
    if not listing_out.empty:
        prior_path = dest / "listing_dates.csv"
        if prior_path.is_file() and prior_path.stat().st_size > 0:
            prior = pd.read_csv(prior_path, dtype=str)
            listing_out = pd.concat([prior, listing_out.astype(str)], ignore_index=True).drop_duplicates(subset=["code"], keep="last")
        listing_out.to_csv(dest / "listing_dates.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    summary_path = dest / "batch_summary.csv"
    if summary_path.is_file() and not summary.empty:
        try:
            prior = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            prior = pd.DataFrame()
        if not prior.empty:
            summary = pd.concat([prior, summary], ignore_index=True).drop_duplicates(subset=["code"], keep="last")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(dest / "from_listing_errors.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] listing out={dest} ok={len(summary_rows)} skip={len(errors)}")
    return summary


def _listing_worker(code, frame, name, cfg, as_of, start_override, dest, prev_dir, known_listing, anchor_close=None):
    lookup = {code: known_listing} if known_listing else {}
    configs = {code: cfg} if cfg is not None else {}
    frames = {code: frame}
    # Re-enter the sequential body by calling the same decisions inline.
    listing = first_bar_date_from_frame(frame)
    if known_listing and pd.Timestamp(known_listing) >= pd.Timestamp(listing):
        listing = pd.Timestamp(known_listing).strftime("%Y-%m-%d")
    plot_start = listing
    if start_override:
        plot_start = max(pd.Timestamp(start_override), pd.Timestamp(listing)).strftime("%Y-%m-%d")
    last_day = pd.Timestamp(frame["datetime"].max()).strftime("%Y-%m-%d")
    plot_end = min(pd.Timestamp(as_of), pd.Timestamp(last_day)).strftime("%Y-%m-%d")
    listing_row = {
        "code": code,
        "listing": listing,
        "plot_start": plot_start,
        "plot_end": plot_end,
        "as_of": as_of,
        "close_last": _close_on(frame, plot_end),
    }
    prev = Path(prev_dir) if prev_dir else None
    if prev is not None and (prev / "listing_dates.csv").is_file():
        prev_frame = pd.read_csv(prev / "listing_dates.csv", dtype=str)
        hit = prev_frame[prev_frame["code"].astype(str) == code]
        html = _prev_html(prev, code)
        if html is not None and not hit.empty and _can_reuse_prev(frame, hit.iloc[0], html, name=name):
            shutil.copy2(html, Path(dest) / html.name)
            return listing_row, {"code": code, "name": name, "out": html.name, "reused": True, "method": (cfg or {}).get("method")}, None
    if cfg is None:
        raise KeyError(f"no frozen config for {code}")
    row = oos_one(
        code, cfg, frame, start_date=plot_start, end_date=plot_end, name=name,
        out_dir=Path(dest), anchor_close=anchor_close,
    )
    row["reused"] = False
    return listing_row, row, None


def run_listing_from_qlib(
    *,
    as_of: str,
    start_date: str | None = None,
    config_source_dir: Path | None = None,
    out_dir: Path | None = None,
    jobs: int = 1,
    codes: list[str] | None = None,
    incremental_from: Path | None = None,
    no_incremental: bool = False,
) -> int:
    from sw_daily.market_bars import prepare_ohlcv
    from sw_daily.pool.data_loading import load_ohlcv
    from sw_daily.pool.selector import CSV_SELECTED_OUT, load_sw_universe

    anchor = _load_anchor_close(as_of)
    source = resolve_config_source(as_of, config_source_dir)
    _, name_map = load_sw_universe()
    if codes:
        selected = list(codes)
    else:
        if not CSV_SELECTED_OUT.is_file():
            raise SystemExit(f"missing {CSV_SELECTED_OUT}; run: sw-daily pool")
        selected = pd.read_csv(CSV_SELECTED_OUT, dtype=str)["code"].astype(str).str.strip().tolist()
    configs: dict[str, dict] = {}
    for code in selected:
        path = source / "configs" / f"{code}.json"
        if path.is_file():
            configs[code] = json.loads(path.read_text(encoding="utf-8"))
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    dest = Path(out_dir) if out_dir is not None else ADAPTIVE_DIR / f"{tag}_from_listing"
    seed_out_dir(dest, source, [code for code in selected if code in configs])
    cache_path = ADAPTIVE_DIR / "_listing_dates.csv"
    lookup = load_listing_lookup(cache_path)
    frames: dict[str, pd.DataFrame] = {}
    for code in selected:
        if code not in configs:
            print(f"[SKIP] {code} no frozen config in {source}")
            continue
        frame = prepare_ohlcv(load_ohlcv(code, "2005-01-01", as_of))
        if frame.empty:
            print(f"[SKIP] {code} empty ohlcv")
            continue
        frames[code] = frame
        lookup[code] = first_bar_date_from_frame(frame)
    prev = None if no_incremental else (Path(incremental_from) if incremental_from else find_prev_listing_dir(as_of))
    if prev is not None:
        print(f"[INFO] incremental_from={prev}")
        lookup = merge_listing_lookups(load_listing_lookup(prev / "listing_dates.csv"), lookup)
    run_listing(
        frames,
        {code: name_map.get(code, code) for code in frames},
        configs,
        as_of=as_of,
        start_override=start_date,
        out_dir=dest,
        jobs=jobs,
        prev_dir=prev,
        listing_lookup=lookup,
        anchor_close=anchor,
    )
    write_listing_cache(cache_path, merge_listing_lookups(lookup, load_listing_lookup(dest / "listing_dates.csv")))
    return 0
