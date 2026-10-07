#!/usr/bin/env python
"""Unified runner for the Shenwan industry index workflow.

Daily use:
    sw-daily etl daily

Backfill:
    sw-daily etl info
    sw-daily etl history

Pool:
    sw-daily pool
    sw-daily cluster-review

Regime (same as etf-daily regime --skip-rebuild --skip-html):
    sw-daily regime
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shenwan industry index workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    etl = sub.add_parser("etl", help="Collect Shenwan industry index data and dump qlib.")
    etl_sub = etl.add_subparsers(dest="etl_command", required=True)
    etl_sub.add_parser("info", help="save first- and second-level industry info CSVs")
    history = etl_sub.add_parser("history", help="backfill second-level daily history and dump qlib")
    history.add_argument("--force", action="store_true", help="re-download codes that already have a CSV")
    daily = etl_sub.add_parser("daily", help="append today's realtime bars and dump qlib")
    daily.add_argument("--date", default=None, help="bar date YYYY-MM-DD (default: today)")
    daily.add_argument("--force-today", action="store_true", help="replace an existing bar for --date")

    pool = sub.add_parser("pool", help="Cluster second-level industries and write the selected pool CSVs.")
    pool.add_argument("--start-date", default="2020-01-01", help="lookback window start YYYY-MM-DD")
    pool.add_argument("--end-date", default="2026-08-31", help="lookback window end YYYY-MM-DD")

    review = sub.add_parser("cluster-review", help="Audit the selected pool against the full cluster mapping.")
    review.add_argument("--mapping-csv", default=None, help="full mapping CSV (default: pool output)")
    review.add_argument("--selected-csv", default=None, help="selected pool CSV (default: pool output)")
    review.add_argument("--future-end", default=None, help="review anchor date YYYY-MM-DD (default: today)")
    review.add_argument("--windows", default="5,20,60", help="trailing return windows, comma separated")
    review.add_argument("--min-regret", type=float, default=0.01)
    review.add_argument("--min-cluster-n", type=int, default=2)
    review.add_argument("--top-n", type=int, default=15)
    review.add_argument("--out-dir", default=None)

    regime = sub.add_parser(
        "regime",
        help="Frozen HMM regime-transition validation (no reversal rebuild, no HTML).",
    )
    regime.add_argument("--as-of", default=None, help="signal end date YYYY-MM-DD (default: today)")
    regime.add_argument("--eval-start", default="2024-06-01", help="first scored date YYYY-MM-DD")
    regime.add_argument("--horizon", type=int, default=10, help="trend-label horizon in trading days")
    regime.add_argument("--jobs", type=int, default=8, help="parallel HMM fits")
    regime.add_argument("--output-dir", default=None, help="decision pack directory")
    regime.add_argument("--cache-dir", default=None, help="frozen-model pickle cache")
    regime.add_argument("--no-cache", action="store_true")
    regime.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    from sw_daily.etl.sw_index import run_daily, run_history, run_info, setup_logger

    setup_logger()
    args = build_parser().parse_args(argv)
    if args.command == "etl" and args.etl_command == "info":
        run_info()
    elif args.command == "etl" and args.etl_command == "history":
        run_history(force=args.force)
    elif args.command == "etl" and args.etl_command == "daily":
        run_daily(trade_date=args.date, force_today=args.force_today)
    elif args.command == "pool":
        from sw_daily.pool.selector import select_pool

        select_pool(test_period=(args.start_date, args.end_date))
    elif args.command == "cluster-review":
        from sw_daily.pool.review import main as review_main

        review_args = []
        if args.mapping_csv:
            review_args += ["--mapping-csv", args.mapping_csv]
        if args.selected_csv:
            review_args += ["--selected-csv", args.selected_csv]
        if args.future_end:
            review_args += ["--future-end", args.future_end]
        review_args += ["--windows", args.windows]
        review_args += ["--min-regret", str(args.min_regret)]
        review_args += ["--min-cluster-n", str(args.min_cluster_n)]
        review_args += ["--top-n", str(args.top_n)]
        if args.out_dir:
            review_args += ["--out-dir", args.out_dir]
        return review_main(review_args)
    elif args.command == "regime":
        from pathlib import Path

        from sw_daily.regime.backtest import run_regime

        return run_regime(
            eval_start=args.eval_start,
            eval_end=args.as_of,
            horizon=int(args.horizon),
            jobs=int(args.jobs),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            use_cache=not args.no_cache,
            resume=not args.no_resume,
        )
    else:
        raise SystemExit(f"unknown command {args.command}")
    return 0
