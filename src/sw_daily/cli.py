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

Adaptive (same as etf-daily adaptive --as-of):
    sw-daily adaptive --as-of 2026-09-30

Listing (same as etf-daily listing --as-of):
    sw-daily listing --as-of 2026-09-30

HRP (same as etf-daily hrp --as-of):
    sw-daily hrp --as-of 2026-09-30

Rules7 (same as etf-daily rules7 --listing-dir):
    sw-daily rules7 --listing-dir ~/temp/sw/adaptive/20260930_from_listing
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

    adaptive = sub.add_parser("adaptive", help="Per-industry regime method and hold-up HTML.")
    adaptive.add_argument("--as-of", required=True, help="plot window end YYYY-MM-DD")
    adaptive.add_argument("--start-date", default="2026-04-01", help="plot window start")
    adaptive.add_argument("--train-cutoff", default=None, help="train on bars before this date (default: start-date)")
    adaptive.add_argument("--jobs", type=int, default=8)
    adaptive.add_argument("--retrain", action="store_true", help="ignore the frozen method cache")
    adaptive.add_argument("--code", action="append", default=None, help="limit to these industry codes")
    adaptive.add_argument("--out-dir", default=None)

    listing = sub.add_parser("listing", help="Hold-up HTML from each industry's first bar through --as-of.")
    listing.add_argument("--as-of", required=True, help="plot window end YYYY-MM-DD")
    listing.add_argument("--start-date", default=None, help="override plot start; still not before the first bar")
    listing.add_argument("--config-source-dir", default=None, help="adaptive dir with configs/ (default: that day's all_adaptive)")
    listing.add_argument("--out-dir", default=None)
    listing.add_argument("--jobs", type=int, default=1)
    listing.add_argument("--code", action="append", default=None, help="limit to these industry codes")
    listing.add_argument("--incremental-from", default=None, help="previous *_from_listing directory")
    listing.add_argument("--no-incremental", action="store_true", help="do not reuse an earlier from_listing page")

    hrp = sub.add_parser("hrp", help="Ward cluster reps and HRP-CVaR dendrogram for the selected pool.")
    hrp.add_argument("--as-of", required=True, help="window end YYYY-MM-DD")
    hrp.add_argument("--lookback-days", type=int, default=252)
    hrp.add_argument("--dist-t", type=float, default=0.40, help="Ward distance cut, dist = 1 - corr")
    hrp.add_argument("--min-corr", type=float, default=0.55, help="split a cluster when pairwise corr is below this")
    hrp.add_argument("--rep-window", type=int, default=20)
    hrp.add_argument("--min-rep-move", type=float, default=5.0, help="minimum |return| percent for an extreme mover")
    hrp.add_argument("--n-clusters", type=int, default=None, help="force this many clusters instead of --dist-t")
    hrp.add_argument("--out", default=None, help="HTML path (CSV is written beside it)")

    rules7 = sub.add_parser("rules7", help="Today's buy/sell checklist for a from_listing directory.")
    rules7.add_argument("--listing-dir", required=True, help="directory of regime_transition_*_adaptive.html")
    rules7.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: date in the directory name)")
    rules7.add_argument("--anchor-code", default=None, help="gap anchor (default: SH510300)")
    rules7.add_argument("--jobs", type=int, default=8)
    rules7.add_argument("--cluster-csv", default=None, help="HRP cluster_representatives CSV")
    rules7.add_argument("--out", default=None, help="checklist CSV (markdown is written beside it)")
    rules7.add_argument("--rewrite-from-csv", action="store_true", help="re-attach cluster columns without recomputing")
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
    elif args.command == "adaptive":
        from pathlib import Path

        from sw_daily.adaptive.run import run_from_qlib

        return run_from_qlib(
            as_of=args.as_of,
            start_date=args.start_date,
            train_cutoff=args.train_cutoff,
            jobs=int(args.jobs),
            retrain=bool(args.retrain),
            codes=args.code,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
    elif args.command == "listing":
        from pathlib import Path

        from sw_daily.adaptive.run import run_listing_from_qlib

        return run_listing_from_qlib(
            as_of=args.as_of,
            start_date=args.start_date,
            config_source_dir=Path(args.config_source_dir) if args.config_source_dir else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            jobs=int(args.jobs),
            codes=args.code,
            incremental_from=Path(args.incremental_from) if args.incremental_from else None,
            no_incremental=bool(args.no_incremental),
        )
    elif args.command == "hrp":
        from pathlib import Path

        from sw_daily.hrp.dendrogram import run_hrp_from_qlib

        return run_hrp_from_qlib(
            asof_date=args.as_of,
            lookback_days=int(args.lookback_days),
            dist_t=float(args.dist_t),
            min_corr=float(args.min_corr),
            rep_window=int(args.rep_window),
            min_rep_move=float(args.min_rep_move),
            n_clusters=args.n_clusters,
            output=Path(args.out) if args.out else None,
        )
    elif args.command == "rules7":
        from sw_daily.paths import ANCHOR_CODE
        from sw_daily.rules7.scan import main as rules7_main

        argv = ["--listing-dir", args.listing_dir, "--jobs", str(args.jobs)]
        if args.as_of:
            argv += ["--as-of", args.as_of]
        argv += ["--anchor-code", args.anchor_code or ANCHOR_CODE]
        if args.cluster_csv:
            argv += ["--cluster-csv", args.cluster_csv]
        if args.out:
            argv += ["--out", args.out]
        if args.rewrite_from_csv:
            argv.append("--rewrite-from-csv")
        return rules7_main(argv)
    else:
        raise SystemExit(f"unknown command {args.command}")
    return 0
