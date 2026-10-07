#!/usr/bin/env bash
# Append today's Shenwan second-level industry bars and refresh the standalone
# qlib dataset. Paths come from configs/sw_daily.json, or the defaults in
# src/sw_daily/paths.py.
#
#   ./scripts/daily_sw_update.sh
#   ./scripts/daily_sw_update.sh --force-today
#   ./scripts/daily_sw_update.sh --date 2026-10-07
#   PY=/home/huangtuo/python/envs/py312/bin/python ./scripts/daily_sw_update.sh
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_HERE}/daily_env.sh"
exec "${PY}" -m sw_daily etl daily "$@"
