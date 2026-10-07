# Sourced by the daily shells. One place for the repo root and local config.
# Missing config.env keeps the public defaults.
_DAILY_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_DAILY_ENV_DIR}/.." && pwd)"
_SAVED_PY="${PY:-}"
if [[ -f "${PROJECT_ROOT}/config.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/config.env"
  set +a
fi
if [[ -n "${_SAVED_PY}" ]]; then
  PY="${_SAVED_PY}"
fi
PY="${PY:-python3}"
_SRC="${PROJECT_ROOT}/src"
case ":${PYTHONPATH:-}:" in
  *":${_SRC}:"*) ;;
  *) PYTHONPATH="${_SRC}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export PYTHONPATH PY PROJECT_ROOT

cd "${PROJECT_ROOT}"
