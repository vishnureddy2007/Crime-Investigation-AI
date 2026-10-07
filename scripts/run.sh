#!/usr/bin/env bash
#
# run.sh - one-command launcher for Linux / macOS.
#
# Creates a .venv on first run, installs requirements.txt,
# then launches Streamlit. Honours $APP_PORT (default 8501)
# and $APP_ADDRESS (default localhost).
#
# Usage:
#   ./scripts/run.sh                 # normal
#   ./scripts/run.sh --server.port 9000   # extra args passed to streamlit
#
set -euo pipefail

# ---- Locate repo root (one level up from this script) ---------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Resolve Python 3.10+ -------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[run.sh] ERROR: no python3 or python on PATH." >&2
    exit 1
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "[run.sh] Using ${PYTHON_BIN} ${PY_VERSION}"

# ---- Create venv on first run ----------------------------------------
VENV_DIR="${REPO_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[run.sh] Creating virtualenv at ${VENV_DIR} ..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---- Install requirements (idempotent; pip skips up-to-date) --------
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

# ---- Launch ---------------------------------------------------------
APP_PORT="${APP_PORT:-8501}"
APP_ADDRESS="${APP_ADDRESS:-localhost}"
APP_THEME="${APP_THEME:-light}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
URL="http://${APP_ADDRESS}:${APP_PORT}"
echo "[run.sh] Launching Streamlit on ${URL}"
echo "[run.sh] (Stop the server with Ctrl-C.)"
echo "[run.sh] APP_THEME=${APP_THEME} LOG_LEVEL=${LOG_LEVEL}"

export APP_PORT APP_ADDRESS APP_HEADLESS APP_THEME LOG_LEVEL LOG_FILE
exec streamlit run app.py \
    --server.port "${APP_PORT}" \
    --server.address "${APP_ADDRESS}" \
    --server.headless "${APP_HEADLESS:-false}" \
    "$@"
