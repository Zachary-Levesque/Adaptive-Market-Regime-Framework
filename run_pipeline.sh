#!/usr/bin/env bash
# AMRF refresh-and-launch script.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
DEFAULT_SOURCE="$ROOT/d_us_txt.zip"

SOURCE=""
ALLOW_REMOTE_DOWNLOADS=0
WITH_RL=0
NO_DASHBOARD=0

usage() {
  cat <<'EOF'
Usage: ./run_pipeline.sh [--source PATH] [--allow-remote-downloads] [--with-rl] [--no-dashboard]

Options:
  --source PATH               Import local vendor data from a ZIP archive or directory.
  --allow-remote-downloads    Allow Phase 1 to fill missing data from remote providers.
  --with-rl                  Train and backtest the PPO agent after the static pipeline.
  --no-dashboard             Skip launching the Streamlit dashboard at the end.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --allow-remote-downloads)
      ALLOW_REMOTE_DOWNLOADS=1
      shift
      ;;
    --with-rl)
      WITH_RL=1
      shift
      ;;
    --no-dashboard)
      NO_DASHBOARD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_venv() {
  if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
}

needs_bootstrap() {
  "$PYTHON" - <<'PY'
import importlib.util
required = ["pandas", "plotly", "streamlit", "pyarrow"]
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in required) else 1)
PY
}

bootstrap_env() {
  echo "Bootstrapping .venv"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$ROOT/requirements-phase1.txt" -r "$ROOT/requirements.txt" -r "$ROOT/dashboard/requirements.txt"
}

run_module() {
  local module="$1"
  shift
  echo
  echo "==> python -m ${module} $*"
  "$PYTHON" -m "$module" "$@"
}

ensure_venv
if ! needs_bootstrap; then
  bootstrap_env
fi

cd "$ROOT"

if [[ -n "$SOURCE" ]]; then
  run_module src.data.import_price_files --config configs/config.yaml --source "$SOURCE"
elif [[ -f "$DEFAULT_SOURCE" ]]; then
  run_module src.data.import_price_files --config configs/config.yaml --source "$DEFAULT_SOURCE"
fi

run_module src.data.validate_phase1_inputs --config configs/config.yaml

if [[ "$ALLOW_REMOTE_DOWNLOADS" -eq 1 ]]; then
  run_module src.data.build_phase1 --config configs/config.yaml --allow-remote-downloads
else
  run_module src.data.build_phase1 --config configs/config.yaml
fi

run_module src.regime.build_phase2 --config configs/config.yaml
run_module src.alpha.build_model_comparison --config configs/config.yaml --skip-ensemble
run_module src.risk.build_phase4 --config configs/config.yaml
run_module src.alpha.build_diagnostics --config configs/config.yaml
run_module src.alpha.build_readiness --config configs/config.yaml

if [[ "$WITH_RL" -eq 1 ]]; then
  run_module src.rl.train_ppo --config configs/config.yaml
  run_module src.rl.backtest_rl --config configs/config.yaml
fi

run_module src.build_completion_report --config configs/config.yaml

if [[ "$NO_DASHBOARD" -eq 1 ]]; then
  echo
  echo "Pipeline complete. Dashboard launch skipped."
  exit 0
fi

echo
echo "Launching Streamlit dashboard..."
exec "$PYTHON" -m streamlit run dashboard/app.py
