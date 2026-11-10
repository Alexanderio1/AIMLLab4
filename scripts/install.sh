#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reports/logs screenshots
if [ ! -d .venv ]; then
  if command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    bash scripts/bootstrap_python.sh
  fi
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12), "Existing .venv is not Python 3.12"'
.venv/bin/python -m pip install -r requirements.lock.txt 2>&1 | tee reports/logs/install.log
.venv/bin/python -m pip check 2>&1 | tee reports/logs/pip-check.log
.venv/bin/python -m pip freeze > reports/installed-requirements.txt
.venv/bin/python src/workflow.py environment
.venv/bin/python -m pytest -q 2>&1 | tee reports/logs/unit-tests.log
printf '\nУСТАНОВКА ЗАВЕРШЕНА. Далее: bash scripts/run.sh experiments\n'
