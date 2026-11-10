#!/usr/bin/env bash
# Ubuntu 22.04: use system Python only to install uv into a private tool venv.
# uv installs a separate CPython; /usr/bin/python3 and apt remain untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools reports/logs
exec > >(tee -a reports/logs/bootstrap-python.log) 2>&1
if [ -d .venv ]; then
  .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3,12)'
  echo 'Existing Python 3.12 .venv retained.'
  exit 0
fi
if [ ! -x .tools/uv/bin/python ]; then
  python3 -m venv .tools/uv
fi
.tools/uv/bin/python -m pip install --disable-pip-version-check 'uv==0.13.5'
export UV_CACHE_DIR="$PWD/.tools/cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
.tools/uv/bin/uv --version
.tools/uv/bin/uv --system-certs python install 3.12.14 --no-bin
.tools/uv/bin/uv --system-certs venv --python 3.12.14 --seed .venv
.venv/bin/python --version
echo 'Separate Python 3.12 is ready. System Python has not been replaced.'
