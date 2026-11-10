#!/usr/bin/env bash
# Run from the existing AIMLLab4 project directory. Preserve all runtime data.
set -euo pipefail
if [ ! -f src/workflow.py ] || [ ! -x .venv/bin/python ]; then
  echo 'Сначала: cd ~/lab4-ubuntu22/AIMLLab4' >&2
  exit 1
fi
base=https://raw.githubusercontent.com/Alexanderio1/AIMLLab4/main
download_dir=$(mktemp -d "$PWD/.rh-update-XXXXXX")
trap 'rm -rf "$download_dir"' EXIT
files=(src/workflow.py src/app.py tests/test_contract.py)
for file in "${files[@]}"; do
  mkdir -p "$download_dir/$(dirname "$file")"
  curl -fL "$base/$file" -o "$download_dir/$file"
done
.venv/bin/python - "$download_dir" <<'PY'
import sys
from pathlib import Path
for path in Path(sys.argv[1]).rglob('*.py'):
    compile(path.read_text(), str(path), 'exec')
PY
backup_dir="reports/source-before-rh-fix-$(date +%Y%m%d-%H%M%S)"
for file in "${files[@]}"; do
  mkdir -p "$backup_dir/$(dirname "$file")" "$(dirname "$file")"
  if [ -f "$file" ]; then cp -p "$file" "$backup_dir/$file.before"; fi
  mv "$download_dir/$file" "$file"
done
.venv/bin/python -m pytest -q
echo "Исправление установлено. Резервные копии: $backup_dir"
echo 'Далее: bash scripts/run.sh experiments'
