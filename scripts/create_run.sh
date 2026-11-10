#!/usr/bin/env bash
# Create a fresh working directory without overwriting recorded report evidence.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [ "$#" -ne 1 ]; then echo 'Usage: bash scripts/create_run.sh /path/to/new-empty-project'; exit 2; fi
target="$1"
if [ -e "$target" ]; then echo "Папка уже существует: $target. Выберите новое имя; ничего не перезаписано."; exit 1; fi
mkdir -p "$target"
for directory in src tests; do
  mkdir -p "$target/$directory"
  find "$root/$directory" -maxdepth 1 -type f -name '*.py' -exec cp -p {} "$target/$directory/" \;
done
mkdir -p "$target/scripts" "$target/screenshots" "$target/data/raw"
for name in install.sh bootstrap_python.sh install_docker_ubuntu.sh mlflow_server.py run.sh collect.py update_humidity_fix.sh; do
  cp -p "$root/scripts/$name" "$target/scripts/$name"
done
for name in requirements.txt requirements.lock.txt requirements-runtime.txt requirements-runtime.lock.txt Dockerfile compose.yaml .dockerignore .gitignore; do
  cp -p "$root/$name" "$target/$name"
done
cp -p "$root/docs/RUN_UBUNTU.md" "$target/README.md"
cp -p "$root/data/raw/ccpp.zip" "$root/data/raw/Folds5x2_pp.xlsx" "$target/data/raw/"
echo "Чистый проект создан: $target"
echo 'Он содержит тот же исходный архив и зафиксированные зависимости.'
echo 'Перейдите в эту папку и выполните bash scripts/install.sh и bash scripts/run.sh experiments.'
