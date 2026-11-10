#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
phase="${1:-}"
case "$phase" in experiments|v1|v2|rollback|compose|collect|stop) ;; *) echo 'Usage: bash scripts/run.sh experiments|v1|v2|rollback|compose|collect|stop'; exit 2;; esac
mkdir -p reports/logs screenshots
exec > >(tee -a "reports/logs/${phase}.log") 2>&1
printf '\n=== %s — %s ===\n' "$phase" "$(date --iso-8601=seconds)"
PY=.venv/bin/python
if [ ! -x "$PY" ]; then echo 'Сначала: bash scripts/install.sh'; exit 1; fi
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
if [ "$phase" = experiments ]; then
  if [ -f reports/registry.json ]; then
    "$PY" scripts/mlflow_server.py start
    if [ ! -f reports/test_metrics.json ]; then "$PY" src/workflow.py evaluate; fi
    echo 'Эксперименты уже выполнены. Повторный запуск не нужен: переходите к v1.'
    echo 'Для нового опыта распакуйте чистый комплект в другую папку.'
    exit 0
  fi
  "$PY" src/workflow.py environment
  "$PY" scripts/mlflow_server.py start
  "$PY" src/workflow.py prepare
  "$PY" src/workflow.py train
  "$PY" src/workflow.py register
  "$PY" src/workflow.py evaluate
  echo 'EXPERIMENTS COMPLETE. Откройте MLflow, сделайте скриншоты таблицы и Registry.'
  exit 0
fi
DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then DOCKER=(sudo -n docker)
  else echo 'Docker недоступен. Установите его по README; затем выполните sudo -v и повторите команду.'; exit 1; fi
fi
"${DOCKER[@]}" version
"${DOCKER[@]}" compose version
remove_standalone() {
  if "${DOCKER[@]}" container inspect ccpp-lab17-api >/dev/null 2>&1; then
    label=$("${DOCKER[@]}" inspect --format '{{ index .Config.Labels "lab" }}' ccpp-lab17-api)
    if [ "$label" != ccpp-17 ]; then echo 'Имя ccpp-lab17-api занято чужим контейнером'; exit 1; fi
    "${DOCKER[@]}" rm -f ccpp-lab17-api
  fi
}
standalone() {
  local tag="$1"
  remove_standalone
  "${DOCKER[@]}" run -d --name ccpp-lab17-api --label lab=ccpp-17 -p 127.0.0.1:8000:8000 "ccpp-api:$tag"
}
wait_health() {
  local id="$1"
  for ((i=0; i<90; i++)); do
    state=$("${DOCKER[@]}" inspect --format '{{.State.Health.Status}}' "$id")
    if [ "$state" = healthy ]; then echo "DOCKER HEALTHCHECK: healthy ($id)"; return; fi
    if [ "$state" = unhealthy ]; then break; fi
    sleep 1
  done
  "${DOCKER[@]}" logs "$id"
  echo "Docker healthcheck failed: $state"; exit 1
}
build_image() {
  local tag="$1"
  if [ ! -f reports/base-image.txt ]; then
    "${DOCKER[@]}" pull python:3.12-slim
    "${DOCKER[@]}" image inspect python:3.12-slim --format '{{index .RepoDigests 0}}' > reports/base-image.txt
  fi
  local base
  base=$(cat reports/base-image.txt)
  local extra=()
  # Cloud-only optional trust mount; normally unset on the student's Ubuntu.
  if [ -n "${CCPP_BUILD_CA:-}" ]; then extra+=(--secret "id=proxy_ca,src=$CCPP_BUILD_CA"); fi
  "${DOCKER[@]}" build "${extra[@]}" --build-arg "BASE_IMAGE=$base" --build-arg "MODEL_PATH=exported_models/$tag" -t "ccpp-api:$tag" .
  "${DOCKER[@]}" image inspect "ccpp-api:$tag" --format '{{json .}}' > "reports/image_$tag.json"
}
case "$phase" in
v1|v2)
  "$PY" scripts/mlflow_server.py start
  if [ "$phase" = v1 ]; then role=baseline; else role=selected; fi
  "$PY" src/workflow.py export --role "$role" --label "$phase"
  build_image "$phase"
  standalone "$phase"
  "$PY" src/workflow.py check --label "$phase" --phase "$phase"
  wait_health ccpp-lab17-api
  "${DOCKER[@]}" inspect ccpp-lab17-api --format '{{json .State.Health}}' > "reports/health_$phase.json"
  "${DOCKER[@]}" ps --filter name=ccpp-lab17-api
  ;;
rollback)
  "$PY" scripts/mlflow_server.py start
  # Reuse the original image; no export, fit or build during rollback.
  "$PY" -c 'import mlflow,json; from mlflow import MlflowClient; d=json.load(open("reports/registry.json")); v=d["versions"]["baseline"]["model_version"]; MlflowClient().set_registered_model_alias(d["model_name"], "champion", v); print("champion ->", v)'
  standalone v1
  "$PY" src/workflow.py check --label v1 --phase rollback
  wait_health ccpp-lab17-api
  "${DOCKER[@]}" inspect ccpp-lab17-api --format '{{json .State.Health}}' > reports/health_rollback.json
  "${DOCKER[@]}" ps --filter name=ccpp-lab17-api
  ;;
compose)
  "$PY" scripts/mlflow_server.py start
  "$PY" -c 'import json; from mlflow import MlflowClient; d=json.load(open("reports/registry.json")); v=d["versions"]["selected"]["model_version"]; MlflowClient().set_registered_model_alias(d["model_name"], "champion", v); print("champion ->", v)'
  remove_standalone
  "${DOCKER[@]}" compose -p ccpp-lab17 up -d --wait --wait-timeout 120
  "$PY" src/workflow.py check --label v2 --phase compose
  "${DOCKER[@]}" compose -p ccpp-lab17 ps
  "${DOCKER[@]}" compose -p ccpp-lab17 ps --format json > reports/compose_ps.json
  id=$("${DOCKER[@]}" compose -p ccpp-lab17 ps -q api)
  wait_health "$id"
  "${DOCKER[@]}" inspect "$id" --format '{{json .State.Health}}' > reports/health_compose.json
  ;;
collect|stop)
  remove_standalone
  "${DOCKER[@]}" compose -p ccpp-lab17 down
  "$PY" scripts/mlflow_server.py stop
  if [ "$phase" = collect ]; then "$PY" scripts/collect.py; fi
  ;;
esac
if [ "$phase" = v1 ] || [ "$phase" = v2 ] || [ "$phase" = rollback ] || [ "$phase" = compose ]; then
  if [ "$phase" = compose ]; then target="$id"; else target=ccpp-lab17-api; fi
  "${DOCKER[@]}" inspect "$target" --format '{"container_id":{{json .Id}},"image_id":{{json .Image}},"image_tag":{{json .Config.Image}},"health":{{json .State.Health.Status}}}' > "reports/container_$phase.json"
  cat "reports/container_$phase.json"
fi
printf '\nЭТАП %s ЗАВЕРШЁН УСПЕШНО\n' "$phase"
