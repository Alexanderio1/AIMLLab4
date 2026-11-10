# Лабораторная № 4: MLOps — вариант 17

Прогноз электрической мощности **Combined Cycle Power Plant** по `AT, V, AP, RH`.
Работа выполнена **10.11.2026 на Ubuntu 22.04.5 ARM64**, Python 3.12.14,
MLflow 3.5.0, scikit-learn 1.7.2, Docker 29.9.3 и Compose v5.7.0.

- **[answers_lab4.pdf](answers_lab4.pdf)** — подробные ответы на все **49 вопросов
  к защите**: 29 по отчёту и 20 по теории, с командами, содержанием и примерами
  из проекта. [Исходный текст](answers_lab4.md).
- **[lab4.ipynb](lab4.ipynb)** — выполненный notebook с таблицами, кодом,
  пояснениями и 14 оригинальными скриншотами Ubuntu, по образцу первой лабораторной.
- **[report.md](report.md)** — полный отчёт для чтения прямо на GitHub.
- **[report.html](report.html)** — версия для браузера. Скачайте файл и откройте
  локально: рисунки встроены, формулы читаются без внешнего MathJax.
- **[docs/RUN_UBUNTU.md](docs/RUN_UBUNTU.md)** — пошаговый запуск лабораторной.
- [data/](data/) — исходный ZIP/XLSX, очищенная таблица, split и три выборки.
- [reports/](reports/) — исходные метрики, run_id, Registry, запросы/ответы,
  Docker ID, healthcheck и журналы. [screenshots/](screenshots/) — оригинальные снимки.
- [src/](src/), [scripts/](scripts/), [tests/](tests/), [Dockerfile](Dockerfile),
  [compose.yaml](compose.yaml) и lock-файлы — приложение и воспроизведение.

## Основные результаты

Из 9568 строк **Sheet1** удалён 41 полный дубликат, осталось 9527.
Train/validation/test — **5715/1906/1906**, seed=42; другие листы не объединяются.
Измерения RH до 100,16% сохранены. Шесть запусков: три Ridge и три RF.

| Выбранная модель | Validation MAE, МВт | Test MAE, МВт | Test RMSE, МВт | Test R² |
|---|---:|---:|---:|---:|
| RF, 250 деревьев, max_depth=None | 2,472198 | **2,490805** | 3,601666 | 0,956215 |

Run ID победителя: `503b388935b047bca0fa775e9603e454`.
Registry: **ccpp-power**, version **1 — Ridge**, **2 — RF**; после Compose
`champion → 2`. Проверены `ccpp-api:v1 → v2 → откат v1`, совпадение API с
экспортом, восемь ошибочных сценариев и Compose **healthy**.

### Примечание к исходному флагу проверки

В [оригинальном evidence_status.json](reports/evidence_status.json) сохранено
`update_verified: false`. Сборщик сравнивал float-ответы через точное `==`:
v2 и Compose различаются на **5,68×10⁻¹⁴ МВт** при одинаковых image ID и хеше
модели. Исходный файл не исправлен. [Повторная проверка](reports/verification.json)
использует исходный допуск `rtol=1e-10, atol=1e-9` и подтверждает результаты;
разбор приведён в разделе 13 отчёта. Компаратор исправлен для будущих сборов.

## Проверка отчёта без обучения и Docker

При наличии Python 3.12, из корня репозитория:

```bash
python3.12 -m venv .report-venv
.report-venv/bin/python -m pip install -r requirements.lock.txt -r requirements-report.txt
.report-venv/bin/python scripts/validate_artifacts.py
.report-venv/bin/python scripts/run_notebook.py
```

Notebook заново читает сохранённые результаты и проверяет их. Он **не запускает
обучение или контейнеры**, не выдаёт новое выполнение за исходный опыт на Ubuntu.
Для регенерации структуры notebook: `python scripts/build_report.py`, затем
`python scripts/run_notebook.py`. После изменения методики/результатов пересмотрите
текстовые выводы в генераторе. Исходные логи и evidence_status не перезаписываются.

На Ubuntu 22.04 без отдельного Python 3.12 можно подготовить его в чистом рабочем
каталоге по следующему разделу, затем использовать его `.venv/bin/python` для
команд проверки отчёта. Системный Python заменять не требуется.

## Полное воспроизведение обучения и сервиса

**В этом репозитории `reports/` — уже полученные доказательства.** Для нового опыта
сначала создайте отдельный рабочий каталог, чтобы сохранить опубликованные результаты:

```bash
bash scripts/create_run.sh "$HOME/AIMLLab4-new-run"
cd "$HOME/AIMLLab4-new-run"
bash scripts/install_docker_ubuntu.sh
bash scripts/install.sh
bash scripts/run.sh experiments
sudo -v
bash scripts/run.sh v1
bash scripts/run.sh v2
bash scripts/run.sh rollback
bash scripts/run.sh compose
# После сохранения новых скриншотов:
bash scripts/run.sh collect
```

`create_run.sh` откажется перезаписывать существующую папку. Он копирует исходный
архив, код и зависимости, но не старые Registry/run_id, отчёты, экспорты и процессы.
Python 3.12.14 на Ubuntu 22.04 установится отдельно через uv; для Docker нужны права
пользователя или sudo. Новые run_id и номера версий могут отличаться; для оценок
float используется численный допуск, побитовая идентичность между архитектурами
не заявляется.

Исходная база MLflow и бинарные joblib **не входят** в архив результатов для GitHub.
Их остановленная резервная копия хранится отдельно на Ubuntu. В репозитории есть
все исходные данные и команды для нового обучения; наличие JSON Registry не
восстанавливает сервер MLflow само по себе.

Источник: [UCI CCPP](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant),
[DOI 10.24432/C5002N](https://doi.org/10.24432/C5002N). Полный список источников — в отчёте.

## Пересборка PDF с ответами

Для PDF нужны ReportLab и шрифты DejaVu (пакет Ubuntu `fonts-dejavu-core`).
Генерация не запускает обучение и не изменяет исходные результаты:

```bash
.venv/bin/python -m pip install -r requirements-defense.txt
.venv/bin/python scripts/build_defense_pdf.py
```

PDF содержит встроенные кириллические шрифты, оглавление с переходами и закладки.
Длинные команды удобнее копировать из [answers_lab4.md](answers_lab4.md).
