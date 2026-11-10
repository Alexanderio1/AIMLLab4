"""Build a report notebook from the submitted CCPP artifacts, without retraining."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text, explanation, images=()):
    cell = nbf.v4.new_code_cell(text.strip())
    if images:
        cell.metadata['report_images'] = list(images)
    cells.append(cell)
    md(explanation)


md(r'''
# Лабораторная работа № 4
## Основы MLOps: эксперименты, реестр моделей и контейнеризация

**Вариант 17 — прогноз выработки электростанции**<br>
**Набор данных:** UCI Combined Cycle Power Plant (CCPP).<br>
**Задача:** `AT, V, AP, RH → PE`, регрессия.<br>
**Основные алгоритмы:** Ridge и RandomForestRegressor. **Критерий выбора:** MAE на validation.<br>
**Дата выполнения на Ubuntu:** 10 ноября 2026 года, UTC+07:00. **Год:** 2026.

Отчёт оформлен по структуре первой лабораторной: выполненные кодовые ячейки,
таблицы, пояснения после вычислений, рисунки и полный исходный код в приложении.
Измерения обучения и Docker получены в Ubuntu 22.04.5 ARM64. Этот notebook
загружает и проверяет сохранённые результаты; его выполнение не запускает
повторное обучение, контейнеры или новый MLflow. Скриншоты оставлены без изменений.
''')
md(r'''
### Содержание

1. Цель и постановка задачи.
2. Набор данных и источник.
3. Подготовка и разбиение.
4. Программное окружение и структура.
5. Последовательность выполнения.
6. Сравнение шести экспериментов.
7. Выбор модели и итоговая оценка.
8. Model Registry и champion.
9. Экспорт и контракт API.
10. Сборка Docker и проверка предсказаний.
11. Обновление и откат.
12. Docker Compose и healthcheck.
13. Воспроизводимость и проверка артефактов.
14. Выводы.
15. Источники.
16. Приложение: программный код и примеры запросов.

### 1. Цель и постановка задачи

Цель — освоить учёт и сравнение экспериментов в MLflow, регистрацию версий
моделей, доставку сервиса предсказаний через Docker и Compose, проверку API,
обновление и откат без переобучения.

Для варианта 17 требуется прогнозировать PE по параметрам среды. Нужно:
выбрать источник и метрики; подготовить train/validation/test; выполнить не менее
шести экспериментов; сохранить параметры, метрики, теги и артефакты;
зарегистрировать минимум две версии и назначить champion; экспортировать Pipeline;
создать API и Dockerfile; проверить корректные/некорректные запросы и численное
совпадение; продемонстрировать обновление, откат и Compose healthcheck;
зафиксировать исходные данные, зависимости и разбиение.

Особенности варианта: используется одна таблица, повторные листы не объединяются;
единицы измерения входят в описание, теги/артефакты MLflow и метаданные экспорта.

MLflow Tracking группирует запуски в experiment и хранит параметры/метрики;
Model Registry даёт модели имя и версии. Alias — изменяемая ссылка на версию.
Docker-образ содержит приложение и конкретный снимок модели, а контейнер является
запущенным экземпляром образа. Смена alias сама по себе не обновляет контейнер.
''')
code('''
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = Path.cwd()
assert (ROOT / "reports/experiments.json").is_file(), "Run this notebook from the repository root"

def read_json(path):
    return json.loads((ROOT / path).read_text())

def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\\\|").replace("\\n", " ")
    text = "| " + " | ".join(map(cell, headers)) + " |\\n"
    text += "|" + "|".join("---" for _ in headers) + "|\\n"
    text += "\\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    display(Markdown(text))

def figure(path, caption):
    display(Image(filename=str(ROOT / path), width=1200))
    display(Markdown("**" + caption + "**"))

audit = read_json("reports/data_audit.json")
registry = read_json("reports/registry.json")
runs = read_json("reports/experiments.json")
test = read_json("reports/test_metrics.json")
environment = read_json("reports/environment.json")
print("Сохранённых экспериментов:", len(runs))
print("Registry:", registry["model_name"], "; выбранная версия:", registry["versions"]["selected"]["model_version"])
''', '''Инициализация читает результаты, полученные на Ubuntu, и определяет общие
функции вывода таблиц и рисунков. Путь notebook не используется как доказательство
платформы обучения: сведения о ней берутся из `reports/environment.json` и скриншотов.''')
md(r'''
### 2. Набор данных и источник

Используется [Combined Cycle Power Plant, UCI, ID 294](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant),
DOI: [10.24432/C5002N](https://doi.org/10.24432/C5002N).
Прямой архив: `https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip`.

По `CCPP/Readme.txt` исходного архива наблюдения собраны в 2006–2011 годах
на электростанции комбинированного цикла при полной нагрузке. Газовая и паровая
турбины работают в едином цикле. Признаки — средние почасовые параметры;
выходная переменная отражает электрическую мощность в MW. Хотя в названии
задания употреблено слово «выработка», **PE в этой работе измеряется в МВт,
а не в МВт·ч**. В workbook цель названа `PE`; в исходном Readme встречается `EP`.

Источник содержит XLSX/ODS с пятью перемешиваниями одной выборки для 5×2 CV.
В работе берётся только `Sheet1` файла `Folds5x2_pp.xlsx`: 9568 строк, четыре
числовых признака и одна цель. Повторные листы не увеличивают размер датасета.
''')
code('''
raw = pd.read_excel(ROOT / "data/raw/Folds5x2_pp.xlsx", sheet_name="Sheet1")
clean = pd.read_csv(ROOT / "data/processed/data.csv")
variables = [
    ("AT", "Температура окружающей среды", "°C"),
    ("V", "Вакуум на выхлопе паровой турбины", "cm Hg, см рт. ст."),
    ("AP", "Атмосферное давление", "mbar, мбар"),
    ("RH", "Относительная влажность", "%"),
    ("PE", "Целевая электрическая мощность", "MW, МВт"),
]
table(["Переменная", "Назначение", "Единицы", "Минимум", "Максимум"],
      [[name, description, unit, f"{raw[name].min():.2f}", f"{raw[name].max():.2f}"] for name, description, unit in variables])
print("Исходная форма:", raw.shape, "; пропусков:", int(raw.isna().sum().sum()))
''', '''Все пять столбцов числовые, категориальное кодирование не требуется.
Диапазоны относятся к исходной таблице; они используются для описания и контроля
источника, а не для подбора модели по test. Диапазон RH достигает 100,16%:
измерения немного выше теоретических 100% документируются, сохраняются без
обрезки и не удаляются как ошибочные автоматически.''')
code('''
excel = pd.ExcelFile(ROOT / "data/raw/Folds5x2_pp.xlsx")
reference_hashes = np.sort(pd.util.hash_pandas_object(raw, index=False).to_numpy())
sheet_rows = []
for name in excel.sheet_names:
    frame = pd.read_excel(excel, sheet_name=name)
    same = np.array_equal(np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy()), reference_hashes)
    sheet_rows.append([name, len(frame), "да" if same else "нет", "да" if name == "Sheet1" else "нет"])
table(["Лист", "Строк", "То же мультимножество строк, что Sheet1", "Используется для обучения"], sheet_rows)
''', '''Проверка сравнивает строки с учётом их кратности и без учёта порядка:
все пять листов являются перестановками одной таблицы. Их объединение дало бы
47 840 строк с повторным представлением наблюдений. Другие листы прочитаны
здесь только для аудита повторов, не объединяются и не участвуют в обучении.''')
md(r'''
### 3. Подготовка данных и корректное разбиение

1. Проверены колонки, 9568 исходных строк, отсутствие пропусков и бесконечных
значений, положительность V/AP и измеренный диапазон RH.
2. Удалён **41 полный дубликат**; осталось **9527** строк. Исходный номер строки
сохранён как `row_id`; этот служебный столбец не входит в признаки.
3. **55 исходных измерений RH выше 100%** сохранены. В раннем установщике граница
100% ошибочно отклоняла источник; проверка исправлена до обучения, а источник
не изменялся. Фактический предел датасета — 100,16%.
4. `GroupShuffleSplit`, `random_state=42`: сначала test ≈20%, затем validation
≈25% оставшейся development-части. Итог — около 60/20/20.
5. Одинаковые векторы `AT,V,AP,RH` относятся к одной группе, даже при разных PE.
После дедупликации дополнительных повторов X в этом наборе нет.

Параметры StandardScaler для Ridge обучаются только на train в составе Pipeline.
Для RF используется `passthrough`. Validation выбирает конфигурацию; test
используется лишь для итоговой оценки победителя. После выбора переобучения
на train+validation нет: оценивается тот же сохранённый артефакт.
''')
code('''
split = read_json("data/split.json")
table(["Часть", "Строк", "Доля, %", "Назначение"], [
    ["train", len(split["train"]), f"{len(split['train']) / len(clean) * 100:.2f}", "Обучение Pipeline"],
    ["validation", len(split["validation"]), f"{len(split['validation']) / len(clean) * 100:.2f}", "Выбор по MAE"],
    ["test", len(split["test"]), f"{len(split['test']) / len(clean) * 100:.2f}", "Итоговая оценка выбранной модели"],
])
figure("screenshots/05-data.png", "Рисунок 1 — аудит данных: один лист, 41 дубликат, RH и разбиение.")
''', '''Получены **5715 / 1906 / 1906** наблюдений. Индексы зафиксированы в
`data/split.json`, сами части — в CSV. Автоматическая проверка в разделе 13
повторяет split с seed=42 и проверяет отсутствие общих row_id и векторов X.
В датасете нет доступных временных меток: такое разбиение оценивает прогноз по
параметрам среды в рамках данного набора, а не качество прогноза будущих лет.''', ['screenshots/05-data.png'])
md(r'''
### 4. Программное окружение и структура проекта

Выполнение проведено в виртуальной машине **Ubuntu 22.04.5 LTS, aarch64**
на Mac M1. В отличие от эталонной Ubuntu 24.04 из методички, системный Python
здесь 3.10.12. Для проекта отдельно установлен **CPython 3.12.14** через
`uv==0.13.5`; системный интерпретатор сохранён. Ядро Linux —
`5.15.0-203-generic`, glibc 2.35. Docker Engine — **29.9.3**, Compose — **v5.7.0**
(фактически записанные версии, не номера из облачной проверки стартового комплекта).
''')
code('''
table(["Компонент", "Версия в Ubuntu"], [[name, version] for name, version in environment["packages"].items()])
figure("screenshots/01-system.png", "Рисунок 2 — Ubuntu 22.04.5, ARM64 и исходный системный Python 3.10.12.")
figure("screenshots/02-docker.png", "Рисунок 3 — Docker Engine 29.9.3 и Docker Compose v5.7.0 на ARM64.")
figure("screenshots/03-install.png", "Рисунок 4 — установка зависимостей, pip check и 14 исходных тестов.")
figure("screenshots/04-python.png", "Рисунок 5 — отдельный Python 3.12.14 проекта и сохранённый Python 3.10.12.")
''', '''Скриншоты показывают исходную установку, поэтому на них **14 passed**.
После исправления обработки RH были добавлены три регрессионные проверки
(17 тестов); при подготовке отчёта добавлены проверки численного сравнения
ответов сборщика, и текущая тестовая версия содержит 25 тестов. Эти более
поздние проверки не выдаются за первоначальный запуск на Ubuntu.
`requirements.lock.txt` фиксирует 139 пакетов; отдельный runtime lock содержит
22 зависимости сервиса. `reports/installed-requirements.txt` сохраняет снимок Ubuntu.''', ['screenshots/01-system.png', 'screenshots/02-docker.png', 'screenshots/03-install.png', 'screenshots/04-python.png'])
md(r'''
```text
AIMLLab4/
  lab4.ipynb, report.md, report.html   # отчёт и версии для просмотра
  requirements*.txt                  # прямые зависимости и lock-файлы
  Dockerfile, compose.yaml, .dockerignore
  src/workflow.py                    # данные, 6 run, Registry, test, export, check
  src/app.py                         # FastAPI, схема входа, загрузка модели
  src/evidence.py                    # численное сравнение ответов сборщика
  scripts/                          # установка, MLflow, этапы и проверки
  tests/                            # тесты API, разбиения, данных, сравнения
  data/raw/                         # исходные ZIP и XLSX
  data/processed/, data/split.json   # очищенная таблица и train/validation/test
  exported_models/v1/, v2/           # metadata, запросы и эталонные прогнозы
  reports/                          # метрики, run_id, JSON-проверки и логи
  screenshots/                      # 14 оригинальных снимков Ubuntu
```

При выполнении также создаются `.venv`, `.tools`, `mlflow-data` и бинарные
`model.joblib`. Они не включены в Git. Переданный архив результатов содержит
метаданные моделей, но не сами joblib и не SQLite MLflow. Их резервная копия
`AIMLLab4-mlflow-backup.tar.gz` сохранена отдельно у исполнителя.

### 5. Последовательность выполнения с командами

Команды ниже описывают **выполненный в Ubuntu** процесс; notebook не запускает их.
Терминал находился в `~/lab4-ubuntu22/AIMLLab4`.

```bash
sudo apt update
sudo apt install -y curl unzip git ca-certificates python3-venv python3-pip gnome-screenshot
bash scripts/install_docker_ubuntu.sh
bash scripts/install.sh
python3 --version
.venv/bin/python --version
bash scripts/run.sh experiments
```

Стартовый комплект распакован с GitHub. MLflow запущен скриптом на
`127.0.0.1:5000`; SQLite хранит метаданные, `mlflow-data/artifacts` — файлы моделей.
Проверка `/health` подтверждает готовность перед логированием.
После исправления диапазона RH команда experiments использовала уже скачанный
исходный архив. Ошибка подготовки не была результатом обучения и не добавляла
успешных run в таблицу.

Далее последовательно выполнены:

```bash
sudo -v
bash scripts/run.sh v1
bash scripts/run.sh v2
bash scripts/run.sh rollback
bash scripts/run.sh compose
# После сохранения скриншотов:
bash scripts/run.sh collect
```

Каждый этап пишет `reports/logs/<этап>.log`. Образ v1 построен для Ridge,
v2 — для выбранного RF; откат использует уже построенный v1. В конце Compose
и MLflow остановлены сборщиком перед резервным копированием. Сборка/запуск,
проверки и результаты подробно разобраны ниже.

### 6. Сравнение шести экспериментов

Experiment: **`lab4-ccpp-variant17`**. Все модели используют один train и один
validation. Ridge проверяется при `alpha=0.1, 1, 10`; RF — с разным количеством
деревьев, глубиной и минимальным числом наблюдений в листе. RF имеет
`random_state=42`, `n_jobs=2`; Ridge использует StandardScaler.

Используются следующие метрики (n — число наблюдений, yᵢ — фактическая
мощность, ŷᵢ — прогноз, ȳ — среднее фактических значений):

- **MAE = (1/n) Σᵢ |yᵢ − ŷᵢ|**, МВт — основной критерий, меньше лучше.
- **RMSE = √[(1/n) Σᵢ (yᵢ − ŷᵢ)²]**, МВт — сильнее реагирует на большие ошибки.
- **R² = 1 − [Σᵢ (yᵢ − ŷᵢ)²] / [Σᵢ (yᵢ − ȳ)²]**, безразмерная — больше лучше.

R² не является accuracy. Формулы записаны текстом и читаются без внешнего MathJax.

Параметры и размеры выборок логируются как params; `val_mae`, `val_rmse`,
`val_r2`, `fit_seconds` — как metrics. Теги включают вариант 17, цель PE,
`selection_metric=val_mae`, направление minimize, единицы и SHA-256 данных/split.
Артефакты — Pipeline с signature и input_example, график validation, `units.json`,
аудит, split, код и зависимости. Dataset input логируется в контексте training.
''')
code('''
ordered = sorted(runs, key=lambda row: (row["val_mae"], row["name"]))
table(["Запуск / алгоритм", "Гиперпараметры", "MAE, МВт", "RMSE, МВт", "R²", "fit, с", "run_id"], [
    [r["name"] + " / " + r["algorithm"], json.dumps(r["parameters"], ensure_ascii=False),
     f"{r['val_mae']:.6f}", f"{r['val_rmse']:.6f}", f"{r['val_r2']:.6f}", f"{r['fit_seconds']:.6f}", "`" + r["run_id"] + "`"]
    for r in ordered
])
figure("screenshots/07-mlflow-runs.png", "Рисунок 6 — шесть успешных run и метрики validation в MLflow.")
figure("reports/experiment_comparison.png", "Рисунок 7 — сравнение validation MAE, меньшая ошибка предпочтительна.")
''', '''Лучший результат — **rf-unlimited**, MAE **2,472198 МВт**.
Ridge почти не меняет ошибку при изменении alpha; глубина 8 ограничивает качество
RF. `fit_seconds` — время только `fit`, а Duration в MLflow дополнительно включает
логирование и сериализацию. Поэтому 48,5 с на скриншоте run не равны 1,8502 с
обучения из таблицы. Значения получены на конкретной виртуальной машине и не
являются сравнением производительности на других компьютерах.''', ['screenshots/07-mlflow-runs.png', 'reports/experiment_comparison.png'])
md(r'''
### 7. Обоснование выбора и итоговая оценка

Выбран **RandomForestRegressor, 250 деревьев, max_depth=None,
min_samples_leaf=1**. Validation MAE ниже лучшего Ridge примерно на **32,16%**.
Разница с RF depth16 небольшая — **0,008920 МВт** (около 0,36%); по заранее
зафиксированному правилу минимума MAE победителем становится rf-unlimited.
Статистическая значимость такого малого отрыва не исследовалась.

Test не используется для выбора алгоритма/гиперпараметров и не определяет alias.
Итоговая оценка проведена для сохранённой версии 2, без переобучения. Test-метрики
дописаны в исходный run победителя `503b388935b047bca0fa775e9603e454`.
''')
code('''
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
y, yhat = predictions["actual_PE_MW"], predictions["predicted_PE_MW"]
recomputed = {
    "test_mae": float(mean_absolute_error(y, yhat)),
    "test_rmse": float(np.sqrt(mean_squared_error(y, yhat))),
    "test_r2": float(r2_score(y, yhat)),
}
for key in recomputed:
    assert np.isclose(recomputed[key], test[key], rtol=0, atol=1e-12)
table(["Объект", "Значение"], [
    ["Registry version", test["model_version"]], ["Run ID", "`" + test["run_id"] + "`"],
    ["Test, строк", test["n_test"]], ["MAE, МВт", f"{test['test_mae']:.9f}"],
    ["RMSE, МВт", f"{test['test_rmse']:.9f}"], ["R²", f"{test['test_r2']:.9f}"],
])
figure("screenshots/06-test.png", "Рисунок 8 — итоговые test-метрики в терминале Ubuntu.")
figure("reports/test_residuals.png", "Рисунок 9 — распределение остатков выбранной модели на test.")
''', '''Повторный расчёт по сохранённым 1906 парам «истина–прогноз» подтверждает
**MAE 2,490805 МВт, RMSE 3,601666 МВт, R² 0,956215**. Test MAE близка к validation
MAE 2,472198 МВт. Превышение RMSE над MAE указывает на влияние более крупных
ошибок; их наличие видно на гистограмме. R² характеризует долю объяснённой
вариации на данном test и не означает «95,6% точных ответов».''', ['screenshots/06-test.png', 'reports/test_residuals.png'])
md(r'''
### 8. Регистрация версий и alias champion

В Model Registry создана одна модель **`ccpp-power`**. Версия 1 — лучший запуск
другого алгоритма, Ridge; версия 2 — выбранный RF. Теги версий содержат роль
baseline/selected и единицу цели MW. Каждая версия связана с исходным run и
URI логированной модели MLflow 3.
''')
code('''
table(["Роль", "Registry version", "Запуск", "run_id", "model_uri"], [
    [role, item["model_version"], item["name"], "`" + item["run_id"] + "`", "`" + item["model_uri"] + "`"]
    for role, item in registry["versions"].items()
])
figure("screenshots/08-registry.png", "Рисунок 10 — две версии ccpp-power; champion назначен версии 2.")
''', '''На скриншоте после экспериментов alias **champion → version 2**.
При демонстрации v1 он временно назначается версии 1, при обновлении — 2,
при откате — 1, после Compose — снова 2. Последнее назначение также зафиксировано
строкой `champion -> 2` в `reports/logs/compose.log`.
Run ID, номер Registry, тег Docker и image ID — разные идентификаторы.''', ['screenshots/08-registry.png'])
md(r'''
### 9. Экспорт модели и контракт API

Экспорт читает `models:/ccpp-power@champion`, сохраняет sklearn Pipeline
в `model.joblib`, вычисляет SHA-256 и создаёт `metadata.json`, `sample_request.json`,
`expected_predictions.json`. В v1 StandardScaler поставляется вместе с Ridge;
v2 содержит Pipeline с passthrough и RF. Контейнер использует конкретный экспорт,
не обращаясь к MLflow во время предсказания. Смена alias требует нового экспорта
и сборки для доставки нового снимка.

FastAPI загружает модель при старте и проверяет хеш. При повреждении файла
приложение не проходит запуск. `/health` возвращает готовность загруженной модели,
`/metadata` — её версию, run_id, признаки, единицы и хеш, `/docs` — Swagger UI.

| Метод и путь | Назначение |
|---|---|
| GET `/health` | status и model_version |
| GET `/metadata` | Метаданные конкретного экспорта |
| POST `/predict` | Пакет предсказаний в MW |
| GET `/docs` | Интерактивная документация |

Вход — JSON с единственным верхним ключом `instances`, от 1 до 1000 объектов.
В каждом ровно `AT`, `V`, `AP`, `RH`; конечные JSON-числа, не числовые строки и
не bool. V и AP > 0, RH от 0 до 100,16% с учётом измерений источника.
Лишние/пропущенные поля запрещены. Эти проверки не являются полноценным
контролем сдвига распределения: например, диапазон AT не ограничен наблюдённым
минимумом/максимумом.

Пример фактически выполненного вручную запроса из Swagger:

```json
{"instances": [{"AT": 20.0, "V": 50.0, "AP": 1010.0, "RH": 70.0}]}
```

Ответ HTTP 200:

```json
{"model_version": "2", "predictions_MW": [454.9589200000001], "unit": "MW"}
```

Это другая входная строка, чем пакет из трёх validation-наблюдений в
автоматическом smoke-test. Их прогнозы не должны совпадать между собой.
''')
code('''
api = {phase: read_json(f"reports/api_{phase}.json") for phase in ["v1", "v2", "rollback", "compose"]}
table(["Экспорт", "Registry version", "SHA-256 model.joblib (из метаданных)"], [
    [label, read_json(f"exported_models/{label}/metadata.json")["model_version"], "`" + read_json(f"exported_models/{label}/metadata.json")["model_sha256"] + "`"]
    for label in ["v1", "v2"]
])
case_labels = {
    "empty_batch": "Пустой instances", "missing_feature": "Нет обязательного признака",
    "numeric_string": "Число передано строкой", "extra_feature": "Лишнее поле PE",
    "invalid_humidity": "RH=101, вне контракта", "wrong_body": "data вместо instances",
    "too_many_rows": "Больше 1000 строк", "nonfinite_input": "NaN в AT",
}
table(["Проверка", "Ожидается", "Факт v1", "Факт v2", "Факт откат", "Факт Compose"], [
    [case_labels[item["case"]], 422] + [next(c["actual_status"] for c in api[phase]["invalid_requests"] if c["case"] == item["case"]) for phase in api]
    for item in api["v1"]["invalid_requests"]
])
figure("screenshots/13-api-correct.png", "Рисунок 11 — корректный ручной запрос: HTTP 200, версия 2, прогноз 454,95892 МВт.")
figure("screenshots/14-api-invalid.png", "Рисунок 12 — без RH: HTTP 422, Field required с указанием отсутствующего поля.")
''', '''Все восемь автоматических отрицательных сценариев получили 422 на каждом
из четырёх этапов. Отдельный ручной запрос без RH также отклонён с указанием
поля. NaN технически не входит в стандарт JSON; поскольку некоторые парсеры его
допускают, приложение отдельно запрещает нечисловые/бесконечные значения и
возвращает корректный JSON ошибки вместо 500.''', ['screenshots/13-api-correct.png', 'screenshots/14-api-invalid.png'])
md(r'''
### 10. Dockerfile, сборка и совпадение предсказаний

Базовый образ — мультиархитектурный `python:3.12-slim`; при первом запуске
сохранён его RepoDigest. Обе сборки используют этот digest, а не независимо
разрешают изменяемый тег. Итоговые образы имеют архитектуру **linux/arm64**.
Runtime-зависимости фиксируются `requirements-runtime.lock.txt`; образ содержит
`src/app.py` и один конкретный экспорт. Приложение работает от `appuser`,
Uvicorn слушает `0.0.0.0:8000` внутри контейнера. Публикация
`127.0.0.1:8000:8000` ограничивает доступ интерфейсом loopback хоста.

Скрипт `bash scripts/run.sh v1` выполняет эквивалент последовательности:

```bash
.venv/bin/python src/workflow.py export --role baseline --label v1
docker build --build-arg BASE_IMAGE="$(cat reports/base-image.txt)" \
  --build-arg MODEL_PATH=exported_models/v1 -t ccpp-api:v1 .
docker run -d --name ccpp-lab17-api --label lab=ccpp-17 \
  -p 127.0.0.1:8000:8000 ccpp-api:v1
.venv/bin/python src/workflow.py check --label v1 --phase v1
```

При необходимости Docker вызывается через sudo самим скриптом. BuildKit secret
`proxy_ca` опционален для сред с собственным доверенным CA; в Ubuntu проверка TLS
не отключалась. `.dockerignore` исключает данные, MLflow, .venv и прочие ненужные
файлы из контекста, оставляя runtime lock, app и экспорт.
''')
code('''
print("Зафиксированный базовый образ:")
print((ROOT / "reports/base-image.txt").read_text().strip())
containers = {phase: read_json(f"reports/container_{phase}.json") for phase in api}
table(["Этап", "Registry version", "Docker tag", "Архитектура", "Image ID"], [
    [phase, api[phase]["model_version"], containers[phase]["image_tag"], "linux/arm64", "`" + containers[phase]["image_id"] + "`"]
    for phase in api
])
table(["Этап", "API против локального экспорта: max |Δ|, МВт", "Допуск"], [
    [phase, f"{api[phase]['max_abs_difference_MW']:.6g}", "rtol=1e-10, atol=1e-9 МВт"] for phase in api
])
figure("screenshots/09-v1.png", "Рисунок 13 — сборка v1, HTTP 200, отрицательные запросы и Docker healthy.")
''', '''Проверка в Ubuntu загрузила экспортированный Pipeline локально и
сравнила его прогнозы с API через `np.testing.assert_allclose`. У v1, v2 и
отката сохранённое максимальное отклонение равно 0; у Compose —
**5,68434×10⁻¹⁴ МВт**, что значительно меньше заданного допуска.
Один PID или открытый порт не использовались как достаточное подтверждение:
выполнялись запросы, проверка версии/метаданных, сравнение прогнозов и healthcheck.''', ['screenshots/09-v1.png'])
md(r'''
### 11. Обновление и откат без повторного обучения

Порядок версий сервиса: **v1 / Registry 1 → v2 / Registry 2 → откат v1 / Registry 1**.
Команда `v2` экспортирует уже обученную выбранную версию, собирает новый образ
и заменяет контейнер. Команда `rollback` только возвращает champion на 1 и
запускает первоначальный образ `ccpp-api:v1`; в её коде нет обучения,
экспорта или сборки.

```bash
bash scripts/run.sh v2
bash scripts/run.sh rollback
```

Для одного и того же входного пакета записаны следующие прогнозы:
''')
code('''
table(["Этап", "Версия", "PE₁, МВт", "PE₂, МВт", "PE₃, МВт"], [
    [phase, api[phase]["model_version"]] + [f"{value:.9f}" for value in api[phase]["valid_response"]["predictions_MW"]]
    for phase in api
])
assert containers["v1"]["image_id"] == containers["rollback"]["image_id"]
assert api["v1"]["model_sha256"] == api["rollback"]["model_sha256"]
np.testing.assert_allclose(api["v1"]["valid_response"]["predictions_MW"], api["rollback"]["valid_response"]["predictions_MW"], rtol=1e-10, atol=1e-9)
print("Откат подтверждён: тот же image ID, хеш модели и прогнозы.")
figure("screenshots/10-v2.png", "Рисунок 14 — обновление до v2: другая версия, другой образ и изменённые прогнозы.")
figure("screenshots/11-rollback.png", "Рисунок 15 — возврат к исходному v1 без обучения и сборки, healthy.")
''', '''В v1 первый прогноз **463,955885 МВт**, в v2 — **465,653520 МВт**;
после отката снова **463,955885 МВт**. У контейнеров до и после отката разные
container ID, поскольку это новые экземпляры, но один и тот же image ID
`sha256:5ad83f412ed49351b1c4fc9e0c0e137041782b93e925b4b6b2bc09fdaa95bbbd`.
Это подтверждает повторное использование исходного образа, а не переобучение
модели под прежним именем. Теги v1/v2 различны; image ID записаны полностью,
так как один и тот же тег технически можно перезаписать.''', ['screenshots/10-v2.png', 'screenshots/11-rollback.png'])
md(r'''
### 12. Docker Compose и healthcheck

`compose.yaml` запускает уже собранный `ccpp-api:v2`, без повторной сборки.
`pull_policy: never` использует локальный образ; порт — `127.0.0.1:8000:8000`,
`restart: unless-stopped`. Проверка здоровья обращается к `/health` через
стандартный `urllib.request` Python: interval=5s, timeout=3s, start_period=20s,
retries=12. Сначала сервер загружает и проверяет модель, затем отвечает healthy.

```bash
bash scripts/run.sh compose
# Внутри этапа:
docker compose -p ccpp-lab17 up -d --wait --wait-timeout 120
docker compose -p ccpp-lab17 ps
```

Одиночный контейнер лабораторной освобождает порт перед Compose. Alias снова
назначается версии 2. После `--wait` отдельно проверяются API и `.State.Health`.
''')
code('''
table(["Этап", "Статус Docker healthcheck", "Проверка API"], [
    [phase, read_json(f"reports/health_{phase}.json")["Status"], api[phase]["all_checks_passed"]] for phase in api
])
assert containers["compose"]["image_id"] == containers["v2"]["image_id"]
print("Compose использует тот же образ v2:", containers["compose"]["image_id"])
figure("screenshots/12-compose.png", "Рисунок 16 — Compose up --wait, сервис healthy и успешные API-проверки.")
''', '''Все четыре этапа имеют статус **healthy**; Compose запустил ту же
версию модели и тот же image ID, что отдельный контейнер v2. HTTP-проверка
включает реальное предсказание и отрицательные сценарии. После сохранения
доказательств `collect` останавливает процессы и создаёт резервную копию MLflow;
наличие скриншота не означает, что сервис продолжает работать после остановки.''', ['screenshots/12-compose.png'])
md(r'''
### 13. Воспроизводимость и независимая проверка артефактов

В Git включены исходные ZIP/XLSX, очищенная таблица, три части CSV, индексы,
seed=42, прямые зависимости и оба lock-файла, исходники и исходные логи.
Исходный архив результатов имеет SHA-256 в `reports/submission_manifest.json`;
там же зафиксированы хеши всех 94 полученных файлов. Оригинальные данные,
метрики, API-ответы и скриншоты не исправлялись при оформлении отчёта.

Для нового воспроизведения установки и обучения используйте новую копию проекта:
обучение создаст другие run_id и может дать последние биты float иначе на другой
архитектуре. Публикация отчёта сама по себе не восстанавливает исходный Registry:
его SQLite и бинарные модели остались в отдельной резервной копии Ubuntu.

Для просмотра и проверки уже полученных результатов обучение не нужно:

```bash
.venv/bin/python scripts/validate_artifacts.py
.venv/bin/python -m pip install -r requirements-report.txt
.venv/bin/python scripts/run_notebook.py
```

**Разбор одного отрицательного флага сборщика.** В оригинальном
`reports/evidence_status.json` сохранены `update_verified: false` и
`all_checks_passed: false`. Причина — сравнение словарей ответов через точное `==`.
В v2 первая оценка равна `465.65352000000087`, в Compose —
`465.6535200000009`: разница **5,684341886080802×10⁻¹⁴ МВт**, один шаг float64.
У многопоточного RF порядок суммирования может менять последние биты результата.

Отдельные функциональные проверки на Ubuntu уже использовали `allclose` и прошли.
Повторная проверка ниже также проверяет image ID, номер/хеш модели, входной пакет
и предсказания с исходным допуском. Исходный отрицательный флаг **не заменён на true**;
в `reports/verification.json` сохранён отдельный вывод с объяснением. Для будущих
сборов исправлен только численный компаратор сборщика, его исходный вариант
сохранён как `reports/submitted_collect.py.before`.
''')
code('''
verification = runpy.run_path(str(ROOT / "scripts/validate_artifacts.py"))["validate"]()
assert verification["all_checks_passed"]
print("Проверок архива:", len(verification["checks"]))
print("Все проверки архива:", verification["all_checks_passed"])
print("Оригинальный update_verified:", verification["original_collector"]["checks"]["update_verified"])
print("v2 vs Compose: max |Δ|, МВт =", verification["v2_vs_compose_max_abs_difference_MW"])
table(["Файл", "SHA-256"], [
    [name, "`" + hashlib.sha256((ROOT / name).read_bytes()).hexdigest() + "`"]
    for name in ["data/raw/ccpp.zip", "data/raw/Folds5x2_pp.xlsx", "data/processed/data.csv", "data/split.json"]
])
''', '''Проверка заново пересчитала test-метрики, восстановила split с seed=42,
проверила отсутствие утечки и неизменность исходных файлов, согласованность
run_id/версий, восемь отрицательных сценариев каждого этапа, образы и healthcheck.
Для API используются сохранённые запросы/ответы: это проверка доказательств,
не повторное обращение к остановленному сервису. Хеши моделей сопоставляются
между метаданными; пересчитать их по бинарным файлам из этого ZIP невозможно,
поскольку joblib в него намеренно не включены.''')
md(r'''
### 14. Выводы

1. Для варианта 17 использован первый лист CCPP: **9568** исходных наблюдений,
**41** полный дубликат удалён, **9527** строк разделены на **5715/1906/1906**.
Повторные листы не объединялись; RH до 100,16% сохранена как измерение источника.
2. В MLflow проведены **шесть** запусков с параметрами, метриками, тегами,
единицами, графиками, split и Pipeline. По validation MAE выбран RF с 250
деревьями и неограниченной глубиной: **2,472198 МВт**, примерно на **32,16%**
лучше лучшего Ridge в этом сравнении.
3. На отложенном test выбранная модель получила **MAE 2,490805 МВт,
RMSE 3,601666 МВт, R² 0,956215**. Test не участвовал в выборе; значимость малого
отрыва от RF depth16 отдельно не оценивалась.
4. Созданы **две версии Registry**, назначен champion; Pipeline экспортирован
с преобразованиями и метаданными. FastAPI возвращает версии и MW, корректные
запросы проходят с 200, восемь классов ошибок — с 422.
5. Собраны два ARM64 Docker-образа. Сравнение API с локальным экспортом проходит
с заданным допуском. Продемонстрированы обновление **v1→v2** и откат **v2→v1**
без обучения: при откате image ID, хеш модели и прогнозы вернулись к исходным.
6. Compose запустил выбранный v2 с **healthy**, автоматические проверки API
прошли. Обнаруженный строгий компаратор сборщика объясняет ложный отрицательный
флаг: различие одного ULP не является сменой модели или ошибкой развёртывания.
7. Зафиксированы данные, split, seed, зависимости, исходники, логи и 14 снимков.
Результат ограничен данным режимом полной нагрузки и случайным разбиением без
временных меток; он не доказывает качество прогноза будущих режимов и не заменяет
мониторинг дрейфа, CI/CD или производственный контур эксплуатации.

### 15. Список использованных источников

1. **UCI Combined Cycle Power Plant.** [Страница набора](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant).
DOI: [10.24432/C5002N](https://doi.org/10.24432/C5002N). Описание и исходный архив.
2. **CCPP/Readme.txt** в исходном ZIP: годы, полная нагрузка, почасовые признаки
и назначение пяти перемешиваний. Архив включён в `data/raw/ccpp.zip`.
3. Tüfekci P. **Prediction of full load electrical power output of a base load
operated combined cycle power plant using machine learning methods.**
International Journal of Electrical Power & Energy Systems. 2014. Vol. 60.
P. 126–140. DOI: [10.1016/j.ijepes.2014.02.027](https://doi.org/10.1016/j.ijepes.2014.02.027).
4. MLflow: [Tracking](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/),
[Model Registry](https://mlflow.org/docs/latest/ml/model-registry/),
[sklearn API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).
5. scikit-learn: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html),
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
6. Docker: [Engine на Ubuntu](https://docs.docker.com/engine/install/ubuntu/),
[Dockerfile](https://docs.docker.com/reference/dockerfile/),
[Compose services / healthcheck](https://docs.docker.com/reference/compose-file/services/#healthcheck).
7. [FastAPI](https://fastapi.tiangolo.com/) и [Pydantic](https://docs.pydantic.dev/latest/).
8. Методические указания **«Лабораторная работа 4. MLOps»**, предоставленный PDF;
индивидуальный вариант 17.
9. [Репозиторий первой лабораторной](https://github.com/Alexanderio1/AIML) — образец
оформления notebook/HTML, а не источник метрик настоящей работы.

### 16. Приложение: исходный код, зависимости и примеры запросов

Полные исполняемые файлы размещены в `src/`, `scripts/`, `tests/`. Ниже приведён
листинг основной программы обучения/экспорта, API, оркестратора, MLflow-сервера,
Dockerfile, Compose и зависимостей. Полные lock-файлы и примеры запросов также
доступны отдельными файлами. Показанный `scripts/collect.py` содержит исправление
компаратора, внесённое после выполнения; первоначальный вариант сохранён отдельно.
''')
code('''
listing = [
    ("src/workflow.py", "python"), ("src/app.py", "python"),
    ("scripts/run.sh", "bash"), ("scripts/mlflow_server.py", "python"),
    ("scripts/collect.py", "python"), ("src/evidence.py", "python"),
    ("Dockerfile", "dockerfile"), ("compose.yaml", "yaml"),
    (".dockerignore", "text"), ("requirements.txt", "text"),
    ("requirements-runtime.lock.txt", "text"),
]
for name, language in listing:
    display(Markdown("<details>\\n<summary>" + name + " — полный листинг</summary>\\n\\n```" + language + "\\n" + (ROOT / name).read_text().rstrip() + "\\n```\\n\\n</details>"))
print("Фиксированные прямые и транзитивные зависимости: requirements.lock.txt, requirements-runtime.lock.txt")
print("Пример пакета для проверки:")
print((ROOT / "exported_models/v2/sample_request.json").read_text())
print("Эталонные предсказания этого пакета:")
print((ROOT / "exported_models/v2/expected_predictions.json").read_text())
''', '''Листинги читаются непосредственно из файлов репозитория, поэтому соответствуют
опубликованному коду. При следующем выполнении notebook выводы должны быть
пересмотрены, если изменяются данные, модели, split или протокол проверки.
В исходном MLflow run зафиксирован код обучения на момент запуска; последующее
исправление сборщика не меняет сохранённые модели и метрики.''')
md('''
Дополнительные файлы приложения:

- [requirements.lock.txt](requirements.lock.txt) — полный lock обучения.
- [requirements-runtime.lock.txt](requirements-runtime.lock.txt) — lock контейнера.
- [scripts/install.sh](scripts/install.sh), [bootstrap_python.sh](scripts/bootstrap_python.sh),
  [install_docker_ubuntu.sh](scripts/install_docker_ubuntu.sh) — установка на Ubuntu.
- [scripts/validate_artifacts.py](scripts/validate_artifacts.py) — проверка результатов.
- [tests/test_contract.py](tests/test_contract.py), [test_evidence.py](tests/test_evidence.py) — тесты.
- [exported_models/v2/sample_request.json](exported_models/v2/sample_request.json) — пакет запроса.
- [reports/experiments.csv](reports/experiments.csv), [test_metrics.json](reports/test_metrics.json),
  [registry.json](reports/registry.json) — исходные численные результаты.
- [reports/verification.json](reports/verification.json) — повторная проверка;
  [evidence_status.json](reports/evidence_status.json) — неизменённый исходный флаг.
- [reports/logs](reports/logs) — полные журналы выполненных этапов.
''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    'kernelspec': {'display_name': 'Python 3 (CCPP report)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
    'lab4': {'variant': 17, 'evidence': 'reports/submission_manifest.json', 'execution_scope': 'Read-only artifact report; original training and containers ran on Ubuntu ARM64'},
})
nbf.write(notebook, ROOT / 'lab4.ipynb')
print('Created', len(cells), 'cells in lab4.ipynb')
