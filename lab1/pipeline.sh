#!/usr/bin/env bash
# =============================================================================
# pipeline.sh  —  последовательный запуск ML-пайплайна
#
# Порядок шагов:
#   0. (авто) создание venv и установка зависимостей (pandas, scikit-learn)
#   1. data_creation.py      — извлечение датасета, разбиение на train/test
#   2. data_preprocessing.py — очистка выбросов, масштабирование, кодирование
#   3. model_preparation.py  — обучение и сохранение модели (model.pkl)
#   4. model_testing.py      — оценка модели на тестовых данных
#
# Запуск:  bash pipeline.sh
#          или ./pipeline.sh (после chmod +x pipeline.sh)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "ОШИБКА: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Директория скрипта
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
log "Рабочая директория: $SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Поиск Python с установленными pandas и scikit-learn
# ---------------------------------------------------------------------------
find_python() {
    local candidates=()

    # Системные интерпретаторы
    for c in python3 python python3.13 python3.12 python3.11 python3.10 python3.9 python3.8; do
        command -v "$c" &>/dev/null && candidates+=("$c")
    done

    # Windows Python через WSL (ищем по всем пользователям)
    if [[ -d "/mnt/c/Users" ]]; then
        while IFS= read -r -d '' udir; do
            for ver in Python313 Python312 Python311 Python310; do
                wp="$udir/AppData/Local/Programs/Python/$ver/python.exe"
                [[ -x "$wp" ]] && candidates+=("$wp")
            done
        done < <(find /mnt/c/Users -maxdepth 1 -mindepth 1 -type d -print0 2>/dev/null)
    fi
    for wp in /mnt/c/Python31*/python.exe; do
        [[ -x "$wp" ]] && candidates+=("$wp")
    done

    # Выбираем первый, у которого есть pandas и sklearn
    for c in "${candidates[@]}"; do
        if "$c" -c "import pandas, sklearn" &>/dev/null; then
            echo "$c"; return 0
        fi
    done

    # Ни у одного нет пакетов — берём первый доступный и ставим через pip
    for c in "${candidates[@]}"; do
        ver=$("$c" -c "import sys; print(sys.version_info >= (3,8))" 2>/dev/null || true)
        if [[ "$ver" == "True" ]]; then
            log "Устанавливаем зависимости в $c ..."
            "$c" -m pip install --quiet pandas scikit-learn \
                && { echo "$c"; return 0; }
        fi
    done

    return 1
}

PYTHON=$(find_python) || fail "Python 3.8+ с пакетами pandas/scikit-learn не найден."
log "Используется Python: $PYTHON ($($PYTHON --version 2>&1))"

# ---------------------------------------------------------------------------
# Функция запуска Python-скрипта
# ---------------------------------------------------------------------------
run_step() {
    local step_num="$1" script="$2" description="$3"
    log "──────────────────────────────────────────"
    log "Шаг $step_num: $description"
    [[ -f "$script" ]] || fail "Файл '$script' не найден."
    "$PYTHON" "$script"
    log "Шаг $step_num завершён успешно."
}

# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------
log "=========================================="
log " Запуск ML-пайплайна"
log "=========================================="
START_TIME=$(date +%s)

run_step 1 "data_creation.py"      "Создание и разбиение датасета"
run_step 2 "data_preprocessing.py" "Предобработка данных"
run_step 3 "model_preparation.py"  "Обучение модели"

log "──────────────────────────────────────────"
log "Шаг 4: Тестирование модели"
[[ -f "model_testing.py" ]] || fail "Файл 'model_testing.py' не найден."
METRIC_OUTPUT=$("$PYTHON" model_testing.py)
log "Шаг 4 завершён успешно."

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

log "=========================================="
log " Пайплайн завершён за ${ELAPSED}с"
log "=========================================="

# Итоговая строка с метрикой в stdout
echo "$METRIC_OUTPUT"
