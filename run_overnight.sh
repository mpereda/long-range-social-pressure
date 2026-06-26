#!/bin/bash
# run_overnight.sh — ejecuta los notebooks de simulación en secuencia.
# Uso: bash run_overnight.sh
# Cada notebook solo arranca si el anterior terminó sin error.
# Los outputs se guardan en el propio .ipynb.

set -e   # para si cualquier comando falla

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

TIMEOUT=86400   # 24 horas por notebook
LOG="$REPO/overnight_run.log"

echo "=============================" | tee "$LOG"
echo "Inicio: $(date)"               | tee -a "$LOG"
echo "Directorio: $REPO"             | tee -a "$LOG"
echo "=============================" | tee -a "$LOG"

run_notebook() {
    local nb="$1"
    echo "" | tee -a "$LOG"
    echo ">>> Arrancando $nb  ($(date))" | tee -a "$LOG"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=$TIMEOUT \
        "$nb" 2>&1 | tee -a "$LOG"
    echo "<<< Terminado  $nb  ($(date))" | tee -a "$LOG"
}

run_notebook "02-long-range-correlated.ipynb"
run_notebook "03-long-range-uncorrelated.ipynb"
run_notebook "04-lambda-sensitivity.ipynb"

echo "" | tee -a "$LOG"
echo "=============================" | tee -a "$LOG"
echo "Todo terminado: $(date)"       | tee -a "$LOG"
echo "=============================" | tee -a "$LOG"
