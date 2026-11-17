#!/usr/bin/env bash
# Parte 2 — curva de orçamento: piso (Parte 0b) e método (FixMatch flex + ajuste anticolapso)
# com 1, 5 e 20 rótulos por classe; no orçamento 5, 3 seeds.
#
#   bash run_part2.sh            # da raiz do projeto
#   WORKERS=1 bash run_part2.sh  # se a RAM estiver apertada (padrão: 2)
#
# Pode interromper e rodar de novo: runs que já têm summary.json são pulados.
# Os runs de seed 0 no orçamento 5 já existem (p0b_resnet18 e fm_tau0.95_flex_ls05r30).

export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONPATH=src
WORKERS=${WORKERS:-2}
mkdir -p runs/logs

# Hiperparâmetros do método = os escolhidos no val na Parte 1 (docs/parte1_fixmatch.md)
FM_ARGS="--tau 0.95 --correction flex --label-smoothing 0.05 --lambda-u-ramp 0.3"
# Piso = mesma receita da Parte 0b
PISO_ARGS="--model resnet18.tv_in1k --epochs 60"

roda() {  # roda NOME comando...  (até 3 tentativas; a última sem workers)
  local nome=$1; shift
  if [ -f "runs/$nome/summary.json" ]; then echo "--- $nome já existe, pulando"; return; fi
  for w in "$WORKERS" 1 0; do
    echo "=== $nome (workers=$w) início $(date +%H:%M)"
    rm -rf "runs/$nome"
    "$@" --workers "$w" --name "$nome" > "runs/logs/$nome.log" 2>&1
    if [ -f "runs/$nome/summary.json" ]; then
      echo "    ok $(date +%H:%M) | $(tail -1 "runs/logs/$nome.log")"; return
    fi
    echo "    falhou: $(tail -1 "runs/logs/$nome.log")"
  done
  echo "!!! $nome falhou 3 vezes (ver runs/logs/$nome.log)"
}

# 1) piso: rápido (~5-10 min cada)
for S in 1 2; do roda piso_b5_s$S uv run python src/train.py $PISO_ARGS --budget 5 --seed $S; done
roda piso_b1_s0  uv run python src/train.py $PISO_ARGS --budget 1  --seed 0
roda piso_b20_s0 uv run python src/train.py $PISO_ARGS --budget 20 --seed 0

# 2) método: ~20 min cada com 2 workers, ~30 min com 1
for S in 1 2; do roda fm_flex_b5_s$S uv run python src/fixmatch.py $FM_ARGS --budget 5 --seed $S; done
roda fm_flex_b1_s0  uv run python src/fixmatch.py $FM_ARGS --budget 1  --seed 0
roda fm_flex_b20_s0 uv run python src/fixmatch.py $FM_ARGS --budget 20 --seed 0

echo "PARTE 2 TERMINADA $(date +%H:%M)"
