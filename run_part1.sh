#!/usr/bin/env bash
# Roteiro da Parte 1 (Trilha B). Rode da raiz do projeto, na ordem.
set -e
export PYTHONPATH=src

# 0) Piso (Parte 0b), se ainda não rodou
# uv run python src/train.py --model resnet18.tv_in1k --epochs 60 --name p0b_resnet18

# 1) Escolha de τ NO VAL (FixMatch puro, seed 0)
for TAU in 0.7 0.8 0.9 0.95; do
  uv run python src/fixmatch.py --budget 5 --tau $TAU --correction none --name fm_tau${TAU}_none
done
# veja o val_f1 de cada um e escolha o melhor τ (NÃO olhe o test_public para isso):
uv run python - << 'PY'
import json, glob
for f in sorted(glob.glob("runs/fm_tau*_none/summary.json")):
    s = json.load(open(f)); print(f"{s['name']:20s} τ={s['tau']}  val_F1={s['val_f1']:.3f}  melhor passo={s['best_step']}")
PY

# 2) Correção do viés, com o melhor τ da etapa 1 (ajuste BEST_TAU)
BEST_TAU=0.9
uv run python src/fixmatch.py --budget 5 --tau $BEST_TAU --correction da   --name fm_tau${BEST_TAU}_da
uv run python src/fixmatch.py --budget 5 --tau $BEST_TAU --correction flex --name fm_tau${BEST_TAU}_flex

# 3) Diagnósticos obrigatórios (precisão/cobertura vs τ e viés) -> analysis/
uv run python src/pseudolabel_diag.py --baseline p0b_resnet18 \
    --runs fm_tau${BEST_TAU}_none fm_tau${BEST_TAU}_da fm_tau${BEST_TAU}_flex --tau $BEST_TAU
