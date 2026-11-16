# PA3 — Poucos Rótulos: Fauna em Câmeras Trap (Trilha B: FixMatch)

Este repositório contém a implementação do **Programming Assignment 3 (Aprendizado Profundo)** focado na classificação de 21 espécies da fauna em imagens de câmeras trap (`ENA24-detection`) utilizando aprendizado semi-supervisionado (**Trilha B — FixMatch**).

> **Status Atual (Handoff para GPU):** Toda a lógica do pipeline (`src/data.py`, `src/train.py`, `src/fixmatch.py` e `src/pseudolabel_diag.py`) já foi validada via *smoke tests* na CPU (sem erros de dimensão, importação ou cálculo de perda). Como o treino completo exige processamento intensivo de aumentação de dados (*RandAugment*) e milhares de passos, **nenhum modelo definitivo foi treinado ainda**. Este guia descreve a sequência exata para executar os treinos na GPU e consolidar os resultados até a Parte 2.

---

## 1. Ambiente e Dados

### Requisitos e Instalação
O projeto utiliza `uv` para gerenciamento de ambiente e dependências (`PyTorch`, `torchvision`, `timm`, `albumentations`, `scikit-learn`).

```bash
# 1. Sincronizar o ambiente virtual do projeto
uv sync

# 2. Definir o PYTHONPATH para os scripts em src/
export PYTHONPATH=src
# (No Windows PowerShell, use: $env:PYTHONPATH="src")
```

### Estrutura de Dados
Certifique-se de que o pacote `pa3_dados.zip` (415 MB) está extraído no diretório esperado pelos scripts, contendo:
- `train`: 6.505 imagens (pool não rotulado + subconjuntos rotulados).
- `rotulos_1.csv`, `rotulos_5.csv`, `rotulos_20.csv`: orçamentos de 1, 5 e 20 rótulos por classe (21, 105 e 420 imagens).
- `val`: 202 imagens (usado exclusivamente para seleção de época, limiar $\tau$ e correção de viés).
- `test_public`: 811 imagens (usado apenas para reportar métricas finais; **nunca** usar para escolher hiperparâmetros).

### Resumo Rápido (Comandos Principais)
- **Comando que treina (modelo final da Trilha B):**
  ```bash
  PYTHONPATH=src uv run python src/fixmatch.py --budget 5 --tau <BEST_TAU> --correction <BEST_CORR> --seed 0 --name fm_final
  ```
- **Comando que avalia (inferência offline em CPU — entregável final):**
  ```bash
  uv run python predict.py imagens <PASTA_IMAGENS> -saida preds.csv
  ```

---

## 2. Parte 0 — O Piso (Baseline Supervisionado)

O objetivo da Parte 0 é estabelecer a linha de base treinando apenas com os 105 rótulos (`rotulos_5.csv`) e verificar a contagem de parâmetros (limite obrigatório $\le 12.000.000$).

### 2.1. Piso (b): Backbone ImageNet com Fine-Tuning (`ResNet-18`)
Este run é **obrigatório antes da Parte 1**, pois gera `runs/p0b_resnet18` (utilizado como referência nos diagnósticos da Trilha B e reaproveitado como `seed 0` do orçamento 5 na Parte 2).

```bash
export PYTHONPATH=src
uv run python src/train.py --model resnet18.tv_in1k --budget 5 --seed 0 --epochs 60 --workers 4 --name p0b_resnet18
```
* **Conferir:** O log deve imprimir **`11,187,285` parâmetros** e salvar os pesos da melhor época no `val`, além das métricas no `test_public` e dados para a matriz de confusão e curva de treino/validação.

### 2.2. Piso (a): CNN Pequena Treinada do Zero
Para completar os entregáveis da Parte 0, rode também a versão (a) com uma CNN pequena sem pesos pré-treinados (verifique a flag/arquitetura configurada em `src/train.py` para o modelo *from scratch*):

```bash
uv run python src/train.py --model <NOME_CNN_PEQUENA> --budget 5 --seed 0 --epochs 60 --workers 4 --name p0a_small_cnn
```

---

## 3. Parte 1 — Trilha B: FixMatch e Correção de Viés

O roteiro automatizado está em `run_part1.sh`. Como cada run completo de 3.000 passos leva entre 15 e 50 minutos dependendo da CPU/GPU, recomenda-se rodar em segundo plano (`nohup` ou `tmux`) e ajustar `--workers 3` ou `4` caso o uso da GPU fique abaixo de 50%.

### Etapa 1.1: Varredura do Limiar $\tau$ no `val` (FixMatch Puro)
Rode o FixMatch sem correção de viés (`--correction none`) variando $\tau \in \{0.7, 0.8, 0.9, 0.95\}$:

```bash
export PYTHONPATH=src
for TAU in 0.7 0.8 0.9 0.95; do
  uv run python src/fixmatch.py --budget 5 --seed 0 --tau $TAU --correction none --workers 4 --name fm_tau${TAU}_none
done
```
*(Dica de otimização de tempo: se o tempo de GPU estiver apertado, avalie apenas `{0.8, 0.95}` ou rode a varredura com `--steps 1500`, mantendo `3000` passos nos runs definitivos).*

**Como escolher o `BEST_TAU`:**
Inspecione o `val_f1` de cada experimento (sem olhar o `test_public`):
```bash
uv run python - << 'PY'
import json, glob
for f in sorted(glob.glob("runs/fm_tau*_none/summary.json")):
    s = json.load(open(f))
    print(f"{s['name']:20s} tau={s['tau']}  val_F1={s['val_f1']:.3f}  melhor_passo={s['best_step']}")
PY
```
Escolha o $\tau$ com maior `val_f1` (em caso de empate técnico com diferença $< 0.02$, prefira o $\tau$ que ativa a máscara de pseudo-rótulos de forma estável sem colapsar para zero).

### Etapa 1.2: Teste das Correções de Desbalanceamento (`da` e `flex`)
Com o `BEST_TAU` definido na etapa anterior, treine as duas estratégias de mitigação de viés para classes raras (*Distribution Alignment* e *FlexMatch*):

```bash
BEST_TAU=0.8  # <-- SUBSTITUA pelo melhor tau encontrado no val na Etapa 1.1

uv run python src/fixmatch.py --budget 5 --seed 0 --tau $BEST_TAU --correction da   --workers 4 --name fm_tau${BEST_TAU}_da
uv run python src/fixmatch.py --budget 5 --seed 0 --tau $BEST_TAU --correction flex --workers 4 --name fm_tau${BEST_TAU}_flex
```

### Etapa 1.3: Diagnósticos Obrigatórios da Trilha B
Gere as tabelas de **precisão vs. cobertura por classe** e o diagnóstico de **viés nas classes raras** (salvos em `analysis/`):

```bash
uv run python src/pseudolabel_diag.py --baseline p0b_resnet18 \
    --runs fm_tau${BEST_TAU}_none fm_tau${BEST_TAU}_da fm_tau${BEST_TAU}_flex --tau $BEST_TAU
```

**Decisão para a Parte 2:** Compare `fm_tau${BEST_TAU}_none`, `fm_tau${BEST_TAU}_da` e `fm_tau${BEST_TAU}_flex` pelo **`val_f1`** (e verifique no diagnóstico se a correção reduziu a concentração nas classes frequentes e melhorou o desempenho nas 5 classes mais raras). Defina essa configuração vencedora como `BEST_CORR` (`none`, `da` ou `flex`).

---

## 4. Parte 2 — Curva de Orçamento (1, 5 e 20 Rótulos por Classe)

Na Parte 2, comparamos o **Piso (`train.py`)** contra o **Método Vencedor da Trilha B (`fixmatch.py`)** nos orçamentos de `1`, `5` e `20` rótulos por classe, tendo como referência o teto do monitor (`F1 macro = 0.686`).

### Economia de Tempo (Reaproveitamento da `seed 0`)
Para o orçamento de **5 rótulos**, o enunciado exige **3 seeds** (`média ± desvio padrão`), enquanto para os orçamentos de **1 e 20 rótulos** basta **1 seed**.
- **Já estão prontos (não rodar novamente):**
  - Piso, orçamento 5, `seed 0`: `runs/p0b_resnet18`
  - FixMatch, orçamento 5, `seed 0`: `runs/fm_tau${BEST_TAU}_${BEST_CORR}`

### Comandos Restantes da Parte 2
Substitua `<BEST_TAU>` e `<BEST_CORR>` pelos valores escolhidos na Parte 1 e execute apenas os 6 treinos faltantes:

```bash
export PYTHONPATH=src
BEST_TAU=0.8      # <-- AJUSTAR
BEST_CORR=flex    # <-- AJUSTAR (none, da ou flex)

# 1) Orçamento 5 — Seeds adicionais (seed 1 e seed 2)
for SEED in 1 2; do
  uv run python src/train.py --model resnet18.tv_in1k --budget 5 --seed $SEED --epochs 60 --workers 4 --name p0b_b5_s${SEED}
  uv run python src/fixmatch.py --budget 5 --seed $SEED --tau $BEST_TAU --correction $BEST_CORR --workers 4 --name fm_b5_s${SEED}
done

# 2) Orçamento 1 (21 imagens rotuladas) — Seed 0
uv run python src/train.py --model resnet18.tv_in1k --budget 1 --seed 0 --epochs 60 --workers 4 --name p0b_b1_s0
uv run python src/fixmatch.py --budget 1 --seed 0 --tau $BEST_TAU --correction $BEST_CORR --workers 4 --name fm_b1_s0

# 3) Orçamento 20 (420 imagens rotuladas) — Seed 0
uv run python src/train.py --model resnet18.tv_in1k --budget 20 --seed 0 --epochs 60 --workers 4 --name p0b_b20_s0
uv run python src/fixmatch.py --budget 20 --seed 0 --tau $BEST_TAU --correction $BEST_CORR --workers 4 --name fm_b20_s0
```

### Consolidação do `curva_orcamento.csv`
Após o término dos runs acima, extraia os resultados de cada `summary.json` para gerar o arquivo obrigatório `curva_orcamento.csv` na raiz do repositório com as colunas exatas:
`método, orçamento, seed, F1 macro, acurácia`

---

## 5. Checklist de Integridade e Próximos Passos

Antes de fechar a apresentação e o repositório final, valide os itens abaixo:
- [ ] **Nenhum vazamento do `test_public`:** Todo hiperparâmetro ($\tau$, época/passo, correção `da`/`flex`) foi escolhido exclusivamente pelo `val`.
- [ ] **Limite de parâmetros:** Todos os modelos possuem `11.187.285` parâmetros ($\le 12\text{M}$).
- [ ] **Parte 3 (Galeria de Falhas e Acertos):** Gerar mapas de saliência (Grad-CAM ou oclusão) usando os pesos do melhor modelo (`seed 0` do orçamento 5) em 3 erros e 5 acertos do `test_public`.
- [ ] **Script `predict.py`:** Implementar/validar na raiz do projeto o comando `python predict.py imagens PASTA -saida preds.csv` rodando 100% em CPU, sem internet e carregando os pesos salvos do modelo final.
- [ ] **Logs de IA (`AI_LOG.md`):** Documentar os episódios em que ferramentas de IA foram utilizadas durante o desenvolvimento.