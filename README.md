# PA3 — Poucos rótulos: fauna em armadilhas fotográficas (Trilha B: FixMatch)

Classificação de 21 espécies em imagens de armadilhas fotográficas (ENA24) com poucos rótulos,
usando aprendizado semi-supervisionado (**FixMatch**) sobre uma ResNet-18 pré-treinada no ImageNet.

**Modelo entregue:** FixMatch + label smoothing + rampa de λ_u + limiar por classe (`flex`),
orçamento de 20 rótulos por classe. `entrega/model.pt`, 11.187.285 parâmetros,
**F1 macro 0,626** no test_public (teto do monitor: 0,686).

Todas as escolhas (época, τ, correção, modelo final) foram feitas no **val**; o test_public só foi
usado para reportar números.

---

## Ambiente

Requer [uv](https://docs.astral.sh/uv/) e Python 3.12. O `pyproject.toml` instala o PyTorch com
CUDA 12.8 (funciona também só em CPU).

```bash
uv sync
```

Dados (`pa3_dados.zip`, 415 MB, link do enunciado), extraídos em `data/pa3_dados/`:

```bash
uv run gdown 1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui -O pa3_dados.zip
uv run python -c "import zipfile; zipfile.ZipFile('pa3_dados.zip').extractall('data')"
```

Todos os comandos rodam **da raiz do repositório**. No Windows, se redirecionar a saída para um
arquivo, defina `PYTHONUTF8=1` (os scripts imprimem a letra τ).

## Comando que treina (modelo entregue)

```bash
uv run python src/fixmatch.py --budget 20 --seed 0 --tau 0.95 --correction flex \
    --label-smoothing 0.05 --lambda-u-ramp 0.3 --workers 1 --name fm_flex_b20_s0
```

~30 min numa RTX 5050. Salva `runs/fm_flex_b20_s0/` (`model.pt`, `history.csv`, `summary.json`,
`preds_test.csv`, `pseudo_stats.csv`). Para entregar: copiar `model.pt` para `entrega/model.pt`.
Com pouca RAM, use `--workers 1` (ou `0`); com folga, `--workers 2` é ~2× mais rápido.

## Comando que avalia

```bash
uv run python predict.py --imagens PASTA --saida preds.csv
```

Roda em CPU, sem internet, imprime o número de parâmetros e escreve `image_id,label` com os nomes
de `classes.csv`. Usa `entrega/model.pt` por padrão (`--pesos` para outro checkpoint).
Tempo medido: 811 imagens em ~45 s (~3 min com 1 thread).

---

## Resultados

| | detalhes em |
|---|---|
| **Parte 0** — piso: CNN do zero 0,210 × ResNet-18 ImageNet 0,415 (test F1, 105 rótulos) | [docs/parte0.md](docs/parte0.md) |
| **Parte 1** — FixMatch: varredura de τ, colapso por viés de confirmação, ajuste anticolapso, precisão/cobertura dos pseudo-rótulos, viés e correções (`da`, `flex`) | [docs/parte1_fixmatch.md](docs/parte1_fixmatch.md) |
| **Parte 2** — curva de orçamento (1, 5, 20 rótulos por classe; 3 seeds no 5) | [docs/parte2.md](docs/parte2.md), [curva_orcamento.csv](curva_orcamento.csv) |
| **Parte 3** — galeria de erros e acertos com Grad-CAM; atalho de fundo | [docs/parte3.md](docs/parte3.md) |

| rótulos por classe | piso (test F1) | FixMatch (test F1) |
|---|---|---|
| 1 | 0,259 | 0,192 |
| 5 (3 seeds) | 0,456 ± 0,039 | 0,502 ± 0,016 |
| 20 | 0,592 | **0,626** |

---

## Estrutura

```
predict.py              inferência (entregável)
run_part2.sh            todos os treinos da Parte 2 (pula os que já existem)
src/
  data.py               leitura dos CSVs, transformações (entrada 224×320), datasets
  models.py             SmallCNN e ResNet-18 (timm)
  engine.py             laço de treino/avaliação, métricas
  train.py              treino supervisionado (piso, Partes 0 e 2)
  fixmatch.py           FixMatch (+ da, flex, label smoothing, rampa de λ_u)
  pseudolabel_diag.py   precisão/cobertura dos pseudo-rótulos vs τ e viés (Parte 1)
  plots_part0.py        curvas, matriz de confusão e exemplos (Parte 0)
  plots_part1.py        figura do colapso (Parte 1)
  part2_curve.py        curva_orcamento.csv, gráfico e ganho por classe (Parte 2)
  gradcam.py            Grad-CAM e galeria (Parte 3)
notebooks/              exploração dos dados
runs/<nome>/            resultados de cada treino (sem os pesos)
analysis/, figures/     tabelas e figuras usadas na apresentação
docs/                   descrição e análise de cada parte
entrega/model.pt        pesos do modelo entregue
```

## Reproduzir tudo

```bash
# Parte 0
uv run python src/train.py --model smallcnn --budget 5 --epochs 150 --name p0a_smallcnn
uv run python src/train.py --model resnet18.tv_in1k --budget 5 --epochs 60 --name p0b_resnet18
uv run python src/plots_part0.py

# Parte 1 (comandos completos em docs/parte1_fixmatch.md)
for TAU in 0.7 0.8 0.9 0.95; do
  uv run python src/fixmatch.py --budget 5 --seed 0 --tau $TAU --correction none --name fm_tau${TAU}_none
done
for CORR in none da flex; do
  uv run python src/fixmatch.py --budget 5 --seed 0 --tau 0.95 --correction $CORR \
      --label-smoothing 0.05 --lambda-u-ramp 0.3 --name fm_tau0.95_${CORR}_ls05r30
done
uv run python src/pseudolabel_diag.py --baseline p0b_resnet18 \
    --runs fm_tau0.95_none fm_tau0.95_none_ls05r30 fm_tau0.95_da_ls05r30 fm_tau0.95_flex_ls05r30 --tau 0.95
uv run python src/plots_part1.py

# Parte 2
bash run_part2.sh
uv run python src/part2_curve.py

# Parte 3 (depois de copiar runs/fm_flex_b20_s0/model.pt para entrega/model.pt)
uv run python src/gradcam.py --erros a6f1f0fd730dbbe7 4a538678f32700c7 53fe911df2430f06 \
    --acertos e2e960a9c33f1f9e 5f2e2bc78c073630 d7286e00316c3303 168048c13eda2985 0bae2965e3b5ac36
uv run python src/gradcam.py --mosaico eastern_chipmunk
```

Os resultados de todos os treinos já estão em `runs/` (exceto os pesos), então as análises e figuras
podem ser regeneradas sem treinar de novo. Os números podem variar um pouco entre máquinas
(não determinismo da GPU).

## Uso de IA

Ver [AI_LOG.md](AI_LOG.md).
