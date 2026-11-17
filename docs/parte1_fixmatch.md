# Parte 1 — Trilha B (FixMatch): diário de experimentos

Todos os números de **seleção** abaixo são do `val` (202 imagens). O `test_public` aparece
só como referência, nunca foi usado para escolher τ, passo ou ajuste.

Configuração comum: `resnet18.tv_in1k`, orçamento 5 (105 rótulos), seed 0, 3000 passos,
B=32 rotuladas + 64 não rotuladas por passo (μ=2), avaliação no val a cada 200 passos com o
modelo EMA. Referência: **piso (Parte 0b) = val F1 0,415** (test F1 0,415).

---

## 1. Varredura de τ (FixMatch original, sem correção)

```bash
for TAU in 0.7 0.8 0.9 0.95; do
  PYTHONUTF8=1 PYTHONPATH=src uv run python src/fixmatch.py --budget 5 --seed 0 \
      --tau $TAU --correction none --workers 2 --name fm_tau${TAU}_none
done
```

| τ    | melhor val F1 | melhor passo | máscara no fim | val loss no fim | % pseudo-rótulos "urso" | classes com mais pseudo-rótulos |
|------|---------------|--------------|----------------|-----------------|-------------------------|---------------------------------|
| 0,70 | 0,381         | 200          | 1,00           | 6,64            | 3,5 %                   | opossum 12,5 · gray_squirrel 11,7 · skunk 11,3 |
| 0,80 | 0,394         | 200          | 1,00           | 6,56            | 3,7 %                   | skunk 11,9 · opossum 11,1 · crow 10,8 |
| 0,90 | 0,405         | 200          | 0,99           | 5,82            | 3,5 %                   | crow 10,9 · opossum 9,9 · chicken 9,9 |
| **0,95** | **0,412** | 200          | 0,96           | 5,84            | 3,7 %                   | crow 11,5 · opossum 10,5 · chicken 10,2 |

**τ escolhido no val: 0,95.** Tendência monotônica: τ maior → menos pseudo-rótulos errados → val F1 maior.
Mesmo assim, **nenhum τ supera o piso**, e o melhor passo é sempre o 200 (a 1ª avaliação).

### Diagnóstico: colapso por viés de confirmação

Histórico típico (τ = 0,8):

| passo | perda sup. | máscara | val loss | val F1 |
|-------|------------|---------|----------|--------|
| 200   | 0,872      | 19 %    | 2,36     | 0,394  |
| 400   | **0,024**  | 71 %    | 3,70     | 0,346  |
| 600   | 0,015      | **91 %**| 4,67     | 0,360  |
| 3000  | 0,002      | ~100 %  | **6,56** | 0,350  |

1. **O ramo rotulado decora as 105 imagens em ~400 passos** (cada passo vê 32 delas; em 400
   passos cada imagem foi vista ~120 vezes). Perda supervisionada → 0.
2. Uma rede que decorou fica **superconfiante em tudo** (o mesmo fenômeno da Parte 0: a val loss
   sobe enquanto o F1 quase não muda). Com confiança alta em qualquer imagem, quase todo o pool
   passa do limiar: a **máscara vai a ~100 %**.
3. Mas a rede acerta só ~35 % → a maioria dos pseudo-rótulos aceitos está **errada**, e ela passa
   a treinar nos próprios erros, cada vez mais confiante (val loss 2,4 → 6,6).

τ maior só **atrasa** o colapso (máscara no passo 400: 71 % com τ=0,8; 31 % com τ=0,95), não o evita.

### Viés de classe: segue o fundo, não o tamanho da classe

O enunciado prevê pseudo-rótulos indo para as classes grandes. Aqui **não**: o urso, maior classe
do pool (690 imagens), recebe só ~3,5 % dos pseudo-rótulos. Quem domina são opossum, skunk, crow,
chicken e gray_squirrel. Na exploração (`notebooks/00_exploracao.ipynb`) vimos que **as 5 imagens
rotuladas de virginia_opossum são a mesma cena noturna** e que chicken aparece no corredor com tela
de galinheiro. Hipótese: a rede aprende o **fundo** ("noite + chão de folhas", "tela") e carimba
essas classes em todo o pool parecido. É o atalho que a Parte 3 vai investigar com Grad-CAM.

---

## 2. Ajuste anticolapso (implementado em `src/fixmatch.py`)

Duas opções novas, **desligadas por padrão** (os runs da seção 1 continuam reproduzíveis):

| opção | o que faz | por quê |
|-------|-----------|---------|
| `--label-smoothing ε` | cross-entropy com label smoothing **só no ramo rotulado** | ataca a superconfiança: a rede deixa de ser empurrada para 100 % nas 105 imagens decoradas, então a máscara não explode |
| `--lambda-u-ramp f` | λ_u sobe linearmente de 0 até `--lambda-u` nos primeiros `f · steps` passos | nos primeiros passos a rede é ruim; os pseudo-rótulos dessa fase entravam com peso cheio e "contaminavam" o treino |

**Interação ε × τ (cuidado!).** Com label smoothing ε, a confiança-alvo máxima no ramo rotulado é
`1 − ε + ε/21`. Se ela ficar abaixo de τ, quase nenhum pseudo-rótulo passa e o FixMatch vira o piso.
ε = 0,1 → teto 0,905 (incompatível com τ = 0,95); **ε = 0,05 → teto 0,952** (compatível).
O script imprime um aviso quando a combinação é inválida.

Valores usados: **τ = 0,95** (escolhido no val, seção 1), **ε = 0,05**, **rampa = 0,3** (λ_u chega a
1 no passo 900, cobrindo a janela 400–1000 em que o colapso acontecia). λ_u, lr e o resto iguais.

Os mesmos ajustes valem para as correções de viés (`da`, `flex`), para que a comparação entre as
três isole só o efeito da correção.

```bash
for CORR in none da flex; do
  PYTHONUTF8=1 PYTHONPATH=src uv run python src/fixmatch.py --budget 5 --seed 0 --tau 0.95 \
      --correction $CORR --label-smoothing 0.05 --lambda-u-ramp 0.3 --workers 1 \
      --name fm_tau0.95_${CORR}_ls05r30
done
```

### Resultados

| run (τ = 0,95) | melhor val F1 | melhor passo | máscara no melhor passo | máscara no fim | val loss no fim | % pseudo-rótulos aceitos "urso" | classes com mais pseudo-rótulos aceitos | test F1 (ref.) |
|----------------|---------------|--------------|-------------------------|----------------|-----------------|---------------------------------|------------------------------------------|----------------|
| none, sem ajuste (seção 1) | 0,412 | 200 | — | 0,96 | 5,84 | 3,7 % | crow · opossum · chicken | — |
| none + ajuste  | 0,416 | 800 | 0,07 | 0,17 | 2,08 | 1,3 % | chipmunk 19,1 · chicken 15,0 · gray_squirrel 11,7 | 0,494 |
| da + ajuste    | 0,406 | 800 | 0,05 | 0,13 | 2,12 | 1,8 % | chipmunk 23,2 · gray_squirrel 14,6 · deer 10,1 | 0,502 |
| **flex + ajuste** | **0,433** | 600 | 0,87 | 1,00 | 4,35 | 7,5 % | opossum 8,1 · raccoon 7,8 · chicken 7,7 | 0,485 |
| *piso (Parte 0b)* | *0,415* | | | | | | | *0,415* |

(% de pseudo-rótulos: distribuição no passo 3000, `pseudo_stats.csv`.)

**O que o ajuste fez (`none`):**
- **Evitou o colapso.** A perda supervisionada estabiliza em ~0,34 em vez de ir a 0 (o label smoothing
  impede a confiança de 100 % nas 105 imagens), a máscara sobe devagar até ~17 % e a val loss fica
  em ~2,1, contra 5,8 sem ajuste.
- **O melhor passo foi de 200 para 800**: o modelo aprende por mais tempo antes de piorar.
- **Mas no val o ganho é nulo**: 0,416 vs piso 0,415. Com só ~10–17 % do pool aceito, e concentrado em
  poucas classes (esquilo/chipmunk, chicken), o sinal não rotulado é pequeno e enviesado.

**Correções de viés:**
- `da` (DistAlign) espalha mais as predições brutas, mas os aceitos continuam concentrados em
  chipmunk/esquilo; val F1 0,406, um pouco abaixo do `none`.
- `flex` (limiares por classe) **baixa o limiar das classes pouco aceitas**, então a máscara volta a ~100 %.
  A distribuição fica bem mais equilibrada (urso recebe 7,5 %, maior classe ≤ 8,1 %) e o pico de
  val F1 é o melhor de todos (**0,433 no passo 600**), mas depois o colapso volta (val loss 4,35 no fim).
  É o único run que supera o piso no val.

**Escolha pelo val: `flex` + ajuste, passo 600 (val F1 0,433).** Ressalva: a diferença para o piso
(+0,018) é de ~4 imagens em 202 e veio de um único seed, então é frágil.

**val × test.** No test os três runs com ajuste ficam em ~0,49–0,50, bem acima do piso (0,415), e a
ordem entre eles é diferente da do val. Como o val tem só 202 imagens (≈10 por classe), a seleção é
ruidosa; mesmo assim seguimos a regra: quem escolhe é o val, o test é só referência.

---

## Notas de execução (máquina com 8 GB de RAM)

- Com outros programas abertos, os workers do DataLoader estouram a RAM (`error code 1455`,
  `CUDA unknown error` ou segfault). Feche jogos/navegadores; `--workers 1` é mais seguro, `--workers 2`
  ~2× mais rápido quando há RAM livre.
- Ao redirecionar o log para arquivo no Windows, use `PYTHONUTF8=1` (o `print` usa a letra τ).
- Tempo por run (3000 passos, RTX 5050): ~20 min com 2 workers, ~28 min com 1 worker.
