# Parte 3 — Galeria de falhas e atalhos

Modelo final: **FixMatch flex + ajuste, orçamento 20** (`entrega/model.pt`, run `fm_flex_b20_s0`;
test F1 0,626). Mapa de saliência: **Grad-CAM** da classe **prevista** na última camada
convolucional (`layer4`, mapa 7×10 ampliado para a imagem). Implementação em `src/gradcam.py`.

```bash
uv run python src/gradcam.py --todas      # Grad-CAM das 811 imagens do test em gradcam_todas/ (fora do git)
uv run python src/gradcam.py --erros a6f1f0fd730dbbe7 4a538678f32700c7 53fe911df2430f06
```

Cada exemplo tem uma pasta em `figures/p3/` com `imagem.png` (original), `gradcam.png` (mapa
sobreposto), `lado_a_lado.png` (as duas) e `info.json`. Índice: `figures/p3/galeria.csv`.
As imagens não têm texto; classe, previsão e confiança estão abaixo e no `info.json`.

Os exemplos foram escolhidos a olho entre os Grad-CAMs de todo o test_public.

---

## Erro 1 — gambá classificado como urso

| | |
|---|---|
| imagem | `figures/p3/erro_1_virginia_opossum__american_black_bear/` (id `a6f1f0fd730dbbe7`) |
| classe verdadeira | virginia_opossum |
| classe prevista | american_black_bear |
| confiança | 0,11 |

**Diagnóstico:** foto noturna (infravermelho) com dois postes de luz acesos ao fundo. O gambá está
borrado pelo movimento, atravessando o gramado no terço inferior (cerca de 130×30 pixels na imagem
de 224×320, contando a cauda). O mapa está **nos postes e nas luzes**, com fundo preto ao redor,
e não no animal. A confiança é muito baixa (0,11): o modelo não reconheceu nada e caiu numa
classe associada à cena noturna.

## Erro 2 — cavalo classificado como veado

| | |
|---|---|
| imagem | `figures/p3/erro_2_horse__white_tailed_deer/` (id `4a538678f32700c7`) |
| classe verdadeira | horse |
| classe prevista | white_tailed_deer |
| confiança | 0,33 |

**Diagnóstico:** foto diurna de floresta. O cavalo (com sela) está cortado pela borda direita,
só a traseira e as pernas aparecem. O mapa se divide entre **as árvores do centro** e a anca/pernas
do cavalo. O padrão se repete no test: várias imagens com árvores foram previstas como
white_tailed_deer, com o mapa nas árvores. Hipótese: as imagens rotuladas de veado são de mata, e o
modelo associou "árvores" a veado. Cavalo também é a classe mais rara (4 imagens no test).

## Erro 3 — corvo classificado como peru selvagem

| | |
|---|---|
| imagem | `figures/p3/erro_3_american_crow__wild_turkey/` (id `53fe911df2430f06`) |
| classe verdadeira | american_crow |
| classe prevista | wild_turkey |
| confiança | 0,40 |

**Diagnóstico:** aqui o mapa está **no animal**: o corvo, no chão, em frente a uma pilha de galhos.
O modelo olhou para o lugar certo, mas confundiu duas aves escuras, de perfil, no chão da mata. A
confusão acontece várias vezes nos dois sentidos (crow → turkey e turkey → crow). É um erro de
classes parecidas, não de atalho de fundo, e serve de contraste com os erros 1 e 2.

---

## Acertos (pergunta inversa: olhou para o animal ou para o fundo?)

```bash
uv run python src/gradcam.py --acertos e2e960a9c33f1f9e 5f2e2bc78c073630 d7286e00316c3303 168048c13eda2985 0bae2965e3b5ac36
uv run python src/gradcam.py --mosaico eastern_chipmunk
```

| # | pasta em `figures/p3/` | classe (real = prevista) | confiança | olhou para |
|---|---|---|---|---|
| 1 | `acerto_1_eastern_fox_squirrel__eastern_fox_squirrel` (`e2e960a9c33f1f9e`) | eastern_fox_squirrel | 0,16 | **fundo** |
| 2 | `acerto_2_american_black_bear__american_black_bear` (`5f2e2bc78c073630`) | american_black_bear | 0,17 | **animal** |
| 3 | `acerto_3_grey_fox__grey_fox` (`d7286e00316c3303`) | grey_fox | 0,20 | **entorno** |
| 4 | `acerto_4_eastern_fox_squirrel__eastern_fox_squirrel` (`168048c13eda2985`) | eastern_fox_squirrel | 0,31 | **fundo** |
| 5 | `acerto_5_eastern_chipmunk__eastern_chipmunk` (`0bae2965e3b5ac36`) | eastern_chipmunk | 0,94 | **fundo (galão)** |

1. **Esquilo (0,16):** o esquilo está de cabeça para baixo no tronco, no canto superior esquerdo.
   O mapa está no **chão de folhas**, embaixo e à direita, longe do animal.
2. **Urso (0,17):** foto noturna; o mapa está **no urso**, à esquerda da imagem. Único acerto
   "pelo motivo certo" da galeria.
3. **Raposa-cinzenta (0,20):** foto noturna; a raposa está bem visível no centro, mas o mapa está
   na **área escura à direita da cabeça**, não no corpo dela.
4. **Esquilo (0,31):** o esquilo é um ponto escuro sobre a neve, embaixo à direita (cerca de 25×15
   pixels na imagem de 224×320). O mapa está na **faixa de árvores do meio** da imagem.
5. **Chipmunk (0,94):** não dá para achar o chipmunk na cena; o mapa está **no galão preto**. É o
   acerto mais confiante da galeria e o mais claramente "pelo motivo errado".

**Evidência do atalho do galão** (`figures/p3/rotulados_eastern_chipmunk.png`): das 20 imagens
**rotuladas** de eastern_chipmunk no orçamento 20 (as que o modelo final viu com rótulo), **16 mostram
o mesmo galão preto**. O modelo aprendeu "galão → chipmunk". Isso também explica por que chipmunk
tem o F1 mais alto em todas as partes (~0,97): o test_public tem imagens da mesma câmera (o split
separa câmera-noite, não câmera; o enunciado avisa que "o fundo se repete").

**Resposta à pergunta inversa:** dos 5 acertos, só **1** (o urso) tem o mapa no animal. Nos outros 4
o modelo acertou olhando para o fundo ou para o entorno. Junto com os erros 1 e 2, a galeria sugere
que boa parte do desempenho vem de reconhecer **a câmera/cena** associada a cada classe, não o
animal. Ressalva: os exemplos foram escolhidos a olho, não sorteados; isto é uma ilustração, não
uma medida da frequência do atalho.
