"""Parte 3: Grad-CAM do modelo final (galeria de erros e acertos).

    uv run python src/gradcam.py --todas                         # 811 imagens em gradcam_todas/
    uv run python src/gradcam.py --erros ID ID ID --acertos ID ID ID ID ID   # galeria final
    uv run python src/gradcam.py                                 # galeria automática (mais confiantes)
    uv run python src/gradcam.py --mosaico eastern_chipmunk      # imagens rotuladas de uma classe
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import data
import models

PESOS = "entrega/model.pt"
RUN = "fm_flex_b20_s0"  # o run que gerou o modelo final (tem o preds_test.csv)
FIG = Path("figures/p3")


class GradCAM:
    """Guarda a ativação da camada-alvo no forward e o gradiente dela no backward."""

    def __init__(self, model, camada):
        self.model = model
        camada.register_forward_hook(self._guarda)

    def _guarda(self, modulo, entrada, saida):
        self.ativ = saida                                         # (1, 512, 7, 10)
        saida.register_hook(lambda g: setattr(self, "grad", g))  # chamado no backward

    def __call__(self, x, classe):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, classe].backward()               # gradiente do score DESSA classe
        pesos = self.grad.mean(dim=(2, 3), keepdim=True)          # importância de cada mapa
        cam = F.relu((pesos * self.ativ).sum(dim=1, keepdim=True))  # soma ponderada
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear")[0, 0]
        cam = cam / (cam.max() + 1e-8)                            # 0..1
        return cam.detach().numpy(), logits.softmax(1)[0].detach()


def carrega_modelo():
    ckpt = torch.load(PESOS, map_location="cpu")
    model = models.build_model(ckpt["model"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["classes"]


def painel(ax_img, ax_cam, path, cam, titulo):
    """Imagem do jeito que o modelo viu (224x320) e o mapa por cima."""
    img = np.asarray(Image.open(path).convert("RGB").resize(data.IMG_SIZE[::-1]))
    ax_img.imshow(img); ax_img.set_title(titulo, fontsize=8)
    ax_cam.imshow(img); ax_cam.imshow(cam, cmap="jet", alpha=0.45)
    ax_cam.set_title("Grad-CAM", fontsize=8)
    ax_cam.axis("off")
    ax_img.tick_params(labelsize=6)  # eixos em pixels da imagem 224x320, para medir o animal


def escolhe(p, n, erros):
    """n exemplos de classes reais diferentes, dos mais confiantes para os menos."""
    sel = p[(p["label"] != p["pred"]) if erros else (p["label"] == p["pred"])]
    return sel.sort_values("conf", ascending=False).drop_duplicates("label").head(n)


def figura(gc, classes, e, arq, dpi=150):
    """Uma figura (imagem + Grad-CAM da classe prevista) para a linha e de preds_test.csv."""
    path = data.DATA_ROOT / "images" / "test_public" / f"{e['image_id']}.jpg"
    x = data.eval_transform()(Image.open(path).convert("RGB")).unsqueeze(0)
    pred = classes.index(e["pred"])
    cam, probs = gc(x, pred)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    painel(axs[0], axs[1], path, cam,
           f"real: {e['label']} | previsto: {e['pred']} ({probs[pred]:.2f})")
    fig.tight_layout()
    fig.savefig(arq, dpi=dpi)
    plt.close(fig)


def exporta(gc, classes, e, pasta):
    """Versão "limpa" para slides, sem texto, na resolução original da imagem:
    imagem.png, gradcam.png (mapa sobreposto), lado_a_lado.png e info.json."""
    pasta.mkdir(parents=True, exist_ok=True)
    path = data.DATA_ROOT / "images" / "test_public" / f"{e['image_id']}.jpg"
    original = Image.open(path).convert("RGB")
    x = data.eval_transform()(original).unsqueeze(0)
    pred = classes.index(e["pred"])
    cam, probs = gc(x, pred)

    # o mapa sai em 224x320 (o que o modelo viu); esticamos para o tamanho original
    W, H = original.size
    cam = F.interpolate(torch.from_numpy(cam)[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
    img = np.asarray(original) / 255.0
    sobreposto = 0.55 * img + 0.45 * plt.get_cmap("jet")(cam)[..., :3]

    plt.imsave(pasta / "imagem.png", img)
    plt.imsave(pasta / "gradcam.png", sobreposto.clip(0, 1))
    espaco = np.ones((H, 12, 3))
    plt.imsave(pasta / "lado_a_lado.png", np.concatenate([img, espaco, sobreposto.clip(0, 1)], axis=1))

    info = {"image_id": e["image_id"], "classe_real": e["label"], "classe_prevista": e["pred"],
            "confianca": round(float(probs[pred]), 3), "acertou": bool(e["label"] == e["pred"]),
            "tamanho_original": [W, H]}
    (pasta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def galeria(gc, classes, exemplos, nome):
    """Uma pasta por exemplo: figures/p3/<nome>_<i>_<real>__<previsto>/"""
    for i, (_, e) in enumerate(exemplos.iterrows(), start=1):
        pasta = FIG / f"{nome}_{i}_{e['label']}__{e['pred']}"
        info = exporta(gc, classes, e, pasta)
        print(f"salvo: {pasta}  ({info['classe_real']} -> {info['classe_prevista']}, {info['confianca']})")


def mosaico(classe, budget=20):
    """Todas as imagens ROTULADAS de uma classe no orçamento do modelo final (sem texto),
    para checar se elas compartilham o fundo (atalho)."""
    d = data.load_split(f"rotulos_{budget}")
    d = d[d["label"] == classe]
    cols = 5
    linhas = int(np.ceil(len(d) / cols))
    fig, axs = plt.subplots(linhas, cols, figsize=(3 * cols, 2.1 * linhas), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, path in zip(axs.flat, d["path"]):
        ax.imshow(Image.open(path).convert("RGB"))
    fig.tight_layout()
    arq = FIG / f"rotulados_{classe}.png"
    fig.savefig(arq, dpi=120)
    plt.close(fig)
    print("salvo:", arq)


def indice():
    """figures/p3/galeria.csv com todos os exemplos exportados (lê os info.json)."""
    linhas = []
    for f in sorted(FIG.glob("*/info.json")):
        linhas.append({"pasta": f.parent.name, **json.loads(f.read_text(encoding="utf-8"))})
    pd.DataFrame(linhas).drop(columns="tamanho_original").to_csv(FIG / "galeria.csv", index=False)
    print("índice:", FIG / "galeria.csv")


def todas(gc, classes, p, pasta=Path("gradcam_todas")):
    """Grad-CAM de TODO o test_public, para escolher a olho (pasta fora do git).

    Nome: <conf>_<real>__<previsto>_<id>.jpg; ordenar por nome = do menos para o mais confiante.
    """
    for sub in ("erros", "acertos"):
        (pasta / sub).mkdir(parents=True, exist_ok=True)
    for k, (_, e) in enumerate(p.iterrows(), start=1):
        sub = "acertos" if e["label"] == e["pred"] else "erros"
        arq = pasta / sub / f"{e['conf']:.3f}_{e['label']}__{e['pred']}_{e['image_id']}.jpg"
        figura(gc, classes, e, arq, dpi=100)
        if k % 100 == 0:
            print(f"{k}/{len(p)}")
    print(f"pronto: {pasta}/erros e {pasta}/acertos")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true",
                    help="gera o Grad-CAM das 811 imagens em gradcam_todas/ (para escolher)")
    ap.add_argument("--erros", nargs="+", help="image_ids dos erros escolhidos para a galeria")
    ap.add_argument("--acertos", nargs="+", help="image_ids dos acertos escolhidos para a galeria")
    ap.add_argument("--mosaico", help="classe: mosaico das imagens rotuladas dela (rotulos_20)")
    args = ap.parse_args()

    if args.mosaico:
        FIG.mkdir(parents=True, exist_ok=True)
        mosaico(args.mosaico)
        raise SystemExit

    model, classes = carrega_modelo()
    gc = GradCAM(model, model.layer4)
    p = pd.read_csv(Path("runs") / RUN / "preds_test.csv")

    if args.todas:
        todas(gc, classes, p)
    else:
        FIG.mkdir(parents=True, exist_ok=True)
        pid = p.set_index("image_id", drop=False)
        if args.erros or args.acertos:  # escolha manual (pode passar só um dos dois)
            if args.erros:
                galeria(gc, classes, pid.loc[args.erros], "erro")
            if args.acertos:
                galeria(gc, classes, pid.loc[args.acertos], "acerto")
        else:  # automático: os mais confiantes
            galeria(gc, classes, escolhe(p, 3, erros=True), "erro")
            galeria(gc, classes, escolhe(p, 5, erros=False), "acerto")
        indice()
