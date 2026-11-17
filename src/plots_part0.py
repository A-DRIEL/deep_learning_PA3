"""Figuras da Parte 0: curvas de treino/validação e matriz de confusão.

    uv run python src/plots_part0.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import data


RUNS = Path("runs")
FIG = Path("figures")


def plot_curva(run, titulo, axes):
    """Duas colunas: perda (treino × val) e acurácia/F1, com a época escolhida marcada."""
    h = pd.read_csv(RUNS / run / "history.csv")
    melhor = h.loc[h["val_f1"].idxmax(), "epoch"]  # mesma regra do train.py

    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], label="treino")
    ax.plot(h["epoch"], h["val_loss"], label="val")
    ax.set(title=f"{titulo}: perda", xlabel="época", ylabel="cross-entropy")

    ax = axes[1]
    ax.plot(h["epoch"], h["train_acc"], label="acc treino")
    ax.plot(h["epoch"], h["val_acc"], label="acc val")
    ax.plot(h["epoch"], h["val_f1"], label="F1 macro val")
    ax.set(title=f"{titulo}: acurácia / F1", xlabel="época", ylim=(0, 1.02))

    for ax in axes:
        ax.axvline(melhor, color="gray", ls="--", lw=1, label=f"melhor época ({melhor})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)


def figura_curvas():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_curva("p0b_resnet18", "(b) ResNet-18 ImageNet", axes[0])
    plot_curva("p0a_smallcnn", "(a) SmallCNN do zero", axes[1])
    fig.tight_layout()
    fig.savefig(FIG / "p0_curvas.png", dpi=150)
    print("salvo:", FIG / "p0_curvas.png")

def figura_confusao(run="p0b_resnet18"):
    """Matriz de confusão no test_public, normalizada por linha (cada linha = recall da classe)."""
    p = pd.read_csv(RUNS / run / "preds_test.csv")
    classes = data.load_classes()
    cm = confusion_matrix(p["label"], p["pred"], labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, cbar_kws={"label": "fração da linha"},
                annot_kws={"fontsize": 7}, ax=ax)
    ax.set(xlabel="predito", ylabel="verdadeiro",
           title=f"(b) ResNet-18, 105 rótulos: matriz de confusão no test_public (n={cm.sum()})")
    fig.tight_layout()
    fig.savefig(FIG / "p0b_confusao.png", dpi=150)
    print("salvo:", FIG / "p0b_confusao.png")

    recall = pd.Series(np.diag(cm_norm), index=classes).sort_values()
    print("\nrecall por classe (piores primeiro):")
    print(recall.round(2).to_string())



def figura_exemplos():
    """Uma imagem rotulada de cada classe (rotulos_5), para apresentar o problema."""
    from PIL import Image

    d = data.load_split("rotulos_5").drop_duplicates("label").sort_values("label")
    fig, axs = plt.subplots(3, 7, figsize=(21, 7))
    for ax, (_, r) in zip(axs.flat, d.iterrows()):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(r["label"], fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "p0_exemplos_classes.png", dpi=100)
    print("salvo:", FIG / "p0_exemplos_classes.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    figura_curvas()
    figura_confusao()
    figura_exemplos()
