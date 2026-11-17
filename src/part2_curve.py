"""Parte 2 — curva de orçamento (lê os runs; não treina nada).

    uv run python src/part2_curve.py

Gera:
  curva_orcamento.csv                 (entregável: método, orçamento, seed, F1 macro, acurácia)
  figures/p2_curva_orcamento.png      (F1 macro no test_public × nº de rótulos, eixo x log)
  analysis/p2_ganho_por_classe.csv    (F1 por classe no orçamento 5, média das 3 seeds)
  figures/p2_ganho_por_classe.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import data

RUNS = Path("runs")
TETO = 0.686  # ResNet-18 com todos os rótulos (dado pelo monitor, enunciado)
N_CLASSES = 21

# (método, orçamento, seed) -> pasta em runs/
RUN_MAP = {
    ("piso", 1, 0): "piso_b1_s0",
    ("piso", 5, 0): "p0b_resnet18",
    ("piso", 5, 1): "piso_b5_s1",
    ("piso", 5, 2): "piso_b5_s2",
    ("piso", 20, 0): "piso_b20_s0",
    ("fixmatch_flex", 1, 0): "fm_flex_b1_s0",
    ("fixmatch_flex", 5, 0): "fm_tau0.95_flex_ls05r30",
    ("fixmatch_flex", 5, 1): "fm_flex_b5_s1",
    ("fixmatch_flex", 5, 2): "fm_flex_b5_s2",
    ("fixmatch_flex", 20, 0): "fm_flex_b20_s0",
}
NOMES = {"piso": "piso (supervisionado)", "fixmatch_flex": "FixMatch flex + ajuste"}


def tabela_runs():
    linhas = []
    for (metodo, budget, seed), run in RUN_MAP.items():
        f = RUNS / run / "summary.json"
        if not f.exists():
            print(f"faltando: {run}")
            continue
        s = json.loads(f.read_text())
        linhas.append({"metodo": metodo, "orcamento": budget, "n_rotulos": budget * N_CLASSES,
                       "seed": seed, "f1_macro": s["test_f1_macro"], "acuracia": s["test_acc"],
                       "val_f1": s["val_f1"], "run": run})
    return pd.DataFrame(linhas)


def figura_curva(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metodo, g in df.groupby("metodo"):
        agg = g.groupby("n_rotulos")["f1_macro"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["n_rotulos"], agg["mean"], yerr=agg["std"].fillna(0), marker="o",
                    capsize=4, label=NOMES[metodo])
        ax.scatter(g["n_rotulos"], g["f1_macro"], s=12, alpha=0.4)  # seeds individuais
    ax.axhline(TETO, color="gray", ls="--", label=f"teto (todos os rótulos) = {TETO}")
    ax.set_xscale("log")
    ax.set_xticks([21, 105, 420], labels=["21\n(1/classe)", "105\n(5/classe)", "420\n(20/classe)"])
    ax.minorticks_off()
    ax.set(xlabel="nº de rótulos (escala log)", ylabel="F1 macro no test_public", ylim=(0, 0.8),
           title="Curva de orçamento (orçamento 5: média ± desvio de 3 seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/p2_curva_orcamento.png", dpi=150)
    plt.close(fig)


def f1_por_classe(run, classes):
    p = pd.read_csv(RUNS / run / "preds_test.csv")
    return f1_score(p["label"], p["pred"], labels=classes, average=None, zero_division=0)


def ganho_por_classe(classes):
    """F1 por classe no orçamento 5, média das 3 seeds de cada método."""
    medias = {}
    for metodo in ["piso", "fixmatch_flex"]:
        runs = [RUN_MAP[(metodo, 5, s)] for s in range(3)
                if (RUNS / RUN_MAP[(metodo, 5, s)] / "preds_test.csv").exists()]
        medias[metodo] = np.mean([f1_por_classe(r, classes) for r in runs], axis=0)
    g = pd.DataFrame({"classe": classes, "f1_piso": medias["piso"],
                      "f1_fixmatch": medias["fixmatch_flex"]})
    g["ganho"] = g["f1_fixmatch"] - g["f1_piso"]
    g = g.sort_values("ganho", ascending=False)
    g.to_csv("analysis/p2_ganho_por_classe.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ["tab:green" if v > 0 else "tab:red" for v in g["ganho"]]
    ax.barh(g["classe"], g["ganho"], color=cores)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="ganho de F1 (FixMatch − piso), média de 3 seeds",
           title="Orçamento 5: ganho por classe no test_public")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/p2_ganho_por_classe.png", dpi=150)
    plt.close(fig)
    return g


def main():
    Path("figures").mkdir(exist_ok=True)
    Path("analysis").mkdir(exist_ok=True)
    df = tabela_runs()
    df[["metodo", "orcamento", "seed", "f1_macro", "acuracia"]].to_csv("curva_orcamento.csv", index=False)

    resumo = (df.groupby(["metodo", "orcamento"])
                .agg(n_seeds=("seed", "count"), f1_media=("f1_macro", "mean"),
                     f1_desvio=("f1_macro", "std"), acc_media=("acuracia", "mean"),
                     val_f1_media=("val_f1", "mean"))
                .round(3))
    print(resumo.to_string())

    figura_curva(df)
    g = ganho_por_classe(data.load_classes())
    print("\nganho por classe (orçamento 5, média das seeds):")
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
