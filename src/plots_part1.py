"""Figura do colapso do FixMatch (Parte 1): máscara, val loss e val F1 ao longo dos passos.

    uv run python src/plots_part1.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("runs")
PISO_VAL_F1 = 0.415

CURVAS = {
    "fm_tau0.8_none": "τ=0,8, sem ajuste",
    "fm_tau0.95_none": "τ=0,95, sem ajuste",
    "fm_tau0.95_none_ls05r30": "τ=0,95 + ajuste",
    "fm_tau0.95_flex_ls05r30": "τ=0,95 + ajuste + flex",
}


def main():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for run, nome in CURVAS.items():
        h = pd.read_csv(RUNS / run / "history.csv")
        axs[0].plot(h["step"], h["mask_rate"], marker=".", label=nome)
        axs[1].plot(h["step"], h["val_loss"], marker=".", label=nome)
        axs[2].plot(h["step"], h["val_f1"], marker=".", label=nome)
    axs[2].axhline(PISO_VAL_F1, color="gray", ls="--", label="piso (val F1 0,415)")
    titulos = ["máscara (fração do pool aceita)", "val loss", "val F1 macro"]
    for ax, t in zip(axs, titulos):
        ax.set(title=t, xlabel="passo")
        ax.grid(alpha=0.3)
    axs[0].set_ylim(0, 1.02)
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/p1_colapso.png", dpi=150)
    print("salvo: figures/p1_colapso.png")


if __name__ == "__main__":
    main()
