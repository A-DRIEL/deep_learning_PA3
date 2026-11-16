"""Diagnósticos obrigatórios da Trilha B (rodam DEPOIS dos treinos; não treinam nada).

    uv run python src/pseudolabel_diag.py --baseline p0b_resnet18 \
        --runs fm_tau095_none fm_tau095_da fm_tau095_flex --tau 0.95

(1) PRECISÃO E COBERTURA DOS PSEUDO-RÓTULOS vs τ, por classe.
    O "pseudo-rotulador" (modelo de cada run) é aplicado ao test_public COMO SE ele não
    tivesse rótulo: pseudo-rótulo = argmax, aceito se confiança >= τ. Só depois olhamos os
    rótulos reais para medir. É DIAGNÓSTICO: a escolha de τ foi feita no val.
      precisão_c(τ)  = dos aceitos com pseudo-rótulo c, fração em que o rótulo real é c
      cobertura_c(τ) = das imagens cuja classe REAL é c, fração que recebeu algum
                       pseudo-rótulo (certo ou errado) com conf >= τ
      cobertura útil_c(τ) = das imagens reais de c, fração aceita COM o pseudo-rótulo certo
                       (é o recall do pseudo-rotulador)

(2) VIÉS PARA CLASSES GRANDES.
    Compara a distribuição de classes dos pseudo-rótulos ACEITOS com a distribuição real,
    para o piso (Parte 0b) e para cada variante do FixMatch (sem correção, da, flex).

Usa os preds_test.csv que cada treino já salvou (label, pred, conf), então não precisa de GPU.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

import data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="runs do FixMatch (pastas em runs/)")
    p.add_argument("--baseline", default=None, help="run do piso (ex.: p0b_resnet18), opcional")
    p.add_argument("--tau", type=float, default=0.95, help="τ de referência para a análise de viés")
    p.add_argument("--taus", nargs="+", type=float,
                   default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99])
    p.add_argument("--rare-k", type=int, default=5, help="quantas classes menores contam como 'raras'")
    p.add_argument("--out", default="analysis")
    return p.parse_args()


def load_preds(run):
    return pd.read_csv(Path("runs") / run / "preds_test.csv")


def tau_table(df, classes, taus):
    """Tabela longa: (τ, classe) -> n_true, n_aceitos_como_c, n_corretos, precisão, coberturas."""
    rows = []
    for tau in taus:
        aceito = df["conf"].to_numpy() >= tau
        for c in classes + ["ALL"]:
            if c == "ALL":
                n_true = len(df)
                n_as_c = int(aceito.sum())
                n_ok = int((aceito & (df["pred"] == df["label"])).sum())
                cov = n_as_c / n_true
                cov_util = n_ok / n_true
            else:
                eh_c = (df["label"] == c).to_numpy()
                as_c = aceito & (df["pred"] == c).to_numpy()
                n_true = int(eh_c.sum())
                n_as_c = int(as_c.sum())
                n_ok = int((as_c & eh_c).sum())
                cov = (aceito & eh_c).sum() / max(n_true, 1)
                cov_util = n_ok / max(n_true, 1)
            rows.append({"tau": tau, "class": c, "n_true": n_true, "n_accepted_as_class": n_as_c,
                         "n_correct": n_ok,
                         "precision": n_ok / n_as_c if n_as_c > 0 else np.nan,
                         "coverage": cov, "useful_coverage": cov_util})
    return pd.DataFrame(rows)


def plot_precision_coverage(tab, run, out):
    g = tab[tab["class"] == "ALL"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(g["tau"], g["precision"], "o-", label="precisão")
    ax.plot(g["tau"], g["coverage"], "s-", label="cobertura")
    ax.set_xlabel("τ"); ax.set_ylabel("fração"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Pseudo-rótulos no test_public — {run}"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / f"prec_cov_global_{run}.png", dpi=150); plt.close(fig)


def plot_heatmaps(tab, classes, run, out):
    """Duas heatmaps classes x τ (classes ordenadas da maior para a menor)."""
    ordem = tab[(tab["class"] != "ALL")].groupby("class")["n_true"].first().sort_values(ascending=False).index
    fig, axs = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, col, titulo in zip(axs, ["precision", "coverage"], ["Precisão", "Cobertura"]):
        m = tab[tab["class"] != "ALL"].pivot(index="class", columns="tau", values=col).loc[ordem]
        m.index = [f"{c} (n={int(tab[(tab['class']==c)]['n_true'].iloc[0])})" for c in m.index]
        sns.heatmap(m, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax,
                    annot_kws={"size": 7}, cbar=False)
        ax.set_title(f"{titulo} por classe vs τ — {run}"); ax.set_xlabel("τ")
    fig.tight_layout(); fig.savefig(out / f"prec_cov_heatmap_{run}.png", dpi=150); plt.close(fig)


def kl(p, q, eps=1e-9):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def bias_analysis(preds, classes, tau, rare_k, out):
    """preds: {nome: DataFrame}. Gera bias_table.csv, bias_summary.csv e a figura de barras."""
    # classes ordenadas pela frequência REAL no test_public (proxy da distribuição do pool)
    ref = next(iter(preds.values()))
    freq_real = ref["label"].value_counts().reindex(classes).fillna(0)
    ordem = list(freq_real.sort_values(ascending=False).index)
    raras = ordem[-rare_k:]
    share_real = (freq_real / freq_real.sum()).reindex(ordem)

    tabela = pd.DataFrame({"class": ordem, "share_true": share_real.values})
    resumo = []
    for nome, df in preds.items():
        aceito = df["conf"] >= tau
        cont = df[aceito]["pred"].value_counts().reindex(ordem).fillna(0)
        share_acc = cont / max(cont.sum(), 1)
        f1c = f1_score(df["label"], df["pred"], average=None, labels=ordem, zero_division=0)
        tabela[f"share_accepted__{nome}"] = share_acc.values
        tabela[f"f1__{nome}"] = f1c
        rec_raras = np.mean([((df["label"] == c) & aceito & (df["pred"] == c)).sum() /
                             max((df["label"] == c).sum(), 1) for c in raras])
        resumo.append({"run": nome, "tau": tau,
                       "coverage": float(aceito.mean()),
                       "precision": float((df[aceito]["pred"] == df[aceito]["label"]).mean()) if aceito.any() else np.nan,
                       "KL(aceitos||real)": kl(share_acc.values, share_real.values),
                       "top3_share_accepted": float(share_acc.iloc[:3].sum()),
                       "top3_share_true": float(share_real.iloc[:3].sum()),
                       f"useful_coverage_rare{rare_k}": float(rec_raras),
                       f"f1_rare{rare_k}": float(np.mean(f1c[-rare_k:])),
                       "f1_macro": float(f1_score(df["label"], df["pred"], average="macro",
                                                  labels=classes, zero_division=0))})
    tabela.to_csv(out / "bias_table.csv", index=False)
    pd.DataFrame(resumo).to_csv(out / "bias_summary.csv", index=False)

    # figura: distribuição real x aceita, por run
    n = len(preds)
    fig, axs = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True, squeeze=False)
    x = np.arange(len(ordem))
    for ax, nome in zip(axs[:, 0], preds):
        ax.bar(x - 0.2, tabela["share_true"], 0.4, label="real (test_public)", color="#999")
        ax.bar(x + 0.2, tabela[f"share_accepted__{nome}"], 0.4, label=f"pseudo-rótulos aceitos (τ={tau})")
        ax.set_ylabel("fração"); ax.set_title(nome, fontsize=10); ax.legend(fontsize=8)
    axs[-1, 0].set_xticks(x); axs[-1, 0].set_xticklabels(ordem, rotation=60, ha="right")
    fig.tight_layout(); fig.savefig(out / "bias_distribution.png", dpi=150); plt.close(fig)
    return pd.DataFrame(resumo), tabela, ordem


def plot_training_pool_bias(runs, classes, ordem, out):
    """Viés NO TREINO: soma, ao longo do treino, dos pseudo-rótulos aceitos por classe no
    pool não rotulado (pseudo_stats.csv). Compara com a distribuição real como proxy."""
    existem = [r for r in runs if (Path("runs") / r / "pseudo_stats.csv").exists()]
    if not existem:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(ordem)); w = 0.8 / len(existem)
    for i, r in enumerate(existem):
        ps = pd.read_csv(Path("runs") / r / "pseudo_stats.csv")
        acc = ps.groupby("class")["n_accepted"].sum().reindex(ordem).fillna(0)
        ax.bar(x + i * w, acc / max(acc.sum(), 1), w, label=r)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ordem, rotation=60, ha="right")
    ax.set_ylabel("fração dos pseudo-rótulos usados no treino")
    ax.set_title("Classes dos pseudo-rótulos realmente usados no treino (pool não rotulado)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "bias_training_pool.png", dpi=150); plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True, parents=True)
    classes = data.load_classes()
    nomes = ([args.baseline] if args.baseline else []) + args.runs
    preds = {n: load_preds(n) for n in nomes}

    # (1) precisão/cobertura vs τ, por classe
    for nome, df in preds.items():
        tab = tau_table(df, classes, args.taus)
        tab.to_csv(out / f"prec_cov_{nome}.csv", index=False)
        plot_precision_coverage(tab, nome, out)
        plot_heatmaps(tab, classes, nome, out)
        g = tab[(tab["class"] == "ALL")][["tau", "precision", "coverage"]]
        print(f"\n== {nome}: precisão/cobertura globais no test_public ==\n{g.round(3).to_string(index=False)}")

    # (2) viés para classes grandes
    resumo, tabela, ordem = bias_analysis(preds, classes, args.tau, args.rare_k, out)
    plot_training_pool_bias(args.runs, classes, ordem, out)
    print(f"\n== Viés (τ={args.tau}) ==\n{resumo.round(3).to_string(index=False)}")
    print(f"\nArquivos em {out}/")


if __name__ == "__main__":
    main()
