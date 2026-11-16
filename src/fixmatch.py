"""Trilha B: FixMatch com 105 rótulos (orçamento 5) + train inteiro como pool não rotulado.

Exemplos (a partir da raiz do projeto):

    # FixMatch "puro" (sem correção de desbalanceamento)
    uv run python src/fixmatch.py --budget 5 --tau 0.95 --correction none --name fm_tau095_none

    # com correção: alinhamento de distribuição (da) ou limiar por classe (flex)
    uv run python src/fixmatch.py --budget 5 --tau 0.95 --correction da   --name fm_tau095_da
    uv run python src/fixmatch.py --budget 5 --tau 0.95 --correction flex --name fm_tau095_flex

Saídas em runs/<name>/: history.csv, pseudo_stats.csv, preds_test.csv, model.pt, summary.json
(mesmo formato do train.py, então o predict.py da entrega funciona com qualquer um dos dois).

Regras do enunciado respeitadas:
  * val  -> seleção de modelo (melhor passo) e escolha de τ / correção.
  * test_public -> lido UMA vez, no fim, com os pesos do melhor passo do val.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, RandomSampler

import data
import models
from engine import autocast, compute_metrics, get_device, predict, set_seed


# ----------------------------------------------------------------------------
# Argumentos
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=models.BACKBONE)
    p.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    p.add_argument("--steps", type=int, default=3000, help="passos de otimização (FixMatch conta em passos)")
    p.add_argument("--batch-size", type=int, default=32, help="B: lote rotulado")
    p.add_argument("--mu", type=int, default=2, help="lote não rotulado = mu * B")
    p.add_argument("--tau", type=float, default=0.95, help="limiar de confiança do pseudo-rótulo")
    p.add_argument("--lambda-u", type=float, default=1.0, help="peso da perda não supervisionada")
    p.add_argument("--correction", default="none", choices=["none", "da", "flex"],
                   help="none | da (alinhamento de distribuição) | flex (limiar por classe)")
    p.add_argument("--da-momentum", type=float, default=0.99)
    p.add_argument("--flex-momentum", type=float, default=0.99)
    p.add_argument("--tau-floor", type=float, default=0.5, help="menor limiar que o 'flex' pode dar a uma classe")
    p.add_argument("--lr", type=float, default=1e-3, help="lr da cabeça")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05, help="fração dos passos com warmup linear")
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2, help="use 0 se faltar RAM (8 GB)")
    p.add_argument("--no-pretrained", action="store_true", help="só para depuração offline")
    p.add_argument("--name", required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Peças do método
# ----------------------------------------------------------------------------
class EMA:
    """Média móvel exponencial dos pesos. É o modelo EMA que é avaliado e entregue: ele
    suaviza o ruído de treinar em pseudo-rótulos. O decay efetivo começa pequeno e sobe
    (min(decay, (1+t)/(10+t))), senão o EMA ficaria preso nos pesos iniciais em treino curto."""

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.t = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.mul_(d).add_(m.detach(), alpha=1 - d)
        for eb, mb in zip(self.model.buffers(), model.buffers()):  # BatchNorm: copia estatísticas
            eb.copy_(mb)


class DistAlign:
    """Alinhamento de distribuição (ReMixMatch). Mantém q = média móvel das probabilidades
    previstas no pool e reescala cada previsão por (alvo / q). Se o modelo despeja tudo em
    'bear', q_bear fica alto e a probabilidade de 'bear' é reduzida; as raras sobem.
    Alvo = uniforme: é a distribuição dos 105 rótulos (5 por classe)."""

    def __init__(self, num_classes, momentum, device):
        self.q = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.target = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.m = momentum

    @torch.no_grad()
    def __call__(self, probs):
        self.q = self.m * self.q + (1 - self.m) * probs.mean(0)
        p = probs * (self.target / self.q.clamp_min(1e-6))
        return p / p.sum(1, keepdim=True)


class ClassThresholds:
    """Limiar por classe (ideia do FlexMatch). σ_c = média móvel de quantas imagens do pool
    foram previstas como c com confiança >= τ ('status de aprendizado' de c).
    β_c = σ_c / max σ ; τ_c = τ * β_c / (2 - β_c), limitado a [tau_floor, τ].
    Classe bem aprendida (β→1) mantém τ; classe rara (β→0) ganha limiar baixo e passa
    a contribuir com pseudo-rótulos."""

    def __init__(self, num_classes, tau, floor, momentum, device):
        self.sigma = torch.zeros(num_classes, device=device)
        self.tau, self.floor, self.m = tau, floor, momentum
        self.C = num_classes

    @torch.no_grad()
    def update(self, conf, pred):
        counts = torch.bincount(pred[conf >= self.tau], minlength=self.C).float()
        self.sigma = self.m * self.sigma + (1 - self.m) * counts

    @torch.no_grad()
    def thresholds(self):
        if self.sigma.max() <= 0:  # nada confiante ainda: limiar cheio para todos
            return torch.full((self.C,), self.tau, device=self.sigma.device)
        beta = self.sigma / self.sigma.max()
        return (self.tau * beta / (2.0 - beta)).clamp(min=self.floor, max=self.tau)


def make_optimizer(model, args):
    """Mesma receita da Parte 0: AdamW, cabeça com lr cheio e backbone com lr * 0.1."""
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": head, "lr": args.lr},
        {"params": backbone, "lr": args.lr * args.backbone_lr_mult},
    ], weight_decay=args.weight_decay)


def make_scheduler(optimizer, total_steps, warmup_frac):
    """Warmup linear + cosseno até 0, por passo."""
    warmup = max(1, int(warmup_frac * total_steps))

    def fator(step):
        if step < warmup:
            return (step + 1) / warmup
        progresso = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progresso))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fator)


def infinite_loader(dataset, batch_size, n_batches, workers):
    """Amostra COM reposição exatamente n_batches lotes (o pool tem 6.505 imagens, mas o
    treino dura `steps` passos, não épocas). Evita itertools.cycle, que guardaria tudo em RAM."""
    sampler = RandomSampler(dataset, replacement=True, num_samples=n_batches * batch_size)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                      num_workers=workers, persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


# ----------------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    classes = data.load_classes()
    C = len(classes)
    lab = data.load_split(f"rotulos_{args.budget}")
    pool = data.load_split("train")  # TODOS os 6.505, sem rótulo (inclui as rotuladas, como no FixMatch)
    va = data.load_split("val")
    te = data.load_split("test_public")
    B, uB = args.batch_size, args.batch_size * args.mu

    # ramo rotulado: MESMA aumentação do piso (Parte 0) -> comparação justa
    lab_loader = infinite_loader(data.ImageDataset(lab["path"], lab["y"], data.train_transform()),
                                 B, args.steps, args.workers)
    # ramo não rotulado: (fraca, forte) da mesma imagem
    unl_loader = infinite_loader(data.TwoViewDataset(pool["path"], data.weak_transform(), data.strong_transform()),
                                 uB, args.steps, args.workers)
    va_loader = DataLoader(data.ImageDataset(va["path"], va["y"], data.eval_transform()), batch_size=64)
    te_loader = DataLoader(data.ImageDataset(te["path"], te["y"], data.eval_transform()), batch_size=64)

    model = models.build_model(args.model, C, pretrained=not args.no_pretrained).to(device)
    n_params = models.count_params(model)
    print(f"{args.model}: {n_params:,} parâmetros | rotuladas={len(lab)} pool={len(pool)} "
          f"val={len(va)} | B={B} uB={uB} | τ={args.tau} correção={args.correction} | {device}")

    ema = EMA(model, args.ema_decay)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args.steps, args.warmup_frac)
    da = DistAlign(C, args.da_momentum, device) if args.correction == "da" else None
    flex = ClassThresholds(C, args.tau, args.tau_floor, args.flex_momentum, device) \
        if args.correction == "flex" else None

    # acumuladores (zerados a cada avaliação) para o diagnóstico de viés
    n_pred = torch.zeros(C, device=device)  # quantas imagens do pool foram previstas como c
    n_acc = torch.zeros(C, device=device)   # quantas dessas passaram do limiar (viraram pseudo-rótulo)
    sup_sum = uns_sum = mask_sum = 0.0
    n_iter = 0
    historico, pseudo_rows = [], []
    melhor = {"val_f1": -1.0, "step": 0}
    melhores_pesos = None
    t0 = time.time()

    for step, ((xl, yl), (xw, xs)) in enumerate(zip(lab_loader, unl_loader), start=1):
        model.train()
        # UM forward com os três blocos juntos: o BatchNorm vê rotulado + fraca + forte
        # (evita que as estatísticas de BN venham só de um tipo de imagem).
        x = torch.cat([xl, xw, xs]).to(device, non_blocking=True)
        yl = yl.to(device)
        with autocast(device):
            logits = model(x)
        logits_l, logits_w, logits_s = logits.float().split([B, uB, uB])

        # (1) perda supervisionada
        loss_sup = F.cross_entropy(logits_l, yl)

        # (2) pseudo-rótulo a partir da visão FRACA (sem gradiente)
        with torch.no_grad():
            probs = logits_w.softmax(1)
            if da is not None:
                probs = da(probs)
            conf, pl = probs.max(1)
            if flex is not None:
                flex.update(conf, pl)
                thr = flex.thresholds()[pl]  # limiar da classe prevista de cada imagem
            else:
                thr = args.tau
            mask = (conf >= thr).float()

        # (3) perda não supervisionada: a visão FORTE tem de prever o pseudo-rótulo.
        # Média sobre TODAS as uB imagens (as rejeitadas contribuem 0), como no paper.
        loss_uns = (F.cross_entropy(logits_s, pl, reduction="none") * mask).mean()

        loss = loss_sup + args.lambda_u * loss_uns
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)

        with torch.no_grad():
            n_pred += torch.bincount(pl, minlength=C).float()
            n_acc += torch.bincount(pl[mask.bool()], minlength=C).float()
        sup_sum += loss_sup.item()
        uns_sum += loss_uns.item()
        mask_sum += mask.mean().item()
        n_iter += 1

        # ---- avaliação no val com o modelo EMA (seleção do melhor passo: SÓ pelo val) ----
        if step % args.eval_every == 0 or step == args.steps:
            vm = compute_metrics(*predict(ema.model, va_loader, device), C)
            historico.append({"step": step, "sup_loss": sup_sum / n_iter, "unsup_loss": uns_sum / n_iter,
                              "mask_rate": mask_sum / n_iter, "val_loss": vm["loss"],
                              "val_acc": vm["acc"], "val_f1": vm["f1_macro"]})
            for c in range(C):
                pseudo_rows.append({"step": step, "class": classes[c],
                                    "n_pred": int(n_pred[c]), "n_accepted": int(n_acc[c])})
            if vm["f1_macro"] > melhor["val_f1"]:
                melhor = {"step": step, "val_f1": vm["f1_macro"], "val_acc": vm["acc"]}
                melhores_pesos = {k: v.detach().clone() for k, v in ema.model.state_dict().items()}
            marca = "  *" if melhor["step"] == step else ""
            print(f"step {step:5d}/{args.steps} | sup {sup_sum/n_iter:.3f} uns {uns_sum/n_iter:.3f} "
                  f"máscara {mask_sum/n_iter:.2f} | val acc {vm['acc']:.3f} F1 {vm['f1_macro']:.3f}{marca} "
                  f"| {time.time()-t0:.0f}s")
            n_pred.zero_(); n_acc.zero_()
            sup_sum = uns_sum = mask_sum = 0.0
            n_iter = 0

    pd.DataFrame(historico).to_csv(out / "history.csv", index=False)
    pd.DataFrame(pseudo_rows).to_csv(out / "pseudo_stats.csv", index=False)

    # ---- test_public: UMA vez, com os pesos EMA do melhor passo no val ----
    ema.model.load_state_dict(melhores_pesos)
    probs_te, y_te = predict(ema.model, te_loader, device)
    tm = compute_metrics(probs_te, y_te, C)
    f1_por_classe = f1_score(y_te, probs_te.argmax(1), average=None, labels=list(range(C)), zero_division=0)

    torch.save({"model": args.model, "classes": classes, "img_size": data.IMG_SIZE,
                "state_dict": melhores_pesos}, out / "model.pt")
    pd.DataFrame({"image_id": te["image_id"], "label": te["label"],
                  "pred": [classes[i] for i in probs_te.argmax(1)], "conf": probs_te.max(1)}
                 ).to_csv(out / "preds_test.csv", index=False)

    resumo = {"name": args.name, "method": "fixmatch", "model": args.model, "budget": args.budget,
              "seed": args.seed, "tau": args.tau, "correction": args.correction, "params": n_params,
              "best_step": melhor["step"], "val_f1": melhor["val_f1"],
              "test_f1_macro": tm["f1_macro"], "test_acc": tm["acc"],
              "test_f1_per_class": dict(zip(classes, map(float, f1_por_classe))),
              "final_class_thresholds": None if flex is None else dict(zip(classes, flex.thresholds().tolist())),
              "final_da_q": None if da is None else dict(zip(classes, da.q.tolist())),
              "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(resumo, indent=2))

    print(f"\nmelhor passo (val) = {melhor['step']}  val F1 = {melhor['val_f1']:.3f}")
    print(f"test_public: F1 macro = {tm['f1_macro']:.3f}  acc = {tm['acc']:.3f}")


if __name__ == "__main__":
    main()
