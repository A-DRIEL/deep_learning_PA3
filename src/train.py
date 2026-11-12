"""Treino supervisionado só com os rótulos do orçamento (Parte 0 e piso da Parte 2).

    uv run python src/train.py --model resnet18.tv_in1k --epochs 60 --name p0b_resnet18
    uv run python src/train.py --model smallcnn --epochs 150 --name p0a_smallcnn
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import data
import models
from engine import compute_metrics, get_device, predict, set_seed, train_one_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=models.BACKBONE)
    p.add_argument("--budget", type=int, default=5, choices=[1, 5, 20])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="lr da cabeça")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="lr do backbone = lr * isto")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)  # 8 GB de RAM: carregar no processo principal
    p.add_argument("--name", required=True, help="nome da pasta em runs/")
    return p.parse_args()


def make_optimizer(model, args):
    """Cabeça com lr cheio; backbone pré-treinado com lr menor."""
    if args.model == "smallcnn":  # tudo aleatório: um lr só
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([
        {"params": head, "lr": args.lr},
        {"params": backbone, "lr": args.lr * args.backbone_lr_mult},
    ], weight_decay=args.weight_decay)


def make_scheduler(optimizer, args, steps_per_epoch):
    """Warmup linear + decaimento em cosseno, multiplicando o lr de cada grupo."""
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def fator(step):
        if step < warmup:
            return (step + 1) / warmup
        progresso = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progresso))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fator)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    classes = data.load_classes()
    C = len(classes)
    tr = data.load_split(f"rotulos_{args.budget}")
    va = data.load_split("val")
    te = data.load_split("test_public")

    tr_loader = DataLoader(data.ImageDataset(tr["path"], tr["y"], data.train_transform()),
                           batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, persistent_workers=args.workers > 0)
    # avaliação: sem embaralhar, sem aumentação, sem workers extras (economiza RAM)
    va_loader = DataLoader(data.ImageDataset(va["path"], va["y"], data.eval_transform()), batch_size=64)
    te_loader = DataLoader(data.ImageDataset(te["path"], te["y"], data.eval_transform()), batch_size=64)

    model = models.build_model(args.model, C, pretrained=True).to(device)
    n_params = models.count_params(model)
    print(f"{args.model}: {n_params:,} parâmetros | treino={len(tr)} val={len(va)} | {device}")

    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args, len(tr_loader))

    historico = []
    melhor = {"val_f1": -1.0}
    melhores_pesos = None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, scheduler, device)
        vm = compute_metrics(*predict(model, va_loader, device), C)
        historico.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_loss": vm["loss"], "val_acc": vm["acc"], "val_f1": vm["f1_macro"]})

        if vm["f1_macro"] > melhor["val_f1"]:  # seleção de época: SÓ pelo val
            melhor = {"epoch": epoch, "val_f1": vm["f1_macro"], "val_acc": vm["acc"]}
            melhores_pesos = {k: v.detach().clone() for k, v in model.state_dict().items()}

        marca = "  *" if melhor["epoch"] == epoch else ""
        print(f"ep {epoch:3d} | treino loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {vm['loss']:.3f} acc {vm['acc']:.3f} F1 {vm['f1_macro']:.3f}{marca}")

    pd.DataFrame(historico).to_csv(out / "history.csv", index=False)

    # ---- test_public: UMA vez, com os pesos da época escolhida no val ----
    model.load_state_dict(melhores_pesos)
    probs, y = predict(model, te_loader, device)
    tm = compute_metrics(probs, y, C)

    torch.save({"model": args.model, "classes": classes, "img_size": data.IMG_SIZE,
                "state_dict": melhores_pesos}, out / "model.pt")
    pd.DataFrame({"image_id": te["image_id"], "label": te["label"],
                  "pred": [classes[i] for i in probs.argmax(1)], "conf": probs.max(1)}
                 ).to_csv(out / "preds_test.csv", index=False)

    resumo = {"name": args.name, "model": args.model, "budget": args.budget, "seed": args.seed,
              "params": n_params, "best_epoch": melhor["epoch"], "val_f1": melhor["val_f1"],
              "test_f1_macro": tm["f1_macro"], "test_acc": tm["acc"], "args": vars(args)}
    (out / "summary.json").write_text(json.dumps(resumo, indent=2))

    print(f"\nmelhor época (val) = {melhor['epoch']}  val F1 = {melhor['val_f1']:.3f}")
    print(f"test_public: F1 macro = {tm['f1_macro']:.3f}  acc = {tm['acc']:.3f}")


if __name__ == "__main__": 
    main()
