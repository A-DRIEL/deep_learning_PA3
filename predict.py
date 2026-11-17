"""Classifica uma pasta de imagens com o modelo entregue.

    uv run python predict.py --imagens PASTA --saida preds.csv [--pesos entrega/model.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))  # nosso código mora em src/
import data    # noqa: E402
import models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imagens", required=True, help="pasta com as imagens (.jpg/.jpeg/.png)")
    p.add_argument("--saida", required=True, help="CSV de saída com colunas image_id,label")
    p.add_argument("--pesos", default="entrega/model.pt")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    ckpt = torch.load(args.pesos, map_location="cpu")
    classes = ckpt["classes"]
    # pretrained=False: só monta a arquitetura; os pesos vêm do checkpoint (não baixa nada)
    model = models.build_model(ckpt["model"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"modelo: {ckpt['model']} | parâmetros: {models.count_params(model):,}")

    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(p for p in Path(args.imagens).iterdir() if p.suffix.lower() in exts)
    loader = DataLoader(data.ImageDataset(paths, transform=data.eval_transform()),
                        batch_size=args.batch_size)

    preds = []
    with torch.inference_mode():
        for x, _ in loader:
            preds += model(x).argmax(1).tolist()

    pd.DataFrame({"image_id": [p.stem for p in paths],
                  "label": [classes[i] for i in preds]}).to_csv(args.saida, index=False)
    print(f"{len(paths)} imagens -> {args.saida} em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
