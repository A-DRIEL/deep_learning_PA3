
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import torch

IMG_SIZE = (224, 320)  # (altura, largura)
MEAN = (0.485, 0.456, 0.406)  # estatísticas do ImageNet
STD = (0.229, 0.224, 0.225)

def train_transform(size=IMG_SIZE):
    h, w = size
    return T.Compose([
        T.ToImage(),
        # escala >= 0.5: raramente corta o animal fora
        # ratio em torno de w/h: o crop mantém o formato retangular
        T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.8 * w / h, 1.25 * w / h), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)], p=0.8),
        T.RandomGrayscale(p=0.2),  # imita fotos IR
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])

def eval_transform(size=IMG_SIZE):
    # val/test/predict: nada aleatório, só redimensiona
    return T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])

DATA_ROOT = Path("data/pa3_dados")  # relativo à raiz do projeto


def load_classes(root=DATA_ROOT):
    """Lista das 21 classes, na ordem do índice (posição na lista = número da classe)."""
    return pd.read_csv(root / "classes.csv").sort_values("idx")["label"].tolist()


def load_split(name, root=DATA_ROOT):
    """Lê um conjunto e devolve DataFrame com image_id, path e, se rotulado, label e y.

    name: 'rotulos_1' | 'rotulos_5' | 'rotulos_20' | 'val' | 'test_public' | 'train'
    'train' = pool inteiro, SEM rótulo.
    """
    if name == "train":
        paths = sorted((root / "images" / "train").glob("*.jpg"))
        return pd.DataFrame({"image_id": [p.stem for p in paths], "path": [str(p) for p in paths]})

    df = pd.read_csv(root / "labels" / f"{name}.csv")
    # rotulos_K são subconjuntos do train: as imagens moram em images/train/
    pasta = "train" if name.startswith("rotulos") else name
    df["path"] = [str(root / "images" / pasta / f"{i}.jpg") for i in df["image_id"]]

    cls2idx = {c: i for i, c in enumerate(load_classes(root))}
    df["y"] = df["label"].map(cls2idx)
    return df


class ImageDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = list(paths)
        self.labels = None if labels is None else list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # sem rótulo (pool do train): devolve -1, porque o DataLoader não aceita None
        y = -1 if self.labels is None else int(self.labels[i])
        return img, y