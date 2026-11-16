from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
import torch

IMG_SIZE = (224, 320)  # (altura, largura)
MEAN = (0.485, 0.456, 0.406)  # estatísticas do ImageNet
STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Transformações
# ----------------------------------------------------------------------------
def train_transform(size=IMG_SIZE):
    """Aumentação supervisionada da Parte 0. Na Trilha B ela também é usada no ramo
    rotulado do FixMatch, para que o ÚNICO fator diferente do piso seja o dado sem rótulo."""
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


def weak_transform(size=IMG_SIZE, pad=16):
    """Aumentação FRACA do FixMatch (a que gera o pseudo-rótulo): flip + deslocamento de até
    `pad` pixels (padding refletido + crop). Quase não muda o conteúdo, então a previsão é
    confiável. Não usa RandomResizedCrop: com animal em <2% do quadro, zoom pode cortá-lo."""
    return T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size, padding=pad, padding_mode="reflect"),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def strong_transform(size=IMG_SIZE, pad=16, ra_ops=2, ra_mag=9, cutout_p=0.5):
    """Aumentação FORTE do FixMatch (a que tenta 'enganar' o aluno): a fraca + RandAugment
    + escala de cinza ocasional (imita IR) + Cutout (RandomErasing).
    RandAugment exige uint8, por isso vem ANTES do ToDtype; o Cutout atua depois do Normalize
    (apaga com 0 = cor média do ImageNet)."""
    return T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size, padding=pad, padding_mode="reflect"),
        T.RandAugment(num_ops=ra_ops, magnitude=ra_mag),
        T.RandomGrayscale(p=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=cutout_p, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
    ])


# ----------------------------------------------------------------------------
# Leitura dos conjuntos
# ----------------------------------------------------------------------------
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


class TwoViewDataset(Dataset):
    """Pool NÃO rotulado do FixMatch: lê a imagem UMA vez e devolve duas visões dela,
    (fraca, forte). O rótulo não existe: o pseudo-rótulo sai da visão fraca e é cobrado
    na forte."""

    def __init__(self, paths, weak, strong):
        self.paths = list(paths)
        self.weak, self.strong = weak, strong

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.weak(img), self.strong(img)