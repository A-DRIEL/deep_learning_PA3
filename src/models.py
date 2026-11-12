import timm
import torch.nn as nn

BACKBONE = "resnet18.tv_in1k"  


def conv_block(c_in, c_out):
    """Duas convoluções 3x3 + max pooling: extrai features e reduz a resolução pela metade."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Parte 0 (a): CNN pequena treinada do zero (~1,2M parâmetros)."""

    def __init__(self, num_classes, width=32, dropout=0.3):
        super().__init__()
        w = width
        # 224x320 -> 112x160 -> 56x80 -> 28x40 -> 14x20 -> 7x10
        self.features = nn.Sequential(
            conv_block(3, w),
            conv_block(w, 2 * w),
            conv_block(2 * w, 4 * w),
            conv_block(4 * w, 8 * w),
            conv_block(8 * w, 8 * w),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling: qualquer HxW -> 1x1
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(8 * w, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_model(name, num_classes, pretrained=True):
    """'smallcnn' -> CNN do zero; qualquer outro nome -> modelo do timm."""
    if name == "smallcnn":
        return SmallCNN(num_classes)
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def count_params(model):
    """Conta exatamente como o enunciado pede (limite: 12M)."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import torch

    for nome in ["smallcnn", BACKBONE]:
        m = build_model(nome, num_classes=21)
        saida = m(torch.randn(2, 3, 224, 320))  # lote falso: 2 imagens 224x320
        print(f"{nome:20s} params={count_params(m):>11,}  saída={tuple(saida.shape)}")

