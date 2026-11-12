import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed):
    """Fixa as fontes de aleatoriedade (pesos iniciais, embaralhamento, aumentações)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(device):
    """Precisão mista (bfloat16) na GPU: mais rápido e usa menos memória. Na CPU, desligado."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Uma passada por todo o conjunto de treino. Devolve (loss média, acurácia)."""
    model.train()  # liga Dropout e o modo de treino do BatchNorm
    total_loss, acertos, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast(device):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)

        optimizer.zero_grad()  # zera gradientes do passo anterior
        loss.backward()        # calcula gradientes
        optimizer.step()       # atualiza os pesos
        scheduler.step()       # ajusta o learning rate (a cada passo, não a cada época)

        total_loss += loss.item() * len(y)
        acertos += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, acertos / n


@torch.no_grad()  # avaliação: não precisa de gradientes
def predict(model, loader, device):
    """Devolve (probabilidades [N, C], rótulos [N]) como arrays numpy."""
    model.eval()  # desliga Dropout; BatchNorm usa as estatísticas aprendidas
    probs, labels = [], []
    for x, y in loader:
        with autocast(device):
            logits = model(x.to(device))
        probs.append(logits.float().softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def compute_metrics(probs, labels, num_classes):
    pred = probs.argmax(1)
    # loss no conjunto avaliado: -log(prob dada à classe certa), em média
    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
    return {
        "loss": float(loss),
        "acc": float(accuracy_score(labels, pred)),
        "f1_macro": float(f1_score(labels, pred, average="macro",
                                   labels=list(range(num_classes)), zero_division=0)),
    }
