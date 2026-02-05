import torch
import torch.nn.functional as F


def _ensure_tensor(x, like=None):
    if torch.is_tensor(x):
        return x
    if like is not None:
        return torch.tensor(x, device=like.device, dtype=like.dtype)
    return torch.tensor(x, dtype=torch.float32)


def distance_func(name, x1, x2, eps: float = 0.0):
    if name == "l1":
        ax = 1
        return l1_dist(x1, x2, ax, eps)
    if name == "l2":
        ax = 1
        return l2_dist(x1, x2, ax, eps)
    if name == "cosine":
        ax = -1
        return cosine_dist(x1, x2, ax, eps)
    raise ValueError(f"Unknown distance function: {name}")


def l1_dist(x1, x2, ax: int, eps: float = 0.0):
    x1_t = _ensure_tensor(x1)
    x2_t = _ensure_tensor(x2, like=x1_t)
    return torch.sum(torch.abs(x1_t - x2_t), dim=ax) + eps


def l2_dist(x1, x2, ax: int, eps: float = 0.0):
    x1_t = _ensure_tensor(x1)
    x2_t = _ensure_tensor(x2, like=x1_t)
    return torch.sqrt(torch.sum((x1_t - x2_t) ** 2, dim=ax) + eps)


def cosine_dist(x1, x2, ax: int, eps: float = 0.0):
    x1_t = _ensure_tensor(x1)
    x2_t = _ensure_tensor(x2, like=x1_t)
    normalize_x1 = F.normalize(x1_t, p=2, dim=1)
    normalize_x2 = F.normalize(x2_t, p=2, dim=1)
    dist = 1.0 - torch.sum(normalize_x1 * normalize_x2, dim=ax)
    return dist + eps
