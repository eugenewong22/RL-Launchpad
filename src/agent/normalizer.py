"""Running observation normalizer (OpenAI-baselines-HER style).

Sparse-reward DDPG-family training on Fetch is known to destabilize on
raw observations (positions ~1.3, velocities ~0.01 in the same vector):
the critic surface degenerates and the actor saturates at the action
bounds. Normalizing inputs to ~N(0,1), clipped to [-5, 5], is one of
the reference implementation's core stabilizers.

Stats update incrementally from stored episodes (Chan et al. parallel
variance combination), so normalization is part of the learned artifact:
`save()` writes an .npz that evaluation must load alongside the policy
checkpoint.
"""

from pathlib import Path

import numpy as np
import torch


class RunningNormalizer:
    def __init__(self, dim: int, clip: float = 5.0, eps: float = 1e-4):
        self.dim = dim
        self.clip = clip
        self.eps = eps
        self.count = 0.0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)  # sum of squared deviations

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, self.dim)
        n = batch.shape[0]
        if n == 0:
            return
        b_mean = batch.mean(axis=0)
        b_m2 = ((batch - b_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = float(n), b_mean, b_m2
            return
        delta = b_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += b_m2 + delta**2 * self.count * n / total
        self.count = total

    @property
    def std(self) -> np.ndarray:
        if self.count == 0:
            return np.ones(self.dim)
        return np.sqrt(np.maximum(self.m2 / self.count, self.eps**2))

    def normalize(self, x):
        if self.count == 0:
            return x  # identity until stats exist
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
            return torch.clamp((x - mean) / std, -self.clip, self.clip)
        x = np.asarray(x, dtype=np.float32)
        out = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, count=self.count, mean=self.mean, m2=self.m2,
                 clip=self.clip, eps=self.eps)

    @classmethod
    def load(cls, path) -> "RunningNormalizer":
        data = np.load(path)
        norm = cls(dim=int(data["mean"].shape[0]), clip=float(data["clip"]), eps=float(data["eps"]))
        norm.count = float(data["count"])
        norm.mean = data["mean"]
        norm.m2 = data["m2"]
        return norm
