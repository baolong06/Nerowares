"""Small nonlinear EEG-to-text anchor encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .retrieval import text_embed

TextEmbedFn = Callable[[str, int], np.ndarray]


class MlpAnchorEncoder:
    """Two-layer ReLU map trained with full-batch anchor regression."""

    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 200,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        seed: int = 0,
        embed_fn: TextEmbedFn = text_embed,
    ):
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.embed_fn = embed_fn
        rng = np.random.default_rng(self.seed)
        input_dim = self.n_channels * self.n_times
        self.w1 = (
            rng.standard_normal((input_dim, self.hidden_dim))
            * np.sqrt(2.0 / max(input_dim, 1))
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2 = (
            rng.standard_normal((self.hidden_dim, self.embed_dim))
            * np.sqrt(2.0 / max(self.hidden_dim, 1))
        ).astype(np.float32)
        self.b2 = np.zeros(self.embed_dim, dtype=np.float32)
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.fitted = False

    def _flatten(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        flat = data.reshape(n, -1).astype(np.float32)
        expected = self.n_channels * self.n_times
        if flat.shape[1] > expected:
            flat = flat[:, :expected]
        elif flat.shape[1] < expected:
            flat = np.pad(flat, ((0, 0), (0, expected - flat.shape[1])))
        return flat

    def _scale_features(self, flat: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            self.x_mean = flat.mean(axis=0).astype(np.float32)
            scale = flat.std(axis=0).astype(np.float32)
            self.x_scale = np.maximum(scale, 1e-6)
        if self.x_mean is None or self.x_scale is None:
            return flat
        return ((flat - self.x_mean) / self.x_scale).astype(np.float32)

    def _forward(self, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = flat @ self.w1 + self.b1
        hidden = np.maximum(z1, 0.0)
        raw = hidden @ self.w2 + self.b2
        return z1, hidden, raw

    def encode(self, data: np.ndarray) -> np.ndarray:
        flat = self._scale_features(self._flatten(data), fit=False)
        _, _, raw = self._forward(flat)
        norm = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        return (raw / norm).astype(np.float32)

    def fit(
        self,
        data: np.ndarray,
        labels: list[str],
        *,
        ridge: float = 1e-4,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> "MlpAnchorEncoder":
        """Fit by deterministic full-batch gradient descent in NumPy."""
        if len(data) != len(labels):
            raise ValueError("Data and label counts must match")
        if len(data) == 0:
            raise ValueError("At least one epoch is required")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        X = self._scale_features(self._flatten(data), fit=True)
        Y = np.stack([self.embed_fn(label, self.embed_dim) for label in labels]).astype(np.float32)
        n_samples = float(len(labels))

        for _ in range(int(epochs)):
            z1, hidden, raw = self._forward(X)
            grad_raw = (2.0 / n_samples) * (raw - Y)
            grad_w2 = hidden.T @ grad_raw + 2.0 * ridge * self.w2
            grad_b2 = grad_raw.sum(axis=0)
            grad_hidden = grad_raw @ self.w2.T
            grad_z1 = grad_hidden * (z1 > 0.0)
            grad_w1 = X.T @ grad_z1 + 2.0 * ridge * self.w1
            grad_b1 = grad_z1.sum(axis=0)

            self.w2 -= (lr * grad_w2).astype(np.float32)
            self.b2 -= (lr * grad_b2).astype(np.float32)
            self.w1 -= (lr * grad_w1).astype(np.float32)
            self.b1 -= (lr * grad_b1).astype(np.float32)

        self.fitted = True
        return self

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            x_mean=self.x_mean if self.x_mean is not None else np.array([], dtype=np.float32),
            x_scale=self.x_scale if self.x_scale is not None else np.array([], dtype=np.float32),
            n_channels=self.n_channels,
            n_times=self.n_times,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MlpAnchorEncoder":
        data = np.load(path, allow_pickle=False)
        enc = cls(
            int(data["n_channels"]),
            int(data["n_times"]),
            int(data["embed_dim"]),
            int(data["hidden_dim"]),
            int(data["seed"]),
            embed_fn=text_embed,
        )
        enc.w1 = data["w1"].astype(np.float32)
        enc.b1 = data["b1"].astype(np.float32)
        enc.w2 = data["w2"].astype(np.float32)
        enc.b2 = data["b2"].astype(np.float32)
        x_mean = data["x_mean"].astype(np.float32)
        x_scale = data["x_scale"].astype(np.float32)
        enc.x_mean = x_mean if x_mean.size else None
        enc.x_scale = x_scale if x_scale.size else None
        enc.fitted = True
        return enc
