"""EEG encoder — ridge map from epoch features onto text-anchor space.

Untrained: random projection (chance-level retrieval).
After fit(epochs, labels): linear W such that flatten(eeg) @ W ≈ text_embed(label),
so synthetic / labeled eval can beat noise and shuffled-label controls.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from .retrieval import text_embed


TextEmbedFn = Callable[[str, int], np.ndarray]


class AnchorEncoder:
    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 200,
        embed_dim: int = 64,
        seed: int = 0,
        embed_fn: TextEmbedFn = text_embed,
    ):
        self.n_channels = n_channels
        self.n_times = n_times
        self.embed_dim = embed_dim
        self.seed = seed
        self.embed_fn = embed_fn
        rng = np.random.default_rng(seed)
        self.proj = rng.standard_normal((n_channels * n_times, embed_dim)).astype(np.float32) * 0.02
        self.fitted = False

    def _flatten(self, epochs: np.ndarray) -> np.ndarray:
        n = epochs.shape[0]
        flat = epochs.reshape(n, -1).astype(np.float32)
        expected = self.n_channels * self.n_times
        if flat.shape[1] > expected:
            flat = flat[:, :expected]
        elif flat.shape[1] < expected:
            flat = np.pad(flat, ((0, 0), (0, expected - flat.shape[1])))
        return flat

    def encode(self, epochs: np.ndarray) -> np.ndarray:
        flat = self._flatten(epochs)
        emb = flat @ self.proj
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return (emb / norm).astype(np.float32)

    def fit(self, epochs: np.ndarray, labels: list[str], ridge: float = 1.0) -> "AnchorEncoder":
        """Least-squares map from flattened EEG onto L2-normalized text embeddings of labels."""
        X = self._flatten(epochs)
        Y = np.stack([self.embed_fn(lab, self.embed_dim) for lab in labels]).astype(np.float32)
        xtx = X.T @ X
        dim = xtx.shape[0]
        W = np.linalg.solve(xtx + ridge * np.eye(dim, dtype=np.float32), X.T @ Y)
        self.proj = W.astype(np.float32)
        self.fitted = True
        return self

    def save(self, path: str | Path) -> None:
        np.savez(path, proj=self.proj, n_channels=self.n_channels, n_times=self.n_times, embed_dim=self.embed_dim, seed=self.seed)

    @classmethod
    def load(cls, path: str | Path) -> "AnchorEncoder":
        data = np.load(path, allow_pickle=False)
        enc = cls(
            int(data["n_channels"]),
            int(data["n_times"]),
            int(data["embed_dim"]),
            int(data["seed"]),
            embed_fn=text_embed,
        )
        enc.proj = data["proj"].astype(np.float32)
        enc.fitted = True
        return enc


def label_template(label: str, n_channels: int, n_times: int) -> np.ndarray:
    """Deterministic spatiotemporal template used to structure synthetic EEG by keyword."""
    digest = hashlib.sha256(label.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**31))
    return rng.standard_normal((n_channels, n_times)).astype(np.float32)


def encode_epochs(epochs: np.ndarray, encoder: AnchorEncoder | None = None) -> np.ndarray:
    enc = encoder or AnchorEncoder(n_channels=epochs.shape[1], n_times=epochs.shape[2])
    return enc.encode(epochs)
