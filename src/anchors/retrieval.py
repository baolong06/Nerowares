"""Contrastive retrieval — EEG embedding vs text anchor embeddings."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalResult:
    anchor_id: str
    surface: str
    score: float
    rank: int
    provenance: list[str]


def text_embed(surface: str, dim: int = 64) -> np.ndarray:
    h = hashlib.sha256(surface.strip().lower().encode()).digest()
    vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    rep = np.tile(vals, (dim // len(vals) + 1))[:dim]
    rep = (rep - 128.0) / 64.0
    norm = np.linalg.norm(rep) + 1e-8
    return (rep / norm).astype(np.float32)


def brennan_text_embed(surface: str, dim: int = 64) -> np.ndarray:
    """Semantic embedding for Brennan Alice sentences.

    Uses a fixed char n-gram HashingVectorizer so held-out story sentences
    share cosine similarity with train sentences (unlike the hash embedder
    where test->train mean cosine ~ 0). No fitted vocabulary, no network,
    no extra dependency beyond scikit-learn.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    text = surface.strip().lower() or " "
    vectorizer = HashingVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=int(dim),
        alternate_sign=False,
        norm="l2",
        binary=False,
    )
    vec = vectorizer.transform([text]).toarray()[0].astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        # Degenerate bucket collision (extremely rare); fall back to hash
        return text_embed(surface, dim)
    return (vec / norm).astype(np.float32)


def build_brennan_tfidf_embedder(train_texts: list[str], dim: int = 64):
    """Word TFIDF embedder fitted on train sentences only (anti-leakage).

    Returns a ``(surface, dim) -> L2 vec`` callable closed over a fitted
    ``TfidfVectorizer``. Kept deliberately small (max_features=dim) so the
    ridge target stays in 64-d and test sentences share moderate cosine with
    train neighbours (unlike the SHA-256 hash ~0 nor char_wb ~0.66).
    No network, no fitted global vocab — caller must fit on train split only.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    d = int(dim)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=d,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    corpus = [t.strip().lower() or " " for t in train_texts]
    vectorizer.fit(corpus)

    def embed_fn(surface: str, dim: int = 64, **_kw) -> np.ndarray:
        dd = int(dim)
        text = surface.strip().lower() or " "
        vec = vectorizer.transform([text]).toarray()[0].astype(np.float32)
        if len(vec) < dd:
            out = np.zeros(dd, dtype=np.float32)
            out[: len(vec)] = vec
        else:
            out = vec[:dd].astype(np.float32)
        norm = float(np.linalg.norm(out))
        if norm < 1e-8:
            return text_embed(surface, dim)
        return (out / norm).astype(np.float32)

    embed_fn.vectorizer = vectorizer  # type: ignore[attr-defined]
    return embed_fn


# Back-compat alias used by older call sites
_text_embed = text_embed


def retrieve_anchors(
    eeg_emb: np.ndarray,
    vocab_keywords: list[str],
    top_k: int = 5,
    epoch_id: str = "eeg:epoch_00001",
    embed_fn=text_embed,
) -> list[RetrievalResult]:
    if not epoch_id.startswith("eeg:"):
        epoch_id = f"eeg:{epoch_id}"
    dim = int(eeg_emb.shape[0])
    scores = []
    for kw in vocab_keywords:
        t = embed_fn(kw, dim)
        scores.append((kw, float(np.dot(eeg_emb, t))))
    scores.sort(key=lambda x: x[1], reverse=True)
    out: list[RetrievalResult] = []
    for rank, (surf, sc) in enumerate(scores[:top_k], start=1):
        conf = (sc + 1.0) / 2.0
        out.append(
            RetrievalResult(
                anchor_id=f"anchor:{surf.replace(' ', '_')}",
                surface=surf,
                score=float(conf),
                rank=rank,
                provenance=[epoch_id],
            )
        )
    return out
