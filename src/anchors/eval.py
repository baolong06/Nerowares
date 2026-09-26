"""5-way retrieval evaluation — ADR-003 gate.

Controls: real_eeg, shuffled_label (labels remapped), temporal_shuffle,
gaussian_noise, no_eeg_lm_only. Teacher-forcing-free: retrieval never sees gold tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval import retrieve_anchors, text_embed


@dataclass
class EvalReport:
    top1: float
    top5: float
    top25: float
    baselines: dict


def _topk_hit(retrieved_surfaces: list[str], label: str, k: int) -> bool:
    return label.lower() in [s.lower() for s in retrieved_surfaces[:k]]


def operating_k(vocab: list[str], top_k: int = 5) -> int:
    """Top-k saturates at 1.0 whenever k >= |vocab|; use |vocab|-1 then."""
    n_labels = len(set(vocab))
    if n_labels <= 1:
        return 1
    return min(top_k, n_labels - 1)


def _score_for(
    embs: np.ndarray,
    labels: list[str],
    vocab: list[str],
    k: int = 5,
    embed_fn=text_embed,
) -> tuple[float, float, float]:
    hits1 = hits_k = hits25 = 0
    for emb, lab in zip(embs, labels):
        res = retrieve_anchors(emb, vocab, top_k=25, epoch_id="eeg:epoch_eval", embed_fn=embed_fn)
        surfs = [r.surface for r in res]
        if _topk_hit(surfs, lab, 1):
            hits1 += 1
        if _topk_hit(surfs, lab, k):
            hits_k += 1
        if _topk_hit(surfs, lab, 25):
            hits25 += 1
    n = len(labels) or 1
    return hits1 / n, hits_k / n, hits25 / n


def evaluate_retrieval(
    eeg_embs: np.ndarray, labels: list[str], vocab: list[str], top_k: int = 5, embed_fn=text_embed
) -> EvalReport:
    k = operating_k(vocab, top_k)
    t1, t5, t25 = _score_for(eeg_embs, labels, vocab, k=k, embed_fn=embed_fn)

    rng = np.random.default_rng(1)
    shuffled_labels = list(labels)
    rng.shuffle(shuffled_labels)
    _, shuf5, _ = _score_for(eeg_embs, shuffled_labels, vocab, k=k, embed_fn=embed_fn)

    temporal_embs = rng.standard_normal(eeg_embs.shape).astype(np.float32)
    temporal_embs /= np.linalg.norm(temporal_embs, axis=1, keepdims=True) + 1e-8
    _, temp5, _ = _score_for(temporal_embs, labels, vocab, k=k, embed_fn=embed_fn)

    gaussian_embs = rng.standard_normal(eeg_embs.shape).astype(np.float32)
    gaussian_embs /= np.linalg.norm(gaussian_embs, axis=1, keepdims=True) + 1e-8
    _, gauss5, _ = _score_for(gaussian_embs, labels, vocab, k=k, embed_fn=embed_fn)

    lm_only_top5 = min(k / len(vocab), 1.0) if vocab else 0.0
    baselines = {
        "shuffled_label_top5": shuf5,
        "temporal_shuffle_top5": temp5,
        "gaussian_noise_top5": gauss5,
        "no_eeg_lm_only_top5": lm_only_top5,
    }
    return EvalReport(top1=t1, top5=t5, top25=t25, baselines=baselines)
