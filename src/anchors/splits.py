"""Split strategies — ADR-003 anti-leakage. Full LOSO / cross-session loops."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplitConfig:
    strategy: str = "cross_session"  # cross_session | loso | stimulus_identity
    n_splits: int = 4
    holdout_subject: str | None = None
    allow_random_fallback: bool = False


def make_splits(
    n_epochs: int,
    session_ids: list[str] | None = None,
    subject_ids: list[str] | None = None,
    stimulus_ids: list[str] | None = None,
    config: SplitConfig | None = None,
):
    cfg = config or SplitConfig()
    if cfg.strategy == "cross_session" and session_ids is not None:
        uniq = sorted(set(session_ids))
        yielded = False
        for held in uniq:
            train_idx = [i for i, s in enumerate(session_ids) if s != held]
            test_idx = [i for i, s in enumerate(session_ids) if s == held]
            if train_idx and test_idx:
                yielded = True
                yield train_idx, test_idx
        if yielded:
            return
    if cfg.strategy == "loso" and subject_ids is not None:
        uniq = sorted(set(subject_ids))
        if cfg.holdout_subject:
            uniq = [cfg.holdout_subject] if cfg.holdout_subject in uniq else uniq
        yielded = False
        for held in uniq:
            train_idx = [i for i, s in enumerate(subject_ids) if s != held]
            test_idx = [i for i, s in enumerate(subject_ids) if s == held]
            if train_idx and test_idx:
                yielded = True
                yield train_idx, test_idx
        if yielded:
            return
    if cfg.strategy == "stimulus_identity" and stimulus_ids is not None:
        uniq = sorted(set(stimulus_ids))
        yielded = False
        for held in uniq:
            train_idx = [i for i, s in enumerate(stimulus_ids) if s != held]
            test_idx = [i for i, s in enumerate(stimulus_ids) if s == held]
            if train_idx and test_idx:
                yielded = True
                yield train_idx, test_idx
        if yielded:
            return
    if not cfg.allow_random_fallback:
        raise ValueError(
            f"Split strategy {cfg.strategy!r} requires matching ids; random-epoch split is banned (ADR-003)"
        )
    rng = np.random.default_rng(0)
    idx = np.arange(n_epochs)
    rng.shuffle(idx)
    cut = int(n_epochs * 0.8)
    yield idx[:cut].tolist(), idx[cut:].tolist()
