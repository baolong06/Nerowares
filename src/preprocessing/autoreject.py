"""Autoreject-style epoch rejection.

Uses the autoreject package when installed; otherwise a robust per-channel
peak-to-peak + MAD rule that does not require MNE.
"""
from __future__ import annotations

import numpy as np


def reject_epochs(
    epochs: np.ndarray,
    z_thresh: float = 4.0,
    max_bad_frac: float = 0.25,
    sfreq: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs, keep_mask). Does not drop rows — mask only, matching pipeline contract."""
    x = np.asarray(epochs, dtype=np.float32)
    n, c, t = x.shape
    try:
        from autoreject import AutoReject  # type: ignore
        from mne import create_info
        from mne.epochs import EpochsArray

        info = create_info(ch_names=[f"EEG{i:03d}" for i in range(c)], sfreq=float(sfreq), ch_types="eeg")
        epo = EpochsArray(x * 1e-6, info, verbose=False)
        ar = AutoReject(n_interpolate=0, verbose=False)
        ar.fit(epo)
        log = ar.get_reject_log(epo)
        mask = ~np.asarray(log.bad_epochs, dtype=bool)
        return x, mask
    except Exception:
        ptp = x.max(axis=2) - x.min(axis=2)  # (n, c)
        med = np.median(ptp, axis=0)
        mad = np.median(np.abs(ptp - med), axis=0) + 1e-6
        z = np.abs(ptp - med) / (1.4826 * mad)
        bad_frac = (z > z_thresh).mean(axis=1)
        mask = bad_frac <= max_bad_frac
        # Never reject everything — keep the cleanest half as a safety net.
        if not mask.any():
            order = np.argsort(bad_frac)
            mask = np.zeros(n, dtype=bool)
            mask[order[: max(1, n // 2)]] = True
        return x, mask
