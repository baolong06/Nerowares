"""ICA artifact attenuation.

Prefers MNE ICA when installed; otherwise sklearn FastICA; otherwise SVD deflation
of the highest-kurtosis components (EOG/blink-like).
"""
from __future__ import annotations

import numpy as np


def _kurtosis(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    m2 = np.mean(x**2, axis=1) + 1e-8
    m4 = np.mean(x**4, axis=1)
    return m4 / (m2**2) - 3.0


def apply_ica(
    epochs: np.ndarray,
    n_components: int = 8,
    drop_top_k: int = 1,
    sfreq: float = 500.0,
    highpass: float = 1.0,
) -> np.ndarray:
    x = np.asarray(epochs, dtype=np.float32)
    n, c, t = x.shape
    n_components = max(2, min(n_components, c))
    try:
        from mne import create_info
        from mne.epochs import EpochsArray
        from mne.preprocessing import ICA

        info = create_info(ch_names=[f"EEG{i:03d}" for i in range(c)], sfreq=float(sfreq), ch_types="eeg")
        info["highpass"] = float(highpass)
        epo = EpochsArray(x * 1e-6, info, verbose=False)
        ica = ICA(n_components=n_components, random_state=0, max_iter="auto")
        ica.fit(epo)
        sources = ica.get_sources(epo).get_data()
        k = np.abs(_kurtosis(sources.reshape(n * n_components, t)[:n_components]))
        # MNE ICA object: exclude highest-kurtosis components
        ica.exclude = list(np.argsort(k)[::-1][:drop_top_k])
        cleaned = ica.apply(epo.copy(), verbose=False).get_data() * 1e6
        return cleaned.astype(np.float32)
    except Exception:
        pass
    flat = x.transpose(0, 2, 1).reshape(-1, c)  # (n*T, C)
    try:
        from sklearn.decomposition import FastICA

        ica = FastICA(n_components=n_components, whiten="unit-variance", random_state=0, max_iter=400)
        s = ica.fit_transform(flat)
        k = np.abs(_kurtosis(s.T))
        drop = set(np.argsort(k)[::-1][:drop_top_k].tolist())
        s[:, list(drop)] = 0
        recon = ica.inverse_transform(s)
        return recon.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)
    except Exception:
        # SVD: zero the top-kurtosis right-singular directions in channel space
        u, s, vt = np.linalg.svd(flat, full_matrices=False)
        k = np.abs(_kurtosis((u * s)[:, :n_components].T))
        drop = np.argsort(k)[::-1][:drop_top_k]
        s = s.copy()
        s[drop] = 0
        recon = (u * s) @ vt
        return recon.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)
