"""Preprocessing pipeline — bandpass (scipy, MNE if present) + average ref + detrend + baseline + PTP mask."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class PreprocessingConfig:
    sampling_rate_hz: float = 500.0
    bandpass_low: float = 1.0
    bandpass_high: float = 40.0
    reference: Literal["average", "mastoid", "none"] = "average"
    baseline: tuple[float, float] | None = (-0.2, 0.0)
    bad_epoch_threshold_uv: float = 100.0
    detrend: bool = True
    use_autoreject: bool = False
    use_ica: bool = False
    apply_bandpass: bool = True
    ica_n_components: int = 8
    profile_name: str = "thinking-v1"


@dataclass
class PreprocessingResult:
    epochs: np.ndarray
    mask_keep: np.ndarray
    config: PreprocessingConfig
    info: dict = field(default_factory=dict)


def _bandpass(x: np.ndarray, low: float, high: float, sfreq: float) -> np.ndarray:
    """Zero-phase bandpass. Prefer MNE, else scipy SOS, else FFT fallback."""
    n_epochs, n_channels, n_times = x.shape
    nyq = sfreq / 2.0
    hi = min(high, nyq * 0.95)
    lo = max(low, 0.01)
    if hi <= lo:
        return x
    try:
        from mne.filter import filter_data

        flat = x.reshape(n_epochs * n_channels, n_times)
        filtered = filter_data(flat.astype(np.float64), sfreq, lo, hi, verbose=False)
        return filtered.reshape(n_epochs, n_channels, n_times).astype(np.float32)
    except Exception:
        pass
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        out = np.empty_like(x, dtype=np.float32)
        for i in range(n_epochs):
            out[i] = sosfiltfilt(sos, x[i], axis=-1).astype(np.float32)
        return out
    except Exception:
        # FFT brick-wall fallback
        freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
        spec = np.fft.rfft(x, axis=-1)
        mask = (freqs >= lo) & (freqs <= hi)
        spec[..., ~mask] = 0
        return np.fft.irfft(spec, n=n_times, axis=-1).astype(np.float32)


def preprocess_epochs(epochs: np.ndarray, config: PreprocessingConfig) -> PreprocessingResult:
    x = np.asarray(epochs, dtype=np.float32)
    n_epochs, n_channels, n_times = x.shape
    if config.apply_bandpass:
        x = _bandpass(x, config.bandpass_low, config.bandpass_high, config.sampling_rate_hz)

    if config.reference == "average":
        x = x - x.mean(axis=1, keepdims=True)

    if config.detrend:
        x = x - x.mean(axis=2, keepdims=True)

    if config.baseline is not None:
        b = max(1, int(n_times * 0.2))
        x = x - x[:, :, :b].mean(axis=2, keepdims=True)

    raw = np.asarray(epochs, dtype=np.float32)
    ptp_per_epoch = raw.max(axis=2).max(axis=1) - raw.min(axis=2).min(axis=1)
    mask_keep = ptp_per_epoch < config.bad_epoch_threshold_uv
    info: dict = {
        "ptp_per_epoch": ptp_per_epoch.tolist(),
        "bandpass": "applied" if config.apply_bandpass else "pre_applied",
    }

    if config.use_autoreject:
        from .autoreject import reject_epochs

        x, ar_mask = reject_epochs(x, sfreq=config.sampling_rate_hz)
        mask_keep = mask_keep & ar_mask
        info["autoreject"] = "applied"

    if config.use_ica:
        from .ica import apply_ica

        x = apply_ica(
            x,
            n_components=config.ica_n_components,
            sfreq=config.sampling_rate_hz,
            highpass=config.bandpass_low,
        )
        info["ica"] = "applied"

    return PreprocessingResult(
        epochs=x,
        mask_keep=mask_keep,
        config=config,
        info=info,
    )
