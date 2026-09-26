"""EEG readers — only through safe_resolve / validate_extension. No pickle."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_npy_safe, load_parquet_safe, safe_resolve, validate_extension


def load_csv_epochs(path: str | Path) -> np.ndarray:
    p = safe_resolve(path)
    validate_extension(p)
    if p.suffix.lower() != ".csv":
        raise ValueError("load_csv_epochs expects a csv file")
    arr = np.loadtxt(str(p), delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    return arr.astype(np.float32)


def load_mne_raw(path: str | Path) -> np.ndarray:
    """Optional MNE path for .edf/.bdf/.fif. Raises if MNE is not installed."""
    p = safe_resolve(path)
    validate_extension(p)
    if p.suffix.lower() not in {".edf", ".bdf", ".fif"}:
        raise ValueError("load_mne_raw expects .edf, .bdf, or .fif")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is not installed; cannot read EDF/BDF/FIF") from exc
    raw = mne.io.read_raw(str(p), preload=True, verbose=False)
    data = raw.get_data()  # (n_channels, n_times) in volts
    return (data.astype(np.float32) * 1e6)[None, :, :]  # µV, add epoch axis


def load_epochs(path: str | Path) -> np.ndarray:
    p = safe_resolve(path)
    validate_extension(p)
    ext = p.suffix.lower()
    if ext in {".npy", ".npz"}:
        arr = load_npy_safe(p)
        if ext == ".npz":
            key = "epochs" if "epochs" in arr.files else arr.files[0]
            arr = arr[key]
        return np.asarray(arr, dtype=np.float32)
    if ext == ".parquet":
        table = load_parquet_safe(p)
        return np.asarray(table.to_pandas().values, dtype=np.float32)
    if ext == ".csv":
        return load_csv_epochs(p)
    if ext in {".edf", ".bdf", ".fif"}:
        return load_mne_raw(p)
    raise ValueError("Unsupported EEG format")
