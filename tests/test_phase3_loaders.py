"""Phase 3 dataset contracts — RED phase.

D01  ds003626 inner speech  (BDF raw + derivatives FIF, 10 subjects x 3 sessions, 4 classes)
D03  Chisco subject02       (EDF 45 runs + xlsx text labels, 64ch imagined speech)
D09  BCI2020 imagined speech (MAT epo_train/epo_validation, 64ch, 5 classes, 300 epochs)
SparrKULee is not present locally — loader must fail closed with a clear message.
"""
from pathlib import Path

import numpy as np
import pytest

from src.training.dataset import LabeledEpochs


def test_d01_loader_exists_and_contract():
    from src.training.dataset import D01_DIR, _resolve_phase3_file, load_d01_epochs

    assert callable(load_d01_epochs)
    assert D01_DIR.name == "ds003626_inner_speech"
    # The fixture contains .pkl reports, but the loader's root-scoped contract
    # must reject them instead of deserializing them.
    pkl = next(D01_DIR.glob("derivatives/**/*.pkl"), None)
    if pkl is not None:
        with pytest.raises(ValueError, match="extension"):
            _resolve_phase3_file(pkl, D01_DIR, {".fif"})


def test_d01_loader_smoke_if_present():
    pytest.importorskip("mne")
    from src.training.dataset import D01_DIR  # type: ignore[import]

    if not D01_DIR.exists():
        pytest.skip("D01 root not present")
    from src.training.dataset import load_d01_epochs

    data = load_d01_epochs(max_subjects=1, max_sessions=1, n_times=32, n_channels=16, use_autoreject=False, use_ica=False)
    assert isinstance(data, LabeledEpochs)
    assert data.epochs.ndim == 3
    assert data.epochs.shape[1] == 16
    assert data.epochs.shape[2] == 32
    assert len(data.labels) == data.epochs.shape[0] >= 4
    assert set(data.labels).issubset({"arriba", "abajo", "derecha", "izquierda", "up", "down", "right", "left"})
    assert len(set(data.subject_ids)) == 1
    assert len(set(data.session_ids)) >= 1
    # No random split — subject/session identity must be preserved
    assert data.subject_ids[0].startswith("sub-")


def test_chisco_loader_exists_and_contract():
    from src.training.dataset import load_chisco_epochs

    assert callable(load_chisco_epochs)


def test_chisco_loader_smoke_if_present():
    pytest.importorskip("mne")
    from src.training.dataset import CHISCO_DIR  # type: ignore[import]

    if not CHISCO_DIR.exists():
        pytest.skip("Chisco root not present")
    from src.training.dataset import load_chisco_epochs

    # Chisco subject02 local fixture — at least a few epochs
    data = load_chisco_epochs(max_runs=2, n_times=32, n_channels=16, use_autoreject=False, use_ica=False)
    assert isinstance(data, LabeledEpochs)
    assert data.epochs.ndim == 3
    assert data.epochs.shape[2] == 32
    assert len(data.labels) >= 2
    assert len(data.subject_ids) == len(data.labels)


def test_d09_loader_exists_and_contract():
    from src.training.dataset import load_d09_epochs

    assert callable(load_d09_epochs)


def test_d09_loader_smoke_if_present():
    from src.training.dataset import D09_DIR  # type: ignore[import]

    if not D09_DIR.exists():
        pytest.skip("D09 root not present")
    from src.training.dataset import load_d09_epochs

    data = load_d09_epochs(split="train", max_epochs=16, n_times=32, n_channels=16)
    assert isinstance(data, LabeledEpochs)
    assert data.epochs.shape[0] >= 4
    assert data.epochs.shape[2] == 32
    assert set(data.labels).issubset({"hello", "helpme", "stop", "thankyou", "yes"})


def test_sparrkulee_loader_fails_closed_when_absent():
    from src.training.dataset import load_sparrkulee_epochs

    assert callable(load_sparrkulee_epochs)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_sparrkulee_epochs()


def test_bciciv2a_loader_exists_and_contract():
    from src.training.dataset import BCICIV2A_DIR, _resolve_phase3_file, load_bciciv2a_epochs

    assert callable(load_bciciv2a_epochs)
    assert BCICIV2A_DIR.name == "aymanmostafa11__eeg-motor-imagery-bciciv-2a"
    with pytest.raises(ValueError, match="extension"):
        _resolve_phase3_file(BCICIV2A_DIR / "untrusted.mat", BCICIV2A_DIR, {".csv"})


def test_bciciv2a_loader_smoke_if_present():
    from src.training.dataset import BCICIV2A_CSV, load_bciciv2a_epochs

    if not BCICIV2A_CSV.exists():
        pytest.skip("BCICIV-2a CSV is not present")
    data = load_bciciv2a_epochs(
        max_epochs=16,
        n_times=32,
        n_channels=16,
        use_autoreject=False,
        use_ica=False,
    )
    assert isinstance(data, LabeledEpochs)
    assert data.epochs.shape == (16, 16, 32)
    assert set(data.labels).issubset({"left", "right", "foot", "tongue"})
    assert all(subject.startswith("sub-") for subject in data.subject_ids)
    assert len(set(data.session_ids)) == len(data.session_ids)
    assert data.metadata is not None
    assert all(item["sfreq_hz"] == pytest.approx(250.0) for item in data.metadata)
    assert all(item["format"] == "csv-windowed" for item in data.metadata)


def test_bciciv2a_loader_rejects_invalid_time_cadence(tmp_path, monkeypatch):
    import csv

    import src.preprocessing.io as iomod
    import src.training.dataset as dataset

    monkeypatch.setattr(iomod, "_allowed_roots", lambda: [tmp_path.resolve()])
    csv_path = tmp_path / "BCICIV_2a_all_patients.csv"
    header = ["patient", "time", "label", "epoch", *dataset._BCICIV2A_EEG_COLUMNS]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(201):
            # The locked contract is 250 Hz / 4 ms; this fixture is deliberately invalid.
            writer.writerow(["1", f"{-0.1 + index * 0.005:.6f}", "left", "1", *(["0"] * 22)])
    monkeypatch.setattr(dataset, "BCICIV2A_DIR", tmp_path)
    monkeypatch.setattr(dataset, "BCICIV2A_CSV", csv_path)

    with pytest.raises(ValueError, match="time"):
        dataset.load_bciciv2a_epochs(max_epochs=1, n_times=16, n_channels=8)


def test_phase3_loaders_do_not_expand_global_allowlist():
    from src.preprocessing.io import ALLOWED_EXTENSIONS

    # Existing .bdf/.edf support is unchanged; Phase 3 must not add unsafe or
    # metadata-only formats to the generic reader allowlist.
    assert ".mat" not in ALLOWED_EXTENSIONS
    assert ".dat" not in ALLOWED_EXTENSIONS
    assert ".pkl" not in ALLOWED_EXTENSIONS
    assert ".xlsx" not in ALLOWED_EXTENSIONS
    assert ".vhdr" not in ALLOWED_EXTENSIONS
    assert ".vmrk" not in ALLOWED_EXTENSIONS
    assert ".eeg" not in ALLOWED_EXTENSIONS


def test_phase3_eval_is_leakage_safe():
    """Each Phase 3 loader result must be evaluatable with LOSO/cross_session without random fallback."""
    from src.training.train import train_and_eval

    rng = np.random.default_rng(11)
    labels = ["hello", "stop", "yes", "thankyou"] * 8
    epochs = np.stack([rng.normal(size=(4, 16)).astype(np.float32) for _ in labels])
    data = LabeledEpochs(
        epochs=epochs,
        labels=labels,
        subject_ids=[f"sub-{i % 4:02d}" for i in range(len(labels))],
        session_ids=[f"sub-{i % 4:02d}:ses-{i % 2 + 1:02d}" for i in range(len(labels))],
        source="phase3:test",
    )
    report = train_and_eval(data, ridge=0.1, embed_dim=16)
    assert "loso" in report and "cross_session" in report
