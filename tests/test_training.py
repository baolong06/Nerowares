from pathlib import Path

import numpy as np
import pytest

from src.anchors.encoder import label_template
from src.anchors.eval import operating_k
from src.training.dataset import (
    BRENNAN_DIR,
    BRENNAN_PROTOCOL,
    LabeledEpochs,
    _bids_ids,
    _load_brennan_protocol,
    _resolve_brennan_brainvision,
    _select_diverse_edfs,
    load_brennan_epochs,
)
from src.training.train import _evaluation_vocab, train_and_eval, train_and_eval_brennan


def test_training_uses_closed_dataset_label_space():
    assert _evaluation_vocab(["topic-a", "topic-b", "topic-a"]) == ["topic-a", "topic-b"]
    assert operating_k(["a", "b", "c"]) == 2
    assert operating_k(["a"] * 6) == 1
    assert operating_k([f"c{i}" for i in range(20)]) == 5


def test_grouped_training_reports_controls_without_random_split():
    rng = np.random.default_rng(9)
    labels = ["open", "window"] * 16
    epochs = np.stack([
        label_template(label, 4, 12) + 0.05 * rng.standard_normal((4, 12))
        for label in labels
    ]).astype(np.float32)
    data = LabeledEpochs(
        epochs=epochs,
        labels=labels,
        subject_ids=[f"sub-{index % 4}" for index in range(len(labels))],
        session_ids=[f"ses-{index % 4}" for index in range(len(labels))],
        source="test",
    )
    report = train_and_eval(data, ridge=0.1)
    assert report["n_folds"] == 4
    assert "loso" in report and "cross_session" in report
    assert set(report["controls"]) == {
        "shuffled_label_top5",
        "temporal_shuffle_top5",
        "gaussian_noise_top5",
        "no_eeg_lm_only_top5",
    }


def test_failed_gate_does_not_publish_encoder(tmp_path):
    data = LabeledEpochs(
        epochs=np.random.default_rng(3).normal(size=(8, 2, 6)).astype(np.float32),
        labels=["topic-a", "topic-b"] * 4,
        subject_ids=[f"sub-{index % 2}" for index in range(8)],
        session_ids=[f"ses-{index % 2}" for index in range(8)],
        source="test",
    )
    report = train_and_eval(data)
    from src.training.train import save_artifacts

    path = save_artifacts(report, tmp_path)
    saved = __import__("json").loads((tmp_path / "train_report.json").read_text())
    if not report["gate_pass"]:
        assert path is None
        assert not (tmp_path / "encoder.npz").exists()
        assert saved["deployment_status"] == "blocked_below_control"


def test_select_diverse_edfs_spans_subjects_and_sessions():
    paths = [
        Path("root/sub-01/ses-01/eeg/sub-01_ses-01_task-para1_run-01_eeg.edf"),
        Path("root/sub-01/ses-01/eeg/sub-01_ses-01_task-para1_run-02_eeg.edf"),
        Path("root/sub-01/ses-02/eeg/sub-01_ses-02_task-para1_run-01_eeg.edf"),
        Path("root/sub-02/ses-01/eeg/sub-02_ses-01_task-para1_run-01_eeg.edf"),
        Path("root/sub-02/ses-02/eeg/sub-02_ses-02_task-para1_run-01_eeg.edf"),
    ]
    selected = _select_diverse_edfs(paths, 4)
    ids = [_bids_ids(path) for path in selected]
    assert len(selected) == 4
    assert {subject for subject, _ in ids} == {"sub-01", "sub-02"}
    assert {session for _, session in ids} == {"ses-01", "ses-02"}


def test_brennan_protocol_contract():
    if not BRENNAN_PROTOCOL.exists():
        pytest.skip("Brennan protocol fixture is not available")
    protocol = _load_brennan_protocol()
    primary = protocol["cohort"]["primary_subjects"]
    analyzed = protocol["cohort"]["analyzed_yes"]
    sentences = protocol["task"]["sentences"]
    block = protocol["splits"]["protocol_B_stimulus_block"]

    assert protocol["cohort"]["primary_n"] == 27 == len(primary)
    assert protocol["cohort"]["analyzed_yes_n"] == 31 == len(analyzed)
    assert protocol["task"]["closed_set_size"] == 29 == len(sentences)
    assert set(block["train_sentence_ids"]).isdisjoint(set(block["test_sentence_ids"]))
    assert block["held_out_segments"] == [10, 11, 12]
    assert block["temporal_buffer_s"] >= 1.0


def test_brennan_brainvision_triplet_helper_rejects_standalone_files(tmp_path, monkeypatch):
    from src.preprocessing import io as iomod

    monkeypatch.setattr(iomod, "_allowed_roots", lambda: [tmp_path.resolve()])
    root = tmp_path / "nm000180"
    eeg_dir = root / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    base = eeg_dir / "sub-001_task-alicelistening_eeg"
    vhdr = base.with_suffix(".vhdr")
    vhdr.write_text("Brain Vision Data Exchange Header File Version 1.0", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing BrainVision sibling"):
        _resolve_brennan_brainvision(vhdr, root=root)

    base.with_suffix(".eeg").write_bytes(b"eeg")
    base.with_suffix(".vmrk").write_text("Brain Vision Data Exchange Marker File", encoding="utf-8")
    assert _resolve_brennan_brainvision(vhdr, root=root) == vhdr.resolve()

    with pytest.raises(ValueError, match=".vhdr"):
        _resolve_brennan_brainvision(base.with_suffix(".eeg"), root=root)


def test_brennan_loader_smoke_if_dataset_present():
    pytest.importorskip("mne")
    vhdr = BRENNAN_DIR / "sub-001" / "eeg" / "sub-001_task-alicelistening_eeg.vhdr"
    if not vhdr.exists():
        pytest.skip("Brennan BrainVision fixture is not available")

    data = load_brennan_epochs(
        max_subjects=1,
        max_sentences=2,
        n_times=16,
        n_channels=8,
        use_autoreject=False,
        use_ica=False,
    )
    assert data.epochs.shape == (2, 8, 16)
    assert data.subject_ids == ["sub-001", "sub-001"]
    assert data.session_ids == ["sub-001:run-01", "sub-001:run-01"]
    assert data.stimulus_ids == ["segment:01", "segment:01"]
    assert data.metadata is not None
    assert [item["sentence_id"] for item in data.metadata] == [3, 5]
    assert data.labels == [
        "thought Alice without pictures or conversation",
        "when suddenly a White Rabbit with pink eyes ran close by",
    ]


def test_brennan_no_fake_cross_session_eval():
    labels = ["train one", "held one", "train two", "held two"]
    data = LabeledEpochs(
        epochs=np.random.default_rng(7).normal(size=(4, 2, 6)).astype(np.float32),
        labels=labels,
        subject_ids=["sub-001", "sub-001", "sub-003", "sub-003"],
        session_ids=["sub-001:run-01", "sub-001:run-01", "sub-003:run-01", "sub-003:run-01"],
        source="brennan:test",
        stimulus_ids=["segment:01", "segment:10", "segment:01", "segment:10"],
        metadata=[
            {"subject_id": "sub-001", "sentence_id": 3, "segment": 1, "onset": 0.0, "offset": 2.0},
            {"subject_id": "sub-001", "sentence_id": 68, "segment": 10, "onset": 10.0, "offset": 12.0},
            {"subject_id": "sub-003", "sentence_id": 3, "segment": 1, "onset": 20.0, "offset": 22.0},
            {"subject_id": "sub-003", "sentence_id": 68, "segment": 10, "onset": 30.0, "offset": 32.0},
        ],
    )
    report = train_and_eval_brennan(data, ridge=0.1, embed_dim=16)
    assert "cross_session" not in report
    assert "loso_same_stimulus" in report
    assert "stimulus_block" in report
    assert report["stimulus_block"]["n_train_epochs"] == 2
    assert report["stimulus_block"]["n_test_epochs"] == 2
