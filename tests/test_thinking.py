import json
from pathlib import Path

import numpy as np
import pytest

from src.anchors.encoder import AnchorEncoder, label_template
from src.anchors.eval import evaluate_retrieval
from src.anchors.splits import SplitConfig, make_splits
from src.anchors.vocab import load_vocab
from src.kg.ontology import load_ontology
from src.kg.resolver import resolve_anchors
from src.preprocessing.io import safe_resolve, validate_extension
from src.preprocessing.pipeline import PreprocessingConfig, preprocess_epochs
from src.prompt.compiler import compile_prompt
from src.prompt.generator import generate_text
from src.prompt.validator import validate_prompt


def test_preprocess_mask_and_bandpass():
    cfg = PreprocessingConfig(bad_epoch_threshold_uv=100)
    epochs = np.random.randn(4, 8, 100).astype(np.float32) * 10
    epochs[2, 0, 0] = 1000
    epochs[2, 0, 1] = -1000
    res = preprocess_epochs(epochs, cfg)
    assert res.mask_keep.tolist() == [True, True, False, True]
    assert res.info["bandpass"] == "applied"


def test_preprocess_autoreject_and_ica_optional_paths():
    rng = np.random.default_rng(101)
    epochs = rng.normal(size=(6, 4, 80)).astype(np.float32)
    result = preprocess_epochs(
        epochs,
        PreprocessingConfig(
            sampling_rate_hz=200,
            bad_epoch_threshold_uv=1_000_000,
            use_autoreject=True,
            use_ica=True,
            ica_n_components=3,
        ),
    )
    assert result.epochs.shape == epochs.shape
    assert result.mask_keep.shape == (6,)
    assert result.info["autoreject"] == "applied"
    assert result.info["ica"] == "applied"


def test_vocab_allowlist_filters_invalid_ontology_surface(tmp_path: Path):
    ontology = {"nodes": [{"label": "normal"}, {"label": "<script>"}], "synonyms": [{"surface": "open", "concept_id": "x:y"}, {"surface": "bad\nname", "concept_id": "x:y"}]}
    path = tmp_path / "ont.json"
    path.write_text(json.dumps(ontology), encoding="utf-8")
    vocab = load_vocab(path)
    assert vocab.contains("normal")
    assert not vocab.contains("script")
    with pytest.raises(ValueError):
        vocab.validate_or_raise("not-in-vocab-xyz-999")


@pytest.mark.parametrize("path", ["../../etc/passwd", "%2e%2e/%2e%2e/etc/passwd", "..%252fsecret.npy"])
def test_safe_resolve_blocks_traversal(path):
    with pytest.raises(ValueError, match="blocked"):
        safe_resolve(path)


def test_extension_rejects_compound_suffix():
    with pytest.raises(ValueError):
        validate_extension(Path("sample.npy.exe"))


def test_kg_resolver_separates_eeg_and_kg_provenance():
    ont = load_ontology(True)
    anchors = [{"surface": "window", "score": 0.9, "provenance": ["eeg:epoch_00001"]}]
    sg = resolve_anchors(anchors, ont)
    accepted = [r for r in sg.resolved if r.status == "accepted"]
    suggested = [r for r in sg.resolved if r.status == "suggested"]
    assert accepted and all(p.startswith("eeg:") for r in accepted for p in r.provenance)
    assert suggested and all(p.startswith("kg:edge:") and "edge:edge:" not in p for r in suggested for p in r.provenance)
    assert any(r.operator == "increase" for r in suggested)


def test_kg_rejects_spoofed_accepted_provenance():
    with pytest.raises(ValueError, match="eeg"):
        resolve_anchors([{"surface": "open", "score": 0.9, "provenance": ["kg:edge:spoof"]}], load_ontology(True))


def test_prompt_validator_strict_and_dynamic_ontology_version():
    ont = load_ontology(True)
    anchors = [{"surface": "window", "score": 0.9, "provenance": ["eeg:epoch_00001"]}]
    sg = resolve_anchors(anchors, ont)
    prompt = compile_prompt(anchors, sg, modality="text")
    validate_prompt(prompt)
    assert prompt["kg"]["ontology_version"] == ont.ontology_version == "text-v0.1"
    prompt["untrusted_extra"] = True
    with pytest.raises(AssertionError, match="Additional properties|additionalProperties"):
        validate_prompt(prompt)


def test_full_cross_session_and_loso_splits_no_random_fallback():
    sessions = ["a", "a", "b", "b", "c", "c"]
    subjects = ["s1", "s1", "s2", "s2", "s3", "s3"]
    assert len(list(make_splits(6, session_ids=sessions, config=SplitConfig("cross_session")))) == 3
    assert len(list(make_splits(6, subject_ids=subjects, config=SplitConfig("loso")))) == 3
    with pytest.raises(ValueError, match="banned"):
        list(make_splits(6, config=SplitConfig("cross_session")))


def test_fitted_encoder_beats_shuffled_and_noise_controls():
    vocab = load_vocab().keywords
    labels = [vocab[i % 5] for i in range(80)]
    rng = np.random.default_rng(8)
    epochs = np.stack([label_template(label, 8, 20) + 0.08 * rng.standard_normal((8, 20)) for label in labels]).astype(np.float32)
    train, test = np.arange(60), np.arange(60, 80)
    encoder = AnchorEncoder(8, 20, 64).fit(epochs[train], [labels[i] for i in train], ridge=0.1)
    report = evaluate_retrieval(encoder.encode(epochs[test]), [labels[i] for i in test], vocab)
    assert report.top5 > max(report.baselines.values())


def test_npy_pickle_is_rejected(tmp_path, monkeypatch):
    from src.preprocessing import io as iomod

    monkeypatch.setattr(iomod, "_allowed_roots", lambda: [tmp_path.resolve()])
    path = tmp_path / "evil.npy"
    np.save(path, np.array([{"rce": True}], dtype=object), allow_pickle=True)
    with pytest.raises((ValueError, OSError)):
        iomod.load_npy_safe(str(path))


def test_production_jwt_secret_is_fail_closed(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    from src.config import Settings, _DEFAULT_JWT_SECRET

    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(env="production", jwt_secret=_DEFAULT_JWT_SECRET)


def test_d05_loader_uses_topic_labels_and_real_data():
    from src.training.dataset import load_d05

    data = load_d05(max_rows=32, n_times=32, label_field="topic", max_shards=2)
    assert data.epochs.shape == (32, 32, 32)
    assert data.source.startswith("d05")
    assert len(data.labels) == 32
    assert all(label for label in data.labels)
    assert len(set(data.session_ids)) >= 2


def test_generator_marks_kg_as_suggestion_not_eeg_read():
    prompt = {
        "subject": {"label": "window"},
        "attributes": [{"status": "suggested", "value": "natural light", "provenance": ["kg:edge:window_light"]}],
    }
    text = generate_text(prompt, [{"surface": "window"}], use_llm=False)
    assert "not EEG-read" in text
