"""Full-scale BCICIV-2a artifact lineage and signing contracts."""
import json
import shutil
import subprocess

import pytest

from src.training.train import save_artifacts


def _ed25519_keys(tmp_path):
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("OpenSSL CLI is required for detached Ed25519 signature tests")
    tmp_path.mkdir(parents=True, exist_ok=True)
    private = tmp_path / "signing-key.pem"
    public = tmp_path / "verification-key.pem"
    subprocess.run(
        [openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [openssl, "pkey", "-in", str(private), "-pubout", "-out", str(public)],
        check=True,
        capture_output=True,
    )
    return private.read_bytes(), public.read_bytes()


def test_failed_fullscale_gate_writes_lineage_without_encoder(tmp_path):
    result = {
        "source": "bciciv2a:mi4",
        "n_epochs": 2448,
        "n_channels": 16,
        "n_times": 32,
        "gate_pass": False,
        "encoder": object(),
        "top5": 0.8,
        "controls": {"shuffled_label_top5": 0.8},
    }
    assert save_artifacts(result, tmp_path) is None
    summary = json.loads((tmp_path / "train_report.json").read_text(encoding="utf-8"))
    assert summary["deployment_status"] == "blocked_below_control"
    assert summary["model_artifact"] is None
    assert not (tmp_path / "encoder.npz").exists()


def test_fullscale_manifest_requires_source_sha_and_split_counts(tmp_path):
    from scripts.artifact_manifest import build_manifest

    manifest = build_manifest(
        artifact_dir=tmp_path,
        source_csv=tmp_path / "source.csv",
        result={
            "source": "bciciv2a:mi4",
            "n_epochs": 2448,
            "n_channels": 16,
            "n_times": 32,
            "gate_pass": True,
            "cross_session": {"n_folds": 2448},
            "loso": {"n_folds": 9},
        },
        subject_counts={"sub-01": 288, "sub-02": 288},
    )
    assert manifest["dataset"]["n_epochs"] == 2448
    assert manifest["dataset"]["source_sha256"]
    assert manifest["evaluation"]["cross_session_folds"] == 2448
    assert manifest["evaluation"]["loso_folds"] == 9


def test_manifest_signature_verifies_with_public_key(tmp_path):
    from scripts.artifact_manifest import build_manifest, write_signed_manifest, verify_manifest_signature

    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    private_key, public_key = _ed25519_keys(tmp_path)
    other_private, other_public = _ed25519_keys(tmp_path / "other")
    manifest = build_manifest(
        artifact_dir=tmp_path,
        source_csv=source,
        result={"source": "bciciv2a:mi4", "n_epochs": 1, "n_channels": 1, "n_times": 1, "gate_pass": False},
        subject_counts={"sub-01": 1},
    )
    path = write_signed_manifest(manifest, tmp_path / "manifest.json", private_key)
    assert verify_manifest_signature(path, public_key) is True
    assert verify_manifest_signature(path, other_public) is False


def test_manifest_tampering_is_rejected(tmp_path):
    from scripts.artifact_manifest import build_manifest, write_signed_manifest, verify_manifest_signature

    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    private_key, public_key = _ed25519_keys(tmp_path)
    manifest = build_manifest(
        artifact_dir=tmp_path,
        source_csv=source,
        result={"source": "bciciv2a:mi4", "n_epochs": 1, "n_channels": 1, "n_times": 1, "gate_pass": False},
        subject_counts={"sub-01": 1},
    )
    path = write_signed_manifest(manifest, tmp_path / "manifest.json", private_key)
    path.write_text(path.read_text(encoding="utf-8").replace('"n_epochs": 1', '"n_epochs": 2'), encoding="utf-8")
    assert verify_manifest_signature(path, public_key) is False


def test_save_artifacts_writes_blocked_lineage_manifest(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    result = {
        "source": "bciciv2a:mi4",
        "source_file": str(source),
        "n_epochs": 2448,
        "n_channels": 16,
        "n_times": 32,
        "gate_pass": False,
        "encoder": object(),
        "cross_session": {"n_folds": 2448},
        "loso": {"n_folds": 9},
    }
    assert save_artifacts(result, tmp_path) is None
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["source_sha256"]
    assert manifest["evaluation"]["cross_session_folds"] == 2448
    assert manifest["evaluation"]["loso_folds"] == 9
    assert manifest["evaluation"]["gate_pass"] is False
