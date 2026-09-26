"""Family-release verification, download preflight, CLI flags, and operator docs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.data.manifest import DatasetManifest, DatasetShard
from src.data.release.models import FamilyReleaseManifest
from tests.test_dataset_release_signing import make_ed25519_pem_pair

_WORKTREE = Path(__file__).resolve().parents[1]
_IMMUTABLE_REVISION = "a" * 40
_RELEASE_KEYS = ("release_id", "manifest_signature_status")
_PARSER_FLAGS = (
    "release_manifest",
    "signature",
    "verification_key",
    "max_cache_bytes",
    "min_free_bytes",
)
_TELEMETRY_EVENTS = (
    "dataset_inventory_reconciliation_failed",
    "dataset_release_plan_created",
    "dataset_shard_build_failed",
    "dataset_upload_blocked",
    "dataset_upload_batch_failed",
    "dataset_release_signature_failed",
    "dataset_shard_verification_failed",
    "dataset_release_verified",
    "dataset_release_verification_failed",
)
_OPERATOR_TOPICS = (
    "plan generation",
    "governance review",
    "direct upload",
    "clean-machine download",
    "cache verification",
    "family-scoped training",
    "artifact lineage",
    "private repository credentials",
    "observed-local-package",
    "full-upstream",
)


def _governance(*, visibility: str = "private-team") -> dict[str, object]:
    return {
        "visibility": visibility,
        "license_status": "verified",
        "dua_status": "not_applicable",
        "custodian_status": "verified",
        "deidentification_status": "reviewed",
        "approved_by": "custodian",
        "evidence_reference": "ticket-1",
    }


def _file_record(path: str = "sample.edf", payload: bytes = b"sample-edf") -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _family_manifest_dict(**overrides: object) -> dict[str, object]:
    files = overrides.pop("files", None)
    if files is None:
        files = [_file_record()]
    document: dict[str, object] = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "governance": _governance(),
        "name": "inner-speech",
        "source": "openneuro",
        "license_note": "research-use",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "revision": _IMMUTABLE_REVISION,
    }
    document.update(overrides)
    return document


def _write_family_json(path: Path, **overrides: object) -> dict[str, object]:
    document = _family_manifest_dict(**overrides)
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return document


def _sign_family_json(path: Path) -> tuple[Path, bytes]:
    from src.data.release.signing import write_detached_signature

    private_pem, public_pem = make_ed25519_pem_pair()
    signature = write_detached_signature(path, private_pem)
    return signature, public_pem


def _hf_catalog_manifest(payload: Path) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="bciciv2a",
        provider="huggingface",
        repo_id="baolong06/thinking-bciciv2a",
        revision=_IMMUTABLE_REVISION,
        format="csv",
        sampling_rate_hz=250.0,
        n_channels_source=22,
        n_times_source=201,
        labels=("left", "right", "foot", "tongue"),
        subject_field="patient",
        session_field="epoch",
        shards=(
            DatasetShard(
                path="data.csv",
                size_bytes=payload.stat().st_size,
                sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
            ),
        ),
    )


def test_signed_family_manifest_is_required_before_download(tmp_path):
    from src.data.huggingface import verify_release_manifest

    _, public_key = make_ed25519_pem_pair()
    manifest_path = tmp_path / "release_manifest.json"
    _write_family_json(manifest_path)

    with pytest.raises(ValueError, match="signature"):
        verify_release_manifest(
            manifest_path=manifest_path,
            signature_path=tmp_path / "release_manifest.json.sig",
            verification_key=public_key,
        )


def test_download_preflight_rejects_insufficient_quota_without_network(tmp_path):
    from src.data.huggingface import HuggingFaceDatasetSource

    payload = tmp_path / "downloaded.csv"
    payload.write_text("payload", encoding="utf-8")
    manifest = _hf_catalog_manifest(payload)
    calls: list[dict] = []

    def downloader(**kwargs):
        calls.append(kwargs)
        raise AssertionError("downloader must not be called during quota preflight")

    source = HuggingFaceDatasetSource(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        downloader=downloader,
        max_cache_bytes=1,
    )

    with pytest.raises(ValueError, match="quota"):
        source.resolve()
    assert calls == []


def test_preflight_download_quota_rejects_free_space_with_injected_disk(tmp_path):
    from src.data.huggingface import preflight_download_quota

    class Usage:
        def __init__(self, free: int) -> None:
            self.free = free

    inspected: list[Path] = []

    def disk_usage(path):
        inspected.append(Path(path))
        return Usage(free=1)

    with pytest.raises(ValueError, match="free"):
        preflight_download_quota(
            total_size_bytes=100,
            cache_dir=tmp_path / "cache",
            min_free_bytes=50,
            disk_usage=disk_usage,
        )
    assert inspected


def test_signed_family_manifest_verifies_without_network(tmp_path):
    from src.data.huggingface import VerifiedRelease, verify_release_manifest

    manifest_path = tmp_path / "release_manifest.json"
    document = _write_family_json(manifest_path)
    signature_path, public_pem = _sign_family_json(manifest_path)

    verified = verify_release_manifest(
        manifest_path=manifest_path,
        signature_path=signature_path,
        verification_key=public_pem,
    )

    assert isinstance(verified, VerifiedRelease)
    assert verified.dataset_id == document["dataset_id"]
    assert verified.release_id == document["release_id"]
    assert verified.repo_id == document["repo_id"]
    assert verified.revision == _IMMUTABLE_REVISION
    assert verified.manifest_signature_status == "verified"
    assert verified.governance_status == "private-team"
    assert verified.scope_status == "observed-local-package"
    assert verified.file_count == 1
    assert verified.total_size_bytes == document["total_size_bytes"]
    assert verified.manifest.dataset_id == "D01"
    assert not list(tmp_path.rglob("*.pem"))


def test_tampered_or_wrong_key_family_manifest_fails_signature(tmp_path):
    from src.data.huggingface import verify_release_manifest

    manifest_path = tmp_path / "release_manifest.json"
    document = _write_family_json(manifest_path)
    signature_path, public_pem = _sign_family_json(manifest_path)
    _, other_public = make_ed25519_pem_pair()

    with pytest.raises(ValueError, match="signature"):
        verify_release_manifest(
            manifest_path=manifest_path,
            signature_path=signature_path,
            verification_key=other_public,
        )

    document["release_id"] = "thinking-d01-tampered"
    manifest_path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        verify_release_manifest(
            manifest_path=manifest_path,
            signature_path=signature_path,
            verification_key=public_pem,
        )


def test_moving_revision_on_signed_family_release_fails_closed(tmp_path):
    from src.data.huggingface import verify_release_manifest

    manifest_path = tmp_path / "release_manifest.json"
    _write_family_json(manifest_path, revision="main")
    signature_path, public_pem = _sign_family_json(manifest_path)

    with pytest.raises(ValueError, match="revision"):
        verify_release_manifest(
            manifest_path=manifest_path,
            signature_path=signature_path,
            verification_key=public_pem,
        )


def test_catalog_only_huggingface_source_omits_release_lineage(tmp_path):
    from src.data.huggingface import HuggingFaceDatasetSource

    payload = tmp_path / "downloaded.csv"
    payload.write_text("payload", encoding="utf-8")
    manifest = _hf_catalog_manifest(payload)

    def downloader(**kwargs):
        assert kwargs["repo_type"] == "dataset"
        return str(payload)

    handle = HuggingFaceDatasetSource(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        downloader=downloader,
    ).resolve()

    lineage = handle.to_lineage()
    assert lineage["dataset_id"] == "bciciv2a"
    assert lineage["mode"] == "hf-cache"
    for key in _RELEASE_KEYS:
        assert key not in lineage
        assert getattr(handle, key) is None


def test_download_and_verify_parsers_expose_release_flags():
    from scripts.dataset_download import build_parser as download_parser
    from scripts.dataset_verify import build_parser as verify_parser

    for builder in (download_parser, verify_parser):
        dests = {action.dest for action in builder()._actions}
        for flag in _PARSER_FLAGS:
            assert flag in dests


def test_dataset_download_cli_requires_complete_release_triple(tmp_path):
    from scripts.dataset_download import build_parser, main

    dests = {action.dest for action in build_parser()._actions}
    assert {"release_manifest", "signature", "verification_key"} <= dests
    manifest_path = tmp_path / "release_manifest.json"
    _write_family_json(manifest_path)

    with pytest.raises(ValueError, match="signature"):
        main(
            [
                "--dataset",
                "D01",
                "--provider",
                "huggingface",
                "--release-manifest",
                str(manifest_path),
            ]
        )


def test_dataset_verify_cli_requires_complete_release_triple(tmp_path):
    from scripts.dataset_verify import build_parser, main

    dests = {action.dest for action in build_parser()._actions}
    assert {"release_manifest", "signature", "verification_key"} <= dests
    manifest_path = tmp_path / "release_manifest.json"
    _write_family_json(manifest_path)

    with pytest.raises(ValueError, match="signature"):
        main(
            [
                "--dataset",
                "D01",
                "--provider",
                "huggingface",
                "--release-manifest",
                str(manifest_path),
            ]
        )


def test_d03_full_upstream_still_fails_at_family_manifest_from_dict():
    document = _family_manifest_dict(dataset_id="D03", scope_status="full-upstream")
    with pytest.raises(ValueError, match="full-upstream"):
        FamilyReleaseManifest.from_dict(document)


def test_huggingface_exports_verified_release_api():
    from src.data import huggingface as hf_module

    for name in ("VerifiedRelease", "verify_release_manifest", "preflight_download_quota"):
        assert name in hf_module.__all__


def test_dataset_release_docs_cover_operator_runbook_and_telemetry():
    text = (_WORKTREE / "docs" / "dataset_release.md").read_text(encoding="utf-8")
    lowered = text.lower()
    for event in _TELEMETRY_EVENTS:
        assert event in text
    for topic in _OPERATOR_TOPICS:
        assert topic in lowered
    assert "subject-02" in lowered or "subject 02" in lowered
    assert "d03" in lowered
    assert "d15" in lowered
    assert "madteam/thinking-bciciv2a" in text
    assert "never overwrites" in lowered or "never overwrite" in lowered
    assert "hf_token" in text
    assert "repo_type=dataset" in lowered or "repo_type = dataset" in lowered or 'repo_type="dataset"' in text
    assert "{.bdf,.edf,.fif,.parquet,.npy,.npz,.csv}" in text or ".bdf,.edf,.fif,.parquet,.npy,.npz,.csv" in text


def test_catalog_and_architecture_docs_mention_family_release():
    catalog = (_WORKTREE / "dataset_catalog" / "README.md").read_text(encoding="utf-8")
    architecture = (_WORKTREE / "architecture" / "dataset_hub.md").read_text(encoding="utf-8")
    for text in (catalog, architecture):
        lowered = text.lower()
        assert "family release" in lowered or "family-release" in lowered
        assert "governance" in lowered
        assert "d15" in lowered
        assert "madteam/thinking-bciciv2a" in text
        assert "1cd0deae08404efd86bf6b84d6beb2575c245a5e" in text
        assert "no local raw copy" in lowered
