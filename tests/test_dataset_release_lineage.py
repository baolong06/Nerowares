"""Release metadata on DatasetHandle and training lineage without fabricating catalog-only fields."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.data.manifest import DatasetManifest, DatasetShard
from src.data.source import DatasetHandle, LocalDatasetSource
from src.training.dataset import _BCICIV2A_EEG_COLUMNS

_IDENTITY_KEYS = (
    "dataset_id",
    "revision",
    "manifest_sha256",
    "provider",
    "mode",
)
_RELEASE_KEYS = (
    "release_id",
    "scope_status",
    "manifest_signature_status",
    "governance_status",
    "file_count",
    "total_size_bytes",
)


def _manifest(
    path: Path,
    *,
    provider: str = "local",
    repo_id: str | None = None,
    revision: str = "local-fixture-v1",
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="bciciv2a",
        provider=provider,
        repo_id=repo_id,
        revision=revision,
        format="csv",
        sampling_rate_hz=250.0,
        n_channels_source=22,
        n_times_source=201,
        labels=("left", "right", "foot", "tongue"),
        subject_field="patient",
        session_field="epoch",
        shards=(
            DatasetShard(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        ),
    )


def _write_tiny_bciciv2a_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "time", "label", "epoch", *_BCICIV2A_EEG_COLUMNS])
        for index in range(201):
            writer.writerow(["1", f"{index * 0.004:.6f}", "left", "1", *("0" for _ in range(22))])
    return path


def make_verified_handle_with_release_metadata() -> DatasetHandle:
    manifest = DatasetManifest(
        dataset_id="bciciv2a",
        provider="local",
        repo_id=None,
        revision="local-fixture-v1",
        format="csv",
        sampling_rate_hz=250.0,
        n_channels_source=22,
        n_times_source=201,
        labels=("left", "right", "foot", "tongue"),
        subject_field="patient",
        session_field="epoch",
        shards=(
            DatasetShard(path="data.csv", size_bytes=7, sha256="0" * 64),
        ),
    )
    return DatasetHandle(
        dataset_id=manifest.dataset_id,
        revision=manifest.revision,
        manifest=manifest,
        files=(Path("data.csv"),),
        manifest_sha256=manifest.sha256(),
        mode="local",
        release_id="thinking-d15-observed-v1",
        scope_status="observed-local-package",
        manifest_signature_status="verified",
        governance_status="private-team",
        file_count=1,
        total_size_bytes=7,
    )


def _assert_no_secrets(lineage: dict[str, object]) -> None:
    dumped = json.dumps(lineage)
    assert "HF_TOKEN" not in dumped
    assert "raw_payload" not in dumped
    assert "token" not in dumped.lower()
    assert "private_key" not in dumped.lower()
    assert "secret" not in dumped.lower()


def test_release_metadata_is_preserved_without_token_or_raw_content(tmp_path):
    handle = make_verified_handle_with_release_metadata()
    lineage = handle.to_lineage()

    assert lineage["manifest_signature_status"] == "verified"
    assert "HF_TOKEN" not in json.dumps(lineage)
    assert "raw_payload" not in json.dumps(lineage)
    _assert_no_secrets(lineage)


def test_catalog_only_handle_lineage_omits_release_fields(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("payload", encoding="utf-8")
    manifest = _manifest(path)
    handle = DatasetHandle(
        dataset_id=manifest.dataset_id,
        revision=manifest.revision,
        manifest=manifest,
        files=(path,),
        manifest_sha256=manifest.sha256(),
        mode="local",
    )

    lineage = handle.to_lineage()

    assert lineage["dataset_id"] == "bciciv2a"
    assert lineage["revision"] == "local-fixture-v1"
    assert lineage["manifest_sha256"] == manifest.sha256()
    assert lineage["provider"] == "local"
    assert lineage["mode"] == "local"
    for key in _IDENTITY_KEYS:
        assert key in lineage
    for key in _RELEASE_KEYS:
        assert key not in lineage
    assert "files" not in lineage
    assert "manifest" not in lineage
    _assert_no_secrets(lineage)


def test_local_source_resolve_without_release_kwargs_leaves_fields_none(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("payload", encoding="utf-8")

    handle = LocalDatasetSource(root=tmp_path, manifest=_manifest(path)).resolve()

    assert handle.release_id is None
    assert handle.scope_status is None
    assert handle.manifest_signature_status is None
    assert handle.governance_status is None
    assert handle.file_count is None
    assert handle.total_size_bytes is None
    for key in _RELEASE_KEYS:
        assert key not in handle.to_lineage()


def test_resolve_dataset_source_without_release_kwargs_does_not_fabricate_metadata(
    tmp_path, monkeypatch
):
    from src.data.resolve import resolve_dataset_source

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("payload", encoding="utf-8")
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    manifest = _manifest(csv_path)
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    monkeypatch.setenv("THINKING_DATA_ROOT", str(tmp_path))

    source = resolve_dataset_source(
        dataset_id="bciciv2a",
        provider="local",
        catalog_dir=catalog_dir,
    )
    handle = source.resolve()

    for key in _RELEASE_KEYS:
        assert getattr(handle, key) is None
        assert key not in handle.to_lineage()


def test_resolve_dataset_source_copies_release_kwargs_into_handle_and_lineage(
    tmp_path, monkeypatch
):
    from src.data.resolve import resolve_dataset_source

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("payload", encoding="utf-8")
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    manifest = _manifest(csv_path)
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    monkeypatch.setenv("THINKING_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "not-for-lineage")

    source = resolve_dataset_source(
        dataset_id="bciciv2a",
        provider="local",
        catalog_dir=catalog_dir,
        release_id="thinking-d15-observed-v1",
        scope_status="observed-local-package",
        manifest_signature_status="verified",
        governance_status="private-team",
        file_count=1,
        total_size_bytes=csv_path.stat().st_size,
    )
    handle = source.resolve()
    lineage = handle.to_lineage()

    assert handle.release_id == "thinking-d15-observed-v1"
    assert handle.scope_status == "observed-local-package"
    assert handle.manifest_signature_status == "verified"
    assert handle.governance_status == "private-team"
    assert handle.file_count == 1
    assert handle.total_size_bytes == csv_path.stat().st_size
    assert lineage["release_id"] == "thinking-d15-observed-v1"
    assert lineage["scope_status"] == "observed-local-package"
    assert lineage["manifest_signature_status"] == "verified"
    assert lineage["governance_status"] == "private-team"
    assert lineage["file_count"] == 1
    assert lineage["total_size_bytes"] == csv_path.stat().st_size
    assert "not-for-lineage" not in json.dumps(lineage)
    _assert_no_secrets(lineage)


def test_huggingface_source_forwards_release_kwargs_without_hub(tmp_path, monkeypatch):
    from src.data.huggingface import HuggingFaceDatasetSource

    payload = tmp_path / "downloaded.csv"
    payload.write_text("payload", encoding="utf-8")
    manifest = DatasetManifest(
        dataset_id="bciciv2a",
        provider="huggingface",
        repo_id="baolong06/thinking-bciciv2a",
        revision="a" * 40,
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
    monkeypatch.setenv("HF_TOKEN", "not-for-lineage")

    handle = HuggingFaceDatasetSource(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        downloader=lambda **_: str(payload),
        release_id="thinking-d15-observed-v1",
        manifest_signature_status="verified",
        file_count=1,
        total_size_bytes=payload.stat().st_size,
    ).resolve()
    lineage = handle.to_lineage()

    assert handle.mode == "hf-cache"
    assert handle.release_id == "thinking-d15-observed-v1"
    assert handle.manifest_signature_status == "verified"
    assert lineage["provider"] == "huggingface"
    assert lineage["mode"] == "hf-cache"
    assert lineage["manifest_signature_status"] == "verified"
    assert "not-for-lineage" not in json.dumps(lineage)
    _assert_no_secrets(lineage)


def test_load_bciciv2a_epochs_catalog_only_keeps_identity_lineage(tmp_path):
    from src.training.dataset import load_bciciv2a_epochs

    csv_path = tmp_path / "data.csv"
    _write_tiny_bciciv2a_csv(csv_path)
    source = LocalDatasetSource(root=tmp_path, manifest=_manifest(csv_path))

    data = load_bciciv2a_epochs(
        dataset_source=source,
        max_epochs=1,
        n_times=16,
        n_channels=8,
        use_autoreject=False,
        use_ica=False,
    )

    lineage = data.data_lineage
    assert lineage is not None
    assert lineage["dataset_id"] == "bciciv2a"
    assert lineage["revision"] == "local-fixture-v1"
    assert lineage["manifest_sha256"] == source.manifest.sha256()
    assert lineage["provider"] == "local"
    assert lineage["mode"] == "local"
    for key in _RELEASE_KEYS:
        assert key not in lineage
    _assert_no_secrets(lineage)


def test_load_bciciv2a_epochs_propagates_verified_signature_status(tmp_path):
    from src.training.dataset import load_bciciv2a_epochs

    csv_path = tmp_path / "data.csv"
    _write_tiny_bciciv2a_csv(csv_path)
    source = LocalDatasetSource(
        root=tmp_path,
        manifest=_manifest(csv_path),
        manifest_signature_status="verified",
    )

    data = load_bciciv2a_epochs(
        dataset_source=source,
        max_epochs=1,
        n_times=16,
        n_channels=8,
        use_autoreject=False,
        use_ica=False,
    )

    assert data.data_lineage is not None
    assert data.data_lineage["manifest_signature_status"] == "verified"
    assert data.data_lineage["dataset_id"] == "bciciv2a"
    _assert_no_secrets(data.data_lineage)
