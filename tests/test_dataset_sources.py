"""Dataset source contracts for reproducible local and Hugging Face inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import pytest

from src.data.manifest import DatasetManifest, DatasetShard
from src.data.source import LocalDatasetSource


def _manifest(path: Path) -> DatasetManifest:
    return DatasetManifest(
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
            DatasetShard(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        ),
    )


def test_manifest_round_trip_and_rejects_moving_revision(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("patient,epoch\n1,1\n", encoding="utf-8")
    manifest = _manifest(source)
    document = manifest.to_dict()
    restored = DatasetManifest.from_dict(document)

    assert restored == manifest
    assert restored.require_revision() == "local-fixture-v1"
    with pytest.raises(ValueError, match="revision"):
        DatasetManifest.from_dict({**document, "revision": "main"})


def test_local_source_verifies_shard_before_returning_path(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("payload", encoding="utf-8")
    manifest = _manifest(source)
    catalog = tmp_path / "manifest.json"
    catalog.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")

    handle = LocalDatasetSource(root=tmp_path, manifest=manifest).resolve()

    assert handle.revision == "local-fixture-v1"
    assert handle.files == (source.resolve(),)
    assert handle.manifest_sha256 == manifest.sha256()
    assert catalog.exists()


def test_local_source_fails_closed_on_checksum_mismatch(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("payload", encoding="utf-8")
    document = _manifest(source).to_dict()
    document["shards"][0]["sha256"] = "0" * 64
    manifest = DatasetManifest.from_dict(document)

    with pytest.raises(ValueError, match="checksum"):
        LocalDatasetSource(root=tmp_path, manifest=manifest).resolve()


def test_huggingface_source_requires_immutable_revision():
    from src.data.huggingface import HuggingFaceDatasetSource

    with pytest.raises(ValueError, match="revision"):
        HuggingFaceDatasetSource(
            repo_id="baolong06/thinking-bciciv2a",
            revision="main",
            manifest=DatasetManifest(
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
                shards=(DatasetShard(path="data.csv", size_bytes=0, sha256="0" * 64),),
            ),
            cache_dir=Path("cache"),
        )


def test_huggingface_source_downloads_and_verifies_shards(tmp_path):
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
    calls: list[dict] = []

    def downloader(**kwargs):
        calls.append(kwargs)
        return str(payload)

    handle = HuggingFaceDatasetSource(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        downloader=downloader,
    ).resolve()

    assert handle.mode == "hf-cache"
    assert handle.files == (payload.resolve(),)
    assert calls[0]["revision"] == manifest.revision
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["local_files_only"] is False


def test_bciciv2a_loader_accepts_huggingface_blob_without_csv_suffix(tmp_path):
    from src.data.huggingface import HuggingFaceDatasetSource
    from src.training.dataset import _BCICIV2A_EEG_COLUMNS, load_bciciv2a_epochs

    blob = tmp_path / "cache-blob"
    with blob.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "time", "label", "epoch", *_BCICIV2A_EEG_COLUMNS])
        for index in range(201):
            writer.writerow(["1", f"{index * 0.004:.6f}", "left", "1", *("0" for _ in range(22))])

    manifest = DatasetManifest(
        dataset_id="bciciv2a",
        provider="huggingface",
        repo_id="madteam/thinking-bciciv2a",
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
                path="BCICIV_2a_all_patients.csv",
                size_bytes=blob.stat().st_size,
                sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
            ),
        ),
    )

    source = HuggingFaceDatasetSource(
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        manifest=manifest,
        cache_dir=tmp_path / "cache",
        downloader=lambda **_: str(blob),
    )

    data = load_bciciv2a_epochs(
        dataset_source=source,
        max_epochs=1,
        n_times=16,
        n_channels=8,
        use_autoreject=False,
        use_ica=False,
    )

    assert data.epochs.shape == (1, 8, 16)
    assert data.data_lineage["provider"] == "huggingface"


def test_bciciv2a_loader_accepts_verified_dataset_source(tmp_path):
    from src.training.dataset import _BCICIV2A_EEG_COLUMNS, load_bciciv2a_epochs

    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "time", "label", "epoch", *_BCICIV2A_EEG_COLUMNS])
        for index in range(201):
            writer.writerow(["1", f"{index * 0.004:.6f}", "left", "1", *(["0"] * 22)])
    source = LocalDatasetSource(root=tmp_path, manifest=_manifest(csv_path))

    data = load_bciciv2a_epochs(
        dataset_source=source,
        max_epochs=1,
        n_times=16,
        n_channels=8,
        use_autoreject=False,
        use_ica=False,
    )

    assert data.epochs.shape == (1, 8, 16)
    assert data.data_lineage is not None
    assert data.data_lineage["revision"] == "local-fixture-v1"
    assert data.data_lineage["manifest_sha256"] == source.manifest.sha256()


def test_training_result_preserves_dataset_lineage(tmp_path):
    from src.anchors.encoder import label_template
    from src.training.dataset import LabeledEpochs
    from src.training.train import train_and_eval

    labels = ["left", "right"] * 4
    epochs = np.stack([
        label_template(label, 2, 6) for label in labels
    ]).astype(np.float32)
    lineage = {
        "dataset_id": "bciciv2a",
        "revision": "local-fixture-v1",
        "manifest_sha256": "a" * 64,
        "provider": "local",
        "mode": "local",
    }
    data = LabeledEpochs(
        epochs=epochs,
        labels=labels,
        subject_ids=[f"sub-{index % 2}" for index in range(len(labels))],
        session_ids=[f"sub-{index % 2}:epoch-{index:03d}" for index in range(len(labels))],
        source="bciciv2a:mi4",
        data_lineage=lineage,
    )

    result = train_and_eval(data, ridge=0.1, embed_dim=8)

    assert result["data_lineage"] == lineage


def test_catalog_loads_pinned_local_manifest(tmp_path):
    from src.data.catalog import load_catalog_manifest

    csv_path = tmp_path / "BCICIV_2a_all_patients.csv"
    csv_path.write_text("patient,epoch\n1,1\n", encoding="utf-8")
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    manifest = _manifest(csv_path)
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )

    loaded = load_catalog_manifest("bciciv2a", provider="local", catalog_dir=catalog_dir)

    assert loaded.dataset_id == "bciciv2a"
    assert loaded.provider == "local"
    assert loaded.require_revision() == "local-fixture-v1"


def test_catalog_rejects_huggingface_moving_revision(tmp_path):
    from src.data.catalog import load_catalog_manifest

    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    (catalog_dir / "bciciv2a.huggingface.json").write_text(
        json.dumps(
            {
                "dataset_id": "bciciv2a",
                "provider": "huggingface",
                "repo_id": "baolong06/thinking-bciciv2a",
                "revision": "main",
                "format": "csv",
                "sampling_rate_hz": 250.0,
                "n_channels_source": 22,
                "n_times_source": 201,
                "labels": ["left", "right", "foot", "tongue"],
                "subject_field": "patient",
                "session_field": "epoch",
                "shards": [{"path": "data.csv", "size_bytes": 1, "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="revision"):
        load_catalog_manifest("bciciv2a", provider="huggingface", catalog_dir=catalog_dir)


def test_resolve_local_source_uses_data_root_not_hardcoded_hub_path(tmp_path, monkeypatch):
    from src.data.resolve import resolve_dataset_source

    csv_path = tmp_path / "kaggle" / "aymanmostafa11__eeg-motor-imagery-bciciv-2a" / "BCICIV_2a_all_patients.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("payload", encoding="utf-8")
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
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
            DatasetShard(
                path="kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv",
                size_bytes=csv_path.stat().st_size,
                sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            ),
        ),
    )
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )
    monkeypatch.setenv("THINKING_DATA_ROOT", str(tmp_path))

    source = resolve_dataset_source(
        dataset_id="bciciv2a",
        provider="local",
        catalog_dir=catalog_dir,
    )
    handle = source.resolve()

    assert handle.files == (csv_path.resolve(),)
    assert handle.mode == "local"


def test_train_cli_rejects_mutable_huggingface_revision(tmp_path):
    from src.training.train import main

    with pytest.raises(ValueError, match="revision"):
        main([
            "--dataset", "bciciv2a",
            "--data-source", "huggingface",
            "--repo-id", "baolong06/thinking-bciciv2a",
            "--revision", "main",
            "--max-rows", "1",
            "--artifacts-dir", str(tmp_path),
        ])


def test_dataset_info_reports_manifest_metadata_without_token(tmp_path, monkeypatch, capsys):
    from scripts.dataset_info import main

    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("payload", encoding="utf-8")
    manifest = _manifest(csv_path)
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    monkeypatch.setenv("HF_TOKEN", "not-for-output")

    assert main([
        "--dataset", "bciciv2a",
        "--provider", "local",
        "--catalog-dir", str(catalog_dir),
    ]) == 0

    output = capsys.readouterr().out
    assert '"dataset_id": "bciciv2a"' in output
    assert '"manifest_sha256"' in output
    assert "not-for-output" not in output


def test_dataset_verify_resolves_and_verifies_complete_local_manifest(
    tmp_path, monkeypatch, capsys
):
    from scripts.dataset_verify import main

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("payload", encoding="utf-8")
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    manifest = _manifest(csv_path)
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    monkeypatch.setenv("THINKING_DATA_ROOT", str(tmp_path))

    assert main([
        "--dataset", "bciciv2a",
        "--provider", "local",
        "--catalog-dir", str(catalog_dir),
    ]) == 0

    output = capsys.readouterr().out
    document = json.loads(output)
    assert document["status"] == "verified"
    assert document["files"] == ["data.csv"]


def test_dataset_download_resolves_every_shard_once(tmp_path, monkeypatch, capsys):
    from scripts.dataset_download import main

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = DatasetManifest(
        dataset_id="bciciv2a",
        provider="local",
        repo_id=None,
        revision="local-fixture-v2",
        format="csv",
        sampling_rate_hz=250.0,
        n_channels_source=22,
        n_times_source=201,
        labels=("left", "right", "foot", "tongue"),
        subject_field="patient",
        session_field="epoch",
        shards=tuple(
            DatasetShard(
                path=path.name,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in (first, second)
        ),
    )
    catalog_dir = tmp_path / "dataset_catalog"
    catalog_dir.mkdir()
    (catalog_dir / "bciciv2a.local.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    monkeypatch.setenv("THINKING_DATA_ROOT", str(tmp_path))

    assert main([
        "--dataset", "bciciv2a",
        "--provider", "local",
        "--catalog-dir", str(catalog_dir),
    ]) == 0

    output = capsys.readouterr().out
    assert '"status": "downloaded"' in output
    assert '"first.csv"' in output
    assert '"second.csv"' in output


def test_committed_bciciv2a_local_catalog_matches_measured_shard():
    from src.data.catalog import load_catalog_manifest

    manifest = load_catalog_manifest("bciciv2a", provider="local")

    assert manifest.dataset_id == "bciciv2a"
    assert manifest.provider == "local"
    assert manifest.revision == "local-bciciv2a-2026-09-24"
    assert manifest.shards[0].path == (
        "kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a/"
        "BCICIV_2a_all_patients.csv"
    )
    assert manifest.shards[0].size_bytes == 214848010
    assert manifest.shards[0].sha256 == (
        "fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd"
    )


def test_committed_bciciv2a_huggingface_catalog_is_immutable():
    from src.data.catalog import load_catalog_manifest

    manifest = load_catalog_manifest("bciciv2a", provider="huggingface")

    assert manifest.repo_id == "madteam/thinking-bciciv2a"
    assert len(manifest.require_revision()) == 40
    assert manifest.shards[0].path == "BCICIV_2a_all_patients.csv"
    assert manifest.shards[0].size_bytes == 214848010
    assert manifest.shards[0].sha256 == (
        "fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd"
    )
