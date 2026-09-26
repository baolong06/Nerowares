"""Streaming inventory and canonical family-release manifest generation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.data.release.models import FamilyReleaseConfig, FamilyReleaseManifest
from src.data.release.scope import discover_scope


def _blocked_governance() -> dict[str, object]:
    return {
        "visibility": "blocked",
        "license_status": "unknown",
        "dua_status": "unknown",
        "custodian_status": "unknown",
        "deidentification_status": "not_reviewed",
        "approved_by": "",
        "evidence_reference": "",
    }


def make_config(**overrides: object) -> FamilyReleaseConfig:
    document: dict[str, object] = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "local_root": "family",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "governance": _blocked_governance(),
        "name": "inner-speech",
        "source": "openneuro",
        "license_note": "research-use",
    }
    document.update(overrides)
    return FamilyReleaseConfig.from_dict(document)


def _write_family_files(tmp_path: Path) -> Path:
    root = tmp_path / "family"
    (root / "nested").mkdir(parents=True)
    (root / "z.edf").write_bytes(b"z-payload")
    (root / "a.edf").write_bytes(b"a-payload")
    (root / "B.edf").write_bytes(b"B-payload")
    (root / "nested" / "b.edf").write_bytes(b"nested-b")
    (root / "keep.mat").write_bytes(b"opaque-matlab-bytes")
    return root


def test_inventory_records_exact_size_and_sha256(tmp_path):
    from src.data.release.inventory import build_family_manifest

    payload_bytes = b"sample-payload"
    root = tmp_path / "family"
    root.mkdir()
    payload = root / "sample.edf"
    payload.write_bytes(payload_bytes)
    config = make_config()
    manifest = build_family_manifest(config, discover_scope(config, tmp_path))

    assert manifest.file_count == 1
    assert manifest.total_size_bytes == len(payload_bytes)
    assert manifest.files[0].path == "sample.edf"
    assert manifest.files[0].size_bytes == len(payload_bytes)
    assert manifest.files[0].sha256 == hashlib.sha256(payload_bytes).hexdigest()
    assert manifest.dataset_id == config.dataset_id
    assert manifest.release_id == config.release_id
    assert manifest.repo_id == config.repo_id
    assert manifest.provider == config.provider
    assert manifest.scope_status == config.scope_status
    assert manifest.governance == config.governance
    assert manifest.revision == ""
    assert manifest.manifest_sha256 == ""
    assert manifest.manifest_signature_status == "unsigned"
    assert manifest.observed_local_files == 1
    assert manifest.observed_local_bytes == len(payload_bytes)
    assert manifest.provenance_references == config.provenance_references
    assert manifest.source == config.source
    assert manifest.name == config.name


def test_manifest_order_and_canonical_digest_are_deterministic(tmp_path):
    from src.data.release.inventory import build_family_manifest

    _write_family_files(tmp_path)
    config = make_config()
    first = build_family_manifest(config, discover_scope(config, tmp_path))
    second = build_family_manifest(config, discover_scope(config, tmp_path))

    assert [item.path for item in first.files] == [
        "B.edf",
        "a.edf",
        "keep.mat",
        "nested/b.edf",
        "z.edf",
    ]
    assert all("/" in item.path or "\\" not in item.path for item in first.files)
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.sha256() == second.sha256()
    assert first.files == second.files
    assert first.file_count == 5
    assert first.total_size_bytes == sum(item.size_bytes for item in first.files)
    payload = json.loads(first.canonical_bytes().decode("utf-8"))
    assert "manifest_sha256" not in payload
    assert "manifest_signature_status" not in payload
    assert "revision" not in payload


def test_hash_file_streams_bounded_chunks_and_counts_bytes_from_same_read(tmp_path, monkeypatch):
    from src.data.release.inventory import hash_file

    path = tmp_path / "blob.bin"
    payload = b"0123456789abcdef" * 40
    path.write_bytes(payload)
    original_stat = os.stat
    chunk_sizes: list[int] = []
    target_key = os.path.normcase(os.path.abspath(os.fspath(path)))

    def _same_file(candidate: object) -> bool:
        return os.path.normcase(os.path.abspath(os.fspath(candidate))) == target_key

    def lying_os_stat(target, *args, **kwargs):
        result = original_stat(target, *args, **kwargs)
        if _same_file(target):
            return result._replace(st_size=1)
        return result

    monkeypatch.setattr(os, "stat", lying_os_stat)

    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if not _same_file(self):
            return handle
        original_read = handle.read

        def tracking_read(size=-1):
            chunk = original_read(size)
            chunk_sizes.append(len(chunk))
            if isinstance(size, int) and size > 0:
                assert len(chunk) <= size
            return chunk

        handle.read = tracking_read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    size, digest = hash_file(path, chunk_size=16)
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert digest == digest.lower()
    assert chunk_sizes
    assert max(chunk_sizes) <= 16
    assert sum(chunk_sizes) == len(payload)


def test_write_manifest_pretty_json_is_not_canonical_bytes(tmp_path):
    from src.data.release.inventory import build_family_manifest, read_manifest, write_manifest

    root = tmp_path / "family"
    root.mkdir()
    (root / "sample.edf").write_bytes(b"sample-payload")
    config = make_config()
    manifest = build_family_manifest(config, discover_scope(config, tmp_path))
    path = write_manifest(manifest, tmp_path / "release_manifest.json")
    text = path.read_text(encoding="utf-8")
    pretty = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    assert path == tmp_path / "release_manifest.json"
    assert text == pretty
    assert text.endswith("\n")
    assert manifest.canonical_bytes() != text.encode("utf-8")
    restored = read_manifest(path)
    assert restored.files == manifest.files
    assert restored.file_count == manifest.file_count
    assert restored.total_size_bytes == manifest.total_size_bytes
    assert restored.scope_status == "observed-local-package"
    parsed = json.loads(text)
    assert "manifest_sha256" not in parsed
    assert "revision" not in parsed
    assert parsed["governance"] == manifest.governance.to_dict()
    assert parsed["provenance_references"] == list(config.provenance_references)
    assert parsed["source"] == config.source


def test_source_mutation_reports_old_and_new_digest_without_upload(tmp_path):
    from src.data.release.inventory import build_family_manifest, revalidate_source

    root = tmp_path / "family"
    root.mkdir()
    payload = root / "sample.edf"
    original = b"sample-payload"
    mutated = b"mutated-payload-bytes"
    payload.write_bytes(original)
    config = make_config()
    manifest = build_family_manifest(config, discover_scope(config, tmp_path))
    payload.write_bytes(mutated)

    mismatches = revalidate_source(manifest, config, tmp_path)

    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert mismatch.path == "sample.edf"
    assert mismatch.expected_size_bytes == len(original)
    assert mismatch.expected_sha256 == hashlib.sha256(original).hexdigest()
    assert mismatch.observed_size_bytes == len(mutated)
    assert mismatch.observed_sha256 == hashlib.sha256(mutated).hexdigest()
    source = Path("src/data/release/inventory.py").read_text(encoding="utf-8")
    assert "huggingface_hub" not in source
    assert "HubClient" not in source
    assert shutil_not_used_for_payload(source)


def shutil_not_used_for_payload(source: str) -> bool:
    return "shutil.copy" not in source and "zipfile" not in source and "tarfile" not in source


def test_inventory_fails_if_scoped_file_is_no_longer_regular(tmp_path):
    from src.data.release.inventory import build_family_manifest

    root = tmp_path / "family"
    root.mkdir()
    payload = root / "sample.edf"
    payload.write_bytes(b"sample-payload")
    config = make_config()
    scope = discover_scope(config, tmp_path)
    payload.unlink()
    payload.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        build_family_manifest(config, scope)


def test_inventory_fails_if_scoped_path_escapes_family_root(tmp_path):
    from src.data.release.inventory import build_family_manifest
    from src.data.release.scope import ScopeResult

    root = tmp_path / "family"
    root.mkdir()
    (root / "ok.edf").write_bytes(b"ok")
    outside = tmp_path / "outside.edf"
    outside.write_bytes(b"outside")
    config = make_config()
    scope = discover_scope(config, tmp_path)
    poisoned = ScopeResult(
        files=scope.files + (outside.resolve(),),
        excluded=scope.excluded,
        scope_sha256=scope.scope_sha256,
    )

    with pytest.raises(ValueError, match="family root"):
        build_family_manifest(config, poisoned)


def test_d03_stays_observed_local_package_when_expected_counts_differ(tmp_path):
    from src.data.release.inventory import build_family_manifest

    root = tmp_path / "family"
    root.mkdir()
    (root / "subject02.edf").write_bytes(b"chisco-subject-02")
    config = make_config(
        dataset_id="D03",
        release_id="thinking-d03-observed-v1",
        repo_id="madteam/thinking-d03",
        scope_status="observed-local-package",
        expected_source_files=99,
        expected_source_bytes=12345,
        provenance_references=["https://openneuro.org/datasets/ds005170"],
    )
    manifest = build_family_manifest(config, discover_scope(config, tmp_path))

    assert manifest.dataset_id == "D03"
    assert manifest.scope_status == "observed-local-package"
    assert manifest.scope_status != "full-upstream"
    assert manifest.expected_source_files == 99
    assert manifest.expected_source_bytes == 12345
    assert manifest.observed_local_files == 1
    assert manifest.observed_local_bytes == len(b"chisco-subject-02")
    assert manifest.observed_local_files != manifest.expected_source_files
    with pytest.raises(ValueError, match="observed-local-package"):
        FamilyReleaseManifest.from_dict(
            {
                **manifest.to_dict(),
                "scope_status": "full-upstream",
            }
        )


def test_parser_allowlist_remains_unchanged():
    from src.preprocessing.io import ALLOWED_EXTENSIONS

    assert ALLOWED_EXTENSIONS == {".bdf", ".edf", ".fif", ".parquet", ".npy", ".npz", ".csv"}


def test_inventory_hashes_opaque_mat_without_copy_or_parser(tmp_path):
    from src.data.release.inventory import build_family_manifest

    root = tmp_path / "family"
    root.mkdir()
    mat = root / "signal.mat"
    payload = b"opaque-matlab-bytes"
    mat.write_bytes(payload)
    (root / "keep.edf").write_bytes(b"edf")
    before = {path.resolve() for path in root.rglob("*")}
    config = make_config()
    manifest = build_family_manifest(config, discover_scope(config, tmp_path))
    after = {path.resolve() for path in root.rglob("*")}
    source = Path("src/data/release/inventory.py").read_text(encoding="utf-8")

    assert after == before
    paths = {item.path for item in manifest.files}
    assert paths == {"keep.edf", "signal.mat"}
    mat_record = next(item for item in manifest.files if item.path == "signal.mat")
    assert mat_record.size_bytes == len(payload)
    assert mat_record.sha256 == hashlib.sha256(payload).hexdigest()
    assert "mne" not in source
    assert "src.preprocessing" not in source
    assert "zipfile" not in source
    assert "tarfile" not in source
    assert "shutil.copy" not in source
    assert "make_archive" not in source


def test_inventory_fails_if_scoped_file_cannot_be_hashed(tmp_path, monkeypatch):
    from src.data.release import inventory

    root = tmp_path / "family"
    root.mkdir()
    (root / "sample.edf").write_bytes(b"sample-payload")
    config = make_config()
    scope = discover_scope(config, tmp_path)

    def boom(_path, chunk_size=1024 * 1024):
        raise OSError("cannot read")

    monkeypatch.setattr(inventory, "hash_file", boom)
    with pytest.raises(ValueError):
        inventory.build_family_manifest(config, scope)


def _try_symlink(link: Path, target: Path) -> bool:
    try:
        link.symlink_to(target)
    except OSError:
        return False
    return link.is_symlink()


def test_inventory_fails_if_scoped_file_is_replaced_by_symlink(tmp_path):
    from src.data.release.inventory import build_family_manifest

    root = tmp_path / "family"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = root / "sample.edf"
    payload.write_bytes(b"sample-payload")
    target = nested / "b.edf"
    target.write_bytes(b"nested-b-payload")
    config = make_config()
    scope = discover_scope(config, tmp_path)

    assert payload.resolve() in scope.files
    assert target.resolve() in scope.files
    assert all(path.is_file() and not path.is_symlink() for path in scope.files)

    payload.unlink()
    symlink_created = _try_symlink(payload, target)
    if not symlink_created:
        pytest.skip("symlink creation refused by the OS")

    with pytest.raises(ValueError, match="regular file"):
        build_family_manifest(config, scope)


def test_inventory_relative_paths_use_deepest_local_root_ancestor(tmp_path):
    from src.data.release.inventory import build_family_manifest, revalidate_source

    data_root = tmp_path / "family" / "data"
    family = data_root / "family"
    family.mkdir(parents=True)
    payload_bytes = b"sample-payload"
    (family / "sample.edf").write_bytes(payload_bytes)
    (family / "nested").mkdir()
    (family / "nested" / "b.edf").write_bytes(b"nested-b")
    config = make_config()
    scope = discover_scope(config, data_root)
    manifest = build_family_manifest(config, scope)

    assert [item.path for item in manifest.files] == ["nested/b.edf", "sample.edf"]
    assert manifest.files[1].sha256 == hashlib.sha256(payload_bytes).hexdigest()
    assert revalidate_source(manifest, config, data_root) == ()
