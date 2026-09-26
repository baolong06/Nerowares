"""Root-scoped deterministic family-release discovery."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from src.data.release.models import FamilyReleaseConfig


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


def make_config(local_root: str = "family") -> FamilyReleaseConfig:
    return FamilyReleaseConfig.from_dict(
        {
            "dataset_id": "D01",
            "release_id": "thinking-d01-observed-v1",
            "local_root": local_root,
            "repo_id": "madteam/thinking-d01",
            "provider": "huggingface",
            "scope_status": "observed-local-package",
            "visibility": "blocked",
            "provenance_references": ["https://openneuro.org/datasets/ds003626"],
            "governance": _blocked_governance(),
        }
    )


def expected_scope_sha256(relative_paths: list[str]) -> str:
    ordered = sorted(relative_paths, key=lambda item: item.encode("utf-8"))
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _try_symlink(link: Path, target: Path, *, target_is_directory: bool = False) -> bool:
    try:
        if target_is_directory:
            link.symlink_to(target, target_is_directory=True)
        else:
            link.symlink_to(target)
    except OSError:
        return False
    return link.is_symlink()


def test_scope_rejects_symlinks_and_paths_outside_family_root(tmp_path):
    from src.data.release.scope import discover_scope

    root = tmp_path / "family"
    root.mkdir()
    (root / "ok.edf").write_bytes(b"ok")
    outside = tmp_path / "outside.edf"
    outside.write_bytes(b"outside")
    sibling = tmp_path / "other"
    sibling.mkdir()
    (sibling / "leak.edf").write_bytes(b"leak")
    link = root / "link.edf"
    symlink_created = _try_symlink(link, outside)

    result = discover_scope(make_config(), tmp_path)

    assert result.files == ((root / "ok.edf").resolve(),)
    assert all(path.resolve() != outside.resolve() for path in result.files)
    assert all(path.name != "leak.edf" for path in result.files)
    if not symlink_created:
        pytest.skip("symlink creation refused by the OS")
    assert any(item.path == "link.edf" for item in result.excluded)


def test_scope_is_sorted_and_excludes_control_files(tmp_path):
    from src.data.release.scope import discover_scope

    root = tmp_path / "family"
    (root / "nested").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "cache").mkdir()
    (root / "__pycache__").mkdir()
    (root / ".cache").mkdir()
    (root / "reports").mkdir()
    (root / "z.edf").write_bytes(b"z")
    (root / "a.edf").write_bytes(b"a")
    (root / "B.edf").write_bytes(b"B")
    (root / "keep.mat").write_bytes(b"matlab-bytes")
    (root / "nested" / "b.edf").write_bytes(b"b")
    (root / ".git" / "config").write_text("gitdir", encoding="utf-8")
    (root / "cache" / "tmp.bin").write_bytes(b"cache")
    (root / "__pycache__" / "mod.pyc").write_bytes(b"pyc")
    (root / ".cache" / "hf.bin").write_bytes(b"hf")
    (root / "download_dataset.ps1").write_text("download", encoding="utf-8")
    (root / "download.sh").write_text("download", encoding="utf-8")
    (root / "reports" / "summary.md").write_text("report", encoding="utf-8")
    (root / "report.md").write_text("report-file", encoding="utf-8")

    result = discover_scope(make_config(), tmp_path)

    expected_relatives = ["B.edf", "a.edf", "keep.mat", "nested/b.edf", "z.edf"]
    assert tuple(path.name for path in result.files) == (
        "B.edf",
        "a.edf",
        "keep.mat",
        "b.edf",
        "z.edf",
    )
    assert [path.resolve() for path in result.files] == [
        (root / rel.replace("/", os.sep)).resolve() for rel in expected_relatives
    ]
    assert result.scope_sha256 == expected_scope_sha256(expected_relatives)
    excluded_paths = {item.path for item in result.excluded}
    assert any(path == ".git" or path.startswith(".git/") for path in excluded_paths)
    assert any(path == "cache" or path.startswith("cache/") for path in excluded_paths)
    assert any(path == "__pycache__" or path.startswith("__pycache__/") for path in excluded_paths)
    assert any(path == ".cache" or path.startswith(".cache/") for path in excluded_paths)
    assert "download_dataset.ps1" in excluded_paths
    assert "download.sh" in excluded_paths
    assert any(path == "reports" or path.startswith("reports/") for path in excluded_paths)
    assert "report.md" in excluded_paths
    assert all(item.path != "keep.mat" for item in result.excluded)


def test_scope_rejects_missing_family_root(tmp_path):
    from src.data.release.scope import discover_scope

    with pytest.raises(ValueError, match="family root"):
        discover_scope(make_config(), tmp_path)


def test_scope_rejects_family_root_symlink_escape(tmp_path):
    from src.data.release.scope import discover_scope

    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside_family"
    outside.mkdir()
    (outside / "secret.edf").write_bytes(b"secret")
    family = data_root / "family"
    if not _try_symlink(family, outside, target_is_directory=True):
        pytest.skip("symlink creation refused by the OS")

    with pytest.raises(ValueError):
        discover_scope(make_config(), data_root)


def test_scope_excludes_non_regular_files(tmp_path):
    from src.data.release.scope import discover_scope

    root = tmp_path / "family"
    nested = root / "subdir"
    nested.mkdir(parents=True)
    (root / "ok.edf").write_bytes(b"ok")
    (nested / "nested.edf").write_bytes(b"n")
    fifo = root / "pipe.fifo"
    fifo_created = False
    if hasattr(os, "mkfifo"):
        try:
            os.mkfifo(fifo)
            fifo_created = stat.S_ISFIFO(os.lstat(fifo).st_mode)
        except (OSError, NotImplementedError, AttributeError):
            fifo_created = False

    result = discover_scope(make_config(), tmp_path)

    assert (root / "ok.edf").resolve() in result.files
    assert (nested / "nested.edf").resolve() in result.files
    assert all(path.is_file() and not path.is_symlink() for path in result.files)
    assert all(path.name != "subdir" for path in result.files)
    if not fifo_created:
        pytest.skip("non-regular file creation is unavailable")
    assert any(item.path == "pipe.fifo" for item in result.excluded)


def test_parser_allowlist_remains_unchanged():
    from src.preprocessing.io import ALLOWED_EXTENSIONS

    assert ALLOWED_EXTENSIONS == {".bdf", ".edf", ".fif", ".parquet", ".npy", ".npz", ".csv"}


def test_scope_includes_opaque_mat_without_reading_or_parsing(tmp_path, monkeypatch):
    from src.data.release.scope import discover_scope

    root = tmp_path / "family"
    root.mkdir()
    mat = root / "signal.mat"
    mat.write_bytes(b"opaque-matlab-bytes")
    (root / "keep.edf").write_bytes(b"edf")

    source = Path("src/data/release/scope.py").read_text(encoding="utf-8")

    def _blocked_open(*_args, **_kwargs):
        raise AssertionError("scope must not read file contents")

    monkeypatch.setattr("builtins.open", _blocked_open)
    result = discover_scope(make_config(), tmp_path)

    assert mat.resolve() in result.files
    assert (root / "keep.edf").resolve() in result.files
    assert "preprocessing.readers" not in source
    assert "mne" not in source
    assert "src.preprocessing.io" not in source


def test_scope_result_is_frozen_and_repeatable(tmp_path):
    from src.data.release.scope import ExcludedPath, ScopeResult, discover_scope

    root = tmp_path / "family"
    root.mkdir()
    (root / "ok.edf").write_bytes(b"ok")
    result = discover_scope(make_config(), tmp_path)
    again = discover_scope(make_config(), tmp_path)

    assert result == again
    assert isinstance(result.files, tuple)
    assert isinstance(result.excluded, tuple)
    assert isinstance(result, ScopeResult)
    with pytest.raises(AttributeError):
        result.files.append(result.files[0])  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        result.excluded.append(ExcludedPath(path="x", reason="symlink"))  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        result.scope_sha256 = "0" * 64
    assert len(result.scope_sha256) == 64
    assert int(result.scope_sha256, 16) >= 0
