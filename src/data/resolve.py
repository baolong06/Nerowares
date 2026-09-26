"""Resolve a DatasetSource from catalog + environment, not a hardcoded hub path."""
from __future__ import annotations

import os
from pathlib import Path

from .catalog import load_catalog_manifest
from .huggingface import HuggingFaceDatasetSource
from .manifest import DatasetManifest
from .source import DatasetSource, LocalDatasetSource

DATA_ROOT_ENV = "THINKING_DATA_ROOT"
CACHE_ENV = "THINKING_DATA_CACHE"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_DATA_ROOT = Path("E:/AI_thucchien/THINKING/datasets")


def data_root(root: Path | None = None) -> Path:
    """Trusted local dataset root: env first, then worktree, then legacy path if present."""
    if root is not None:
        return Path(root).expanduser()
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        return Path(env).expanduser()
    worktree = WORKTREE_ROOT / "datasets"
    if worktree.exists():
        return worktree
    if _LEGACY_DATA_ROOT.exists():
        return _LEGACY_DATA_ROOT
    return worktree


def cache_root(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "thinking" / "datasets"


def resolve_dataset_source(
    *,
    dataset_id: str,
    provider: str,
    catalog_dir: Path | None = None,
    manifest: DatasetManifest | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
    root: Path | None = None,
    token: str | None = None,
    local_files_only: bool = False,
    release_id: str | None = None,
    scope_status: str | None = None,
    manifest_signature_status: str | None = None,
    governance_status: str | None = None,
    file_count: int | None = None,
    total_size_bytes: int | None = None,
) -> DatasetSource:
    """Build a verified local or Hugging Face source from a pinned catalog."""
    if manifest is None:
        manifest = load_catalog_manifest(dataset_id, provider, catalog_dir=catalog_dir)
    if manifest.dataset_id != dataset_id:
        raise ValueError("catalog dataset_id does not match the requested dataset")
    if manifest.provider != provider:
        raise ValueError("catalog provider does not match the requested provider")
    release_kwargs = {
        "release_id": release_id,
        "scope_status": scope_status,
        "manifest_signature_status": manifest_signature_status,
        "governance_status": governance_status,
        "file_count": file_count,
        "total_size_bytes": total_size_bytes,
    }
    if provider == "local":
        return LocalDatasetSource(root=data_root(root), manifest=manifest, **release_kwargs)
    if provider == "huggingface":
        requested_repo = repo_id or manifest.repo_id
        requested_revision = revision or manifest.revision
        if not requested_repo:
            raise ValueError("Hugging Face datasets require repo_id")
        return HuggingFaceDatasetSource(
            repo_id=requested_repo,
            revision=requested_revision,
            manifest=manifest,
            cache_dir=cache_root(cache_dir),
            token=token,
            local_files_only=local_files_only,
            **release_kwargs,
        )
    raise ValueError("dataset provider must be local or huggingface")


__all__ = [
    "CACHE_ENV",
    "DATA_ROOT_ENV",
    "cache_root",
    "data_root",
    "resolve_dataset_source",
]
