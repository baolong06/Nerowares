"""Dataset source abstraction with fail-closed checksum verification."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .manifest import DatasetManifest, DatasetShard


@dataclass(frozen=True)
class DatasetHandle:
    """Verified files ready for a dataset-specific loader."""

    dataset_id: str
    revision: str
    manifest: DatasetManifest
    files: tuple[Path, ...]
    manifest_sha256: str
    mode: str
    release_id: str | None = None
    scope_status: str | None = None
    manifest_signature_status: str | None = None
    governance_status: str | None = None
    file_count: int | None = None
    total_size_bytes: int | None = None

    def to_lineage(self) -> dict[str, object]:
        """Identity metadata for training reports; omit unset release fields."""
        lineage: dict[str, object] = {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "provider": self.manifest.provider,
            "mode": self.mode,
        }
        optional = {
            "release_id": self.release_id,
            "scope_status": self.scope_status,
            "manifest_signature_status": self.manifest_signature_status,
            "governance_status": self.governance_status,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }
        for key, value in optional.items():
            if value is not None:
                lineage[key] = value
        return lineage


class DatasetSource(Protocol):
    """Resolve a versioned dataset into verified local files."""

    def resolve(self) -> DatasetHandle:
        ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    """Resolve a manifest path and prevent symlink/path escape."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("dataset shard escapes the configured root") from exc
    return candidate


def _verify_shard(path: Path, shard: DatasetShard) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"dataset shard is missing: {shard.path}")
    actual_size = path.stat().st_size
    if actual_size != shard.size_bytes:
        raise ValueError(
            f"dataset shard size mismatch for {shard.path}: "
            f"expected {shard.size_bytes}, got {actual_size}"
        )
    actual_sha = _sha256_file(path)
    if actual_sha != shard.sha256.lower():
        raise ValueError(f"dataset shard checksum mismatch for {shard.path}")


class LocalDatasetSource:
    """Resolve already-downloaded files below a trusted local dataset root."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: DatasetManifest,
        release_id: str | None = None,
        scope_status: str | None = None,
        manifest_signature_status: str | None = None,
        governance_status: str | None = None,
        file_count: int | None = None,
        total_size_bytes: int | None = None,
    ) -> None:
        if manifest.provider != "local":
            raise ValueError("LocalDatasetSource requires a local manifest")
        self.root = Path(root)
        self.manifest = manifest
        self.release_id = release_id
        self.scope_status = scope_status
        self.manifest_signature_status = manifest_signature_status
        self.governance_status = governance_status
        self.file_count = file_count
        self.total_size_bytes = total_size_bytes

    def resolve(self) -> DatasetHandle:
        revision = self.manifest.require_revision()
        files: list[Path] = []
        for shard in self.manifest.shards:
            path = _safe_child(self.root, shard.path)
            _verify_shard(path, shard)
            files.append(path)
        return DatasetHandle(
            dataset_id=self.manifest.dataset_id,
            revision=revision,
            manifest=self.manifest,
            files=tuple(files),
            manifest_sha256=self.manifest.sha256(),
            mode="local",
            release_id=self.release_id,
            scope_status=self.scope_status,
            manifest_signature_status=self.manifest_signature_status,
            governance_status=self.governance_status,
            file_count=self.file_count,
            total_size_bytes=self.total_size_bytes,
        )


__all__ = [
    "DatasetHandle",
    "DatasetSource",
    "LocalDatasetSource",
    "_safe_child",
    "_verify_shard",
]
