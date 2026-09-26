"""Immutable dataset contracts used by local and remote sources."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REVISIONS = {"main", "master", "latest", "head", "default"}


@dataclass(frozen=True)
class DatasetShard:
    """One immutable, checksum-addressed dataset file."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith(("/", "\\")) or ".." in self.path.replace("\\", "/").split("/"):
            raise ValueError("dataset shard path must be relative and stay inside the dataset root")
        if self.size_bytes < 0:
            raise ValueError("dataset shard size must be non-negative")
        if not _SHA256_RE.fullmatch(self.sha256.lower()):
            raise ValueError("dataset shard sha256 must be a 64-character hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256.lower(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetShard":
        try:
            return cls(
                path=str(value["path"]),
                size_bytes=int(value["size_bytes"]),
                sha256=str(value["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid dataset shard manifest") from exc


@dataclass(frozen=True)
class DatasetManifest:
    """Versioned dataset schema and split identity contract."""

    dataset_id: str
    provider: str
    repo_id: str | None
    revision: str
    format: str
    sampling_rate_hz: float
    n_channels_source: int
    n_times_source: int
    labels: tuple[str, ...]
    subject_field: str
    session_field: str
    shards: tuple[DatasetShard, ...]

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required")
        if self.provider not in {"local", "huggingface"}:
            raise ValueError("dataset provider must be local or huggingface")
        if self.provider == "huggingface" and not self.repo_id:
            raise ValueError("Hugging Face datasets require repo_id")
        if not self.revision or self.revision.lower() in _MUTABLE_REVISIONS:
            raise ValueError("dataset revision must be immutable, not a moving revision")
        if self.provider == "huggingface" and not _GIT_SHA_RE.fullmatch(self.revision.lower()):
            raise ValueError("Hugging Face dataset revision must be a 40-character commit SHA")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.n_channels_source < 1 or self.n_times_source < 1:
            raise ValueError("dataset dimensions must be positive")
        if not self.labels or any(not label.strip() for label in self.labels):
            raise ValueError("dataset labels must be non-empty")
        if not self.subject_field or not self.session_field:
            raise ValueError("subject_field and session_field are required")
        if not self.shards:
            raise ValueError("dataset manifest must contain at least one shard")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "provider": self.provider,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "format": self.format,
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_channels_source": self.n_channels_source,
            "n_times_source": self.n_times_source,
            "labels": list(self.labels),
            "subject_field": self.subject_field,
            "session_field": self.session_field,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetManifest":
        try:
            shards = tuple(DatasetShard.from_dict(item) for item in value["shards"])
            labels = tuple(str(item) for item in value["labels"])
            return cls(
                dataset_id=str(value["dataset_id"]),
                provider=str(value["provider"]),
                repo_id=None if value.get("repo_id") is None else str(value["repo_id"]),
                revision=str(value["revision"]),
                format=str(value["format"]),
                sampling_rate_hz=float(value["sampling_rate_hz"]),
                n_channels_source=int(value["n_channels_source"]),
                n_times_source=int(value["n_times_source"]),
                labels=labels,
                subject_field=str(value["subject_field"]),
                session_field=str(value["session_field"]),
                shards=shards,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and "revision" in str(exc):
                raise
            raise ValueError("invalid dataset manifest") from exc

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_json(cls, path: Path) -> "DatasetManifest":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("unable to read dataset manifest JSON") from exc
        if not isinstance(document, dict):
            raise ValueError("dataset manifest JSON must be an object")
        return cls.from_dict(document)

    def write_json(self, path: Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination

    def require_revision(self) -> str:
        """Return the pinned revision after re-validating the immutability contract."""
        if not self.revision or self.revision.lower() in _MUTABLE_REVISIONS:
            raise ValueError("dataset revision is not immutable")
        return self.revision
