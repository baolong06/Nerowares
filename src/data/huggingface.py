"""Hugging Face Hub dataset source with immutable revision and checksum checks."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .manifest import DatasetManifest
from .release.hub import require_immutable_revision
from .release.models import FamilyReleaseManifest
from .release.signing import verify_detached_signature
from .source import DatasetHandle, _verify_shard

try:  # Optional dependency: local-only workflows must not require the Hub client.
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover - exercised in environments without the extra
    hf_hub_download = None  # type: ignore[assignment]


@dataclass(frozen=True)
class VerifiedRelease:
    """Family-release manifest that passed detached Ed25519 verification."""

    manifest: FamilyReleaseManifest

    @property
    def release_id(self) -> str:
        return self.manifest.release_id

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    @property
    def repo_id(self) -> str:
        return self.manifest.repo_id

    @property
    def revision(self) -> str:
        return self.manifest.revision

    @property
    def manifest_sha256(self) -> str:
        return self.manifest.manifest_sha256 or self.manifest.sha256()

    @property
    def manifest_signature_status(self) -> str:
        return self.manifest.manifest_signature_status

    @property
    def governance_status(self) -> str:
        return self.manifest.governance.visibility

    @property
    def scope_status(self) -> str:
        return self.manifest.scope_status

    @property
    def file_count(self) -> int:
        return self.manifest.file_count

    @property
    def total_size_bytes(self) -> int:
        return self.manifest.total_size_bytes

    def source_kwargs(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "scope_status": self.scope_status,
            "manifest_signature_status": "verified",
            "governance_status": self.governance_status,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
        }


def verify_release_manifest(
    *,
    manifest_path: Path,
    signature_path: Path,
    verification_key: bytes,
) -> VerifiedRelease:
    """Load and fail-closed-verify a signed family-release JSON document."""
    path = Path(manifest_path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("unable to read family release manifest") from exc
    if not isinstance(loaded, dict):
        raise ValueError("family release manifest must be a JSON object")
    manifest = FamilyReleaseManifest.from_dict(loaded)
    if not verify_detached_signature(path, Path(signature_path), verification_key):
        raise ValueError("signature is missing, untrusted, or does not match the manifest")
    if manifest.provider == "huggingface":
        require_immutable_revision(manifest.revision, git_sha=True)
    verified_manifest = replace(
        manifest,
        manifest_signature_status="verified",
        manifest_sha256=manifest.manifest_sha256 or manifest.sha256(),
    )
    return VerifiedRelease(manifest=verified_manifest)


def preflight_download_quota(
    *,
    total_size_bytes: int,
    cache_dir: Path,
    max_cache_bytes: int | None = None,
    min_free_bytes: int | None = None,
    disk_usage: Callable[..., Any] | None = None,
) -> None:
    """Reject oversized or under-provisioned downloads before any Hub call."""
    if max_cache_bytes is not None and total_size_bytes > max_cache_bytes:
        raise ValueError("download exceeds cache quota")
    if min_free_bytes is None:
        return
    cache = Path(cache_dir).expanduser()
    if cache.exists():
        probe = cache
    elif cache.parent.exists():
        probe = cache.parent
    else:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache
    usage = (disk_usage or shutil.disk_usage)(probe)
    free = int(getattr(usage, "free"))
    if free < total_size_bytes + min_free_bytes:
        raise ValueError("insufficient free space for dataset download")


def load_verified_release_from_flags(
    *,
    release_manifest: Path | None,
    signature: Path | None,
    verification_key: Path | None,
) -> VerifiedRelease | None:
    """Require the CLI release triple together; catalog-only when all omitted."""
    flags = (release_manifest, signature, verification_key)
    if any(flag is not None for flag in flags) and any(flag is None for flag in flags):
        raise ValueError("release-manifest, signature, and verification-key are required together")
    if release_manifest is None:
        return None
    return verify_release_manifest(
        manifest_path=Path(release_manifest),
        signature_path=Path(signature),
        verification_key=Path(verification_key).read_bytes(),
    )


def align_cli_with_verified_release(
    *,
    dataset_id: str,
    repo_id: str | None,
    revision: str | None,
    verified: VerifiedRelease,
) -> tuple[str | None, str | None, dict[str, object]]:
    """Bind --dataset/--repo-id/--revision to a verified family release."""
    if dataset_id != verified.dataset_id:
        raise ValueError("dataset does not match verified family release")
    if repo_id is None:
        repo_id = verified.repo_id
    elif repo_id != verified.repo_id:
        raise ValueError("repo-id does not match verified family release")
    if revision is None:
        revision = verified.revision or None
    elif revision != verified.revision:
        raise ValueError("revision does not match verified family release")
    return repo_id, revision, verified.source_kwargs()


class HuggingFaceDatasetSource:
    """Resolve version-pinned Hub shards into the Hugging Face local cache.

    The manifest is intentionally supplied by the caller and must contain the same
    immutable revision and repository as the request. This prevents a moving branch
    or a changed remote manifest from silently changing a scientific run.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        revision: str,
        manifest: DatasetManifest,
        cache_dir: Path,
        token: str | None = None,
        downloader: Callable[..., str] | None = None,
        local_files_only: bool = False,
        release_id: str | None = None,
        scope_status: str | None = None,
        manifest_signature_status: str | None = None,
        governance_status: str | None = None,
        file_count: int | None = None,
        total_size_bytes: int | None = None,
        max_cache_bytes: int | None = None,
        min_free_bytes: int | None = None,
        disk_usage: Callable[..., Any] | None = None,
    ) -> None:
        if manifest.provider != "huggingface":
            raise ValueError("HuggingFaceDatasetSource requires a huggingface manifest")
        if not repo_id.strip() or manifest.repo_id != repo_id:
            raise ValueError("Hugging Face repo_id does not match the dataset manifest")
        if manifest.require_revision() != revision:
            raise ValueError("Hugging Face revision does not match the dataset manifest")
        self.repo_id = repo_id
        self.revision = revision
        self.manifest = manifest
        self.cache_dir = Path(cache_dir).expanduser()
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self._downloader = downloader
        self.local_files_only = local_files_only
        self.release_id = release_id
        self.scope_status = scope_status
        self.manifest_signature_status = manifest_signature_status
        self.governance_status = governance_status
        self.file_count = file_count
        self.total_size_bytes = total_size_bytes
        self.max_cache_bytes = max_cache_bytes
        self.min_free_bytes = min_free_bytes
        self._disk_usage = disk_usage

    def resolve(self, *, local_files_only: bool | None = None) -> DatasetHandle:
        """Download/cache and verify every manifest shard at the pinned revision."""
        if self.max_cache_bytes is not None or self.min_free_bytes is not None:
            shard_total = sum(shard.size_bytes for shard in self.manifest.shards)
            total = shard_total
            if self.total_size_bytes is not None:
                total = max(int(self.total_size_bytes), shard_total)
            preflight_download_quota(
                total_size_bytes=total,
                cache_dir=self.cache_dir,
                max_cache_bytes=self.max_cache_bytes,
                min_free_bytes=self.min_free_bytes,
                disk_usage=self._disk_usage,
            )
        downloader = self._downloader or hf_hub_download
        if downloader is None:
            raise RuntimeError(
                "Hugging Face support requires the optional 'data' dependency"
            )
        if local_files_only is None:
            local_files_only = self.local_files_only
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for shard in self.manifest.shards:
            try:
                downloaded = downloader(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    filename=shard.path,
                    revision=self.revision,
                    cache_dir=str(self.cache_dir),
                    token=self.token,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"failed to resolve Hugging Face dataset shard: {shard.path}"
                ) from exc
            path = Path(downloaded).resolve()
            _verify_shard(path, shard)
            files.append(path)
        return DatasetHandle(
            dataset_id=self.manifest.dataset_id,
            revision=self.revision,
            manifest=self.manifest,
            files=tuple(files),
            manifest_sha256=self.manifest.sha256(),
            mode="hf-cache",
            release_id=self.release_id,
            scope_status=self.scope_status,
            manifest_signature_status=self.manifest_signature_status,
            governance_status=self.governance_status,
            file_count=self.file_count,
            total_size_bytes=self.total_size_bytes,
        )


__all__ = [
    "HuggingFaceDatasetSource",
    "VerifiedRelease",
    "preflight_download_quota",
    "verify_release_manifest",
]
