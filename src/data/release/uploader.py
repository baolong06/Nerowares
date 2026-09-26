"""Resumable direct family upload from existing package roots."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.data.release.governance import validate_governance
from src.data.release.hub import (
    UploadOperation,
    is_git_sha,
    reject_canonical_repo,
    require_immutable_revision,
)
from src.data.release.inventory import build_family_manifest, hash_file, revalidate_source
from src.data.release.models import FamilyReleaseConfig, FamilyReleaseManifest, UploadState
from src.data.release.scope import discover_scope

_SAMPLE_LIMIT = 3
_VERIFICATION_MODES = {"all", "sample"}


@dataclass(frozen=True)
class UploadResult:
    revision: str
    repo_id: str
    verification_status: str
    completed_batch_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RemoteVerification:
    revision: str
    verification_status: str
    file_count: int = 0


class ReleaseUploader:
    """Plan, upload, and verify one family against an injected Hub client."""

    def __init__(
        self,
        client: object,
        data_root: Path,
        config: FamilyReleaseConfig,
        *,
        batch_size: int = 32,
        verification_mode: str = "all",
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if verification_mode not in _VERIFICATION_MODES:
            raise ValueError("verification_mode must be 'all' or 'sample'")
        self._client = client
        self.data_root = Path(data_root)
        self.config = config
        self._batch_size = int(batch_size)
        self._verification_mode = verification_mode

    def plan(self, config: FamilyReleaseConfig, data_root: Path) -> FamilyReleaseManifest:
        self._require_bound(config, data_root)
        self._require_uploadable(config)
        scope = discover_scope(config, Path(data_root))
        return build_family_manifest(config, scope)

    def upload(self, manifest: FamilyReleaseManifest, state_path: Path) -> UploadResult:
        self._require_uploadable(self.config)
        reject_canonical_repo(manifest.repo_id)
        reject_canonical_repo(self.config.repo_id)
        self._revalidate_all(manifest)

        digest = manifest.sha256()
        batches = self._split_batches(manifest.files)
        state = _read_state(Path(state_path))
        needed: list[tuple[str, tuple]] = []
        completed: list[str] = []
        for batch_id, records in batches:
            if self._batch_remotely_confirmed(state, manifest, batch_id, records, digest):
                completed.append(batch_id)
            else:
                needed.append((batch_id, records))

        revision = ""
        if state is not None and state.manifest_sha256 == digest and is_git_sha(state.remote_revision):
            revision = state.remote_revision

        if needed:
            private = self.config.visibility == "private-team"
            self._client.create_dataset_repo(manifest.repo_id, private=private)

        for batch_id, records in needed:
            self._revalidate_batch(records)
            operations = tuple(
                UploadOperation(path=record.path, source_path=self._source_path(record.path))
                for record in records
            )
            revision = self._client.upload_batch(
                manifest.repo_id, operations, f"upload {batch_id}"
            )
            require_immutable_revision(revision, git_sha=True)
            listing = self._client.list_revision(manifest.repo_id, revision)
            listed = {_listing_path(item) for item in listing.files}
            if any(record.path not in listed for record in records):
                raise ValueError("remote listing does not contain uploaded files")
            completed.append(batch_id)
            state = UploadState(
                release_id=manifest.release_id,
                dataset_id=manifest.dataset_id,
                repo_id=manifest.repo_id,
                provider=manifest.provider,
                manifest_sha256=digest,
                completed_batch_ids=tuple(completed),
                remote_revision=revision,
                verification_status="pending",
            )
            _write_state(Path(state_path), state)

        if not is_git_sha(revision):
            raise ValueError("revision must be a 40-character commit SHA")

        verification_status = "pending"
        if manifest.manifest_signature_status == "verified":
            verified = self.verify_remote(manifest, revision)
            verification_status = verified.verification_status
            state = UploadState(
                release_id=manifest.release_id,
                dataset_id=manifest.dataset_id,
                repo_id=manifest.repo_id,
                provider=manifest.provider,
                manifest_sha256=digest,
                completed_batch_ids=tuple(completed),
                remote_revision=revision,
                verification_status=verification_status,
            )
            _write_state(Path(state_path), state)
        elif state is None:
            state = UploadState(
                release_id=manifest.release_id,
                dataset_id=manifest.dataset_id,
                repo_id=manifest.repo_id,
                provider=manifest.provider,
                manifest_sha256=digest,
                completed_batch_ids=tuple(completed),
                remote_revision=revision,
                verification_status=verification_status,
            )
            _write_state(Path(state_path), state)

        return UploadResult(
            revision=revision,
            repo_id=manifest.repo_id,
            verification_status=verification_status,
            completed_batch_ids=tuple(completed),
        )

    def verify_remote(self, manifest: FamilyReleaseManifest, revision: str) -> RemoteVerification:
        revision = require_immutable_revision(revision, git_sha=True)
        if manifest.manifest_signature_status != "verified":
            raise ValueError("manifest signature is not verified")
        listing = self._client.list_revision(manifest.repo_id, revision)
        listed = {_listing_path(item): item for item in listing.files}
        expected = {item.path: item for item in manifest.files}
        if set(listed) != set(expected):
            raise ValueError("remote listing does not match the planned manifest")
        for path, record in expected.items():
            remote = listed[path]
            size = getattr(remote, "size_bytes", None)
            if size is None:
                size = getattr(remote, "size", None)
            if size is not None and int(size) != record.size_bytes:
                raise ValueError("remote listing does not match the planned manifest")

        targets = list(manifest.files)
        if self._verification_mode == "sample":
            targets = targets[:_SAMPLE_LIMIT]
        for record in targets:
            self._verify_downloaded_file(manifest.repo_id, record, revision)
        return RemoteVerification(
            revision=revision,
            verification_status="verified",
            file_count=len(manifest.files),
        )

    def _require_bound(self, config: FamilyReleaseConfig, data_root: Path) -> None:
        if config != self.config:
            raise ValueError("plan config must match the bound uploader")
        if Path(data_root).resolve() != self.data_root.resolve():
            raise ValueError("plan data_root must match the bound uploader")

    def _require_uploadable(self, config: FamilyReleaseConfig) -> None:
        if not config.governance.can_upload:
            raise ValueError("family is not approved for upload")
        validate_governance(config)

    def _revalidate_all(self, manifest: FamilyReleaseManifest) -> None:
        mismatches = revalidate_source(manifest, self.config, self.data_root)
        if mismatches:
            raise ValueError("source file changed since the planned manifest")

    def _revalidate_batch(self, records: tuple) -> None:
        for record in records:
            path = self._source_path(record.path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("source file changed since the planned manifest")
            try:
                size, digest = hash_file(path)
            except OSError as exc:
                raise ValueError("source file changed since the planned manifest") from exc
            if size != record.size_bytes or digest != record.sha256.lower():
                raise ValueError("source file changed since the planned manifest")

    def _source_path(self, relative: str) -> Path:
        parts = [part for part in relative.replace("\\", "/").split("/") if part and part != "."]
        if not parts or any(part == ".." for part in parts):
            raise ValueError("source path escapes the family root")
        local = [part for part in str(self.config.local_root).replace("\\", "/").split("/") if part]
        return self.data_root.joinpath(*local, *parts)

    def _split_batches(self, files: tuple) -> list[tuple[str, tuple]]:
        items = tuple(files)
        batches: list[tuple[str, tuple]] = []
        for index, start in enumerate(range(0, len(items), self._batch_size)):
            batches.append((f"batch-{index}", items[start : start + self._batch_size]))
        return batches

    def _batch_remotely_confirmed(
        self,
        state: UploadState | None,
        manifest: FamilyReleaseManifest,
        batch_id: str,
        records: tuple,
        digest: str,
    ) -> bool:
        if state is None:
            return False
        if state.manifest_sha256 != digest:
            return False
        if not is_git_sha(state.remote_revision):
            return False
        if batch_id not in state.completed_batch_ids:
            return False
        listing = self._client.list_revision(manifest.repo_id, state.remote_revision)
        listed = {_listing_path(item) for item in listing.files}
        return all(record.path in listed for record in records)

    def _verify_downloaded_file(self, repo_id: str, record: object, revision: str) -> None:
        handle, name = tempfile.mkstemp(prefix="thinking-verify-")
        os.close(handle)
        dest = Path(name)
        try:
            self._client.download_for_verify(repo_id, record.path, revision, dest)
            size, digest = hash_file(dest)
        finally:
            try:
                dest.unlink()
            except OSError:
                pass
        if size != record.size_bytes or digest != record.sha256.lower():
            raise ValueError("remote file digest does not match the planned manifest")


def _listing_path(item: object) -> str:
    path = getattr(item, "path", "")
    return str(path).replace("\\", "/")


def _read_state(path: Path) -> UploadState | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read upload state") from exc
    return UploadState.from_dict(document)


def _write_state(path: Path, state: UploadState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
