"""Streaming size/SHA-256 inventory and canonical family-manifest I/O."""
from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path

from src.data.release.models import FamilyReleaseConfig, FamilyReleaseManifest, FileRecord
from src.data.release.scope import ScopeResult


def _posix_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part and part != "."]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parts_match(actual: tuple[str, ...], expected: list[str]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(left.casefold() == right.casefold() for left, right in zip(actual, expected))


def _is_regular_file(path: Path) -> bool:
    try:
        if path.is_symlink():
            return False
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)


def _family_root_from_path(file: Path, local_root: str) -> Path:
    local_parts = _posix_parts(local_root)
    if not local_parts or any(part == ".." for part in local_parts):
        raise ValueError("scoped path escapes the family root")
    count = len(local_parts)
    candidate = Path(file).parent
    while True:
        parts = candidate.parts
        if len(parts) >= count and _parts_match(parts[-count:], local_parts):
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    raise ValueError("scoped path escapes the family root")


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[int, str]:
    """Return (size_bytes, lowercase sha256) from one bounded streaming read."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def build_family_manifest(
    config: FamilyReleaseConfig, scope: ScopeResult
) -> FamilyReleaseManifest:
    """Hash scoped files in one pass and build a deterministic family manifest."""
    local_parts = _posix_parts(config.local_root)
    if not local_parts or any(part == ".." for part in local_parts):
        raise ValueError("scoped path escapes the family root")

    records: list[FileRecord] = []
    family_root: Path | None = None
    for file in scope.files:
        original = Path(file)
        if not _is_regular_file(original):
            raise ValueError("scoped path is no longer a regular file")
        root = _family_root_from_path(original, config.local_root)
        if family_root is None:
            family_root = root
        elif family_root.resolve() != root.resolve():
            raise ValueError("scoped path escapes the family root")
        try:
            relative = original.relative_to(family_root)
        except ValueError as exc:
            raise ValueError("scoped path escapes the family root") from exc
        posix = relative.as_posix()
        if not posix or posix.startswith("/") or any(part == ".." for part in posix.split("/")):
            raise ValueError("scoped path escapes the family root")
        try:
            size, digest = hash_file(original)
        except OSError as exc:
            raise ValueError("cannot hash scoped file") from exc
        records.append(FileRecord(path=posix, size_bytes=size, sha256=digest))

    records.sort(key=lambda item: item.path.encode("utf-8"))
    total = sum(item.size_bytes for item in records)
    return FamilyReleaseManifest(
        dataset_id=config.dataset_id,
        release_id=config.release_id,
        repo_id=config.repo_id,
        provider=config.provider,
        scope_status=config.scope_status,
        file_count=len(records),
        total_size_bytes=total,
        files=tuple(records),
        governance=config.governance,
        name=config.name,
        source=config.source,
        license_note=config.license_note,
        provenance_references=config.provenance_references,
        expected_source_files=config.expected_source_files,
        expected_source_bytes=config.expected_source_bytes,
        observed_local_files=len(records),
        observed_local_bytes=total,
        manifest_sha256="",
        manifest_signature_status="unsigned",
        revision="",
    )


def write_manifest(manifest: FamilyReleaseManifest, path: Path) -> Path:
    """Write pretty JSON of to_dict(); signing uses canonical_bytes(), not this text."""
    target = Path(path)
    text = json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target.write_text(text, encoding="utf-8")
    return target


def read_manifest(path: Path) -> FamilyReleaseManifest:
    """Load a family release manifest from pretty or compact JSON."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read family release manifest") from exc
    return FamilyReleaseManifest.from_dict(document)


@dataclass(frozen=True)
class SourceMismatch:
    """One planned file whose current size/digest no longer matches the manifest."""

    path: str
    expected_size_bytes: int
    expected_sha256: str
    observed_size_bytes: int | None
    observed_sha256: str | None
    reason: str = "changed"


def revalidate_source(
    manifest: FamilyReleaseManifest,
    config: FamilyReleaseConfig,
    data_root: Path,
) -> tuple[SourceMismatch, ...]:
    """Re-hash planned files and report mismatches. Does not upload."""
    local_parts = _posix_parts(config.local_root)
    if not local_parts or any(part == ".." for part in local_parts):
        raise ValueError("scoped path escapes the family root")
    family = Path(data_root).resolve().joinpath(*local_parts)
    mismatches: list[SourceMismatch] = []
    for record in manifest.files:
        relative_parts = _posix_parts(record.path)
        if not relative_parts or any(part == ".." for part in relative_parts):
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=None,
                    observed_sha256=None,
                    reason="escaped",
                )
            )
            continue
        candidate = family.joinpath(*relative_parts)
        if candidate.is_symlink() or not _is_regular_file(candidate):
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=None,
                    observed_sha256=None,
                    reason="missing",
                )
            )
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=None,
                    observed_sha256=None,
                    reason="missing",
                )
            )
            continue
        if not _is_relative_to(resolved, family.resolve()):
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=None,
                    observed_sha256=None,
                    reason="escaped",
                )
            )
            continue
        try:
            size, digest = hash_file(resolved)
        except OSError:
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=None,
                    observed_sha256=None,
                    reason="unreadable",
                )
            )
            continue
        if size != record.size_bytes or digest != record.sha256.lower():
            mismatches.append(
                SourceMismatch(
                    path=record.path,
                    expected_size_bytes=record.size_bytes,
                    expected_sha256=record.sha256.lower(),
                    observed_size_bytes=size,
                    observed_sha256=digest,
                    reason="changed",
                )
            )
    return tuple(mismatches)
