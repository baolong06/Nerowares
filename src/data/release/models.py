"""Immutable family-release contracts used by later inventory and upload tasks."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FAMILY_ID_RE = re.compile(r"^D(0[1-9]|1[0-9])$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_MUTABLE_REVISIONS = {"main", "master", "latest", "head", "default"}
_VISIBILITY = {"public", "private-team", "blocked"}
_LICENSE_STATUS = {"verified", "other", "unknown", "denied", "not_reviewed"}
_DUA_STATUS = {"verified", "not_applicable", "unknown", "denied", "not_reviewed"}
_CUSTODIAN_STATUS = {"verified", "unknown", "denied", "not_reviewed"}
_DEIDENTIFICATION_STATUS = {"reviewed", "not_reviewed", "unknown", "denied"}
_SCOPE_STATUS = {"observed-local-package", "full-upstream", "blocked"}
_PROVIDERS = {"huggingface", "local"}
_SIGNATURE_STATUS = {"unsigned", "signed", "verified", "missing", "mismatched", "untrusted"}
_VERIFICATION_STATUS = {"pending", "verified", "failed", "incomplete"}
_BLOCKING_STATUS = {"unknown", "denied", "not_reviewed"}
_FORBIDDEN_KEYS = {
    "hf_token",
    "token",
    "access_token",
    "private_key",
    "secret",
    "password",
    "api_key",
}
_FAMILY_IDS = tuple(f"D{index:02d}" for index in range(1, 20))
_TOP_LEVEL_KEYS = {"schema_version", "release_id", "canonical_exclusions", *_FAMILY_IDS}
CANONICAL_BCICIV2A_REPO_ID = "madteam/thinking-bciciv2a"


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_choice(name: str, value: object, allowed: set[str]) -> str:
    text = str(value)
    if text not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return text


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    number = int(value)
    if isinstance(value, float) and value != number:
        raise ValueError("integer fields must be whole numbers")
    return number


def _normalized_parts(path: str) -> list[str]:
    return path.replace("\\", "/").split("/")


def _is_absolute_path(path: str) -> bool:
    if path.startswith(("/", "\\")):
        return True
    if _DRIVE_PATH_RE.match(path):
        return True
    return Path(path).is_absolute()


def _relative_path_is_unsafe(path: str) -> bool:
    if not path:
        return True
    if _is_absolute_path(path):
        return True
    return any(part == ".." for part in _normalized_parts(path))


def _immutable_revision(value: str, *, allow_empty: bool, git_sha: bool) -> str:
    revision = str(value or "")
    if not revision:
        if allow_empty:
            return ""
        raise ValueError("revision must be immutable, not a moving revision")
    if revision.lower() in _MUTABLE_REVISIONS:
        raise ValueError("revision must be immutable, not a moving revision")
    if git_sha and not _GIT_SHA_RE.fullmatch(revision.lower()):
        raise ValueError("revision must be a 40-character commit SHA")
    return revision


def _require_repo_id(value: object) -> str:
    repo_id = str(value).strip()
    if repo_id.count("/") != 1 or repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError("repo_id must be a namespaced Hub repository")
    namespace, name = repo_id.split("/")
    if not namespace.strip() or not name.strip() or ".." in repo_id:
        raise ValueError("repo_id must be a namespaced Hub repository")
    return repo_id


def _is_canonical_bciciv2a_repo(repo_id: str) -> bool:
    return repo_id.strip().casefold() == CANONICAL_BCICIV2A_REPO_ID.casefold()


def _reject_secrets(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_KEYS or lowered.endswith("_token"):
                raise ValueError("secrets are not allowed in release configuration")
            _reject_secrets(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_secrets(item)
        return
    if isinstance(value, str) and (
        "BEGIN PRIVATE KEY" in value
        or "BEGIN RSA PRIVATE KEY" in value
        or "BEGIN OPENSSH PRIVATE KEY" in value
    ):
        raise ValueError("secrets are not allowed in release configuration")


def _reject_absolute_strings(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_absolute_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_absolute_strings(item)
        return
    if isinstance(value, str) and _is_absolute_path(value):
        raise ValueError("absolute paths are not allowed in release state")


@dataclass(frozen=True)
class GovernanceDecision:
    """Fail-closed publication decision for one family."""

    visibility: str
    license_status: str
    dua_status: str
    custodian_status: str
    deidentification_status: str
    approved_by: str
    evidence_reference: str

    def __post_init__(self) -> None:
        _require_choice("visibility", self.visibility, _VISIBILITY)
        _require_choice("license_status", self.license_status, _LICENSE_STATUS)
        _require_choice("dua_status", self.dua_status, _DUA_STATUS)
        _require_choice("custodian_status", self.custodian_status, _CUSTODIAN_STATUS)
        _require_choice(
            "deidentification_status",
            self.deidentification_status,
            _DEIDENTIFICATION_STATUS,
        )
    @property
    def can_upload(self) -> bool:
        if self.visibility not in {"public", "private-team"}:
            return False
        if not isinstance(self.approved_by, str) or not isinstance(self.evidence_reference, str):
            return False
        if not self.approved_by.strip() or not self.evidence_reference.strip():
            return False
        return (
            self.license_status not in _BLOCKING_STATUS
            and self.dua_status not in _BLOCKING_STATUS
            and self.custodian_status not in _BLOCKING_STATUS
            and self.deidentification_status not in _BLOCKING_STATUS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "visibility": self.visibility,
            "license_status": self.license_status,
            "dua_status": self.dua_status,
            "custodian_status": self.custodian_status,
            "deidentification_status": self.deidentification_status,
            "approved_by": self.approved_by,
            "evidence_reference": self.evidence_reference,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "GovernanceDecision":
        document = _require_mapping(value, "governance")
        if "visibility" not in document:
            raise ValueError("visibility is required")
        try:
            return cls(
                visibility=str(document["visibility"]),
                license_status=str(document["license_status"]),
                dua_status=str(document["dua_status"]),
                custodian_status=str(document["custodian_status"]),
                deidentification_status=str(document["deidentification_status"]),
                approved_by=document.get("approved_by", ""),
                evidence_reference=document.get("evidence_reference", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid governance decision") from exc


@dataclass(frozen=True)
class FileRecord:
    """One relative payload file with exact size and digest."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if _relative_path_is_unsafe(self.path):
            raise ValueError("file path must be relative and stay inside the family root")
        if self.size_bytes < 0:
            raise ValueError("file size must be non-negative")
        if not _SHA256_RE.fullmatch(self.sha256.lower()):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256.lower(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FileRecord":
        document = _require_mapping(value, "file record")
        try:
            return cls(
                path=str(document["path"]),
                size_bytes=int(document["size_bytes"]),
                sha256=str(document["sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid file record") from exc


@dataclass(frozen=True)
class FamilyReleaseConfig:
    """Reviewed local-root, repository, and governance mapping for one family."""

    dataset_id: str
    release_id: str
    local_root: str
    repo_id: str
    provider: str
    scope_status: str
    visibility: str
    provenance_references: tuple[str, ...]
    governance: GovernanceDecision
    name: str = ""
    source: str = ""
    license_note: str = ""
    expected_source_files: int | None = None
    expected_source_bytes: int | None = None
    observed_local_files: int | None = None
    observed_local_bytes: int | None = None

    def __post_init__(self) -> None:
        if not _FAMILY_ID_RE.fullmatch(self.dataset_id):
            raise ValueError("dataset_id must be D01 through D19")
        if not self.release_id.strip():
            raise ValueError("release_id is required")
        if _relative_path_is_unsafe(self.local_root) or _is_absolute_path(self.local_root):
            raise ValueError("local_root must be relative to THINKING_DATA_ROOT")
        object.__setattr__(self, "repo_id", _require_repo_id(self.repo_id))
        if _is_canonical_bciciv2a_repo(self.repo_id):
            raise ValueError("family repository IDs must not target the canonical BCICIV-2a repository")
        object.__setattr__(self, "provenance_references", tuple(self.provenance_references))
        _require_choice("provider", self.provider, _PROVIDERS)
        _require_choice("scope_status", self.scope_status, _SCOPE_STATUS)
        _require_choice("visibility", self.visibility, _VISIBILITY)
        if self.visibility != self.governance.visibility:
            raise ValueError("visibility must match governance.visibility")
        if self.dataset_id == "D03" and self.scope_status == "full-upstream":
            raise ValueError("D03 scope_status must be observed-local-package, never full-upstream")
        if not self.provenance_references or any(not str(item).strip() for item in self.provenance_references):
            raise ValueError("provenance_references must be non-empty")
        for count in (
            self.expected_source_files,
            self.expected_source_bytes,
            self.observed_local_files,
            self.observed_local_bytes,
        ):
            if count is not None and count < 0:
                raise ValueError("source inventory counts must be non-negative")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FamilyReleaseConfig":
        document = _require_mapping(value, "family release config")
        _reject_secrets(document)
        if _is_absolute_path(str(document.get("local_root", ""))):
            raise ValueError("local_root must be relative to THINKING_DATA_ROOT")
        try:
            visibility = str(document["visibility"]) if "visibility" in document else ""
            governance_raw = dict(_require_mapping(document["governance"], "governance"))
            if "visibility" not in governance_raw:
                if not visibility:
                    raise ValueError("visibility is required")
                governance_raw["visibility"] = visibility
            elif visibility and str(governance_raw["visibility"]) != visibility:
                raise ValueError("visibility must match governance.visibility")
            references = tuple(str(item) for item in document["provenance_references"])
            return cls(
                dataset_id=str(document["dataset_id"]),
                release_id=str(document["release_id"]),
                local_root=str(document["local_root"]),
                repo_id=str(document["repo_id"]),
                provider=str(document.get("provider", "huggingface")),
                scope_status=str(document["scope_status"]),
                visibility=str(governance_raw["visibility"]),
                provenance_references=references,
                governance=GovernanceDecision.from_dict(governance_raw),
                name=str(document.get("name", "") or ""),
                source=str(document.get("source", "") or ""),
                license_note=str(document.get("license_note", "") or ""),
                expected_source_files=_optional_int(document.get("expected_source_files")),
                expected_source_bytes=_optional_int(document.get("expected_source_bytes")),
                observed_local_files=_optional_int(document.get("observed_local_files")),
                observed_local_bytes=_optional_int(document.get("observed_local_bytes")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid family release config") from exc


@dataclass(frozen=True)
class FamilyReleaseManifest:
    """Deterministic inventory contract for one family release."""

    dataset_id: str
    release_id: str
    repo_id: str
    provider: str
    scope_status: str
    file_count: int
    total_size_bytes: int
    files: tuple[FileRecord, ...]
    governance: GovernanceDecision
    name: str = ""
    source: str = ""
    license_note: str = ""
    provenance_references: tuple[str, ...] = ()
    expected_source_files: int | None = None
    expected_source_bytes: int | None = None
    observed_local_files: int | None = None
    observed_local_bytes: int | None = None
    manifest_sha256: str = ""
    manifest_signature_status: str = "unsigned"
    revision: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "provenance_references", tuple(self.provenance_references))
        if not self.dataset_id.strip() or not self.release_id.strip():
            raise ValueError("dataset_id and release_id are required")
        object.__setattr__(self, "repo_id", _require_repo_id(self.repo_id))
        _require_choice("provider", self.provider, _PROVIDERS)
        _require_choice("scope_status", self.scope_status, _SCOPE_STATUS)
        if self.dataset_id == "D03" and self.scope_status == "full-upstream":
            raise ValueError("D03 scope_status must be observed-local-package, never full-upstream")
        if self.file_count != len(self.files):
            raise ValueError("file_count must match files")
        if self.total_size_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("total_size_bytes must match files")
        if self.file_count < 0 or self.total_size_bytes < 0:
            raise ValueError("inventory totals must be non-negative")
        for count in (
            self.expected_source_files,
            self.expected_source_bytes,
            self.observed_local_files,
            self.observed_local_bytes,
        ):
            if count is not None and count < 0:
                raise ValueError("source inventory counts must be non-negative")
        if self.observed_local_files is not None and self.observed_local_files != self.file_count:
            raise ValueError("observed_local_files must match file_count")
        if self.observed_local_bytes is not None and self.observed_local_bytes != self.total_size_bytes:
            raise ValueError("observed_local_bytes must match total_size_bytes")
        _require_choice(
            "manifest_signature_status",
            self.manifest_signature_status,
            _SIGNATURE_STATUS,
        )
        _immutable_revision(
            self.revision,
            allow_empty=True,
            git_sha=self.provider == "huggingface" and bool(self.revision),
        )
        if self.manifest_sha256:
            if not _SHA256_RE.fullmatch(self.manifest_sha256.lower()):
                raise ValueError("manifest_sha256 must be a 64-character hexadecimal digest")
            if self.manifest_sha256.lower() != hashlib.sha256(self.canonical_bytes()).hexdigest():
                raise ValueError("manifest_sha256 does not match canonical bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "release_id": self.release_id,
            "repo_id": self.repo_id,
            "provider": self.provider,
            "scope_status": self.scope_status,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "files": [item.to_dict() for item in self.files],
            "governance": self.governance.to_dict(),
            "name": self.name,
            "source": self.source,
            "license_note": self.license_note,
            "provenance_references": list(self.provenance_references),
            "expected_source_files": self.expected_source_files,
            "expected_source_bytes": self.expected_source_bytes,
            "observed_local_files": self.observed_local_files,
            "observed_local_bytes": self.observed_local_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "FamilyReleaseManifest":
        document = _require_mapping(value, "family release manifest")
        try:
            files = tuple(FileRecord.from_dict(item) for item in document["files"])
            references = document.get("provenance_references", ()) or ()
            return cls(
                dataset_id=str(document["dataset_id"]),
                release_id=str(document["release_id"]),
                repo_id=str(document["repo_id"]),
                provider=str(document.get("provider", "huggingface")),
                scope_status=str(document["scope_status"]),
                file_count=int(document["file_count"]),
                total_size_bytes=int(document["total_size_bytes"]),
                files=files,
                governance=GovernanceDecision.from_dict(
                    _require_mapping(document["governance"], "governance")
                ),
                name=str(document.get("name", "") or ""),
                source=str(document.get("source", "") or ""),
                license_note=str(document.get("license_note", "") or ""),
                provenance_references=tuple(str(item) for item in references),
                expected_source_files=_optional_int(document.get("expected_source_files")),
                expected_source_bytes=_optional_int(document.get("expected_source_bytes")),
                observed_local_files=_optional_int(document.get("observed_local_files")),
                observed_local_bytes=_optional_int(document.get("observed_local_bytes")),
                manifest_sha256=str(document.get("manifest_sha256", "") or ""),
                manifest_signature_status=str(
                    document.get("manifest_signature_status", "unsigned") or "unsigned"
                ),
                revision=str(document.get("revision", "") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid family release manifest") from exc

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class UploadState:
    """Resumable metadata-only upload checkpoint."""

    release_id: str
    dataset_id: str
    repo_id: str
    provider: str
    manifest_sha256: str
    completed_batch_ids: tuple[str, ...]
    remote_revision: str
    verification_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_batch_ids", tuple(self.completed_batch_ids))
        if not self.release_id.strip() or not self.dataset_id.strip():
            raise ValueError("release_id and dataset_id are required")
        object.__setattr__(self, "repo_id", _require_repo_id(self.repo_id))
        _require_choice("provider", self.provider, _PROVIDERS)
        if not _SHA256_RE.fullmatch(self.manifest_sha256.lower()):
            raise ValueError("manifest_sha256 must be a 64-character hexadecimal digest")
        if any(not str(item).strip() or _is_absolute_path(str(item)) for item in self.completed_batch_ids):
            raise ValueError("completed_batch_ids must be relative identifiers")
        _immutable_revision(
            self.remote_revision,
            allow_empty=True,
            git_sha=self.provider == "huggingface" and bool(self.remote_revision),
        )
        _require_choice("verification_status", self.verification_status, _VERIFICATION_STATUS)
        _reject_absolute_strings(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "dataset_id": self.dataset_id,
            "repo_id": self.repo_id,
            "provider": self.provider,
            "manifest_sha256": self.manifest_sha256,
            "completed_batch_ids": list(self.completed_batch_ids),
            "remote_revision": self.remote_revision,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "UploadState":
        document = _require_mapping(value, "upload state")
        _reject_secrets(document)
        _reject_absolute_strings(document)
        try:
            return cls(
                release_id=str(document["release_id"]),
                dataset_id=str(document["dataset_id"]),
                repo_id=str(document["repo_id"]),
                provider=str(document.get("provider", "huggingface")),
                manifest_sha256=str(document["manifest_sha256"]),
                completed_batch_ids=tuple(str(item) for item in document["completed_batch_ids"]),
                remote_revision=str(document.get("remote_revision", "") or ""),
                verification_status=str(document.get("verification_status", "pending") or "pending"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid upload state") from exc


@dataclass(frozen=True)
class CanonicalExclusion:
    """Hub repository that family upload must never overwrite."""

    repo_id: str
    revision: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", _require_repo_id(self.repo_id))
        _immutable_revision(self.revision, allow_empty=True, git_sha=bool(self.revision))

    @classmethod
    def from_dict(cls, value: dict[str, object] | str) -> "CanonicalExclusion":
        if isinstance(value, str):
            return cls(repo_id=value)
        document = _require_mapping(value, "canonical exclusion")
        return cls(
            repo_id=str(document["repo_id"]),
            revision=str(document.get("revision", "") or ""),
            reason=str(document.get("reason", "") or ""),
        )


@dataclass(frozen=True)
class ReleaseConfig:
    """Reviewed D01-D19 family mapping plus protected canonical repositories."""

    schema_version: int
    release_id: str
    canonical_exclusions: tuple[CanonicalExclusion, ...]
    families: tuple[FamilyReleaseConfig, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_exclusions", tuple(self.canonical_exclusions))
        object.__setattr__(self, "families", tuple(self.families))
        if self.schema_version < 1:
            raise ValueError("schema_version must be >= 1")
        if not self.release_id.strip():
            raise ValueError("release_id is required")
        if not self.canonical_exclusions:
            raise ValueError("canonical_exclusions is required")
        family_ids = tuple(family.dataset_id for family in self.families)
        if family_ids != _FAMILY_IDS:
            raise ValueError("release configuration must contain D01 through D19 in order")
        excluded = {item.repo_id.strip().casefold() for item in self.canonical_exclusions}
        if CANONICAL_BCICIV2A_REPO_ID.casefold() not in excluded:
            raise ValueError("canonical BCICIV-2a repository must be excluded from overwrite")
        repo_ids = [family.repo_id.strip().casefold() for family in self.families]
        if len(set(repo_ids)) != len(repo_ids):
            raise ValueError("family repository IDs must be unique")
        if excluded.intersection(repo_ids):
            raise ValueError("family repository IDs must not target canonical exclusions")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ReleaseConfig":
        document = _require_mapping(value, "release configuration")
        _reject_secrets(document)
        unknown = set(document) - _TOP_LEVEL_KEYS
        if unknown:
            raise ValueError(f"unsupported release configuration keys: {sorted(unknown)}")
        try:
            families = []
            for family_id in _FAMILY_IDS:
                raw = dict(_require_mapping(document[family_id], f"family {family_id}"))
                raw.setdefault("dataset_id", family_id)
                if str(raw["dataset_id"]) != family_id:
                    raise ValueError("dataset_id must match family key")
                families.append(FamilyReleaseConfig.from_dict(raw))
            exclusions = tuple(
                CanonicalExclusion.from_dict(item) for item in document["canonical_exclusions"]
            )
            return cls(
                schema_version=int(document["schema_version"]),
                release_id=str(document["release_id"]),
                canonical_exclusions=exclusions,
                families=tuple(families),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid release configuration") from exc

    @classmethod
    def from_json(cls, path: Path) -> "ReleaseConfig":
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("unable to read release configuration JSON") from exc
        return cls.from_dict(document)
