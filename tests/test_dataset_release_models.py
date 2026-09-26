"""Immutable family-release models and reviewed D01-D19 configuration."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from src.data.catalog import DEFAULT_CATALOG_DIR, load_catalog_manifest

_RELEASE_CONFIG_PATH = DEFAULT_CATALOG_DIR / "families.release.json"
_FAMILY_IDS = tuple(f"D{index:02d}" for index in range(1, 20))
_SECRET_MARKERS = (
    "HF_TOKEN",
    "hf_token",
    "private_key",
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
)


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


def _approved_governance(*, visibility: str = "private-team") -> dict[str, object]:
    return {
        "visibility": visibility,
        "license_status": "verified",
        "dua_status": "not_applicable",
        "custodian_status": "verified",
        "deidentification_status": "reviewed",
        "approved_by": "custodian",
        "evidence_reference": "ticket-1",
    }


def test_governance_requires_explicit_visibility_and_evidence():
    from src.data.release.models import GovernanceDecision

    with pytest.raises(ValueError, match="visibility"):
        GovernanceDecision.from_dict({
            "license_status": "verified",
            "dua_status": "verified",
            "custodian_status": "verified",
            "deidentification_status": "reviewed",
            "approved_by": "custodian",
            "evidence_reference": "ticket-1",
        })


def test_blocked_family_is_not_uploadable():
    from src.data.release.models import GovernanceDecision

    decision = GovernanceDecision.from_dict({
        "visibility": "blocked",
        "license_status": "unknown",
        "dua_status": "unknown",
        "custodian_status": "unknown",
        "deidentification_status": "not_reviewed",
        "approved_by": "",
        "evidence_reference": "",
    })
    assert decision.can_upload is False


def test_file_record_rejects_parent_path_and_bad_digest():
    from src.data.release.models import FileRecord

    with pytest.raises(ValueError):
        FileRecord(path="../escape.edf", size_bytes=1, sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        FileRecord(path="ok.edf", size_bytes=1, sha256="bad")


def test_file_record_rejects_windows_drive_relative_paths():
    from src.data.release.models import FileRecord

    with pytest.raises(ValueError):
        FileRecord(path="C:foo", size_bytes=1, sha256="0" * 64)
    with pytest.raises(ValueError):
        FileRecord(path="C:..\\x", size_bytes=1, sha256="0" * 64)


def test_governance_can_upload_only_for_public_or_private_team_with_evidence():
    from src.data.release.models import GovernanceDecision

    for visibility in ("public", "private-team"):
        decision = GovernanceDecision.from_dict(_approved_governance(visibility=visibility))
        assert decision.can_upload is True

    other_license = dict(_approved_governance())
    other_license["license_status"] = "other"
    assert GovernanceDecision.from_dict(other_license).can_upload is True


def test_governance_unknown_denied_or_not_reviewed_cannot_upload():
    from src.data.release.models import GovernanceDecision

    base = _approved_governance()
    blocked_cases = (
        ("license_status", "unknown"),
        ("dua_status", "denied"),
        ("custodian_status", "unknown"),
        ("deidentification_status", "not_reviewed"),
        ("approved_by", ""),
        ("evidence_reference", "  "),
        ("visibility", "blocked"),
    )
    for field, value in blocked_cases:
        decision = GovernanceDecision.from_dict({**base, field: value})
        assert decision.can_upload is False, field


def test_governance_null_or_bool_approver_and_evidence_cannot_upload():
    from src.data.release.models import GovernanceDecision

    base = _approved_governance()
    blocked_cases = (
        ("approved_by", None),
        ("evidence_reference", None),
        ("approved_by", True),
        ("evidence_reference", False),
        ("approved_by", True),
        ("evidence_reference", True),
    )
    for field, value in blocked_cases:
        decision = GovernanceDecision.from_dict({**base, field: value})
        assert decision.can_upload is False, (field, value)


def test_file_record_preserves_source_path_without_rewrite():
    from src.data.release.models import FileRecord

    posix = FileRecord(path="sub-01/ses-01/file.edf", size_bytes=1, sha256="a" * 64)
    mixed = FileRecord(path="sub-01\\ses-01\\file.edf", size_bytes=1, sha256="a" * 64)

    assert posix.path == "sub-01/ses-01/file.edf"
    assert mixed.path == "sub-01\\ses-01\\file.edf"


def test_family_release_manifest_canonical_bytes_and_digest():
    from src.data.release.models import FamilyReleaseManifest

    document = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "file_count": 1,
        "total_size_bytes": 4,
        "files": [{"path": "a.edf", "size_bytes": 4, "sha256": "b" * 64}],
        "governance": _blocked_governance(),
        "manifest_sha256": "",
        "manifest_signature_status": "unsigned",
        "revision": "",
    }
    manifest = FamilyReleaseManifest.from_dict(document)
    payload = json.loads(manifest.canonical_bytes().decode("utf-8"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    assert manifest.canonical_bytes() == encoded
    assert manifest.sha256() == hashlib.sha256(encoded).hexdigest()
    assert "manifest_sha256" not in payload
    assert "manifest_signature_status" not in payload
    assert "revision" not in payload
    assert payload["dataset_id"] == "D01"
    assert payload["files"][0]["path"] == "a.edf"


def test_family_release_manifest_files_are_immutable_and_cannot_desync_file_count():
    from src.data.release.models import FamilyReleaseManifest, FileRecord, GovernanceDecision

    record = FileRecord(path="a.edf", size_bytes=4, sha256="b" * 64)
    extra = FileRecord(path="b.edf", size_bytes=4, sha256="c" * 64)
    files = [record]
    manifest = FamilyReleaseManifest(
        dataset_id="D01",
        release_id="thinking-d01-observed-v1",
        repo_id="madteam/thinking-d01",
        provider="huggingface",
        scope_status="observed-local-package",
        file_count=1,
        total_size_bytes=4,
        files=files,
        governance=GovernanceDecision.from_dict(_blocked_governance()),
    )

    with pytest.raises(AttributeError):
        manifest.files.append(extra)
    files.append(extra)

    assert isinstance(manifest.files, tuple)
    assert manifest.file_count == 1
    assert len(manifest.files) == 1
    assert manifest.file_count == len(manifest.files)
    assert manifest.total_size_bytes == 4


def test_family_release_manifest_rejects_moving_revision():
    from src.data.release.models import FamilyReleaseManifest

    document = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "file_count": 1,
        "total_size_bytes": 4,
        "files": [{"path": "a.edf", "size_bytes": 4, "sha256": "b" * 64}],
        "governance": _blocked_governance(),
        "manifest_sha256": "",
        "manifest_signature_status": "unsigned",
        "revision": "main",
    }
    with pytest.raises(ValueError, match="revision"):
        FamilyReleaseManifest.from_dict(document)


def test_upload_state_round_trip_excludes_secrets_and_absolute_paths():
    from src.data.release.models import UploadState

    document = {
        "release_id": "thinking-d01-observed-v1",
        "dataset_id": "D01",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "manifest_sha256": "c" * 64,
        "completed_batch_ids": ["batch-0"],
        "remote_revision": "",
        "verification_status": "pending",
    }
    state = UploadState.from_dict(document)
    restored = UploadState.from_dict(state.to_dict())
    serialized = json.dumps(state.to_dict())

    assert restored == state
    assert "token" not in serialized.lower()
    assert "private" not in serialized.lower()
    assert "E:\\" not in serialized
    with pytest.raises(ValueError, match="revision"):
        UploadState.from_dict({**document, "remote_revision": "main"})


def test_family_release_config_rejects_absolute_local_root():
    from src.data.release.models import FamilyReleaseConfig

    document = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "local_root": "E:/AI_thucchien/THINKING/datasets/openneuro/ds003626_inner_speech",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "governance": _blocked_governance(),
    }
    with pytest.raises(ValueError, match="local_root"):
        FamilyReleaseConfig.from_dict(document)


def test_family_release_config_rejects_windows_drive_relative_local_root():
    from src.data.release.models import FamilyReleaseConfig

    document = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "local_root": "C:foo",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "governance": _blocked_governance(),
    }
    with pytest.raises(ValueError, match="local_root"):
        FamilyReleaseConfig.from_dict(document)
    with pytest.raises(ValueError, match="local_root"):
        FamilyReleaseConfig.from_dict({**document, "local_root": "C:..\\x"})


def test_family_release_config_rejects_canonical_bciciv2a_repo():
    from src.data.release.models import FamilyReleaseConfig

    document = {
        "dataset_id": "D15",
        "release_id": "thinking-d15-observed-v1",
        "local_root": "kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a",
        "repo_id": "madteam/thinking-bciciv2a",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": ["https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a"],
        "governance": _blocked_governance(),
    }
    with pytest.raises(ValueError, match="canonical"):
        FamilyReleaseConfig.from_dict(document)


def test_family_release_config_rejects_canonical_bciciv2a_repo_case_and_whitespace():
    from src.data.release.models import FamilyReleaseConfig

    document = {
        "dataset_id": "D15",
        "release_id": "thinking-d15-observed-v1",
        "local_root": "kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a",
        "repo_id": "madteam/thinking-d15",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": [
            "https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a"
        ],
        "governance": _blocked_governance(),
    }
    accepted = FamilyReleaseConfig.from_dict(document)
    assert accepted.repo_id == "madteam/thinking-d15"

    for repo_id in (
        "MadTeam/thinking-bciciv2a",
        "madteam/thinking-BCICIV2a",
        " madteam/thinking-bciciv2a",
        "madteam/thinking-bciciv2a ",
    ):
        with pytest.raises(ValueError, match="canonical"):
            FamilyReleaseConfig.from_dict({**document, "repo_id": repo_id})


def test_d03_full_upstream_claim_is_rejected():
    from src.data.release.models import FamilyReleaseConfig

    document = {
        "dataset_id": "D03",
        "release_id": "thinking-d03-observed-v1",
        "local_root": "openneuro/ds005170_chisco_subject02",
        "repo_id": "madteam/thinking-d03",
        "provider": "huggingface",
        "scope_status": "full-upstream",
        "visibility": "blocked",
        "provenance_references": ["https://openneuro.org/datasets/ds005170"],
        "governance": _blocked_governance(),
    }
    with pytest.raises(ValueError, match="observed-local-package"):
        FamilyReleaseConfig.from_dict(document)


def test_release_config_contains_all_d01_d19_ids():
    from src.data.release.models import ReleaseConfig

    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)
    family_ids = tuple(family.dataset_id for family in config.families)

    assert config.release_id
    assert int(config.schema_version) >= 1
    assert family_ids == _FAMILY_IDS


def test_blocked_config_entries_cannot_upload():
    from src.data.release.models import ReleaseConfig

    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)
    uploadable = [family.dataset_id for family in config.families if family.governance.can_upload]

    assert uploadable == []
    assert all(family.governance.visibility == "blocked" for family in config.families)


def test_release_config_has_no_absolute_paths_or_secrets():
    from src.data.release.models import ReleaseConfig

    text = _RELEASE_CONFIG_PATH.read_text(encoding="utf-8")
    document = json.loads(text)
    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)

    for marker in _SECRET_MARKERS:
        assert marker not in text
    assert re.search(r'"[A-Za-z]:[\\/]', text) is None
    assert "E:\\" not in text
    assert "C:\\" not in text
    for family in config.families:
        assert not Path(family.local_root).is_absolute()
        assert not family.local_root.startswith(("/", "\\"))
        assert ".." not in family.local_root.replace("\\", "/").split("/")
        assert family.provenance_references
        assert family.repo_id
        assert family.scope_status in {"observed-local-package", "full-upstream", "blocked"}
    dumped = json.dumps(document)
    assert "token" not in dumped.lower()
    assert "private_key" not in dumped.lower()


def test_canonical_bciciv2a_repository_is_protected():
    from src.data.release.models import ReleaseConfig

    canonical = load_catalog_manifest("bciciv2a", "huggingface")
    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)
    excluded_repos = {item.repo_id for item in config.canonical_exclusions}

    assert canonical.repo_id == "madteam/thinking-bciciv2a"
    assert canonical.repo_id in excluded_repos
    assert all(family.repo_id != canonical.repo_id for family in config.families)
    for item in config.canonical_exclusions:
        if item.repo_id == canonical.repo_id and item.revision:
            assert item.revision == canonical.revision


def test_d03_is_observed_local_package_never_full_upstream():
    from src.data.release.models import ReleaseConfig

    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)
    d03 = next(family for family in config.families if family.dataset_id == "D03")

    assert d03.scope_status == "observed-local-package"
    assert d03.scope_status != "full-upstream"
    assert "chisco" in d03.local_root.lower()
    assert "subject02" in d03.local_root.replace("-", "").replace("_", "").lower() or "subject02" in d03.local_root.lower()


def test_d15_full_raw_repository_is_distinct_from_canonical():
    from src.data.release.models import ReleaseConfig

    canonical = load_catalog_manifest("bciciv2a", "huggingface")
    config = ReleaseConfig.from_json(_RELEASE_CONFIG_PATH)
    d15 = next(family for family in config.families if family.dataset_id == "D15")

    assert d15.repo_id != canonical.repo_id
    assert d15.repo_id != "madteam/thinking-bciciv2a"
    assert "bciciv" in d15.local_root.lower() or "2a" in d15.local_root.lower()


def test_existing_bciciv2a_catalog_contract_unchanged():
    canonical = load_catalog_manifest("bciciv2a", "huggingface")

    assert canonical.dataset_id == "bciciv2a"
    assert canonical.provider == "huggingface"
    assert canonical.repo_id == "madteam/thinking-bciciv2a"
    assert canonical.revision == "1cd0deae08404efd86bf6b84d6beb2575c245a5e"
    assert canonical.shards[0].path == "BCICIV_2a_all_patients.csv"
    assert canonical.shards[0].size_bytes == 214848010
    assert canonical.shards[0].sha256 == (
        "fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd"
    )
