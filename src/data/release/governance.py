"""Fail-closed governance loader and upload-plan filter."""
from __future__ import annotations

from pathlib import Path

from src.data.release.models import (
    CANONICAL_BCICIV2A_REPO_ID,
    FamilyReleaseConfig,
    ReleaseConfig,
)

_BLOCKING_STATUS = {"unknown", "denied", "not_reviewed"}
_UPLOADABLE_VISIBILITY = {"public", "private-team"}


def _normalized_repo_id(repo_id: str) -> str:
    return str(repo_id).strip().casefold()


def validate_canonical_exclusions(config: ReleaseConfig) -> None:
    canonical = _normalized_repo_id(CANONICAL_BCICIV2A_REPO_ID)
    excluded = {_normalized_repo_id(item.repo_id) for item in config.canonical_exclusions}
    if canonical not in excluded:
        raise ValueError("canonical BCICIV-2a repository must be excluded from overwrite")
    for family in config.families:
        repo_id = _normalized_repo_id(family.repo_id)
        if repo_id == canonical or repo_id in excluded:
            raise ValueError("family repository IDs must not target canonical exclusions")


def validate_governance(config: FamilyReleaseConfig) -> None:
    if config.visibility == "blocked":
        return None
    if config.visibility not in _UPLOADABLE_VISIBILITY:
        raise ValueError("unknown visibility")
    governance = config.governance
    if governance.license_status in _BLOCKING_STATUS:
        raise ValueError("license status is unknown, denied, or not_reviewed")
    if governance.dua_status in _BLOCKING_STATUS:
        raise ValueError("dua status is unknown, denied, or not_reviewed")
    if governance.custodian_status in _BLOCKING_STATUS:
        raise ValueError("custodian status is unknown, denied, or not_reviewed")
    if governance.deidentification_status in _BLOCKING_STATUS:
        raise ValueError("deidentification status is unknown, denied, or not_reviewed")
    if not isinstance(governance.approved_by, str) or not governance.approved_by.strip():
        raise ValueError("approved_by is missing or empty")
    if not isinstance(governance.evidence_reference, str) or not governance.evidence_reference.strip():
        raise ValueError("evidence_reference is missing or empty")
    return None


def approved_families(config: ReleaseConfig) -> tuple[FamilyReleaseConfig, ...]:
    approved: list[FamilyReleaseConfig] = []
    for family in config.families:
        if family.visibility == "blocked" or not family.governance.can_upload:
            continue
        try:
            validate_governance(family)
        except ValueError:
            continue
        approved.append(family)
    return tuple(approved)


def load_release_config(path: Path) -> ReleaseConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"release configuration not found: {config_path}")
    config = ReleaseConfig.from_json(config_path)
    validate_canonical_exclusions(config)
    for family in config.families:
        validate_governance(family)
    return config
