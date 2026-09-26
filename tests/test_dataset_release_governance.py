"""Fail-closed governance loader and upload-plan filter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.catalog import DEFAULT_CATALOG_DIR

_RELEASE_CONFIG_PATH = DEFAULT_CATALOG_DIR / "families.release.json"
_FAMILY_IDS = tuple(f"D{index:02d}" for index in range(1, 20))
_CANONICAL_BCICIV2A_REPO_ID = "madteam/thinking-bciciv2a"
_CANONICAL_REVISION = "1cd0deae08404efd86bf6b84d6beb2575c245a5e"


def _catalog_document() -> dict[str, object]:
    return json.loads(_RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))


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


def family_config(*, visibility: str = "blocked", dataset_id: str = "D01", **governance_fields):
    from src.data.release.models import FamilyReleaseConfig

    document = dict(_catalog_document()[dataset_id])
    if visibility == "blocked" and not governance_fields:
        governance = _blocked_governance()
    else:
        governance = _approved_governance(visibility=visibility)
        governance.update(governance_fields)
        governance["visibility"] = visibility
    document["visibility"] = visibility
    document["governance"] = governance
    return FamilyReleaseConfig.from_dict(document)


def release_config_with_family(*, dataset_id: str = "D01", visibility: str = "blocked", **governance_fields):
    from src.data.release.models import ReleaseConfig

    document = _catalog_document()
    family = dict(document[dataset_id])
    if visibility == "blocked" and not governance_fields:
        governance = dict(family["governance"])
    else:
        governance = _approved_governance(visibility=visibility)
        governance.update(governance_fields)
        governance["visibility"] = visibility
    family["visibility"] = visibility
    family["governance"] = governance
    document[dataset_id] = family
    return ReleaseConfig.from_dict(document)


def _write_catalog(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "families.release.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_blocked_family_is_excluded_from_upload_plan():
    from src.data.release.governance import approved_families

    config = release_config_with_family(visibility="blocked")
    assert approved_families(config) == ()


def test_missing_dua_or_custodian_evidence_fails_closed():
    from src.data.release.governance import validate_governance

    config = family_config(visibility="private-team", dua_status="unknown")
    with pytest.raises(ValueError, match="dua"):
        validate_governance(config)


def test_current_catalog_loads_blocked_and_is_not_an_upload_plan():
    from src.data.release.governance import approved_families, load_release_config

    source = (DEFAULT_CATALOG_DIR.parent / "src/data/release/governance.py").read_text(encoding="utf-8")
    config = load_release_config(_RELEASE_CONFIG_PATH)
    family_ids = tuple(family.dataset_id for family in config.families)
    d03 = next(family for family in config.families if family.dataset_id == "D03")
    d15 = next(family for family in config.families if family.dataset_id == "D15")
    canonical = next(
        item for item in config.canonical_exclusions if item.repo_id == _CANONICAL_BCICIV2A_REPO_ID
    )

    assert "huggingface_hub" not in source
    assert family_ids == _FAMILY_IDS
    assert all(family.visibility == "blocked" for family in config.families)
    assert all(family.governance.can_upload is False for family in config.families)
    assert approved_families(config) == ()
    assert d03.scope_status == "observed-local-package"
    assert "ds005170_chisco_subject02" in d03.local_root
    assert canonical.repo_id == "madteam/thinking-bciciv2a"
    assert canonical.revision == _CANONICAL_REVISION
    assert d15.repo_id == "madteam/thinking-d15"


def test_validate_governance_is_noop_for_blocked_family_with_unknown_evidence():
    from src.data.release.governance import validate_governance

    family = family_config(visibility="blocked", dua_status="unknown", approved_by="")
    assert validate_governance(family) is None


@pytest.mark.parametrize("visibility", ("public", "private-team"))
@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("license_status", "unknown", "license"),
        ("license_status", "denied", "license"),
        ("license_status", "not_reviewed", "license"),
        ("dua_status", "denied", "dua"),
        ("dua_status", "not_reviewed", "dua"),
        ("custodian_status", "unknown", "custodian"),
        ("custodian_status", "denied", "custodian"),
        ("custodian_status", "not_reviewed", "custodian"),
        ("deidentification_status", "unknown", "deidentification"),
        ("deidentification_status", "denied", "deidentification"),
        ("deidentification_status", "not_reviewed", "deidentification"),
        ("approved_by", "", "approved"),
        ("approved_by", "   ", "approved"),
        ("evidence_reference", "", "evidence"),
        ("evidence_reference", "   ", "evidence"),
    ),
)
def test_uploadable_visibility_fails_closed_on_blocking_governance(visibility, field, value, match):
    from src.data.release.governance import validate_governance

    family = family_config(visibility=visibility, **{field: value})
    with pytest.raises(ValueError, match=match):
        validate_governance(family)


@pytest.mark.parametrize("visibility", ("public", "private-team"))
@pytest.mark.parametrize("field", ("approved_by", "evidence_reference"))
@pytest.mark.parametrize("value", (None, True, False))
def test_uploadable_visibility_fails_closed_on_non_string_approver_or_evidence(visibility, field, value):
    from src.data.release.governance import validate_governance

    family = family_config(visibility=visibility, **{field: value})
    match = "approved" if field == "approved_by" else "evidence"
    with pytest.raises(ValueError, match=match):
        validate_governance(family)


def test_approved_family_is_included_when_others_remain_blocked():
    from src.data.release.governance import approved_families, validate_governance

    config = release_config_with_family(
        dataset_id="D01",
        visibility="private-team",
        license_status="verified",
        dua_status="not_applicable",
        custodian_status="verified",
        deidentification_status="reviewed",
        approved_by="custodian",
        evidence_reference="ticket-1",
    )
    approved = approved_families(config)

    assert validate_governance(approved[0]) is None
    assert tuple(family.dataset_id for family in approved) == ("D01",)
    assert approved[0].governance.can_upload is True
    assert len(config.families) == 19
    assert sum(1 for family in config.families if family.visibility == "blocked") == 18


def test_license_other_is_not_blocking_for_private_team():
    from src.data.release.governance import approved_families, validate_governance

    family = family_config(visibility="private-team", license_status="other")
    config = release_config_with_family(visibility="private-team", license_status="other")

    assert validate_governance(family) is None
    assert tuple(item.dataset_id for item in approved_families(config)) == ("D01",)


def test_unresolved_private_team_family_is_excluded_from_upload_plan():
    from src.data.release.governance import approved_families

    config = release_config_with_family(visibility="private-team", dua_status="unknown")
    assert approved_families(config) == ()


def test_approved_families_preserve_catalog_order():
    from src.data.release.governance import approved_families
    from src.data.release.models import ReleaseConfig

    document = _catalog_document()
    for dataset_id in ("D05", "D02"):
        family = dict(document[dataset_id])
        family["visibility"] = "private-team"
        family["governance"] = _approved_governance(visibility="private-team")
        document[dataset_id] = family
    config = ReleaseConfig.from_dict(document)

    assert tuple(family.dataset_id for family in approved_families(config)) == ("D02", "D05")


def test_validate_canonical_exclusions_accepts_current_catalog():
    from src.data.release.governance import load_release_config, validate_canonical_exclusions

    config = load_release_config(_RELEASE_CONFIG_PATH)
    assert validate_canonical_exclusions(config) is None


def test_validate_canonical_exclusions_requires_canonical_repo():
    from src.data.release.governance import load_release_config, validate_canonical_exclusions

    config = load_release_config(_RELEASE_CONFIG_PATH)
    object.__setattr__(config, "canonical_exclusions", ())
    with pytest.raises(ValueError, match="canonical"):
        validate_canonical_exclusions(config)


def test_validate_canonical_exclusions_rejects_family_targeting_canonical_repo():
    from src.data.release.governance import load_release_config, validate_canonical_exclusions

    config = load_release_config(_RELEASE_CONFIG_PATH)
    d15 = next(family for family in config.families if family.dataset_id == "D15")
    object.__setattr__(d15, "repo_id", " MadTeam/thinking-bciciv2a")
    with pytest.raises(ValueError, match="canonical"):
        validate_canonical_exclusions(config)


def test_validate_canonical_exclusions_rejects_family_repo_matching_exclusion():
    from src.data.release.governance import load_release_config, validate_canonical_exclusions
    from src.data.release.models import CanonicalExclusion

    config = load_release_config(_RELEASE_CONFIG_PATH)
    extra = CanonicalExclusion(repo_id="madteam/thinking-d01", reason="must not overlap")
    object.__setattr__(config, "canonical_exclusions", config.canonical_exclusions + (extra,))
    with pytest.raises(ValueError, match="canonical"):
        validate_canonical_exclusions(config)


def test_load_rejects_d15_targeting_canonical_bciciv2a(tmp_path):
    from src.data.release.governance import load_release_config

    document = _catalog_document()
    document["D15"]["repo_id"] = _CANONICAL_BCICIV2A_REPO_ID
    path = _write_catalog(tmp_path, document)
    with pytest.raises(ValueError, match="canonical"):
        load_release_config(path)


def test_missing_config_path_fails_closed(tmp_path):
    from src.data.release.governance import load_release_config

    missing = tmp_path / "missing.families.release.json"
    with pytest.raises(ValueError):
        load_release_config(missing)


def test_duplicate_repo_ids_are_rejected(tmp_path):
    from src.data.release.governance import load_release_config

    document = _catalog_document()
    document["D02"]["repo_id"] = document["D01"]["repo_id"]
    path = _write_catalog(tmp_path, document)
    with pytest.raises(ValueError):
        load_release_config(path)


def test_absolute_local_root_is_rejected(tmp_path):
    from src.data.release.governance import load_release_config

    document = _catalog_document()
    document["D01"]["local_root"] = "E:/AI_thucchien/THINKING/datasets/openneuro/ds003626_inner_speech"
    path = _write_catalog(tmp_path, document)
    with pytest.raises(ValueError, match="local_root"):
        load_release_config(path)
