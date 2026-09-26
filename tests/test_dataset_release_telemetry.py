"""Allowlisted dataset-release SIEM events and SOC detection text."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGMA_RULE = ROOT / "soc" / "sigma_rules" / "thinking-dataset-release.yml"
KQL_RULE = ROOT / "soc" / "kql" / "thinking_dataset_release.kql"
SPLUNK_RULE = ROOT / "soc" / "splunk" / "thinking_dataset_release.spl"
RELEASE_DOCS = ROOT / "docs" / "dataset_release.md"

SPEC_EVENT_TYPES = (
    "dataset_inventory_reconciliation_failed",
    "dataset_release_plan_created",
    "dataset_shard_build_failed",
    "dataset_upload_blocked",
    "dataset_upload_batch_failed",
    "dataset_release_signature_failed",
    "dataset_shard_verification_failed",
    "dataset_release_verified",
)
DETECTION_EVENT_TYPES = (
    "dataset_release_verification_failed",
    "dataset_inventory_reconciliation_failed",
    "dataset_upload_blocked",
)
ALLOWLISTED_FIELDS = (
    "event_type",
    "timestamp",
    "correlation_id",
    "user_sub",
    "dataset_id",
    "release_id",
    "manifest_sha256",
    "path_hash",
    "provider",
    "mode",
    "result",
    "reason",
    "count",
    "total_size_bytes",
)
DISALLOWED_EVENT_FIELDS = (
    "path",
    "raw_eeg",
    "prompt",
    "HF_TOKEN",
    "hf_token",
    "token",
    "secret",
    "private_key",
    "password",
    "credential",
)
FORBIDDEN_RULE_MARKERS = (
    "HF_TOKEN",
    "hf_token",
    "private_key",
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "raw_eeg",
    ".edf",
    "C:/",
    "C:\\",
    "/home/",
    "/Users/",
    "password",
    "api_key",
    "siem_token",
)


def test_release_telemetry_drops_raw_data_tokens_and_prompts():
    from src.soc.siem import sanitize_release_event

    safe = sanitize_release_event({
        "event_type": "dataset_release_verified",
        "dataset_id": "D01",
        "release_id": "r1",
        "manifest_sha256": "a" * 64,
        "raw_eeg": "must-drop",
        "prompt": "must-drop",
        "HF_TOKEN": "must-drop",
        "path": "C:/sensitive/raw/file.edf",
    })
    assert "raw_eeg" not in safe
    assert "prompt" not in safe
    assert "HF_TOKEN" not in safe
    assert "path" not in safe


def test_release_telemetry_hashes_path_when_supplied():
    from src.soc.siem import sanitize_release_event

    raw_path = "C:/sensitive/raw/file.edf"
    safe = sanitize_release_event({
        "event_type": "dataset_shard_verification_failed",
        "path": raw_path,
    })
    assert "path" not in safe
    assert safe["path_hash"] == hashlib.sha256(raw_path.encode("utf-8")).hexdigest()


def test_release_telemetry_keeps_allowlisted_fields():
    from src.soc.siem import sanitize_release_event

    event = {
        "event_type": "dataset_release_plan_created",
        "timestamp": "2026-09-25T00:00:00Z",
        "correlation_id": "corr-1",
        "user_sub": "operator",
        "dataset_id": "D01",
        "release_id": "r1",
        "manifest_sha256": "b" * 64,
        "path_hash": "c" * 64,
        "provider": "huggingface",
        "mode": "plan",
        "result": "ok",
        "reason": "created",
        "count": 3,
        "total_size_bytes": 1024,
    }
    safe = sanitize_release_event(event)
    assert safe == event


def test_release_telemetry_drops_extra_keys():
    from src.soc.siem import sanitize_release_event

    safe = sanitize_release_event({
        "event_type": "dataset_upload_blocked",
        "dataset_id": "D02",
        "tenant_id": "must-drop-from-release-sanitizer",
        "extra": "nope",
        "token": "nope",
        "eeg": [0.1, 0.2],
        "model": "must-drop",
    })
    assert safe == {
        "event_type": "dataset_upload_blocked",
        "dataset_id": "D02",
    }


def test_release_telemetry_allows_spec_and_detection_event_types():
    from src.soc.siem import sanitize_release_event

    for event_type in (*SPEC_EVENT_TYPES, "dataset_release_verification_failed"):
        safe = sanitize_release_event({"event_type": event_type, "reason": None})
        assert safe == {"event_type": event_type}


def test_soc_rules_name_repeated_release_detections():
    for rule_path in (SIGMA_RULE, KQL_RULE, SPLUNK_RULE):
        assert rule_path.is_file(), rule_path
        text = rule_path.read_text(encoding="utf-8")
        for event_type in DETECTION_EVENT_TYPES:
            assert event_type in text, f"{rule_path.name} missing {event_type}"


def test_soc_rule_text_uses_only_allowlisted_fields():
    field_pattern = re.compile(r"\b(" + "|".join(map(re.escape, (
        *ALLOWLISTED_FIELDS,
        *DISALLOWED_EVENT_FIELDS,
    ))) + r")\b")
    for rule_path in (SIGMA_RULE, KQL_RULE, SPLUNK_RULE):
        assert rule_path.is_file(), rule_path
        text = rule_path.read_text(encoding="utf-8")
        named = set(field_pattern.findall(text))
        assert named <= set(ALLOWLISTED_FIELDS), f"{rule_path.name} named {named - set(ALLOWLISTED_FIELDS)}"
        for marker in FORBIDDEN_RULE_MARKERS:
            assert marker not in text, f"{rule_path.name} contains {marker!r}"
        assert "PEM" not in text
        sigma_text = SIGMA_RULE.read_text(encoding="utf-8")
        assert "author: Techwaves EGY" in sigma_text
        assert "product: thinking-eeg" in sigma_text


def test_dataset_release_docs_describe_event_schema():
    assert RELEASE_DOCS.is_file()
    text = RELEASE_DOCS.read_text(encoding="utf-8")
    for event_type in (*SPEC_EVENT_TYPES, "dataset_release_verification_failed"):
        assert event_type in text
    for field in ALLOWLISTED_FIELDS:
        assert field in text
    for marker in ("HF_TOKEN=", "BEGIN PRIVATE KEY", "raw_eeg"):
        assert marker not in text
