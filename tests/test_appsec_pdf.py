"""Regression checks for AppSec report rendering modes."""

import copy

from scripts import generate_appsec_pdf


def _phase4_report_fixture() -> dict:
    return {
        "meta": {
            "title": "Pre-Fix Security Assessment Report — Phase 4 API",
            "subtitle": "THINKING EEG Platform",
            "classification": "PRE-FIX | CONFIDENTIAL",
            "company": "Techwaves EGY",
            "contact": "info@techwaves-egy.com",
            "date": "2026-09-22",
            "project": "thinking-eeg 0.2.0",
            "worktree": "ci-fixture",
            "assessor": "Blue Team AppSec",
            "authorization": "Internal assessment of an authorized local codebase.",
        },
        "hard_stop": "Pre-fix evidence gate",
        "executive_summary": "Phase 4 summary.",
        "scope": ["API routes"],
        "out_of_scope": ["External probing"],
        "evidence": ["Offline regression evidence"],
        "domains": [{"id": 1, "name": "Authentication", "score": 6.0, "rationale": "JWT and MFA controls exist."}],
        "dfd": {
            "entities": ["Client"],
            "processes": ["FastAPI"],
            "stores": ["Dataset root"],
            "trust_boundaries": ["Client to API"],
        },
        "stride": [{"element": "API", "S": "spoof", "T": "tamper", "R": "repudiate", "I": "disclose", "D": "deny", "E": "elevate"}],
        "findings": [
            {
                "id": "TW-2026-003",
                "title": "Durable MFA evidence gap",
                "severity": "High",
                "cvss": 7.1,
                "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N",
                "cwe": "CWE-287",
                "kev": False,
                "component": "src/api/mfa.py",
                "exploitability_here": "Process-local enrollment loses state across restart.",
                "poc": "Restarting the API clears process memory.",
                "remediation": "Persist MFA enrollment in an approved durable store.",
            }
        ],
        "cve_table": [{"package": "starlette", "version": "via FastAPI", "cve": "N/A", "kev": False, "note": "No unresolved local KEV."}],
        "planned_change_constraints": ["Do not claim durable MFA while process-local."],
    }


def test_post_fix_report_is_detected_from_classification():
    report = {"meta": {"classification": "POST-FIX | CONFIDENTIAL"}}

    assert generate_appsec_pdf._is_post_fix(report) is True


def test_pre_fix_report_remains_pre_fix_by_default():
    report = {"meta": {"classification": "PRE-FIX | CONFIDENTIAL"}}

    assert generate_appsec_pdf._is_post_fix(report) is False


def test_post_fix_story_uses_post_fix_report_sections():
    report = copy.deepcopy(_phase4_report_fixture())
    report["meta"]["classification"] = "POST-FIX | CONFIDENTIAL"
    report["hard_stop"] = "Post-fix evidence gate"
    report["dfd_narrative"] = "post-fix DFD narrative"
    report["cve_narrative"] = "post-fix CVE narrative"
    report["soc_narrative"] = "post-fix SOC narrative"
    report["constraints_heading"] = "Residual controls"
    report["conclusion"] = "post-fix conclusion"
    report["telemetry_findings"] = ["TW-2026-003"]

    story = generate_appsec_pdf.build_story(report, generate_appsec_pdf._styles())
    text = "\n".join(
        getattr(item, "text", getattr(item, "getPlainText", lambda: "")())
        for item in story
    )

    assert "post-fix DFD narrative" in text
    assert "post-fix CVE narrative" in text
    assert "post-fix SOC narrative" in text
    assert "Residual controls" in text
    assert "post-fix conclusion" in text
    assert "Planned Brennan loader reads" not in text
