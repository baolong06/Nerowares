"""Regression checks for AppSec report rendering modes."""

import copy
import json
from pathlib import Path

from scripts import generate_appsec_pdf


ROOT = Path(__file__).resolve().parents[1]


def test_post_fix_report_is_detected_from_classification():
    report = {"meta": {"classification": "POST-FIX | CONFIDENTIAL"}}

    assert generate_appsec_pdf._is_post_fix(report) is True


def test_pre_fix_report_remains_pre_fix_by_default():
    report = {"meta": {"classification": "PRE-FIX | CONFIDENTIAL"}}

    assert generate_appsec_pdf._is_post_fix(report) is False


def test_post_fix_story_uses_post_fix_report_sections():
    source = ROOT / "reports" / "Pre-Fix-AppSec-Assessment-Phase4-2026-09-22.json"
    report = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
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
