"""Render a Techwaves EGY AppSec PDF from a JSON assessment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = HexColor("#0B1F3A")
GOLD = HexColor("#C4A35A")
RED = HexColor("#8B1E1E")
AMBER = HexColor("#8A5A00")
GREEN = HexColor("#1F4E3D")
SLATE = HexColor("#243040")
LIGHT = HexColor("#F4F1EA")
ROW = HexColor("#EEF2F6")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "Pre-Fix-AppSec-Assessment-Brennan-2026-09-21.json"
DEFAULT_OUTPUT = ROOT / "reports" / "Pre-Fix-AppSec-Report-Brennan-2026-09-21.pdf"


def _is_post_fix(report: dict) -> bool:
    classification = str(report.get("meta", {}).get("classification", ""))
    return "POST-FIX" in classification.upper()


def _finding_summary(report: dict) -> str:
    domains = report.get("domains", [])
    mean = sum(float(d["score"]) for d in domains) / len(domains) if domains else 0.0
    findings = report.get("findings", [])
    if not _is_post_fix(report):
        counts = {level: sum(f.get("severity") == level for f in findings) for level in ("Critical", "High", "Medium", "Low")}
        kev = sum(bool(f.get("kev")) for f in findings)
        return (
            f"Mean of {len(domains)} domains: <b>{mean:.2f} / 10.0</b>. "
            f"Critical: {counts['Critical']}. High: {counts['High']}. "
            f"Medium: {counts['Medium']}. CISA KEV findings: {kev}."
        )
    remediated = sum(str(f.get("status", "")).lower() in {"remediated", "resolved", "closed"} for f in findings)
    residual = len(findings) - remediated
    kev = [f.get("id", "") for f in findings if f.get("kev") and f.get("status", "").lower() not in {"remediated", "resolved", "closed"}]
    kev_text = ", ".join(kev) if kev else "none"
    return (
        f"Mean of {len(domains)} domains: <b>{mean:.2f} / 10.0</b>. "
        f"Findings tracked: {len(findings)}. Remediated: {remediated}. "
        f"Residual: {residual}. Residual CISA KEV: {kev_text}."
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle("cover_kicker", parent=base["Normal"], fontName="Times-Bold", fontSize=11, textColor=GOLD, alignment=TA_CENTER, tracking=1),
        "cover_title": ParagraphStyle("cover_title", parent=base["Title"], fontName="Times-Bold", fontSize=22, textColor=NAVY, alignment=TA_CENTER, leading=26, spaceAfter=6),
        "cover_sub": ParagraphStyle("cover_sub", parent=base["Normal"], fontName="Times-Italic", fontSize=11, textColor=SLATE, alignment=TA_CENTER, leading=14, spaceAfter=8),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Times-Bold", fontSize=14, textColor=NAVY, spaceBefore=10, spaceAfter=6, leading=18),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Times-Bold", fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4, leading=15),
        "body": ParagraphStyle("body", parent=base["Normal"], fontName="Times-Roman", fontSize=9.5, textColor=SLATE, alignment=TA_JUSTIFY, leading=13, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["Normal"], fontName="Times-Roman", fontSize=8, textColor=SLATE, leading=11),
        "cell": ParagraphStyle("cell", parent=base["Normal"], fontName="Times-Roman", fontSize=8, textColor=SLATE, leading=10),
        "cell_b": ParagraphStyle("cell_b", parent=base["Normal"], fontName="Times-Bold", fontSize=8, textColor=NAVY, leading=10),
        "stop": ParagraphStyle("stop", parent=base["Normal"], fontName="Times-Bold", fontSize=9.5, textColor=RED, alignment=TA_LEFT, leading=13, spaceAfter=8),
        "meta": ParagraphStyle("meta", parent=base["Normal"], fontName="Times-Roman", fontSize=9, textColor=SLATE, alignment=TA_CENTER, leading=12),
        "code": ParagraphStyle("code", parent=base["Code"], fontName="Courier", fontSize=7, textColor=SLATE, leading=9, leftIndent=4, rightIndent=4),
        "footer": ParagraphStyle("footer", parent=base["Normal"], fontName="Times-Roman", fontSize=8, textColor=white, alignment=TA_LEFT),
        "footer_r": ParagraphStyle("footer_r", parent=base["Normal"], fontName="Times-Roman", fontSize=8, textColor=GOLD, alignment=TA_RIGHT),
    }


def _severity_color(level: str) -> HexColor:
    return {"Critical": RED, "High": HexColor("#B45309"), "Medium": AMBER, "Low": GREEN}.get(level, SLATE)


def _header_footer(canvas, doc, meta: dict) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 16 * mm, width, 16 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, height - 17.2 * mm, width, 1.2 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Times-Bold", 11)
    canvas.drawString(16 * mm, height - 9 * mm, meta["company"].upper())
    canvas.setFont("Times-Roman", 8)
    canvas.drawRightString(width - 16 * mm, height - 7.5 * mm, meta["contact"])
    canvas.drawRightString(width - 16 * mm, height - 12 * mm, meta["classification"])
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 12 * mm, width, 0.8 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Times-Roman", 8)
    canvas.drawString(16 * mm, 5 * mm, f"{meta['company']}  |  {meta['contact']}")
    canvas.drawRightString(width - 16 * mm, 5 * mm, f"Page {doc.page}  |  {meta['date']}")
    canvas.restoreState()


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def _kv_table(rows: list[tuple[str, str]], styles: dict, col1=38 * mm) -> Table:
    data = [[_p(k, styles["cell_b"]), _p(v, styles["cell"])] for k, v in rows]
    table = Table(data, colWidths=[col1, 142 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("BACKGROUND", (1, 0), (1, -1), white),
                ("BOX", (0, 0), (-1, -1), 0.3, NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.2, HexColor("#C5CDD6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _score_table(domains: list[dict], styles: dict) -> Table:
    header = [
        _p("#", styles["cell_b"]),
        _p("Domain", styles["cell_b"]),
        _p("Score", styles["cell_b"]),
        _p("Rationale", styles["cell_b"]),
    ]
    data = [header]
    for item in domains:
        data.append(
            [
                _p(str(item["id"]), styles["cell"]),
                _p(item["name"], styles["cell_b"]),
                _p(f"{item['score']:.1f}", styles["cell_b"]),
                _p(item["rationale"], styles["cell"]),
            ]
        )
    mean = sum(d["score"] for d in domains) / len(domains)
    data.append([_p("", styles["cell"]), _p("Mean", styles["cell_b"]), _p(f"{mean:.2f}", styles["cell_b"]), _p("0 = missing control, 10 = production-hardened", styles["cell"])])
    table = Table(data, colWidths=[12 * mm, 48 * mm, 18 * mm, 102 * mm])
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.4, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, HexColor("#C5CDD6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (2, 1), (2, -1), "CENTER"),
    ]
    for index, item in enumerate(domains, start=1):
        if index % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, index), (-1, index), ROW))
        score = item["score"]
        color = RED if score < 5 else (AMBER if score < 7 else GREEN)
        style_cmds.append(("TEXTCOLOR", (2, index), (2, index), color))
    table.setStyle(TableStyle(style_cmds))
    return table


def _finding_block(finding: dict, styles: dict) -> list:
    color = _severity_color(finding["severity"])
    title = _p(f"{finding['id']}  —  {finding['title']}", styles["h2"])
    rows = [
        ("Severity", f"{finding['severity']}  |  CVSS {finding['cvss']:.1f}"),
        ("Vector", finding["vector"]),
        ("CWE", finding["cwe"]),
        ("CISA KEV", "YES" if finding.get("kev") else "No"),
        ("Component", finding["component"]),
        ("Exploitability here", finding["exploitability_here"]),
    ]
    if finding.get("status"):
        rows.insert(1, ("Status", finding["status"]))
    meta = _kv_table(rows, styles)
    body = [
        title,
        meta,
        Spacer(1, 2 * mm),
        _p("<b>PoC / evidence.</b> " + finding["poc"], styles["body"]),
        _p("<b>Remediation.</b> " + finding["remediation"], styles["body"]),
        HRFlowable(width="100%", thickness=0.4, color=color, spaceAfter=6),
    ]
    return [KeepTogether(body)]


def _sigma_for(finding_id: str) -> str:
    rules = {
        "TW-2026-001": '''title: THINKING untrusted PyTorch checkpoint load
id: 7c1a9e2b-4d80-4b11-9f3a-brennan-torch-load
status: experimental
description: Detects torch.load / weights_only checkpoint loads from dataset roots.
author: Techwaves EGY
date: 2026-09-21
logsource:
  product: thinking-eeg
  service: training
detection:
  selection:
    event_type: model_load
    loader: torch.load
  condition: selection
level: critical
tags:
  - attack.execution
  - cve.2025-32434
  - cisa.kev''',
        "TW-2026-002": '''title: THINKING API availability collapse
id: 2e6c0d41-aa11-4f08-9c77-brennan-starlette-dos
status: experimental
description: Burst of 5xx or >2s elapsed on unauthenticated or authenticated API paths.
author: Techwaves EGY
date: 2026-09-21
logsource:
  product: thinking-eeg
  service: api
detection:
  selection:
    event_type: api_request
  high_status:
    status: 500
  slow:
    elapsed_ms: '>2000'
  condition: selection and (high_status or slow) | count(correlation_id) by src_ip > 20
  timeframe: 5m
level: high
tags:
  - attack.impact
  - attack.t1499''',
        "TW-2026-003": '''title: THINKING decode without durable MFA
id: 9b71d3c0-1a22-4e5f-8d44-brennan-mfa-bypass
status: experimental
description: Successful decode/ingest without a prior MFA success for the same sub.
author: Techwaves EGY
date: 2026-09-21
logsource:
  product: thinking-eeg
  service: api
detection:
  selection:
    event_type: api_request
    path|contains:
      - /decode
      - /ingest/validate
    status: 200
  filter_mfa:
    event_type: mfa_verification_succeeded
  condition: selection
level: high
tags:
  - attack.defense_evasion
  - attack.t1556''',
        "TW-2026-004": '''title: THINKING BrainVision ingest attempt
id: 5a88e0f1-77c3-4b90-a219-brennan-brainvision
status: experimental
description: Ingest validation involving .eeg .vhdr or .vmrk before or after allowlist change.
author: Techwaves EGY
date: 2026-09-21
logsource:
  product: thinking-eeg
  service: api
detection:
  selection:
    event_type: api_request
    path: /ingest/validate
    source_path|contains:
      - .vhdr
      - .eeg
      - .vmrk
  condition: selection
level: high
tags:
  - attack.ingress_tool_transfer
  - attack.t1105''',
    }
    return rules[finding_id]


def _kql_for(finding_id: str) -> str:
    mapping = {
        "TW-2026-001": "AuditLogs\n| where event_type == 'model_load' and loader == 'torch.load'\n| project timestamp, user_sub, path, src_ip",
        "TW-2026-002": "AuditLogs\n| where event_type == 'api_request' and (status >= 500 or elapsed_ms > 2000)\n| summarize c=count() by src_ip, bin(timestamp, 5m)\n| where c > 20",
        "TW-2026-003": "let ok = AuditLogs | where event_type == 'mfa_verification_succeeded';\nAuditLogs\n| where event_type == 'api_request' and status == 200 and path in ('/decode','/ingest/validate')\n| join kind=leftanti ok on user_sub\n| summarize c=count() by user_sub, bin(timestamp, 15m)",
        "TW-2026-004": "AuditLogs\n| where path == '/ingest/validate' and source_path matches regex r'\\.(vhdr|eeg|vmrk)$'\n| project timestamp, user_sub, status, src_ip",
    }
    return mapping[finding_id]


def _spl_for(finding_id: str) -> str:
    mapping = {
        "TW-2026-001": 'index=thinking event_type=model_load loader="torch.load" | stats count by user_sub, path',
        "TW-2026-002": "index=thinking event_type=api_request (status>=500 OR elapsed_ms>2000) | bin span=5m _time | stats count by src_ip | where count>20",
        "TW-2026-003": 'index=thinking status=200 (path="/decode" OR path="/ingest/validate") NOT [search index=thinking event_type=mfa_verification_succeeded | fields user_sub] | stats count by user_sub',
        "TW-2026-004": r'index=thinking path="/ingest/validate" source_path="*.vhdr" OR source_path="*.eeg" OR source_path="*.vmrk"',
    }
    return mapping[finding_id]


def build_story(report: dict, styles: dict) -> list:
    meta = report["meta"]
    post_fix = _is_post_fix(report)
    story: list = []
    dfd_narrative = report.get("dfd_narrative") if post_fix else (
        "Client --HTTPS--> FastAPI (CORS, 60/min limiter, correlation) --JWT/MFA--> /decode | "
        "/ingest/validate. /ingest/validate --safe_resolve--> dataset root. Optional Vault KV for "
        "JWT_SECRET. Optional SIEM HTTPS POST of allowlisted audit fields. Planned Brennan loader "
        "reads BrainVision under the NEMAR root; that code is not present."
    )
    cve_narrative = report.get("cve_narrative") if post_fix else (
        "Local path references/cve-library/ is absent. The table is from pip-audit of pinned THINKING "
        "extras plus public KEV knowledge for CVE-2025-32434. Ambient site-packages (for example "
        "ChromaDB) are out of this application's declared dependency set and are not scored as "
        "product findings."
    )
    soc_narrative = report.get("soc_narrative") if post_fix else (
        "Existing committed rules: soc/sigma_rules/thinking-jwt-rejections.yml and "
        "thinking-mfa-failures.yml (validated). Additional detections below are specified in this "
        "report only; they were not written into soc/ because this is a Pre-Fix gate."
    )
    telemetry_ids = report.get("telemetry_findings") if post_fix else [
        "TW-2026-001", "TW-2026-002", "TW-2026-003", "TW-2026-004"
    ]
    constraints_heading = report.get("constraints_heading") if post_fix else (
        "10. Constraints if the Brennan loader is later approved"
    )
    conclusion = report.get("conclusion") if post_fix else (
        "No evidence of active external compromise was found. Do not implement load_brennan_epochs "
        "or widen ALLOWED_EXTENSIONS until this Pre-Fix report is explicitly approved. Contact: "
        "info@techwaves-egy.com."
    )
    end_label = "End of Post-Fix report." if post_fix else "End of Pre-Fix report."

    story += [
        Spacer(1, 8 * mm),
        _p("APPLICATION SECURITY ASSESSMENT", styles["cover_kicker"]),
        Spacer(1, 3 * mm),
        _p(meta["title"], styles["cover_title"]),
        _p(meta["subtitle"], styles["cover_sub"]),
        HRFlowable(width="100%", thickness=1.2, color=GOLD, spaceAfter=8),
        _p(f"{meta['company']}  ·  {meta['contact']}", styles["meta"]),
        _p(f"{meta['date']}  ·  {meta['classification']}", styles["meta"]),
        Spacer(1, 4 * mm),
        _p(report["hard_stop"], styles["stop"]),
        _kv_table(
            [
                ("Project", meta["project"]),
                ("Worktree", meta["worktree"]),
                ("Assessor", meta["assessor"]),
                ("Authorization", meta["authorization"]),
                ("Active compromise", "None observed. IR 15-minute escalation was not triggered."),
            ],
            styles,
            col1=42 * mm,
        ),
        Spacer(1, 4 * mm),
        _p("1. Executive summary", styles["h1"]),
        _p(report["executive_summary"], styles["body"]),
        _p(_finding_summary(report), styles["body"]),
        _p("2. Scope", styles["h1"]),
        _p("<b>In scope.</b> " + "; ".join(report["scope"]) + ".", styles["body"]),
        _p("<b>Out of scope.</b> " + "; ".join(report["out_of_scope"]) + ".", styles["body"]),
        _p("3. Evidence snapshot", styles["h1"]),
    ]
    for item in report["evidence"]:
        story.append(_p("• " + item, styles["body"]))
    story += [
        _p("4. Eleven-domain scores (0.0–10.0)", styles["h1"]),
        _score_table(report["domains"], styles),
        PageBreak(),
        _p("5. Data-flow diagram (narrative)", styles["h1"]),
        _p("<b>External entities.</b> " + ", ".join(report["dfd"]["entities"]) + ".", styles["body"]),
        _p("<b>Processes.</b> " + ", ".join(report["dfd"]["processes"]) + ".", styles["body"]),
        _p("<b>Stores.</b> " + ", ".join(report["dfd"]["stores"]) + ".", styles["body"]),
        _p("<b>Trust boundaries.</b> " + ", ".join(report["dfd"]["trust_boundaries"]) + ".", styles["body"]),
        _p(dfd_narrative, styles["body"]),
        _p("6. STRIDE", styles["h1"]),
    ]
    for row in report["stride"]:
        story.append(_p(f"<b>{row['element']}</b>", styles["body"]))
        story.append(
            _p(
                f"S: {row['S']}<br/>T: {row['T']}<br/>R: {row['R']}<br/>I: {row['I']}<br/>D: {row['D']}<br/>E: {row['E']}",
                styles["small"],
            )
        )
    story += [_p("7. Vulnerability dockets (CVSS v3.1)", styles["h1"])]
    for finding in report["findings"]:
        story.extend(_finding_block(finding, styles))
    story += [
        PageBreak(),
        _p("8. CVE / CISA KEV cross-reference", styles["h1"]),
        _p(cve_narrative, styles["body"]),
    ]
    cve_header = [
        _p("Package", styles["cell_b"]),
        _p("Version", styles["cell_b"]),
        _p("CVE", styles["cell_b"]),
        _p("KEV", styles["cell_b"]),
        _p("Note", styles["cell_b"]),
    ]
    cve_data = [cve_header]
    for row in report["cve_table"]:
        cve_data.append(
            [
                _p(row["package"], styles["cell"]),
                _p(row["version"], styles["cell"]),
                _p(row["cve"], styles["cell_b"]),
                _p("YES" if row["kev"] else "No", styles["cell"]),
                _p(row["note"], styles["cell"]),
            ]
        )
    cve_table = Table(cve_data, colWidths=[28 * mm, 22 * mm, 38 * mm, 14 * mm, 78 * mm])
    cve_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BOX", (0, 0), (-1, -1), 0.3, NAVY),
                ("INNERGRID", (0, 0), (-1, -1), 0.2, HexColor("#C5CDD6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(cve_table)
    story += [
        Spacer(1, 4 * mm),
        _p("9. SOC telemetry for High / Critical findings", styles["h1"]),
        _p(soc_narrative, styles["body"]),
    ]
    findings_by_id = {item["id"]: item for item in report["findings"]}
    for finding_id in telemetry_ids or []:
        finding = findings_by_id.get(finding_id)
        if finding is None or finding_id not in {"TW-2026-001", "TW-2026-002", "TW-2026-003", "TW-2026-004"}:
            continue
        story.append(_p(f"{finding_id} — {finding['title']}", styles["h2"]))
        story.append(_p("Sigma", styles["cell_b"]))
        story.append(Preformatted(_sigma_for(finding_id), styles["code"]))
        story.append(_p("KQL", styles["cell_b"]))
        story.append(Preformatted(_kql_for(finding_id), styles["code"]))
        story.append(_p("Splunk SPL", styles["cell_b"]))
        story.append(Preformatted(_spl_for(finding_id), styles["code"]))
    story += [_p(constraints_heading, styles["h1"])]
    for item in report["planned_change_constraints"]:
        story.append(_p("• " + item, styles["body"]))
    story += [
        _p("11. Conclusion", styles["h1"]),
        _p(conclusion, styles["body"]),
        _p(end_label, styles["meta"]),
    ]
    return story


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Techwaves EGY AppSec PDF")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    source = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = json.loads(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title=report["meta"]["title"],
        author=report["meta"]["company"],
        subject=report["meta"]["subtitle"],
    )
    doc.build(
        build_story(report, styles),
        onFirstPage=lambda c, d: _header_footer(c, d, report["meta"]),
        onLaterPages=lambda c, d: _header_footer(c, d, report["meta"]),
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
