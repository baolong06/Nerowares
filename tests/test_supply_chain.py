"""Supply-chain evidence must distinguish declared and ambient dependencies."""
from __future__ import annotations

import json
from pathlib import Path


def test_cve_checker_uses_structured_status_and_ignores_readme_keywords(tmp_path, monkeypatch):
    from scripts import check_cve

    library = tmp_path / "cve-library"
    library.mkdir()
    (library / "README.md").write_text("CRITICAL and CISA KEV are tracked here\n", encoding="utf-8")
    (library / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "records": [
                {"cve": "CVE-1", "severity": "CRITICAL", "status": "remediated", "kev": False},
                {"cve": "CVE-2", "severity": "MEDIUM", "status": "open", "kev": False},
            ],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_cve, "LOCAL_CVE", library)
    assert check_cve.main() == 0


def test_cve_checker_fails_for_unresolved_critical_or_kev(tmp_path, monkeypatch):
    from scripts import check_cve

    library = tmp_path / "cve-library"
    library.mkdir()
    (library / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "records": [
                {"cve": "CVE-9", "severity": "HIGH", "status": "open", "kev": False},
                {"cve": "CVE-10", "severity": "MEDIUM", "status": "open", "kev": True},
            ],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_cve, "LOCAL_CVE", library)
    assert check_cve.main() == 1


def test_declared_and_ambient_sbom_have_distinct_scope(tmp_path, monkeypatch):
    from scripts import generate_sbom

    declared = tmp_path / "declared.json"
    assert generate_sbom.main(["--mode", "declared", "--output", str(declared)]) == 0
    declared_doc = json.loads(declared.read_text(encoding="utf-8"))
    assert declared_doc["metadata"]["properties"]["dependency_scope"] == "declared"
    declared_names = {component["name"].lower() for component in declared_doc["components"]}
    assert "fastapi" in declared_names
    assert "transformers" not in declared_names

    monkeypatch.setattr(generate_sbom, "_pip_freeze", lambda: "fastapi==0.141.1\ntransformers==4.45.0\n")
    ambient = tmp_path / "ambient.json"
    assert generate_sbom.main(["--mode", "ambient", "--output", str(ambient)]) == 0
    ambient_doc = json.loads(ambient.read_text(encoding="utf-8"))
    assert ambient_doc["metadata"]["properties"]["dependency_scope"] == "ambient"
    ambient_names = {component["name"].lower() for component in ambient_doc["components"]}
    assert "transformers" in ambient_names
