"""Offline CVE snapshot updater contracts."""
from __future__ import annotations

import json
from pathlib import Path


def test_cve_updater_imports_approved_records_without_network(tmp_path):
    from scripts import update_cve_library

    source = tmp_path / "approved.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "cve": "CVE-2099-0001",
                        "severity": "HIGH",
                        "status": "open",
                        "kev": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    library = tmp_path / "cve-library"
    library.mkdir(parents=True, exist_ok=True)
    (library / "README.md").write_text("# Local CVE library\n", encoding="utf-8")

    assert update_cve_library.main(["--input", str(source), "--library", str(library)]) == 0

    record = json.loads((library / "cve-2099-0001.json").read_text(encoding="utf-8"))
    assert record["cve"] == "CVE-2099-0001"
    assert record["snapshot_date"]
    assert "Snapshot date:" in (library / "README.md").read_text(encoding="utf-8")


def test_cve_updater_rejects_invalid_identifier(tmp_path):
    from scripts import update_cve_library

    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"cve": "not-a-cve", "severity": "HIGH"}), encoding="utf-8")

    try:
        update_cve_library.main(["--input", str(source), "--library", str(tmp_path / "library")])
    except ValueError as exc:
        assert "invalid CVE identifier" in str(exc)
    else:
        raise AssertionError("invalid CVE identifier was accepted")
