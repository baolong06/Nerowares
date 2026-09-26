"""Grep references/cve-library (when present) for CRITICAL / CISA KEV records."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CVE = ROOT / "references" / "cve-library"
_RESOLVED = {"remediated", "resolved", "closed", "accepted"}


def _records() -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    for path in sorted(LOCAL_CVE.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and isinstance(document.get("records"), list):
            records.extend(
                (path, item) for item in document["records"] if isinstance(item, dict)
            )
        elif isinstance(document, dict) and document.get("cve"):
            records.append((path, document))
    return records


def main() -> int:
    if not LOCAL_CVE.exists():
        print("local CVE library not present at references/cve-library — failing closed", file=sys.stderr)
        print("hint: run python scripts/update_cve_library.py or vendor the library before CI", file=sys.stderr)
        return 2

    records = _records()
    unresolved_critical: list[tuple[Path, dict]] = []
    unresolved_kev: list[tuple[Path, dict]] = []
    for path, record in records:
        status = str(record.get("status", "open")).lower()
        if status in _RESOLVED:
            continue
        if str(record.get("severity", "")).upper() == "CRITICAL":
            unresolved_critical.append((path, record))
        if bool(record.get("kev")):
            unresolved_kev.append((path, record))

    print(f"CRITICAL unresolved: {len(unresolved_critical)}")
    for path, record in unresolved_critical:
        print(f"{path}: {record.get('cve', 'unknown')}")
    print(f"CISA KEV unresolved: {len(unresolved_kev)}")
    for path, record in unresolved_kev:
        print(f"{path}: {record.get('cve', 'unknown')}")
    print(f"local CVE library records: {len(records)}")
    return 1 if unresolved_critical or unresolved_kev else 0


if __name__ == "__main__":
    raise SystemExit(main())
