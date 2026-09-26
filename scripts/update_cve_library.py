"""Refresh the vendored local CVE/KEV evidence manifest.

The project keeps a small, reviewable offline snapshot rather than silently
trusting network access during CI. This command can import a JSON document with
records supplied by an approved NVD/CISA review, or refresh the snapshot
metadata without inventing advisory records.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "references" / "cve-library"
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def _read_records(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and isinstance(document.get("records"), list):
        return [item for item in document["records"] if isinstance(item, dict)]
    if isinstance(document, dict) and document.get("cve"):
        return [document]
    raise ValueError("input must be a CVE record object or {records: [...]} document")


def _fetch_kev(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "thinking-eeg-cve-refresh/1.0"})
    with urlopen(request, timeout=15) as response:  # noqa: S310 - explicit approved CISA URL
        return json.loads(response.read().decode("utf-8"))


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the offline CVE/KEV evidence snapshot")
    parser.add_argument("--input", type=Path, help="approved JSON file containing CVE records")
    parser.add_argument("--fetch-kev", action="store_true", help="fetch the CISA KEV catalog")
    parser.add_argument("--kev-url", default=KEV_URL)
    parser.add_argument("--library", type=Path, default=LIBRARY)
    args = parser.parse_args(argv)

    library = args.library if args.library.is_absolute() else ROOT / args.library
    library.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    if args.input:
        source = args.input if args.input.is_absolute() else ROOT / args.input
        records = _read_records(source)
        for record in records:
            cve = str(record.get("cve", "")).lower()
            if not cve.startswith("cve-"):
                raise ValueError(f"invalid CVE identifier: {record.get('cve')!r}")
            record = {**record, "source": record.get("source", f"approved snapshot {today}"), "snapshot_date": today}
            _write_json(library / f"{cve}.json", record)
        print(f"wrote {len(records)} CVE record(s) to {library}")

    if args.fetch_kev:
        document = _fetch_kev(args.kev_url)
        vulnerabilities = document.get("vulnerabilities", []) if isinstance(document, dict) else []
        _write_json(
            library / "kev.json",
            {
                "source": args.kev_url,
                "snapshot_date": today,
                "kev_count": len(vulnerabilities),
                "vulnerabilities": vulnerabilities,
            },
        )
        print(f"wrote CISA KEV snapshot ({len(vulnerabilities)} entries)")

    readme = library / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        lines = text.splitlines()
        lines = [line for line in lines if not line.startswith("Snapshot date:")]
        lines.insert(3, f"Snapshot date: {today}")
        readme.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    elif not args.input and not args.fetch_kev:
        raise SystemExit("no refresh source selected; use --input and/or --fetch-kev")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
