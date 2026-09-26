"""Minimal structural validation for committed Sigma YAML detections."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "soc" / "sigma_rules"
REQUIRED = ("title:", "id:", "logsource:", "detection:", "level:")


def main() -> int:
    rules = sorted(RULES.glob("*.yml"))
    if not rules:
        print("No Sigma rules found")
        return 1
    invalid = []
    for rule in rules:
        text = rule.read_text(encoding="utf-8")
        missing = [field for field in REQUIRED if field not in text]
        if missing:
            invalid.append(f"{rule.name}: {', '.join(missing)}")
    if invalid:
        print("Invalid Sigma rules:\n" + "\n".join(invalid))
        return 1
    print(f"validated {len(rules)} Sigma rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
