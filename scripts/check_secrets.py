"""Run Gitleaks when available, otherwise a conservative redacted regex scan."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".venv-st", "__pycache__", ".pytest_cache", "artifacts"}
ALLOWLIST_SUBSTRINGS = ("runtime-secret-without-shared-store", "dev-only-change-me", "THINKING_ALLOW_DEV_SECRET")
# High-signal forms; values are never printed.
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "anthropic-key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
    "github-token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "generic-assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9/+_=-]{20,}['\"]"
    ),
}


def _fallback_scan() -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED for part in path.parts):
            continue
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".parquet", ".edf"}:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if any(value in line for value in ALLOWLIST_SUBSTRINGS):
                continue
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    hits.append((path.relative_to(ROOT), number, kind))
    return hits


def main() -> int:
    executable = shutil.which("gitleaks")
    if executable:
        command = [
            executable,
            "detect",
            "--source", str(ROOT),
            "--config", str(ROOT / ".gitleaks.toml"),
            "--redact",
            "--no-banner",
            "--exit-code", "1",
        ]
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    hits = _fallback_scan()
    if hits:
        print(f"fallback secret scan found {len(hits)} candidate(s); values redacted")
        for path, line, kind in hits:
            print(f"{path}:{line}: {kind}")
        return 1
    print("Gitleaks CLI unavailable; conservative fallback scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
