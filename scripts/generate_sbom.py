"""Create a CycloneDX JSON SBOM from `pip freeze`, with cyclonedx-py as fallback."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "thinking-eeg"
APP_VERSION = "0.2.0"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)


def _pip_freeze() -> str:
    result = _run([sys.executable, "-m", "pip", "freeze"])
    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _parse_freeze(text: str) -> list[tuple[str, str]]:
    packages: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-e ") or " @ " in line:
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        name = name.split("[", 1)[0].strip()
        version = version.strip()
        if name and version:
            packages.append((name, version))
    packages.sort(key=lambda item: item[0].lower())
    return packages


def _write_from_freeze(packages: list[tuple[str, str]], output: Path) -> None:
    components = []
    for name, version in packages:
        purl = f"pkg:pypi/{name.lower()}@{version}"
        components.append(
            {
                "type": "library",
                "name": name,
                "version": version,
                "purl": purl,
                "bom-ref": purl,
            }
        )
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": [{"vendor": "thinking-eeg", "name": "generate_sbom.py", "version": APP_VERSION}],
            "component": {"type": "application", "name": APP_NAME, "version": APP_VERSION},
        },
        "components": components,
    }
    output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")


def _cyclonedx_py() -> str | None:
    found = shutil.which("cyclonedx-py")
    if found:
        return found
    result = _run([sys.executable, "-m", "cyclonedx_py", "--help"])
    if result.returncode == 0:
        return sys.executable
    return None


def _run_cyclonedx(cli: str, args: list[str], output: Path) -> bool:
    command = [cli, *args] if cli != sys.executable else [sys.executable, "-m", "cyclonedx_py", *args]
    result = subprocess.run(command, cwd=ROOT, check=False)
    return result.returncode == 0 and output.exists() and output.stat().st_size > 0


def _packages_from_requirements(requirements: list[str]) -> list[tuple[str, str]]:
    packages: list[tuple[str, str]] = []
    for raw in requirements:
        cleaned = str(raw).strip().strip(",").strip('"').strip("'")
        if not cleaned or cleaned.startswith("#"):
            continue
        name = cleaned.split("==")[0].split(">=")[0].split("[")[0].strip()
        version = cleaned.split("==")[1].strip().split(",")[0] if "==" in cleaned else "declared"
        if name:
            packages.append((name, version))
    packages.sort(key=lambda item: item[0].lower())
    return packages


def _array_entries_after(text: str, header_prefix: str) -> list[str]:
    entries: list[str] = []
    in_array = False
    for raw in text.splitlines():
        line = raw.strip()
        if not in_array:
            if line.startswith(header_prefix):
                in_array = True
            continue
        if line.startswith("]"):
            break
        entries.append(line)
    return entries


def _declared_packages() -> list[tuple[str, str]]:
    try:
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        import tomllib

        document = tomllib.loads(text)
        project = document.get("project", {})
        requirements = list(project.get("dependencies") or [])
        extras = project.get("optional-dependencies") or {}
        requirements.extend(extras.get("data") or [])
        return _packages_from_requirements([str(item) for item in requirements])
    except ImportError:
        requirements = _array_entries_after(text, "dependencies")
        requirements.extend(_array_entries_after(text, "data"))
        return _packages_from_requirements(requirements)


def _write_with_scope(packages: list[tuple[str, str]], output: Path, scope: str) -> None:
    components = []
    for name, version in packages:
        purl = f"pkg:pypi/{name.lower()}@{version}"
        components.append({"type": "library", "name": name, "version": version, "purl": purl, "bom-ref": purl})
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tools": [{"vendor": "thinking-eeg", "name": "generate_sbom.py", "version": APP_VERSION}],
            "component": {"type": "application", "name": APP_NAME, "version": APP_VERSION},
            "properties": {"dependency_scope": scope},
        },
        "components": components,
    }
    output.write_text(json.dumps(bom, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a CycloneDX SBOM from pip freeze")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "sbom.cdx.json")
    parser.add_argument("--mode", choices=["declared", "ambient", "auto"], default="auto")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else (ROOT / args.output)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "declared":
        packages = _declared_packages()
        if not packages:
            print("failed to generate declared SBOM: pyproject.toml unavailable", file=sys.stderr)
            return 1
        _write_with_scope(packages, output, "declared")
        print(f"wrote {output} declared SBOM ({len(packages)} components)")
        return 0
    if args.mode == "ambient":
        freeze_text = _pip_freeze()
        packages = _parse_freeze(freeze_text)
        if not packages:
            print("failed to generate ambient SBOM: pip freeze empty", file=sys.stderr)
            return 1
        _write_with_scope(packages, output, "ambient")
        print(f"wrote {output} ambient SBOM ({len(packages)} components)")
        return 0

    freeze_text = _pip_freeze()
    freeze_path = output.parent / "pip-freeze.requirements.txt"
    freeze_path.write_text(freeze_text, encoding="utf-8")

    cli = _cyclonedx_py()
    if cli and freeze_text.strip():
        if _run_cyclonedx(
            cli,
            ["requirements", str(freeze_path), "--output-format", "JSON", "--output-file", str(output)],
            output,
        ):
            print(f"wrote {output} via cyclonedx-py requirements (pip freeze)")
            return 0

    packages = _parse_freeze(freeze_text)
    if packages:
        _write_from_freeze(packages, output)
        print(f"wrote {output} from pip freeze ({len(packages)} components)")
        return 0

    if cli and _run_cyclonedx(
        cli,
        ["environment", "--output-format", "JSON", "--output-file", str(output)],
        output,
    ):
        print(f"wrote {output} via cyclonedx-py environment")
        return 0

    print("failed to generate SBOM: pip freeze empty and cyclonedx-py unavailable", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
