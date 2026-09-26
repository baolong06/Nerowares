"""Release-tool data extra, SCA install, and declared SBOM contract."""
from __future__ import annotations

import json
from pathlib import Path


def test_data_extra_uses_validated_exact_huggingface_version():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["optional-dependencies"]["data"] == ["huggingface_hub==0.36.2"]


def test_dev_extra_uses_pytest_stack_versions_without_declared_vulnerabilities():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dev = document["project"]["optional-dependencies"]["dev"]
    assert "pytest==9.0.3" in dev
    assert "pytest-asyncio==1.4.0" in dev


def test_sca_installs_declared_dependencies_without_auditing_local_package():
    text = Path(".github/workflows/sca.yml").read_text(encoding="utf-8")
    assert 'for extra in ("dev", "security", "data"):' in text
    assert "python -m pip install -r artifacts/ci-requirements.txt" in text
    assert "python -m pip_audit -r artifacts/ci-requirements.txt --progress-spinner off --strict" in text
    assert 'python -m pip install ".[dev,security,data]"' not in text
    assert 'python -m pip install -e' not in text
    assert 'HF_HUB_OFFLINE: "1"' in text
    assert 'TRANSFORMERS_OFFLINE: "1"' in text
    assert 'python scripts/generate_sbom.py --mode declared --output artifacts/sbom-declared.cdx.json' in text
    assert text.index("Generate declared SBOM") < text.index("Pytest (full regression, no raw dataset download)")


def test_security_extra_includes_pdf_renderer_used_by_appsec_reports():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    security = document["project"]["optional-dependencies"]["security"]
    assert "reportlab==5.0.1" in security


def test_security_extra_includes_release_signing_crypto():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    security = document["project"]["optional-dependencies"]["security"]
    assert "cryptography==50.0.0" in security


def test_dev_extra_includes_semantic_retrieval_dependency_used_by_tests():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dev = document["project"]["optional-dependencies"]["dev"]
    assert "scikit-learn==1.5.2" in dev


def test_declared_sbom_includes_exact_huggingface_hub(tmp_path):
    from scripts import generate_sbom

    output = tmp_path / "sbom-declared.cdx.json"
    assert generate_sbom.main(["--mode", "declared", "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    components = {item["name"]: item["version"] for item in document["components"]}
    assert components.get("huggingface_hub") == "0.36.2"
    names = {name.lower() for name in components}
    assert "torch" not in names
    assert "pytest" not in names
    assert "mne" not in names
