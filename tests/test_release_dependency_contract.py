"""Release-tool data extra, SCA install, and declared SBOM contract."""
from __future__ import annotations

import json
from pathlib import Path


def test_data_extra_uses_validated_exact_huggingface_version():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["optional-dependencies"]["data"] == ["huggingface_hub==0.36.2"]


def test_sca_installs_data_extra_and_keeps_offline_pytest():
    text = Path(".github/workflows/sca.yml").read_text(encoding="utf-8")
    assert 'python -m pip install -e ".[dev,security,data]"' in text
    assert 'HF_HUB_OFFLINE: "1"' in text
    assert 'TRANSFORMERS_OFFLINE: "1"' in text


def test_security_extra_includes_pdf_renderer_used_by_appsec_reports():
    import tomllib

    document = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    security = document["project"]["optional-dependencies"]["security"]
    assert "reportlab==5.0.1" in security


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
