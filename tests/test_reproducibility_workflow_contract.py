"""Manual family reproducibility must fail closed and keep CI offline."""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from tests.test_dataset_release_integration import _governance, _sign_family_json, _write_family_json


_WORKFLOW = Path(".github/workflows/reproducibility.yml")


def test_reproducibility_workflow_is_manual_and_does_not_run_on_push():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "pull_request:" not in text
    assert "HF_TOKEN" not in text
    assert 'TRANSFORMERS_OFFLINE: "1"' not in text
    assert "--upload" not in text


def test_reproducibility_workflow_requires_protected_secrets_and_family_inputs():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "environment: dataset-reproducibility" in text
    assert "runs-on: [self-hosted, linux, dataset-reproducibility]" in text
    assert "timeout-minutes: 120" in text
    assert "group: reproducibility-${{ github.repository }}\n" in text
    assert "cancel-in-progress: false" in text
    assert "family:" in text
    assert "release_manifest:" in text
    assert "THINKING_HF_DATASET_TOKEN: ${{ secrets.THINKING_HF_DATASET_TOKEN }}" in text
    assert "THINKING_RELEASE_VERIFY_KEY: ${{ secrets.THINKING_RELEASE_VERIFY_KEY }}" in text
    assert "token=token" in text
    before_verify = text.split("      - name: Verify approved signed family and pinned shards", 1)[0]
    assert "secrets.THINKING_HF_DATASET_TOKEN" not in before_verify
    assert "secrets.THINKING_RELEASE_VERIFY_KEY" not in before_verify


def test_reproducibility_workflow_verifies_signed_release_and_quota_before_download():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "verify_release_manifest(" in text
    assert "manifest_path=release_path" in text
    assert "signature_path=signature_path" in text
    assert "governance.can_upload" in text
    assert 'verified.manifest.provider != "huggingface"' in text
    assert "load_release_config(" in text
    assert "approved.governance != verified.manifest.governance" in text
    assert "approved.release_id != verified.release_id" in text
    assert "approved.scope_status != verified.scope_status" in text
    assert "DatasetManifest" not in text  # scientific schema cannot be inferred from inventory
    assert "preflight_download_quota(" in text
    assert "max_cache_bytes = 80_000_000_000" in text
    assert "min_free_bytes = 10_000_000_000" in text
    assert "for record in verified.manifest.files:" in text
    assert "hf_hub_download(" in text
    assert 'repo_type="dataset"' in text
    assert "size, digest = hash_file(downloaded)" in text
    assert "(size, digest) != (record.size_bytes, record.sha256)" in text
    assert text.index("verify_release_manifest(") < text.index("preflight_download_quota(")
    assert text.index("preflight_download_quota(") < text.index("hf_hub_download(")


def test_reproducibility_workflow_publishes_only_sanitized_lineage():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "actions/upload-artifact@v4" in text
    assert "lineage.json" in text
    assert "release_id" in text
    assert "manifest_sha256" in text
    assert "manifest_signature_status" in text
    assert "no raw EEG" in text
    assert "if-no-files-found: error" in text
    assert "artifacts/release-manifests/" not in text.split("uses: actions/upload-artifact@v4", 1)[1]
    assert "cache/" not in text.split("uses: actions/upload-artifact@v4", 1)[1]


def test_reproducibility_workflow_embedded_python_compiles():
    text = _WORKFLOW.read_text(encoding="utf-8")
    program = text.split("          python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    compile(textwrap.dedent(program), str(_WORKFLOW), "exec")


def test_manual_workflow_fails_closed_for_current_blocked_family_without_downloading(tmp_path, monkeypatch):
    from tests.test_dataset_release_signing import make_ed25519_pem_pair

    from src.data.release.signing import write_detached_signature

    # Use tmp_path as repository root so no file is written into the source tree.
    sandbox = tmp_path / "sandbox"
    (sandbox / "dataset_catalog").mkdir(parents=True)
    target = sandbox / "dataset_catalog" / "D01.json"
    document = _write_family_json(target)
    document["governance"] = _governance(visibility="blocked")
    target.write_text(json.dumps(document), encoding="utf-8")
    private_key, public_key = make_ed25519_pem_pair()
    write_detached_signature(target, private_key)
    (sandbox / "dataset_catalog" / "families.release.json").write_bytes(
        Path("dataset_catalog/families.release.json").read_bytes()
    )
    text = _WORKFLOW.read_text(encoding="utf-8")
    program = text.split("          python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("REQUESTED_FAMILY", "D01")
    monkeypatch.setenv("RELEASE_MANIFEST_PATH", "dataset_catalog/D01.json")
    monkeypatch.setenv("THINKING_RELEASE_VERIFY_KEY", public_key.decode("utf-8"))
    monkeypatch.setenv("THINKING_HF_DATASET_TOKEN", "dummy-token")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    fake_hub = ModuleType("huggingface_hub")

    def forbidden_download(**kwargs):
        raise AssertionError("download must not run for blocked governance")

    fake_hub.hf_hub_download = forbidden_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with pytest.raises(SystemExit) as stopped:
        exec(compile(textwrap.dedent(program), str(_WORKFLOW), "exec"), {"__name__": "__main__"})
    assert stopped.value.code == 1
    assert not (sandbox / "artifacts/reproducibility/lineage.json").exists()


def test_manual_workflow_rejects_manifest_outside_repository_before_hub_call(tmp_path, monkeypatch):
    text = _WORKFLOW.read_text(encoding="utf-8")
    program = text.split("          python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("REQUESTED_FAMILY", "D01")
    monkeypatch.setenv("RELEASE_MANIFEST_PATH", "../untrusted.json")
    monkeypatch.setenv("THINKING_RELEASE_VERIFY_KEY", "not-a-key")
    monkeypatch.setenv("THINKING_HF_DATASET_TOKEN", "dummy-token")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    fake_hub = ModuleType("huggingface_hub")

    def forbidden_download(**kwargs):
        raise AssertionError("download must not run for untrusted path")

    fake_hub.hf_hub_download = forbidden_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    with pytest.raises(SystemExit) as stopped:
        exec(compile(textwrap.dedent(program), str(_WORKFLOW), "exec"), {"__name__": "__main__"})
    assert stopped.value.code == 1
    assert "THINKING_HF_DATASET_TOKEN" not in os.environ
    assert "THINKING_RELEASE_VERIFY_KEY" not in os.environ
    assert not (sandbox / "artifacts/reproducibility/lineage.json").exists()


def test_manual_workflow_verifies_approved_signed_inventory_and_writes_only_lineage(
    tmp_path, monkeypatch
):
    from src.data import huggingface

    sandbox = tmp_path / "sandbox"
    catalog = sandbox / "dataset_catalog"
    catalog.mkdir(parents=True)
    release = catalog / "D01.json"
    _write_family_json(release)
    _, public_key = _sign_family_json(release)
    config = json.loads(Path("dataset_catalog/families.release.json").read_text(encoding="utf-8"))
    config["D01"]["visibility"] = "private-team"
    config["D01"]["governance"] = _governance()
    (catalog / "families.release.json").write_text(json.dumps(config), encoding="utf-8")
    payload = tmp_path / "sample.edf"
    payload.write_bytes(b"sample-edf")
    events = []
    real_preflight = huggingface.preflight_download_quota

    def checked_preflight(**kwargs):
        events.append("preflight")
        return real_preflight(**kwargs, disk_usage=lambda unused: type("Usage", (), {"free": 80_000_000_000})())

    def fake_download(**kwargs):
        assert events == ["preflight"]
        assert kwargs["repo_id"] == "madteam/thinking-d01"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["filename"] == "sample.edf"
        assert kwargs["revision"] == "a" * 40  # synthetic test-only revision
        assert kwargs["token"] == "test-credential"
        events.append("download")
        return str(payload)

    monkeypatch.setattr(huggingface, "preflight_download_quota", checked_preflight)
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("REQUESTED_FAMILY", "D01")
    monkeypatch.setenv("RELEASE_MANIFEST_PATH", "dataset_catalog/D01.json")
    monkeypatch.setenv("THINKING_RELEASE_VERIFY_KEY", public_key.decode("utf-8"))
    monkeypatch.setenv("THINKING_HF_DATASET_TOKEN", "test-credential")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    text = _WORKFLOW.read_text(encoding="utf-8") if _WORKFLOW.exists() else Path(__file__).resolve().parents[1].joinpath(_WORKFLOW).read_text(encoding="utf-8")
    program = text.split("          python - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    exec(compile(textwrap.dedent(program), str(_WORKFLOW), "exec"), {"__name__": "__main__"})
    assert events == ["preflight", "download"]
    assert "THINKING_HF_DATASET_TOKEN" not in os.environ
    lineage = json.loads((sandbox / "artifacts/reproducibility/lineage.json").read_text(encoding="utf-8"))
    assert lineage["dataset_id"] == "D01"
    assert lineage["manifest_signature_status"] == "verified"
    assert lineage["verification_status"] == "verified"
    assert lineage["file_count"] == 1
    assert lineage["total_size_bytes"] == len(b"sample-edf")
    assert "test-credential" not in json.dumps(lineage)


def test_sca_remains_offline_and_compiles_release_modules():
    text = Path(".github/workflows/sca.yml").read_text(encoding="utf-8")
    assert 'HF_HUB_OFFLINE: "1"' in text
    assert 'TRANSFORMERS_OFFLINE: "1"' in text
    assert "src/data/release/*.py" in text
    assert "scripts/dataset_release.py" in text
    sbom_paths = text.split("      - name: Upload SBOMs", 1)[1].split("      - name: pip-audit", 1)[0]
    assert "artifacts/sbom-declared.cdx.json" in sbom_paths
    assert "artifacts/sbom-ambient.cdx.json" in sbom_paths
    assert "artifacts/sbom.cdx.json" not in sbom_paths
