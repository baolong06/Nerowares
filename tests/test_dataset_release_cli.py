"""Plan/upload CLI for family release. Offline FakeHubClient only; no live Hub."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_dataset_release_signing import make_ed25519_pem_pair
from tests.test_dataset_release_uploader import FakeHubClient

PAYLOAD = b"payload-one"
_PRIVATE_KEY_ENV = "THINKING_RELEASE_PRIVATE_KEY"
_CANONICAL_REPO = "madteam/thinking-bciciv2a"
_HOST = "huggingface" + ".co"
_PKG = "huggingface" + "_hub"
_LIVE = "Hf" + "Api"
_TOKEN = "HF_" + "TOKEN"
_PEM = "BEGIN PRIVATE " + "KEY"


def fail_if_constructed(*args, **kwargs):
    raise AssertionError("HubClient must not be constructed")


def _catalog_document() -> dict[str, object]:
    return json.loads(Path("dataset_catalog/families.release.json").read_text(encoding="utf-8"))


def _approved_governance() -> dict[str, object]:
    return {
        "visibility": "private-team",
        "license_status": "verified",
        "dua_status": "not_applicable",
        "custodian_status": "verified",
        "deidentification_status": "reviewed",
        "approved_by": "custodian",
        "evidence_reference": "ticket-1",
    }


def write_release_config(
    tmp_path: Path,
    *,
    approved_id: str | None = "D01",
    local_root: str = "family",
    repo_id: str | None = None,
) -> Path:
    document = _catalog_document()
    if approved_id is not None:
        family = dict(document[approved_id])
        family["local_root"] = local_root
        family["visibility"] = "private-team"
        family["governance"] = _approved_governance()
        if repo_id is not None:
            family["repo_id"] = repo_id
        document[approved_id] = family
    path = tmp_path / "families.release.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def populate_family(data_root: Path, local_root: str = "family") -> None:
    root = data_root / local_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "sample.edf").write_bytes(PAYLOAD)


def cli_argv(
    tmp_path: Path,
    config: Path,
    *extra: str,
    family: str = "D01",
) -> list[str]:
    return [
        *extra,
        "--family",
        family,
        "--config",
        str(config),
        "--data-root",
        str(tmp_path),
        "--manifest-dir",
        str(tmp_path / "manifests"),
        "--state-dir",
        str(tmp_path / "state"),
    ]


def install_keys(monkeypatch, tmp_path: Path, *, env_name: str = _PRIVATE_KEY_ENV):
    private_pem, public_pem = make_ed25519_pem_pair()
    monkeypatch.setenv(env_name, private_pem.decode("ascii"))
    public_path = tmp_path / "verification.pub"
    public_path.write_bytes(public_pem)
    return env_name, public_path, private_pem, public_pem


def patch_fail_hub(monkeypatch) -> None:
    monkeypatch.setattr("src.data.release.hub.HubClient", fail_if_constructed)


def patch_fake_hub(monkeypatch, calls: list[dict] | None = None) -> list[dict]:
    bucket = calls if calls is not None else []

    class PatchedHubClient(FakeHubClient):
        def __init__(self, api=None, *args, **kwargs):
            super().__init__(bucket)

    monkeypatch.setattr("src.data.release.hub.HubClient", PatchedHubClient)
    return bucket


def load_cli():
    from scripts.dataset_release import build_parser, main

    return build_parser, main


def stdout_json(capsys) -> dict[str, object]:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def assert_safe_text(text: str, data_root: Path) -> None:
    assert _TOKEN not in text
    assert "hf_supersecret" not in text
    assert _PEM not in text
    assert "payload-one" not in text
    assert str(data_root) not in text
    assert "E:\\" not in text
    assert "C:\\" not in text
    lowered = text.lower()
    assert "hf_token" not in lowered


def test_plan_mode_never_constructs_a_hub_client(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(
        [
            "--plan",
            "--family",
            "D01",
            "--config",
            str(config),
            "--data-root",
            str(tmp_path),
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "plan"
    assert payload.get("dataset_id") == "D01" or payload.get("family") == "D01"
    assert payload["file_count"] == 1
    assert payload["total_size_bytes"] == len(PAYLOAD)
    assert payload["scope_status"] == "observed-local-package"
    assert payload["visibility"] == "private-team" or payload.get("governance_status")
    assert isinstance(payload["manifest_sha256"], str)
    assert len(payload["manifest_sha256"]) == 64
    written = list((tmp_path / "manifests").glob("*.json"))
    assert written
    assert list((tmp_path / "manifests").rglob("*.edf")) == []


def test_cli_output_does_not_contain_token_or_absolute_source_root(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    monkeypatch.setenv(_TOKEN, "hf_supersecret_should_never_appear")
    patch_fail_hub(monkeypatch)

    code = main(cli_argv(tmp_path, config, "--plan"))
    assert code == 0
    plan_out = capsys.readouterr().out
    assert_safe_text(plan_out, tmp_path)
    plan = json.loads(plan_out)
    assert plan["mode"] == "plan"

    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    calls = patch_fake_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code == 0
    upload_out = capsys.readouterr().out
    assert_safe_text(upload_out, tmp_path)
    upload = json.loads(upload_out)
    assert upload["mode"] == "upload"
    assert upload["revision"] == "a" * 40
    assert "verification_status" in upload
    assert upload["manifest_sha256"] == plan["manifest_sha256"]
    uploaded = [op.path for call in calls for op in call.get("operations") or ()]
    assert uploaded
    assert set(uploaded) <= {"sample.edf"}
    assert "README.md" not in uploaded


def test_plan_with_allow_network_does_not_construct_hub_client(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(cli_argv(tmp_path, config, "--plan", "--allow-network"))
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_blocked_family_plan_never_constructs_hub_client(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path, approved_id=None)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(cli_argv(tmp_path, config, "--plan"))
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_blocked_family_upload_never_constructs_hub_client(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path, approved_id=None)
    populate_family(tmp_path)
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_without_allow_network_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_missing_plan_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_missing_private_key_env_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    _, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    monkeypatch.delenv(_PRIVATE_KEY_ENV, raising=False)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_empty_private_key_env_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    _, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    monkeypatch.setenv(_PRIVATE_KEY_ENV, "")
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            _PRIVATE_KEY_ENV,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_malformed_private_key_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    _, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    monkeypatch.setenv(_PRIVATE_KEY_ENV, "not-a-pem")
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            _PRIVATE_KEY_ENV,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_missing_verification_key_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    env_name, _, _, _ = install_keys(monkeypatch, tmp_path)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(tmp_path / "missing.pub"),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_untrusted_verification_key_fails_closed(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    env_name, _, _, _ = install_keys(monkeypatch, tmp_path)
    _, other_public = make_ed25519_pem_pair()
    other_path = tmp_path / "other.pub"
    other_path.write_bytes(other_public)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(other_path),
        )
    )
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_changed_source_aborts_before_hub_write(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    capsys.readouterr()
    (tmp_path / "family" / "sample.edf").write_bytes(b"mutated-payload-bytes")
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    calls = patch_fake_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code != 0
    assert calls == []
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_canonical_d15_repo_in_config_is_rejected(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(
        tmp_path, approved_id="D15", local_root="family", repo_id=_CANONICAL_REPO
    )
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    code = main(cli_argv(tmp_path, config, "--plan", family="D15"))
    assert code != 0
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_canonical_repo_swapped_into_saved_plan_is_rejected(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path, approved_id="D15", local_root="family")
    populate_family(tmp_path)
    patch_fail_hub(monkeypatch)
    assert main(cli_argv(tmp_path, config, "--plan", family="D15")) == 0
    capsys.readouterr()
    plans = list((tmp_path / "manifests").glob("*.json"))
    assert plans
    for plan_path in plans:
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        document["repo_id"] = _CANONICAL_REPO
        plan_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    calls = patch_fake_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
            family="D15",
        )
    )
    assert code != 0
    assert calls == []
    assert_safe_text(capsys.readouterr().out, tmp_path)


def test_upload_with_fake_hub_client_succeeds_for_approved_fixture(monkeypatch, tmp_path, capsys):
    _, main = load_cli()
    config = write_release_config(tmp_path)
    populate_family(tmp_path)
    assert main(cli_argv(tmp_path, config, "--plan")) == 0
    plan = stdout_json(capsys)
    env_name, public_path, _, _ = install_keys(monkeypatch, tmp_path)
    calls = patch_fake_hub(monkeypatch)
    code = main(
        cli_argv(
            tmp_path,
            config,
            "--upload",
            "--allow-network",
            "--private-key-env",
            env_name,
            "--verification-key",
            str(public_path),
        )
    )
    assert code == 0
    payload = stdout_json(capsys)
    assert payload["mode"] == "upload"
    assert payload["revision"] == "a" * 40
    assert len(str(payload["revision"])) == 40
    assert "verification_status" in payload
    assert payload["manifest_sha256"] == plan["manifest_sha256"]
    uploaded = {op.path for call in calls for op in call.get("operations") or ()}
    assert uploaded == {"sample.edf"}
    assert all(call.get("repo_type") == "dataset" for call in calls if "repo_type" in call)


def test_cli_module_does_not_use_network_or_live_hub():
    source = Path(__file__).read_text(encoding="utf-8")
    assert f"from {_PKG}" not in source
    assert f"import {_PKG}" not in source
    assert _HOST not in source
    assert f"{_LIVE}(" not in source
    cli_path = Path("scripts/dataset_release.py")
    if cli_path.is_file():
        production = cli_path.read_text(encoding="utf-8")
        assert f"from {_PKG}" not in production
        assert f"import {_PKG}" not in production
        assert _HOST not in production
        assert f"{_LIVE}(" not in production
        assert _TOKEN not in production
        assert _PEM not in production
        assert "from src.data.release.hub import HubClient" not in production


def test_build_parser_requires_mode_and_core_options():
    build_parser, _ = load_cli()
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plan"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plan", "--upload"])
