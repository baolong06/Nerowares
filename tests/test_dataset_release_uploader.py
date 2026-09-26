"""Hub dataset adapter and resumable direct uploader (offline FakeHubClient only)."""
from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.data.release.models import (
    CANONICAL_BCICIV2A_REPO_ID,
    FamilyReleaseConfig,
    FamilyReleaseManifest,
)


def make_approved_config(**overrides: object) -> FamilyReleaseConfig:
    document: dict[str, object] = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "local_root": "family",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "private-team",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "governance": {
            "visibility": "private-team",
            "license_status": "verified",
            "dua_status": "not_applicable",
            "custodian_status": "verified",
            "deidentification_status": "reviewed",
            "approved_by": "custodian",
            "evidence_reference": "ticket-1",
        },
    }
    document.update(overrides)
    return FamilyReleaseConfig.from_dict(document)


def make_blocked_config(**overrides: object) -> FamilyReleaseConfig:
    document: dict[str, object] = {
        "dataset_id": "D01",
        "release_id": "thinking-d01-observed-v1",
        "local_root": "family",
        "repo_id": "madteam/thinking-d01",
        "provider": "huggingface",
        "scope_status": "observed-local-package",
        "visibility": "blocked",
        "provenance_references": ["https://openneuro.org/datasets/ds003626"],
        "governance": {
            "visibility": "blocked",
            "license_status": "unknown",
            "dua_status": "unknown",
            "custodian_status": "unknown",
            "deidentification_status": "not_reviewed",
            "approved_by": "",
            "evidence_reference": "",
        },
    }
    document.update(overrides)
    return FamilyReleaseConfig.from_dict(document)


def _populate_family(tmp_path: Path, payloads: dict[str, bytes] | None = None) -> None:
    family = tmp_path / "family"
    family.mkdir(parents=True, exist_ok=True)
    files = payloads if payloads is not None else {"sample.edf": b"payload-one"}
    for relative, data in files.items():
        target = family.joinpath(*relative.replace("\\", "/").split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    git_dir = family / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (family / "download.ps1").write_text("Write-Host 'blocked'\n", encoding="utf-8")


class FakeHubClient:
    """Test-only duck type of HubClient. No token. No network."""

    def __init__(self, calls: list[dict] | None = None) -> None:
        self.calls = calls if calls is not None else []
        self._blobs: dict[tuple[str, str], dict[str, bytes]] = {}

    def create_dataset_repo(self, repo_id: str, private: bool) -> None:
        self.calls.append(
            {
                "method": "create_dataset_repo",
                "repo_id": repo_id,
                "private": private,
                "repo_type": "dataset",
                "operations": [],
            }
        )

    def upload_batch(self, repo_id: str, operations, commit_message: str) -> str:
        from src.data.release.hub import UploadOperation

        revision = "a" * 40
        stored = dict(self._blobs.get((repo_id, revision), {}))
        recorded: list[UploadOperation] = []
        for operation in operations:
            stored[operation.path] = Path(operation.source_path).read_bytes()
            recorded.append(operation)
        self._blobs[(repo_id, revision)] = stored
        self.calls.append(
            {
                "method": "upload_batch",
                "repo_id": repo_id,
                "commit_message": commit_message,
                "repo_type": "dataset",
                "operations": recorded,
            }
        )
        return revision

    def list_revision(self, repo_id: str, revision: str):
        from src.data.release.hub import RemoteListing

        self.calls.append(
            {
                "method": "list_revision",
                "repo_id": repo_id,
                "revision": revision,
                "repo_type": "dataset",
                "operations": [],
            }
        )
        blobs = self._blobs.get((repo_id, revision), {})
        files = tuple(
            _ListingFile(path=path, size_bytes=len(payload))
            for path, payload in sorted(blobs.items())
        )
        return RemoteListing(revision=revision, files=files)

    def download_for_verify(self, repo_id: str, path: str, revision: str, destination: Path) -> Path:
        self.calls.append(
            {
                "method": "download_for_verify",
                "repo_id": repo_id,
                "path": path,
                "revision": revision,
                "repo_type": "dataset",
                "operations": [],
            }
        )
        payload = self._blobs[(repo_id, revision)][path]
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return dest

    def drop_remote_files(self, repo_id: str, revision: str) -> None:
        self._blobs.pop((repo_id, revision), None)


class _ListingFile:
    def __init__(self, path: str, size_bytes: int) -> None:
        self.path = path
        self.size_bytes = size_bytes


def make_uploader(client, tmp_path: Path, config: FamilyReleaseConfig | None = None, **kwargs):
    from src.data.release.uploader import ReleaseUploader

    _populate_family(tmp_path)
    bound = config if config is not None else make_approved_config()
    return ReleaseUploader(client, tmp_path, bound, **kwargs)


def approved_fixture_manifest(uploader) -> FamilyReleaseManifest:
    return uploader.plan(uploader.config, uploader.data_root)


def _upload_operations(calls: list[dict]):
    return [operation for call in calls for operation in call.get("operations") or ()]


def _methods(calls: list[dict], name: str) -> list[dict]:
    return [call for call in calls if call.get("method") == name]


def test_upload_uses_dataset_repo_type_and_does_not_upload_blocked_files(tmp_path):
    calls: list[dict] = []
    client = FakeHubClient(calls)
    uploader = make_uploader(client, tmp_path)
    manifest = approved_fixture_manifest(uploader)

    result = uploader.upload(manifest, state_path=tmp_path / "state.json")

    assert calls[0]["repo_type"] == "dataset"
    assert all(op.path not in {".git/config", "download.ps1"} for op in calls[0]["operations"])
    assert result.revision == "a" * 40
    assert all(call["repo_type"] == "dataset" for call in calls)
    uploaded = {op.path for op in _upload_operations(calls)}
    assert uploaded
    assert ".git/config" not in uploaded
    assert "download.ps1" not in uploaded
    assert uploaded <= {item.path for item in manifest.files}
    assert "sample.edf" in uploaded
    assert ".git/config" not in {item.path for item in manifest.files}
    assert "download.ps1" not in {item.path for item in manifest.files}


def test_changed_source_file_aborts_before_remote_write(tmp_path):
    calls: list[dict] = []
    client = FakeHubClient(calls)
    uploader = make_uploader(client, tmp_path)
    manifest = approved_fixture_manifest(uploader)
    (tmp_path / "family" / "sample.edf").write_bytes(b"mutated-payload-bytes")

    with pytest.raises(ValueError, match="source"):
        uploader.upload(manifest, state_path=tmp_path / "state.json")

    assert _methods(calls, "upload_batch") == []
    assert _methods(calls, "create_dataset_repo") == []
    assert calls == []


def test_resume_skips_verified_batches_without_duplicate_operations(tmp_path):
    _populate_family(
        tmp_path,
        {"sample.edf": b"payload-one", "extra.edf": b"payload-two"},
    )
    calls: list[dict] = []
    client = FakeHubClient(calls)
    from src.data.release.uploader import ReleaseUploader

    uploader = ReleaseUploader(
        client, tmp_path, make_approved_config(), batch_size=1, verification_mode="all"
    )
    manifest = uploader.plan(uploader.config, tmp_path)
    state_path = tmp_path / "state.json"

    first = uploader.upload(manifest, state_path)
    first_batches = _methods(calls, "upload_batch")
    assert len(first_batches) == 2
    first_paths = [tuple(op.path for op in call["operations"]) for call in first_batches]

    second = uploader.upload(manifest, state_path)
    second_batches = _methods(calls, "upload_batch")
    assert second_batches == first_batches
    assert [tuple(op.path for op in call["operations"]) for call in second_batches] == first_paths
    assert second.revision == first.revision == "a" * 40


def test_blocked_family_never_calls_hub(tmp_path):
    calls: list[dict] = []
    client = FakeHubClient(calls)
    _populate_family(tmp_path)
    from src.data.release.uploader import ReleaseUploader

    uploader = ReleaseUploader(client, tmp_path, make_blocked_config())
    with pytest.raises(ValueError):
        uploader.plan(uploader.config, tmp_path)
    assert calls == []

    approved = make_uploader(FakeHubClient([]), tmp_path)
    manifest = approved_fixture_manifest(approved)
    with pytest.raises(ValueError):
        uploader.upload(manifest, state_path=tmp_path / "state.json")
    assert calls == []


def test_canonical_bciciv2a_repo_is_rejected_case_and_whitespace(tmp_path):
    from src.data.release.hub import HubClient, UploadOperation
    from src.data.release.uploader import ReleaseUploader

    class _ForbiddenApi:
        def create_repo(self, *args, **kwargs):
            raise AssertionError("canonical repo must not reach the Hub API")

        def create_commit(self, *args, **kwargs):
            raise AssertionError("canonical repo must not reach the Hub API")

    client = HubClient(_ForbiddenApi())
    source = tmp_path / "sample.edf"
    source.write_bytes(b"x")
    operation = UploadOperation(path="sample.edf", source_path=source)
    for repo_id in (
        CANONICAL_BCICIV2A_REPO_ID,
        " MADTEAM/thinking-bciciv2a ",
        "madteam/THINKING-bciciv2a",
    ):
        with pytest.raises(ValueError, match="canonical"):
            client.create_dataset_repo(repo_id, private=True)
        with pytest.raises(ValueError, match="canonical"):
            client.upload_batch(repo_id, [operation], "upload")

    calls: list[dict] = []
    uploader = make_uploader(FakeHubClient(calls), tmp_path)
    planned = approved_fixture_manifest(uploader)
    colliding = replace(planned, repo_id=CANONICAL_BCICIV2A_REPO_ID)
    with pytest.raises(ValueError, match="canonical"):
        uploader.upload(colliding, state_path=tmp_path / "state.json")
    assert calls == []


def test_state_json_has_no_token_absolute_source_roots_or_file_bytes(tmp_path):
    calls: list[dict] = []
    uploader = make_uploader(FakeHubClient(calls), tmp_path)
    manifest = approved_fixture_manifest(uploader)
    state_path = tmp_path / "state.json"
    uploader.upload(manifest, state_path=state_path)

    text = state_path.read_text(encoding="utf-8")
    document = json.loads(text)
    serialized = json.dumps(document)
    assert "HF_TOKEN" not in text
    assert "hf_token" not in text
    assert "token" not in serialized.lower()
    assert "BEGIN PRIVATE" not in text
    assert "raw_payload" not in text
    assert "E:\\" not in text
    assert "C:\\" not in text
    assert "payload-one" not in text
    assert str(tmp_path) not in text
    assert "completed_batch_ids" in document
    assert all(not Path(item).is_absolute() for item in document["completed_batch_ids"])


def test_create_dataset_repo_private_flag_follows_visibility(tmp_path):
    private_calls: list[dict] = []
    private_uploader = make_uploader(FakeHubClient(private_calls), tmp_path)
    private_uploader.upload(
        approved_fixture_manifest(private_uploader), state_path=tmp_path / "private.json"
    )
    created = _methods(private_calls, "create_dataset_repo")
    assert created
    assert created[0]["private"] is True
    assert created[0]["repo_type"] == "dataset"

    public_root = tmp_path / "public"
    public_root.mkdir()
    public_calls: list[dict] = []
    public_config = make_approved_config(
        visibility="public",
        governance={
            "visibility": "public",
            "license_status": "verified",
            "dua_status": "not_applicable",
            "custodian_status": "verified",
            "deidentification_status": "reviewed",
            "approved_by": "custodian",
            "evidence_reference": "ticket-1",
        },
    )
    public_uploader = make_uploader(FakeHubClient(public_calls), public_root, public_config)
    public_uploader.upload(
        approved_fixture_manifest(public_uploader), state_path=public_root / "public.json"
    )
    public_created = _methods(public_calls, "create_dataset_repo")
    assert public_created
    assert public_created[0]["private"] is False


def test_verify_remote_rejects_revision_main(tmp_path):
    uploader = make_uploader(FakeHubClient([]), tmp_path)
    manifest = replace(approved_fixture_manifest(uploader), manifest_signature_status="verified")
    with pytest.raises(ValueError, match="revision"):
        uploader.verify_remote(manifest, "main")


def test_unsigned_upload_returns_revision_but_is_not_verified(tmp_path):
    uploader = make_uploader(FakeHubClient([]), tmp_path)
    manifest = approved_fixture_manifest(uploader)
    assert manifest.manifest_signature_status == "unsigned"
    result = uploader.upload(manifest, state_path=tmp_path / "state.json")
    assert result.revision == "a" * 40
    assert result.verification_status != "verified"
    with pytest.raises(ValueError, match="signature"):
        uploader.verify_remote(manifest, result.revision)


def test_verify_remote_binds_signed_manifest_and_immutable_revision(tmp_path):
    calls: list[dict] = []
    uploader = make_uploader(FakeHubClient(calls), tmp_path, verification_mode="all")
    manifest = replace(approved_fixture_manifest(uploader), manifest_signature_status="verified")
    result = uploader.upload(manifest, state_path=tmp_path / "state.json")
    verification = uploader.verify_remote(manifest, result.revision)
    assert verification.verification_status == "verified"
    assert verification.revision == "a" * 40
    downloads = _methods(calls, "download_for_verify")
    assert {call["path"] for call in downloads} >= {item.path for item in manifest.files}


def test_resume_reuploads_when_remote_listing_lacks_files(tmp_path):
    calls: list[dict] = []
    client = FakeHubClient(calls)
    uploader = make_uploader(client, tmp_path, batch_size=32)
    manifest = approved_fixture_manifest(uploader)
    state_path = tmp_path / "state.json"
    uploader.upload(manifest, state_path)
    first_count = len(_methods(calls, "upload_batch"))
    assert first_count >= 1
    client.drop_remote_files(manifest.repo_id, "a" * 40)
    uploader.upload(manifest, state_path)
    assert len(_methods(calls, "upload_batch")) == first_count * 2


def test_hub_client_passes_dataset_repo_type_to_injected_api(tmp_path):
    from src.data.release.hub import HubClient, UploadOperation

    recorded: list[dict] = []

    class RecordingApi:
        def create_repo(self, repo_id, **kwargs):
            recorded.append({"name": "create_repo", "repo_id": repo_id, **kwargs})

        def create_commit(self, repo_id, operations, **kwargs):
            recorded.append(
                {"name": "create_commit", "repo_id": repo_id, "operations": operations, **kwargs}
            )

            class _Info:
                oid = "b" * 40

            return _Info()

        def list_repo_tree(self, repo_id, **kwargs):
            recorded.append({"name": "list_repo_tree", "repo_id": repo_id, **kwargs})
            return []

        def hf_hub_download(self, repo_id, filename, **kwargs):
            recorded.append(
                {"name": "hf_hub_download", "repo_id": repo_id, "filename": filename, **kwargs}
            )
            dest = tmp_path / "downloaded.bin"
            dest.write_bytes(b"remote")
            return str(dest)

    client = HubClient(RecordingApi())
    client.create_dataset_repo("madteam/thinking-d01", private=True)
    source = tmp_path / "sample.edf"
    source.write_bytes(b"payload-one")
    revision = client.upload_batch(
        "madteam/thinking-d01",
        [UploadOperation(path="sample.edf", source_path=source)],
        "add files",
    )
    client.list_revision("madteam/thinking-d01", revision)
    client.download_for_verify(
        "madteam/thinking-d01", "sample.edf", revision, tmp_path / "verify.edf"
    )
    assert revision == "b" * 40
    assert recorded
    assert all(item.get("repo_type") == "dataset" for item in recorded)
    commit = next(item for item in recorded if item["name"] == "create_commit")
    added = commit["operations"][0]
    fileobj = getattr(added, "path_or_fileobj")
    assert Path(fileobj) == source or os_fspath_is_source(fileobj, source)
    assert "payload-one" not in str(added)


def os_fspath_is_source(fileobj, source: Path) -> bool:
    return Path(fileobj) == source


def _simulate_missing_data_extra(monkeypatch) -> None:
    import builtins
    import sys

    pkg = "huggingface" + "_hub"
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == pkg or name.startswith(pkg + "."):
            raise ImportError("simulated missing data extra")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    for key in list(sys.modules):
        if key == pkg or key.startswith(pkg + "."):
            monkeypatch.delitem(sys.modules, key, raising=False)


def test_hub_client_upload_batch_with_injected_api_does_not_require_data_extra(tmp_path, monkeypatch):
    from src.data.release.hub import HubClient, UploadOperation

    _simulate_missing_data_extra(monkeypatch)
    recorded: list[dict] = []

    class RecordingApi:
        def create_commit(self, repo_id, operations, **kwargs):
            recorded.append(
                {"name": "create_commit", "repo_id": repo_id, "operations": operations, **kwargs}
            )

            class _Info:
                oid = "b" * 40

            return _Info()

    client = HubClient(RecordingApi())
    source = tmp_path / "sample.edf"
    source.write_bytes(b"payload-one")
    revision = client.upload_batch(
        "madteam/thinking-d01",
        [UploadOperation(path="sample.edf", source_path=source)],
        "add files",
    )
    assert revision == "b" * 40
    assert recorded
    assert all(item.get("repo_type") == "dataset" for item in recorded)
    added = recorded[0]["operations"][0]
    assert added.path_in_repo == "sample.edf"
    assert Path(added.path_or_fileobj) == source
    assert "payload-one" not in str(added)


def test_hub_client_without_api_fails_closed_when_data_extra_missing(tmp_path, monkeypatch):
    from src.data.release.hub import HubClient, UploadOperation

    _simulate_missing_data_extra(monkeypatch)
    source = tmp_path / "sample.edf"
    source.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="optional"):
        HubClient().upload_batch(
            "madteam/thinking-d01",
            [UploadOperation(path="sample.edf", source_path=source)],
            "add files",
        )


def test_hub_client_sanitizes_injected_api_errors(tmp_path):
    from src.data.release.hub import HubClient

    class BoomApi:
        def create_repo(self, *args, **kwargs):
            raise RuntimeError("denied HF_TOKEN=hf_supersecret raw_payload=EEG")

    with pytest.raises(RuntimeError) as caught:
        HubClient(BoomApi()).create_dataset_repo("madteam/thinking-d01", private=True)
    message = str(caught.value)
    assert "hf_supersecret" not in message
    assert "HF_TOKEN" not in message
    assert "raw_payload" not in message


def test_release_uploader_constructor_has_no_token_field():
    from src.data.release.uploader import ReleaseUploader

    params = inspect.signature(ReleaseUploader.__init__).parameters
    assert "token" not in params
    assert "hf_token" not in params
    assert "HF_TOKEN" not in params


def test_test_module_does_not_use_network_or_live_hub():
    source = Path(__file__).read_text(encoding="utf-8")
    pkg = "huggingface" + "_hub"
    host = "huggingface" + ".co"
    live = "Hf" + "Api"
    assert f"from {pkg}" not in source
    assert f"import {pkg}" not in source
    assert host not in source
    assert f"{live}(" not in source
    production = (
        Path("src/data/release/hub.py").read_text(encoding="utf-8")
        + Path("src/data/release/uploader.py").read_text(encoding="utf-8")
    )
    assert host not in production
    assert "HF_" + "TOKEN" not in production
    assert "BEGIN PRIVATE " + "KEY" not in production
