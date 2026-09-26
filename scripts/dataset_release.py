"""Plan or upload one approved dataset family. JSON-only safe stdout."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from src.data.release import hub as release_hub
from src.data.release.governance import load_release_config, validate_governance
from src.data.release.inventory import revalidate_source
from src.data.release.models import FamilyReleaseConfig, FamilyReleaseManifest, ReleaseConfig
from src.data.release.signing import verify_detached_signature, write_detached_signature
from src.data.release.uploader import ReleaseUploader


class _OfflineClient:
    """Non-network stand-in so --plan can call ReleaseUploader.plan without HubClient."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or upload one approved dataset family release"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--upload", action="store_true")
    parser.add_argument("--family", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--private-key-env", default=None)
    parser.add_argument("--verification-key", type=Path, default=None)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Required for --upload; rejected with --plan",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, TypeError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        mode = "plan" if args.plan else "upload"
        return _fail(_safe_message(exc), mode=mode)


def _run(args: argparse.Namespace) -> int:
    if args.plan and args.allow_network:
        return _fail("allow-network is not valid with --plan", mode="plan")
    if args.upload and not args.allow_network:
        return _fail("allow-network is required for upload", mode="upload")

    config = load_release_config(Path(args.config))
    family = _select_family(config, args.family)
    data_root = Path(args.data_root)
    if args.plan:
        return _plan(family, data_root, Path(args.manifest_dir))
    return _upload(args, family, data_root)


def _plan(family: FamilyReleaseConfig, data_root: Path, manifest_dir: Path) -> int:
    _require_uploadable(family)
    uploader = ReleaseUploader(_OfflineClient(), data_root, family)
    manifest = uploader.plan(family, data_root)
    plan_path = _plan_path(manifest_dir, family.dataset_id)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _emit(
        {
            "mode": "plan",
            "dataset_id": family.dataset_id,
            "family": family.dataset_id,
            "release_id": manifest.release_id,
            "repo_id": manifest.repo_id,
            "file_count": manifest.file_count,
            "total_size_bytes": manifest.total_size_bytes,
            "scope_status": manifest.scope_status,
            "visibility": family.governance.visibility,
            "governance_status": "approved",
            "manifest_sha256": manifest.sha256(),
        }
    )
    return 0


def _upload(args: argparse.Namespace, family: FamilyReleaseConfig, data_root: Path) -> int:
    _require_uploadable(family)
    private_pem, public_pem = _load_keys(args)
    plan_path = _plan_path(Path(args.manifest_dir), family.dataset_id)
    manifest = _load_plan(plan_path)
    if manifest.dataset_id != family.dataset_id:
        raise ValueError("plan does not match family")
    release_hub.reject_canonical_repo(manifest.repo_id)
    release_hub.reject_canonical_repo(family.repo_id)
    if manifest.repo_id.strip().casefold() != family.repo_id.strip().casefold():
        release_hub.reject_canonical_repo(manifest.repo_id)
        raise ValueError("plan does not match family")
    mismatches = revalidate_source(manifest, family, data_root)
    if mismatches:
        raise ValueError("source file changed since the planned manifest")
    try:
        sidecar = write_detached_signature(plan_path, private_pem)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise ValueError("private key is malformed") from exc
    if not verify_detached_signature(plan_path, sidecar, public_pem):
        raise ValueError("verification key is untrusted")
    signed = replace(manifest, manifest_signature_status="verified")
    state_path = Path(args.state_dir) / f"{family.dataset_id}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    client = release_hub.HubClient()
    uploader = ReleaseUploader(client, data_root, family)
    result = uploader.upload(signed, state_path)
    _emit(
        {
            "mode": "upload",
            "dataset_id": family.dataset_id,
            "family": family.dataset_id,
            "revision": result.revision,
            "verification_status": result.verification_status,
            "manifest_sha256": signed.sha256(),
            "repo_id": result.repo_id,
        }
    )
    return 0


def _select_family(config: ReleaseConfig, dataset_id: str) -> FamilyReleaseConfig:
    for family in config.families:
        if family.dataset_id == dataset_id:
            return family
    raise ValueError("unknown family")


def _require_uploadable(family: FamilyReleaseConfig) -> None:
    if not family.governance.can_upload:
        raise ValueError("family is not approved for upload")
    validate_governance(family)


def _plan_path(manifest_dir: Path, dataset_id: str) -> Path:
    return Path(manifest_dir) / f"{dataset_id}.json"


def _load_plan(path: Path) -> FamilyReleaseManifest:
    if not path.is_file():
        raise ValueError("validated plan not found")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("validated plan not found") from exc
    return FamilyReleaseManifest.from_dict(document)


def _load_keys(args: argparse.Namespace) -> tuple[bytes, bytes]:
    if not args.private_key_env:
        raise ValueError("private key is missing")
    value = os.environ.get(args.private_key_env, "")
    if not value.strip():
        raise ValueError("private key is missing")
    if args.verification_key is None:
        raise ValueError("verification key is missing")
    key_path = Path(args.verification_key)
    if not key_path.is_file():
        raise ValueError("verification key is missing")
    public_pem = key_path.read_bytes()
    if b"PRIVATE KEY" in public_pem:
        raise ValueError("verification key is untrusted")
    return value.encode("utf-8"), public_pem


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _fail(message: str, **fields: object) -> int:
    payload: dict[str, object] = {"error": message}
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    _emit(payload)
    return 1


def _safe_message(exc: BaseException) -> str:
    text = str(exc)
    lowered = text.casefold()
    if "hf_token" in lowered or "begin private" in lowered:
        return "release failed"
    if any(marker in text for marker in ("E:\\", "C:\\", "\\\\", "/home/", "/Users/")):
        if "canonical" in lowered:
            return "canonical BCICIV-2a repository must not be overwritten"
        if "source" in lowered:
            return "source file changed since the planned manifest"
        if "plan" in lowered:
            return "validated plan not found"
        return "release failed"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
