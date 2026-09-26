"""Download and verify a complete pinned dataset once; never per epoch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.huggingface import (
    align_cli_with_verified_release,
    load_verified_release_from_flags,
    preflight_download_quota,
)
from src.data.manifest import DatasetManifest
from src.data.resolve import cache_root, resolve_dataset_source

_MUTABLE_REVISIONS = {"main", "master", "latest", "head", "default"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download every shard of a pinned dataset into the verified cache"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", choices=["local", "huggingface"], required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--catalog-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--release-manifest", type=Path, default=None)
    parser.add_argument("--signature", type=Path, default=None)
    parser.add_argument("--verification-key", type=Path, default=None)
    parser.add_argument("--max-cache-bytes", type=int, default=None)
    parser.add_argument("--min-free-bytes", type=int, default=None)
    return parser


def _status_payload(handle, status: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "dataset_id": handle.dataset_id,
        "provider": handle.manifest.provider,
        "revision": handle.revision,
        "mode": handle.mode,
        "manifest_sha256": handle.manifest_sha256,
        "files": [shard.path for shard in handle.manifest.shards],
    }
    lineage = handle.to_lineage()
    for key in (
        "release_id",
        "scope_status",
        "manifest_signature_status",
        "governance_status",
        "file_count",
        "total_size_bytes",
    ):
        if key in lineage:
            payload[key] = lineage[key]
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.revision and args.revision.lower() in _MUTABLE_REVISIONS:
        raise ValueError("dataset revision must be immutable, not a moving revision")
    if args.provider == "huggingface" and (
        args.revision is not None and args.revision.lower() in _MUTABLE_REVISIONS
    ):
        raise ValueError("dataset revision must be immutable, not a moving revision")

    verified = load_verified_release_from_flags(
        release_manifest=args.release_manifest,
        signature=args.signature,
        verification_key=args.verification_key,
    )
    repo_id = args.repo_id
    revision = args.revision
    release_kwargs: dict[str, object] = {}
    if verified is not None:
        repo_id, revision, release_kwargs = align_cli_with_verified_release(
            dataset_id=args.dataset,
            repo_id=repo_id,
            revision=revision,
            verified=verified,
        )

    manifest = DatasetManifest.from_json(args.manifest) if args.manifest else None
    source = resolve_dataset_source(
        dataset_id=args.dataset,
        provider=args.provider,
        catalog_dir=args.catalog_dir,
        manifest=manifest,
        repo_id=repo_id,
        revision=revision,
        cache_dir=args.cache_dir,
        local_files_only=False,
        **release_kwargs,
    )
    if args.max_cache_bytes is not None or args.min_free_bytes is not None:
        if verified is not None:
            total = verified.total_size_bytes
        else:
            total = sum(shard.size_bytes for shard in source.manifest.shards)
        preflight_download_quota(
            total_size_bytes=total,
            cache_dir=args.cache_dir or cache_root(),
            max_cache_bytes=args.max_cache_bytes,
            min_free_bytes=args.min_free_bytes,
        )
    handle = source.resolve()
    print(
        json.dumps(
            _status_payload(handle, "downloaded"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
