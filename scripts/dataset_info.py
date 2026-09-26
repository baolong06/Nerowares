"""Display non-secret metadata for a pinned dataset manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.catalog import load_catalog_manifest
from src.data.manifest import DatasetManifest


def _manifest_from_args(args: argparse.Namespace) -> DatasetManifest:
    if args.manifest is not None:
        return DatasetManifest.from_json(args.manifest)
    return load_catalog_manifest(
        args.dataset,
        provider=args.provider,
        catalog_dir=args.catalog_dir,
    )


def _document(manifest: DatasetManifest) -> dict[str, object]:
    return {
        "dataset_id": manifest.dataset_id,
        "provider": manifest.provider,
        "repo_id": manifest.repo_id,
        "revision": manifest.require_revision(),
        "format": manifest.format,
        "sampling_rate_hz": manifest.sampling_rate_hz,
        "n_channels_source": manifest.n_channels_source,
        "n_times_source": manifest.n_times_source,
        "labels": list(manifest.labels),
        "subject_field": manifest.subject_field,
        "session_field": manifest.session_field,
        "manifest_sha256": manifest.sha256(),
        "shards": [shard.to_dict() for shard in manifest.shards],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display metadata for an immutable dataset manifest"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--provider", choices=["local", "huggingface"], required=True)
    parser.add_argument("--catalog-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest_from_args(args)
    print(json.dumps(_document(manifest), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
