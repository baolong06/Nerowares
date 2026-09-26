"""Pinned dataset catalogs committed with the source repository."""
from __future__ import annotations

from pathlib import Path

from .manifest import DatasetManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_DIR = REPO_ROOT / "dataset_catalog"


def catalog_dir(path: Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_CATALOG_DIR


def catalog_path(dataset_id: str, provider: str, directory: Path | None = None) -> Path:
    if not dataset_id or any(part in {"", ".", ".."} for part in dataset_id.replace("\\", "/").split("/")):
        raise ValueError("dataset_id must be a simple catalog key")
    if "/" in dataset_id or "\\" in dataset_id:
        raise ValueError("dataset_id must be a simple catalog key")
    if provider not in {"local", "huggingface"}:
        raise ValueError("dataset provider must be local or huggingface")
    return catalog_dir(directory) / f"{dataset_id}.{provider}.json"


def load_catalog_manifest(
    dataset_id: str,
    provider: str,
    catalog_dir: Path | None = None,
) -> DatasetManifest:
    """Load `{dataset_id}.{provider}.json` and reject moving revisions."""
    path = catalog_path(dataset_id, provider, catalog_dir)
    if not path.is_file():
        raise FileNotFoundError(f"dataset catalog is missing: {path}")
    return DatasetManifest.from_json(path)


__all__ = ["DEFAULT_CATALOG_DIR", "catalog_dir", "catalog_path", "load_catalog_manifest"]
