"""Versioned dataset sources and verified cache helpers."""

from .catalog import load_catalog_manifest
from .huggingface import HuggingFaceDatasetSource
from .manifest import DatasetManifest, DatasetShard
from .resolve import resolve_dataset_source
from .source import DatasetHandle, DatasetSource, LocalDatasetSource

__all__ = [
    "DatasetHandle",
    "DatasetManifest",
    "DatasetShard",
    "DatasetSource",
    "HuggingFaceDatasetSource",
    "LocalDatasetSource",
    "load_catalog_manifest",
    "resolve_dataset_source",
]
