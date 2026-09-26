"""Artifact manifests bind the SBOM evidence used for publication."""
from __future__ import annotations

import hashlib


def test_manifest_records_declared_and_ambient_sbom_hashes(tmp_path):
    from scripts.artifact_manifest import build_manifest

    source = tmp_path / "source.csv"
    source.write_text("fixture", encoding="utf-8")
    artifact_dir = tmp_path / "artifacts" / "bciciv2a-full"
    artifact_dir.mkdir(parents=True)
    declared = artifact_dir.parent / "sbom-declared.cdx.json"
    ambient = artifact_dir.parent / "sbom-ambient.cdx.json"
    declared.write_text('{"scope":"declared"}\n', encoding="utf-8")
    ambient.write_text('{"scope":"ambient"}\n', encoding="utf-8")

    manifest = build_manifest(
        artifact_dir=artifact_dir,
        source_csv=source,
        result={
            "source": "bciciv2a:mi4",
            "n_epochs": 2448,
            "n_channels": 16,
            "n_times": 32,
            "gate_pass": False,
        },
        subject_counts={"sub-01": 288},
    )

    assert manifest["supply_chain"]["sboms"]["declared"]["sha256"] == hashlib.sha256(declared.read_bytes()).hexdigest()
    assert manifest["supply_chain"]["sboms"]["ambient"]["sha256"] == hashlib.sha256(ambient.read_bytes()).hexdigest()
