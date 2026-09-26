# Dataset Hub architecture

GitHub is not a dataset store. Another researcher clones the source
repository, downloads a pinned Hub revision, verifies checksums, and
trains on real EEG.

```text
GitHub repository
  = source + tests + manifests + CLI + CI

Hugging Face Dataset Hub
  = versioned shards + dataset card + license + checksums

Local verified cache
  = downloaded and verified data for large training runs

DatasetSource
  = local | huggingface, resolved from a pinned catalog

Training pipeline
  = resolve revision → verify → cache/stream → preprocess → leakage-safe evaluate

Artifact lineage
  = repository, revision, manifest hash, shard hashes, preprocessing, split, gate status
```

## Invariants

- Real EEG only for scientific results.
- Filter at native sampling rate before epoching or downsampling.
- No random epoch splits.
- Brennan, D01, Chisco, D09, BCICIV-2a, and SparrKULee stay separate.
- Brennan canonical SHA remains
  `b2cb578d77f61e65ac36c2e4ccb0099dc30fa6f5e9b595d87e58552d00831461`.
- Do not claim BCICIV-2a is deployable while its scientific gate is false.
- Do not invent a Hugging Face commit SHA.
- `ALLOWED_EXTENSIONS` stays closed:
  `{.bdf,.edf,.fif,.parquet,.npy,.npz,.csv}`.
- `HF_TOKEN` is environment/secret-manager only.

## Resolution order

Local data root:

1. `THINKING_DATA_ROOT`
2. worktree `datasets/` if it exists
3. legacy `E:/AI_thucchien/THINKING/datasets` if it exists
4. worktree `datasets/` as the fail-closed default

Cache root:

1. `--cache-dir`
2. `THINKING_DATA_CACHE`
3. `~/.cache/thinking/datasets`

## Modes

- `stream`: download each manifest shard once at the pinned revision,
  then verify size and SHA-256. No HTTP request per epoch.
- `verified`: `local_files_only=True`. Training fails closed if the
  cache is empty; run `scripts/dataset_download.py` first.

## Current wiring

Verified sources are implemented for BCICIV-2a. Other families keep
their existing local loaders. The public Hugging Face dataset is now pinned
at `madteam/thinking-bciciv2a` revision
`1cd0deae08404efd86bf6b84d6beb2575c245a5e`.

The published shard is `BCICIV_2a_all_patients.csv`, with 214,848,010 bytes
and SHA-256
`fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd`.
The committed catalog is `dataset_catalog/bciciv2a.huggingface.json`.
The repository is public because the operator confirmed redistribution rights;
consumers remain responsible for upstream terms and applicable review rules.

## Family release

A family release is one Hub dataset repository per D01–D19 family, with
fail-closed governance (`public` / `private-team` / `blocked`), a signed
inventory manifest, and an immutable 40-character revision. Direct upload
streams from existing package roots and creates **no local raw copy**.

The canonical D15 exception: full raw D15 never overwrites
`madteam/thinking-bciciv2a` revision
`1cd0deae08404efd86bf6b84d6beb2575c245a5e`. That catalog pin remains the
BCICIV-2a CSV workflow. Family-scoped training records optional release
lineage only after signature verification; catalog-only BCICIV-2a omits
those keys.
