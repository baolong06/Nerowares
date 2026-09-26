# Dataset catalog

Pinned manifests live next to the source repository. Raw EEG does not.

Each catalog file is `{dataset_id}.{provider}.json` and must satisfy
`src.data.manifest.DatasetManifest`:

- `provider` is `local` or `huggingface`
- `revision` is immutable (`main`, `master`, `latest`, `head`, and `default` are rejected)
- Hugging Face revisions are a 40-character commit SHA
- every shard lists a relative path, exact byte size, and SHA-256

## Local catalogs

A local catalog pins files under `THINKING_DATA_ROOT` (worktree `datasets/`
if present, otherwise the legacy machine path when it exists). Example
relative shard:

```text
kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv
```

Do not commit a local catalog until the shard checksum is measured from
the real file.

```bash
python scripts/dataset_info.py --dataset bciciv2a --provider local
python scripts/dataset_verify.py --dataset bciciv2a --provider local
```

## Hugging Face catalogs

The BCICIV-2a public Dataset Hub repository is:

- repo: `madteam/thinking-bciciv2a`
- immutable revision: `1cd0deae08404efd86bf6b84d6beb2575c245a5e`
- shard: `BCICIV_2a_all_patients.csv`
- bytes: `214848010`
- SHA-256: `fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd`

The pinned catalog is `bciciv2a.huggingface.json`. Do not replace its
40-character revision with `main` or another moving reference. Upload is
out of band and is not performed by these verification scripts.

```bash
python scripts/dataset_download.py --dataset bciciv2a --provider huggingface
python scripts/dataset_verify.py --dataset bciciv2a --provider huggingface --local-files-only
```

`HF_TOKEN` is read from the environment when the Hub requires it. Tokens
are never printed, logged, or written into manifests.

## Family release

Family-scoped Hugging Face Dataset Hub releases (D01–D19) are separate
from the canonical BCICIV-2a catalog. Governance is fail-closed
(`public`, `private-team`, `blocked`). Direct upload reads existing
package roots and creates **no local raw copy**.

D15 full raw family release never overwrites the canonical BCICIV-2a
repository `madteam/thinking-bciciv2a` revision
`1cd0deae08404efd86bf6b84d6beb2575c245a5e`. Use a separate family
repository. D03 is the Chisco subject-02 subset
(`observed-local-package`), never `full-upstream`.

Clean-machine download requires a signed family release manifest,
detached signature, and verification key together. Catalog-only
BCICIV-2a stays valid without those flags.

## GitHub vs Hub

GitHub holds source, tests, catalogs, CLI, and CI. Versioned raw shards
belong on Hugging Face Dataset Hub. CI must stay offline
(`HF_HUB_OFFLINE=1`) and must not download the 133 GB tree.
