# Dataset release

Operator runbook for family-scoped Hugging Face Dataset Hub releases, plus the Task 9 telemetry schema.

## Operator workflow

### Plan generation

Generate a dry-run plan with no remote writes:

```bash
PYTHONPATH=. python scripts/dataset_release.py --plan --all \
  --config dataset_catalog/families.release.json \
  --data-root "$THINKING_DATA_ROOT" \
  --manifest-dir artifacts/release-manifests \
  --state-dir artifacts/release-state
```

Plan generation inventories existing package roots, records governance and scope, and writes small metadata only. It does not upload, archive, or duplicate raw EEG.

### Governance review

Governance is fail-closed. A family is uploadable only after an explicit decision:

- `public` — evidence supports public redistribution
- `private-team` — evidence supports the approved team audience and storage ACL
- `blocked` — no upload operation is permitted

`private-team` is not a substitute for missing license, DUA, custodian, or de-identification permission. Missing or `not_reviewed` / `unknown` / `denied` statuses stay blocked.

### Direct upload

Approved files are streamed from the existing family package roots. The release must create **no local raw copy**, tar.zst, canonical duplicate, or full-tree staging directory. Only small metadata and resumable upload state are written locally. `repo_type=dataset` on every Hub call.

### Clean-machine download

On a machine without the local `datasets/` tree, verify the signed family manifest before any shard download:

```bash
PYTHONPATH=. python scripts/dataset_download.py \
  --dataset D01 \
  --provider huggingface \
  --release-manifest artifacts/release-manifests/D01.release.json \
  --signature artifacts/release-manifests/D01.release.json.ed25519.sig \
  --verification-key "$THINKING_RELEASE_VERIFY_KEY" \
  --max-cache-bytes 200000000000 \
  --min-free-bytes 20000000000
```

`--release-manifest`, `--signature`, and `--verification-key` are required together. An unsigned full-family manifest is never treated as release-verified. Hugging Face family revisions must be a 40-character commit SHA (`main` / `master` / `latest` / `head` / `default` are rejected).

### Cache verification

After download, or against an existing cache:

```bash
PYTHONPATH=. python scripts/dataset_verify.py \
  --dataset D01 \
  --provider huggingface \
  --local-files-only \
  --release-manifest artifacts/release-manifests/D01.release.json \
  --signature artifacts/release-manifests/D01.release.json.ed25519.sig \
  --verification-key "$THINKING_RELEASE_VERIFY_KEY"
```

Cache verification checks shard size and SHA-256 at the pinned revision. Quota and free-space preflight (`--max-cache-bytes`, `--min-free-bytes`) run before any downloader or Hub client is constructed.

### Family-scoped training

Train only on a verified family revision. Artifact lineage may include optional release fields (`release_id`, `scope_status`, `manifest_signature_status`, `governance_status`, `file_count`, `total_size_bytes`) copied from a `VerifiedRelease`. Catalog-only BCICIV-2a handles omit those keys; do not fabricate them.

### Artifact lineage

Identity keys remain `dataset_id`, `revision`, `manifest_sha256`, `provider`, and `mode`. Release fields are omitted when unset. Lineage never includes `HF_TOKEN`, PEM material, raw EEG, prompts, or absolute dataset roots.

### Private repository credentials

Private repository credentials: `HF_TOKEN` / `hf_token` comes from the environment or a secret manager only. Tokens and private keys must never appear in logs, manifests, dataset cards, CLI JSON, or SIEM payloads.

### `observed-local-package` vs `full-upstream`

- `observed-local-package` — the release is exactly the local package that was inventoried; it does not claim upstream completeness.
- `full-upstream` — the release claims completeness relative to the declared upstream source and must have evidence for that claim.
- `blocked` — out of scope for upload and for release-verified download.

D03 is the Chisco **subject-02** subset. D03 `scope_status` must stay `observed-local-package` and must never be `full-upstream`.

D15 full raw family release **never overwrites** the canonical BCICIV-2a CSV repository `madteam/thinking-bciciv2a` revision `1cd0deae08404efd86bf6b84d6beb2575c245a5e`. Full D15 uses a separate family repository.

Immutable revisions only. Parser allowlist stays `{.bdf,.edf,.fif,.parquet,.npy,.npz,.csv}`. Preservation of other extensions as opaque bytes does not widen the parser.

## Dataset release telemetry

Allowlisted, hash-only audit events for family release. Events never include raw EEG, prompts, token values, credentials, shard contents, or filesystem paths. Task 11 expands the operator runbook; this file documents the event schema only.

## Event types

- `dataset_inventory_reconciliation_failed`
- `dataset_release_plan_created`
- `dataset_shard_build_failed`
- `dataset_upload_blocked`
- `dataset_upload_batch_failed`
- `dataset_release_signature_failed`
- `dataset_shard_verification_failed`
- `dataset_release_verified`
- `dataset_release_verification_failed` (detection name for repeated verification failures)

`sanitize_release_event` keeps both the spec types and the detection name. Sigma, KQL, and Splunk rules watch repeated `dataset_release_verification_failed`, `dataset_inventory_reconciliation_failed`, and `dataset_upload_blocked`.

## Fields

| Field | Meaning |
|---|---|
| `event_type` | One of the event types above |
| `timestamp` | Event time |
| `correlation_id` | Request or job correlation identifier |
| `user_sub` | Actor subject |
| `dataset_id` | Family identifier (D01–D19) |
| `release_id` | Release identifier |
| `manifest_sha256` | Canonical manifest digest |
| `path_hash` | SHA-256 of a shard or filesystem identifier; raw path is never emitted |
| `provider` | Remote provider name when present |
| `mode` | Operator mode (for example plan or upload) |
| `result` | Outcome token |
| `reason` | Short allowlisted reason token |
| `count` | Item count when useful |
| `total_size_bytes` | Byte total when useful |

HTTP SIEM events continue to use `path` as the request path. Release telemetry uses `path_hash` instead of a filesystem path. Existing SIEM transport policy is unchanged: production endpoints must be HTTPS; forwarding failures are logged without exception text.
