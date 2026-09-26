# Nerowares — THINKING EEG Semantic Anchors Platform

Nguồn mở cho team clone về chạy / train ngay. **Không chứa 130GB raw EEG** — raw shards ở Hugging Face Dataset Hub hoặc `THINKING_DATA_ROOT` cục bộ, CI giữ `HF_HUB_OFFLINE=1`.

## Clone & chạy smoke

```bash
git clone https://github.com/baolong06/Nerowares.git
cd Nerowares
python3.11 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,ml,security,data]"

# Test offline (không tải dataset)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. pytest -q

# Train smoke không cần raw 130GB
PYTHONPATH=. python -m src.training.train --dataset d05 --max-rows 240 --n-times 64
PYTHONPATH=. python -m src.training.train --dataset brennan --max-sentences 8 --max-subjects 2

# Serve API
uvicorn src.main:app --reload --port 8000  # http://localhost:8000/docs (chỉ khi docs_enabled + development)
```

## BCICIV2a verified (mã nguồn đã hỗ trợ, dữ liệu cần token/catalog)

```bash
# Local (nếu có THINKING_DATA_ROOT/datasets)
python scripts/dataset_info.py --dataset bciciv2a --provider local
python scripts/dataset_verify.py --dataset bciciv2a --provider local

# Hugging Face verified (cần HF_TOKEN trong env, không commit token)
# HF_TOKEN chỉ đọc từ environment/secret manager
python scripts/dataset_download.py --dataset bciciv2a --provider huggingface
python scripts/dataset_verify.py --dataset bciciv2a --provider huggingface --local-files-only
```

Catalog: `dataset_catalog/` — BCICIV-2a canonical `madteam/thinking-bciciv2a@1cd0deae08404efd86bf6b84d6beb2575c245a5e` (40-char immutable). Family D01–D19 hiện `blocked` (fail-closed), cần `approved_by` + `evidence_reference` mới upload.

## Cấu trúc

`src/` (api, data, preprocessing, training, anchors, prompt, soc) · `tests/` · `scripts/` (dataset_*, check_*, generate_*) · `architecture/` · `dataset_catalog/` · `soc/` (Sigma/KQL/SPL) · `.github/workflows/` (sca + reproducibility manual)

## Family release (D01–D19) — khi có phê duyệt governance

```bash
# Dry-run từng family (không --all, không --upload)
PYTHONPATH=. python scripts/dataset_release.py --plan --family D01 --config dataset_catalog/families.release.json --data-root /path/datasets --manifest-dir artifacts/release-manifests --state-dir artifacts/release-state
# Upload có ký detached Ed25519 + verify (cần THINKING_HF_DATASET_TOKEN trong env)
PYTHONPATH=. python scripts/dataset_release.py --upload --family D01 --config ... --data-root ... --manifest-dir ... --state-dir ... --signing-key /run/secrets/ed25519.pem
```

## Bảo mật

`HF_TOKEN`/`THINKING_HF_DATASET_TOKEN` và private key không bao giờ commit/log/SIEM. `python scripts/check_secrets.py` và `python scripts/check_cve.py` trước mỗi push. Governance `blocked` mặc định.

License: MIT (giáo dục).
