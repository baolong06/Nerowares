# Nerowares — THINKING EEG Semantic Anchors Platform

Nguồn mở cho team clone về chạy / train smoke ngay. Repo **không chứa 130GB raw EEG**; thay vào đó có `sample_data/` là bộ fixture rất nhỏ để kiểm tra loader, test và pipeline huấn luyện cơ bản.

## Clone & chạy ngay với sample data

### Linux / macOS

```bash
git clone https://github.com/baolong06/Nerowares.git
cd Nerowares
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml,security,data]"

export THINKING_DATA_ROOT=sample_data
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

pytest -q
python scripts/create_sample_data.py --check
python -m src.training.train --dataset d05 --max-rows 32 --n-times 32 --artifacts-dir artifacts/sample-d05
python -c "from src.training.dataset import load_bciciv2a_epochs; d=load_bciciv2a_epochs(max_epochs=8,n_times=32,n_channels=8,use_autoreject=False,use_ica=False); print(d.epochs.shape, sorted(set(d.labels)))"
uvicorn src.main:app --reload --port 8000
```

### Windows PowerShell

```bash
git clone https://github.com/baolong06/Nerowares.git
cd Nerowares
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,ml,security,data]"

$env:THINKING_DATA_ROOT = "sample_data"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = "."

pytest -q
python scripts/create_sample_data.py --check
python -m src.training.train --dataset d05 --max-rows 32 --n-times 32 --artifacts-dir artifacts/sample-d05
python -c "from src.training.dataset import load_bciciv2a_epochs; d=load_bciciv2a_epochs(max_epochs=8,n_times=32,n_channels=8,use_autoreject=False,use_ica=False); print(d.epochs.shape, sorted(set(d.labels)))"
uvicorn src.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs` khi chạy ở development và docs được bật.

## Sample data đi kèm

`sample_data/` chứa fixture nhỏ, deterministic, chỉ dùng cho onboarding/smoke test:

- `sample_data/huggingface/EEG-semantic-text-relevance/data/train-00000-of-00001.parquet` — D05-shaped semantic relevance sample.
- `sample_data/kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv` — BCICIV2a-shaped motor imagery sample.

Đây **không phải** raw EEG corpus 130GB, không phải Hugging Face release chính thức, và không phải bằng chứng scientific/deployment readiness. Có thể kiểm tra hoặc tái tạo fixture bằng:

```bash
python scripts/create_sample_data.py --check
python scripts/create_sample_data.py
```

## Dữ liệu thật / advanced

Nếu có raw datasets thật ở local, đặt `THINKING_DATA_ROOT` trỏ tới thư mục dữ liệu đó. BCICIV2a verified source vẫn được hỗ trợ qua catalog/token riêng:

```bash
python scripts/dataset_info.py --dataset bciciv2a --provider local
python scripts/dataset_verify.py --dataset bciciv2a --provider local

# Hugging Face verified path cần HF_TOKEN trong env, không commit token
python scripts/dataset_download.py --dataset bciciv2a --provider huggingface
python scripts/dataset_verify.py --dataset bciciv2a --provider huggingface --local-files-only
```

Catalog: `dataset_catalog/` — BCICIV-2a canonical `madteam/thinking-bciciv2a@1cd0deae08404efd86bf6b84d6beb2575c245a5e` (40-char immutable). Family D01–D19 hiện `blocked` fail-closed, cần governance thật (`approved_by` + `evidence_reference`) mới upload.

## Cấu trúc

`src/` (api, data, preprocessing, training, anchors, prompt, soc) · `tests/` · `scripts/` (dataset_*, check_*, generate_*) · `sample_data/` · `architecture/` · `dataset_catalog/` · `soc/` (Sigma/KQL/SPL) · `.github/workflows/`.

## Bảo mật

`HF_TOKEN`/`THINKING_HF_DATASET_TOKEN` và private key không bao giờ commit/log/SIEM. Chạy `python scripts/check_secrets.py` và `python scripts/check_cve.py` trước mỗi push. Governance mặc định fail-closed.

License: MIT (giáo dục).
