# Phase 3 — Mở rộng dataset D01 / Chisco / D09 / BCICIV-2a (không gộp)

**Ngày:** 2026-09-23  
**Worktree:** `E:\AI_thucchien\THINKING\.claude\worktrees\stoic-haslett-10e9dc`  
**Gate:** [`reports/Pre-Fix-AppSec-Report-Phase3-2026-09-22.pdf`](../reports/Pre-Fix-AppSec-Report-Phase3-2026-09-22.pdf) + [`reports/Pre-Fix-AppSec-Assessment-Phase3-2026-09-22.json`](../reports/Pre-Fix-AppSec-Assessment-Phase3-2026-09-22.json) — mean **5.59/10**, hard-stop đã duyệt trước khi code loader  
**Canonical Brennan giữ nguyên:** [`artifacts/brennan-primary-mlp-best/`](../artifacts/brennan-primary-mlp-best/) SHA `b2cb578d77f61e65ac36c2e4ccb0099dc30fa6f5e9b595d87e58552d00831461`  
**Manifest máy:** [`artifacts/phase3_manifest.json`](../artifacts/phase3_manifest.json)

## 1) Phạm vi

Phase 3 chỉ thêm **loader đọc dữ liệu thật + hợp đồng tách anti-leakage theo từng dataset**. Không gộp dataset, không claim tổng quát liên-dataset, không đổi `ALLOWED_EXTENSIONS` toàn cục, không pickle, không commit/push/PR.

- D01 `ds003626_inner_speech` — đã có.
- Chisco `ds005170` subject02 — đã có (45 EDF + 45 xlsx).
- D09 BCI2020 — đã có (Training/Validation/Test, MAT v5 + v7.3).
- SparrKULee — chưa có local → **fail-closed**.

## 2) Hợp đồng format (đã khóa trong code)

| Dataset | Input được phép | Parser | Từ chối | Size cap |
|---|---|---|---|---|
| D01 | `derivatives/sub-*/ses-*/sub-*_eeg-epo.fif` | `mne.read_epochs` | `.pkl` báo cáo, raw BDF, mọi suffix khác | 2 GB/file |
| Chisco | `sub-*/ses-*/eeg/*_eeg.edf` + `textdataset/split_data_{run}.xlsx` ghép **theo vị trí** với `run = _run-0*(\d+)_` | `mne.io.read_raw_edf` + `openpyxl` read_only | thiếu cặp, xlsx >5 MB, EDF ngoài root | EDF 1 GB, XLSX 5 MB |
| D09 | `Training set/Data_Sample*.mat` / `Validation set/...` (MAT v5) | `scipy.io.loadmat` (`epo_train`/`epo_validation`, `x(795,64,300)`) | Test HDF5 v7.3 (báo lỗi giải mã), mọi `.mat` ngoài root | 2 GB/file |
| SparrKULee | — | — | Mọi suy đoán format | — |

Tất cả đi qua `safe_resolve` + `_is_relative_to(root)` + `suffix ∈ {closed set}` + `exists/is_file` + `stat size cap`. `.mat/.xlsx/.pkl/.vhdr/.vmrk/.eeg/.dat` **không** được thêm vào `src/preprocessing/io.py:ALLOWED_EXTENSIONS` (vẫn `{.bdf,.edf,.fif,.parquet,.npy,.npz,.csv}`).

Mọi EEG được lọc **1–40 Hz FIR ở native sfreq** trước khi epoch/downsample/mask: D01 256 Hz, Chisco 1000 Hz (pick `eeg` trước), D09 256 Hz qua `_phase3_preprocess(..., fs)`.

## 3) Hợp đồng BCICIV-2a — đã khóa

BCICIV-2a là family mới được thêm sau gate mở rộng, vẫn tách biệt hoàn toàn với D01/Chisco/D09/Brennan.

| Thuộc tính | Hợp đồng đã khóa |
|---|---|
| `local_root` | `kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a` |
| Input mặc định | `BCICIV_2a_all_patients.csv` (một CSV consolidated; các CSV per-patient chỉ là dữ liệu tham chiếu) |
| Parser / security | `csv.DictReader` + `safe_resolve`/root-scope + suffix đóng `{.csv}` + `is_file` + size cap 300,000,000 bytes; không pickle, không MAT, không giải nén |
| Native schema | 22 EEG channels, 201 samples/window, `time` step 0.004 s ⇒ 250 Hz; cửa sổ đã windowed, không áp dụng FIR lần hai |
| Labels | Cột `label`, lowercase, đúng closed set `{left,right,foot,tongue}` |
| Identity | `subject_id = sub-XX` từ `patient`; `session_id = sub-XX:epoch-XXX` từ `epoch`; không suy diễn identity khác |
| Split contract | LOSO trên `subject_id` và cross-session trên `session_id`, chỉ khả dụng khi có ít nhất hai group; `allow_random_fallback=False` và thiếu group phải fail-closed |
| Bounded selection | `max_subjects` giới hạn subject đầu theo thứ tự số; `max_epochs` chọn round-robin theo subject để cap nhỏ vẫn giữ đa subject, deterministic, không random |
| Preprocessing | Nếu bật Autoreject/ICA, chạy tại native 250 Hz trước spatial/temporal pooling; mặc định smoke tắt để kiểm tra schema |

Observed local CSV: 214,848,010 bytes, 2,448 epochs, phân bố `sub-01..sub-09 = 288,288,288,144,288,288,288,288,288`; mỗi epoch 201 rows và 22 channels. Smoke `max_epochs=64, n_times=32, n_channels=16` giữ 9 subjects, 64 unique sessions; `train_and_eval` tạo được LOSO 9 folds và cross-session 64 folds, không random fallback.

## 4) Identity và split — không bịa, không random fallback

`SplitConfig(allow_random_fallback=False)` và `train_and_eval` không bao giờ tự rơi về random-epoch.

- **D01:** `subject_id = sub-XX`, `session_id = sub-XX:ses-YY`. Cần ≥2 subject/session khác nhau thì cả `cross_session` lẫn `LOSO` mới khả dụng. Smoke `max_subjects=2,max_sessions=1` ⇒ 2 folds mỗi chiều (đã kiểm). `1sub/2ses` mất LOSO và **phải fail-closed** — đúng ADR-003.
- **Chisco:** một subject duy nhất local (`sub-02`). `session_id = sub-02:run-XXX` (run là ranh giới thu nhận, `ses-01` đơn lẻ không phải unit tách). Do đó `cross_session` trên run là hợp lệ (3 runs ⇒ 3 folds trong smoke), còn `LOSO` **phải từ chối** bằng `ValueError: random-epoch split is banned`.
- **D09:** một subject (`sub-1`), `session_id = sub-1:train` / `sub-1:validation`. Cả `LOSO` và `cross_session` đều **phải từ chối** — đúng hành vi đã kiểm. Headline duy nhất không rò rỉ là **train→validation holdout retrieval** (không phải multipath LOSO/cross_session), và cấm gộp với D01/Chisco.

## 4) Kết quả smoke (không phải claim khoa học)

Chạy thật, slice nhỏ, `ridge=0.1, embed_dim=16, n_times=32, n_channels=16`:

- D01 `2subs/1ses` 400 epochs — `loso` Top-5 0.765, `cross_session` Top-5 0.765, gate `Top5 > max controls` = **false** (slice 4-class, không claim).
- Chisco `3runs` 500 epochs (EDF 2140 s/1608 s/1678 s đã đọc) — `cross_session` 3 folds Top-5 0.146, `loso` từ chối đúng.
- D09 `train 32 epochs` — cả hai grouped splits từ chối đúng; demo holdout `train 80 → validation 50` là đường đánh giá đúng cho single-subject.

## 5) Bảo toàn

- Brennan canonical và regression giữ nguyên: `PYTHONPATH=. pytest -q` → **58 passed, 48 warnings** (warnings là MNE/ICA + dọn `Temp` trên Windows).
- Không commit/push/PR. Artefact mới duy nhất là `artifacts/phase3_manifest.json` và addendum này; Pre-Fix PDF/JSON không sửa.
