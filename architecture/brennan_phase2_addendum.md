# Brennan Alice — Phase 2 kết quả và artifact addendum

**Ngày ghi nhận:** 2026-09-22  
**Protocol ID:** `brennan-alice-eval-v1`  
**Trạng thái:** Phase 2 baseline đã được xác thực qua anti-leakage gate

## Phạm vi và tính bất biến của protocol

Bản khóa khoa học vẫn là:

- `E:/AI_thucchien/THINKING/datasets/nemar/nm000180_brennan2019_alice/PROTOCOL.json`
- `E:/AI_thucchien/THINKING/datasets/nemar/nm000180_brennan2019_alice/PROTOCOL.md`

Các file trên không bị sửa. Các quy tắc primary cohort 27 người, sentence retrieval 29 câu, LOSO-same-stimulus, stimulus-block holdout, temporal buffer 1 giây, năm control và cấm random-epoch/fake-cross-session vẫn giữ nguyên.

Pre-Fix AppSec gate đã được duyệt trước khi loader/MLP được triển khai; hồ sơ tham chiếu là [`reports/Pre-Fix-AppSec-Report-Brennan-2026-09-21.pdf`](../reports/Pre-Fix-AppSec-Report-Brennan-2026-09-21.pdf) và JSON tương ứng.

## Canonical baseline

Artifact chính thức:

- Thư mục: [`artifacts/brennan-primary-mlp-best/`](../artifacts/brennan-primary-mlp-best/)
- Báo cáo: [`train_report.json`](../artifacts/brennan-primary-mlp-best/train_report.json)
- Model: [`encoder.npz`](../artifacts/brennan-primary-mlp-best/encoder.npz)
- Cấu hình: `MlpAnchorEncoder`, `hidden_dim=128`, `mlp_epochs=120`, `mlp_lr=0.002`, `ridge=1e-4`, `embed_dim=64`, `n_channels=32`, `n_times=64`, seed `0`
- SHA-256 encoder: `b2cb578d77f61e65ac36c2e4ccb0099dc30fa6f5e9b595d87e58552d00831461`

Bản kiểm chứng độc lập tại [`artifacts/brennan-primary-mlp-best-verify/`](../artifacts/brennan-primary-mlp-best-verify/) có cùng SHA-256 và cùng kết quả report. Runner-up được lưu tại [`artifacts/brennan-primary-mlp-runnerup/`](../artifacts/brennan-primary-mlp-runnerup/) với `hidden_dim=128`, `mlp_epochs=200`, `mlp_lr=0.002`.

## Phạm vi thực tế của run

Protocol khóa có primary cohort 27 người. Tuy nhiên report của artifact canonical ghi rõ source `primary25`, `n_epochs=614` và `loso_same_stimulus.n_folds=25`. Addendum này giữ nguyên đúng provenance của report: đây là kết quả của **25 fold / 25 subject materialized run**, không mở rộng hoặc suy diễn thành kết quả 27-subject. Hai subject không xuất hiện trong report không được tự động điền hoặc gán kết quả.

Stimulus-block report ghi:

- Held-out segments: `10, 11, 12`
- Train/test: `439 / 175` epochs
- Train/test labels: `21 / 8`
- Temporal buffer: `1.0 s`

## Kết quả anti-leakage gate

### Stimulus-block holdout

| Chỉ số | Real EEG | Control mạnh nhất | Kết luận |
|---|---:|---:|---|
| Top-1 | 0.0229 | — | Không dùng để claim giải mã Top-1 |
| Top-5 | **0.2286** | Gaussian noise `0.1886` | Vượt `0.0400` điểm tuyệt đối |
| Top-25 | 0.8629 | — | Chỉ là metric phụ |

Các control Top-5: shuffled-label `0.1600`, temporal-shuffle `0.1371`, Gaussian-noise `0.1886`, no-EEG LM-only `0.1724`. Stimulus-block pass.

### LOSO-same-stimulus

| Chỉ số | Real EEG | Control mạnh nhất | Kết luận |
|---|---:|---:|---|
| Top-1 | 0.0379 | chance `1/29 = 0.0345` | Chỉ nhỉnh hơn chance trong protocol này |
| Top-5 | **0.1770** | No-EEG LM-only `0.1724` | Vượt `0.0046` điểm tuyệt đối |
| Top-25 | 0.8962 | — | Chỉ là metric phụ |

LOSO pass và combined `gate_pass=true` vì cả LOSO lẫn stimulus-block đều vượt toàn bộ control tương ứng.

## Diễn giải bị giới hạn

- Kết quả được phép báo cáo là **retrieval Top-5 vượt controls** trong hai protocol đã khóa.
- Top-1 stimulus-block `0.0229` thấp hơn chance `0.0345`; không được gọi là sentence decoding chính xác.
- LOSO là generalization giữa người nghe cùng stimulus, không phải zero-shot semantic generalization.
- Đây không phải word-level classification trên 601 word types và không phải cross-session evaluation.
- `encoder.npz` là artifact đã qua gate cho benchmark retrieval; không tự động đồng nghĩa với production deployment hoặc EEG-to-text generation hoàn chỉnh.

## Reproducibility và regression

Lệnh verify full pipeline đã tái tạo đúng các con số canonical và sinh encoder có cùng SHA-256. Full regression:

```text
49 passed, 45 warnings
```

Các warning là warning kỹ thuật đã biết của MNE/ICA và dọn thư mục tạm trên Windows; không có test failure.

Manifest tổng hợp nằm tại [`artifacts/brennan_phase2_manifest.json`](../artifacts/brennan_phase2_manifest.json).
