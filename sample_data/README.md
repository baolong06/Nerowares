# THINKING sample data

This directory contains tiny deterministic fixtures for clone-and-run smoke tests.
They are committed so a teammate can clone the repository, set `THINKING_DATA_ROOT=sample_data`, and exercise the real dataset loaders without downloading the full EEG corpus.

Included fixtures:

- `huggingface/EEG-semantic-text-relevance/data/train-00000-of-00001.parquet` — D05-shaped semantic relevance sample.
- `kaggle/aymanmostafa11__eeg-motor-imagery-bciciv-2a/BCICIV_2a_all_patients.csv` — BCICIV2a-shaped motor imagery sample.

These files are synthetic toy fixtures. They are not the 130GB raw dataset collection, not a Hugging Face release, and not evidence for scientific headline claims or deployment readiness.

Regenerate or validate them with:

```bash
python scripts/create_sample_data.py
python scripts/create_sample_data.py --check
```
