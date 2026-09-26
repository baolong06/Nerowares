"""Create tiny deterministic sample datasets for clone-and-run smoke tests."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.anchors.encoder import label_template
from src.training.dataset import (
    _BCICIV2A_EEG_COLUMNS,
    _BCICIV2A_NATIVE_TIMES,
    _BCICIV2A_TIME_STEP,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "sample_data"
D05_SAMPLE = SAMPLE_ROOT / "huggingface" / "EEG-semantic-text-relevance" / "data" / "train-00000-of-00001.parquet"
BCICIV2A_SAMPLE = SAMPLE_ROOT / "kaggle" / "aymanmostafa11__eeg-motor-imagery-bciciv-2a" / "BCICIV_2a_all_patients.csv"
D05_TOPICS = ["open", "window", "left", "right", "yes"]
BCICIV2A_LABELS = ["left", "right", "foot", "tongue"]


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError("Creating D05 sample parquet requires pyarrow; install .[data] or .[ml].") from exc
    return pa, pq


def create_d05_sample(path: Path = D05_SAMPLE) -> None:
    pa, pq = _require_pyarrow()
    rng = np.random.default_rng(20260926)
    rows: dict[str, list[object]] = {
        "word": [],
        "topic": [],
        "selected_topic": [],
        "participant": [],
        "eeg": [],
        "semantic_relevance": [],
    }
    for index in range(64):
        topic = D05_TOPICS[index % len(D05_TOPICS)]
        epoch = label_template(topic, 32, 64) + (0.04 * rng.standard_normal((32, 64))).astype(np.float32)
        rows["word"].append(f"{topic}-sample-{index:02d}")
        rows["topic"].append(topic)
        rows["selected_topic"].append(topic)
        rows["participant"].append(f"sample-sub-{index % 4 + 1:02d}")
        rows["eeg"].append(epoch.astype(np.float32).tolist())
        rows["semantic_relevance"].append(1)

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows), path)


def create_bciciv2a_sample(path: Path = BCICIV2A_SAMPLE) -> None:
    rng = np.random.default_rng(20260927)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient", "time", "label", "epoch", *_BCICIV2A_EEG_COLUMNS])
        for patient in range(1, 5):
            for epoch_id in range(1, 5):
                label = BCICIV2A_LABELS[(patient + epoch_id - 2) % len(BCICIV2A_LABELS)]
                epoch = label_template(label, len(_BCICIV2A_EEG_COLUMNS), _BCICIV2A_NATIVE_TIMES)
                epoch = epoch + (0.03 * rng.standard_normal(epoch.shape)).astype(np.float32)
                for time_index in range(_BCICIV2A_NATIVE_TIMES):
                    writer.writerow([
                        str(patient),
                        f"{time_index * _BCICIV2A_TIME_STEP:.6f}",
                        label,
                        str(epoch_id),
                        *[f"{float(value):.8f}" for value in epoch[:, time_index]],
                    ])


def check_d05_sample(path: Path = D05_SAMPLE) -> None:
    _, pq = _require_pyarrow()
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {"word", "topic", "selected_topic", "participant", "eeg", "semantic_relevance"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"D05 sample is missing columns: {sorted(missing)}")
    if parquet.metadata.num_rows < 32:
        raise ValueError("D05 sample needs at least 32 rows for smoke training")
    first = parquet.read_row_group(0, columns=["eeg"]).to_pydict()["eeg"][0]
    arr = np.asarray(first, dtype=np.float32)
    if arr.shape != (32, 64):
        raise ValueError(f"D05 sample EEG shape must be (32, 64), got {arr.shape}")


def check_bciciv2a_sample(path: Path = BCICIV2A_SAMPLE) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: set[str] = set()
    patients: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        required = {"patient", "time", "label", "epoch", *_BCICIV2A_EEG_COLUMNS}
        missing = required - header
        if missing:
            raise ValueError(f"BCICIV2a sample is missing columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            labels.add(str(row["label"]).strip().lower())
            patients.add(str(row["patient"]).strip())
    if rows != 4 * 4 * _BCICIV2A_NATIVE_TIMES:
        raise ValueError(f"BCICIV2a sample row count mismatch: {rows}")
    if labels != set(BCICIV2A_LABELS):
        raise ValueError(f"BCICIV2a sample labels mismatch: {sorted(labels)}")
    if len(patients) != 4:
        raise ValueError(f"BCICIV2a sample needs 4 patients, got {sorted(patients)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or validate bundled THINKING sample data")
    parser.add_argument("--check", action="store_true", help="validate committed sample files without rewriting them")
    args = parser.parse_args(argv)

    if not args.check:
        create_d05_sample()
        create_bciciv2a_sample()
    check_d05_sample()
    check_bciciv2a_sample()
    print(f"sample data OK under {SAMPLE_ROOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
