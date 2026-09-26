"""Labeled EEG loaders for D05, COFETT, and Brennan BIDS data."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.anchors.encoder import label_template
from src.anchors.vocab import load_vocab
from src.preprocessing.io import safe_resolve


def _discover_data_root() -> Path:
    env = os.environ.get("THINKING_DATA_ROOT")
    if env:
        return Path(env)
    worktree = Path(__file__).resolve().parents[2] / "datasets"
    legacy = Path("E:/AI_thucchien/THINKING/datasets")
    if worktree.exists():
        return worktree
    if legacy.exists():
        return legacy
    return worktree


DATA_ROOT = _discover_data_root()
D05_DIR = DATA_ROOT / "huggingface" / "EEG-semantic-text-relevance" / "data"
COFETT_DIR = DATA_ROOT / "openneuro" / "ds006317_cofett"
BRENNAN_DIR = DATA_ROOT / "nemar" / "nm000180_brennan2019_alice"
BRENNAN_PROTOCOL = BRENNAN_DIR / "PROTOCOL.json"
D01_DIR = DATA_ROOT / "openneuro" / "ds003626_inner_speech"
CHISCO_DIR = DATA_ROOT / "openneuro" / "ds005170_chisco_subject02"
D09_DIR = DATA_ROOT / "kaggle" / "abdulkareembageri__imagined-speech-eeg-signal-bci2020" / "BCI2020 EEG Signal for Words"
BCICIV2A_DIR = DATA_ROOT / "kaggle" / "aymanmostafa11__eeg-motor-imagery-bciciv-2a"
BCICIV2A_CSV = BCICIV2A_DIR / "BCICIV_2a_all_patients.csv"
SPARRKULEE_DIR = DATA_ROOT / "sparrkulee"  # not present locally — fail-closed


@dataclass
class LabeledEpochs:
    """EEG epochs and the metadata needed for leakage-safe evaluation."""

    epochs: np.ndarray  # (n_epochs, n_channels, n_times)
    labels: list[str]
    subject_ids: list[str]
    session_ids: list[str]
    source: str
    stimulus_ids: list[str] | None = None
    metadata: list[dict[str, object]] | None = None
    # Versioned source metadata is kept separate from per-epoch metadata so
    # training reports can bind a run to a Hub revision without copying raw data.
    data_lineage: dict[str, object] | None = None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _block_downsample(eeg: np.ndarray, n_times: int) -> np.ndarray:
    """Mean-pool one EEG epoch to its requested temporal resolution."""
    if eeg.ndim != 2:
        raise ValueError("EEG epochs must have shape (channels, times)")
    if n_times < 1:
        raise ValueError("n_times must be positive")

    channels, times = eeg.shape
    if times < 1:
        raise ValueError("EEG epochs must contain at least one time point")
    if times == n_times:
        return eeg.astype(np.float32, copy=False)

    boundaries = np.linspace(0, times, n_times + 1, dtype=int)
    pooled = np.empty((channels, n_times), dtype=np.float32)
    for index in range(n_times):
        start = boundaries[index]
        stop = max(start + 1, boundaries[index + 1])
        pooled[:, index] = eeg[:, start:stop].mean(axis=1)
    return pooled


def d05_parquet_shards(directory: Path = D05_DIR) -> list[Path]:
    """Return D05 training shards that reside below an approved data root."""
    if not directory.exists():
        return []
    return [safe_resolve(path) for path in sorted(directory.glob("train-*.parquet"))]


def synthetic_d05(
    max_rows: int = 240,
    n_times: int = 64,
    n_channels: int = 8,
    seed: int = 0,
) -> LabeledEpochs:
    """Build a deterministic, label-structured D05-shaped smoke-test dataset.

    Every subject has two sessions, so both cross-session and LOSO evaluation
    remain available when the real parquet shards are absent.
    """
    if max_rows < 4:
        raise ValueError("Synthetic D05 requires at least four rows for grouped evaluation")
    if n_channels < 1:
        raise ValueError("n_channels must be positive")

    labels_vocab = load_vocab().keywords[:5]
    if not labels_vocab:
        raise RuntimeError("Anchor vocabulary is empty")

    rng = np.random.default_rng(seed)
    epochs: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    for index in range(max_rows):
        label = labels_vocab[index % len(labels_vocab)]
        subject = f"synthetic-sub-{index % 4:02d}"
        session = f"{subject}:ses-{(index // 4) % 2 + 1:02d}"
        noise = 0.08 * rng.standard_normal((n_channels, n_times))
        epochs.append(label_template(label, n_channels, n_times) + noise.astype(np.float32))
        labels.append(label)
        subjects.append(subject)
        sessions.append(session)

    return LabeledEpochs(
        epochs=np.stack(epochs).astype(np.float32),
        labels=labels,
        subject_ids=subjects,
        session_ids=sessions,
        source="synthetic_d05",
    )


def load_d05(
    max_rows: int = 400,
    n_times: int = 64,
    shard: int = 0,
    relevant_only: bool = False,
    label_field: str = "topic",
    max_shards: int = 8,
) -> LabeledEpochs:
    """Load D05 through PyArrow; fall back to synthetic epochs when unavailable.

    Data is sampled evenly from consecutive parquet shards. This preserves each
    shard as a session boundary and normally exposes several participants for
    both cross-session and leave-one-subject-out evaluation.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if shard < 0:
        raise ValueError("shard must be non-negative")
    if max_shards < 1:
        raise ValueError("max_shards must be positive")
    if label_field not in {"topic", "selected_topic", "word"}:
        raise ValueError("label_field must be topic, selected_topic, or word")

    shards = d05_parquet_shards()
    if not shards:
        return synthetic_d05(max_rows=max_rows, n_times=n_times)

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("D05 parquet exists but loading it requires pyarrow") from exc

    start = min(shard, len(shards) - 1)
    selected_shards = shards[start : start + max_shards]
    rows_per_shard = max(1, max_rows // len(selected_shards))
    required_columns = {"word", "topic", "selected_topic", "participant", "eeg", "semantic_relevance"}
    epochs: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    columns = sorted(required_columns)
    for shard_offset, path in enumerate(selected_shards):
        parquet_file = pq.ParquetFile(str(path))
        missing_columns = required_columns - set(parquet_file.schema_arrow.names)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"D05 parquet is missing required columns: {missing}")

        shard_rows = 0
        budget = min(rows_per_shard, max_rows - len(epochs))
        for batch in parquet_file.iter_batches(batch_size=32, columns=columns):
            rows = batch.to_pydict()
            for index, word_value in enumerate(rows["word"]):
                if relevant_only and int(rows["semantic_relevance"][index] or 0) != 1:
                    continue
                raw_epoch = np.asarray(rows["eeg"][index], dtype=np.float32)
                if raw_epoch.ndim != 2 or raw_epoch.shape[0] < 1 or raw_epoch.shape[1] < 1:
                    continue
                word = str(word_value or "").strip().lower()
                topic = str(rows["topic"][index] or "").strip().lower()
                selected = str(rows["selected_topic"][index] or "").strip().lower()
                label = {"topic": topic, "selected_topic": selected, "word": word}[label_field]
                if not label:
                    label = word or topic
                if not label:
                    continue
                subject = str(rows["participant"][index] or "unknown")
                epochs.append(_block_downsample(raw_epoch, n_times))
                labels.append(label)
                subjects.append(subject)
                # D05 has no explicit session field, so the physical parquet
                # shard is the defensible acquisition-session boundary.
                sessions.append(f"{subject}:shard-{start + shard_offset}")
                shard_rows += 1
                if shard_rows >= budget or len(epochs) >= max_rows:
                    break
            if shard_rows >= budget or len(epochs) >= max_rows:
                break
        if len(epochs) >= max_rows:
            break

    if not epochs:
        raise RuntimeError("D05 parquet produced no valid epochs")
    return LabeledEpochs(
        epochs=np.stack(epochs),
        labels=labels,
        subject_ids=subjects,
        session_ids=sessions,
        source=f"d05:{label_field}",
    )


def _bids_ids(path: Path) -> tuple[str, str]:
    """Extract BIDS subject/session entities from directories or a BIDS filename."""
    subject = "unknown"
    session = "unknown"
    # Directory names are authoritative when present.
    for part in path.parts[:-1]:
        if part.startswith("sub-"):
            subject = part
        elif part.startswith("ses-"):
            session = part
    # COFETT files encode both entities in their basename, e.g.
    # sub-01_ses-01_task-para1_run-01_events.tsv.
    for entity in path.name.split("_"):
        if entity.startswith("sub-"):
            subject = entity
        elif entity.startswith("ses-"):
            session = entity
    return subject, session


_SKIP_EVENT_VALUES = {"65480", "65481", "event_65480", "event_65481"}


def _spatial_downsample(eeg: np.ndarray, n_channels: int) -> np.ndarray:
    """Mean-pool channels so ridge stays tractable on high-density caps."""
    if eeg.ndim != 2:
        raise ValueError("EEG epochs must have shape (channels, times)")
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    channels, times = eeg.shape
    if channels <= n_channels:
        return eeg.astype(np.float32, copy=False)
    boundaries = np.linspace(0, channels, n_channels + 1, dtype=int)
    pooled = np.empty((n_channels, times), dtype=np.float32)
    for index in range(n_channels):
        start = boundaries[index]
        stop = max(start + 1, boundaries[index + 1])
        pooled[index] = eeg[start:stop].mean(axis=0)
    return pooled


def _select_diverse_edfs(edfs: list[Path], max_files: int) -> list[Path]:
    """Pick EDFs round-robin by session then subject for grouped splits."""
    if max_files < 1:
        raise ValueError("max_files must be positive")
    by_group: dict[tuple[str, str], list[Path]] = {}
    for path in edfs:
        by_group.setdefault(_bids_ids(path), []).append(path)
    subjects = sorted({subject for subject, _ in by_group})
    sessions = sorted({session for _, session in by_group})
    selected: list[Path] = []
    while len(selected) < max_files:
        progressed = False
        for session in sessions:
            for subject in subjects:
                if len(selected) >= max_files:
                    break
                files = by_group.get((subject, session), [])
                if not files:
                    continue
                selected.append(files.pop(0))
                progressed = True
            if len(selected) >= max_files:
                break
        if not progressed:
            break
    return selected


def _parse_event_file(event_path: Path) -> list[dict[str, str]]:
    """Read one BIDS events TSV and add stable subject/session metadata."""
    safe_path = safe_resolve(event_path)
    lines = [line for line in safe_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    subject, session = _bids_ids(safe_path)
    records: list[dict[str, str]] = []
    for line in lines[1:]:
        record = dict(zip(header, line.split("\t")))
        record["subject_id"] = subject
        # Qualify session by subject so cross-session folds do not mix people.
        record["session_id"] = f"{subject}:{session}"
        record["source_file"] = safe_path.name
        records.append(record)
    return records


def _take_event_budget(records: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    """Keep a strided subset of usable events, dropping session-marker codes."""
    usable: list[dict[str, str]] = []
    for record in records:
        label = str(record.get("trial_type") or record.get("value") or "event")
        value = str(record.get("value") or "")
        if label in _SKIP_EVENT_VALUES or value in _SKIP_EVENT_VALUES:
            continue
        usable.append(record)
    if budget < 1 or len(usable) <= budget:
        return usable
    step = len(usable) / budget
    return [usable[int(index * step)] for index in range(budget)]


def load_cofett_events(max_files: int = 2) -> list[dict[str, str]]:
    """Load COFETT BIDS event records without accessing unapproved paths."""
    if max_files < 1:
        raise ValueError("max_files must be positive")
    if not COFETT_DIR.exists():
        raise FileNotFoundError("COFETT BIDS root not found")
    rows: list[dict[str, str]] = []
    for event_path in _select_diverse_edfs(sorted(COFETT_DIR.rglob("*_events.tsv")), max_files):
        rows.extend(_parse_event_file(event_path))
    return rows


def load_cofett_epochs(
    max_files: int = 4,
    n_times: int = 64,
    epoch_sec: float = 0.8,
    max_epochs: int = 240,
    n_channels: int = 32,
    use_autoreject: bool = True,
    use_ica: bool = True,
) -> LabeledEpochs:
    """Epoch COFETT EDF data around BIDS events. Requires optional MNE.

    Bandpass / Autoreject / ICA run at the native sampling rate. Temporal
    downsampling happens only after that, so FIR length stays shorter than the
    epoch and ICA sees high-passed data.
    """
    if max_files < 1:
        raise ValueError("max_files must be positive")
    if epoch_sec <= 0:
        raise ValueError("epoch_sec must be positive")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("COFETT EDF epoching requires MNE") from exc
    if not COFETT_DIR.exists():
        raise FileNotFoundError("COFETT BIDS root not found")

    edfs = _select_diverse_edfs(sorted(COFETT_DIR.rglob("*_eeg.edf")), max_files)
    if not edfs:
        raise FileNotFoundError("No COFETT EDF files")

    per_file = max(1, max_epochs // len(edfs))
    native_epochs: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    sample_rate = 1000.0
    for edf in edfs:
        safe_edf = safe_resolve(edf)
        event_path = safe_edf.with_name(safe_edf.name.replace("_eeg.edf", "_events.tsv"))
        if not event_path.exists():
            continue
        event_records = _take_event_budget(_parse_event_file(event_path), per_file)
        if not event_records:
            continue
        raw = mne.io.read_raw_edf(str(safe_edf), preload=True, verbose=False)
        try:
            raw.load_data()
            raw.filter(
                l_freq=1.0,
                h_freq=40.0,
                picks="eeg",
                method="fir",
                fir_window="hamming",
                verbose=False,
            )
            sample_rate = float(raw.info["sfreq"])
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            eeg_data = raw.get_data(picks=picks) * 1e6
        finally:
            del raw
        subject, session = _bids_ids(safe_edf)
        session_id = f"{subject}:{session}"
        width = int(epoch_sec * sample_rate)
        for record in event_records:
            if len(native_epochs) >= max_epochs:
                break
            start = int(float(record.get("onset") or 0) * sample_rate)
            stop = start + width
            if start < 0 or stop > eeg_data.shape[1]:
                continue
            native_epochs.append(_spatial_downsample(eeg_data[:, start:stop], n_channels))
            labels.append(str(record.get("trial_type") or record.get("value") or "event"))
            subjects.append(record.get("subject_id") or subject)
            sessions.append(record.get("session_id") or session_id)
        del eeg_data
        if len(native_epochs) >= max_epochs:
            break

    if not native_epochs:
        raise RuntimeError("COFETT epoching produced no windows")

    stacked = np.stack(native_epochs).astype(np.float32)
    if use_autoreject or use_ica:
        from src.preprocessing.pipeline import PreprocessingConfig, preprocess_epochs

        processed = preprocess_epochs(
            stacked,
            PreprocessingConfig(
                sampling_rate_hz=sample_rate,
                apply_bandpass=False,
                use_autoreject=use_autoreject,
                use_ica=use_ica,
                bad_epoch_threshold_uv=800.0,
                ica_n_components=min(8, stacked.shape[1]),
            ),
        )
        keep = processed.mask_keep
        n_subjects = len({sid for sid, flag in zip(subjects, keep) if flag})
        n_sessions = len({sid for sid, flag in zip(sessions, keep) if flag})
        if int(keep.sum()) >= 16 and n_subjects >= 2 and n_sessions >= 2:
            stacked = processed.epochs[keep]
            labels = [label for label, flag in zip(labels, keep) if flag]
            subjects = [sid for sid, flag in zip(subjects, keep) if flag]
            sessions = [sid for sid, flag in zip(sessions, keep) if flag]
        else:
            stacked = processed.epochs

    downsampled = np.stack([_block_downsample(epoch, n_times) for epoch in stacked])
    return LabeledEpochs(
        epochs=downsampled,
        labels=labels,
        subject_ids=subjects,
        session_ids=sessions,
        source="cofett",
    )


def _load_brennan_protocol(protocol_path: Path = BRENNAN_PROTOCOL) -> dict:
    safe_path = safe_resolve(protocol_path)
    if not safe_path.exists():
        raise FileNotFoundError("Brennan protocol lock not found")
    return json.loads(safe_path.read_text(encoding="utf-8"))


def _resolve_brennan_brainvision(vhdr_path: str | Path, root: Path = BRENNAN_DIR) -> Path:
    """Return a safe Brennan BrainVision header after validating its required triplet."""
    safe_vhdr = safe_resolve(vhdr_path)
    safe_root = safe_resolve(root).resolve()
    if safe_vhdr.suffix.lower() != ".vhdr":
        raise ValueError("BrainVision ingest must start from a .vhdr header")
    if not _is_relative_to(safe_vhdr.resolve(), safe_root):
        raise ValueError("BrainVision path must stay inside the Brennan root")

    limits = {".vhdr": 1_000_000, ".vmrk": 2_000_000, ".eeg": 512_000_000}
    for suffix, limit in limits.items():
        sibling = safe_vhdr.with_suffix(suffix)
        safe_sibling = safe_resolve(sibling)
        if safe_sibling.parent != safe_vhdr.parent or safe_sibling.stem != safe_vhdr.stem:
            raise ValueError("BrainVision triplet filenames must share one stem")
        if not _is_relative_to(safe_sibling.resolve(), safe_root):
            raise ValueError("BrainVision triplet must stay inside the Brennan root")
        if not safe_sibling.exists() or not safe_sibling.is_file():
            raise FileNotFoundError(f"Missing BrainVision sibling {suffix}")
        if safe_sibling.stat().st_size > limit:
            raise ValueError(f"BrainVision sibling {suffix} exceeds the size limit")
    return safe_vhdr


def _read_brennan_events(event_path: Path) -> list[dict[str, str]]:
    safe_path = safe_resolve(event_path)
    with safe_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _brennan_eeg_channel_names(channels_path: Path) -> list[str]:
    safe_path = safe_resolve(channels_path)
    with safe_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [str(row["name"]) for row in rows if str(row.get("type") or "").upper() == "EEG"]


def _group_brennan_sentences(events: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for row in events:
        sentence_id = int(float(row["sentence_id"]))
        groups.setdefault(sentence_id, []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row["onset"]))
    return groups


def _brennan_subjects_for_cohort(protocol: dict, cohort: str) -> list[str]:
    cohort_data = protocol["cohort"]
    if cohort == "primary":
        return list(cohort_data["primary_subjects"])
    if cohort == "analyzed":
        return list(cohort_data["analyzed_yes"])
    if cohort == "events_available":
        excluded = set(cohort_data["eeg_without_events"])
        return sorted(path.name for path in BRENNAN_DIR.glob("sub-*") if path.is_dir() and path.name not in excluded)
    raise ValueError("cohort must be primary, analyzed, or events_available")


def load_brennan_epochs(
    cohort: str = "primary",
    n_times: int = 64,
    n_channels: int = 32,
    max_subjects: int | None = None,
    max_sentences: int | None = None,
    use_autoreject: bool = True,
    use_ica: bool = True,
    tmin: float = -0.2,
    max_epoch_sec: float = 5.0,
) -> LabeledEpochs:
    """Load Brennan Alice sentence-retrieval epochs from the locked protocol.

    BrainVision files are accepted only as a Brennan-local .vhdr/.eeg/.vmrk
    triplet; the global ingestion allowlist remains unchanged.
    """
    if n_times < 1:
        raise ValueError("n_times must be positive")
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max_subjects must be positive when provided")
    if max_sentences is not None and max_sentences < 1:
        raise ValueError("max_sentences must be positive when provided")
    if max_epoch_sec <= 0:
        raise ValueError("max_epoch_sec must be positive")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Brennan BrainVision epoching requires MNE") from exc
    if not BRENNAN_DIR.exists():
        raise FileNotFoundError("Brennan BIDS root not found")

    protocol = _load_brennan_protocol()
    subjects = _brennan_subjects_for_cohort(protocol, cohort)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    sentence_specs = list(protocol["task"]["sentences"])
    if max_sentences is not None:
        sentence_specs = sentence_specs[:max_sentences]
    sentence_text = {int(spec["sentence_id"]): str(spec["text"]) for spec in sentence_specs}
    sentence_order = [int(spec["sentence_id"]) for spec in sentence_specs]

    native_epochs: list[np.ndarray] = []
    labels: list[str] = []
    subject_ids: list[str] = []
    session_ids: list[str] = []
    stimulus_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    sample_rate = float(protocol["dataset"].get("sfreq_hz", 500.0))
    native_width = int(round((max_epoch_sec - tmin) * sample_rate))

    for subject in subjects:
        eeg_dir = BRENNAN_DIR / subject / "eeg"
        vhdr = _resolve_brennan_brainvision(eeg_dir / f"{subject}_task-alicelistening_eeg.vhdr")
        event_path = eeg_dir / f"{subject}_task-alicelistening_events.tsv"
        channels_path = eeg_dir / f"{subject}_task-alicelistening_channels.tsv"
        if not event_path.exists():
            continue
        events = _group_brennan_sentences(_read_brennan_events(event_path))
        selected: list[tuple[int, list[dict[str, str]], float, float]] = []
        for sentence_id in sentence_order:
            rows = events.get(sentence_id)
            if not rows:
                continue
            onset = min(float(row["onset"]) for row in rows)
            offset = max(float(row["onset"]) + float(row["duration"]) for row in rows)
            selected.append((sentence_id, rows, onset, min(offset, onset + max_epoch_sec)))
        if not selected:
            continue

        raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose=False)
        try:
            eeg_names = _brennan_eeg_channel_names(channels_path)
            present = [name for name in eeg_names if name in raw.ch_names]
            if not present:
                raise RuntimeError(f"No EEG channels found for {subject}")
            raw.pick_channels(present, ordered=True)
            crop_start = max(0.0, min(onset + tmin for _, _, onset, _ in selected))
            crop_stop = min(float(raw.times[-1]), max(offset for _, _, _, offset in selected))
            raw.crop(tmin=crop_start, tmax=crop_stop, include_tmax=True)
            raw.load_data()
            raw.filter(
                l_freq=1.0,
                h_freq=40.0,
                picks="all",
                method="fir",
                fir_window="hamming",
                verbose=False,
            )
            sample_rate = float(raw.info["sfreq"])
            eeg_data = raw.get_data() * 1e6
        finally:
            del raw

        for sentence_id, rows, onset, offset in selected:
            start = int(round((onset + tmin - crop_start) * sample_rate))
            duration = min(offset - onset, max_epoch_sec)
            stop = int(round((onset + duration - crop_start) * sample_rate))
            if start < 0 or stop <= start or stop > eeg_data.shape[1]:
                continue
            epoch = eeg_data[:, start:stop].astype(np.float32)
            if epoch.shape[1] < native_width:
                epoch = np.pad(epoch, ((0, 0), (0, native_width - epoch.shape[1])), mode="constant")
            elif epoch.shape[1] > native_width:
                epoch = epoch[:, :native_width]
            pooled = _spatial_downsample(epoch, n_channels)
            segment = int(float(rows[0]["segment"]))
            stim = f"segment:{segment:02d}"
            native_epochs.append(pooled)
            labels.append(sentence_text[sentence_id])
            subject_ids.append(subject)
            session_ids.append(f"{subject}:run-01")
            stimulus_ids.append(stim)
            metadata.append(
                {
                    "subject_id": subject,
                    "session_id": f"{subject}:run-01",
                    "sentence_id": sentence_id,
                    "segment": segment,
                    "stimulus_id": stim,
                    "onset": float(onset),
                    "offset": float(offset),
                    "epoch_tmin": float(tmin),
                    "epoch_tmax": float(duration),
                    "n_words": len(rows),
                    "label": sentence_text[sentence_id],
                    "source_file": event_path.name,
                }
            )
        del eeg_data

    if not native_epochs:
        raise RuntimeError("Brennan epoching produced no windows")

    stacked = np.stack(native_epochs).astype(np.float32)
    if use_autoreject or use_ica:
        from src.preprocessing.pipeline import PreprocessingConfig, preprocess_epochs

        processed = preprocess_epochs(
            stacked,
            PreprocessingConfig(
                sampling_rate_hz=sample_rate,
                apply_bandpass=False,
                use_autoreject=use_autoreject,
                use_ica=use_ica,
                bad_epoch_threshold_uv=800.0,
                ica_n_components=min(8, stacked.shape[1]),
            ),
        )
        keep = processed.mask_keep
        if int(keep.sum()) >= max(1, min(8, len(labels) // 4)):
            stacked = processed.epochs[keep]
            labels = [label for label, flag in zip(labels, keep) if flag]
            subject_ids = [sid for sid, flag in zip(subject_ids, keep) if flag]
            session_ids = [sid for sid, flag in zip(session_ids, keep) if flag]
            stimulus_ids = [sid for sid, flag in zip(stimulus_ids, keep) if flag]
            metadata = [item for item, flag in zip(metadata, keep) if flag]
        else:
            stacked = processed.epochs

    downsampled = np.stack([_block_downsample(epoch, n_times) for epoch in stacked])
    source = f"brennan:sentence_retrieval_2to5s:{cohort}{len(set(subject_ids))}"
    return LabeledEpochs(
        epochs=downsampled,
        labels=labels,
        subject_ids=subject_ids,
        session_ids=session_ids,
        source=source,
        stimulus_ids=stimulus_ids,
        metadata=metadata,
    )


def _resolve_phase3_file(path: str | Path, root: Path, extensions: set[str]) -> Path:
    """Resolve one Phase 3 input below a dataset root with a closed extension set."""
    safe_root = safe_resolve(root).resolve()
    candidate = safe_resolve(path)
    if not _is_relative_to(candidate.resolve(), safe_root):
        raise ValueError("Phase 3 path must stay inside its dataset root")
    if candidate.suffix.lower() not in extensions:
        raise ValueError("Phase 3 file extension is not allowed")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Phase 3 input not found: {candidate.name}")
    return candidate


def _phase3_validate_limit(path: Path, max_bytes: int) -> None:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Phase 3 input exceeds the size limit: {path.name}")


def _phase3_preprocess(
    epochs: np.ndarray,
    sample_rate: float,
    *,
    use_autoreject: bool,
    use_ica: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Run optional cleanup at native rate and return epochs plus keep mask."""
    if not use_autoreject and not use_ica:
        return epochs.astype(np.float32, copy=False), np.ones(len(epochs), dtype=bool)
    from src.preprocessing.pipeline import PreprocessingConfig, preprocess_epochs

    result = preprocess_epochs(
        epochs,
        PreprocessingConfig(
            sampling_rate_hz=sample_rate,
            apply_bandpass=False,
            use_autoreject=use_autoreject,
            use_ica=use_ica,
            bad_epoch_threshold_uv=800.0,
            ica_n_components=min(8, epochs.shape[1]),
        ),
    )
    return result.epochs.astype(np.float32, copy=False), result.mask_keep


def _apply_keep(
    epochs: np.ndarray,
    labels: list[str],
    subjects: list[str],
    sessions: list[str],
    stimulus_ids: list[str],
    metadata: list[dict[str, object]],
    keep: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str], list[str], list[str], list[dict[str, object]]]:
    if len(keep) != len(labels):
        raise ValueError("Phase 3 preprocessing mask length does not match metadata")
    return (
        epochs[keep],
        [value for value, flag in zip(labels, keep) if flag],
        [value for value, flag in zip(subjects, keep) if flag],
        [value for value, flag in zip(sessions, keep) if flag],
        [value for value, flag in zip(stimulus_ids, keep) if flag],
        [value for value, flag in zip(metadata, keep) if flag],
    )


def _phase3_return(
    epochs: list[np.ndarray],
    labels: list[str],
    subjects: list[str],
    sessions: list[str],
    stimulus_ids: list[str],
    metadata: list[dict[str, object]],
    *,
    n_times: int,
    source: str,
    data_lineage: dict[str, object] | None = None,
) -> LabeledEpochs:
    if not epochs:
        raise RuntimeError(f"{source} produced no valid epochs")
    stacked = np.stack([_block_downsample(epoch, n_times) for epoch in epochs]).astype(np.float32)
    return LabeledEpochs(
        epochs=stacked,
        labels=labels,
        subject_ids=subjects,
        session_ids=sessions,
        source=source,
        stimulus_ids=stimulus_ids,
        metadata=metadata,
        data_lineage=data_lineage,
    )


def _d01_label(event_code: int) -> str:
    return {31: "arriba", 32: "abajo", 33: "derecha", 34: "izquierda"}.get(event_code, "")


def load_d01_epochs(
    *,
    max_subjects: int | None = None,
    max_sessions: int | None = None,
    n_times: int = 64,
    n_channels: int = 32,
    use_autoreject: bool = True,
    use_ica: bool = True,
    max_epochs: int | None = None,
) -> LabeledEpochs:
    """Load D01 Inner Speech derivative FIF epochs with preserved BIDS identities.

    The raw BDF is not reparsed because D01's event annotations are stored in
    derivative FIF event tables. FIF is read through MNE; pickle reports are
    never considered input. Native 256 Hz filtering/cleanup occurs before
    temporal downsampling.
    """
    if n_times < 1 or n_channels < 1:
        raise ValueError("n_times and n_channels must be positive")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("D01 loading requires MNE") from exc
    if not D01_DIR.exists():
        raise FileNotFoundError("D01 dataset root not found")
    subjects = sorted({path.parts[-3] for path in D01_DIR.glob("derivatives/sub-*/ses-*/sub-*_eeg-epo.fif")})
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    selected: list[Path] = []
    for subject in subjects:
        sessions = sorted(D01_DIR.glob(f"derivatives/{subject}/ses-*/{subject}_*_eeg-epo.fif"))
        selected.extend(sessions[:max_sessions] if max_sessions is not None else sessions)
    if not selected:
        raise FileNotFoundError("No D01 derivative FIF epochs found")

    epochs_out: list[np.ndarray] = []
    labels: list[str] = []
    subjects_out: list[str] = []
    sessions_out: list[str] = []
    stimulus_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    for fif in selected:
        safe_fif = _resolve_phase3_file(fif, D01_DIR, {".fif"})
        _phase3_validate_limit(safe_fif, 2_000_000_000)
        ep = mne.read_epochs(str(safe_fif), preload=True, verbose=False)
        sample_rate = float(ep.info["sfreq"])
        data = ep.get_data(copy=True).astype(np.float32) * 1e6
        data, keep = _phase3_preprocess(data, sample_rate, use_autoreject=use_autoreject, use_ica=use_ica)
        # preprocess mask is applied to event rows before spatial/temporal pooling.
        events = ep.events[keep]
        subject = fif.parts[-3]
        session = fif.parts[-2]
        for index, epoch in enumerate(data):
            label = _d01_label(int(events[index, 2]))
            if not label:
                continue
            epochs_out.append(_spatial_downsample(epoch, n_channels))
            labels.append(label)
            subjects_out.append(subject)
            sessions_out.append(f"{subject}:{session}")
            stimulus_ids.append(label)
            metadata.append({
                "subject_id": subject,
                "session_id": f"{subject}:{session}",
                "label": label,
                "event_code": int(events[index, 2]),
                "source_file": safe_fif.name,
                "sfreq_hz": sample_rate,
                "format": "fif",
            })
            if max_epochs is not None and len(epochs_out) >= max_epochs:
                break
        if max_epochs is not None and len(epochs_out) >= max_epochs:
            break
    return _phase3_return(epochs_out, labels, subjects_out, sessions_out, stimulus_ids, metadata, n_times=n_times, source="d01:inner_speech")


def _chisco_run_number(path: Path) -> int:
    import re

    match = re.search(r"_run-0*(\d+)_", path.name)
    if not match:
        raise ValueError(f"Chisco EDF has no numeric run entity: {path.name}")
    return int(match.group(1))


def _read_chisco_labels(path: Path) -> list[tuple[str, str]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Chisco labels require openpyxl") from exc
    _phase3_validate_limit(path, 5_000_000)
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(workbook.active.iter_rows(min_row=2, values_only=True))
    result: list[tuple[str, str]] = []
    for row in rows:
        if len(row) < 2:
            continue
        sentence, label = str(row[0] or "").strip(), str(row[1] or "").strip()
        if sentence and label:
            result.append((sentence, label))
    return result


def load_chisco_epochs(
    *,
    max_runs: int | None = None,
    n_times: int = 64,
    n_channels: int = 32,
    epoch_tmin: float = 0.0,
    epoch_sec: float = 5.0,
    use_autoreject: bool = True,
    use_ica: bool = True,
    max_epochs: int | None = None,
) -> LabeledEpochs:
    """Load local Chisco EDF runs paired positionally with locked XLSX labels."""
    if n_times < 1 or n_channels < 1 or epoch_sec <= 0 or epoch_tmin < 0:
        raise ValueError("invalid Chisco epoch parameters")
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Chisco loading requires MNE") from exc
    if not CHISCO_DIR.exists():
        raise FileNotFoundError("Chisco dataset root not found")
    edfs = sorted(CHISCO_DIR.glob("sub-*/ses-*/eeg/*_eeg.edf"), key=lambda path: (_chisco_run_number(path), str(path)))
    if max_runs is not None:
        edfs = edfs[:max_runs]
    if not edfs:
        raise FileNotFoundError("No Chisco EDF runs found")

    epochs_out: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    stimulus_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    for edf in edfs:
        safe_edf = _resolve_phase3_file(edf, CHISCO_DIR, {".edf"})
        _phase3_validate_limit(safe_edf, 1_000_000_000)
        run = _chisco_run_number(safe_edf)
        label_path = CHISCO_DIR / "textdataset" / f"split_data_{run}.xlsx"
        safe_label_path = _resolve_phase3_file(label_path, CHISCO_DIR, {".xlsx"})
        label_rows = _read_chisco_labels(safe_label_path)
        raw = mne.io.read_raw_edf(str(safe_edf), preload=False, verbose=False)
        try:
            sfreq = float(raw.info["sfreq"])
            annotations = list(zip(raw.annotations.onset.tolist(), raw.annotations.description.tolist()))
            if not annotations:
                continue
            # Chisco annotations are one marker per imagined-speech trial. Read only
            # the requested windows, filter native-rate, then downsample each window.
            n_samples = int(round(epoch_sec * sfreq))
            raw.load_data()
            raw.pick(picks="eeg")
            raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
            eeg = raw.get_data() * 1e6
        finally:
            del raw
        native: list[np.ndarray] = []
        run_labels: list[str] = []
        run_meta: list[dict[str, object]] = []
        for index, (onset, description) in enumerate(annotations):
            start = int(round((float(onset) + epoch_tmin) * sfreq))
            stop = start + n_samples
            if start < 0 or stop > eeg.shape[1] or index >= len(label_rows):
                continue
            sentence, label = label_rows[index]
            native.append(_spatial_downsample(eeg[:, start:stop].astype(np.float32), n_channels))
            run_labels.append(label)
            run_meta.append({
                "subject_id": safe_edf.parts[-4],
                "session_id": f"{safe_edf.parts[-4]}:{safe_edf.parts[-3]}",
                "run": run,
                "trial_index": index,
                "sentence": sentence,
                "annotation": str(description),
                "onset": float(onset),
                "offset": float(onset) + epoch_sec,
                "source_file": safe_edf.name,
                "sfreq_hz": sfreq,
                "format": "edf",
            })
        if not native:
            continue
        processed, keep = _phase3_preprocess(np.stack(native), sfreq, use_autoreject=use_autoreject, use_ica=use_ica)
        for index, epoch in enumerate(processed):
            if not keep[index]:
                continue
            item = run_meta[index]
            epochs_out.append(epoch)
            labels.append(run_labels[index])
            subjects.append(str(item["subject_id"]))
            # Session boundary is the acquisition run: all trials in ses-01
            # share one BIDS session, so run is the defensible split unit.
            sessions.append(f"{str(item['subject_id'])}:run-{run:03d}")
            stimulus_ids.append(f"run:{run}:trial:{index}")
            metadata.append({**item, "session_id": f"{str(item['subject_id'])}:run-{run:03d}"})
            if max_epochs is not None and len(epochs_out) >= max_epochs:
                break
        del eeg
        if max_epochs is not None and len(epochs_out) >= max_epochs:
            break
    return _phase3_return(epochs_out, labels, subjects, sessions, stimulus_ids, metadata, n_times=n_times, source="chisco:imagined_speech")


def _d09_extract_struct(value: object, field: str) -> object:
    if hasattr(value, field):
        return getattr(value, field)
    raise ValueError(f"D09 MAT structure is missing {field}")


def load_d09_epochs(
    *,
    split: str = "train",
    max_epochs: int | None = None,
    n_times: int = 64,
    n_channels: int = 32,
    use_autoreject: bool = False,
    use_ica: bool = False,
) -> LabeledEpochs:
    """Load BCI2020 MAT v5/v7.3 epochs without pickle or generic MAT ingestion."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if n_times < 1 or n_channels < 1:
        raise ValueError("n_times and n_channels must be positive")
    split_dirs = [D09_DIR / f"{split.title()} set"]
    if split == "train":
        split_dirs.append(D09_DIR / "Training set")
    elif split == "validation":
        split_dirs.append(D09_DIR / "Validation set")
    elif split == "test":
        split_dirs.append(D09_DIR / "Test set")
    split_dir = next((candidate for candidate in split_dirs if candidate.exists()), split_dirs[0])
    files = sorted(split_dir.glob("Data_Sample*.mat"))
    if not files:
        raise FileNotFoundError(f"D09 {split} split not found")
    selected = files
    epochs_out: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    stimulus_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    class_names = ["hello", "helpme", "stop", "thankyou", "yes"]
    for path in selected:
        safe_mat = _resolve_phase3_file(path, D09_DIR, {".mat"})
        _phase3_validate_limit(safe_mat, 2_000_000_000)
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(safe_mat), squeeze_me=True, struct_as_record=False)
            root = mat[f"epo_{'validation' if split == 'validation' else 'train'}"]
            x = np.asarray(_d09_extract_struct(root, "x"), dtype=np.float32)
            y = np.asarray(_d09_extract_struct(root, "y"))
            fs = float(np.asarray(_d09_extract_struct(root, "fs")).squeeze())
            title = str(_d09_extract_struct(root, "title"))
            names = [str(item) for item in np.asarray(_d09_extract_struct(root, "className")).reshape(-1)]
        except NotImplementedError:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("D09 v7.3 loading requires h5py") from exc
            raise RuntimeError("D09 v7.3 test files require HDF5 reference decoding; use train/validation MAT v5 files")
        if x.ndim != 3:
            raise ValueError("D09 epochs must have shape (time, channels, epochs)")
        x = np.transpose(x, (2, 1, 0))
        if y.ndim == 1:
            class_idx = y.astype(int)
        else:
            class_idx = np.argmax(y, axis=0 if y.shape[0] == len(names) else 1)
        for index, epoch in enumerate(x):
            label_index = int(class_idx[index])
            label = names[label_index] if 0 <= label_index < len(names) else class_names[label_index]
            label = label.lower().replace(" ", "")
            epoch = _spatial_downsample(epoch, n_channels)
            processed, keep = _phase3_preprocess(epoch[None, ...], fs, use_autoreject=use_autoreject, use_ica=use_ica)
            if not bool(keep[0]):
                continue
            epochs_out.append(processed[0])
            labels.append(label)
            subject = title.lower().replace("sub", "sub-")
            subjects.append(subject)
            sessions.append(f"{subject}:{split}")
            stimulus_ids.append(f"class:{label}")
            metadata.append({
                "subject_id": subject,
                "session_id": f"{subject}:{split}",
                "class_index": label_index,
                "label": label,
                "source_file": safe_mat.name,
                "sfreq_hz": fs,
                "format": "mat-v5",
            })
            if max_epochs is not None and len(epochs_out) >= max_epochs:
                break
        if max_epochs is not None and len(epochs_out) >= max_epochs:
            break
    return _phase3_return(epochs_out, labels, subjects, sessions, stimulus_ids, metadata, n_times=n_times, source=f"d09:bci2020:{split}")


def load_sparrkulee_epochs(**_: object) -> LabeledEpochs:
    """Fail closed until a verified SparrKULee local manifest is supplied."""
    raise FileNotFoundError(
        "SparrKULee dataset is not present; refusing to infer a format or load an unverified source"
    )


_BCICIV2A_EEG_COLUMNS = [
    "EEG-Fz", "EEG-0", "EEG-1", "EEG-2", "EEG-3", "EEG-4", "EEG-5", "EEG-C3",
    "EEG-6", "EEG-Cz", "EEG-7", "EEG-C4", "EEG-8", "EEG-9", "EEG-10", "EEG-11",
    "EEG-12", "EEG-13", "EEG-14", "EEG-Pz", "EEG-15", "EEG-16",
]
_BCICIV2A_SFREQ = 250.0
_BCICIV2A_NATIVE_TIMES = 201
_BCICIV2A_TIME_STEP = 1.0 / _BCICIV2A_SFREQ
_BCICIV2A_LABELS = {"left", "right", "foot", "tongue"}


def load_bciciv2a_epochs(
    *,
    max_epochs: int | None = None,
    max_subjects: int | None = None,
    n_times: int = 64,
    n_channels: int = 32,
    use_autoreject: bool = False,
    use_ica: bool = False,
    dataset_source: object | None = None,
) -> LabeledEpochs:
    """Load bounded BCICIV-2a motor-imagery windows from its locked CSV contract.

    The consolidated CSV contains 2448 pre-windowed trials (patients 1-9,
    144/288 epochs per patient), 201 samples per epoch, 22 EEG channels, 250 Hz,
    and four labels. The loader allocates one float32 buffer per validated trial,
    keeps at most ``max_epochs`` trials, and selects subjects round-robin so a
    smoke/training cap cannot silently collapse to patient 1. The CSV is already
    windowed; no second FIR is applied. Optional Autoreject/ICA runs at native
    250 Hz before spatial/temporal pooling. No pickle or generic MAT ingestion is
    involved; the input remains a root-scoped, closed ``.csv`` contract.
    """
    if n_times < 1 or n_channels < 1:
        raise ValueError("n_times and n_channels must be positive")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("max_epochs must be positive when provided")
    if max_subjects is not None and max_subjects < 1:
        raise ValueError("max_subjects must be positive when provided")
    data_lineage: dict[str, object] | None = None
    if dataset_source is None:
        safe_csv = _resolve_phase3_file(BCICIV2A_CSV, BCICIV2A_DIR, {".csv"})
    else:
        handle = dataset_source.resolve()
        manifest = handle.manifest
        if manifest.dataset_id != "bciciv2a":
            raise ValueError("BCICIV-2a loader requires a bciciv2a dataset source")
        if manifest.format.lower() != "csv":
            raise ValueError("BCICIV-2a loader requires a CSV dataset source")
        if len(handle.files) != 1:
            raise ValueError("BCICIV-2a loader requires exactly one consolidated CSV shard")
        safe_csv = Path(handle.files[0]).resolve()
        if Path(manifest.shards[0].path).suffix.lower() != ".csv":
            raise ValueError("BCICIV-2a dataset source must resolve to a CSV shard")
        data_lineage = handle.to_lineage()
    _phase3_validate_limit(safe_csv, 300_000_000)
    import csv as _csv

    # Keep compact float32 arrays rather than Python float lists. This bounds
    # the full local fixture to roughly 220 MB of EEG buffers under the 300 MB
    # file cap and avoids an avoidable object-heavy memory multiplier.
    buffers: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    labels_by_group: dict[tuple[str, str], str] = {}
    previous_time: dict[tuple[str, str], float] = {}
    invalid_groups: set[tuple[str, str]] = set()
    with safe_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = _csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [col for col in _BCICIV2A_EEG_COLUMNS if col not in header]
        if missing:
            raise ValueError(f"BCICIV-2a CSV is missing EEG columns: {missing}")
        for col in ("patient", "epoch", "label", "time"):
            if col not in header:
                raise ValueError(f"BCICIV-2a CSV is missing required column: {col}")
        for row in reader:
            patient = str(row["patient"] or "").strip()
            epoch_id = str(row["epoch"] or "").strip()
            label = str(row["label"] or "").strip().lower()
            if not patient or not epoch_id or label not in _BCICIV2A_LABELS:
                raise ValueError("BCICIV-2a CSV contains an invalid patient, epoch, or label")
            try:
                int(patient)
                int(epoch_id)
                tval = float(str(row["time"]).strip())
                eeg_row = np.asarray([float(row[col]) for col in _BCICIV2A_EEG_COLUMNS], dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise ValueError("BCICIV-2a CSV contains non-numeric EEG or time data") from exc
            key = (patient, epoch_id)
            if key not in buffers:
                buffers[key] = np.empty((len(_BCICIV2A_EEG_COLUMNS), _BCICIV2A_NATIVE_TIMES), dtype=np.float32)
                counts[key] = 0
                labels_by_group[key] = label
            elif labels_by_group[key] != label:
                invalid_groups.add(key)
            index = counts[key]
            if index >= _BCICIV2A_NATIVE_TIMES:
                invalid_groups.add(key)
                continue
            previous = previous_time.get(key)
            if previous is not None and abs((tval - previous) - _BCICIV2A_TIME_STEP) > 1e-6:
                invalid_groups.add(key)
            previous_time[key] = tval
            buffers[key][:, index] = eeg_row
            counts[key] = index + 1

    if invalid_groups:
        patient, epoch_id = sorted(invalid_groups, key=lambda item: (int(item[0]), int(item[1])))[0]
        raise ValueError(f"BCICIV-2a time or label contract is invalid for {(patient, epoch_id)}")
    incomplete = [key for key, count in counts.items() if count != _BCICIV2A_NATIVE_TIMES]
    if incomplete:
        patient, epoch_id = sorted(incomplete, key=lambda item: (int(item[0]), int(item[1])))[0]
        raise ValueError(f"BCICIV-2a time cadence is invalid for {(patient, epoch_id)}: expected 201 rows, got {counts[(patient, epoch_id)]}")

    by_patient: dict[str, list[tuple[str, str]]] = {}
    for key in buffers:
        by_patient.setdefault(key[0], []).append(key)
    patients = sorted(by_patient, key=int)
    if max_subjects is not None:
        patients = patients[:max_subjects]
    for patient in patients:
        by_patient[patient].sort(key=lambda key: int(key[1]))

    # Round-robin selection is deterministic and preserves subject diversity.
    if max_epochs is None:
        ordered_keys = [key for patient in patients for key in by_patient[patient]]
    else:
        ordered_keys = []
        for position in range(max(len(by_patient[patient]) for patient in patients)):
            for patient in patients:
                if position < len(by_patient[patient]):
                    ordered_keys.append(by_patient[patient][position])
                    if len(ordered_keys) >= max_epochs:
                        break
            if len(ordered_keys) >= max_epochs:
                break

    epochs_out: list[np.ndarray] = []
    labels: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    stimulus_ids: list[str] = []
    metadata: list[dict[str, object]] = []
    for patient, epoch_id in ordered_keys:
        arr = _spatial_downsample(buffers[(patient, epoch_id)], n_channels)
        if use_autoreject or use_ica:
            arr_batch, keep = _phase3_preprocess(
                arr[None, ...], _BCICIV2A_SFREQ,
                use_autoreject=use_autoreject, use_ica=use_ica,
            )
            if not bool(keep[0]):
                continue
            arr = arr_batch[0]
        label = labels_by_group[(patient, epoch_id)]
        subject = f"sub-{int(patient):02d}"
        session = f"{subject}:epoch-{int(epoch_id):03d}"
        epochs_out.append(arr)
        labels.append(label)
        subjects.append(subject)
        sessions.append(session)
        stimulus_ids.append(label)
        metadata.append({
            "subject_id": subject,
            "session_id": session,
            "patient": patient,
            "epoch": epoch_id,
            "label": label,
            "sfreq_hz": _BCICIV2A_SFREQ,
            "format": "csv-windowed",
            "source_file": safe_csv.name,
            "n_times_native": _BCICIV2A_NATIVE_TIMES,
            "n_channels_native": len(_BCICIV2A_EEG_COLUMNS),
        })
    return _phase3_return(
        epochs_out, labels, subjects, sessions, stimulus_ids, metadata,
        n_times=n_times, source="bciciv2a:mi4", data_lineage=data_lineage,
    )


def load_labeled(dataset: str = "d05", **kwargs: object) -> LabeledEpochs:
    """Load one named training dataset."""
    if dataset == "d05":
        return load_d05(**kwargs)  # type: ignore[arg-type]
    if dataset == "cofett":
        return load_cofett_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset == "brennan":
        return load_brennan_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset == "d01":
        return load_d01_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset == "chisco":
        return load_chisco_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset == "d09":
        return load_d09_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset == "bciciv2a":
        return load_bciciv2a_epochs(**kwargs)  # type: ignore[arg-type]
    if dataset in {"sparrkulee", "sparrkuLee", "sparr-kulee"}:
        return load_sparrkulee_epochs(**kwargs)
    raise ValueError(f"Unknown dataset {dataset}")
