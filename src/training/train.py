"""Train and evaluate the semantic-anchor encoder without random splits."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from src.anchors.encoder import AnchorEncoder
from src.anchors.eval import EvalReport, evaluate_retrieval, operating_k
from src.anchors.mlp_encoder import MlpAnchorEncoder
from src.anchors.retrieval import build_brennan_tfidf_embedder
from src.anchors.splits import SplitConfig, make_splits
from src.training.dataset import LabeledEpochs, _load_brennan_protocol, load_brennan_epochs, load_d05, load_labeled

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SplitStrategy = Literal["cross_session", "loso"]


def _evaluation_vocab(labels: list[str]) -> list[str]:
    """Use the closed dataset label space for a valid supervised retrieval metric.

    The production anchor vocabulary is intentionally broader; including its
    unrelated interior terms during D05 evaluation would artificially depress
    Top-k and turn the no-EEG control into a mismatched task.
    """
    return list(dict.fromkeys(labels))


def _brennan_vocab(data: LabeledEpochs) -> list[str]:
    """Use the locked 29-sentence Brennan vocabulary when labels match it."""
    try:
        protocol = _load_brennan_protocol()
        vocab = [str(item["text"]) for item in protocol["task"]["sentences"]]
    except (FileNotFoundError, KeyError, TypeError):
        return _evaluation_vocab(data.labels)
    return vocab if set(data.labels).issubset(set(vocab)) else _evaluation_vocab(data.labels)


def _mean_report(reports: list[EvalReport]) -> dict[str, float | dict[str, float]]:
    if not reports:
        raise RuntimeError("No evaluation reports were produced")
    return {
        "n_folds": len(reports),
        "top1": float(np.mean([report.top1 for report in reports])),
        "top5": float(np.mean([report.top5 for report in reports])),
        "top25": float(np.mean([report.top25 for report in reports])),
        "controls": {
            key: float(np.mean([report.baselines[key] for report in reports]))
            for key in reports[0].baselines
        },
    }


def _single_report(report: EvalReport) -> dict[str, float | dict[str, float]]:
    return {
        "n_folds": 1,
        "top1": float(report.top1),
        "top5": float(report.top5),
        "top25": float(report.top25),
        "controls": {key: float(value) for key, value in report.baselines.items()},
    }


def _skipped_report(reason: str) -> dict[str, object]:
    return {
        "n_folds": 0,
        "top1": 0.0,
        "top5": 0.0,
        "top25": 0.0,
        "controls": {},
        "skipped": reason,
    }


def _passes_controls(report: dict[str, object]) -> bool:
    controls = report.get("controls")
    top5 = report.get("top5")
    if report.get("skipped") or not isinstance(controls, dict) or not controls or not isinstance(top5, float):
        return False
    return top5 > max(float(value) for value in controls.values())


def _evaluate_indices(
    data: LabeledEpochs,
    train_indices: list[int],
    test_indices: list[int],
    vocab: list[str],
    ridge: float,
    embed_dim: int,
    *,
    brennan_semantic: bool = False,
    encoder_kind: str = "ridge",
    hidden_dim: int = 64,
    mlp_epochs: int = 120,
    mlp_lr: float = 1e-3,
) -> EvalReport:
    if not train_indices or not test_indices:
        raise RuntimeError("Train and test splits must both be non-empty")
    _, n_channels, n_times = data.epochs.shape
    if brennan_semantic:
        train_labels_for_vocab = [data.labels[index] for index in train_indices]
        embed_fn = build_brennan_tfidf_embedder(train_labels_for_vocab, dim=embed_dim)
        kwargs: dict = {"embed_fn": embed_fn}
    else:
        kwargs = {}
        embed_fn = None
    if encoder_kind == "mlp":
        encoder = MlpAnchorEncoder(
            n_channels=n_channels, n_times=n_times, embed_dim=embed_dim,
            hidden_dim=hidden_dim, seed=0, **kwargs,
        )
        encoder.fit(
            data.epochs[train_indices],
            [data.labels[index] for index in train_indices],
            ridge=ridge,
            epochs=mlp_epochs,
            lr=mlp_lr,
        )
    else:
        encoder = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=embed_dim, **kwargs)
        encoder.fit(
            data.epochs[train_indices],
            [data.labels[index] for index in train_indices],
            ridge=ridge,
        )
    return evaluate_retrieval(
        encoder.encode(data.epochs[test_indices]),
        [data.labels[index] for index in test_indices],
        vocab,
        **kwargs,
    )


def _evaluate_grouped(
    data: LabeledEpochs,
    vocab: list[str],
    ridge: float,
    embed_dim: int,
    strategy: SplitStrategy,
    *,
    brennan_semantic: bool = False,
    encoder_kind: str = "ridge",
    hidden_dim: int = 64,
    mlp_epochs: int = 120,
    mlp_lr: float = 1e-3,
) -> dict[str, float | dict[str, float]]:
    """Fit/evaluate every held-out group for one ADR-003 split strategy."""
    n_epochs, _, _ = data.epochs.shape
    ids = data.session_ids if strategy == "cross_session" else data.subject_ids
    split_kwargs = {"session_ids": ids} if strategy == "cross_session" else {"subject_ids": ids}
    # SplitConfig defaults to allow_random_fallback=False. State it here to make
    # the TRAIN pipeline's anti-leakage policy explicit and non-accidental.
    config = SplitConfig(strategy=strategy, allow_random_fallback=False)
    folds = list(make_splits(n_epochs, config=config, **split_kwargs))
    if not folds:
        raise RuntimeError(f"No {strategy} folds could be formed")

    reports = [
        _evaluate_indices(
            data, train, test, vocab, ridge, embed_dim,
            brennan_semantic=brennan_semantic,
            encoder_kind=encoder_kind, hidden_dim=hidden_dim,
            mlp_epochs=mlp_epochs, mlp_lr=mlp_lr,
        )
        for train, test in folds
    ]
    return _mean_report(reports)


def _brennan_stimulus_block_indices(
    data: LabeledEpochs,
    held_segments: set[int] | None = None,
    buffer_s: float = 1.0,
) -> tuple[list[int], list[int]]:
    """Return train/test indices for Brennan segment 10-12 holdout with a time buffer."""
    if data.metadata is None or len(data.metadata) != len(data.labels):
        raise RuntimeError("Brennan stimulus-block evaluation requires per-epoch metadata")
    held = held_segments or {10, 11, 12}
    test_indices: list[int] = []
    test_windows: dict[str, list[tuple[float, float]]] = {}
    for index, item in enumerate(data.metadata):
        segment = int(item["segment"])
        subject = str(item.get("subject_id") or data.subject_ids[index])
        onset = float(item["onset"])
        offset = float(item["offset"])
        if segment in held:
            test_indices.append(index)
            test_windows.setdefault(subject, []).append((onset, offset))

    train_indices: list[int] = []
    for index, item in enumerate(data.metadata):
        segment = int(item["segment"])
        if segment in held:
            continue
        subject = str(item.get("subject_id") or data.subject_ids[index])
        onset = float(item["onset"]) - buffer_s
        offset = float(item["offset"]) + buffer_s
        overlaps_held = any(
            onset < test_offset and offset > test_onset
            for test_onset, test_offset in test_windows.get(subject, [])
        )
        if not overlaps_held:
            train_indices.append(index)
    return train_indices, test_indices


def train_and_eval(data: LabeledEpochs, ridge: float = 1.0, embed_dim: int = 64) -> dict:
    """Evaluate cross-session and LOSO reports, then fit the deployable encoder.

    The final model is fit only after evaluation, on all supplied labels. Neither
    grouped evaluation path permits a random epoch fallback.
    """
    if data.epochs.ndim != 3:
        raise ValueError("epochs must have shape (n_epochs, n_channels, n_times)")
    n_epochs, n_channels, n_times = data.epochs.shape
    if n_epochs != len(data.labels) or n_epochs != len(data.subject_ids) or n_epochs != len(data.session_ids):
        raise ValueError("Epoch, label, subject, and session counts must match")
    if n_epochs < 2:
        raise ValueError("At least two epochs are required for grouped evaluation")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    vocab = _evaluation_vocab(data.labels)
    eval_k = operating_k(vocab)
    cross_session = _evaluate_grouped(data, vocab, ridge, embed_dim, "cross_session")
    loso = _evaluate_grouped(data, vocab, ridge, embed_dim, "loso")

    # The saved model must not be a model from an arbitrary held-out fold.
    encoder = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=embed_dim)
    encoder.fit(data.epochs, data.labels, ridge=ridge)

    controls = cross_session["controls"]
    assert isinstance(controls, dict)
    top5 = cross_session["top5"]
    assert isinstance(top5, float)
    return {
        "source": data.source,
        "data_lineage": data.data_lineage,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_times": n_times,
        # Retain concise top-level metrics for callers that predate the nested
        # evaluation report; they always refer to cross-session generalization.
        "n_folds": cross_session["n_folds"],
        "n_labels": len(vocab),
        "eval_k": eval_k,
        "top1": cross_session["top1"],
        "top5": top5,
        "top25": cross_session["top25"],
        "controls": controls,
        "cross_session": cross_session,
        "loso": loso,
        "gate_pass": top5 > max(controls.values()),
        "encoder": encoder,
    }


def train_and_eval_brennan(
    data: LabeledEpochs,
    ridge: float = 1.0,
    embed_dim: int = 64,
    *,
    encoder_kind: str = "ridge",
    hidden_dim: int = 64,
    mlp_epochs: int = 120,
    mlp_lr: float = 1e-3,
) -> dict:
    """Evaluate Brennan with LOSO and stimulus-block protocols only.

    This intentionally does not produce a ``cross_session`` key: Brennan Alice is
    one continuous recording per subject, so fabricated session folds are banned.
    """
    if data.epochs.ndim != 3:
        raise ValueError("epochs must have shape (n_epochs, n_channels, n_times)")
    n_epochs, n_channels, n_times = data.epochs.shape
    if n_epochs != len(data.labels) or n_epochs != len(data.subject_ids) or n_epochs != len(data.session_ids):
        raise ValueError("Epoch, label, subject, and session counts must match")
    if n_epochs < 1:
        raise ValueError("At least one epoch is required")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    vocab = _brennan_vocab(data)
    eval_k = operating_k(vocab)
    try:
        loso = _evaluate_grouped(
            data, vocab, ridge, embed_dim, "loso", brennan_semantic=True,
            encoder_kind=encoder_kind, hidden_dim=hidden_dim,
            mlp_epochs=mlp_epochs, mlp_lr=mlp_lr,
        )
    except (RuntimeError, ValueError) as exc:
        loso = _skipped_report(str(exc))

    try:
        train_indices, test_indices = _brennan_stimulus_block_indices(data)
        stimulus_report = _single_report(
            _evaluate_indices(
                data, train_indices, test_indices, vocab, ridge, embed_dim,
                brennan_semantic=True, encoder_kind=encoder_kind,
                hidden_dim=hidden_dim, mlp_epochs=mlp_epochs, mlp_lr=mlp_lr,
            )
        )
        stimulus_report["held_out_segments"] = [10, 11, 12]
        stimulus_report["temporal_buffer_s"] = 1.0
        stimulus_report["n_train_epochs"] = len(train_indices)
        stimulus_report["n_test_epochs"] = len(test_indices)
        stimulus_report["n_train_labels"] = len({data.labels[index] for index in train_indices})
        stimulus_report["n_test_labels"] = len({data.labels[index] for index in test_indices})
    except (RuntimeError, ValueError) as exc:
        stimulus_report = _skipped_report(str(exc))
        stimulus_report["held_out_segments"] = [10, 11, 12]
        stimulus_report["temporal_buffer_s"] = 1.0

    # Prefer the unseen-story-time protocol as the concise top-level headline;
    # fall back to LOSO for tiny smoke runs that lack held-out segments.
    headline_name = "stimulus_block" if not stimulus_report.get("skipped") else "loso_same_stimulus"
    headline = stimulus_report if headline_name == "stimulus_block" else loso

    final_embed_fn = build_brennan_tfidf_embedder(data.labels, dim=embed_dim)
    if encoder_kind == "mlp":
        encoder: AnchorEncoder | MlpAnchorEncoder = MlpAnchorEncoder(
            n_channels=n_channels, n_times=n_times, embed_dim=embed_dim,
            hidden_dim=hidden_dim, seed=0, embed_fn=final_embed_fn,
        )
        encoder.fit(data.epochs, data.labels, ridge=ridge, epochs=mlp_epochs, lr=mlp_lr)
    else:
        encoder = AnchorEncoder(
            n_channels=n_channels, n_times=n_times, embed_dim=embed_dim, embed_fn=final_embed_fn
        )
        encoder.fit(data.epochs, data.labels, ridge=ridge)

    gate_pass = _passes_controls(loso) and _passes_controls(stimulus_report)
    return {
        "source": data.source,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_times": n_times,
        "n_labels": len(vocab),
        "eval_k": eval_k,
        "headline_protocol": headline_name,
        "top1": headline["top1"],
        "top5": headline["top5"],
        "top25": headline["top25"],
        "controls": headline["controls"],
        "loso_same_stimulus": loso,
        "stimulus_block": stimulus_report,
        "gate_pass": gate_pass,
        "encoder": encoder,
    }


def _artifact_source_csv(result: dict) -> Path | None:
    """Resolve the source file for lineage without inventing a dataset path."""
    explicit = result.get("source_file")
    if explicit:
        return Path(str(explicit))
    if result.get("source") == "bciciv2a:mi4":
        from src.training.dataset import BCICIV2A_CSV

        return BCICIV2A_CSV
    return None


def _subject_counts(result: dict) -> dict[str, int]:
    subjects = result.get("subject_ids")
    if isinstance(subjects, list):
        return dict(Counter(str(subject) for subject in subjects))
    return {str(key): int(value) for key, value in (result.get("subject_counts") or {}).items()}


def save_artifacts(result: dict, out_dir: Path = ARTIFACTS) -> Path | None:
    """Persist lineage for every run and a deployable model only after the gate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "encoder"}
    deployable = bool(result["gate_pass"])
    summary["deployment_status"] = "validated" if deployable else "blocked_below_control"
    summary["model_artifact"] = "encoder.npz" if deployable else None
    (out_dir / "train_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Every BCICIV-2a run gets a signed lineage manifest. Signing is enabled by
    # an explicit PEM private-key path; without it, lineage remains unsigned and
    # the report records that no deployable artifact was published.
    source_csv = _artifact_source_csv(result)
    if source_csv is not None:
        from scripts.artifact_manifest import build_manifest, write_signed_manifest

        manifest = build_manifest(
            artifact_dir=out_dir,
            source_csv=source_csv,
            result={**result, "deployment_status": summary["deployment_status"], "model_artifact": summary["model_artifact"]},
            subject_counts=_subject_counts(result),
        )
        signing_key_path = os.environ.get("THINKING_ARTIFACT_SIGNING_KEY")
        if signing_key_path:
            key = Path(signing_key_path).read_bytes()
            write_signed_manifest(manifest, out_dir / "artifact_manifest.json", key)
        else:
            (out_dir / "artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if not deployable:
        return None
    encoder: AnchorEncoder = result["encoder"]
    model_path = out_dir / "encoder.npz"
    encoder.save(model_path)
    return model_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train THINKING semantic-anchor encoder")
    parser.add_argument("--dataset", choices=["d05", "cofett", "brennan", "d01", "chisco", "d09", "bciciv2a", "sparrkulee"], default="d05")
    parser.add_argument("--max-rows", type=int, default=240)
    parser.add_argument("--n-times", type=int, default=64)
    parser.add_argument("--n-channels", type=int, default=32)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--label-field", choices=["topic", "selected_topic", "word"], default="topic")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--cohort", choices=["primary", "analyzed", "events_available"], default="primary")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--no-autoreject", action="store_true")
    parser.add_argument("--no-ica", action="store_true")
    parser.add_argument("--encoder", choices=["ridge", "mlp"], default="ridge")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--data-source", choices=["local", "huggingface"], default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cache-mode", choices=["verified", "stream"], default="verified")
    parser.add_argument("--catalog-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.revision and args.revision.lower() in {"main", "master", "latest", "head", "default"}:
        raise ValueError("dataset revision must be immutable, not a moving revision")
    if args.data_source == "huggingface" and (
        not args.revision or args.revision.lower() in {"main", "master", "latest", "head", "default"}
    ):
        raise ValueError("dataset revision must be immutable, not a moving revision")

    dataset_source = None
    if args.data_source:
        if args.dataset != "bciciv2a":
            raise ValueError("verified dataset sources are currently implemented for bciciv2a")
        from src.data.manifest import DatasetManifest
        from src.data.resolve import resolve_dataset_source

        manifest = DatasetManifest.from_json(args.manifest) if args.manifest else None
        dataset_source = resolve_dataset_source(
            dataset_id=args.dataset,
            provider=args.data_source,
            catalog_dir=args.catalog_dir,
            manifest=manifest,
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.cache_mode == "verified",
        )

    if args.dataset == "d05":
        data = load_d05(max_rows=args.max_rows, n_times=args.n_times, shard=args.shard, label_field=args.label_field)
        result = train_and_eval(data, ridge=args.ridge, embed_dim=args.embed_dim)
    elif args.dataset in {"d01", "chisco", "d09", "bciciv2a", "sparrkulee"}:
        load_kwargs: dict[str, object] = {
            "n_times": args.n_times,
            "n_channels": args.n_channels,
            "max_epochs": args.max_rows,
            "use_autoreject": not args.no_autoreject,
            "use_ica": not args.no_ica,
        }
        if dataset_source is not None:
            load_kwargs["dataset_source"] = dataset_source
        data = load_labeled(args.dataset, **load_kwargs)
        result = train_and_eval(data, ridge=args.ridge, embed_dim=args.embed_dim)
    elif args.dataset == "cofett":
        try:
            data = load_labeled(
                "cofett",
                max_files=args.max_files,
                n_times=args.n_times,
                n_channels=args.n_channels,
                max_epochs=args.max_rows,
                use_autoreject=not args.no_autoreject,
                use_ica=not args.no_ica,
            )
        except RuntimeError as exc:
            from src.training.dataset import load_cofett_events

            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            events = load_cofett_events(max_files=8)
            (ARTIFACTS / "cofett_events.json").write_text(
                json.dumps({"n_events": len(events), "reason": str(exc), "sample": events[:8]}, indent=2),
                encoding="utf-8",
            )
            print(f"COFETT EDF skipped ({exc}); events metadata saved; falling back to D05")
            data = load_d05(max_rows=args.max_rows, n_times=args.n_times, shard=args.shard, label_field=args.label_field)
        result = train_and_eval(data, ridge=args.ridge, embed_dim=args.embed_dim)
    else:
        data = load_brennan_epochs(
            cohort=args.cohort,
            n_times=args.n_times,
            n_channels=args.n_channels,
            max_subjects=args.max_subjects,
            max_sentences=args.max_sentences,
            use_autoreject=not args.no_autoreject,
            use_ica=not args.no_ica,
        )
        result = train_and_eval_brennan(
            data, ridge=args.ridge, embed_dim=args.embed_dim,
            encoder_kind=args.encoder, hidden_dim=args.hidden_dim,
            mlp_epochs=args.mlp_epochs, mlp_lr=args.mlp_lr,
        )

    model_path = save_artifacts(result, args.artifacts_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "encoder"}, indent=2))
    if model_path:
        print(f"saved validated model {model_path}")
    else:
        print("model deployment blocked: Top-5 did not exceed all controls")
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
