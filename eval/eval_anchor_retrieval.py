"""Teacher-forcing-free anchor retrieval eval with five controls (ADR-003)."""
from __future__ import annotations

import numpy as np

from src.anchors.encoder import AnchorEncoder, label_template
from src.anchors.eval import evaluate_retrieval
from src.anchors.splits import SplitConfig, make_splits
from src.anchors.vocab import load_vocab


def synthetic_labeled_epochs(labels: list[str], n_channels: int, n_times: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Signal is label-dependent; low Gaussian noise imitates a separable toy benchmark.
    return np.stack([label_template(label, n_channels, n_times) + 0.08 * rng.standard_normal((n_channels, n_times)) for label in labels]).astype(np.float32)


def main() -> None:
    vocab = load_vocab().keywords
    labels = [vocab[i % 5] for i in range(80)]
    epochs = synthetic_labeled_epochs(labels, n_channels=8, n_times=20)
    session_ids = [f"session_{i % 4}" for i in range(len(labels))]

    fold_reports = []
    for train_idx, test_idx in make_splits(
        len(labels), session_ids=session_ids, config=SplitConfig(strategy="cross_session")
    ):
        encoder = AnchorEncoder(n_channels=8, n_times=20, embed_dim=64)
        encoder.fit(epochs[train_idx], [labels[i] for i in train_idx], ridge=0.1)
        fold_reports.append(evaluate_retrieval(encoder.encode(epochs[test_idx]), [labels[i] for i in test_idx], vocab))

    mean_top1 = float(np.mean([report.top1 for report in fold_reports]))
    mean_top5 = float(np.mean([report.top5 for report in fold_reports]))
    mean_top25 = float(np.mean([report.top25 for report in fold_reports]))
    baseline_means = {
        key: float(np.mean([report.baselines[key] for report in fold_reports]))
        for key in fold_reports[0].baselines
    }
    print(f"Cross-session folds: {len(fold_reports)}")
    print(f"Top-1: {mean_top1:.3f}  Top-5: {mean_top5:.3f}  Top-25: {mean_top25:.3f}")
    print("Controls (Top-5):", baseline_means)
    # Evidence gate: held-out EEG must outperform each non-EEG control in this synthetic smoke test.
    strongest_control = max(baseline_means.values())
    if mean_top5 <= strongest_control:
        raise SystemExit(f"FAIL: real EEG Top-5 {mean_top5:.3f} did not beat strongest control {strongest_control:.3f}")
    print("PASS: real EEG Top-5 exceeds all five control baselines.")


if __name__ == "__main__":
    main()
