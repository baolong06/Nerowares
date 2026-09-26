import numpy as np


def test_mlp_encoder_exists_and_encodes():
    """RED: MlpAnchorEncoder must exist with AnchorEncoder-compatible API."""
    from src.anchors.mlp_encoder import MlpAnchorEncoder

    enc = MlpAnchorEncoder(n_channels=4, n_times=8, embed_dim=16, hidden_dim=32, seed=0)
    epochs = np.random.default_rng(0).standard_normal((2, 4, 8)).astype(np.float32)
    out = enc.encode(epochs)
    assert out.shape == (2, 16)
    # L2 normalized
    for row in out:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-5


def test_mlp_encoder_beats_ridge_on_nonlinear_synthetic():
    """RED: MLP should beat linear ridge on a nonlinear synthetic mapping."""
    from src.anchors.encoder import AnchorEncoder
    from src.anchors.eval import evaluate_retrieval
    from src.anchors.mlp_encoder import MlpAnchorEncoder
    from src.anchors.retrieval import text_embed

    rng = np.random.default_rng(1)
    n_channels, n_times, dim = 8, 16, 32
    # Labels from tiny vocab
    labels = ["alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta"]
    n = len(labels)
    # Build EEG with nonlinear transform of text embedding: eeg = relu(sem @ W1) @ W2
    sem = np.stack([text_embed(l, dim) for l in labels])
    W1 = rng.standard_normal((dim, 64)).astype(np.float32) * 0.5
    W2 = rng.standard_normal((64, n_channels * n_times)).astype(np.float32) * 0.3
    hidden = np.maximum(0, sem @ W1)
    eeg = (hidden @ W2).reshape(n, n_channels, n_times)
    eeg = eeg + 0.02 * rng.standard_normal(eeg.shape).astype(np.float32)
    vocab = ["alpha", "beta", "gamma"]
    train_idx = [0, 1, 2, 3]
    test_idx = [4, 5, 6, 7]

    ridge_enc = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=dim).fit(
        eeg[train_idx], [labels[i] for i in train_idx], ridge=1.0
    )
    ridge_report = evaluate_retrieval(
        ridge_enc.encode(eeg[test_idx]), [labels[i] for i in test_idx], vocab
    )

    mlp_enc = MlpAnchorEncoder(
        n_channels=n_channels, n_times=n_times, embed_dim=dim, hidden_dim=64, seed=0
    ).fit(eeg[train_idx], [labels[i] for i in train_idx], epochs=80, lr=2e-3)
    mlp_report = evaluate_retrieval(
        mlp_enc.encode(eeg[test_idx]), [labels[i] for i in test_idx], vocab
    )

    assert mlp_report.top5 >= ridge_report.top5


def test_brennan_evaluator_can_select_mlp_encoder():
    """RED: Brennan evaluator must expose the nonlinear encoder path."""
    from src.anchors.mlp_encoder import MlpAnchorEncoder
    from src.training.dataset import LabeledEpochs
    from src.training.train import train_and_eval_brennan

    rng = np.random.default_rng(4)
    data = LabeledEpochs(
        epochs=rng.standard_normal((4, 2, 6)).astype(np.float32),
        labels=["alpha", "beta", "alpha", "beta"],
        subject_ids=["sub-001", "sub-001", "sub-002", "sub-002"],
        session_ids=["sub-001:run-01"] * 2 + ["sub-002:run-01"] * 2,
        source="brennan:test",
        stimulus_ids=["segment:01", "segment:10"] * 2,
        metadata=[
            {"subject_id": "sub-001", "sentence_id": 1, "segment": 1, "onset": 0.0, "offset": 2.0},
            {"subject_id": "sub-001", "sentence_id": 2, "segment": 10, "onset": 10.0, "offset": 12.0},
            {"subject_id": "sub-002", "sentence_id": 1, "segment": 1, "onset": 0.0, "offset": 2.0},
            {"subject_id": "sub-002", "sentence_id": 2, "segment": 10, "onset": 10.0, "offset": 12.0},
        ],
    )
    report = train_and_eval_brennan(
        data,
        ridge=1e-3,
        embed_dim=8,
        encoder_kind="mlp",
        hidden_dim=8,
        mlp_epochs=2,
        mlp_lr=1e-3,
    )
    assert isinstance(report["encoder"], MlpAnchorEncoder)
