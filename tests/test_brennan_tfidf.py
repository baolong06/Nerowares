import numpy as np


def test_brennan_tfidf_embedder_exists_and_fitted_on_train_only():
    """RED: TFIDF embedder must be fitted on train sentences only."""
    from src.anchors.retrieval import build_brennan_tfidf_embedder
    from src.training.dataset import _load_brennan_protocol

    protocol = _load_brennan_protocol()
    by_id = {item["sentence_id"]: item["text"] for item in protocol["task"]["sentences"]}
    train_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["train_sentence_ids"])
    test_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["test_sentence_ids"])
    train_texts = [by_id[i] for i in train_ids]
    test_texts = [by_id[i] for i in test_ids]

    embed_fn = build_brennan_tfidf_embedder(train_texts, dim=64)
    # Must be callable (text, dim) -> vec
    vec = embed_fn(train_texts[0], 64)
    assert vec.shape == (64,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5

    # Held-out sentences should share moderate similarity, not 0 (hash) nor 0.66 (char hash)
    train_embs = np.stack([embed_fn(t, 64) for t in train_texts])
    test_embs = np.stack([embed_fn(t, 64) for t in test_texts])
    cos = test_embs @ train_embs.T
    # Semantic TFIDF should give non-trivial but discriminative overlap
    assert float(cos.max()) > 0.10, f"max cosine {cos.max()} too low"
    assert float(cos.mean()) > 0.02, f"mean cosine {cos.mean()} too low"
    # Must not be overly collapsed like char_wb (mean 0.66)
    assert float(cos.mean()) < 0.50, f"mean cosine {cos.mean()} too high (collapsed)"


def test_brennan_tfidf_encoder_transfer_synthetic():
    """RED: Ridge on TFIDF should transfer to held-out story sentences."""
    from src.anchors.encoder import AnchorEncoder
    from src.anchors.eval import evaluate_retrieval
    from src.anchors.retrieval import build_brennan_tfidf_embedder, text_embed
    from src.training.dataset import _load_brennan_protocol

    protocol = _load_brennan_protocol()
    by_id = {item["sentence_id"]: item["text"] for item in protocol["task"]["sentences"]}
    train_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["train_sentence_ids"])
    test_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["test_sentence_ids"])
    rng = np.random.default_rng(0)
    n_channels, n_times, dim = 8, 16, 64
    all_ids = train_ids + test_ids
    all_texts = [by_id[i] for i in all_ids]
    train_texts = [by_id[i] for i in train_ids]

    embed_fn_sem = build_brennan_tfidf_embedder(train_texts, dim=dim)
    sem = np.stack([embed_fn_sem(t, dim=dim) for t in all_texts])
    proj = rng.standard_normal((dim, n_channels * n_times)).astype(np.float32) * 0.5
    eeg = (sem @ proj).reshape(len(all_ids), n_channels, n_times)
    eeg = eeg + 0.05 * rng.standard_normal(eeg.shape).astype(np.float32)

    n_train = len(train_ids)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(all_ids)))
    vocab = [by_id[i] for i in sorted(set(train_ids + test_ids))]

    enc_sem = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=dim,
                            embed_fn=embed_fn_sem).fit(
        eeg[train_idx], [all_texts[i] for i in train_idx], ridge=1.0)
    sem_report = evaluate_retrieval(enc_sem.encode(eeg[test_idx]),
                                    [all_texts[i] for i in test_idx],
                                    vocab, embed_fn=embed_fn_sem)

    enc_hash = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=dim,
                             embed_fn=text_embed).fit(
        eeg[train_idx], [all_texts[i] for i in train_idx], ridge=1.0)
    hash_report = evaluate_retrieval(enc_hash.encode(eeg[test_idx]),
                                     [all_texts[i] for i in test_idx],
                                     vocab, embed_fn=text_embed)

    assert sem_report.top5 > hash_report.top5
    assert sem_report.top5 > 0.30
