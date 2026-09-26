import numpy as np
import pytest

def test_brennan_semantic_embedding_exists_and_generalizes_to_heldout_sentences():
    """RED: Brennan block 10/11/12 must share semantic similarity with train sentences.

    With hash embeddings the block is random (mean cosine ~ 0) and ridge cannot
    extrapolate. A semantic embedder must give held-out sentences non-trivial
    cosine to some train sentence.
    """
    from src.anchors.retrieval import brennan_text_embed
    from src.training.dataset import _load_brennan_protocol

    protocol = _load_brennan_protocol()
    train_ids = set(protocol["splits"]["protocol_B_stimulus_block"]["train_sentence_ids"])
    test_ids = set(protocol["splits"]["protocol_B_stimulus_block"]["test_sentence_ids"])
    by_id = {item["sentence_id"]: item["text"] for item in protocol["task"]["sentences"]}

    train_texts = [by_id[i] for i in sorted(train_ids)]
    test_texts = [by_id[i] for i in sorted(test_ids)]

    train_embs = np.stack([brennan_text_embed(t, dim=64) for t in train_texts])
    test_embs = np.stack([brennan_text_embed(t, dim=64) for t in test_texts])

    # Embeddings must be L2-normalized
    for emb in list(train_embs) + list(test_embs):
        assert emb.shape == (64,)
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    # Each held-out sentence must have some train neighbor with cosine > 0.15
    # Hash embeddings fail this; TFIDF/semantic passes.
    cos = test_embs @ train_embs.T
    assert float(cos.max()) > 0.15
    assert float(cos.mean()) > 0.05


def test_brennan_semantic_encoder_beats_hash_on_synthetic_semantic_eeg():
    """RED: Ridge on Brennan semantic embeddings must beat hash on transfer.

    Construct EEG = label_template on semantic TFIDF cluster label so that
    held-out sentences from the same story share EEG subspace. Hash cannot
    transfer; semantic can.
    """
    from src.anchors.encoder import AnchorEncoder
    from src.anchors.retrieval import brennan_text_embed, text_embed
    from src.anchors.eval import evaluate_retrieval
    from src.training.dataset import _load_brennan_protocol

    protocol = _load_brennan_protocol()
    by_id = {item["sentence_id"]: item["text"] for item in protocol["task"]["sentences"]}
    train_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["train_sentence_ids"])
    test_ids = sorted(protocol["splits"]["protocol_B_stimulus_block"]["test_sentence_ids"])

    # Build synthetic EEG where the signal follows semantic embedding, not hash
    rng = np.random.default_rng(0)
    n_channels, n_times, dim = 8, 16, 64
    all_ids = train_ids + test_ids
    all_texts = [by_id[i] for i in all_ids]
    # Project semantic embedding to EEG space via a fixed random matrix
    sem = np.stack([brennan_text_embed(t, dim=dim) for t in all_texts])
    proj_sem_to_eeg = rng.standard_normal((dim, n_channels * n_times)).astype(np.float32) * 0.5
    eeg = (sem @ proj_sem_to_eeg).reshape(len(all_ids), n_channels, n_times)
    eeg = eeg + 0.05 * rng.standard_normal(eeg.shape).astype(np.float32)

    train_idx = list(range(len(train_ids)))
    test_idx = list(range(len(train_ids), len(all_ids)))
    vocab = [by_id[i] for i in sorted(set(train_ids + test_ids))]

    # Semantic encoder trained on train should retrieve held-out test sentences
    enc_sem = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=dim,
                            embed_fn=brennan_text_embed).fit(
        eeg[train_idx], [all_texts[i] for i in train_idx], ridge=1.0)
    sem_report = evaluate_retrieval(
        enc_sem.encode(eeg[test_idx]),
        [all_texts[i] for i in test_idx],
        vocab, embed_fn=brennan_text_embed)

    # Hash encoder on the same EEG cannot
    enc_hash = AnchorEncoder(n_channels=n_channels, n_times=n_times, embed_dim=dim,
                             embed_fn=text_embed).fit(
        eeg[train_idx], [all_texts[i] for i in train_idx], ridge=1.0)
    hash_report = evaluate_retrieval(
        enc_hash.encode(eeg[test_idx]),
        [all_texts[i] for i in test_idx],
        vocab, embed_fn=text_embed)

    assert sem_report.top5 > hash_report.top5
    assert sem_report.top5 > 0.3
