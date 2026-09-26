"""Teacher-forcing-free EEG-to-text smoke evaluation.

This is deliberately a semantic-anchor evaluation, not a claim of verbatim thought decoding.
It verifies held-out labels are recovered via EEG retrieval before deterministic generation.
"""
from __future__ import annotations

import numpy as np

from src.anchors.encoder import AnchorEncoder, label_template
from src.anchors.retrieval import retrieve_anchors
from src.anchors.vocab import load_vocab
from src.prompt.generator import generate_text


def main() -> None:
    vocab = load_vocab().keywords
    labels = [vocab[i % 5] for i in range(50)]
    rng = np.random.default_rng(21)
    epochs = np.stack([label_template(label, 8, 20) + 0.08 * rng.standard_normal((8, 20)) for label in labels]).astype(np.float32)
    train, test = np.arange(0, 40), np.arange(40, 50)
    encoder = AnchorEncoder(8, 20, 64).fit(epochs[train], [labels[i] for i in train], ridge=0.1)

    hits = 0
    outputs: list[str] = []
    for idx in test:
        emb = encoder.encode(epochs[idx : idx + 1])[0]
        retrieved = retrieve_anchors(emb, vocab, top_k=5, epoch_id=f"eeg:heldout_{idx}")
        surfaces = [r.surface for r in retrieved]
        hits += labels[idx] in surfaces
        prompt = {
            "subject": {"label": "semantic topic"},
            "attributes": [],
        }
        outputs.append(generate_text(prompt, [r.__dict__ for r in retrieved], use_llm=False))
    top5 = hits / len(test)
    print(f"Held-out semantic Top-5: {top5:.3f} ({hits}/{len(test)})")
    print(f"Sample output: {outputs[0]}")
    if top5 <= 0.5:
        raise SystemExit("FAIL: held-out semantic retrieval is not above the smoke-test threshold")
    print("PASS: teacher-forcing-free held-out semantic retrieval gate.")


if __name__ == "__main__":
    main()
