"""Anchors — semantic anchor retrieval (T08 Brain-CLIPLM Stage-1 style)."""
from .vocab import AnchorVocab, load_vocab
from .encoder import AnchorEncoder, encode_epochs, label_template
from .mlp_encoder import MlpAnchorEncoder
from .retrieval import retrieve_anchors, RetrievalResult, text_embed
from .splits import make_splits, SplitConfig
from .eval import evaluate_retrieval, EvalReport

__all__ = ["AnchorVocab", "load_vocab", "AnchorEncoder", "MlpAnchorEncoder", "encode_epochs", "retrieve_anchors", "RetrievalResult", "make_splits", "SplitConfig", "evaluate_retrieval", "EvalReport"]
