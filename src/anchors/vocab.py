"""Anchor vocabulary — fixed keyword allowlist (APP-003)."""
from __future__ import annotations
import json, re
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parents[2] / "architecture" / "knowledge_graph" / "ontology_seed.json"
_ALT = Path("E:/AI_thucchien/THINKING/architecture/knowledge_graph/ontology_seed.json")

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9 _-]")
ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9 _-]{1,64}$")

class AnchorVocab:
    def __init__(self, keywords: list[str]):
        seen = set()
        self.keywords: list[str] = []
        for k in keywords:
            kk = k.strip().lower()
            if kk and kk not in seen:
                seen.add(kk)
                self.keywords.append(kk)
        self._set = set(self.keywords)

    def contains(self, surface: str) -> bool:
        return surface.strip().lower() in self._set

    def sanitize(self, surface: str) -> str:
        return _SANITIZE_RE.sub("", surface).strip()[:64]

    def validate_or_raise(self, surface: str) -> str:
        s = self.sanitize(surface)
        if not ALLOWED_PATTERN.match(s):
            raise ValueError(f"Anchor surface violates allowlist pattern: {surface!r} -> {s!r}")
        if s.lower() not in self._set:
            raise ValueError(f"Anchor not in vocab: {surface!r} (sanitized: {s!r})")
        return s

    def __len__(self): return len(self.keywords)

DEFAULT_KEYWORDS = [
    "open", "airy", "sunlight", "minimalist", "modern", "classical",
    "window", "living room", "bedroom", "office", "natural light", "openness",
    "decoration density", "minimal", "increase", "decrease", "preserve",
    "focus", "memory", "recall", "reading", "imagery", "rest", "intent", "topic",
]

def load_vocab(path: Path | None = None) -> AnchorVocab:
    keywords = list(DEFAULT_KEYWORDS)
    ontology_path = path or (VOCAB_PATH if VOCAB_PATH.exists() else (_ALT if _ALT.exists() else None))
    if ontology_path and ontology_path.exists():
        try:
            data = json.loads(ontology_path.read_text(encoding="utf-8"))
            incoming: list[str] = []
            for syn in data.get("synonyms", []) or []:
                if syn.get("surface"):
                    incoming.append(str(syn["surface"]))
            for node in data.get("nodes", []) or []:
                if node.get("label"):
                    incoming.append(str(node["label"]))
            # Ingest only surfaces that already match the allowlist. Do not keep
            # a stripped remnant of an invalid label (e.g. "<script>" -> "script").
            for s in incoming:
                raw = str(s).strip()
                if ALLOWED_PATTERN.match(raw):
                    keywords.append(raw)
        except Exception:
            pass
    return AnchorVocab(keywords)
