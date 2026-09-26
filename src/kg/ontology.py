"""Ontology loader — merges interior-v1.0 + text-v0.1 extension."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

THINKING_ONT = Path("E:/AI_thucchien/THINKING/architecture/knowledge_graph/ontology_seed.json")
WORKTREE_ONT = Path(__file__).resolve().parents[2] / "architecture" / "knowledge_graph" / "ontology_seed.json"

TEXT_EXTENSION = {
    "ontology_version": "text-v0.1",
    "nodes": [
        {"id": "text:Keyword", "type": "Concept", "label": "keyword"},
        {"id": "text:Topic", "type": "Concept", "label": "topic"},
        {"id": "text:Intent", "type": "Concept", "label": "intent"},
        {"id": "text:Reading", "type": "Paradigm", "label": "reading"},
        {"id": "text:InnerRecall", "type": "Paradigm", "label": "inner recall"},
    ],
    "edges": [
        {"id": "edge:keyword_topic", "source": "text:Keyword", "relation": "RELATED_TO", "target": "text:Topic", "confidence": 0.7, "provenance": "text_v01:keyword_topic"},
        {"id": "edge:topic_intent", "source": "text:Topic", "relation": "RELATED_TO", "target": "text:Intent", "confidence": 0.6, "provenance": "text_v01:topic_intent"},
    ],
    "synonyms": [
        {"surface": "recall", "concept_id": "text:InnerRecall"},
        {"surface": "reading", "concept_id": "text:Reading"},
        {"surface": "keyword", "concept_id": "text:Keyword"},
        {"surface": "topic", "concept_id": "text:Topic"},
    ],
}

@dataclass
class Ontology:
    ontology_version: str
    nodes: list[dict]
    edges: list[dict]
    synonyms: list[dict]
    node_ids: set[str]

def load_ontology(with_text_extension: bool = True) -> Ontology:
    p = WORKTREE_ONT if WORKTREE_ONT.exists() else THINKING_ONT
    base = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"ontology_version": "interior-v1.0", "nodes": [], "edges": [], "synonyms": []}
    if with_text_extension:
        # merge
        existing_ids = {n["id"] for n in base["nodes"]}
        for n in TEXT_EXTENSION["nodes"]:
            if n["id"] not in existing_ids:
                base["nodes"].append(n)
        existing_eids = {e["id"] for e in base["edges"]}
        for e in TEXT_EXTENSION["edges"]:
            if e["id"] not in existing_eids:
                base["edges"].append(e)
        base["synonyms"].extend(TEXT_EXTENSION["synonyms"])
        base["ontology_version"] = TEXT_EXTENSION["ontology_version"]
    return Ontology(
        ontology_version=base["ontology_version"],
        nodes=base["nodes"],
        edges=base["edges"],
        synonyms=base["synonyms"],
        node_ids={n["id"] for n in base["nodes"]},
    )
