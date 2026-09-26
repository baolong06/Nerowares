"""KG — ontology grounding with text-v0.1 (Pha 3)."""
from .ontology import load_ontology, Ontology
from .resolver import resolve_anchors, ResolvedConcept, KGSubgraph

__all__ = ["load_ontology", "Ontology", "resolve_anchors", "ResolvedConcept", "KGSubgraph"]
