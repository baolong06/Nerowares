"""Anchor -> concept resolver — confidence threshold + provenance separation.

Accepted concepts MUST carry eeg:* provenance only. Suggested concepts carry
kg:edge:<edge_id_without_edge_prefix> and must never be presented as EEG-read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import hashlib
import math

from .ontology import Ontology

_REL_TO_OP = {
    "INCREASES": "increase",
    "DECREASES": "decrease",
    "CONFLICTS_WITH": "avoid",
    "RELATED_TO": "set",
    "PRESERVES": "preserve",
}


@dataclass
class ResolvedConcept:
    concept_id: str
    label: str
    confidence: float
    provenance: list[str]
    status: str  # accepted | suggested
    operator: str = "set"


@dataclass
class KGSubgraph:
    subgraph_id: str
    nodes: list[dict]
    edges: list[dict]
    resolved: list[ResolvedConcept]
    ontology_version: str = "text-v0.1"


def _sg_id(anchors: list[str]) -> str:
    h = hashlib.sha256("|".join(sorted(anchors)).encode()).hexdigest()[:8]
    return f"sg:{h}"


def _eeg_only(prov: object) -> list[str]:
    items = prov if isinstance(prov, list) else [prov]
    cleaned = [p for p in items if isinstance(p, str) and p.startswith("eeg:") and p.strip()]
    if not cleaned:
        raise ValueError("accepted anchors require eeg:* provenance")
    return cleaned


def _kg_edge_prov(edge_id: str) -> str:
    suffix = edge_id[5:] if edge_id.startswith("edge:") else edge_id
    return f"kg:edge:{suffix}"


def resolve_anchors(anchors: list[dict], ontology: Ontology, threshold: float = 0.5) -> KGSubgraph:
    syn_map = {s["surface"].lower(): s["concept_id"] for s in ontology.synonyms}
    node_label = {n["id"]: n.get("label", n["id"]) for n in ontology.nodes}
    for n in ontology.nodes:
        lab = str(n.get("label", "")).lower().strip()
        if lab and lab not in syn_map and n["id"] in ontology.node_ids:
            syn_map[lab] = n["id"]

    resolved: list[ResolvedConcept] = []
    involved_ids: set[str] = set()

    for a in anchors:
        surf = a.get("surface", "").lower().strip()
        score = float(a.get("score", 0))
        if not math.isfinite(score):
            raise ValueError("anchor score must be finite")
        if not surf or score < threshold:
            continue
        prov = _eeg_only(a.get("provenance", ["eeg:unknown"]))
        cid = syn_map.get(surf)
        if cid and cid in ontology.node_ids:
            resolved.append(
                ResolvedConcept(
                    concept_id=cid,
                    label=node_label.get(cid, cid),
                    confidence=score,
                    provenance=list(prov),
                    status="accepted",
                    operator="set",
                )
            )
            involved_ids.add(cid)
        else:
            resolved.append(
                ResolvedConcept(
                    concept_id="text:Keyword",
                    label=surf,
                    confidence=score * 0.8,
                    provenance=list(prov),
                    status="accepted",
                    operator="set",
                )
            )
            involved_ids.add("text:Keyword")

    suggested: list[ResolvedConcept] = []
    subgraph_edges: list[dict] = []
    for e in ontology.edges:
        if e["source"] not in involved_ids and e["target"] not in involved_ids:
            continue
        subgraph_edges.append(e)
        other = e["target"] if e["source"] in involved_ids else e["source"]
        if other in involved_ids:
            continue
        if any(r.concept_id == other for r in resolved + suggested):
            continue
        suggested.append(
            ResolvedConcept(
                concept_id=other,
                label=node_label.get(other, other),
                confidence=float(e.get("confidence", 0.5)) * 0.7,
                provenance=[_kg_edge_prov(e["id"])],
                status="suggested",
                operator=_REL_TO_OP.get(e.get("relation", ""), "set"),
            )
        )

    all_resolved = resolved + suggested
    involved = {r.concept_id for r in all_resolved} | involved_ids
    subgraph_nodes = [n for n in ontology.nodes if n["id"] in involved]
    return KGSubgraph(
        subgraph_id=_sg_id([a.get("surface", "") for a in anchors]),
        nodes=subgraph_nodes,
        edges=subgraph_edges,
        resolved=all_resolved,
        ontology_version=ontology.ontology_version,
    )
