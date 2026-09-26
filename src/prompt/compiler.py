"""Prompt compiler — anchors + KG subgraph -> structured_prompt.schema.json v1.0"""
from __future__ import annotations

from src.kg.resolver import KGSubgraph


def compile_prompt(
    anchors: list[dict],
    subgraph: KGSubgraph,
    modality: str = "text",
    intent_value: str = "decode",
    subject_fallback: str = "text:Topic",
) -> dict:
    sorted_anchors = sorted(anchors, key=lambda a: float(a.get("score", 0)), reverse=True)
    top = sorted_anchors[0] if sorted_anchors else {"score": 0.5, "provenance": ["eeg:epoch_unknown"]}
    intent = {
        "value": intent_value,
        "confidence": float(top.get("score", 0.5)),
        "provenance": list(top.get("provenance", ["eeg:epoch_unknown"])),
    }

    accepted = [r for r in subgraph.resolved if r.status == "accepted"]
    if accepted:
        subj = accepted[0]
        subject = {
            "concept_id": subj.concept_id,
            "label": subj.label,
            "confidence": subj.confidence,
            "provenance": list(subj.provenance),
        }
    else:
        subject = {
            "concept_id": subject_fallback,
            "label": "topic",
            "confidence": 0.5,
            "provenance": ["kg:fallback"],
        }

    attributes = []
    for r in subgraph.resolved:
        if r.concept_id == subject["concept_id"] and r.status == "accepted":
            continue
        attributes.append(
            {
                "concept_id": r.concept_id if ":" in r.concept_id else "text:Keyword",
                "operator": getattr(r, "operator", "set") or "set",
                "value": r.label,
                "confidence": r.confidence,
                "status": r.status,
                "provenance": list(r.provenance),
            }
        )

    required: list[str] = []
    forbidden: list[str] = []
    for e in subgraph.edges:
        if e.get("relation") == "CONFLICTS_WITH":
            forbidden.append(f"avoid {e.get('target')}")

    return {
        "schema_version": "1.0",
        "modality": modality,
        "intent": intent,
        "subject": subject,
        "attributes": attributes,
        "constraints": {"required": required, "forbidden": forbidden},
        "kg": {
            "ontology_version": subgraph.ontology_version,
            "subgraph_id": subgraph.subgraph_id,
        },
    }
