"""Validate local interior-v1.0 seed and its runtime text-v0.1 extension contract."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_confidence(value: object, context: str) -> None:
    require(isinstance(value, (int, float)), f"{context}: confidence must be numeric")
    require(0 <= value <= 1, f"{context}: confidence must be in [0, 1]")


def check_provenance(value: object, context: str) -> None:
    require(isinstance(value, list) and value, f"{context}: provenance must be non-empty")
    require(all(isinstance(item, str) and item for item in value), f"{context}: invalid provenance")


def check_ontology(ontology: dict) -> None:
    node_ids = [node["id"] for node in ontology["nodes"]]
    require(len(node_ids) == len(set(node_ids)), "Ontology node IDs must be unique")
    known = set(node_ids)
    edge_ids: set[str] = set()
    for edge in ontology["edges"]:
        require(edge["id"] not in edge_ids, f"Duplicate edge ID: {edge['id']}")
        edge_ids.add(edge["id"])
        require(edge["source"] in known, f"Unknown edge source: {edge['source']}")
        require(edge["target"] in known, f"Unknown edge target: {edge['target']}")
        check_confidence(edge["confidence"], edge["id"])
        require(bool(edge.get("provenance")), f"{edge['id']}: missing provenance")
    for synonym in ontology["synonyms"]:
        require(synonym["concept_id"] in known, f"Unknown synonym target: {synonym['concept_id']}")


def main() -> None:
    from src.kg.ontology import load_ontology
    from src.prompt.validator import validate_prompt

    seed = load("ontology_seed.json")
    prompt = load("example_structured_prompt.json")
    schema = load("structured_prompt.schema.json")
    check_ontology(seed)
    validate_prompt(prompt)
    require(prompt["kg"]["ontology_version"] == seed["ontology_version"], "Seed prompt ontology version mismatch")
    require(schema.get("$schema", "").endswith("2020-12/schema"), "Expected Draft 2020-12 schema")

    runtime = load_ontology(with_text_extension=True)
    require(runtime.ontology_version == "text-v0.1", "Runtime ontology must be text-v0.1")
    require(len(runtime.nodes) == 19, f"Expected 19 runtime nodes, got {len(runtime.nodes)}")
    require(len(runtime.edges) == 5, f"Expected 5 runtime edges, got {len(runtime.edges)}")
    print(
        f"Contracts valid: {len(seed['nodes'])} seed nodes/{len(seed['edges'])} seed edges; "
        f"{len(runtime.nodes)} text-v0.1 nodes/{len(runtime.edges)} text-v0.1 edges"
    )


if __name__ == "__main__":
    main()
