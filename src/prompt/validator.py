"""Prompt validator — jsonschema Draft 2020-12 against structured_prompt.schema.json."""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "architecture" / "knowledge_graph" / "structured_prompt.schema.json",
    Path("E:/AI_thucchien/THINKING/architecture/knowledge_graph/structured_prompt.schema.json"),
]

CONCEPT_RE = __import__("re").compile(r"^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9_-]+$")
VALID_MODALITIES = {"image", "text", "music", "video", "3d"}
VALID_STATUSES = {"accepted", "suggested", "rejected", "unknown"}
VALID_OPERATORS = {"set", "increase", "decrease", "preserve", "avoid"}
ALLOWED_TOP_KEYS = {"schema_version", "modality", "intent", "subject", "attributes", "constraints", "kg"}


def _load_schema() -> dict | None:
    for p in SCHEMA_CANDIDATES:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def _manual_strict(p: dict) -> None:
    extra = set(p.keys()) - ALLOWED_TOP_KEYS
    if extra:
        raise AssertionError(f"additionalProperties not allowed: {sorted(extra)}")
    assert p.get("schema_version") == "1.0", "schema_version must be 1.0"
    assert p.get("modality") in VALID_MODALITIES, f"Invalid modality {p.get('modality')}"
    for field in ("intent", "subject"):
        v = p.get(field)
        assert isinstance(v, dict), f"{field} must be object"
        assert 0 <= float(v.get("confidence", -1)) <= 1, f"{field} confidence out of range"
        assert isinstance(v.get("provenance"), list) and v["provenance"], f"{field} provenance required"
        extra_f = set(v.keys()) - ({"value", "confidence", "provenance"} if field == "intent" else {"concept_id", "label", "confidence", "provenance"})
        if extra_f:
            raise AssertionError(f"{field} additionalProperties not allowed: {sorted(extra_f)}")
    assert CONCEPT_RE.match(p["subject"].get("concept_id", "")), "subject concept_id pattern invalid"
    for i, a in enumerate(p.get("attributes", [])):
        ctx = f"attributes[{i}]"
        extra_a = set(a.keys()) - {"concept_id", "operator", "value", "confidence", "status", "provenance"}
        if extra_a:
            raise AssertionError(f"{ctx} additionalProperties not allowed: {sorted(extra_a)}")
        assert CONCEPT_RE.match(a.get("concept_id", "")), f"{ctx} concept_id"
        assert a.get("status") in VALID_STATUSES, f"{ctx} status"
        assert a.get("operator") in VALID_OPERATORS, f"{ctx} operator"
        assert 0 <= float(a.get("confidence", -1)) <= 1, f"{ctx} confidence"
        assert isinstance(a.get("provenance"), list) and a["provenance"], f"{ctx} provenance"
    cons = p.get("constraints")
    assert isinstance(cons, dict) and "required" in cons and "forbidden" in cons
    extra_c = set(cons.keys()) - {"required", "forbidden"}
    if extra_c:
        raise AssertionError(f"constraints additionalProperties not allowed: {sorted(extra_c)}")
    kg = p.get("kg")
    assert isinstance(kg, dict) and kg.get("ontology_version") and kg.get("subgraph_id")
    extra_k = set(kg.keys()) - {"ontology_version", "subgraph_id"}
    if extra_k:
        raise AssertionError(f"kg additionalProperties not allowed: {sorted(extra_k)}")


def validate_prompt(p: dict) -> None:
    schema = _load_schema()
    if schema is not None:
        try:
            from jsonschema import Draft202012Validator

            validator = Draft202012Validator(schema)
            errors = sorted(validator.iter_errors(p), key=lambda e: list(e.path))
            if errors:
                raise AssertionError("; ".join(f"{list(e.path)}: {e.message}" for e in errors[:8]))
            return
        except ImportError:
            pass
    _manual_strict(p)
