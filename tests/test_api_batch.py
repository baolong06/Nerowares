"""Batch decode — RED phase for POST /decode/batch."""
import base64
import json

from fastapi.testclient import TestClient

from src.api.security import create_token
from src.main import app

client = TestClient(app)

_default_token = create_token("tester", tenant_id="default")
_default_headers = {"Authorization": f"Bearer {_default_token}"}
_foreign_token = create_token("tester", tenant_id="foreign")
_foreign_headers = {"Authorization": f"Bearer {_foreign_token}"}


def _eeg(rows=8, cols=20):
    return [[float((i + j) % 5) for j in range(cols)] for i in range(rows)]


def test_batch_decode_requires_auth():
    r = client.post("/decode/batch", json={"items": [{"eeg": _eeg(), "session_id": "s1"}]})
    assert r.status_code == 401


def test_batch_decode_rejects_oversized_batch():
    items = [{"anchors_override": [{"surface": "open"}]} for _ in range(9)]
    r = client.post("/decode/batch", json={"items": items}, headers=_default_headers)
    assert r.status_code == 422


def test_batch_decode_rejects_oversized_per_item():
    # Exceed max_eeg_channels (64) with 65 channels
    eeg = [[1.0] * 10 for _ in range(65)]
    r = client.post("/decode/batch", json={"items": [{"eeg": eeg}]}, headers=_default_headers)
    assert r.status_code in (400, 422)


def test_batch_decode_preserves_provenance_per_item():
    r = client.post(
        "/decode/batch",
        json={
            "items": [
                {"eeg": _eeg(), "session_id": "batch-s1"},
                {"anchors_override": [{"surface": "open", "score": 0.9}]},
            ]
        },
        headers=_default_headers,
    )
    assert r.status_code == 200
    payload = r.json()
    assert "results" in payload
    assert len(payload["results"]) == 2
    for item in payload["results"]:
        assert all(p.startswith("eeg:") for p in item["anchors"][0]["provenance"])
        assert "kg:" in " ".join(item["provenance"])
    assert "X-Correlation-ID" in r.headers


def test_batch_decode_rejects_nonfinite():
    body = json.dumps(
        {"items": [{"eeg": [[float("inf")] * 8 for _ in range(4)]}]},
        allow_nan=True,
    )
    r = client.post(
        "/decode/batch",
        content=body,
        headers={**_default_headers, "Content-Type": "application/json"},
    )
    assert r.status_code in (400, 422)


def test_batch_decode_provider_policy_per_item(monkeypatch):
    from src import config as cfg

    monkeypatch.setattr(cfg.settings, "llm_enabled", True)
    monkeypatch.setattr(cfg.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(cfg.settings, "llm_model", "claude-opus-5")
    monkeypatch.setattr(cfg.settings, "llm_allowed_tenants", ["default"])
    r = client.post(
        "/decode/batch",
        json={"items": [{"anchors_override": [{"surface": "open"}], "use_llm": True}]},
        headers=_foreign_headers,
    )
    assert r.status_code == 403


def test_batch_decode_single_item_anchors_override():
    r = client.post(
        "/decode/batch",
        json={"items": [{"anchors_override": [{"surface": "open"}]}]},
        headers=_default_headers,
    )
    assert r.status_code == 200
    assert r.json()["results"][0]["structured_prompt"]["schema_version"] == "1.0"
