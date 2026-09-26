import asyncio

from src.soc import siem


def test_siem_forwards_only_allowlisted_fields(monkeypatch):
    captured = {}

    def fake_send(endpoint, token, event):
        captured.update(endpoint=endpoint, token=token, event=event)

    monkeypatch.setattr(siem, "_send", fake_send)
    monkeypatch.setattr(siem.settings, "siem_endpoint", "https://siem.example/events")
    monkeypatch.setattr(siem.settings, "siem_token", "not-logged")
    asyncio.run(
        siem.forward_audit_event(
            {
                "event_type": "api_request",
                "correlation_id": "validcorr_123",
                "path": "/decode",
                "secret": "must-not-forward",
            }
        )
    )
    assert captured["endpoint"] == "https://siem.example/events"
    assert captured["event"] == {
        "event_type": "api_request",
        "correlation_id": "validcorr_123",
        "path": "/decode",
    }


def test_siem_rejects_non_http_endpoint(monkeypatch):
    monkeypatch.setattr(siem.settings, "siem_endpoint", "file:///tmp/telemetry")
    asyncio.run(siem.forward_audit_event({"event_type": "api_request"}))
