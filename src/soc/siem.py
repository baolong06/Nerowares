"""Best-effort, bounded SIEM forwarding for security audit events."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlparse

from src.config import settings

logger = logging.getLogger("thinking.siem")

_RELEASE_ALLOWED_KEYS = (
    "event_type",
    "timestamp",
    "correlation_id",
    "user_sub",
    "dataset_id",
    "release_id",
    "manifest_sha256",
    "path_hash",
    "provider",
    "mode",
    "result",
    "reason",
    "count",
    "total_size_bytes",
)


def sanitize_release_event(event: Mapping[str, object]) -> dict[str, object]:
    """Return only allowlisted release fields; hash paths, never keep raw path."""
    safe = {
        key: event[key]
        for key in _RELEASE_ALLOWED_KEYS
        if key in event and event[key] is not None
    }
    path = event.get("path")
    if path is not None:
        safe["path_hash"] = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return safe


def _send(endpoint: str, token: str | None, event: dict) -> None:
    body = json.dumps(event, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "thinking-eeg-siem/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=2) as response:
        response.read(1)


async def forward_audit_event(event: dict) -> None:
    """Forward only allowlisted audit fields; telemetry outages never break decoding."""
    endpoint = settings.siem_endpoint
    if not endpoint:
        return
    parsed = urlparse(endpoint)
    allowed_schemes = {"https"} if settings.env == "production" else {"https", "http"}
    if parsed.scheme not in allowed_schemes or not parsed.netloc:
        logger.error("SIEM forwarding disabled: endpoint violates transport policy")
        return
    safe_event = {
        key: event[key]
        for key in (
            "event_type", "timestamp", "correlation_id", "user_sub", "tenant_id",
            "method", "path", "status", "elapsed_ms", "src_ip", "anchor_count",
            "ontology_version", "prompt_hash", "prompt_hash_alg", "provider", "model",
            "provider_status", "data_egress", "batch_size", "batch_prompt_hashes",
            "dataset_id", "release_id", "manifest_sha256", "path_hash", "mode",
            "result", "reason", "count", "total_size_bytes",
        )
        if key in event and event[key] is not None
    }
    # Batch envelope: allow batch_size and hashed batch_prompt_hashes but never raw EEG/prompt text.
    if "batch_prompt_hashes" in safe_event and isinstance(safe_event["batch_prompt_hashes"], list):
        safe_event["batch_prompt_hashes"] = [str(value)[:64] for value in event["batch_prompt_hashes"] if isinstance(value, str)]
    try:
        await asyncio.to_thread(_send, endpoint, settings.siem_token, safe_event)
    except Exception:
        logger.warning("SIEM forwarding failed")
