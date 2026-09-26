"""Middleware — correlation sanitization and allowlisted audit telemetry."""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone

from fastapi import Request, Response

from src.soc.siem import forward_audit_event

# Safe correlation id: 8-64 alnum/-/_.
_SAFE_CORR_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _sanitize_corr(raw: str | None) -> str:
    if not raw:
        return str(uuid.uuid4())
    if any(c in raw for c in ("\r", "\n", "\x00")) or not raw.isprintable():
        return str(uuid.uuid4())
    raw = raw.strip()[:64]
    if _SAFE_CORR_RE.match(raw):
        return raw
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", raw)[:64]
    if len(sanitized) >= 8 and _SAFE_CORR_RE.match(sanitized):
        return sanitized
    return str(uuid.uuid4())


async def correlation_middleware(request: Request, call_next):
    # Enforce a compact, bounded request envelope before any decode/provider work.
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > 1_048_576:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=413, content={"detail": "Request entity too large"})
    corr = _sanitize_corr(request.headers.get("X-Correlation-ID"))
    request.state.correlation_id = corr
    if not hasattr(request.state, "user_sub"):
        request.state.user_sub = "-"
    start = time.time()
    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr
    elapsed = (time.time() - start) * 1000
    user = getattr(request.state, "user_sub", "-")
    path = request.url.path.replace("\r", "").replace("\n", "")
    await forward_audit_event(
        {
            "event_type": getattr(request.state, "audit_event_type", "api_request"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": corr,
            "user_sub": user,
            "tenant_id": getattr(request.state, "tenant_id", "-"),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "elapsed_ms": round(elapsed, 1),
            "src_ip": request.client.host if request.client else "unknown",
            # Phase 4 semantics — hashes and counts only; never raw EEG or prompt text.
            "anchor_count": getattr(request.state, "anchor_count", None),
            "ontology_version": getattr(request.state, "ontology_version", None),
            "prompt_hash": getattr(request.state, "prompt_hash", None),
            "prompt_hash_alg": "sha256" if getattr(request.state, "prompt_hash", None) else None,
            "provider": getattr(request.state, "provider", None),
            "model": getattr(request.state, "model", None),
            "provider_status": getattr(request.state, "provider_status", None),
            "data_egress": getattr(request.state, "data_egress", None),
            "batch_size": getattr(request.state, "batch_size", None),
            "batch_prompt_hashes": getattr(request.state, "batch_prompt_hashes", None),
        }
    )
    return response
