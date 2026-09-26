"""Bounded rate limiting with optional shared Redis coordination."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from ipaddress import ip_address

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api import shared_state as _shared
from src.config import settings

_HITS: dict[str, deque[float]] = defaultdict(deque)
_SKIP = {"/health"}
_REDIS_CLIENT = None


def _peer_key(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    trusted = set(settings.trusted_proxy_ips)
    if peer in trusted:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            try:
                return str(ip_address(forwarded))
            except ValueError:
                pass
    return peer


def _redis_sync():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None or not settings.redis_url:
        return _REDIS_CLIENT
    try:
        import redis

        _REDIS_CLIENT = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=0.25)
    except Exception:
        _REDIS_CLIENT = False
    return _REDIS_CLIENT


def _local_allowed(key: str, now: float, limit: int) -> bool:
    bucket = _HITS[key]
    while bucket and now - bucket[0] >= 60.0:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def _redis_allowed(client, key: str, limit: int) -> bool:
    try:
        pipe = client.pipeline()
        redis_key = f"thinking:ratelimit:{key}:{int(time.time() // 60)}"
        pipe.incr(redis_key)
        pipe.expire(redis_key, 61)
        count, _ = pipe.execute()
        return int(count) <= limit
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for rate limiting") from exc
        return _local_allowed(key, time.time(), limit)


async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in _SKIP or not settings.rate_limit_enabled:
        return await call_next(request)
    key = _peer_key(request)
    limit = max(1, int(settings.rate_limit_per_minute))
    if str(settings.env).lower() == "production":
        client = await asyncio.to_thread(_shared.require_shared_redis)
        allowed = await asyncio.to_thread(_redis_allowed, client, key, limit)
    else:
        client = await asyncio.to_thread(_redis_sync)
        allowed = await asyncio.to_thread(_redis_allowed, client, key, limit) if client else _local_allowed(key, time.time(), limit)
    if not allowed:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)
