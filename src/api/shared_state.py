"""Shared Redis coordination for durable security state (MFA/JTI/rate limit)."""
from __future__ import annotations

from src.config import settings


def _is_production() -> bool:
    return str(settings.env).lower() == "production"


def get_shared_redis():
    """Return a Redis client when configured, else None.

    In production, missing ``REDIS_URL`` is already rejected at ``Settings``
    construction, so a ``None`` here means the client could not be created.
    """
    if not settings.redis_url:
        return None
    try:
        import redis  # type: ignore[import-not-found]

        client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=0.25)
        # Lazy ping — only when caller requires durability
        return client
    except Exception:
        return None


def require_shared_redis():
    """Return a reachable Redis client or raise in production (fail-closed)."""
    client = get_shared_redis()
    if client is None:
        raise RuntimeError("shared Redis is required in production for MFA/JTI/rate limiting")
    try:
        client.ping()
    except Exception as exc:
        raise RuntimeError("shared Redis is unreachable in production") from exc
    return client
