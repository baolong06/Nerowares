"""JWT authentication, revocation, and ownership checks."""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api import shared_state as _shared
from src.config import settings

security = HTTPBearer(auto_error=False)
_MAX_REVOKED = 10_000
_REVOCATION_PREFIX = "thinking:revoked-jti:"
_REVOKED_JTIS: OrderedDict[str, float] = OrderedDict()
# Test cleanup index for revocations persisted in shared Redis.
_KNOWN_REVOKED_JTIS: set[str] = set()


def _now() -> int:
    return int(time.time())


def _prune_revoked(now: float | None = None) -> None:
    current = time.time() if now is None else now
    expired = [jti for jti, exp in _REVOKED_JTIS.items() if exp <= current]
    for jti in expired:
        _REVOKED_JTIS.pop(jti, None)
    while len(_REVOKED_JTIS) > _MAX_REVOKED:
        _REVOKED_JTIS.popitem(last=False)


def _shared_client():
    return _shared.require_shared_redis() if str(settings.env).lower() == "production" else _shared.get_shared_redis()


def _redis_revocation_key(jti: str) -> str:
    return f"{_REVOCATION_PREFIX}{jti}"


def revoke_token(jti: str, exp: float) -> None:
    if not jti:
        return
    _prune_revoked()
    value = float(exp)
    _REVOKED_JTIS[str(jti)] = value
    _REVOKED_JTIS.move_to_end(str(jti))
    _KNOWN_REVOKED_JTIS.add(str(jti))
    client = _shared_client()
    if client is None:
        return
    ttl = max(1, int(value - time.time()))
    try:
        client.setex(_redis_revocation_key(str(jti)), ttl, "1")
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for token revocation") from exc


def is_revoked(jti: object) -> bool:
    if not isinstance(jti, str) or not jti:
        return True
    _prune_revoked()
    if jti in _REVOKED_JTIS:
        return True
    client = _shared_client()
    if client is None:
        return False
    try:
        return bool(client.get(_redis_revocation_key(jti)))
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for token revocation") from exc
        return False


def clear_revocations_for_test() -> None:
    jtis = set(_KNOWN_REVOKED_JTIS) | set(_REVOKED_JTIS)
    _REVOKED_JTIS.clear()
    _KNOWN_REVOKED_JTIS.clear()
    client = _shared_client()
    if client is None:
        return
    try:
        for jti in jtis:
            client.delete(_redis_revocation_key(jti))
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for token cleanup") from exc


def create_token(
    sub: str,
    tenant_id: str = "default",
    exp_minutes: int | None = None,
    mfa: bool = False,
    mfa_verified: bool | None = None,
) -> str:
    now = _now()
    exp = now + (exp_minutes if exp_minutes is not None else settings.jwt_exp_minutes) * 60
    flag = bool(mfa_verified if mfa_verified is not None else mfa)
    payload = {
        "sub": str(sub),
        "tenant_id": str(tenant_id),
        "mfa_verified": flag,
        "mfa": flag,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "exp": exp,
        "iat": now,
        "nbf": now,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_jwt(token: str) -> dict | None:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            options={"require": ["sub", "iss", "aud", "exp", "iat", "nbf", "jti"]},
            leeway=5,
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("sub"), str):
            return None
        if not isinstance(payload.get("jti"), str) or is_revoked(payload["jti"]):
            return None
        for claim in ("mfa_verified", "mfa"):
            if claim in payload and not isinstance(payload[claim], bool):
                return None
        if "mfa_verified" not in payload:
            payload["mfa_verified"] = bool(payload.get("mfa", False))
        elif "mfa" in payload and payload["mfa"] != payload["mfa_verified"]:
            return None
        return payload
    except (jwt.InvalidTokenError, TypeError, ValueError):
        return None


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication")
    payload = verify_jwt(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    request.state.user_sub = payload.get("sub", "-")
    request.state.tenant_id = payload.get("tenant_id", "default")
    return payload


def enforce_ownership(resource_owner: str, resource_tenant: str, current_user: dict) -> None:
    if current_user.get("sub") != resource_owner or current_user.get("tenant_id") != resource_tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
