"""TOTP MFA enrollment and verification with fail-closed production policy."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import quote

from fastapi import HTTPException, status

from src.api import shared_state as _shared
from src.config import settings

_BACKUP_CODE_COUNT = 8
_MAX_ATTEMPTS = 5
_ATTEMPT_WINDOW_SECONDS = 300.0


@dataclass
class MFAEnrollment:
    secret: str
    backup_hashes: set[str]


_ENROLLMENTS: dict[tuple[str, str], MFAEnrollment] = {}
_ATTEMPTS: dict[tuple[str, str], deque[float]] = defaultdict(deque)
# Test cleanup keeps an index so shared Redis state can be removed after a
# simulated process restart without scanning unrelated application keys.
_KNOWN_KEYS: set[tuple[str, str]] = set()


def _key(user: dict) -> tuple[str, str]:
    sub = str(user.get("sub") or "")
    tenant = str(user.get("tenant_id") or "default")
    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user identity")
    return tenant, sub


def _totp(secret: str, timestamp: float | None = None, interval: int = 30) -> str:
    counter = int((timestamp if timestamp is not None else time.time()) // interval)
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, counter.to_bytes(8, "big"), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF
    return f"{binary % 1_000_000:06d}"


def _valid_totp(secret: str, code: str) -> bool:
    cleaned = code.strip().replace(" ", "")
    if len(cleaned) != 6 or not cleaned.isdecimal():
        return False
    try:
        import pyotp

        return bool(pyotp.TOTP(secret).verify(cleaned, valid_window=1))
    except ImportError:
        now = time.time()
        return any(hmac.compare_digest(cleaned, _totp(secret, now + delta * 30)) for delta in (-1, 0, 1))


def _backup_hash(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def _redis_key(key: tuple[str, str]) -> str:
    tenant, sub = key
    return f"thinking:mfa:{tenant}:{sub}"


def _shared_client():
    return _shared.require_shared_redis() if str(settings.env).lower() == "production" else _shared.get_shared_redis()


def _serialize(enrollment: MFAEnrollment) -> dict[str, str]:
    return {
        "secret": enrollment.secret,
        "backup_hashes": ",".join(sorted(enrollment.backup_hashes)),
    }


def _deserialize(data: dict[str, str]) -> MFAEnrollment | None:
    secret = data.get("secret")
    if not secret:
        return None
    raw_hashes = data.get("backup_hashes", "")
    return MFAEnrollment(secret=secret, backup_hashes={value for value in raw_hashes.split(",") if value})


def _load_enrollment(key: tuple[str, str]) -> MFAEnrollment | None:
    local = _ENROLLMENTS.get(key)
    if local is not None:
        return local
    client = _shared_client()
    if client is None:
        return None
    try:
        enrollment = _deserialize(client.hgetall(_redis_key(key)))
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for MFA") from exc
        return None
    if enrollment is not None:
        _ENROLLMENTS[key] = enrollment
        _KNOWN_KEYS.add(key)
    return enrollment


def _persist_enrollment(key: tuple[str, str], enrollment: MFAEnrollment) -> None:
    client = _shared_client()
    if client is None:
        return
    try:
        client.hset(_redis_key(key), mapping=_serialize(enrollment))
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for MFA") from exc


def _assert_attempt_allowed(key: tuple[str, str]) -> None:
    now = time.monotonic()
    attempts = _ATTEMPTS[key]
    while attempts and now - attempts[0] >= _ATTEMPT_WINDOW_SECONDS:
        attempts.popleft()
    if len(attempts) >= _MAX_ATTEMPTS:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="MFA verification temporarily locked")

    client = _shared_client()
    if client is None:
        return
    try:
        count = int(client.get(f"{_redis_key(key)}:attempts") or 0)
        if count >= _MAX_ATTEMPTS:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="MFA verification temporarily locked")
    except HTTPException:
        raise
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for MFA attempts") from exc


def _record_failed_attempt(key: tuple[str, str]) -> None:
    _ATTEMPTS[key].append(time.monotonic())
    client = _shared_client()
    if client is None:
        return
    try:
        attempt_key = f"{_redis_key(key)}:attempts"
        count = int(client.incr(attempt_key))
        if count == 1:
            client.expire(attempt_key, int(_ATTEMPT_WINDOW_SECONDS))
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for MFA attempts") from exc


def _clear_attempts(key: tuple[str, str]) -> None:
    _ATTEMPTS.pop(key, None)
    client = _shared_client()
    if client is not None:
        try:
            client.delete(f"{_redis_key(key)}:attempts")
        except Exception as exc:
            if str(settings.env).lower() == "production":
                raise RuntimeError("shared Redis is unavailable for MFA attempts") from exc


def is_enrolled(user: dict) -> bool:
    return _load_enrollment(_key(user)) is not None


def enroll(user: dict) -> dict:
    key = _key(user)
    if _load_enrollment(key) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="MFA already enrolled")
    secret = base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")
    codes = [secrets.token_urlsafe(8).upper() for _ in range(_BACKUP_CODE_COUNT)]
    enrollment = MFAEnrollment(secret=secret, backup_hashes={_backup_hash(code) for code in codes})
    _ENROLLMENTS[key] = enrollment
    _KNOWN_KEYS.add(key)
    _persist_enrollment(key, enrollment)
    issuer = quote(settings.mfa_issuer, safe="")
    account = quote(f"{key[0]}:{key[1]}", safe="")
    uri = f"otpauth://totp/{issuer}:{account}?secret={secret}&issuer={issuer}&algorithm=SHA1&digits=6&period=30"
    return {"secret": secret, "otpauth_uri": uri, "backup_codes": codes}


def verify(user: dict, code: str) -> bool:
    key = _key(user)
    enrollment = _load_enrollment(key)
    if enrollment is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MFA is not enrolled")
    _assert_attempt_allowed(key)
    valid = _valid_totp(enrollment.secret, code)
    if not valid:
        hashed = _backup_hash(code)
        if hashed in enrollment.backup_hashes:
            enrollment.backup_hashes.remove(hashed)
            _persist_enrollment(key, enrollment)
            valid = True
    if valid:
        _clear_attempts(key)
        return True
    _record_failed_attempt(key)
    return False


def require_mfa_claim(user: dict) -> None:
    """Require MFA for production/global policy and every enrolled identity."""
    verified = user.get("mfa_verified", user.get("mfa", False))
    production = str(settings.env).lower() == "production"
    if (settings.mfa_required or production or is_enrolled(user)) and verified is not True:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="MFA verification required")


def clear_enrollments_for_test() -> None:
    keys = set(_KNOWN_KEYS) | set(_ENROLLMENTS)
    _ENROLLMENTS.clear()
    _ATTEMPTS.clear()
    _KNOWN_KEYS.clear()
    client = _shared_client()
    if client is None:
        return
    try:
        for key in keys:
            client.delete(_redis_key(key), f"{_redis_key(key)}:attempts")
    except Exception as exc:
        if str(settings.env).lower() == "production":
            raise RuntimeError("shared Redis is unavailable for MFA cleanup") from exc
