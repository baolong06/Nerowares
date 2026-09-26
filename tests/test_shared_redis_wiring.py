"""RED: shared Redis wiring for MFA/JTI/rate limit must be durable in production."""
import pytest


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.ttls = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None, px=None, exat=None):
        self.store[key] = value if isinstance(value, str) else str(value)
        return True

    def setex(self, key, ttl, value):
        self.store[key] = value if isinstance(value, str) else str(value)
        return True

    def setnx(self, key, value):
        if key in self.store:
            return False
        self.store[key] = value
        return True

    def sadd(self, key, *values):
        current = self.store.setdefault(key, set())
        if not isinstance(current, set):
            current = set()
            self.store[key] = current
        before = len(current)
        current.update(values)
        return len(current) - before

    def srem(self, key, *values):
        current = self.store.get(key, set())
        if not isinstance(current, set):
            return 0
        removed = sum(1 for value in values if value in current)
        current.difference_update(values)
        return removed

    def exists(self, *keys):
        return sum(1 for k in keys if k in self.store)

    def delete(self, *keys):
        c = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                c += 1
        return c

    def incr(self, key):
        v = int(self.store.get(key, "0")) + 1
        self.store[key] = str(v)
        return v

    def expire(self, key, ttl):
        return True

    def expireat(self, key, ts):
        return True

    def ttl(self, key):
        return -1

    def pipeline(self):
        parent = self

        class Pipe:
            def __init__(self):
                self.cmds = []

            def incr(self, k):
                self.cmds.append(("incr", k))
                return self

            def expire(self, k, ttl):
                self.cmds.append(("expire", k, ttl))
                return self

            def execute(self):
                res = []
                for cmd in self.cmds:
                    if cmd[0] == "incr":
                        res.append(parent.incr(cmd[1]))
                    elif cmd[0] == "expire":
                        res.append(parent.expire(cmd[1], cmd[2]))
                return res

        return Pipe()

    # hash helpers for MFA
    def hgetall(self, key):
        v = self.store.get(key)
        return v if isinstance(v, dict) else {}

    def hmset(self, key, mapping):
        self.store[key] = dict(mapping)
        return True

    def hset(self, key, mapping=None, **kwargs):
        if mapping is None:
            mapping = kwargs
        else:
            mapping = {**mapping, **kwargs}
        cur = self.store.get(key, {})
        if not isinstance(cur, dict):
            cur = {}
        cur.update({k: str(v) for k, v in mapping.items()})
        self.store[key] = cur
        return 1


def _install_fake_redis(monkeypatch, fake):
    monkeypatch.setattr("src.api.shared_state.get_shared_redis", lambda: fake)
    monkeypatch.setattr("src.api.shared_state.require_shared_redis", lambda: fake)


def test_mfa_enrollment_survives_process_local_clear_when_redis_available(monkeypatch):
    """MFA enrollment must be durable via Redis in production, not only _ENROLLMENTS."""
    from src.api import mfa as mfa_mod

    fake = FakeRedis()
    # force production env for this test via settings patch
    monkeypatch.setattr("src.api.mfa.settings", type("S", (), {"env": "production", "redis_url": "redis://fake", "mfa_required": True, "mfa_issuer": "THINKING EEG Platform"})())
    monkeypatch.setattr("src.api.shared_state.settings", type("S", (), {"env": "production", "redis_url": "redis://fake"})())
    _install_fake_redis(monkeypatch, fake)
    mfa_mod.clear_enrollments_for_test()
    user = {"sub": "alice-redis", "tenant_id": "default"}
    mfa_mod.enroll(user)
    # simulate process restart: clear local dict
    mfa_mod._ENROLLMENTS.clear()
    # RED expectation: is_enrolled must still be True via Redis
    assert mfa_mod.is_enrolled(user) is True


def test_jti_revocation_survives_process_local_clear_when_redis_available(monkeypatch):
    import time
    from src.api import security as sec

    fake = FakeRedis()
    monkeypatch.setattr("src.api.security.settings", type("S", (), {"env": "production", "redis_url": "redis://fake"})())
    monkeypatch.setattr("src.api.shared_state.settings", type("S", (), {"env": "production", "redis_url": "redis://fake"})())
    _install_fake_redis(monkeypatch, fake)
    sec.clear_revocations_for_test()
    jti = "test-jti-123"
    sec.revoke_token(jti, time.time() + 600)
    sec._REVOKED_JTIS.clear()
    assert sec.is_revoked(jti) is True


def test_rate_limit_requires_redis_in_production(monkeypatch):
    """In production Redis unavailable must not silently fall back to local."""
    from src.api import ratelimit as rl

    monkeypatch.setattr("src.api.ratelimit.settings", type("S", (), {"env": "production", "redis_url": "redis://fake", "rate_limit_enabled": True, "rate_limit_per_minute": 60, "trusted_proxy_ips": []})())
    monkeypatch.setattr("src.api.shared_state.settings", type("S", (), {"env": "production", "redis_url": "redis://fake"})())
    # simulate unreachable Redis
    class Unreachable:
        def ping(self):
            raise ConnectionError("down")

        def pipeline(self):
            raise ConnectionError("down")

    monkeypatch.setattr("src.api.shared_state.get_shared_redis", lambda: Unreachable())
    monkeypatch.setattr("src.api.shared_state.require_shared_redis", lambda: (_ for _ in ()).throw(RuntimeError("shared Redis is unreachable in production")))
    # _redis_allowed fallback path must raise or return 429, not silently allow
    # We test the middleware helper: when require fails, local fallback must NOT be used
    import asyncio
    from fastapi import Request

    # Directly test _redis_allowed fallback is disabled in production.
    # Redis failure must not silently allow the request through local state.
    fake_unreachable = Unreachable()
    with pytest.raises(RuntimeError, match="shared Redis"):
        rl._redis_allowed(fake_unreachable, "test-key", 60)
    from src.api.shared_state import require_shared_redis

    with pytest.raises(RuntimeError, match="shared Redis"):
        require_shared_redis()


def test_vault_failure_is_not_swallowed_in_production(monkeypatch):
    """Production must not silently return None when Vault is misconfigured."""
    import src.config_vault as vault

    monkeypatch.setenv("THINKING_ENV", "production")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example.invalid")
    monkeypatch.setenv("VAULT_TOKEN", "bad-token")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    vault.get_secret.cache_clear()

    # Simulate hvac failure
    import sys

    class FakeHvac:
        class Client:
            def __init__(self, url, token):
                raise RuntimeError("vault unreachable")

    monkeypatch.setitem(sys.modules, "hvac", FakeHvac)
    with pytest.raises(RuntimeError, match="(?i)vault"):
        vault.get_secret("JWT_SECRET")
    vault.get_secret.cache_clear()
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
