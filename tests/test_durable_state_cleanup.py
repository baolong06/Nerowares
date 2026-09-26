"""Durable-state contracts for process restart and single-use MFA behavior."""
from __future__ import annotations

import time

from tests.test_shared_redis_wiring import FakeRedis, _install_fake_redis


def _production_settings():
    return type(
        "S",
        (),
        {
            "env": "production",
            "redis_url": "redis://fake",
            "mfa_required": True,
            "mfa_issuer": "THINKING EEG Platform",
        },
    )()


def test_clear_enrollments_for_test_clears_shared_mfa_state(monkeypatch):
    from src.api import mfa

    fake = FakeRedis()
    monkeypatch.setattr("src.api.mfa.settings", _production_settings())
    monkeypatch.setattr("src.api.shared_state.settings", _production_settings())
    _install_fake_redis(monkeypatch, fake)

    user = {"sub": "cleanup-user", "tenant_id": "default"}
    mfa.clear_enrollments_for_test()
    mfa.enroll(user)
    assert mfa.is_enrolled(user)

    mfa.clear_enrollments_for_test()
    assert mfa.is_enrolled(user) is False
    assert not any(str(key).startswith("thinking:mfa:") for key in fake.store)


def test_backup_code_consumption_is_durable_across_local_cache_clear(monkeypatch):
    from src.api import mfa

    fake = FakeRedis()
    monkeypatch.setattr("src.api.mfa.settings", _production_settings())
    monkeypatch.setattr("src.api.shared_state.settings", _production_settings())
    _install_fake_redis(monkeypatch, fake)

    user = {"sub": "single-use-user", "tenant_id": "default"}
    mfa.clear_enrollments_for_test()
    setup = mfa.enroll(user)
    code = setup["backup_codes"][0]
    assert mfa.verify(user, code) is True

    mfa._ENROLLMENTS.clear()
    assert mfa.verify(user, code) is False


def test_clear_revocations_for_test_clears_shared_jti_state(monkeypatch):
    from src.api import security

    fake = FakeRedis()
    settings = type("S", (), {"env": "production", "redis_url": "redis://fake"})()
    monkeypatch.setattr("src.api.security.settings", settings)
    monkeypatch.setattr("src.api.shared_state.settings", settings)
    _install_fake_redis(monkeypatch, fake)

    jti = "cleanup-jti"
    security.clear_revocations_for_test()
    security.revoke_token(jti, time.time() + 600)
    assert security.is_revoked(jti) is True

    security.clear_revocations_for_test()
    assert security.is_revoked(jti) is False
    assert not any(str(key).startswith("thinking:revoked-jti:") for key in fake.store)
