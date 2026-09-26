import pytest
from fastapi.testclient import TestClient

from src.api.mfa import clear_enrollments_for_test
from src.api.security import create_token, verify_jwt
from src.main import app

client = TestClient(app)


def _auth(sub: str, tenant: str = "default", mfa: bool = False):
    tok = create_token(sub, tenant_id=tenant, mfa=mfa)
    return {"Authorization": f"Bearer {tok}"}, tok


def setup_function(_):
    clear_enrollments_for_test()


def test_mfa_setup_issues_secret_and_backup_codes():
    headers, _ = _auth("alice")
    r = client.post("/auth/mfa/setup", headers=headers)
    assert r.status_code == 200
    assert r.headers["Cache-Control"] == "no-store"
    data = r.json()
    assert "secret" in data and "otpauth_uri" in data and "backup_codes" in data
    assert len(data["backup_codes"]) == 8
    assert data["otpauth_uri"].startswith("otpauth://totp/")


def test_mfa_enroll_conflict():
    headers, _ = _auth("bob")
    assert client.post("/auth/mfa/setup", headers=headers).status_code == 200
    assert client.post("/auth/mfa/setup", headers=headers).status_code == 409


def test_mfa_verify_issues_elevated_token_and_decode_gates():
    # Use totp fallback computation for determinism
    from src.api.mfa import _totp, _ENROLLMENTS, _key

    headers, _ = _auth("carol")
    assert client.post("/auth/mfa/setup", headers=headers).status_code == 200
    # Decode without MFA should now be 403 (enrolled users must present mfa=true)
    r = client.post("/decode", json={"anchors_override": [{"surface": "open"}]}, headers=headers)
    assert r.status_code == 403

    # Verify with current TOTP -> elevated token
    payload = verify_jwt(headers["Authorization"].split()[1])
    key = _key(payload)
    secret = _ENROLLMENTS[key].secret
    code = _totp(secret)
    r = client.post("/auth/mfa/verify", json={"code": code}, headers=headers)
    assert r.status_code == 200
    assert r.json()["verified"] is True
    elevated = r.json()["issued_token"]
    assert elevated
    claims = verify_jwt(elevated)
    assert claims["mfa_verified"] is True
    assert claims["mfa"] is True

    # Decode with elevated token succeeds
    elevated_headers = {"Authorization": f"Bearer {elevated}"}
    r2 = client.post("/decode", json={"anchors_override": [{"surface": "open"}]}, headers=elevated_headers)
    assert r2.status_code == 200

    # Backup codes are single-use
    headers2, _ = _auth("dave")
    assert client.post("/auth/mfa/setup", headers=headers2).status_code == 200
    p2 = verify_jwt(headers2["Authorization"].split()[1])
    k2 = _key(p2)
    from src.api.mfa import _ENROLLMENTS as E2
    backup = E2[k2].backup_hashes
    # Extract one backup via setup response instead of internal set — redo setup capture
    clear_enrollments_for_test()
    h3, _ = _auth("dave2")
    setup = client.post("/auth/mfa/setup", headers=h3).json()
    code1 = setup["backup_codes"][0]
    assert client.post("/auth/mfa/verify", json={"code": code1}, headers=h3).json()["verified"] is True
    # Reusing same backup must not verify again (consumed)
    assert client.post("/auth/mfa/verify", json={"code": code1}, headers=h3).json()["verified"] is False


def test_mfa_verify_rate_limit():
    headers, _ = _auth("eve")
    assert client.post("/auth/mfa/setup", headers=headers).status_code == 200
    for _ in range(5):
        r = client.post("/auth/mfa/verify", json={"code": "000000"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["verified"] is False
    r = client.post("/auth/mfa/verify", json={"code": "000000"}, headers=headers)
    assert r.status_code == 429


def test_mfa_verify_rejects_enrollment_missing():
    headers, _ = _auth("frank")
    r = client.post("/auth/mfa/verify", json={"code": "123456"}, headers=headers)
    assert r.status_code == 400


def test_non_enrolled_users_still_decode_without_mfa():
    headers, _ = _auth("grace")
    # No enrollment -> decode must remain open (rolling deployment safe)
    r = client.post("/decode", json={"anchors_override": [{"surface": "open"}]}, headers=headers)
    assert r.status_code == 200
