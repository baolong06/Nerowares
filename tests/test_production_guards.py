"""TDD RED: production durability / fail-closed guards."""
import pytest

def test_vault_production_fails_closed_when_secret_missing(monkeypatch):
    monkeypatch.setenv("THINKING_ENV", "production")
    # ensure no Vault env and no JWT secret env
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    import importlib
    import src.config_vault as vault
    vault.get_secret.cache_clear()
    from src.config import Settings, _DEFAULT_JWT_SECRET
    # In production without Vault/env secret, constructing Settings should fail
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(env="production", jwt_secret=_DEFAULT_JWT_SECRET)
    vault.get_secret.cache_clear()

def test_cve_check_fails_closed_when_library_missing(monkeypatch):
    import pathlib, sys
    from scripts import check_cve
    # point to missing dir
    monkeypatch.setattr(check_cve, "LOCAL_CVE", pathlib.Path("references/cve-library-missing-xyz"))
    assert check_cve.main() == 2

def test_artifact_manifest_tamper_rejected(tmp_path):
    import shutil, subprocess, pathlib
    from scripts.artifact_manifest import build_manifest, write_signed_manifest, verify_manifest_signature
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("openssl required")
    src = tmp_path / "source.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    subprocess.run([openssl, "genpkey", "-algorithm", "ED25519", "-out", str(priv)], check=True, capture_output=True)
    subprocess.run([openssl, "pkey", "-in", str(priv), "-pubout", "-out", str(pub)], check=True, capture_output=True)
    manifest = build_manifest(artifact_dir=tmp_path, source_csv=src, result={"source":"bciciv2a:mi4","n_epochs":2,"n_channels":2,"n_times":4,"gate_pass":False}, subject_counts={"sub-01":2})
    path = write_signed_manifest(manifest, tmp_path / "manifest.json", priv.read_bytes())
    # tamper
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"n_epochs": 2', '"n_epochs": 99'), encoding="utf-8")
    assert verify_manifest_signature(path, pub.read_bytes()) is False

def test_ratelimit_production_requires_redis(monkeypatch):
    from src.config import Settings

    with pytest.raises(ValueError, match="REDIS_URL"):
        Settings(env="production", jwt_secret="runtime-secret-without-shared-store")


def test_shared_state_requires_reachable_redis_in_production(monkeypatch):
    from src.api.shared_state import get_shared_redis, require_shared_redis

    # No Redis configured -> require must fail-closed
    monkeypatch.setattr("src.api.shared_state.settings", type("S", (), {"env": "production", "redis_url": None})())
    assert get_shared_redis() is None
    with pytest.raises(RuntimeError, match="shared Redis"):
        require_shared_redis()
