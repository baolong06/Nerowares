from src.config_vault import get_secret


def test_vault_reads_environment_before_hvac(monkeypatch):
    get_secret.cache_clear()
    monkeypatch.setenv("JWT_SECRET", "from-env-not-logged")
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    assert get_secret("JWT_SECRET") == "from-env-not-logged"
    get_secret.cache_clear()


def test_vault_returns_none_without_addr_or_env(monkeypatch):
    get_secret.cache_clear()
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    assert get_secret("MISSING_SECRET") is None
    get_secret.cache_clear()
