"""App config — Pydantic Settings. Fail-closed JWT secret; no hardcoded production keys."""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_vault import get_secret

_DEFAULT_JWT_SECRET = "dev-only-change-me-32-byte-placeholder!"


def _default_data_root() -> Path:
    env = os.environ.get("THINKING_DATA_ROOT")
    if env:
        return Path(env)
    worktree = Path(__file__).resolve().parents[1] / "datasets"
    legacy = Path("E:/AI_thucchien/THINKING/datasets")
    if worktree.exists():
        return worktree
    if legacy.exists():
        return legacy
    return worktree


def _in_pytest() -> bool:
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


class Settings(BaseSettings):
    app_name: str = "THINKING EEG Platform"
    env: str = "development"
    jwt_secret: str = _DEFAULT_JWT_SECRET
    jwt_issuer: str = "thinking.local"
    jwt_audience: str = "thinking-api"
    jwt_exp_minutes: int = 15
    data_root: Path = Field(default_factory=_default_data_root)
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"
    rate_limit_per_minute: int = 60
    rate_limit_enabled: bool = True
    redis_url: str | None = None
    trusted_proxy_ips: list[str] = []
    trusted_hosts: list[str] = ["localhost", "127.0.0.1", "testserver"]
    docs_enabled: bool = True
    mfa_required: bool = False
    mfa_issuer: str = "THINKING EEG Platform"
    max_request_bytes: int = 1_048_576
    max_eeg_channels: int = 64
    max_eeg_times: int = 2500
    max_anchors: int = 20
    max_anchor_surface: int = 128
    max_prompt_bytes: int = 65_536
    max_output_chars: int = 4_000
    llm_enabled: bool = False
    llm_allowed_tenants: list[str] = []
    llm_model: str = "claude-opus-5"
    llm_model_allowlist: list[str] = ["claude-opus-5"]
    llm_timeout_seconds: float = 10.0
    llm_max_output_tokens: int = 256
    anthropic_api_key: str | None = None
    siem_endpoint: str | None = None
    siem_token: str | None = None

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    def model_post_init(self, __context) -> None:
        vault_jwt_secret = get_secret("JWT_SECRET")
        if vault_jwt_secret:
            object.__setattr__(self, "jwt_secret", vault_jwt_secret)
        if not self.siem_token:
            object.__setattr__(self, "siem_token", get_secret("SIEM_TOKEN"))
        using_default = self.jwt_secret == _DEFAULT_JWT_SECRET
        production = str(self.env).lower() == "production"
        if production and using_default:
            raise ValueError("JWT_SECRET must be set in production (environment or Vault; fail-closed)")
        if production and not self.redis_url:
            raise ValueError(
                "REDIS_URL must be configured in production for shared MFA, revocation, and rate limiting"
            )
        if production and not self.mfa_required:
            raise ValueError("MFA_REQUIRED must be enabled in production")
        if using_default and not _in_pytest() and os.environ.get("THINKING_ALLOW_DEV_SECRET") != "1":
            # Ephemeral per-process secret so the source-code default cannot forge tokens.
            object.__setattr__(self, "jwt_secret", secrets.token_hex(32))


settings = Settings()
