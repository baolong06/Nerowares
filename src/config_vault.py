"""Secret resolution: environment first, HashiCorp Vault KV second.

Vault is optional for local development. Production callers must reject a missing
secret rather than substituting a source-controlled default.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=32)
def get_secret(name: str) -> str | None:
    """Return an environment or Vault KV secret without logging its value."""
    candidates = (name, name.upper(), name.lower())
    for candidate in candidates:
        value = os.environ.get(candidate)
        if value:
            return value

    addr = os.environ.get("VAULT_ADDR")
    token = os.environ.get("VAULT_TOKEN")
    if not addr or not token:
        return None
    try:
        import hvac

        client = hvac.Client(url=addr, token=token)
        if not client.is_authenticated():
            if str(os.environ.get("THINKING_ENV", "")).lower() == "production":
                raise RuntimeError("Vault authentication failed in production")
            return None
        mount = os.environ.get("VAULT_KV_MOUNT", "secret")
        path = os.environ.get("VAULT_KV_PATH", "thinking")
        response: dict[str, Any] = client.secrets.kv.v2.read_secret_version(
            mount_point=mount,
            path=path,
            raise_on_deleted_version=True,
        )
        data = response.get("data", {}).get("data", {})
        for candidate in candidates:
            value = data.get(candidate)
            if isinstance(value, str) and value:
                return value
    except Exception as exc:
        if str(os.environ.get("THINKING_ENV", "")).lower() == "production" and (addr or token):
            raise RuntimeError("Vault is unreachable or misconfigured in production") from exc
        # Do not expose Vault hostnames, tokens, or response detail to logs.
        return None
    return None
