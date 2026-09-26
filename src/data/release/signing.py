"""Detached Ed25519 signatures over canonical family-release JSON."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_CRYPTO_REQUIRED = "cryptography is required for Ed25519 release signing"


def canonical_bytes(document: Mapping[str, object]) -> bytes:
    """Compact UTF-8 JSON with sorted keys; independent of pretty-print layout."""
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _as_bytes(value: object, label: str) -> bytes:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError(f"{label} must be bytes")
    return value


def _require_private_pem(private_key_pem: bytes) -> bytes:
    blob = _as_bytes(private_key_pem, "private_key_pem")
    if b"BEGIN" not in blob or b"PRIVATE KEY" not in blob:
        raise ValueError("signing accepts only PEM private keys")
    return blob


def _is_public_pem(public_key_pem: object) -> bool:
    if isinstance(public_key_pem, memoryview):
        public_key_pem = public_key_pem.tobytes()
    if isinstance(public_key_pem, bytearray):
        public_key_pem = bytes(public_key_pem)
    if not isinstance(public_key_pem, bytes):
        return False
    if b"PRIVATE KEY" in public_key_pem:
        return False
    return b"BEGIN PUBLIC KEY" in public_key_pem or b"BEGIN ED25519 PUBLIC KEY" in public_key_pem


def _load_ed25519_private(private_key_pem: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as exc:
        raise RuntimeError(_CRYPTO_REQUIRED) from exc
    key = load_pem_private_key(_require_private_pem(private_key_pem), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("signing requires an Ed25519 PEM private key")
    return key


def _load_ed25519_public(public_key_pem: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as exc:
        raise RuntimeError(_CRYPTO_REQUIRED) from exc
    if not _is_public_pem(public_key_pem):
        raise ValueError("verification accepts only PEM public keys")
    key = load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("verification requires an Ed25519 PEM public key")
    return key


def _sidecar_path(document_path: Path) -> Path:
    path = Path(document_path)
    return path.with_name(path.name + ".ed25519.sig")


def sign_document(document: Mapping[str, object], private_key_pem: bytes) -> bytes:
    """Return a raw Ed25519 signature over canonical JSON. Private key stays in memory."""
    key = _load_ed25519_private(private_key_pem)
    return key.sign(canonical_bytes(document))


def verify_document(
    document: Mapping[str, object], signature: bytes, public_key_pem: bytes
) -> bool:
    """Verify a raw Ed25519 signature. Any I/O, parse, or crypto error is False."""
    try:
        from cryptography.exceptions import InvalidSignature
    except ImportError:
        return False
    try:
        key = _load_ed25519_public(_as_bytes(public_key_pem, "public_key_pem"))
        payload = canonical_bytes(document)
        key.verify(_as_bytes(signature, "signature"), payload)
        return True
    except (InvalidSignature, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    except Exception:
        return False


def write_detached_signature(document_path: Path, private_key_pem: bytes) -> Path:
    """Sign the JSON object at *document_path* and write `<name>.ed25519.sig` beside it."""
    path = Path(document_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("document must be a JSON object")
    sidecar = _sidecar_path(path)
    sidecar.write_bytes(sign_document(loaded, private_key_pem))
    return sidecar


def verify_detached_signature(
    document_path: Path, signature_path: Path, public_key_pem: bytes
) -> bool:
    """Verify a detached sidecar. Missing, malformed, or untrusted input is False."""
    try:
        loaded = json.loads(Path(document_path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return False
        signature = Path(signature_path).read_bytes()
    except (OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    except Exception:
        return False
    return verify_document(loaded, signature, public_key_pem)
