"""Detached Ed25519 signing for family-release JSON documents."""
from __future__ import annotations

from pathlib import Path

import pytest


def make_ed25519_pem_pair() -> tuple[bytes, bytes]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    public_pem = private.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    return private_pem, public_pem


def _pem_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*.pem") if path.is_file()}


def test_canonical_bytes_are_compact_sorted_utf8():
    from src.data.release.signing import canonical_bytes

    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert canonical_bytes({"name": "tín"}) == '{"name":"tín"}'.encode("utf-8")


def test_signature_round_trip_and_tamper_failure(tmp_path):
    from src.data.release.signing import write_detached_signature, verify_detached_signature

    private_key, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1"}\n', encoding="utf-8")
    signature = write_detached_signature(document, private_key)

    assert signature == tmp_path / "release_manifest.json.ed25519.sig"
    assert signature.read_bytes()
    assert len(signature.read_bytes()) == 64
    assert verify_detached_signature(document, signature, public_key)
    document.write_text('{"release_id":"tampered"}\n', encoding="utf-8")
    assert not verify_detached_signature(document, signature, public_key)


def test_missing_signature_fails_closed(tmp_path):
    from src.data.release.signing import verify_detached_signature

    _, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1"}\n', encoding="utf-8")
    missing = tmp_path / "release_manifest.json.ed25519.sig"

    assert not missing.exists()
    assert verify_detached_signature(document, missing, public_key) is False


def test_wrong_public_key_fails_closed(tmp_path):
    from src.data.release.signing import write_detached_signature, verify_detached_signature

    private_key, _ = make_ed25519_pem_pair()
    _, other_public = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1"}\n', encoding="utf-8")
    signature = write_detached_signature(document, private_key)

    assert verify_detached_signature(document, signature, other_public) is False


def test_malformed_signature_fails_closed(tmp_path):
    from src.data.release.signing import verify_detached_signature

    _, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1"}\n', encoding="utf-8")
    malformed = tmp_path / "release_manifest.json.ed25519.sig"
    malformed.write_bytes(b"not-a-valid-ed25519-signature")

    assert verify_detached_signature(document, malformed, public_key) is False


def test_whitespace_only_json_change_still_verifies(tmp_path):
    from src.data.release.signing import write_detached_signature, verify_detached_signature

    private_key, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1","files":[1]}\n', encoding="utf-8")
    signature = write_detached_signature(document, private_key)
    document.write_text(
        '{\n  "files": [\n    1\n  ],\n  "release_id": "r1"\n}\n',
        encoding="utf-8",
    )

    assert verify_detached_signature(document, signature, public_key) is True


def test_semantic_json_change_fails(tmp_path):
    from src.data.release.signing import write_detached_signature, verify_detached_signature

    private_key, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1","count":1}\n', encoding="utf-8")
    signature = write_detached_signature(document, private_key)
    document.write_text('{"release_id":"r1","count":2}\n', encoding="utf-8")

    assert verify_detached_signature(document, signature, public_key) is False


def test_private_key_never_written_to_disk(tmp_path):
    from src.data.release.signing import write_detached_signature, verify_detached_signature

    worktree = Path(__file__).resolve().parents[1]
    watched = (
        tmp_path,
        worktree / "src" / "data" / "release",
        worktree / "tests",
        worktree / "scripts",
    )
    before = {path for root in watched for path in _pem_files(root)}

    private_key, public_key = make_ed25519_pem_pair()
    document = tmp_path / "release_manifest.json"
    document.write_text('{"release_id":"r1"}\n', encoding="utf-8")
    signature = write_detached_signature(document, private_key)

    assert verify_detached_signature(document, signature, public_key) is True
    after = {path for root in watched for path in _pem_files(root)}
    assert after == before
    assert list(tmp_path.glob("*.pem")) == []
    for path in tmp_path.rglob("*"):
        if path.is_file():
            payload = path.read_bytes()
            assert b"BEGIN PRIVATE KEY" not in payload
            assert b"BEGIN OPENSSH PRIVATE KEY" not in payload


def test_sign_document_round_trip_and_tamper():
    from src.data.release.signing import sign_document, verify_document

    private_key, public_key = make_ed25519_pem_pair()
    document = {"release_id": "r1", "b": 2, "a": 1}
    signature = sign_document(document, private_key)

    assert verify_document(document, signature, public_key) is True
    assert verify_document({"a": 1, "b": 2, "release_id": "r1"}, signature, public_key) is True
    assert verify_document({"release_id": "r2", "b": 2, "a": 1}, signature, public_key) is False


def test_sign_rejects_public_pem_and_hmac_secret():
    from src.data.release.signing import sign_document

    _, public_key = make_ed25519_pem_pair()
    document = {"release_id": "r1"}

    with pytest.raises((TypeError, ValueError)):
        sign_document(document, public_key)
    with pytest.raises((TypeError, ValueError)):
        sign_document(document, b"not-a-pem-hmac-secret")


def test_verify_rejects_private_pem_and_hmac_secret():
    from src.data.release.signing import sign_document, verify_document

    private_key, public_key = make_ed25519_pem_pair()
    document = {"release_id": "r1"}
    signature = sign_document(document, private_key)

    assert verify_document(document, signature, public_key) is True
    assert verify_document(document, signature, private_key) is False
    assert verify_document(document, signature, b"not-a-pem-hmac-secret") is False


def test_verify_io_and_parse_errors_fail_closed(tmp_path):
    from src.data.release.signing import verify_detached_signature, verify_document

    _, public_key = make_ed25519_pem_pair()
    missing_document = tmp_path / "missing.json"
    signature = tmp_path / "missing.json.ed25519.sig"
    signature.write_bytes(b"\x00" * 64)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")

    assert verify_detached_signature(missing_document, signature, public_key) is False
    assert verify_detached_signature(invalid, signature, public_key) is False
    assert verify_document({"release_id": "r1"}, b"\x00" * 64, public_key) is False
    assert verify_document({"release_id": object()}, b"\x00" * 64, public_key) is False
