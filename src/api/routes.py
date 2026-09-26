"""API routes — bounded decode, MFA, ingestion validation, and audit metadata."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from src.api.mfa import enroll, require_mfa_claim, verify
from src.api.security import create_token, enforce_ownership, require_auth
from src.anchors.encoder import AnchorEncoder
from src.anchors.retrieval import retrieve_anchors
from src.anchors.vocab import load_vocab
from src.config import settings
from src.kg.ontology import load_ontology
from src.kg.resolver import resolve_anchors
from src.preprocessing.io import safe_resolve, validate_extension
from src.prompt.compiler import compile_prompt
from src.prompt.generator import generate_text_result
from src.prompt.validator import validate_prompt

router = APIRouter()
_vocab = load_vocab()
_ontology = load_ontology(with_text_extension=True)

_EEG_ROW = Annotated[
    list[FiniteFloat],
    Field(min_length=1, max_length=settings.max_eeg_times),
]
_EEG_MATRIX = Annotated[
    list[_EEG_ROW],
    Field(min_length=1, max_length=settings.max_eeg_channels),
]
_EEG_BATCH = Annotated[
    list[_EEG_MATRIX],
    Field(min_length=1, max_length=1),
]


class MFASetupResponse(BaseModel):
    secret: str
    otpauth_uri: str
    backup_codes: list[str]


class MFAVerifyRequest(BaseModel):
    code: str = Field(min_length=4, max_length=32)


class MFAVerifyResponse(BaseModel):
    verified: bool
    issued_token: str | None = None


class AnchorOverride(BaseModel):
    """Client-controlled anchor input; provenance is always server-assigned."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    surface: str = Field(min_length=1, max_length=settings.max_anchor_surface)
    score: FiniteFloat = Field(default=0.5, ge=0.0, le=1.0)
    rank: int = Field(default=1, ge=1, le=settings.max_anchors)


class DecodeRequest(BaseModel):
    eeg: _EEG_MATRIX | _EEG_BATCH | None = Field(default=None)
    session_id: str | None = Field(default=None, max_length=128)
    owner_id: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)
    anchors_override: list[AnchorOverride] | None = Field(
        default=None,
        max_length=settings.max_anchors,
    )
    modality: Literal["image", "text", "music", "video", "3d"] = "text"
    use_llm: bool = False

    @model_validator(mode="after")
    def _validate_input_contract(self) -> "DecodeRequest":
        has_eeg = self.eeg is not None
        has_override = self.anchors_override is not None
        if has_eeg == has_override:
            raise ValueError("provide exactly one of eeg or anchors_override")
        if has_eeg:
            try:
                arr = np.asarray(self.eeg, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("eeg must be a rectangular numeric array") from exc
            if arr.ndim == 2:
                channels, times = arr.shape
            elif arr.ndim == 3 and arr.shape[0] == 1:
                _, channels, times = arr.shape
            else:
                raise ValueError("eeg must be 2D or a single-item 3D array")
            if channels > settings.max_eeg_channels or times > settings.max_eeg_times:
                raise ValueError("eeg dimensions exceed policy")
            if not np.isfinite(arr).all():
                raise ValueError("eeg values must be finite")
        return self


class DecodeResponse(BaseModel):
    anchors: list[dict]
    kg_subgraph: dict
    structured_prompt: dict
    text: str
    provenance: list[str]
    provider_status: str = "local_template"
    provider: str = "local"
    model: str | None = None


class BatchDecodeItem(BaseModel):
    eeg: _EEG_MATRIX | _EEG_BATCH | None = Field(default=None)
    session_id: str | None = Field(default=None, max_length=128)
    anchors_override: list[AnchorOverride] | None = Field(default=None, max_length=settings.max_anchors)
    modality: Literal["image", "text", "music", "video", "3d"] = "text"
    use_llm: bool = False

    @model_validator(mode="after")
    def _validate_item(self) -> "BatchDecodeItem":
        has_eeg = self.eeg is not None
        has_override = self.anchors_override is not None
        if has_eeg == has_override:
            raise ValueError("provide exactly one of eeg or anchors_override")
        if has_eeg:
            try:
                arr = np.asarray(self.eeg, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("eeg must be a rectangular numeric array") from exc
            if arr.ndim == 2:
                channels, times = arr.shape
            elif arr.ndim == 3 and arr.shape[0] == 1:
                _, channels, times = arr.shape
            else:
                raise ValueError("eeg must be 2D or a single-item 3D array")
            if channels > settings.max_eeg_channels or times > settings.max_eeg_times:
                raise ValueError("eeg dimensions exceed policy")
            if not np.isfinite(arr).all():
                raise ValueError("eeg values must be finite")
        return self


class BatchDecodeRequest(BaseModel):
    items: list[BatchDecodeItem] = Field(min_length=1, max_length=8)
    owner_id: str | None = Field(default=None, max_length=128)
    tenant_id: str | None = Field(default=None, max_length=128)


class BatchDecodeResponse(BaseModel):
    results: list[DecodeResponse]
    batch_size: int


def _bind_owner(req_owner: str | None, req_tenant: str | None, user: dict) -> None:
    owner = req_owner or user.get("sub")
    tenant = req_tenant or user.get("tenant_id", "default")
    enforce_ownership(owner, tenant, user)


def _prompt_hash(prompt: dict) -> str:
    encoded = json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provider_allowed(user: dict) -> bool:
    tenant = str(user.get("tenant_id", "default"))
    return (
        settings.llm_enabled
        and bool(settings.anthropic_api_key)
        and settings.llm_model in settings.llm_model_allowlist
        and tenant in set(settings.llm_allowed_tenants)
    )


@router.get("/health")
async def health():
    return {"status": "ok", "ontology_version": _ontology.ontology_version, "vocab_size": len(_vocab)}


@router.post("/auth/mfa/setup", response_model=MFASetupResponse)
async def mfa_setup(request: Request, response: Response, user: dict = Depends(require_auth)):
    request.state.tenant_id = user.get("tenant_id", "default")
    response.headers["Cache-Control"] = "no-store"
    data = enroll(user)
    request.state.audit_event_type = "mfa_setup"
    return MFASetupResponse(**data)


@router.post("/auth/mfa/verify", response_model=MFAVerifyResponse)
async def mfa_verify(
    payload: MFAVerifyRequest,
    request: Request,
    response: Response,
    user: dict = Depends(require_auth),
):
    request.state.tenant_id = user.get("tenant_id", "default")
    ok = verify(user, payload.code)
    response.headers["Cache-Control"] = "no-store"
    if not ok:
        request.state.audit_event_type = "mfa_verification_failed"
        return MFAVerifyResponse(verified=False, issued_token=None)
    # Issue a fresh elevated token; the original bearer remains valid for ordinary
    # authentication so a failed/reused backup code is reported as a verification
    # failure rather than an unrelated token-expiry response.
    issued = create_token(user["sub"], tenant_id=user.get("tenant_id", "default"), mfa_verified=True)
    request.state.audit_event_type = "mfa_verification_succeeded"
    return MFAVerifyResponse(verified=True, issued_token=issued)


@router.post("/decode", response_model=DecodeResponse)
async def decode(req: DecodeRequest, request: Request, user: dict = Depends(require_auth)):
    _bind_owner(req.owner_id, req.tenant_id, user)
    require_mfa_claim(user)
    request.state.user_sub = user.get("sub", "-")
    request.state.tenant_id = user.get("tenant_id", "default")
    request.state.anchor_count = len(req.anchors_override or [])
    request.state.data_egress = bool(req.use_llm)

    if req.use_llm and not _provider_allowed(user):
        request.state.provider_status = "blocked_policy"
        request.state.audit_event_type = "provider_egress_blocked"
        raise HTTPException(status_code=403, detail="External generation is not enabled for this tenant")

    anchors: list[dict] = []
    if req.anchors_override is not None:
        for a in req.anchors_override:
            raw = a.surface
            try:
                sanitized = _vocab.validate_or_raise(raw)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Anchor not in vocab allowlist") from exc
            anchors.append(
                {
                    "anchor_id": f"anchor:{sanitized.replace(' ', '_')}",
                    "surface": sanitized,
                    "score": float(a.score),
                    "rank": int(a.rank),
                    "provenance": ["eeg:epoch_override"],
                }
            )
    elif req.eeg is not None:
        arr = np.asarray(req.eeg, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.ndim != 3 or arr.shape[0] != 1 or not np.isfinite(arr).all():
            raise HTTPException(status_code=400, detail="eeg must be a finite 2D or single-item 3D array")
        enc = AnchorEncoder(n_channels=arr.shape[1], n_times=arr.shape[2])
        emb = enc.encode(arr)[0]
        epoch_id = f"eeg:epoch_{req.session_id or 'single'}"
        retrieved = retrieve_anchors(emb, _vocab.keywords, top_k=min(5, settings.max_anchors), epoch_id=epoch_id)
        anchors = [
            {
                "anchor_id": r.anchor_id,
                "surface": r.surface,
                "score": r.score,
                "rank": r.rank,
                "provenance": list(r.provenance),
            }
            for r in retrieved
        ]
    else:  # guarded by the request validator
        raise HTTPException(status_code=400, detail="Provide eeg or anchors_override")

    try:
        subgraph = resolve_anchors(anchors, _ontology)
        prompt = compile_prompt(anchors, subgraph, modality=req.modality)
        validate_prompt(prompt)
        serialized_prompt = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(serialized_prompt) > settings.max_prompt_bytes:
            raise ValueError("structured prompt exceeds size policy")
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail="Invalid anchors or prompt") from exc
    except AssertionError as exc:
        raise HTTPException(status_code=400, detail="Structured prompt failed schema validation") from exc

    request.state.prompt_hash = _prompt_hash(prompt)
    request.state.anchor_count = len(anchors)
    request.state.ontology_version = subgraph.ontology_version
    result = generate_text_result(
        prompt,
        anchors,
        use_llm=req.use_llm,
        tenant_id=str(user.get("tenant_id", "default")),
    )
    request.state.provider_status = result.status
    request.state.provider = result.provider
    request.state.model = result.model
    if result.status == "provider_failed":
        request.state.audit_event_type = "provider_generation_failed"
        raise HTTPException(status_code=502, detail="External generation failed")

    provenance = [p for a in anchors for p in a["provenance"]] + [f"kg:{subgraph.subgraph_id}"]
    return DecodeResponse(
        anchors=anchors,
        kg_subgraph={
            "subgraph_id": subgraph.subgraph_id,
            "ontology_version": subgraph.ontology_version,
            "nodes": subgraph.nodes,
            "edges": subgraph.edges,
            "resolved": [r.__dict__ for r in subgraph.resolved],
        },
        structured_prompt=prompt,
        text=result.text,
        provenance=provenance,
        provider_status=result.status,
        provider=result.provider,
        model=result.model,
    )


@router.post("/decode/batch", response_model=BatchDecodeResponse)
async def decode_batch(
    req: BatchDecodeRequest,
    request: Request,
    user: dict = Depends(require_auth),
):
    """Decode a bounded batch while retaining single-item security semantics."""
    _bind_owner(req.owner_id, req.tenant_id, user)
    require_mfa_claim(user)
    request.state.user_sub = user.get("sub", "-")
    request.state.tenant_id = user.get("tenant_id", "default")
    request.state.batch_size = len(req.items)
    request.state.data_egress = any(item.use_llm for item in req.items)

    if any(item.use_llm for item in req.items) and not _provider_allowed(user):
        request.state.provider_status = "blocked_policy"
        request.state.audit_event_type = "provider_egress_blocked"
        raise HTTPException(status_code=403, detail="External generation is not enabled for this tenant")

    results: list[DecodeResponse] = []
    prompt_hashes: list[str] = []
    for item in req.items:
        # Reuse the single-item route's validated, provenance-preserving flow
        # without issuing a second HTTP request or bypassing ownership policy.
        if item.anchors_override is not None:
            anchors: list[dict] = []
            for anchor in item.anchors_override:
                try:
                    sanitized = _vocab.validate_or_raise(anchor.surface)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="Anchor not in vocab allowlist") from exc
                anchors.append({
                    "anchor_id": f"anchor:{sanitized.replace(' ', '_')}",
                    "surface": sanitized,
                    "score": float(anchor.score),
                    "rank": int(anchor.rank),
                    "provenance": ["eeg:epoch_override"],
                })
        else:
            arr = np.asarray(item.eeg, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[None, :, :]
            if arr.ndim != 3 or arr.shape[0] != 1 or not np.isfinite(arr).all():
                raise HTTPException(status_code=400, detail="eeg must be a finite 2D or single-item 3D array")
            enc = AnchorEncoder(n_channels=arr.shape[1], n_times=arr.shape[2])
            emb = enc.encode(arr)[0]
            epoch_id = f"eeg:epoch_{item.session_id or 'batch'}"
            retrieved = retrieve_anchors(emb, _vocab.keywords, top_k=min(5, settings.max_anchors), epoch_id=epoch_id)
            anchors = [
                {
                    "anchor_id": result.anchor_id,
                    "surface": result.surface,
                    "score": result.score,
                    "rank": result.rank,
                    "provenance": list(result.provenance),
                }
                for result in retrieved
            ]

        try:
            subgraph = resolve_anchors(anchors, _ontology)
            prompt = compile_prompt(anchors, subgraph, modality=item.modality)
            validate_prompt(prompt)
            serialized_prompt = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(serialized_prompt) > settings.max_prompt_bytes:
                raise ValueError("structured prompt exceeds size policy")
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(status_code=400, detail="Invalid anchors or prompt") from exc
        except AssertionError as exc:
            raise HTTPException(status_code=400, detail="Structured prompt failed schema validation") from exc

        result = generate_text_result(
            prompt,
            anchors,
            use_llm=item.use_llm,
            tenant_id=str(user.get("tenant_id", "default")),
        )
        if result.status == "provider_failed":
            request.state.audit_event_type = "provider_generation_failed"
            raise HTTPException(status_code=502, detail="External generation failed")
        prompt_hashes.append(_prompt_hash(prompt))
        provenance = [value for anchor in anchors for value in anchor["provenance"]] + [f"kg:{subgraph.subgraph_id}"]
        results.append(DecodeResponse(
            anchors=anchors,
            kg_subgraph={
                "subgraph_id": subgraph.subgraph_id,
                "ontology_version": subgraph.ontology_version,
                "nodes": subgraph.nodes,
                "edges": subgraph.edges,
                "resolved": [resolved.__dict__ for resolved in subgraph.resolved],
            },
            structured_prompt=prompt,
            text=result.text,
            provenance=provenance,
            provider_status=result.status,
            provider=result.provider,
            model=result.model,
        ))

    request.state.anchor_count = sum(len(result.anchors) for result in results)
    request.state.batch_prompt_hashes = prompt_hashes
    request.state.provider_status = ",".join(result.provider_status for result in results)
    return BatchDecodeResponse(results=results, batch_size=len(results))


@router.post("/ingest/validate")
async def ingest_validate(
    request: Request,
    source_path: str = Query(min_length=1, max_length=512),
    owner_id: str | None = Query(default=None, max_length=128),
    tenant_id: str | None = Query(default=None, max_length=128),
    user: dict = Depends(require_auth),
):
    _bind_owner(owner_id, tenant_id, user)
    require_mfa_claim(user)
    request.state.user_sub = user.get("sub", "-")
    request.state.tenant_id = user.get("tenant_id", "default")
    try:
        p = safe_resolve(source_path)
        validate_extension(p)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path traversal blocked") from exc
    request.state.audit_event_type = "ingest_validation"
    return {"resolved": p.name, "exists": p.exists()}
