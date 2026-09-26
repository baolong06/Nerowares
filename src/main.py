"""FastAPI app — THINKING API with production security defaults."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import correlation_middleware
from src.api.ratelimit import rate_limit_middleware
from src.api.routes import router
from src.config import settings

_is_production = str(settings.env).lower() == "production"
# OpenAPI is disabled by default; local developers must opt in explicitly.
_docs_url = "/docs" if settings.docs_enabled and not _is_production and settings.env == "development" else None
_redoc_url = "/redoc" if settings.docs_enabled and not _is_production and settings.env == "development" else None
_openapi_url = "/openapi.json" if settings.docs_enabled and not _is_production and settings.env == "development" else None

app = FastAPI(
    title="THINKING EEG Platform",
    version="0.2.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

if settings.trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(correlation_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Correlation-ID"],
)
app.include_router(router)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    if request.url.scheme == "https" or _is_production:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": "Bad request"})


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": "0.2.0"}
