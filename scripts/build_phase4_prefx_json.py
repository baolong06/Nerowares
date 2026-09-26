import copy, json
from pathlib import Path

root = Path(".")
source = root / "reports" / "Pre-Fix-AppSec-Assessment-Phase3-2026-09-22.json"
out = root / "reports" / "Pre-Fix-AppSec-Assessment-Phase4-2026-09-22.json"
report = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
report["meta"].update({
    "title": "Pre-Fix Security Assessment Report — Phase 4 API and Structured-Prompt Platform",
    "subtitle": "THINKING EEG Platform — API, knowledge graph, provenance, and LLM generation hardening",
    "date": "2026-09-22",
})
report["hard_stop"] = (
    "This report is the Pre-Fix gate for Phase 4 API, structured-prompt, knowledge-graph, "
    "and retrieval-grounded generation work. No Phase 4 source, dependency, API route, "
    "security control, or provider configuration was modified. Implementation waits for "
    "explicit approval on the next user turn."
)
report["executive_summary"] = (
    "Phase 3 dataset contracts and anti-leakage identities are now implemented and verified, "
    "but Phase 4 exposes the authenticated /decode pipeline to EEG arrays, anchor overrides, "
    "knowledge-graph resolution, strict structured-prompt compilation, and an optional Anthropic "
    "provider path. The current code preserves EEG-versus-KG provenance and validates the JSON "
    "schema, yet request sizes, provider egress, output/error semantics, durable MFA, security "
    "headers, and shared rate limiting remain residual risks. Existing Critical/High supply-chain "
    "and authentication findings remain open. Mean domain score 5.09 / 10.0."
)
report["scope"] = [
    "FastAPI /decode, /ingest/validate, health, MFA setup and verification routes",
    "JWT verification, ownership/tenant binding, MFA elevation, CORS, correlation, rate limiting, and audit telemetry",
    "EEG array and anchor_override validation, retrieval, knowledge-graph resolution, and provenance boundaries",
    "Structured-prompt compiler and Draft 2020-12 validator",
    "Optional Anthropic provider path, prompt-injection boundary, data egress, timeout, and fallback behavior",
    "Phase 3 manifest/loader integration boundaries and validated npz artifact handling",
    "CVE/KEV, Gitleaks, SBOM, Vault, and Sigma/KQL/SPL control readiness",
]
report["out_of_scope"] = [
    "Applying any Phase 4 code or dependency change before explicit approval",
    "Calling an external Anthropic endpoint or probing third-party systems",
    "Production deployment, reverse-proxy configuration, Redis provisioning, or identity-provider migration",
    "Claiming EEG-to-text semantic accuracy beyond the locked Phase 2/3 retrieval metrics",
    "Pooling D01, Chisco, D09, SparrKULee, or Brennan results",
]
report["evidence"] = [
    "Phase 3 loader and manifest verification is complete; artifacts/phase3_manifest.json records per-dataset contracts and split outcomes",
    "PYTHONPATH=. pytest -q: 58 passed, 48 warnings (2026-09-22); warnings are known MNE/ICA and Windows temp cleanup warnings",
    "src/api/routes.py exposes authenticated /decode and /ingest/validate; /decode accepts unbounded list/dict payloads before np.asarray or prompt serialization",
    "src/prompt/validator.py enforces the structured_prompt.schema.json contract and src/kg/resolver.py separates eeg:* accepted provenance from kg:edge:* suggestions",
    "src/prompt/generator.py sends structured prompt and accepted anchors to Anthropic when ANTHROPIC_API_KEY is set, catches all provider exceptions, and falls back to a local template",
    "src/api/mfa.py stores enrollment only in process memory and src/config.py defaults mfa_required=False outside production enforcement",
    "Pinned ML/runtime extras still include torch==2.5.1, starlette==0.41.3 via FastAPI, and pyarrow==15.0.2; prior pip-audit findings remain open",
    "references/cve-library/ is absent and the existing report records the CISA KEV torch issue; no new external dependency scan was performed in this pre-fix gate",
]
report["domains"] = [
    {"id": 1, "name": "Authentication", "score": 5.5, "rationale": "JWT authentication and TOTP hooks exist, but MFA is optional by default and enrollment is process-local; Phase 4 adds a high-value decode/provider surface."},
    {"id": 2, "name": "Authorization / Access Control", "score": 6.5, "rationale": "Owner and tenant binding exists for decode and ingest, but provider egress, artifact ownership, and dataset-family authorization are not modeled as policy decisions."},
    {"id": 3, "name": "Session / Token Management", "score": 6.0, "rationale": "Issuer, audience, expiry, nbf, jti, and algorithm checks are present, but there is no revocation/replay store and MFA elevation is stateless."},
    {"id": 4, "name": "Input Validation & Injection", "score": 4.5, "rationale": "Vocabulary and JSON schema validation reduce prompt injection, but EEG arrays, anchor dictionaries, scores, ranks, and serialized prompts lack comprehensive bounds and finite-number checks."},
    {"id": 5, "name": "File Handling / Path Traversal", "score": 6.0, "rationale": "safe_resolve and Phase 3 root-scoped loaders are strong controls, but /ingest/validate still exposes existence behavior and Phase 4 artifact/provider boundaries need explicit policy."},
    {"id": 6, "name": "Cryptography", "score": 6.0, "rationale": "HMAC-SHA256, constant-time comparisons, hashed backup codes, and SHA-256 provenance exist; custom JWT parsing and unsigned provider/artifact metadata remain residual concerns."},
    {"id": 7, "name": "Error Handling & Information Disclosure", "score": 4.5, "rationale": "Generic API errors and root-name suppression exist, but provider failures are silently converted to local output and route-level distinctions can expose operational state without audit context."},
    {"id": 8, "name": "Logging, Monitoring & Telemetry", "score": 5.5, "rationale": "Correlation and allowlisted SIEM forwarding exist, but semantic decode events lack provider/model, prompt hash, anchor count, ontology version, output status, and data-egress fields."},
    {"id": 9, "name": "Transport, CORS & Security Headers", "score": 4.0, "rationale": "CORS is narrowly configured, but docs/OpenAPI are public by default and TrustedHost, HSTS, CSP, frame, MIME, and referrer controls are absent."},
    {"id": 10, "name": "Configuration, Secrets & Supply Chain", "score": 3.5, "rationale": "Vault hooks and fail-closed production JWT secret checks exist, but torch KEV, Starlette/pyarrow advisories, absent local CVE library, and optional external provider configuration remain open."},
    {"id": 11, "name": "Availability, Rate Limiting & Business Logic", "score": 4.0, "rationale": "Per-process rate limiting exists, but unbounded JSON/EEG and LLM work can exhaust memory or provider quota; test-mode bypass and multi-worker multiplication remain."},
]
report["dfd"] = {
    "entities": ["Client", "Operator", "Identity/MFA user", "Phase 3 dataset roots", "Anthropic API (optional)", "HashiCorp Vault", "SIEM"],
    "processes": ["FastAPI (CORS, rate limit, correlation)", "JWT/MFA and owner/tenant policy", "/decode", "/ingest/validate", "AnchorEncoder and retrieval", "KG resolver", "Structured-prompt compiler/validator", "Template or Anthropic generation", "Phase 3 native-rate loaders/evaluators", "npz artifact loader"],
    "stores": ["Process-local MFA dict", "In-memory rate buckets", "Client-held JWT", "ontology_seed.json", "encoder.npz", "Phase 3 manifests and dataset roots", "Vault KV", "SIEM audit stream"],
    "trust_boundaries": ["Internet/client to FastAPI", "JWT claims to owner/tenant policy", "API to local EEG/filesystem", "API to optional Anthropic provider", "Provider response to local template fallback", "API to Vault", "API to SIEM", "Subject/session identity to evaluator", "Artifact file to encoder runtime"],
}
report["stride"] = [
    {"element": "Authenticated /decode and MFA", "S": "Stolen JWT or MFA elevation token", "T": "Owner/tenant claim or MFA state manipulation", "R": "No revocation or durable enrollment audit", "I": "EEG, anchors, and subject/session metadata", "D": "Repeated large decode requests", "E": "Abuse of optional MFA or stateless claims"},
    {"element": "Structured prompt and KG provenance", "S": "Client submits spoofed anchor metadata", "T": "Accepted EEG provenance or KG suggestion boundary is altered", "R": "Output cannot be tied to prompt/ontology version", "I": "Raw EEG or sensitive semantic intent in logs/provider payloads", "D": "Oversized anchor graph/prompt", "E": "Prompt injection or untrusted structured data treated as instructions"},
    {"element": "Optional Anthropic generation", "S": "Compromised provider credential or endpoint", "T": "Provider response/metadata tampering", "R": "Silent fallback hides provider failure", "I": "Structured prompt and accepted EEG anchors leave the trust boundary", "D": "Provider quota exhaustion/timeouts", "E": "Unbounded or misconfigured model/provider invocation"},
    {"element": "Phase 3 loaders and artifacts", "S": "Planted dataset or artifact under a writable root", "T": "Manifest, labels, or encoder replacement", "R": "Unreproducible split/provenance result", "I": "EEG and participant identifiers", "D": "Large parser or fold workload", "E": "Unsafe deserialization or false scientific claim"},
]
for finding in report["findings"]:
    if finding["id"] == "TW-2026-004":
        finding.update({
            "title": "Phase 4 structured decode/LLM path lacks bounded input, egress, and provenance contract",
            "component": "src/api/routes.py /decode; src/prompt/compiler.py; src/prompt/generator.py; optional Anthropic provider",
            "exploitability_here": "An authenticated caller can submit unbounded EEG arrays or repeated allowlisted anchor dictionaries. When ANTHROPIC_API_KEY is configured, the structured prompt and accepted anchors cross the provider trust boundary without an explicit data-classification, redaction, timeout, output-size, or tenant egress policy. Any provider exception is silently converted to a local template.",
            "poc": "With a valid JWT, POST /decode using an oversized nested eeg list or a large anchors_override list containing repeated allowlisted surfaces. Set ANTHROPIC_API_KEY and observe that prompt serialization/provider invocation is selected without a request-size contract; force a provider exception and observe a successful-looking template fallback without a provider-failure event.",
            "remediation": "Bound JSON body, channel/time dimensions, anchor count/surface length, numeric finiteness, and serialized prompt size at Pydantic and proxy layers. Make external generation explicit and tenant-authorized; redact or classify EEG-derived data, allowlist provider/model, set timeout/output limits, audit provider status and prompt hash without raw EEG, and distinguish provider failure from a successful local decode.",
            "cwe": "CWE-20",
            "vector": "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:H",
            "cvss": 8.1,
        })
report["planned_change_constraints"] = [
    "Do not modify Phase 4 source, dependency pins, routes, or provider settings until this report is explicitly approved.",
    "Cap request bytes, EEG channels/times, batch size, anchor count, surface length, rank/score ranges, and serialized prompt size; reject NaN/Infinity and malformed nested arrays.",
    "Keep accepted concepts restricted to eeg:* provenance and KG suggestions restricted to kg:edge:*; validate the final structured prompt after every compiler path.",
    "External LLM generation must be opt-in, tenant-authorized, provider/model allowlisted, timeout-bounded, output-capped, and data-classification/redaction controlled; local template mode remains the safe default.",
    "Never silently report provider failure as a successful semantic decode; preserve a safe local fallback only with explicit status and allowlisted audit telemetry.",
    "Persist MFA enrollments or keep production MFA fail-closed; add revocation/replay handling before expanding sensitive decode access.",
    "Upgrade torch and Starlette before serving additional dataset-derived models; keep npz artifact I/O and deny torch.load on dataset roots.",
    "Disable or protect docs/OpenAPI in production, add TrustedHost/security headers, and use a shared rate limiter with trusted proxy handling.",
    "Emit dataset family, manifest hash, subject/session policy, ontology version, provider/model status, prompt hash, anchor count, and gate outcome to SIEM; never forward raw EEG, secrets, or prompt contents.",
    "Add tests for prompt injection, provenance spoofing, oversized/nonfinite EEG, provider exceptions/timeouts, output bounds, tenant egress policy, and schema rejection before implementation is called complete.",
]
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(str(out.resolve()))
