import copy, json
from pathlib import Path

root = Path(".")
source = root / "reports" / "Pre-Fix-AppSec-Assessment-Brennan-2026-09-21.json"
out = root / "reports" / "Pre-Fix-AppSec-Assessment-Phase3-2026-09-22.json"
report = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
report["meta"].update({
    "title": "Pre-Fix Security Assessment Report \u2014 Phase 3 Dataset Expansion",
    "subtitle": "THINKING EEG Platform \u2014 planned D01/D09, Chisco, SparrKULee loaders",
    "date": "2026-09-22",
})
report["hard_stop"] = (
    "This report is the Pre-Fix gate for Phase 3 dataset expansion. No D01/D09, Chisco, "
    "or SparrKULee loader, split, dependency, allowlist, or API source was modified. "
    "Implementation waits for explicit approval on the next user turn."
)
report["executive_summary"] = (
    "Phase 2 Brennan retrieval artifacts are already validated, but the next phase would "
    "add multiple EEG dataset families (D01/D09, Chisco, and SparrKULee) with potentially "
    "different containers, metadata, sampling rates, channel conventions, labels, and "
    "session structures. The current code has safe path resolution, non-pickle readers, "
    "anti-leakage evaluation primitives, Vault/SIEM hooks, and a locked Brennan contract. "
    "The expansion is blocked until each dataset has a provenance manifest, format-specific "
    "safe loader contract, native-rate preprocessing policy, subject/session identity map, "
    "and explicit split strategy. Highest residual risks remain the CISA KEV torch checkpoint "
    "issue, Starlette advisories, process-local MFA, unbounded EEG payloads, generic file "
    "allowlist expansion, and missing local CVE library. Mean domain score 5.59 / 10.0."
)
report["scope"] = [
    "Planned Phase 3 dataset inventory and loaders for D01/D09, Chisco, and SparrKULee",
    "Dataset provenance, licensing, checksums, container formats, sidecars, and event alignment",
    "Subject/session identity and anti-leakage split contracts for each dataset",
    "Native-rate filtering, Autoreject/ICA applicability, channel policy, and resampling",
    "Existing src/training, src/preprocessing, src/api, Vault, SIEM, and dependency controls",
    "CVE/KEV, Gitleaks, SBOM, and Sigma/KQL/SPL control readiness",
]
report["out_of_scope"] = [
    "Writing or modifying any Phase 3 loader",
    "Downloading or probing external dataset hosts",
    "Changing ALLOWED_EXTENSIONS or API routes",
    "Training or publishing a Phase 3 model",
    "Dependency upgrades or production deployment changes",
]
report["evidence"] = [
    "Phase 2 canonical MLP artifact passes Brennan anti-leakage gate; provenance is recorded in artifacts/brennan_phase2_manifest.json",
    "Phase 2 reproducibility encoder SHA-256 matches canonical artifact: b2cb578d77f61e65ac36c2e4ccb0099dc30fa6f5e9b595d87e58552d00831461",
    "PYTHONPATH=. pytest -q: 49 passed, 45 warnings (2026-09-22)",
    "Brennan protocol explicitly bans random epoch splits, fabricated sessions, and overlapping buffered train windows",
    "Current safe_resolve/validate_extension rejects unapproved BrainVision suffixes globally; Brennan uses a root-scoped triplet helper",
    "Pinned ML extras still include torch==2.5.1, pyarrow==15.0.2, scikit-learn==1.4.2, and starlette via FastAPI; prior pip-audit findings remain open",
    "Phase 3 target dataset inventory and format contracts have not yet been locked",
]
report["domains"] = [
    {"id": 1, "name": "Authentication", "score": 6.0, "rationale": "JWT and MFA hooks exist for API access, but process-local enrollment and default-off MFA remain; new dataset endpoints would increase protected surface."},
    {"id": 2, "name": "Authorization / Access Control", "score": 7.0, "rationale": "Ownership and tenant binding exist, but dataset-family/manifest-level authorization and RBAC are not yet defined for additional loaders."},
    {"id": 3, "name": "Session / Token Management", "score": 6.5, "rationale": "Short-lived HS256 tokens validate core claims, but there is no revocation store and no dataset-operation audit binding."},
    {"id": 4, "name": "Input Validation & Injection", "score": 5.5, "rationale": "Pydantic and vocabulary controls exist, but heterogeneous sidecars, events, headers, and metadata require per-format schemas and size limits."},
    {"id": 5, "name": "File Handling / Path Traversal", "score": 5.5, "rationale": "safe_resolve and non-pickle readers are present, but adding several container/sidecar families creates residual traversal, parser, symlink, and resource-exhaustion risk."},
    {"id": 6, "name": "Cryptography", "score": 6.5, "rationale": "HMAC-SHA256 JWT, SHA-256 artifacts, and hashed backup codes exist; artifact signatures and dataset manifest signing are not yet implemented."},
    {"id": 7, "name": "Error Handling & Information Disclosure", "score": 5.5, "rationale": "Generic API errors exist, but format-specific parser errors, subject identifiers, missing-sidecar behavior, and dataset existence oracles need a contract."},
    {"id": 8, "name": "Logging, Monitoring & Telemetry", "score": 6.5, "rationale": "Correlation and SIEM allowlisting exist, but no standardized telemetry fields yet cover dataset family, manifest hash, parser, or split provenance."},
    {"id": 9, "name": "Transport, CORS & Security Headers", "score": 4.5, "rationale": "Existing API lacks several production headers and strict host/HTTPS controls; broader ingestion would increase exposure without remediation."},
    {"id": 10, "name": "Configuration, Secrets & Supply Chain", "score": 3.5, "rationale": "Vault fallback and SBOM hooks exist, but torch/Starlette/pyarrow findings, missing local CVE library, and unverified third-party dataset provenance remain."},
    {"id": 11, "name": "Availability, Rate Limiting & Business Logic", "score": 4.5, "rationale": "In-memory rate limiting and unbounded EEG/body/parser work remain; multi-dataset parsing could amplify CPU, memory, and disk pressure."},
]
report["dfd"] = {
    "entities": ["Client", "Operator", "Dataset custodians", "D01/D09 sources", "Chisco source", "SparrKULee source", "HashiCorp Vault", "SIEM"],
    "processes": ["FastAPI (CORS, rate limit, correlation)", "JWT/MFA", "/decode", "/ingest/validate", "Dataset inventory/manifest builder (planned)", "Format-specific safe loaders (planned)", "Native-rate preprocessing", "LOSO/cross-session/stimulus split evaluator", "AnchorEncoder/MLP"],
    "stores": ["Process-local MFA dict", "In-memory rate buckets", "Client-held JWT", "encoder.npz", "Dataset manifest/checksum store (planned)", "Vault KV", "E:/AI_thucchien/THINKING/datasets", "SIEM audit stream"],
    "trust_boundaries": ["Internet to API", "API to filesystem", "Operator to dataset inventory", "External dataset source to local mirror", "API to Vault", "API to SIEM", "Training process to each dataset root", "Subject identity to split evaluator"],
}
report["stride"] = [
    {"element": "Dataset manifest and provenance", "S": "Untrusted source or operator account substitutes a dataset", "T": "Manifest/checksum or license metadata tampering", "R": "Missing immutable provenance trail", "I": "Subject identifiers and EEG at rest", "D": "Large mirror or manifest expansion", "E": "Loader executes unsafe parser behavior"},
    {"element": "Format-specific loader", "S": "Planted sidecar/container under dataset root", "T": "Header/event/channel remapping", "R": "Ambiguous parser fallback", "I": "Raw EEG and participant metadata", "D": "Huge file or decompression exhaustion", "E": "Unsafe deserialization or parser escape"},
    {"element": "Subject/session split evaluator", "S": "Forged subject/session identity", "T": "Leakage through random or overlapping windows", "R": "Unreproducible fold assignment", "I": "Cross-subject data exposure", "D": "Excessive fold computation", "E": "False scientific claim from invalid split"},
    {"element": "API and artifact service", "S": "Stolen JWT/MFA bypass", "T": "Anchor or artifact override", "R": "Weak audit correlation", "I": "EEG in request or logs", "D": "Unbounded decode/parser workload", "E": "Serving unvalidated model artifact"},
]
report["planned_change_constraints"] = [
    "Do not add generic extensions or parser fallbacks; each dataset family must have a root-scoped, format-specific contract and size cap.",
    "Create a manifest before loading: source URL/custodian, license, BIDS/version, file hashes, subject/session map, event source, and known exclusions.",
    "Use safe_resolve, reject pickle/joblib/cloudpickle, validate sidecar relationships, and preserve original files as read-only inputs.",
    "Filter at native sampling rate before epoching/downsampling; record channel type policy, units, reference, Autoreject/ICA decisions, and every dropped epoch.",
    "Ban random epoch splits and fabricated sessions; choose only justified cross-session, LOSO, or stimulus-identity/block strategies per dataset.",
    "Keep each dataset result separate; do not pool datasets or claim cross-dataset generalization until identities, labels, and stimulus semantics are proven compatible.",
    "Upgrade torch and Starlette before serving additional dataset-derived models; keep npz artifact I/O and deny torch.load on dataset roots.",
    "Persist MFA or keep mfa_required fail-closed before exposing more ingestion surface; add body/parser limits and shared rate limiting.",
    "Add dataset-family, manifest hash, parser, subject/session, fold strategy, and gate outcome to allowlisted SIEM telemetry.",
]
for finding in report["findings"]:
    if finding["id"] == "TW-2026-004":
        finding["title"] = "Planned multi-dataset ingest expansion without per-format contract"
        finding["component"] = "src/preprocessing/io.py ALLOWED_EXTENSIONS; src/training/dataset.py planned Phase 3 loaders"
        finding["exploitability_here"] = "Current generic readers remain constrained, but adding D01/D09, Chisco, or SparrKULee formats without a per-format root, sidecar, size, and parser contract could introduce local authenticated file parsing and resource-exhaustion paths."
        finding["poc"] = "Current safe_resolve/validate_extension rejects unapproved extensions. The future PoC is a naive allowlist expansion followed by parsing a planted header, sidecar, archive, or container under a dataset root without matching-file, byte-cap, checksum, and parser validation."
        finding["remediation"] = "For each dataset family, require a read-only root-scoped manifest, matching sidecars, file-size and decompression caps, SHA-256 verification, explicit parser choice, no pickle, and a dedicated test fixture. Keep global ALLOWED_EXTENSIONS closed."
        break
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(str(out.resolve()))
