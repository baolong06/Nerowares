"""Safe I/O for EEG ingestion — APP-004 remediation.
Only paths below configured dataset roots and trusted non-pickle formats are accepted.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote

ALLOWED_EXTENSIONS = {".bdf", ".edf", ".fif", ".parquet", ".npy", ".npz", ".csv"}
ALLOWED_ROOT_ENV = "THINKING_DATA_ROOT"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_DATA_ROOT = Path("E:/AI_thucchien/THINKING/datasets")


def _default_allowed_roots() -> list[Path]:
    """Environment first, then the worktree, then a present legacy path."""
    roots: list[Path] = [WORKTREE_ROOT / "datasets", Path("/data/eeg")]
    if _LEGACY_DATA_ROOT.exists():
        roots.insert(1, _LEGACY_DATA_ROOT)
    return roots


DEFAULT_ALLOWED_ROOTS = _default_allowed_roots()

# Reject literal, percent-encoded, and URL double-encoded traversal markers before Path parsing.
_TRAVERSAL_RE = re.compile(r"(?:\.\.(?:[\\/]|$)|%(?:25)?2e%(?:25)?2e(?:%(?:25)?2f|%(?:25)?5c|[\\/]|$))", re.IGNORECASE)


def _allowed_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get(ALLOWED_ROOT_ENV)
    if env:
        roots.append(Path(env).expanduser().resolve())
    roots.extend(root.resolve() for root in DEFAULT_ALLOWED_ROOTS)
    return list(dict.fromkeys(roots))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _decode_until_stable(raw: str) -> str:
    prev = raw.replace("\\", "/")
    for _ in range(5):
        nxt = unquote(prev)
        if nxt == prev:
            return prev
        prev = nxt
    return prev


def _has_parent_segment(raw: str) -> bool:
    text = raw.replace("\\", "/")
    return any(part == ".." for part in Path(text).parts) or "/../" in f"/{text}/" or text.startswith("../")


def safe_resolve(user_path: str | Path, allowed_roots: list[Path] | None = None) -> Path:
    """Resolve a user path only beneath configured roots; never reveal absolute roots."""
    raw = str(user_path)
    decoded = _decode_until_stable(raw)
    if (
        not raw
        or "\x00" in raw
        or _TRAVERSAL_RE.search(raw)
        or _TRAVERSAL_RE.search(decoded)
        or _has_parent_segment(raw)
        or _has_parent_segment(decoded)
    ):
        raise ValueError("Path traversal blocked")

    roots = [Path(root).resolve() for root in (allowed_roots or _allowed_roots())]
    if not roots:
        raise ValueError("No dataset root configured")

    candidate = Path(raw)
    # Relative paths are always interpreted under the first (configured) root.
    resolved = candidate.resolve() if candidate.is_absolute() else (roots[0] / candidate).resolve()
    if any(_is_relative_to(resolved, root) for root in roots):
        return resolved
    raise ValueError("Path traversal blocked")


def validate_extension(path: Path) -> None:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    # Compound/hidden suffixes must not hide an unallowed format (e.g. .npy.exe).
    if len(suffixes) != 1 or suffixes[0] not in ALLOWED_EXTENSIONS:
        raise ValueError("File extension is not allowed")


def load_parquet_safe(path: str | Path):
    """Load parquet through pyarrow. Never deserialize Python pickles."""
    import pyarrow.parquet as pq

    p = safe_resolve(path)
    validate_extension(p)
    if p.suffix.lower() != ".parquet":
        raise ValueError("load_parquet_safe expects a parquet file")
    return pq.read_table(str(p))


def load_npy_safe(path: str | Path):
    """Load .npy/.npz with pickle permanently disabled."""
    import numpy as np

    p = safe_resolve(path)
    validate_extension(p)
    if p.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError("load_npy_safe expects an .npy or .npz file")
    return np.load(str(p), allow_pickle=False)
