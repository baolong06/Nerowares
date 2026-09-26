"""Root-scoped deterministic discovery for one family package."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.data.release.models import FamilyReleaseConfig

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:")
_CACHE_DIR_NAMES = {
    "cache",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}
_DOWNLOADER_SUFFIXES = (".ps1", ".sh")


def _posix_parts(path: str) -> list[str]:
    return path.replace("\\", "/").split("/")


def _is_absolute_path(path: str) -> bool:
    if path.startswith(("/", "\\")):
        return True
    if _DRIVE_PATH_RE.match(path):
        return True
    return Path(path).is_absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _directory_reason(name: str) -> str | None:
    lowered = name.casefold()
    if lowered == ".git":
        return "git"
    if lowered in _CACHE_DIR_NAMES:
        return "cache"
    if lowered == "reports":
        return "report"
    return None


def _is_downloader_script(name: str) -> bool:
    lowered = name.casefold()
    return lowered.startswith("download") and lowered.endswith(_DOWNLOADER_SUFFIXES)


def _is_report_file(name: str) -> bool:
    stem, dot, suffix = name.rpartition(".")
    candidate = stem if dot else name
    return candidate.casefold() == "report"


def _scope_digest(relative_paths: list[str]) -> str:
    ordered = sorted(relative_paths, key=lambda item: item.encode("utf-8"))
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExcludedPath:
    """One path rejected from family scope, with a stable reason token."""

    path: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.replace("\\", "/"))


@dataclass(frozen=True)
class ScopeResult:
    """Deterministic included files, exclusions, and scope digest."""

    files: tuple[Path, ...]
    excluded: tuple[ExcludedPath, ...]
    scope_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "excluded", tuple(self.excluded))
        digest = str(self.scope_sha256).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("scope_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "scope_sha256", digest)


def discover_scope(config: FamilyReleaseConfig, data_root: Path) -> ScopeResult:
    """List regular files under the configured family root without reading contents."""
    local_root = config.local_root
    if not local_root or _is_absolute_path(local_root) or any(part == ".." for part in _posix_parts(local_root)):
        raise ValueError("local_root must be relative to THINKING_DATA_ROOT")

    resolved_data = Path(data_root).resolve()
    family = resolved_data.joinpath(*_posix_parts(local_root))
    if family.is_symlink():
        raise ValueError("family root must not be a symlink")
    if not family.exists():
        raise ValueError("family root does not exist")
    if not family.is_dir():
        raise ValueError("family root is not a directory")

    resolved_family = family.resolve()
    if not _is_relative_to(resolved_family, resolved_data):
        raise ValueError("family root escapes data root")

    included: list[tuple[str, Path]] = []
    excluded: list[ExcludedPath] = []
    seen_relative: set[str] = set()
    seen_resolved: set[Path] = set()

    def _exclude(relative: str, reason: str) -> None:
        excluded.append(ExcludedPath(path=relative, reason=reason))

    def _walk(directory: Path, relative_prefix: str) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            _exclude(relative_prefix or ".", "non-regular")
            return

        for entry in entries:
            relative = f"{relative_prefix}/{entry.name}" if relative_prefix else entry.name
            posix = relative.replace("\\", "/")
            if any(part == ".." for part in _posix_parts(posix)):
                _exclude(posix, "parent-traversal")
                continue
            try:
                is_link = entry.is_symlink()
            except OSError:
                _exclude(posix, "symlink")
                continue
            if is_link:
                _exclude(posix, "symlink")
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                _exclude(posix, "non-regular")
                continue
            if is_dir:
                dir_reason = _directory_reason(entry.name)
                if dir_reason is not None:
                    _exclude(posix, dir_reason)
                    continue
                resolved_dir = Path(entry.path).resolve()
                if not _is_relative_to(resolved_dir, resolved_family):
                    _exclude(posix, "outside-root")
                    continue
                _walk(Path(entry.path), posix)
                continue
            if not is_file:
                _exclude(posix, "non-regular")
                continue
            if _is_downloader_script(entry.name):
                _exclude(posix, "downloader-script")
                continue
            if _is_report_file(entry.name):
                _exclude(posix, "report")
                continue
            resolved_file = Path(entry.path).resolve()
            if not _is_relative_to(resolved_file, resolved_family):
                _exclude(posix, "outside-root")
                continue
            if posix in seen_relative or resolved_file in seen_resolved:
                _exclude(posix, "duplicate")
                continue
            seen_relative.add(posix)
            seen_resolved.add(resolved_file)
            included.append((posix, resolved_file))

    _walk(family, "")
    included.sort(key=lambda item: item[0].encode("utf-8"))
    excluded.sort(key=lambda item: (item.path.encode("utf-8"), item.reason.encode("utf-8")))
    relatives = [relative for relative, _ in included]
    return ScopeResult(
        files=tuple(path for _, path in included),
        excluded=tuple(excluded),
        scope_sha256=_scope_digest(relatives),
    )
