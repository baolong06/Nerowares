"""Injected Hugging Face Dataset Hub adapter. Tests never construct a live API."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.data.release.models import CANONICAL_BCICIV2A_REPO_ID

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REVISIONS = {"main", "master", "latest", "head", "default"}
_HUB_REQUIRED = "Hugging Face support requires the optional 'data' dependency"


def _posix(path: str) -> str:
    return str(path).replace("\\", "/")


def reject_canonical_repo(repo_id: str) -> str:
    text = str(repo_id).strip()
    if text.casefold() == CANONICAL_BCICIV2A_REPO_ID.casefold():
        raise ValueError("canonical BCICIV-2a repository must not be overwritten")
    return text


def require_immutable_revision(revision: str, *, git_sha: bool = True) -> str:
    text = str(revision or "")
    if not text or text.lower() in _MUTABLE_REVISIONS:
        raise ValueError("revision must be immutable, not a moving revision")
    if git_sha and not _GIT_SHA_RE.fullmatch(text.lower()):
        raise ValueError("revision must be a 40-character commit SHA")
    return text


def is_git_sha(revision: str) -> bool:
    return bool(_GIT_SHA_RE.fullmatch(str(revision or "").lower()))


@dataclass(frozen=True)
class UploadOperation:
    """One relative Hub path streamed from an existing local package file."""

    path: str
    source_path: Path = field(repr=False)


@dataclass(frozen=True)
class RemoteListedFile:
    path: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class RemoteListing:
    revision: str
    files: tuple[object, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True)
class _PathCommitAdd:
    """Stand-in for CommitOperationAdd when the optional data extra is absent."""

    path_in_repo: str
    path_or_fileobj: object = field(repr=False)


class HubClient:
    """Narrow Dataset-Hub adapter around an injected huggingface_hub-shaped API."""

    def __init__(self, api: object | None = None) -> None:
        self._api = api

    def _require_api(self) -> object:
        if self._api is not None:
            return self._api
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(_HUB_REQUIRED) from exc
        self._api = HfApi()
        return self._api

    def _commit_operations(self, operations: Sequence[UploadOperation]) -> list[object]:
        try:
            from huggingface_hub import CommitOperationAdd as add_cls
        except ImportError as exc:
            if self._api is None:
                raise RuntimeError(_HUB_REQUIRED) from exc
            add_cls = _PathCommitAdd
        return [
            add_cls(
                path_in_repo=operation.path,
                path_or_fileobj=operation.source_path,
            )
            for operation in operations
        ]

    def create_dataset_repo(self, repo_id: str, private: bool) -> None:
        repo_id = reject_canonical_repo(repo_id)
        api = self._require_api()
        try:
            api.create_repo(
                repo_id,
                private=bool(private),
                repo_type="dataset",
                exist_ok=True,
            )
        except Exception:
            raise RuntimeError("hub request failed") from None

    def upload_batch(
        self, repo_id: str, operations: Sequence[UploadOperation], commit_message: str
    ) -> str:
        repo_id = reject_canonical_repo(repo_id)
        adds = self._commit_operations(operations)
        api = self._require_api()
        try:
            info = api.create_commit(
                repo_id,
                operations=adds,
                commit_message=commit_message,
                repo_type="dataset",
            )
        except Exception:
            raise RuntimeError("hub request failed") from None
        oid = getattr(info, "oid", None)
        if not isinstance(oid, str) or not is_git_sha(oid):
            raise ValueError("revision must be a 40-character commit SHA")
        return oid

    def list_revision(self, repo_id: str, revision: str) -> RemoteListing:
        repo_id = reject_canonical_repo(repo_id)
        revision = require_immutable_revision(revision, git_sha=True)
        api = self._require_api()
        try:
            entries = list(
                api.list_repo_tree(
                    repo_id,
                    recursive=True,
                    expand=True,
                    revision=revision,
                    repo_type="dataset",
                )
            )
        except Exception:
            raise RuntimeError("hub request failed") from None
        files: list[RemoteListedFile] = []
        for entry in entries:
            path = getattr(entry, "path", None)
            if not path:
                continue
            if type(entry).__name__ == "RepoFolder":
                continue
            size = getattr(entry, "size", None)
            if size is None:
                size = getattr(entry, "size_bytes", None)
            files.append(
                RemoteListedFile(
                    path=_posix(path),
                    size_bytes=int(size) if isinstance(size, int) else None,
                )
            )
        return RemoteListing(revision=revision, files=tuple(files))

    def download_for_verify(
        self, repo_id: str, path: str, revision: str, destination: Path
    ) -> Path:
        repo_id = reject_canonical_repo(repo_id)
        revision = require_immutable_revision(revision, git_sha=True)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        api = self._require_api()
        try:
            downloaded = api.hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
                revision=revision,
            )
        except Exception:
            raise RuntimeError("hub request failed") from None
        dest.write_bytes(Path(downloaded).read_bytes())
        return dest
