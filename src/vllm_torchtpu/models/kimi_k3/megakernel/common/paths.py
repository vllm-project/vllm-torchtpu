# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-stdlib local filesystem path validation and atomic staging helpers."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Matches remote URI schemes even after pathlib collapses "://" to ":/"
# (requires >= 2 scheme characters so single-letter drive prefixes do not match).
_COLLAPSED_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:/")


def require_local_path(value: Path | str, *, label: str = "path") -> Path:
    """Validate that ``value`` is a local filesystem path and return it as ``Path``.

    Rejects both explicit ``scheme://...`` URIs and collapsed ``scheme:/...``
    paths (produced when a remote URI was previously wrapped in ``Path``).

    Args:
        value: Candidate local path or string.
        label: Human-readable parameter name for error messages.

    Returns:
        Validated local :class:`pathlib.Path`.

    Raises:
        ValueError: If ``value`` looks like a remote URI rather than a local path.
    """
    raw = os.fspath(value)
    if "://" in raw or _COLLAPSED_SCHEME_RE.match(raw) is not None:
        msg = f"{label} must be a local filesystem path, not remote URI {raw!r}"
        raise ValueError(msg)
    return Path(raw)


@contextlib.contextmanager
def atomic_binary_writer(path: Path | str) -> Iterator[IO[bytes]]:
    """Yield a sibling temporary binary file and replace ``path`` on clean exit.

    Creates missing parent directories before staging. Unlinks the temporary
    file on any exception so partial files are never left behind.

    Args:
        path: Destination local file path.

    Yields:
        Open writable binary file object backed by a sibling temp file.
    """
    target = require_local_path(path, label="path")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.tmp.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            yield handle.file
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(target)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a sibling temporary file.

    Args:
        path: Destination local file path.
        data: Raw byte payload to write.
    """
    with atomic_binary_writer(path) as handle:
        handle.write(data)


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write ``text`` to ``path`` atomically via a sibling temporary file.

    Args:
        path: Destination local file path.
        text: Text payload to encode and write.
        encoding: Character encoding used to encode ``text``.
    """
    atomic_write_bytes(path, text.encode(encoding))


def _publish_staged_directory(staging: Path, dest: Path, *, exist_ok: bool) -> None:
    """Publish ``staging`` to ``dest`` atomically, rolling back on swap error."""
    if not dest.exists():
        staging.rename(dest)
        return
    if not exist_ok:
        msg = f"{dest} already exists"
        raise FileExistsError(msg)
    backup_root = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.bak."))
    backup_dest = backup_root / dest.name
    moved_existing = False
    try:
        dest.rename(backup_dest)
        moved_existing = True
        staging.rename(dest)
    except BaseException:
        if moved_existing and backup_dest.exists() and not dest.exists():
            backup_dest.rename(dest)
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    shutil.rmtree(backup_root, ignore_errors=True)


@contextlib.contextmanager
def staged_directory(
    target: Path | str,
    *,
    exist_ok: bool = False,
) -> Iterator[Path]:
    """Stage directory outputs in a sibling temp dir and publish atomically.

    Ensures ``target.parent`` exists before creating the staging directory.
    On any exception inside the ``with`` block or during publication, removes
    the staging directory and restores any pre-existing ``target`` intact.

    Args:
        target: Destination local directory path.
        exist_ok: When ``True``, replace an existing ``target`` directory via
            a sibling backup swap with rollback; when ``False``, raise
            :class:`FileExistsError` if ``target`` already exists.

    Yields:
        Path to the newly created sibling staging directory.

    Raises:
        FileExistsError: If ``target`` already exists and ``exist_ok`` is ``False``.
    """
    dest = require_local_path(target, label="target")
    if not exist_ok and dest.exists():
        msg = f"{dest} already exists"
        raise FileExistsError(msg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}.tmp."))
    try:
        yield staging
        _publish_staged_directory(staging, dest, exist_ok=exist_ok)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
