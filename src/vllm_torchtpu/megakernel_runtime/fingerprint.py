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
"""Plugin source hashes, fingerprints and agreement tokens (stdlib only).

`source_hash` digests the bytes of a plugin's `*.py` files (never mtimes, so
editable installs work) and fails closed for sources it can't see: namespace
packages without `source_root`, sourceless (`.pyc`-only) and zip installs.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import zipimport
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import MegakernelPluginError

MAX_SOURCE_FILES = 2000
MAX_SOURCE_BYTES = 64 << 20
_FIX_HINT = "install the plugin from source or set TpuMonoSpec.source_root"


def _unhashable(why: str, name: str) -> MegakernelPluginError:
    return MegakernelPluginError(
        f"plugin_source_unhashable:{why}: can't hash the source of {name!r}; "
        f"{_FIX_HINT}"
    )


def _walk_py(root: Path, name: str) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        ]
        names = set(filenames)
        for fn in filenames:
            # A `.pyc` beside its `.py` (e.g. `compileall -b`) is fine: only the
            # source is hashed. A `.pyc` without one is a sourceless install.
            if fn.endswith(".pyc") and fn[: -len("c")] not in names:
                raise _unhashable("sourceless", name)
            if fn.endswith(".py") and not fn.startswith("."):
                p = Path(dirpath) / fn
                files.append((p.relative_to(root).as_posix(), p))
    return files


def hash_files(files: Iterable[tuple[str, Path]], *, name: str = "") -> str:
    """sha256 over `(relpath, path)` records sorted by relpath, each fed as
    `relpath NUL len NUL bytes`. Bounded by file count and total bytes."""
    recs = sorted(files)
    if len(recs) > MAX_SOURCE_FILES:
        raise MegakernelPluginError(
            f"plugin_source_too_large: {name!r} has {len(recs)} files "
            f"(> {MAX_SOURCE_FILES})"
        )
    h, total = hashlib.sha256(), 0
    for rel, path in recs:
        data = path.read_bytes()
        total += len(data)
        if total > MAX_SOURCE_BYTES:
            raise MegakernelPluginError(
                f"plugin_source_too_large: {name!r} exceeds {MAX_SOURCE_BYTES} bytes"
            )
        h.update(rel.encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0")
        h.update(data)
    return h.hexdigest()


@functools.cache
def _hash_root(root: str, single_file: bool, name: str) -> str:
    p = Path(root)
    if single_file:
        return hash_files([(p.name, p)], name=name)
    return hash_files(_walk_py(p, name), name=name)


def source_hash(module: ModuleType, *, source_root: str | None = None) -> str:
    """The digest of a plugin module's source (computed once per process).

    A package hashes every `*.py` under its directory (skipping hidden dirs and
    `__pycache__`, not following directory symlinks); a single-file module
    hashes that one file. Raises `MegakernelPluginError(
    "plugin_source_unhashable:<why>")` for namespace packages (unless
    `source_root` is set), sourceless and zip installs.
    """
    name = module.__name__
    if source_root is not None:
        root = Path(source_root)
        if root.is_file() and root.suffix == ".py":
            return _hash_root(str(root.resolve()), True, name)
        if not root.is_dir():
            raise _unhashable("source_root_missing", name)
        return _hash_root(str(root.resolve()), False, name)
    spec = getattr(module, "__spec__", None)
    if isinstance(getattr(spec, "loader", None), zipimport.zipimporter):
        raise _unhashable("zip", name)
    file = getattr(module, "__file__", None)
    path = getattr(module, "__path__", None)
    if file is None or (path is not None and len(list(path)) != 1):
        raise _unhashable("namespace", name)
    fpath = Path(file)
    if not fpath.is_file():
        raise _unhashable("zip", name)
    if fpath.suffix != ".py":
        raise _unhashable("sourceless", name)
    if path is not None:
        return _hash_root(str(fpath.parent.resolve()), False, name)
    return _hash_root(str(fpath.resolve()), True, name)


@functools.lru_cache(maxsize=1)
def runtime_src_hash() -> str:
    """The digest of the `megakernel/` and `megakernel_runtime/` sources."""
    pkg = Path(__file__).resolve().parent.parent
    files: list[tuple[str, Path]] = []
    for sub in ("megakernel", "megakernel_runtime"):
        files += [(f"{sub}/{rel}", p) for rel, p in _walk_py(pkg / sub, sub)]
    return hash_files(files, name="vllm_torchtpu.megakernel*")


def clear_caches() -> None:
    """Forget cached source hashes (tests that edit a plugin's source)."""
    _hash_root.cache_clear()
    runtime_src_hash.cache_clear()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no spaces); unknown types via `repr`."""

    def norm(x: Any) -> Any:
        if isinstance(x, Mapping):
            return {str(k): norm(v) for k, v in x.items()}
        if isinstance(x, (frozenset, set)):
            return sorted(norm(v) for v in x)
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x

    return json.dumps(norm(obj), sort_keys=True, separators=(",", ":"), default=repr)


def plugin_fingerprint(
    *,
    module: str,
    attr: str,
    plugin: Any,
    source_hash: str,
    config: Mapping[str, Any],
    raw_env: str,
) -> str:
    """sha256 over the plugin identity, its source and the resolved config."""
    doc = {
        "api_version": int(plugin.api_version),
        "module": module,
        "attr": attr,
        "name": plugin.name,
        "version": str(plugin.version),
        "source_hash": source_hash,
        "chips": list(plugin.chips),
        "tp_sizes": list(plugin.tp_sizes),
        "parallelism": sorted(plugin.parallelism),
        "features": sorted(plugin.features),
        "requires": dict(sorted(plugin.requires.items())),
        "numerics": plugin.numerics,
        "collective_domain": plugin.collective_domain,
        "decline_codes": list(plugin.decline_codes),
        "config": dict(config),
        "raw_env": raw_env,
    }
    return hashlib.sha256(canonical_json(doc).encode("utf-8")).hexdigest()


def agreement_token(fingerprint: str, runtime_src: str) -> int:
    """Agreement #1 token: 60 bits, so `-token` fits the int64 vote tensor."""
    digest = hashlib.sha256((fingerprint + runtime_src).encode("ascii")).hexdigest()
    return int(digest[:15], 16)


def src8(runtime_src: str, plugin_src: str) -> str:
    """The source tag secondary / draft program model tags append (`_<src8>`)."""
    return hashlib.sha256((runtime_src + plugin_src).encode("ascii")).hexdigest()[:8]


__all__ = [
    "MAX_SOURCE_BYTES",
    "MAX_SOURCE_FILES",
    "agreement_token",
    "canonical_json",
    "clear_caches",
    "hash_files",
    "plugin_fingerprint",
    "runtime_src_hash",
    "source_hash",
    "src8",
]
