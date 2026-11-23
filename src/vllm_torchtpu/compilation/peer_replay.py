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
"""Dynamic claim-based peer replay scheduling."""

import contextlib
import itertools
import os
import shutil
import time
from collections.abc import Callable
from typing import Any

from vllm_torchtpu import envs, profiler_trace
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_claim_call_counter = itertools.count(1)
_STALE_CLAIM_AGE_S = 86400.0


def _prune_old_claims(claims_root: str) -> None:
    """Best-effort removal of claim directories from runs older than a day."""
    cutoff = time.time() - _STALE_CLAIM_AGE_S
    with contextlib.suppress(OSError):
        for entry in os.scandir(claims_root):
            if entry.is_dir(follow_symlinks=False):
                with contextlib.suppress(OSError):
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                        shutil.rmtree(entry.path, ignore_errors=True)


def _build_claim_dirs(
    pairs: list[tuple[Any, Any]],
) -> tuple[dict[int, str], dict[int, str]] | None:
    """Create per-call, per-fingerprint claim directories for each backend.

    Returns `(claim_dirs, own_rank_dirs)` keyed by `id(backend)`, or `None` if
    any backend lacks a valid `rank_<r>_<dp>` cache directory, cannot be
    fingerprinted, or the cache directory is not writable (signaling fallback
    to rotated compilation).
    """
    call_id = str(next(_claim_call_counter))
    run_id = profiler_trace.profile_session_id(include_starttime=True)
    claim_dirs: dict[int, str] = {}
    own_rank_dirs: dict[int, str] = {}
    for backend, _ in pairs:
        bid = id(backend)
        if bid in claim_dirs:
            continue
        compiler = backend.vllm_backend.compiler_manager.compiler
        parsed = tpu_compiler.TpuCompilerAdaptor.parse_cache_dir(
            getattr(compiler, "cache_dir", None)
        )
        fingerprint = tpu_compiler._graph_fingerprint(backend.graph)
        if parsed is None or fingerprint is None:
            logger.warning_once(
                "[peer_replay] Falling back to rotated precompile: "
                "subgraph has no rank_<r>_<dp> cache_dir or fingerprint."
            )
            return None
        cache_hash_dir, rank_dir, prefix = parsed
        own_rank_dirs[bid] = rank_dir
        claims_root = os.path.join(cache_hash_dir, "peer_claims")
        if call_id == "1":
            _prune_old_claims(claims_root)
        cdir = os.path.join(
            claims_root,
            run_id,
            prefix,
            fingerprint[:24],
            call_id,
        )
        try:
            os.makedirs(cdir, exist_ok=True)
        except OSError as e:
            logger.warning_once(
                "[peer_replay] Falling back to rotated precompile: "
                "cannot create claim directory %s: %s",
                cdir,
                e,
            )
            return None
        claim_dirs[bid] = cdir
    return claim_dirs, own_rank_dirs


def _try_claim(path: str, owner_dir: str) -> bool | None:
    """Atomically claim `path` for `owner_dir` via O_EXCL.

    Returns True if claimed, False if already claimed by a peer, or None if
    writing the claim failed with OSError.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=0o644)
    except FileExistsError:
        return False
    except OSError as e:
        logger.warning_once(
            "[peer_replay] Falling back to rotated precompile: "
            "cannot write claim %s: %s",
            path,
            e,
        )
        return None
    try:
        with os.fdopen(fd, "w") as f:
            f.write(owner_dir)
    except OSError as e:
        logger.warning_once(
            "[peer_replay] Falling back to rotated precompile: "
            "cannot write claim %s: %s",
            path,
            e,
        )
        return None
    return True


def _read_claim_owner(path: str) -> str:
    """Read the claiming rank dir name from `path`, polling briefly for the writer."""
    for _ in range(50):
        try:
            with open(path) as f:
                owner = f.read().strip()
        except OSError:
            owner = ""
        if owner:
            return owner
        time.sleep(0.1)
    return ""


def compile_claimed(
    pairs: list[tuple[Any, Any]],
    compile_entry: Callable[[Any, Any], None],
) -> bool:
    """`additional_config.peer_replay_scheduler='claims'`: list scheduling over ranks.

    Each rank repeatedly claims the next unclaimed range via an atomic `O_EXCL`
    file under `<cache_hash>/peer_claims/<run_id>/<prefix>/<fp>/<call_id>/` and
    compiles it; once every range is claimed, it replays the ranges claimed by
    peers. Only ranks tracing the same graph share a claim directory and no
    collective is needed, so pipeline stages tracing different subgraphs cannot
    deadlock. Returns False if claims cannot be used (e.g. missing fingerprint
    or read-only cache), so the caller falls back to rotated compilation.
    """
    if not pairs:
        return True
    built = _build_claim_dirs(pairs)
    if built is None:
        return False
    claim_dirs, own_rank_dirs = built

    def claim_path(backend: Any, entry: Any) -> str:
        r = entry.compile_range
        sg = getattr(backend, "piecewise_compile_index", 0)
        return os.path.join(claim_dirs[id(backend)], f"{r.start}_{r.end}_sg{sg}")

    pending = [(b, e) for b, e in pairs if not e.compiled]
    if envs.TPU_PRECOMPILE_BIGGEST_FIRST:
        pending.sort(key=lambda p: p[1].compile_range.end, reverse=True)

    claimed_by_peer: list[tuple[Any, Any]] = []
    own = 0
    for backend, entry in pending:
        if entry.compiled:
            continue
        owner_dir = own_rank_dirs[id(backend)]
        claimed = _try_claim(claim_path(backend, entry), owner_dir)
        if claimed is None:
            return False
        if claimed:
            own += 1
            compile_entry(backend, entry)
        else:
            claimed_by_peer.append((backend, entry))

    wait_s = envs.TPU_PRECOMPILE_PEER_WAIT_S
    wait_order = (
        sorted(claimed_by_peer, key=lambda p: p[1].compile_range.end)
        if envs.TPU_PRECOMPILE_BIGGEST_FIRST
        else claimed_by_peer
    )
    for backend, entry in wait_order:
        if entry.compiled:
            continue
        owner = _read_claim_owner(claim_path(backend, entry))
        allowed = {owner} if owner else None
        wait = 0.0 if not owner or owner == own_rank_dirs[id(backend)] else wait_s
        with tpu_compiler.wait_for_peer_artifact(wait, allowed_dirs=allowed):
            compile_entry(backend, entry)

    logger.info(
        "[peer_replay] Claims: compiled %d, replayed or rebuilt %d (%s)",
        own,
        len(claimed_by_peer),
        next(iter(claim_dirs.values())),
    )
    return True
