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
import time
from collections.abc import Callable
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_claim_call_counter = itertools.count(1)


def _run_id() -> str:
    """Return a per-engine-run identifier shared by all workers of one EngineCore.

    Combines the parent PID with its kernel start time from `/proc/<ppid>/stat`
    so an EngineCore restart that reuses a PID never collides with stale claims.
    """
    ppid = os.getppid()
    starttime = ""
    with contextlib.suppress(OSError, IndexError), open(f"/proc/{ppid}/stat") as f:
        # Field 22 (index 19 after the comm field's closing ')') is starttime.
        starttime = f.read().rsplit(")", 1)[1].split()[19]
    return f"{ppid}_{starttime}" if starttime else str(ppid)


def _build_claim_dirs(
    pairs: list[tuple[Any, Any]],
) -> tuple[dict[int, str], set[str]] | None:
    """Create per-call, per-fingerprint claim directories for each backend.

    Returns `(claim_dirs, own_rank_dirs)` or `None` if any backend lacks a
    cache directory or cannot be fingerprinted (signaling fallback to rotated
    compilation).
    """
    call_id = str(next(_claim_call_counter))
    run_id = _run_id()
    claim_dirs: dict[int, str] = {}
    own_rank_dirs: set[str] = set()
    for backend, _ in pairs:
        bid = id(backend)
        if bid in claim_dirs:
            continue
        cache_dir = backend.vllm_backend.compiler_manager.compiler.cache_dir
        fingerprint = tpu_compiler._graph_fingerprint(backend.graph)
        if not cache_dir or fingerprint is None:
            logger.warning_once(
                "[peer_replay] Falling back to rotated precompile: "
                "subgraph has no cache_dir or fingerprint."
            )
            return None
        rank_dir_path, prefix = os.path.split(os.path.normpath(cache_dir))
        own_rank_dirs.add(os.path.basename(rank_dir_path))
        cdir = os.path.join(
            os.path.dirname(rank_dir_path),
            "peer_claims",
            run_id,
            prefix,
            fingerprint[:24],
            call_id,
        )
        os.makedirs(cdir, exist_ok=True)
        claim_dirs[bid] = cdir
    return claim_dirs, own_rank_dirs


def _try_claim(path: str, owner_dir: str) -> bool:
    """Atomically claim `path` for `owner_dir` via O_EXCL. False if already claimed."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(owner_dir)
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
    """VLLM_TPU_PEER_REPLAY_CLAIMS=1: list scheduling over the host's ranks.

    Each rank repeatedly claims the largest unclaimed range via an atomic
    O_EXCL file under `<cache_hash>/peer_claims/<run_id>/<prefix>/<fp>/<call_id>/`
    and compiles it; once every range is claimed, it replays the ranges claimed
    by peers. Only ranks tracing the same graph share a claim directory and no
    collective is needed, so pipeline stages tracing different subgraphs cannot
    deadlock. Returns False if any subgraph cannot be claimed.
    """
    if not pairs:
        return True
    built = _build_claim_dirs(pairs)
    if built is None:
        return False
    claim_dirs, own_rank_dirs = built

    def claim_path(backend: Any, entry: Any) -> str:
        r = entry.compile_range
        return os.path.join(claim_dirs[id(backend)], f"{r.start}_{r.end}")

    def own_dir_for(backend: Any) -> str:
        cache_dir = backend.vllm_backend.compiler_manager.compiler.cache_dir
        return os.path.basename(os.path.dirname(os.path.normpath(cache_dir)))

    pending = sorted(
        ((b, e) for b, e in pairs if not e.compiled),
        key=lambda p: p[1].compile_range.end,
        reverse=True,
    )
    claimed_by_peer: list[tuple[Any, Any]] = []
    own = 0
    for backend, entry in pending:
        if entry.compiled:
            continue
        owner_dir = own_dir_for(backend)
        if _try_claim(claim_path(backend, entry), owner_dir):
            own += 1
            with tpu_compiler.replay_only_from({owner_dir}):
                compile_entry(backend, entry)
        else:
            claimed_by_peer.append((backend, entry))

    wait_s = envs.TPU_PRECOMPILE_PEER_WAIT_S
    for backend, entry in sorted(claimed_by_peer, key=lambda p: p[1].compile_range.end):
        if entry.compiled:
            continue
        owner = _read_claim_owner(claim_path(backend, entry))
        if owner in own_rank_dirs:
            with tpu_compiler.replay_only_from({owner}):
                compile_entry(backend, entry)
            continue
        allowed = {owner} if owner else None
        with (
            tpu_compiler.replay_only_from(allowed),
            tpu_compiler.wait_for_peer_artifact(wait_s if owner else 0.0),
        ):
            compile_entry(backend, entry)

    logger.info(
        "[peer_replay] Claims: compiled %d, replayed or rebuilt %d (%s)",
        own,
        len(claimed_by_peer),
        next(iter(claim_dirs.values())),
    )
    return True
