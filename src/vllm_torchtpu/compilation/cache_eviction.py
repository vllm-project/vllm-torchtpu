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
"""Size cap and LRU eviction for the on-disk compile caches.

Every compile cache vllm-torchtpu writes only grows: vLLM's
``torch_compile_cache/<hash>/`` gains a directory per config, env or code
change, ``profile_cache/<hash>/`` one per profile-run variant, and torch_tpu's
Tier-3 cache one binary per program. Nothing deletes old entries.

This module treats all of them the same way. Each cache is a *region*: a root
directory whose entries at a fixed depth are *units*, evicted whole (a hash
directory, or one file). A unit's last use is the newest mtime of the unit
itself; code that reuses a unit calls :func:`mark_used` so a unit read every
start is not evicted as if it were stale. Regions whose reads cannot be hooked
(torch_tpu's Tier-3 cache) fall back to write time.

When the regions together exceed ``VLLM_TPU_COMPILE_CACHE_MAX_BYTES``,
:func:`evict` deletes the least recently used units until they fit again. It
never deletes a unit used since this process started, and it counts bytes by
inode: caches share files through hard links (the artifact store links vLLM's
per-rank artifacts), and deleting one name of a linked file frees nothing, so a
unit's bytes count as freed only once every name of the file is deleted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import os
import shutil
import sys
import time
from collections.abc import Iterable, Sequence

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Units used at or after this time belong to the running process.
_PROCESS_START = time.time()

_LOCK_NAME = ".compile_cache_eviction.lock"


@dataclasses.dataclass(frozen=True)
class CacheRegion:
    """A cache directory whose entries at ``depth`` are evicted as units.

    ``skip`` names entries above ``depth`` that are not part of this region
    (another region, or bookkeeping), so their contents are never units here.
    """

    name: str
    root: str
    depth: int = 1
    skip: frozenset[str] = frozenset()


@dataclasses.dataclass
class _Unit:
    region: str
    path: str
    last_used: float
    # (st_dev, st_ino) -> (bytes on disk, st_nlink) for each file in the unit.
    inodes: dict[tuple[int, int], tuple[int, int]]


@dataclasses.dataclass(frozen=True)
class EvictionResult:
    total_bytes: int
    freed_bytes: int
    evicted: tuple[str, ...]


def mark_used(path: str | None) -> None:
    """Record that ``path`` (a unit: a cache file or hash directory) was used.

    Non-fatal: a cache that cannot be stamped is only evicted earlier.
    """
    if not path:
        return
    with contextlib.suppress(OSError):
        os.utime(path)


def _units(region: CacheRegion) -> Iterable[str]:
    level = [region.root]
    for depth in range(region.depth):
        nxt = []
        for parent in level:
            try:
                entries = os.scandir(parent)
            except OSError:
                continue
            with entries:
                for entry in entries:
                    if entry.name.startswith(".") or entry.name in region.skip:
                        continue
                    if depth + 1 < region.depth and not entry.is_dir(
                        follow_symlinks=False
                    ):
                        continue
                    nxt.append(entry.path)
        level = nxt
    return level


def _scan(region: CacheRegion, path: str) -> _Unit | None:
    try:
        top = os.lstat(path)
    except OSError:
        return None
    last_used = top.st_mtime
    inodes: dict[tuple[int, int], tuple[int, int]] = {}

    def add(st: os.stat_result) -> None:
        inodes[(st.st_dev, st.st_ino)] = (st.st_blocks * 512, st.st_nlink)

    if os.path.isdir(path) and not os.path.islink(path):
        for dirpath, _dirnames, filenames in os.walk(path):
            for name in filenames:
                with contextlib.suppress(OSError):
                    st = os.lstat(os.path.join(dirpath, name))
                    add(st)
                    last_used = max(last_used, st.st_mtime)
    else:
        add(top)
    return _Unit(region.name, path, last_used, inodes)


def plan_eviction(
    units: Sequence[_Unit], max_bytes: int, protect_since: float
) -> tuple[int, int, list[_Unit]]:
    """Pick the least recently used units whose deletion fits ``max_bytes``.

    Returns (total bytes, bytes the chosen units free, chosen units). A file
    with names outside every region (``st_nlink`` above the names seen here) is
    never counted as freed.
    """
    names: dict[tuple[int, int], int] = {}
    size: dict[tuple[int, int], tuple[int, int]] = {}
    for unit in units:
        for ino, (nbytes, nlink) in unit.inodes.items():
            names[ino] = names.get(ino, 0) + 1
            size[ino] = (nbytes, nlink)
    total = sum(nbytes for nbytes, _ in size.values())
    if total <= max_bytes:
        return total, 0, []

    remaining = dict(names)
    freed = 0
    chosen = []
    for unit in sorted(units, key=lambda u: u.last_used):
        if total - freed <= max_bytes:
            break
        if unit.last_used >= protect_since:
            continue
        chosen.append(unit)
        for ino in unit.inodes:
            remaining[ino] -= 1
            nbytes, nlink = size[ino]
            if remaining[ino] == 0 and names[ino] >= nlink:
                freed += nbytes
    return total, freed, chosen


def _delete(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            os.remove(path)


def evict(
    regions: Sequence[CacheRegion],
    max_bytes: int,
    *,
    protect_since: float = _PROCESS_START,
    lock_dir: str | None = None,
    dry_run: bool = False,
) -> EvictionResult | None:
    """Delete least recently used units until ``regions`` fit ``max_bytes``.

    Returns None when another process holds the eviction lock (it is doing
    the same work) or ``max_bytes`` is not positive. Deleting a file another
    process has open is safe on POSIX, and a cache entry that disappears only
    costs a recompile.
    """
    if max_bytes <= 0:
        return None
    lock_dir = lock_dir or next(
        (r.root for r in regions if os.path.isdir(r.root)), None
    )
    if lock_dir is None:
        return EvictionResult(0, 0, ())
    with open(os.path.join(lock_dir, _LOCK_NAME), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        units = [
            unit
            for region in regions
            for path in _units(region)
            if (unit := _scan(region, path)) is not None
        ]
        total, freed, chosen = plan_eviction(units, max_bytes, protect_since)
        if not dry_run:
            for unit in chosen:
                _delete(unit.path)
    result = EvictionResult(total, freed, tuple(u.path for u in chosen))
    if chosen:
        logger.info(
            "Compile cache %s %d entries, %.2f of %.2f GiB (cap %.2f GiB): %s",
            "would evict" if dry_run else "evicted",
            len(chosen),
            freed / (1 << 30),
            total / (1 << 30),
            max_bytes / (1 << 30),
            ", ".join(sorted({u.region for u in chosen})),
        )
    return result


def default_regions(
    vllm_cache_root: str, tier3_root: str | None = None
) -> list[CacheRegion]:
    """The caches vllm-torchtpu writes under ``vllm_cache_root``, plus Tier-3.

    Regions that do not exist on disk are scanned as empty.
    """
    compile_cache = os.path.join(vllm_cache_root, "torch_compile_cache")
    regions = [
        # One unit per vLLM cache hash; AOT artifacts live a level deeper.
        CacheRegion(
            "torch_compile_cache", compile_cache, skip=frozenset({"torch_aot_compile"})
        ),
        CacheRegion(
            "torch_aot_compile", os.path.join(compile_cache, "torch_aot_compile")
        ),
        CacheRegion("profile_cache", os.path.join(vllm_cache_root, "profile_cache")),
        CacheRegion(
            "tpu_artifact_store", os.path.join(vllm_cache_root, "tpu_artifact_store")
        ),
    ]
    if tier3_root:
        regions.append(CacheRegion("torch_tpu_tier3", tier3_root))
    return regions


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m vllm_torchtpu.compilation.cache_eviction``: evict offline,
    e.g. before a compile-cache archive is uploaded."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm-cache-root", required=True)
    parser.add_argument("--tier3-root", default=None)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = evict(
        default_regions(args.vllm_cache_root, args.tier3_root),
        args.max_bytes,
        # Offline there is no running process to protect.
        protect_since=float("inf"),
        dry_run=args.dry_run,
    )
    out = sys.stdout
    if result is None:
        out.write("skipped: another process is evicting, or --max-bytes <= 0\n")
        return 0
    out.write(
        f"total={result.total_bytes} freed={result.freed_bytes} "
        f"evicted={len(result.evicted)}\n"
    )
    for path in result.evicted:
        out.write(f"  {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
