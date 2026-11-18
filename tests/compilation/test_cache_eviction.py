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
import fcntl
import os

import pytest

from vllm_torchtpu.compilation import cache_eviction
from vllm_torchtpu.compilation.cache_eviction import CacheRegion, evict

KIB = 1 << 10


def _write(path, nbytes, mtime):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(os.urandom(nbytes))
    os.utime(path, (mtime, mtime))
    return path


def _age(path, mtime):
    os.utime(path, (mtime, mtime))


def _du(root):
    seen, total = set(), 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            st = os.lstat(os.path.join(dirpath, name))
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_blocks * 512
    return total


@pytest.fixture
def root(tmp_path):
    return tmp_path / "vllm"


def _hash_dir(root, name, nbytes, mtime):
    d = root / "torch_compile_cache" / name
    _write(str(d / "rank_0_0" / "backbone" / "artifact"), nbytes, mtime)
    _age(str(d), mtime)
    return d


def test_no_cap_or_under_cap_deletes_nothing(root):
    old = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    regions = cache_eviction.default_regions(str(root))

    assert evict(regions, 0) is None
    result = evict(regions, 10 << 20, protect_since=2000)
    assert result.evicted == ()
    assert old.exists()


def test_least_recently_used_units_go_first_until_under_cap(root):
    oldest = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    middle = _hash_dir(root, "bbbb", 64 * KIB, 2000)
    newest = _hash_dir(root, "cccc", 64 * KIB, 3000)
    regions = cache_eviction.default_regions(str(root))
    cap = _du(root) - 32 * KIB

    result = evict(regions, cap, protect_since=10_000)

    assert result.evicted == (str(oldest),)
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert _du(root) <= cap


def test_mark_used_keeps_a_unit_that_was_only_read(root):
    read_every_start = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    stale = _hash_dir(root, "bbbb", 64 * KIB, 2000)
    cache_eviction.mark_used(str(read_every_start))

    result = evict(
        cache_eviction.default_regions(str(root)),
        _du(root) - 32 * KIB,
        protect_since=10_000,
    )

    assert result.evicted == (str(stale),)
    assert read_every_start.exists()


def test_units_used_by_this_process_are_never_evicted(root):
    in_use = _hash_dir(root, "aaaa", 64 * KIB, 5000)
    stale = _hash_dir(root, "bbbb", 64 * KIB, 1000)

    result = evict(cache_eviction.default_regions(str(root)), 1, protect_since=4000)

    assert result.evicted == (str(stale),)
    assert in_use.exists()


def test_regions_are_capped_together(root):
    store = root / "tpu_artifact_store"
    old_entry = _write(str(store / "e1"), 64 * KIB, 1000)
    profile = root / "profile_cache" / "p1"
    _write(str(profile / "rank_0_0" / "artifact"), 64 * KIB, 1500)
    _age(str(profile), 1500)
    newer = _hash_dir(root, "aaaa", 64 * KIB, 3000)

    result = evict(
        cache_eviction.default_regions(str(root)),
        _du(root) - 32 * KIB,
        protect_since=10_000,
    )

    assert result.evicted == (old_entry,)
    assert profile.exists() and newer.exists()


def test_aot_artifacts_are_their_own_units(root):
    aot = root / "torch_compile_cache" / "torch_aot_compile" / "h1"
    _write(str(aot / "model"), 64 * KIB, 1000)
    _age(str(aot), 1000)
    newer = _hash_dir(root, "aaaa", 64 * KIB, 3000)

    result = evict(
        cache_eviction.default_regions(str(root)),
        _du(root) - 32 * KIB,
        protect_since=10_000,
    )

    assert result.evicted == (str(aot),)
    assert (root / "torch_compile_cache" / "torch_aot_compile").exists()
    assert newer.exists()


def test_a_hard_linked_file_is_freed_only_with_its_last_name(root):
    """The artifact store links vLLM's per-rank artifacts: deleting the old
    hash dir alone frees nothing, so eviction goes on to the store entry."""
    old = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    store = root / "tpu_artifact_store"
    store.mkdir(parents=True)
    linked = store / "e1"
    os.link(old / "rank_0_0" / "backbone" / "artifact", linked)
    _age(str(linked), 2000)
    newest = _hash_dir(root, "bbbb", 64 * KIB, 3000)

    regions = cache_eviction.default_regions(str(root))
    total, freed, chosen = cache_eviction.plan_eviction(
        [
            u
            for r in regions
            for p in cache_eviction._units(r)
            if (u := cache_eviction._scan(r, p))
        ],
        _du(root) - 32 * KIB,
        protect_since=10_000,
    )

    assert [u.path for u in chosen] == [str(old), str(linked)]
    assert freed >= 64 * KIB
    assert newest.exists()


def test_a_file_linked_outside_every_region_is_never_counted_freed(root, tmp_path):
    old = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    os.link(old / "rank_0_0" / "backbone" / "artifact", tmp_path / "elsewhere")
    newer = _hash_dir(root, "bbbb", 64 * KIB, 3000)

    result = evict(
        cache_eviction.default_regions(str(root)),
        _du(root) - 32 * KIB,
        protect_since=10_000,
        dry_run=True,
    )

    # Deleting "aaaa" frees nothing, so eviction moves on to "bbbb" as well.
    assert result.evicted == (str(old), str(newer))
    assert result.freed_bytes == _du(newer)
    assert old.exists()  # dry run


def test_dry_run_deletes_nothing(root):
    old = _hash_dir(root, "aaaa", 64 * KIB, 1000)

    result = evict(
        cache_eviction.default_regions(str(root)), 1, protect_since=10_000, dry_run=True
    )

    assert result.evicted == (str(old),)
    assert old.exists()


def test_a_held_lock_skips_eviction(root):
    old = _hash_dir(root, "aaaa", 64 * KIB, 1000)
    regions = cache_eviction.default_regions(str(root))
    lock_path = root / "torch_compile_cache" / cache_eviction._LOCK_NAME
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert evict(regions, 1, protect_since=10_000) is None
    assert old.exists()


def test_a_region_that_does_not_exist_is_empty(tmp_path):
    regions = [CacheRegion("missing", str(tmp_path / "nope"))]
    assert evict(regions, 1).evicted == ()


def test_cli_evicts_offline_without_protecting_recent_units(root, capsys):
    recent = _hash_dir(root, "aaaa", 64 * KIB, 9e9)

    assert (
        cache_eviction.main(["--vllm-cache-root", str(root), "--max-bytes", "1"]) == 0
    )

    assert not recent.exists()
    assert "evicted=1" in capsys.readouterr().out
