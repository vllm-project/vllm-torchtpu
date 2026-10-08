# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the deferred weight page cache eviction.

Eviction must wait until every model is loaded: an MTP draft re-reads the
target checkpoint, and evicting between the two loads makes local rank 0
re-read it from cold disk while every other rank waits. No TPU, no network.
"""

import os
import threading

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu import model_loader_patches as mlp


@pytest.fixture
def evicted(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", True, raising=False)
    monkeypatch.setattr(mlp, "_evict_checkpoint_page_cache", calls.append)
    monkeypatch.setattr(mlp, "_loaded_weight_files", {})
    monkeypatch.setenv("LOCAL_RANK", "0")
    return calls


def test_files_from_every_load_evicted_once_in_load_order(evicted):
    # Target, then an MTP draft over the same shards.
    mlp._loaded_weight_files.update(dict.fromkeys(["a", "b"]))
    mlp._loaded_weight_files.update(dict.fromkeys(["b", "a", "c"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["a", "b", "c"]]
    mlp.evict_loaded_weights_page_cache()
    assert evicted == [["a", "b", "c"]]


def test_files_evicted_concurrently(tmp_path, monkeypatch):
    """Multiple checkpoint files are evicted in parallel rather than serially."""
    paths = []
    for i in range(4):
        p = tmp_path / f"shard-{i}.safetensors"
        p.write_bytes(b"x" * 16)
        paths.append(str(p))

    barrier = threading.Barrier(len(paths), timeout=5.0)
    fadvise_fds: list[int] = []
    lock = threading.Lock()

    def fake_fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        # All 4 files must enter posix_fadvise concurrently for the barrier to pass.
        barrier.wait()
        with lock:
            fadvise_fds.append(fd)

    monkeypatch.setattr(os, "posix_fadvise", fake_fadvise)
    mlp._evict_checkpoint_page_cache(paths)
    assert len(fadvise_fds) == 4


def test_only_local_rank_zero_evicts(evicted, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == []
    assert not mlp._loaded_weight_files


def test_disabled_leaves_files_untouched(evicted, monkeypatch):
    monkeypatch.setattr(envs, "TPU_EVICT_WEIGHTS_PAGE_CACHE", False, raising=False)
    mlp._loaded_weight_files.update(dict.fromkeys(["a"]))
    mlp.evict_loaded_weights_page_cache()
    assert evicted == []
