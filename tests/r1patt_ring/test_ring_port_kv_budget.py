# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""The KV cache manager's block-count override budget leaves room for the ring
hand-off's buffers, as TPUWorker.determine_available_memory does (port of
upstream PR 2527 commit 712a5873b onto base15). Explicit kv_cache_memory_bytes
(the PCP microbench A line) is used as is."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_torchtpu import utils
from vllm_torchtpu.runner import kv_cache_manager as kcm


class _Runner(MagicMock):
    def pp_handoff_hbm_reserve_bytes(self):
        return self._reserve


def _manager(monkeypatch, reserve, kv_bytes=None):
    monkeypatch.setattr(
        utils, "compute_hbm_budget",
        lambda *_: utils.HbmBudget(total_limit=0, total_used=0, cap=0, available=7 * utils.GBYTES),
    )
    monkeypatch.setattr(utils, "estimate_kv_connector_hbm_reserve", lambda *_: 0)
    monkeypatch.setattr(utils, "min_across_ep_dp", lambda _pc, x: x)
    runner = _Runner()
    runner._reserve = reserve
    runner.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_bytes, gpu_memory_utilization=0.9)
    mgr = kcm.KVCacheManager.__new__(kcm.KVCacheManager)
    mgr.runner = runner
    return mgr


def test_override_budget_reserves_the_ring_buffers(monkeypatch):
    assert _manager(monkeypatch, utils.GBYTES)._available_kv_cache_hbm() == 6 * utils.GBYTES
    assert _manager(monkeypatch, 0)._available_kv_cache_hbm() == 7 * utils.GBYTES


def test_explicit_kv_bytes_are_used_as_is(monkeypatch):
    mgr = _manager(monkeypatch, utils.GBYTES, kv_bytes=25_000_000_000)
    assert mgr._available_kv_cache_hbm() == 25_000_000_000
