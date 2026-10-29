# SPDX-License-Identifier: Apache-2.0
"""TPU_OFFLOAD_KEEP_EAGLE_TAIL: offload tier hits keep their last chunk.

With speculative decoding on and no KV cache group annotated as a drafter
group, the pinned vLLM flags every group as one, and every tier lookup pops
the last chunk of the hit. keep_unannotated_eagle_tail clears that flag on
the unannotated groups only (vLLM main's behavior, vllm-project/vllm#52771).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    SchedulerOffloadConfig,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

from vllm_torchtpu.offload import raiden_connector
from vllm_torchtpu.offload.raiden_connector import keep_unannotated_eagle_tail

TPC = 4096


def _group(idx, is_eagle_group):
    return GroupOffloadConfig(
        group_idx=idx,
        tokens_per_block=TPC,
        tokens_per_chunk=TPC,
        hashes_per_chunk=16,
        kv_event_group_spec=MagicMock(),
        sliding_window_size_in_chunks=None,
        is_eagle_group=is_eagle_group,
    )


def _config(*flags):
    return SchedulerOffloadConfig(
        kv_group_configs=tuple(_group(i, f) for i, f in enumerate(flags)),
        blocks_per_chunk=1,
        tokens_per_hash=TPC // 16,
        num_workers=16,
        offload_prompt_only=False,
        supports_partial_tail=False,
    )


def _kv_groups(*annotated):
    return [
        KVCacheGroupSpec(
            layer_names=[f"layers.{i}"],
            kv_cache_spec=MagicMock(),
            is_eagle_group=a,
        )
        for i, a in enumerate(annotated)
    ]


def _flags(config):
    return [g.is_eagle_group for g in config.kv_group_configs]


def test_shared_group_mtp_flag_is_cleared():
    # GLM-5.2: one group holds target and MTP layers; vLLM flags it only
    # because speculative decoding is on.
    config, cleared = keep_unannotated_eagle_tail(_config(True), _kv_groups(False))
    assert cleared == (0,)
    assert _flags(config) == [False]


def test_annotated_drafter_group_keeps_its_flag():
    config, cleared = keep_unannotated_eagle_tail(
        _config(False, True), _kv_groups(False, True)
    )
    assert cleared == ()
    assert _flags(config) == [False, True]


def test_only_unannotated_groups_are_cleared():
    config, cleared = keep_unannotated_eagle_tail(
        _config(True, True), _kv_groups(False, True)
    )
    assert cleared == (0,)
    assert _flags(config) == [False, True]


def test_no_flagged_group_returns_the_same_config():
    original = _config(False, False)
    config, cleared = keep_unannotated_eagle_tail(original, _kv_groups(False, False))
    assert cleared == ()
    assert config is original


def test_other_fields_are_unchanged():
    original = _config(True)
    config, _ = keep_unannotated_eagle_tail(original, _kv_groups(False))
    assert config._replace(kv_group_configs=()) == original._replace(
        kv_group_configs=()
    )
    assert (
        config.kv_group_configs[0]._replace(is_eagle_group=True)
        == original.kv_group_configs[0]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_scheduler_applies_it_only_when_enabled(monkeypatch, enabled):
    monkeypatch.setattr(
        raiden_connector.tpu_envs, "TPU_OFFLOAD_KEEP_EAGLE_TAIL", enabled
    )
    base = raiden_connector.OffloadingConnectorScheduler

    def fake_base_init(self, spec, vllm_config, kv_cache_config):
        self.config = _config(True)
        self.manager = MagicMock(spec=raiden_connector.RaidenOffloadingManager)

    monkeypatch.setattr(base, "__init__", fake_base_init)
    spec = SimpleNamespace(extra_config={})
    kv_cache_config = SimpleNamespace(kv_cache_groups=_kv_groups(False))
    sched = raiden_connector.TPURaidenOffloadingScheduler(
        spec, MagicMock(), kv_cache_config
    )
    assert _flags(sched.config) == [not enabled]
