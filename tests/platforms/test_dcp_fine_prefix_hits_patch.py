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
"""Prefix-cache hits finer than the DCP logical block for one KV cache group.

See `_patch_vllm_dcp_fine_prefix_hits`: with DCP=16 and block_size=256 the
manager block is 4096 tokens, and ``--prefix-match-unit 256`` makes a shared
prefix hit at 256-token granularity, copy-on-writing the shared tail block.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_coordinator, kv_cache_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_torchtpu import _patch_vllm_dcp_fine_prefix_hits

BLOCK_SIZE = 256
DCP = 16
LOGICAL_BLOCK = BLOCK_SIZE * DCP  # 4096
UNIT = 256


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    """Apply the patch on top of upstream, restoring everything afterwards."""
    cls = kv_cache_coordinator.UnitaryKVCacheCoordinator
    monkeypatch.setattr(cls, "__init__", inspect.unwrap(cls.__init__))
    monkeypatch.setattr(
        cls, "find_longest_cache_hit", inspect.unwrap(cls.find_longest_cache_hit)
    )
    monkeypatch.setattr(cls, "_tpu_dcp_fine_prefix_patch", False, raising=False)
    monkeypatch.setattr(
        kv_cache_utils,
        "resolve_kv_cache_block_sizes",
        inspect.unwrap(kv_cache_utils.resolve_kv_cache_block_sizes),
    )
    _patch_vllm_dcp_fine_prefix_hits()


def _mla_spec(dtype=torch.bfloat16, head_size=576):
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=head_size, dtype=dtype
    )


def _kv_cache_config(*specs, num_blocks=64):
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i, spec in enumerate(specs)
        ],
    )


def _stub_config(
    *, unit=UNIT, dcp=DCP, pcp=1, caching=True, connector=False, block_size=BLOCK_SIZE
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=caching,
            prefix_match_unit=unit,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
        ),
        kv_transfer_config=(
            SimpleNamespace(kv_connector=connector)
            if isinstance(connector, str)
            else object()
            if connector
            else None
        ),
    )


def _resolve(kv_cache_config, vllm_config):
    return kv_cache_utils.resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def test_resolve_gated_to_single_group_dcp():
    mla = _kv_cache_config(_mla_spec())
    assert _resolve(mla, _stub_config()) == (LOGICAL_BLOCK, UNIT)
    # Opt-in: every unmet condition keeps upstream (hash == scheduler block).
    for config in (
        _stub_config(unit=None),
        _stub_config(caching=False),
        _stub_config(connector=True),
        _stub_config(unit=LOGICAL_BLOCK),
    ):
        assert _resolve(mla, config) == (LOGICAL_BLOCK, LOGICAL_BLOCK)
    assert _resolve(mla, _stub_config(dcp=1)) == (BLOCK_SIZE, BLOCK_SIZE)
    # CPU offloading connectors keep the fine unit; P/D connectors do not.
    for name in ("TPURaidenOffloadingConnector", "OffloadingConnector"):
        assert _resolve(mla, _stub_config(connector=name)) == (LOGICAL_BLOCK, UNIT)
    assert _resolve(mla, _stub_config(connector="NixlConnector")) == (
        LOGICAL_BLOCK,
        LOGICAL_BLOCK,
    )
    swa = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    assert _resolve(_kv_cache_config(swa), _stub_config()) == (
        LOGICAL_BLOCK,
        LOGICAL_BLOCK,
    )
    with pytest.raises(ValueError, match="divisible"):
        _resolve(mla, _stub_config(unit=384))


def test_resolve_leaves_multi_group_to_previous_resolver():
    mamba = MambaSpec(
        block_size=LOGICAL_BLOCK,
        shapes=((3, 64),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
    )
    config = _kv_cache_config(_mla_spec(), mamba)
    vllm_config = _stub_config()
    upstream = inspect.unwrap(kv_cache_utils.resolve_kv_cache_block_sizes)
    assert _resolve(config, vllm_config) == upstream(config, vllm_config)


def _scheduler(monkeypatch, *, prefix_match_unit, use_mtp=False):
    """Build a real Scheduler at DCP=16 the way EngineCore does."""
    # opt-125m's 2048 positions are irrelevant: the scheduler never runs it.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    model_config = ModelConfig(
        model="facebook/opt-125m", dtype="float16", seed=42, max_model_len=8192
    )
    speculative_config = None
    if use_mtp:
        # Stand in for a multi-module MTP drafter (k=3) without loading one.
        monkeypatch.setattr(SpeculativeConfig, "use_eagle", lambda self: True)
        monkeypatch.setattr(
            SpeculativeConfig, "use_multi_module_mtp", lambda self: True
        )
        speculative_config = SpeculativeConfig(
            model="ngram",
            num_speculative_tokens=3,
            prompt_lookup_max=3,
            prompt_lookup_min=1,
        )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=8192,
            max_model_len=8192,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=True,
        ),
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_config.parallel_config.decode_context_parallel_size = DCP
    vllm_config.cache_config.prefix_match_unit = prefix_match_unit
    kv_cache_config = _kv_cache_config(_mla_spec(), num_blocks=64)
    vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    register_all_kvcache_specs(vllm_config)
    # EngineCore: resolve, then build the scheduler and request hasher.
    scheduler_block_size, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        log_stats=True,
    )
    init_none_hash(sha256)
    return scheduler, get_request_block_hasher(hash_block_size, sha256)


def _request(request_id, prompt, hasher):
    params = SamplingParams(max_tokens=8, ignore_eos=True)
    params.update_from_generation_config({}, 2)
    return Request(request_id, prompt, params, None, block_hasher=hasher)


# A: one full logical block plus 1000 tokens. Its prompt tail registers a
# partial entry at the last 256 boundary inside block 1: 19 * 256 = 4864.
PROMPT_A = list(range(LOGICAL_BLOCK + 1000))
TAIL_BOUNDARY = (len(PROMPT_A) // UNIT) * UNIT
# B shares A's first 4900 tokens, then diverges.
PROMPT_B = PROMPT_A[:4900] + [100_000 + i for i in range(300)]


def _schedule_a_then_b(scheduler, hasher):
    request_a = _request("a", PROMPT_A, hasher)
    scheduler.add_request(request_a)
    out_a = scheduler.schedule()
    assert [r.req_id for r in out_a.scheduled_new_reqs] == ["a"]
    assert not out_a.kv_cache_block_copies
    request_b = _request("b", PROMPT_B, hasher)
    scheduler.add_request(request_b)
    out_b = scheduler.schedule()
    (new_b,) = out_b.scheduled_new_reqs
    assert new_b.req_id == "b"
    return request_a, new_b, out_b


def test_scheduler_mid_block_hit_emits_cow_copy(monkeypatch):
    scheduler, hasher = _scheduler(monkeypatch, prefix_match_unit=UNIT)
    assert scheduler.block_size == LOGICAL_BLOCK
    assert scheduler.hash_block_size == UNIT
    assert scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits

    request_a, new_b, out_b = _schedule_a_then_b(scheduler, hasher)

    # Hit lands on the hash boundary inside A's second 4096-token block.
    assert new_b.num_computed_tokens == TAIL_BOUNDARY
    assert TAIL_BOUNDARY % LOGICAL_BLOCK != 0
    manager = scheduler.kv_cache_manager
    (a_blocks,) = manager.get_block_ids("a")
    (b_blocks,) = manager.get_block_ids("b")
    # Block 0 is shared; B's tail is a private copy of A's block 1.
    assert b_blocks[0] == a_blocks[0]
    assert b_blocks[1] != a_blocks[1]
    copies = out_b.kv_cache_block_copies
    assert [(c.src_block_id, c.dst_block_id) for c in copies] == [
        (a_blocks[1], b_blocks[1])
    ]
    assert out_b.num_scheduled_tokens["b"] == len(PROMPT_B) - TAIL_BOUNDARY


def test_scheduler_default_keeps_block_granularity(monkeypatch):
    """Without --prefix-match-unit the hit stays at the 4096-token block."""
    scheduler, hasher = _scheduler(monkeypatch, prefix_match_unit=None)
    assert scheduler.hash_block_size == LOGICAL_BLOCK
    assert not scheduler.kv_cache_manager.coordinator.enable_partial_hash_hits

    _, new_b, out_b = _schedule_a_then_b(scheduler, hasher)

    assert new_b.num_computed_tokens == LOGICAL_BLOCK
    assert not out_b.kv_cache_block_copies


def test_scheduler_mtp_drops_one_hash_unit(monkeypatch):
    scheduler, hasher = _scheduler(monkeypatch, prefix_match_unit=UNIT, use_mtp=True)
    assert scheduler.use_eagle
    assert scheduler.num_prefill_lookahead == 3

    _, new_b, out_b = _schedule_a_then_b(scheduler, hasher)

    # One 256-token hash unit is dropped, not the whole 4096-token block.
    assert new_b.num_computed_tokens == TAIL_BOUNDARY - UNIT
    manager = scheduler.kv_cache_manager
    (a_blocks,) = manager.get_block_ids("a")
    (b_blocks,) = manager.get_block_ids("b")
    assert [(c.src_block_id, c.dst_block_id) for c in out_b.kv_cache_block_copies] == [
        (a_blocks[1], b_blocks[1])
    ]


def test_coordinator_rejects_unit_below_mtp_lookahead():
    config = _kv_cache_config(_mla_spec())
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        kv_cache_coordinator.UnitaryKVCacheCoordinator(
            config,
            max_model_len=8192,
            max_in_flight_tokens=8192,
            use_eagle=True,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=DCP,
            pcp_world_size=1,
            scheduler_block_size=LOGICAL_BLOCK,
            hash_block_size=2,
            num_prefill_lookahead=3,
        )


def test_base_init_accepts_every_unitary_init_parameter():
    """The fine path forwards UnitaryKVCacheCoordinator.__init__'s bound
    arguments to KVCacheCoordinator.__init__ by name; fail loudly on a vLLM
    bump that makes the two signatures diverge."""
    unitary = inspect.signature(
        kv_cache_coordinator.UnitaryKVCacheCoordinator.__init__.__wrapped__
    ).parameters
    base = inspect.signature(kv_cache_coordinator.KVCacheCoordinator.__init__)
    assert set(unitary) <= set(base.parameters)


def test_patch_is_idempotent():
    resolve = kv_cache_utils.resolve_kv_cache_block_sizes
    init = kv_cache_coordinator.UnitaryKVCacheCoordinator.__init__
    _patch_vllm_dcp_fine_prefix_hits()
    assert kv_cache_utils.resolve_kv_cache_block_sizes is resolve
    assert kv_cache_coordinator.UnitaryKVCacheCoordinator.__init__ is init
