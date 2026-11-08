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
"""Partial-tail offloading under DCP in TPURaidenOffloadingScheduler."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.output import KVConnectorBlockState
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingKVEventsConfig,
    PrepareStoreOutput,
    RequestOffloadingContext,
)
from vllm.v1.request import Request

from vllm_torchtpu.offload import raiden_connector
from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler
from vllm_torchtpu.offload.raiden_store import (
    RaidenLoadStoreSpec,
    RaidenOffloadingManager,
)


def make_scheduler(
    dcp=4,
    recurrent_block_size=None,
    eagle=False,
    prompt_tokens=30,
    dcp_partial_tail=True,
    hash_size=4,
):
    """One attention group (`hash_size` tokens per rank, scaled by DCP) and
    two align-mode recurrent groups, with a mock Raiden manager."""
    recurrent_block_size = recurrent_block_size or hash_size
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=hash_size, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
            is_eagle_group=eagle,
        ),
        *(
            KVCacheGroupSpec(
                [f"recurrent_{i}"],
                MambaSpec(
                    block_size=recurrent_block_size,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
            for i in range(2)
        ),
    ]
    tokens_per_block = (hash_size * dcp, recurrent_block_size, recurrent_block_size)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=dcp, decode_context_parallel_size=dcp
        ),
        speculative_config=None,
        cache_config=SimpleNamespace(
            enable_prefix_caching=True, prefix_cache_retention_interval=None
        ),
    )
    manager = MagicMock(spec=RaidenOffloadingManager)
    manager.on_new_request.return_value = RequestOffloadingContext()
    manager.lookup.return_value = LookupResult.MISS
    manager.prepare_load.side_effect = lambda keys, context: RaidenLoadStoreSpec(keys)
    manager.prepare_store.side_effect = lambda keys, context: PrepareStoreOutput(
        keys_to_store=list(keys), store_spec=RaidenLoadStoreSpec(keys), evicted_keys=[]
    )
    # Only the fields the scheduler reads.
    spec = SimpleNamespace(
        config=SimpleNamespace(
            groups=[
                SimpleNamespace(
                    group_id=idx,
                    tokens_per_block=size,
                    sliding_window_size_in_chunks=None if idx == 0 else 1,
                )
                for idx, size in enumerate(tokens_per_block)
            ]
        ),
        tokens_per_block=tokens_per_block,
        tokens_per_hash=hash_size,
        blocks_per_chunk=1,
        offload_prompt_only=True,
        extra_config={"dcp_partial_tail": dcp_partial_tail},
        kv_events_config=OffloadingKVEventsConfig(
            enable_kv_cache_events=False, self_describing_kv_events=False
        ),
        get_manager=lambda: manager,
    )
    scheduler = TPURaidenOffloadingScheduler(
        spec,
        vllm_config,
        KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups),
    )
    init_none_hash(sha256)
    request = Request(
        request_id="producer",
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, sha256),
    )
    scheduler.on_new_request(request)
    state = scheduler._req_status[request.request_id]
    state.update_offload_keys()
    state.group_states[0].block_ids = [11, 12]
    state.group_states[1].block_ids = [0] * 6 + [21]
    state.group_states[2].block_ids = [0] * 6 + [22]
    return scheduler, request


def step(handoffs=(), scheduled=2, preempted=(), finished=()):
    return SimpleNamespace(
        kv_connector_block_state=KVConnectorBlockState(
            req_ids={"producer"},
            resolve_block_ids=MagicMock(),
            boundary_state_offloads={"producer": list(handoffs)},
        ),
        num_scheduled_tokens={"producer": scheduled},
        preempted_req_ids=set(preempted),
        finished_req_ids=set(finished),
    )


def keys(scheduler, request, boundary):
    context = scheduler._req_status[request.request_id].req_context
    return [
        scheduler._make_boundary_key(request, group, boundary, context)
        for group in range(3)
    ]


def serve(scheduler, stored, pending=()):
    scheduler.manager.lookup.side_effect = lambda key, context: (
        LookupResult.HIT_PENDING
        if key in pending
        else LookupResult.HIT
        if key in stored
        else LookupResult.MISS
    )


@pytest.mark.parametrize("dcp", [4, 32])
def test_dcp_tail_waits_for_the_step_that_writes_its_attention_and_states(dcp):
    scheduler, request = make_scheduler(dcp=dcp)
    request.num_computed_tokens = 16
    assert (
        scheduler._build_partial_tail_store_jobs(
            step([(1, 91, 28), (2, 92, 28)], scheduled=12)
        )
        == {}
    )
    scheduler.manager.prepare_store.assert_not_called()
    request.num_computed_tokens = 28
    [job] = scheduler._build_partial_tail_store_jobs(step()).values()
    assert job.dst_spec.keys == keys(scheduler, request, 28)
    assert job.src_spec.block_ids.tolist() == [12 if dcp == 4 else 11, 91, 92]
    assert job.src_spec.block_indices == [1 if dcp == 4 else 0, 6, 6]
    assert job.src_spec.group_sizes == [1, 1, 1]
    assert not scheduler._deferred_tail_states


@pytest.mark.parametrize("local_tokens", [0, 16])
def test_host_tail_loads_each_group_at_its_own_block_index(local_tokens):
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (
        28 - local_tokens,
        True,
    )
    attention = [KVCacheBlock(31), KVCacheBlock(32)]
    if local_tokens:
        attention[0]._block_hash = b"local-full-block"
    null = KVCacheBlock(0, is_null=True)
    blocks = KVCacheBlocks(
        (attention, [null] * 6 + [KVCacheBlock(41)], [null] * 6 + [KVCacheBlock(42)])
    )
    scheduler.update_state_after_alloc(request, blocks, 28 - local_tokens)
    [job] = scheduler._current_batch_load_jobs.values()
    assert job.src_spec.keys == (
        ([] if local_tokens else [keys(scheduler, request, 16)[0]])
        + keys(scheduler, request, 28)
    )
    assert job.dst_spec.block_ids.tolist() == ([] if local_tokens else [31]) + [
        32,
        41,
        42,
    ]
    assert job.dst_spec.block_indices == [local_tokens // 16, 6, 6]
    state = scheduler._req_status[request.request_id]
    assert [g.next_stored_chunk_idx for g in state.group_states] == [1, 7, 7]


@pytest.mark.parametrize("dcp", [4, 32])
def test_hash_aligned_prompt_tail_uses_its_final_states_without_a_handoff(dcp):
    scheduler, request = make_scheduler(dcp=dcp, prompt_tokens=28)
    request.num_computed_tokens = 16
    [job] = scheduler._build_partial_tail_store_jobs(step(scheduled=12)).values()
    assert job.dst_spec.keys == keys(scheduler, request, 28)
    assert job.src_spec.block_ids.tolist() == [12 if dcp == 4 else 11, 21, 22]
    assert job.src_spec.block_indices == [1 if dcp == 4 else 0, 6, 6]


def test_a_live_decode_state_cannot_replace_a_boundary_handoff():
    scheduler, request = make_scheduler(prompt_tokens=28)
    request.sampling_params.max_tokens = 2
    request.num_computed_tokens = 16
    assert scheduler._build_partial_tail_store_jobs(step(scheduled=12)) == {}
    scheduler.manager.prepare_store.assert_not_called()


@pytest.mark.parametrize("missing_group", [0, 1, 2])
def test_tail_hit_requires_every_group_at_the_same_boundary(missing_group):
    scheduler, request = make_scheduler()
    stored = set(keys(scheduler, request, 16) + keys(scheduler, request, 28))
    stored.remove(keys(scheduler, request, 28)[missing_group])
    serve(scheduler, stored)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, True)


@pytest.mark.parametrize("reason", ["reallocated", "preempted", "finished"])
def test_stale_deferred_sources_are_never_published(reason):
    scheduler, request = make_scheduler()
    request.num_computed_tokens = 16
    scheduler._build_partial_tail_store_jobs(
        step([(1, 91, 28), (2, 92, 28)], scheduled=12)
    )
    request.num_computed_tokens = 28
    output = step()
    if reason == "reallocated":
        scheduler._current_batch_allocated_block_ids = {91}
    elif reason == "preempted":
        output.preempted_req_ids = {request.request_id}
    else:
        output.finished_req_ids = {request.request_id}
    assert scheduler._build_partial_tail_store_jobs(output) == {}
    scheduler.manager.prepare_store.assert_not_called()
    assert not scheduler._deferred_tail_states


@pytest.mark.parametrize("limit", [0, 24])
def test_partial_store_honors_max_offload_tokens(limit):
    scheduler, request = make_scheduler()
    scheduler._req_status[request.request_id].max_offload_tokens = limit
    request.num_computed_tokens = 28
    assert (
        scheduler._build_partial_tail_store_jobs(step([(1, 91, 28), (2, 92, 28)])) == {}
    )
    scheduler.manager.prepare_store.assert_not_called()


@pytest.mark.parametrize("recurrent_block_size,eagle", [(8, False), (4, True)])
def test_unsupported_group_layout_keeps_the_existing_path(recurrent_block_size, eagle):
    scheduler, _ = make_scheduler(
        recurrent_block_size=recurrent_block_size, eagle=eagle
    )
    assert not scheduler.config.supports_partial_tail
    assert not scheduler._mixed_partial_tail


def test_tail_hit_does_not_require_an_evicted_earlier_recurrent_checkpoint():
    scheduler, request = make_scheduler()
    serve(scheduler, {keys(scheduler, request, 16)[0], *keys(scheduler, request, 28)})
    assert scheduler.get_num_new_matched_tokens(request, 0) == (28, True)


@pytest.mark.parametrize("dcp", [4, 32])
@pytest.mark.parametrize("local_blocks", [0, 1])
def test_tail_lookup_extends_an_unaligned_recurrent_checkpoint(dcp, local_blocks):
    block_size = 4 * dcp
    complete_boundary = 2 * block_size - 4
    tail_boundary = 4 * block_size - 8
    local_tokens = local_blocks * block_size
    scheduler, request = make_scheduler(dcp=dcp, prompt_tokens=tail_boundary + 4)
    stored = {
        keys(scheduler, request, boundary)[0]
        for boundary in (block_size, 2 * block_size, 3 * block_size)
    }
    stored.update(keys(scheduler, request, complete_boundary)[1:])
    stored.update(keys(scheduler, request, tail_boundary))
    serve(scheduler, stored)
    state = scheduler._req_status[request.request_id]
    state.num_locally_computed_tokens = local_tokens
    assert scheduler._lookup_complete_chunks(state) == complete_boundary - local_tokens
    assert scheduler.get_num_new_matched_tokens(request, local_tokens) == (
        tail_boundary - local_tokens,
        True,
    )
    assert state.partial_tail_boundary == tail_boundary


def test_tail_cannot_bridge_a_missing_attention_prefix_block():
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 28)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


def test_local_partial_hit_skips_the_host_lookup():
    # A load starts on an attention block boundary, so a local hit that ends
    # inside a block loads nothing.
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    assert scheduler.get_num_new_matched_tokens(request, 12) == (0, False)
    scheduler.manager.lookup.assert_not_called()


def test_ready_tail_is_usable_while_its_full_block_store_is_pending():
    scheduler, request = make_scheduler()
    serve(
        scheduler,
        set(keys(scheduler, request, 12)),
        pending={keys(scheduler, request, 16)[0]},
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (12, True)


@pytest.mark.parametrize("boundary,group_idx", [(16, 0), (28, 0), (28, 1)])
def test_tail_waits_for_overlapping_attention_or_recurrent_loads(boundary, group_idx):
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    scheduler._chunks_being_loaded.add(keys(scheduler, request, boundary)[group_idx])
    assert scheduler.get_num_new_matched_tokens(request, 0) == (None, False)
    assert scheduler._req_status[request.request_id].partial_tail_boundary is None
    scheduler._chunks_being_loaded.clear()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (28, True)


def test_tail_does_not_wait_for_an_earlier_recurrent_state_it_will_not_load():
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    scheduler._chunks_being_loaded.add(keys(scheduler, request, 16)[1])
    assert scheduler.get_num_new_matched_tokens(request, 0) == (28, True)


def test_tail_lookup_honors_the_load_caps():
    scheduler, request = make_scheduler()
    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    assert scheduler.get_num_new_matched_tokens(request, 0, 24) == (16, True)
    scheduler._req_status[request.request_id].max_load_tokens = 24
    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, True)


def test_opt_out_keeps_complete_chunk_offload_for_the_dcp_shape(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(raiden_connector, "logger", logger)
    scheduler, request = make_scheduler(dcp_partial_tail=False)
    assert not scheduler._mixed_partial_tail
    assert not scheduler.config.supports_partial_tail
    assert scheduler._partial_tail_block_size == 0
    logger.info.assert_not_called()

    serve(scheduler, set(keys(scheduler, request, 16) + keys(scheduler, request, 28)))
    assert scheduler.get_num_new_matched_tokens(request, 0) == (16, True)
    assert scheduler._req_status[request.request_id].partial_tail_boundary is None

    # Each recurrent state is stored alone under its chunk key; no partly
    # filled attention block is stored.
    request.num_computed_tokens = 28
    jobs = scheduler._build_partial_tail_store_jobs(step([(1, 91, 28), (2, 92, 28)]))
    assert sorted(job.dst_spec.keys for job in jobs.values()) == sorted(
        [[key] for key in keys(scheduler, request, 28)[1:]]
    )
    assert not scheduler._deferred_tail_states


def test_opt_in_without_dcp_keeps_the_existing_path(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(raiden_connector, "logger", logger)
    scheduler, request = make_scheduler(dcp=1)
    assert not scheduler._mixed_partial_tail
    assert not scheduler.config.supports_partial_tail
    assert scheduler._partial_tail_block_size == 0
    [call] = logger.info.call_args_list
    assert "stays off" in call.args[0]

    request.num_computed_tokens = 28
    jobs = scheduler._build_partial_tail_store_jobs(step([(1, 91, 28), (2, 92, 28)]))
    assert sorted(job.dst_spec.keys for job in jobs.values()) == sorted(
        [[key] for key in keys(scheduler, request, 28)[1:]]
    )
    assert not scheduler._deferred_tail_states


def test_k3_sized_dcp_tail_engages(monkeypatch):
    # 512-token states and hashes with DCP 32: 16384-token attention blocks.
    logger = MagicMock()
    monkeypatch.setattr(raiden_connector, "logger", logger)
    block, tail = 16384, 16384 + 3 * 512
    scheduler, request = make_scheduler(
        dcp=32, hash_size=512, prompt_tokens=block + 8 * 512
    )
    assert scheduler._mixed_partial_tail
    assert scheduler.config.supports_partial_tail
    assert scheduler._partial_tail_block_size == block
    [call] = logger.info.call_args_list
    assert "enabled" in call.args[0]

    serve(
        scheduler, {keys(scheduler, request, block)[0], *keys(scheduler, request, tail)}
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (tail, True)
    assert scheduler._req_status[request.request_id].partial_tail_boundary == tail
