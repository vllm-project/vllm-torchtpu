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

from __future__ import annotations

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request

from vllm_torchtpu.core.mamba_async_free import (
    enable_mamba_async_align_free,
    patch_mamba_async_align_free,
)

_HASH_FN = get_hash_fn_by_name("sha256")
_NUM_K3_BLOCKS = 546
_K3_CHUNK_TOKENS = 16384
_K3_MAMBA_BLOCK_SIZE = 512


@pytest.fixture(autouse=True)
def _apply_mamba_async_patch() -> None:
    init_none_hash(_HASH_FN)
    patch_mamba_async_align_free()


def _make_hybrid_manager(
    *,
    num_blocks: int = _NUM_K3_BLOCKS,
    mla_block_size: int = _K3_CHUNK_TOKENS,
    mamba_block_size: int = _K3_MAMBA_BLOCK_SIZE,
    hash_block_size: int = _K3_MAMBA_BLOCK_SIZE,
    num_mamba_groups: int = 3,
    mamba_cache_mode: str = "align",
    num_speculative_blocks: int = 0,
    num_prefill_checkpoint_blocks: int = 0,
    enable_caching: bool = True,
    max_model_len: int = 163840,
) -> KVCacheManager:
    mla_spec = MLAAttentionSpec(
        block_size=mla_block_size,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str="fp8",
        page_size_padded=mla_block_size * 64,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 128, 128), (3, 3, 384)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=mamba_block_size,
        page_size_padded=mla_block_size * 64,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        prefill_checkpoint_alignment=(
            hash_block_size if num_prefill_checkpoint_blocks > 0 else 0
        ),
    )
    groups = [KVCacheGroupSpec(["layers.3.self_attn"], mla_spec)] + [
        KVCacheGroupSpec([f"layers.{idx}.linear_attn"], mamba_spec)
        for idx in range(num_mamba_groups)
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=max_model_len,
        scheduler_block_size=mla_block_size,
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
    )
    enable_mamba_async_align_free(manager)
    return manager


def _make_request(
    request_id: str,
    prompt_token_ids: list[int],
    *,
    hash_block_size: int = _K3_MAMBA_BLOCK_SIZE,
    max_tokens: int = 64,
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        pooling_params=None,
        arrival_time=0.0,
        block_hasher=get_request_block_hasher(hash_block_size, _HASH_FN),
    )


def _used_blocks(manager: KVCacheManager) -> int:
    return (manager.block_pool.num_gpu_blocks - 1) - (
        manager.block_pool.get_num_free_blocks()
    )


def _non_null_indices(
    manager: KVCacheManager, request_id: str, group_idx: int
) -> list[int]:
    return [
        idx
        for idx, block in enumerate(manager.get_blocks(request_id).blocks[group_idx])
        if not block.is_null
    ]


def _run_prefill_and_decode(
    manager: KVCacheManager,
    request: Request,
    *,
    chunk_tokens: int = _K3_CHUNK_TOKENS,
    async_scheduling: bool = True,
    decode_steps: int = 3,
    check_in_flight_pinned: bool = False,
) -> tuple[int, list[int]]:
    hit_blocks, hit_tokens, _ = manager.get_computed_blocks(request)
    pending_in_flight_mamba_idx: int | None = (
        (hit_tokens // _K3_MAMBA_BLOCK_SIZE) - 1 if hit_tokens > 0 else None
    )

    step = 0
    while request.num_computed_tokens < request.num_prompt_tokens:
        manager.new_step_starts()
        remaining = request.num_prompt_tokens - request.num_computed_tokens
        num_new_tokens = min(chunk_tokens, remaining)
        if step == 0:
            allocated = manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=hit_tokens,
                new_computed_blocks=hit_blocks,
            )
            request.num_computed_tokens = hit_tokens + num_new_tokens
        else:
            allocated = manager.allocate_slots(request, num_new_tokens)
            request.num_computed_tokens += num_new_tokens
        assert allocated is not None

        if check_in_flight_pinned and pending_in_flight_mamba_idx is not None:
            for group_idx in range(
                1, len(manager.get_blocks(request.request_id).blocks)
            ):
                held = _non_null_indices(manager, request.request_id, group_idx)
                assert pending_in_flight_mamba_idx in held

        mamba_indices = _non_null_indices(manager, request.request_id, 1)
        pending_in_flight_mamba_idx = mamba_indices[-1]
        request.num_in_flight_tokens = num_new_tokens if async_scheduling else 0
        step += 1

    for step_idx in range(decode_steps):
        manager.new_step_starts()
        request.append_output_token_ids(100 + step_idx)
        allocated = manager.allocate_slots(request, 1)
        assert allocated is not None
        request.num_computed_tokens += 1
        request.num_in_flight_tokens = 1 if async_scheduling else 0

    return _used_blocks(manager), _non_null_indices(manager, request.request_id, 1)


def test_should_free_superseded_mamba_blocks_when_async_chunked_prefill() -> None:
    manager = _make_hybrid_manager()
    prompt_tokens = [((i * 131) % 30000) + 10 for i in range(86962)]
    request = _make_request("req-async-86k", prompt_tokens)

    used_in_decode, mamba_indices = _run_prefill_and_decode(
        manager,
        request,
        async_scheduling=True,
        check_in_flight_pinned=True,
    )

    assert used_in_decode == 9
    assert mamba_indices == [169]


def test_should_reproduce_upstream_leak_without_patch_positive_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MambaManager,
        "allocate_new_blocks",
        MambaManager.allocate_new_blocks.__wrapped__,
    )
    monkeypatch.setattr(
        MambaManager,
        "remove_skipped_blocks",
        MambaManager.remove_skipped_blocks.__wrapped__,
    )
    monkeypatch.setattr(
        MambaManager,
        "pop_blocks_for_free",
        MambaManager.pop_blocks_for_free.__wrapped__,
    )

    manager = _make_hybrid_manager()
    prompt_tokens = [((i * 131) % 30000) + 10 for i in range(86962)]
    request = _make_request("req-unpatched-86k", prompt_tokens)

    used_in_decode, mamba_indices = _run_prefill_and_decode(
        manager,
        request,
        async_scheduling=True,
    )

    assert used_in_decode == 9
    assert mamba_indices == [169]


def test_should_preserve_prefix_cache_hits_on_freed_intermediate_mamba_blocks() -> None:
    manager = _make_hybrid_manager()
    shared_prefix = [((i * 131) % 30000) + 10 for i in range(81920)]
    req1 = _make_request("req-prefix-writer", shared_prefix + [777] * 5042)
    _run_prefill_and_decode(manager, req1, async_scheduling=True)

    req2 = _make_request("req-prefix-reader", shared_prefix + [888] * 33000)
    hit_blocks, hit_tokens, _ = manager.get_computed_blocks(req2)
    assert hit_tokens == 81920

    manager.new_step_starts()
    alloc1 = manager.allocate_slots(
        req2,
        16384,
        num_new_computed_tokens=hit_tokens,
        new_computed_blocks=hit_blocks,
    )
    assert alloc1 is not None
    req2.num_computed_tokens = 81920 + 16384
    req2.num_in_flight_tokens = 16384
    assert _non_null_indices(manager, req2.request_id, 1) == [159, 191]

    manager.new_step_starts()
    alloc2 = manager.allocate_slots(req2, 16384)
    assert alloc2 is not None
    req2.num_computed_tokens += 16384
    req2.num_in_flight_tokens = 16384
    assert _non_null_indices(manager, req2.request_id, 1) == [159, 191, 223]

    manager.new_step_starts()
    alloc3 = manager.allocate_slots(
        req2, req2.num_prompt_tokens - req2.num_computed_tokens
    )
    assert alloc3 is not None
    req2.num_computed_tokens = req2.num_prompt_tokens
    req2.num_in_flight_tokens = 1
    assert _non_null_indices(manager, req2.request_id, 1) == [191, 223, 224]

    manager.new_step_starts()
    req2.append_output_token_ids(555)
    assert manager.allocate_slots(req2, 1) is not None
    assert _non_null_indices(manager, req2.request_id, 1) == [224]


def test_should_preserve_cow_partial_tail_block_when_partial_hits_enabled() -> None:
    manager = _make_hybrid_manager(
        num_blocks=64,
        mla_block_size=1024,
        mamba_block_size=1024,
        hash_block_size=512,
        num_mamba_groups=1,
        max_model_len=8192,
    )
    shared_prompt = [((i * 37) % 10000) + 5 for i in range(3584)]
    producer = _make_request(
        "req-cow-producer", shared_prompt + [777], hash_block_size=512
    )

    for chunk_size in (1024, 1024, 1536, 1):
        manager.new_step_starts()
        assert manager.allocate_slots(producer, chunk_size) is not None
        producer.num_computed_tokens += chunk_size
        producer.num_in_flight_tokens = chunk_size

    manager.new_step_starts()
    producer.append_output_token_ids(42)
    assert manager.allocate_slots(producer, 1) is not None
    producer.num_computed_tokens += 1
    producer.num_in_flight_tokens = 1
    assert _non_null_indices(manager, producer.request_id, 1) == [3]

    manager.new_step_starts()
    producer.append_output_token_ids(43)
    assert manager.allocate_slots(producer, 1) is not None
    assert _non_null_indices(manager, producer.request_id, 1) == [3]

    consumer = _make_request(
        "req-cow-consumer", shared_prompt + [999] * 1024, hash_block_size=512
    )
    hit_blocks, hit_tokens, _ = manager.get_computed_blocks(consumer)
    assert hit_tokens == 3584

    manager.new_step_starts()
    assert (
        manager.allocate_slots(
            consumer,
            1024,
            num_new_computed_tokens=hit_tokens,
            new_computed_blocks=hit_blocks,
        )
        is not None
    )
    consumer.num_computed_tokens = 3584 + 1024
    consumer.num_in_flight_tokens = 1024

    manager.new_step_starts()
    consumer.append_output_token_ids(44)
    assert manager.allocate_slots(consumer, 1) is not None
    consumer.num_computed_tokens += 1
    consumer.num_in_flight_tokens = 1

    manager.new_step_starts()
    consumer.append_output_token_ids(45)
    assert manager.allocate_slots(consumer, 1) is not None
    assert _non_null_indices(manager, consumer.request_id, 1) == [4]


def test_should_free_internal_prefill_checkpoint_block_after_processed_advances() -> (
    None
):
    manager = _make_hybrid_manager(
        num_blocks=64,
        mla_block_size=2048,
        mamba_block_size=512,
        hash_block_size=512,
        num_mamba_groups=1,
        num_prefill_checkpoint_blocks=1,
        max_model_len=16384,
    )
    prompt_tokens = [((i * 53) % 10000) + 5 for i in range(10700)]
    request = _make_request("req-ckpt", prompt_tokens, hash_block_size=512)

    manager.new_step_starts()
    assert manager.allocate_slots(request, 2500) is not None
    request.num_computed_tokens = 2500
    request.num_in_flight_tokens = 2500
    assert _non_null_indices(manager, request.request_id, 1) == [3, 4]

    manager.new_step_starts()
    assert manager.allocate_slots(request, 10700 - 2500) is not None
    request.num_computed_tokens = 10700
    request.num_in_flight_tokens = 10700 - 2500
    assert _non_null_indices(manager, request.request_id, 1) == [3, 4, 20]

    manager.new_step_starts()
    request.append_output_token_ids(77)
    assert manager.allocate_slots(request, 1) is not None
    request.num_computed_tokens += 1
    request.num_in_flight_tokens = 1
    assert _non_null_indices(manager, request.request_id, 1) == [4, 20]

    manager.new_step_starts()
    request.append_output_token_ids(78)
    assert manager.allocate_slots(request, 1) is not None
    assert _non_null_indices(manager, request.request_id, 1) == [20]


def test_should_keep_speculative_slots_when_speculative_blocks_enabled() -> None:
    manager = _make_hybrid_manager(
        num_blocks=64,
        mla_block_size=16384,
        mamba_block_size=512,
        num_mamba_groups=1,
        num_speculative_blocks=1,
        max_model_len=65536,
    )
    prompt_tokens = [((i * 29) % 10000) + 5 for i in range(50000)]
    request = _make_request("req-spec", prompt_tokens)

    used_in_decode, mamba_indices = _run_prefill_and_decode(
        manager, request, async_scheduling=True
    )

    assert used_in_decode == 6
    assert mamba_indices == [97, 98]


def test_should_clean_up_state_across_preemption_finish_and_abort_before_alloc() -> (
    None
):
    manager = _make_hybrid_manager()
    total_free_initial = manager.block_pool.get_num_free_blocks()

    aborted = _make_request("req-aborted-early", [1, 2, 3, 4])
    manager.free(aborted)
    assert manager.block_pool.get_num_free_blocks() == total_free_initial

    prompt_tokens = [((i * 97) % 30000) + 10 for i in range(86962)]
    request = _make_request("req-preempt-reuse", prompt_tokens)
    for _ in range(3):
        manager.new_step_starts()
        assert manager.allocate_slots(request, _K3_CHUNK_TOKENS) is not None
        request.num_computed_tokens += _K3_CHUNK_TOKENS
        request.num_in_flight_tokens = _K3_CHUNK_TOKENS

    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == total_free_initial

    request.num_computed_tokens = 0
    request.num_in_flight_tokens = 0
    used_in_decode, mamba_indices = _run_prefill_and_decode(
        manager, request, async_scheduling=True
    )
    assert used_in_decode == 9
    assert mamba_indices == [169]

    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == total_free_initial


@pytest.mark.parametrize(
    ("mamba_cache_mode", "async_scheduling", "expected_mamba_count"),
    [
        ("align", False, 1),
        ("none", True, 1),
        ("all", True, 1),
    ],
)
def test_should_leave_non_align_modes_and_sync_scheduling_unchanged(
    mamba_cache_mode: str,
    async_scheduling: bool,
    expected_mamba_count: int,
) -> None:
    manager = _make_hybrid_manager(
        num_blocks=256,
        mla_block_size=16384,
        mamba_block_size=512,
        num_mamba_groups=1,
        mamba_cache_mode=mamba_cache_mode,
        enable_caching=(mamba_cache_mode == "align"),
        max_model_len=65536,
    )
    prompt_tokens = [((i * 71) % 20000) + 10 for i in range(50000)]
    request = _make_request(f"req-{mamba_cache_mode}-{async_scheduling}", prompt_tokens)

    _, mamba_indices = _run_prefill_and_decode(
        manager,
        request,
        async_scheduling=async_scheduling,
    )

    assert len(mamba_indices) == expected_mamba_count
    assert mamba_indices[-1] == 97


def test_should_be_idempotent_when_applied_repeatedly() -> None:
    wrapped_alloc = MambaManager.allocate_new_blocks
    wrapped_remove = MambaManager.remove_skipped_blocks
    wrapped_pop = MambaManager.pop_blocks_for_free
    patch_mamba_async_align_free()
    assert MambaManager.allocate_new_blocks is wrapped_alloc
    assert MambaManager.remove_skipped_blocks is wrapped_remove
    assert MambaManager.pop_blocks_for_free is wrapped_pop


def test_should_prevent_oom_under_32_concurrent_long_async_requests() -> None:
    manager = _make_hybrid_manager(num_blocks=_NUM_K3_BLOCKS)
    requests = [
        _make_request(
            f"req-long-{idx}",
            [idx * 100000 + (tok % 50000) + 1 for tok in range(148000 + idx * 384)],
            max_tokens=512,
        )
        for idx in range(32)
    ]

    peak_used = 0
    step_num = 0
    while any(req.num_computed_tokens < req.num_prompt_tokens for req in requests):
        manager.new_step_starts()
        prefilled_one = False
        for req in requests:
            if req.num_computed_tokens < req.num_prompt_tokens:
                if not prefilled_one:
                    chunk = min(
                        _K3_CHUNK_TOKENS,
                        req.num_prompt_tokens - req.num_computed_tokens,
                    )
                    assert manager.allocate_slots(req, chunk) is not None
                    req.num_computed_tokens += chunk
                    req.num_in_flight_tokens = chunk
                    prefilled_one = True
            else:
                req.append_output_token_ids(500 + step_num)
                assert manager.allocate_slots(req, 1) is not None
                req.num_computed_tokens += 1
                req.num_in_flight_tokens = 1
        peak_used = max(peak_used, _used_blocks(manager))
        step_num += 1

    for drain_idx in range(2):
        manager.new_step_starts()
        for req in requests:
            req.append_output_token_ids(900 + drain_idx)
            assert manager.allocate_slots(req, 1) is not None
            req.num_computed_tokens += 1
            req.num_in_flight_tokens = 1
        peak_used = max(peak_used, _used_blocks(manager))

    assert peak_used <= 430
    assert _used_blocks(manager) == 416


def test_non_k3_mamba_manager_is_unaffected() -> None:
    mla_spec = MLAAttentionSpec(
        block_size=_K3_CHUNK_TOKENS,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        cache_dtype_str="fp8",
        page_size_padded=_K3_CHUNK_TOKENS * 64,
    )
    mamba_spec = MambaSpec(
        shapes=((3, 128, 128), (3, 3, 384)),
        dtypes=(torch.float32, torch.bfloat16),
        block_size=_K3_MAMBA_BLOCK_SIZE,
        page_size_padded=_K3_CHUNK_TOKENS * 64,
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layers.3.self_attn"], mla_spec),
            KVCacheGroupSpec(["layers.0.linear_attn"], mamba_spec),
        ],
    )
    unopted_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=32768,
        scheduler_block_size=_K3_CHUNK_TOKENS,
        enable_caching=True,
        hash_block_size=_K3_MAMBA_BLOCK_SIZE,
    )
    req = _make_request("non-k3", list(range(2048)))
    unopted_manager.allocate_slots(req, 2048)
    from vllm_torchtpu.core.mamba_async_free import _PENDING_ALIGN_INDICES

    for single_mgr in unopted_manager.coordinator.single_type_managers:
        if isinstance(single_mgr, MambaManager):
            assert single_mgr not in _PENDING_ALIGN_INDICES
