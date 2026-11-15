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
"""CPU unit tests for model-agnostic `megakernel.adapter.decode_owner` and scheduler."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu.megakernel.adapter import decode_owner
from vllm_torchtpu.megakernel.adapter.decode_owner import DecodeOwner, DecodeResult
from vllm_torchtpu.megakernel.core.runtime.contract import PAR_TP, TpuMonoSpec


class _FakeOwner:
    def __init__(self, owned: set[str], *, drafts: bool = False) -> None:
        self._owned = set(owned)
        self.drafts = drafts

    def load(self, model: Any, draft_model: Any | None) -> None:
        pass

    def bind_runner(self, runner: Any) -> None:
        pass

    def release(self, req_ids: Any) -> None:
        for r in req_ids:
            self._owned.discard(r)

    def owns(self, req_id: str) -> bool:
        return req_id in self._owned

    def begin_step(self, num_chunks: int) -> None:
        pass

    def chunk_tokens(
        self,
        runner: Any,
        scheduler_output: Any,
        start_index: int,
        num_reqs: int,
        num_tokens_padded: int,
    ) -> None:
        pass

    def flush_chunk(self) -> None:
        pass

    def flush_draft_chunk(self, chunk_index: int) -> None:
        pass

    def take_ownership(
        self,
        runner: Any,
        scheduler_output: Any,
        sampled: Any,
        drafts: Any,
    ) -> None:
        pass

    def run_decode(
        self, runner: Any, indices: Any, scheduler_output: Any
    ) -> dict[str, DecodeResult]:
        return {}


def _fake_create(view: Any, options: Any) -> Any:
    return _FakeOwner(set())


FAKE_SPEC = TpuMonoSpec(
    name="fake_mk",
    version="1.0",
    create=f"{__name__}:_fake_create",
    chips=("TPU7x",),
    tp_sizes=(8,),
    parallelism=frozenset({PAR_TP}),
    top_level_options=frozenset({"max_req"}),
)


class _FakeBatch:
    def __init__(self, req_ids: list[str]) -> None:
        self.req_ids = list(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        self.num_computed_tokens_cpu = np.arange(len(req_ids), dtype=np.int32) * 10
        self.token_ids_cpu = np.zeros((len(req_ids), 64), dtype=np.int32)
        self.num_tokens = [10 * i for i in range(len(req_ids))]
        self.num_tokens_no_spec = [10 * i for i in range(len(req_ids))]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    def swap_states(self, i1: int, i2: int) -> None:
        self.req_ids[i1], self.req_ids[i2] = self.req_ids[i2], self.req_ids[i1]
        self.req_id_to_index = {r: i for i, r in enumerate(self.req_ids)}


def test_move_owned_to_front_reorders_and_preserves_counts() -> None:
    owner = _FakeOwner({"r1", "r3"})
    assert isinstance(owner, DecodeOwner)
    batch = _FakeBatch(["r0", "r1", "r2", "r3", "p0"])
    num_owned, num_decode, num_windowed = decode_owner.move_owned_to_front(
        batch, owner, num_decode=2, num_windowed=4
    )
    assert (num_owned, num_decode, num_windowed) == (2, 3, 4)
    assert batch.req_ids == ["r1", "r3", "r0", "r2", "p0"]

    # Owned request in the prefill tail raises.
    bad_batch = _FakeBatch(["r0", "r1"])
    with pytest.raises(RuntimeError, match="scheduled as a prefill"):
        decode_owner.move_owned_to_front(
            bad_batch, _FakeOwner({"r1"}), num_decode=1, num_windowed=1
        )


def test_owned_step_output_and_prepend_owned() -> None:
    batch = _FakeBatch(["r_owned", "r_vllm"])
    batch.max_num_logprobs = 2
    batch.num_computed_tokens_cpu[:] = [5, 10]
    batch.num_tokens = [5, 11]
    batch.num_tokens_no_spec = [5, 11]
    draft_sink: list[list[int]] = []
    runner = SimpleNamespace(
        input_batch=batch,
        requests={
            "r_owned": SimpleNamespace(output_token_ids=[]),
            "r_vllm": SimpleNamespace(output_token_ids=[]),
        },
        spec_decode_manager=SimpleNamespace(
            peek_draft_token_ids=lambda: [[90, 91, 92]],
            set_draft_token_ids=lambda d: draft_sink.extend(d),
        ),
    )
    results = {
        "r_owned": DecodeResult(
            tokens=[42, 43],
            top_ids=np.array([[42, 7, 8], [43, 9, 10]], dtype=np.int32),
            top_logprobs=np.array(
                [[-0.1, -1.2, -2.3], [-0.2, -1.5, -3.0]], dtype=np.float32
            ),
            drafts=[101, 102, 103],
        )
    }
    owner = _FakeOwner({"r_owned"}, drafts=True)
    owner.run_decode = lambda _r, _idx, _so: results  # type: ignore[method-assign]

    final_output = SimpleNamespace(
        sampled_token_ids=[[77]],
        logprobs=None,
    )
    decode_owner.prepend_owned(
        runner, owner, final_output, num_owned=1, scheduler_output=SimpleNamespace()
    )
    assert final_output.sampled_token_ids == [[42, 43], [77]]
    assert final_output.logprobs is not None
    assert final_output.logprobs.logprob_token_ids[0].tolist() == [42, 42, 7]
    assert draft_sink == [[101, 102, 103], [90, 91, 92]]


def test_tpu_runner_decode_owner_class_annotation_no_shadowing() -> None:
    """`TPUModelRunner` class body must not shadow the `decode_owner` module."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    assert TPUModelRunner.decode_owner is None


def test_suspend_decode_owner_and_spec_chunk_never_skips_logits() -> None:
    """`_suspend_decode_owner` hides `decode_owner` during warmup and restores it;
    `chunk_discards_all_samples` is False whenever spec-decode tokens exist."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    fake_runner: Any = SimpleNamespace(decode_owner=_FakeOwner({"r0"}))
    with TPUModelRunner._suspend_decode_owner(fake_runner):
        assert fake_runner.decode_owner is None
    assert fake_runner.decode_owner is not None

    batch = _FakeBatch(["r0"])
    batch.logits_processing_needs_token_ids = np.zeros(4, dtype=bool)
    runner = SimpleNamespace(
        input_batch=batch,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        requests={"r0": SimpleNamespace(num_computed_tokens=4, num_tokens=16)},
    )
    so_partial = SimpleNamespace(
        num_scheduled_tokens={"r0": 4},
        scheduled_spec_decode_tokens={},
    )
    assert decode_owner.chunk_discards_all_samples(runner, so_partial, 0) is True

    so_spec = SimpleNamespace(
        num_scheduled_tokens={"r0": 4},
        scheduled_spec_decode_tokens={"r0": [1, 2, 3]},
    )
    assert decode_owner.chunk_discards_all_samples(runner, so_spec, 0) is False


def test_decode_owner_scheduler_caps_cache_blocks_and_clips_prefill(
    monkeypatch: Any,
) -> None:
    """`DecodeOwnerScheduler` caps `cache_blocks` at `num_prompt_tokens`,
    enforces non-preemptive admission, and aligns prefill chunks to `block_size`."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel.adapter.decode_owner_scheduler import (
        SCHEDULER_CLS,
        DecodeOwnerScheduler,
        _cap_instance_cache_blocks,
    )

    seen_kvm: list[int] = []
    seen_coord: list[int] = []
    coord = SimpleNamespace(cache_blocks=lambda _r, n: seen_coord.append(n))
    kvm = SimpleNamespace(
        cache_blocks=lambda _r, n: seen_kvm.append(n),
        coordinator=coord,
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 20),
    )
    _cap_instance_cache_blocks(kvm)
    req = SimpleNamespace(num_prompt_tokens=144, max_tokens=288, num_computed_tokens=0)
    kvm.cache_blocks(req, 400)
    coord.cache_blocks(req, 400)
    assert seen_kvm == [144]
    assert seen_coord == [144]

    sched: Any = object.__new__(DecodeOwnerScheduler)
    sched.cache_config = SimpleNamespace(block_size=144)
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=131072)
    )
    sched.scheduler_config = SimpleNamespace(max_num_batched_tokens=4096)
    sched.kv_cache_manager = kvm
    sched._orig_need_mamba_split = False
    sched._prefill_used = 0
    sched.running = [
        SimpleNamespace(num_prompt_tokens=144, max_tokens=1440, num_computed_tokens=144)
    ]
    # Running needs 14 blocks growth; candidate needs 14 blocks -> 28 > 20 free.
    cand = SimpleNamespace(num_prompt_tokens=720, max_tokens=720)
    assert sched._admit_fits(cand) is False
    kvm.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 40)
    assert sched._admit_fits(cand) is True

    # Exercise `DecodeOwnerScheduler.schedule()` for both blocked and unblocked queues.
    from vllm.v1.core.sched.scheduler import Scheduler

    seen_active_caps: list[int] = []
    monkeypatch.setattr(
        Scheduler,
        "schedule",
        lambda self, throttle_prefills=False: seen_active_caps.append(
            self.max_num_active_reqs
        ),
    )

    class _FakeQueue(list[Any]):
        def peek_request(self) -> Any:
            return self[0]

    sched.kv_holding_waiting = _FakeQueue()
    sched.waiting = _FakeQueue([cand])
    sched.num_waiting_for_streaming_input = 0
    sched.max_num_active_reqs = 16
    kvm.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 20)
    sched.schedule()
    kvm.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 40)
    sched.schedule()
    assert seen_active_caps == [1, 2]
    assert sched.max_num_active_reqs == 16

    # Prefill chunk clipping aligns to `block_size=144` once budget is partially used.
    long_req = SimpleNamespace(
        num_prompt_tokens=8192, num_tokens=8192, num_computed_tokens=0
    )
    sched._prefill_used = 1000
    clipped = DecodeOwnerScheduler._mamba_block_aligned_split(sched, long_req, 4096)
    assert clipped == ((4096 - 1000) // 144) * 144

    monkeypatch.setattr(
        envs, "VLLM_TPU_MEGAKERNEL", f"plugin={__name__}:FAKE_SPEC,max_req=16"
    )
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["FakeForCausalLM"], model_type="fake"
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=False),
        additional_config={},
    )
    decode_owner.configure_scheduler(vcfg)
    assert vcfg.scheduler_config.scheduler_cls == SCHEDULER_CLS

    # `async_scheduling=True` must decline in both `configure_scheduler` and `create`.
    vcfg_async = SimpleNamespace(
        model_config=vcfg.model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=True),
        additional_config={},
    )
    decode_owner.configure_scheduler(vcfg_async)
    assert vcfg_async.scheduler_config.scheduler_cls is None
    assert (
        decode_owner.create(
            SimpleNamespace(vllm_config=vcfg_async, model_config=vcfg.model_config)
        )
        is None
    )

    # End-to-end with vLLM's real KVCacheManager + BlockPool: when cached prompt
    # blocks from r1 are evicted and recycled into r2's decode steps, BlockPool
    # evicts r1's old hashes upon allocation and `_cap_instance_cache_blocks`
    # prevents r2's decode blocks from being hashed. Thus r3 (matching r2's
    # prompt + decode) only hits r2's prompt block and never hits recycled decode
    # block IDs.
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    from vllm.v1.request import Request

    bs = 16
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(bs, hash_fn)
    real_kvm = KVCacheManager(
        kv_cache_config=KVCacheConfig(
            num_blocks=6,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["attn"],
                    FullAttentionSpec(
                        block_size=bs,
                        num_kv_heads=1,
                        head_size=16,
                        dtype=torch.bfloat16,
                    ),
                )
            ],
        ),
        max_model_len=1024,
        scheduler_block_size=bs,
        hash_block_size=bs,
        enable_caching=True,
        use_eagle=False,
        log_stats=False,
    )
    _cap_instance_cache_blocks(real_kvm)

    def _mk_req(rid: str, toks: list[int]) -> Request:
        return Request(
            request_id=rid,
            prompt_token_ids=toks,
            sampling_params=SamplingParams(max_tokens=64),
            pooling_params=None,
            arrival_time=0.0,
            block_hasher=hasher,
        )

    r1 = _mk_req("r1", list(range(100, 164)))
    real_kvm.allocate_slots(r1, num_new_tokens=64)
    r1.num_computed_tokens = 64
    r1_blks = [
        b.block_id
        for b in real_kvm.coordinator.single_type_managers[0].req_to_blocks["r1"]
    ]
    real_kvm.free(r1)

    r2_prompt = list(range(200, 216))
    r2 = _mk_req("r2", r2_prompt)
    real_kvm.allocate_slots(r2, num_new_tokens=16)
    r2.num_computed_tokens = 16
    for s in range(48):
        r2.append_output_token_ids(300 + s)
        real_kvm.allocate_slots(r2, num_new_tokens=1)
        r2.num_computed_tokens += 1
    r2_blks = [
        b.block_id
        for b in real_kvm.coordinator.single_type_managers[0].req_to_blocks["r2"]
    ]
    assert set(r2_blks[1:]) & set(r1_blks)

    r3 = _mk_req("r3", r2_prompt + [300 + s for s in range(48)])
    r3_hits, r3_n = real_kvm.get_computed_blocks(r3)[:2]
    assert r3_n == 16
    assert [b.block_id for b in r3_hits.blocks[0]] == [r2_blks[0]]


def test_eagle3_propose_flushes_draft_chunks() -> None:
    """`Eagle3Proposer.propose` flushes MTP draft chunks before loop steps."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    events: list[str] = []
    owner = _FakeOwner(set(), drafts=True)
    owner.flush_draft_chunk = lambda ci: events.append(f"flush:{ci}")  # type: ignore[method-assign]

    proposer: Any = object.__new__(Eagle3Proposer)
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=1),
        decode_owner=owner,
        uses_mrope=False,
        _dp_step_num_chunks=1,
    )
    proposer._draft_md_cache = {}
    proposer.speculative_config = SimpleNamespace(
        num_speculative_tokens=3, method="mtp"
    )
    proposer._draft_uses_aux_hidden_state = lambda: False
    proposer._share_mtp_indices = False
    proposer._dp_lockstep_sharded = lambda: False
    proposer.draft_chunks = [
        SimpleNamespace(
            num_reqs=1,
            hidden_states=torch.zeros((4, 8)),
            sequence_layout_plan=SimpleNamespace(
                aggregate_request_aligned_tensor=lambda t, _m: t
            ),
        )
    ]
    proposer._prepare_draft_inputs = lambda *_a, **_kw: (
        torch.zeros((4,), dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([3], dtype=torch.int64),
        np.zeros((1,), dtype=np.int32),
    )

    def fake_forward_draft(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        events.append(f"step:{kwargs['step_idx']}")
        n = int(kwargs["num_tokens_padded"])
        return torch.zeros((n, 8)), torch.zeros((n, 8))

    proposer._forward_draft = fake_forward_draft
    proposer._draft_gather_carries = lambda h, p, lh, idx: (h[idx], p[idx], lh[idx])
    proposer._draft_propose_token = lambda lh: torch.zeros(
        (lh.shape[0],), dtype=torch.int32
    )
    proposer._loop_bucket = lambda n: n

    # 1. Partial-prefill discard chunk: step 0 + flush_draft_chunk(0), skips steps 1..2.
    out_discard = Eagle3Proposer.propose(
        proposer,
        sampled_token_ids=[],
        discard_sampled_tokens_req_indices=[0],
        num_rejected_tokens_np=None,
        scheduler_output=cast_any(SimpleNamespace()),
        return_device=False,
    )
    assert out_discard == [[]]
    assert events == ["step:0", "flush:0"]


def cast_any(x: Any) -> Any:
    return x
