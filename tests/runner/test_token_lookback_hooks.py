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
"""Spec tests for TokenLookbackModel mixin and TokenLookbackRing staging."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output, init_cpu_runner
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
)

from vllm_torchtpu import utils
from vllm_torchtpu.runner.token_lookback import (
    LOOKBACK_RING_DEPTH,
    TokenLookbackRing,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.vllm_v1.worker.gpu_model_runner import GPUModelRunner


class _DummyLookbackModel(torch.nn.Module, utils.TokenLookbackModel):
    def __init__(self, *, depth: int = 3) -> None:
        super().__init__()
        self.token_lookback_depth = depth
        self.last_kwargs: dict[str, object] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, inputs_embeds
        self.last_kwargs = kwargs
        return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)


class _DummyPlainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: dict[str, object] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del positions, inputs_embeds
        self.last_kwargs = kwargs
        return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)


def _swa_kv_config() -> KVCacheConfig:
    swa_spec = SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=32,
        bounded_replay=True,
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn.0"], kv_cache_spec=swa_spec)
        ],
    )


def _cpu_ring(rows: int, depth: int) -> TokenLookbackRing:
    buf_np = np.zeros((rows, depth), dtype=np.int32)
    return TokenLookbackRing(
        (
            SimpleNamespace(
                np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
            ),
        )
    )


def test_should_expose_token_lookback_mixin_and_ring_depth() -> None:
    assert utils.TokenLookbackModel.token_lookback_depth == 0
    assert LOOKBACK_RING_DEPTH == 4
    with pytest.raises(ValueError, match="at least one staging buffer"):
        TokenLookbackRing(())


def test_should_gather_lookback_on_cold_and_warm_replay_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = init_cpu_runner(monkeypatch)
    runner.kv_cache_config = _swa_kv_config()
    runner._token_lookback = _cpu_ring(4, 3)  # noqa: SLF001 -- attach CPU staging ring
    runner.input_batch = SimpleNamespace(
        _tpu_spec_room_guard=True,
        req_id_to_index={"req0": 0, "req1": 1, "req_cold": 2},
        num_reqs=3,
        req_ids=["req0", "req1", "req_cold"],
        num_computed_tokens_cpu=np.array([5, 6, 2, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 4, 4, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 7, 2, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [0, 42, 99, 77, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    runner.model = _DummyLookbackModel(depth=3)
    runner._bounded_replay.replay_starts = {}  # noqa: SLF001 -- cold admission without SWA replay
    zero_out = runner._token_lookback.prepare(  # noqa: SLF001 -- zero-request padding check
        runner.input_batch,
        0,
        async_scheduling=False,
        device_type="cpu",
        replay_tracker=runner._bounded_replay,  # noqa: SLF001 -- pass replay tracker
    )
    assert (zero_out == -1).all()

    runner.forward_model(
        torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)
    )
    out_prompt_only = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(out_prompt_only, torch.Tensor)
    assert out_prompt_only[0].tolist() == [-1, 13, 12]
    assert out_prompt_only[1].tolist() == [-1, -1, 23]
    # Cold partial lookback preserves valid token ID 0 while padding OOB slots with -1.
    assert out_prompt_only[2].tolist() == [42, 0, -1]
    assert out_prompt_only[3].tolist() == [-1, -1, -1]

    # Warm SWA bounded replay starting at token 6 for req1 exposes history up to
    # num_tokens_no_spec while req0 stays prompt-bounded.
    warm_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="req1",
                replay_start=6,
                num_computed_tokens=6,
                sampling_params=None,
                prompt_token_ids=[],
            )
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(), num_computed_tokens=[]
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(warm_sched)  # noqa: SLF001 -- record warm replay start via _update_states
    runner.input_batch.num_reqs = 2
    runner.forward_model(
        torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    )
    out_replayed = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(out_replayed, torch.Tensor)
    assert out_replayed[0].tolist() == [-1, 13, 12]
    assert out_replayed[1].tolist() == [25, 24, 23]


def test_should_reflect_speculative_rejection_rollback_in_subsequent_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = init_cpu_runner(monkeypatch, max_num_reqs=2, max_num_tokens=16)
    runner.kv_cache_config = _swa_kv_config()
    runner._token_lookback = _cpu_ring(2, 3)  # noqa: SLF001 -- attach CPU staging ring
    runner.model = _DummyLookbackModel(depth=3)
    runner.input_batch = SimpleNamespace(
        _tpu_spec_room_guard=True,
        req_id_to_index={"req0": 0},
        num_reqs=1,
        req_ids=["req0"],
        num_computed_tokens_cpu=np.array([7, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [100, 101, 102, 103, 200, 301, 302, 303],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    resume_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids={"req0"},
            num_computed_tokens=[7],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(resume_sched)  # noqa: SLF001 -- record resume start=7 via real _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    on_resume = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(on_resume, torch.Tensor)
    assert on_resume[0].tolist() == [302, 301, 200]

    # Without a new preemption/resume transition, replay_starts stays at 7; a step
    # whose num_computed_tokens rolled back to 6 (or advanced to 8) masks generated
    # tokens >= num_prompt_tokens (4) to -1 rather than reading past the prompt.
    runner.input_batch.token_ids_cpu[0, 5:7] = [301, 401]
    runner.input_batch.num_computed_tokens_cpu[0] = 6
    runner.input_batch.num_tokens_no_spec[0] = 7
    continue_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids=set(),
            num_computed_tokens=[6],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(continue_sched)  # noqa: SLF001 -- advance non-resumed state via real _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    non_resume_rollback = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(non_resume_rollback, torch.Tensor)
    assert non_resume_rollback[0].tolist() == [-1, -1, 103]

    runner.input_batch.num_computed_tokens_cpu[0] = 8
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    later_decode = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(later_decode, torch.Tensor)
    assert later_decode[0].tolist() == [-1, -1, -1]

    # When preempted and re-resumed at rolled-back computed count 6, _update_states
    # refreshes replay start to 6 so lookback exposes bonus history [301, 200, 103].
    runner.input_batch.num_computed_tokens_cpu[0] = 6
    reresume_sched = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"req0"},
        kv_cache_block_copies=None,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req0"],
            resumed_req_ids={"req0"},
            num_computed_tokens=[6],
        ),
    )
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        runner._update_states(reresume_sched)  # noqa: SLF001 -- re-resume at rolled-back computed=6 via _update_states
    runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    after_reresume = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(after_reresume, torch.Tensor)
    assert after_reresume[0].tolist() == [301, 200, 103]


def test_should_scale_lookback_staging_independent_of_max_model_len(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_runner = init_cpu_runner(monkeypatch, max_num_reqs=4, max_model_len=64)
    large_runner = init_cpu_runner(monkeypatch, max_num_reqs=4, max_model_len=131072)
    outputs = []
    for runner in (small_runner, large_runner):
        ring = _cpu_ring(4, 3)
        batch = SimpleNamespace(
            req_ids=["r0", "r1"],
            num_computed_tokens_cpu=np.array([4, 3, 0, 0], dtype=np.int32),
            num_prompt_tokens=np.array([6, 6, 0, 0], dtype=np.int32),
            num_tokens_no_spec=np.array([6, 6, 0, 0], dtype=np.int32),
            token_ids_cpu=np.array(
                [
                    [10, 20, 30, 40, 50, 60],
                    [70, 80, 90, 100, 110, 120],
                ],
                dtype=np.int32,
            ),
        )
        outputs.append(
            ring.prepare(
                batch,
                2,
                async_scheduling=False,
                device_type="cpu",
                replay_tracker=runner._bounded_replay,  # noqa: SLF001 -- zero replay state
            )
        )

    assert outputs[0].shape == outputs[1].shape == (4, 3)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0][:2].tolist() == [[40, 30, 20], [90, 80, 70]]


def test_should_fence_four_slot_lookback_ring_across_wraparound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        req_ids=["req0"],
        num_computed_tokens_cpu=np.array([3, 0, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([8, 0, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 0, 0, 0], dtype=np.int32),
        token_ids_cpu=np.zeros((4, 8), dtype=np.int32),
    )

    cpu_arrays = [np.zeros((4, 2), dtype=np.int32) for _ in range(LOOKBACK_RING_DEPTH)]
    cpu_ring = TokenLookbackRing(
        tuple(
            SimpleNamespace(
                np=arr, copy_to_gpu=lambda a=arr: torch.from_numpy(a.copy())
            )
            for arr in cpu_arrays
        )
    )
    batch.token_ids_cpu[0, :3] = [11, 12, 13]
    out_cpu = cpu_ring.prepare(batch, 1, async_scheduling=True, device_type="cpu")
    assert out_cpu[0].tolist() == [13, 12]
    assert cpu_ring._fences == [None] * LOOKBACK_RING_DEPTH  # noqa: SLF001 -- no TPU events on CPU

    ring_arrays = [np.zeros((4, 2), dtype=np.int32) for _ in range(LOOKBACK_RING_DEPTH)]
    events_log: list[tuple[str, int, list[int]]] = []

    class _FakeTpuEvent:
        slot_cursor = 0

        def __init__(self) -> None:
            self.slot = _FakeTpuEvent.slot_cursor % LOOKBACK_RING_DEPTH
            _FakeTpuEvent.slot_cursor += 1
            self.recorded_row0: list[int] = []

        def record(self) -> None:
            self.recorded_row0 = ring_arrays[self.slot][0].tolist()
            events_log.append(("record", self.slot, list(self.recorded_row0)))

        def synchronize(self) -> None:
            current_row0 = ring_arrays[self.slot][0].tolist()
            assert current_row0 == self.recorded_row0
            events_log.append(("sync", self.slot, current_row0))

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(Event=_FakeTpuEvent), raising=False
    )
    tpu_ring = TokenLookbackRing(
        tuple(
            SimpleNamespace(
                np=arr,
                copy_to_gpu=lambda slot=i: SimpleNamespace(
                    device=SimpleNamespace(type="tpu"), slot=slot
                ),
            )
            for i, arr in enumerate(ring_arrays)
        )
    )

    for step in range(6):
        base = (step + 1) * 100
        batch.token_ids_cpu[0, :3] = [base + 1, base + 2, base + 3]
        out = tpu_ring.prepare(batch, 1, async_scheduling=True, device_type="tpu")
        slot = step % LOOKBACK_RING_DEPTH
        assert out.slot == slot
        assert ring_arrays[slot][0].tolist() == [base + 3, base + 2]
        if step >= 1:
            prev_slot = (step - 1) % LOOKBACK_RING_DEPTH
            prev_base = step * 100
            assert ring_arrays[prev_slot][0].tolist() == [prev_base + 3, prev_base + 2]

    assert [entry[:2] for entry in events_log] == [
        ("record", 0),
        ("record", 1),
        ("record", 2),
        ("record", 3),
        ("sync", 0),
        ("record", 0),
        ("sync", 1),
        ("record", 1),
    ]


@pytest.mark.parametrize(
    ("is_lookback", "depth"),
    [
        (True, 3),
        (True, 0),
        (False, 0),
    ],
)
def test_should_configure_lookback_ring_on_model_load(
    is_lookback: bool,
    depth: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = init_cpu_runner(monkeypatch, max_num_reqs=16, max_num_tokens=64)
    runner._pp_is_last = False  # noqa: SLF001 -- disable greedy top-k setup
    runner.input_batch = SimpleNamespace(num_reqs=2, block_table=[])
    runner._make_buffer = lambda rows, cols, *, dtype: SimpleNamespace(  # noqa: SLF001 -- stub device staging buffer
        shape=(rows, cols), dtype=dtype
    )
    runner._initialize_pallas_kernels = lambda: None  # noqa: SLF001 -- stub kernel compile
    model = _DummyLookbackModel(depth=depth) if is_lookback else _DummyPlainModel()
    orig_device = torch.device
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.torch.device",
        lambda dev: orig_device("cpu") if dev == "tpu" else orig_device(dev),
    )
    for flag in (
        "TPU_ROPE_CACHE_TRUNCATE",
        "TPU_ROPE_CACHE_ROW_MAJOR",
        "TPU_MOE_HASH_TABLE_ROW_MAJOR",
        "FUSED_MOE_DECODE_KERNEL",
        "TPU_MOE_SKIP_PADDED_TOKENS",
        "USE_MOE_TP_DECODE_KERNEL",
    ):
        monkeypatch.setattr(f"vllm_torchtpu.runner.tpu_runner.envs.{flag}", False)
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.get_model_loader",
        lambda _cfg: SimpleNamespace(load_model=lambda **_kw: model),
    )

    runner.load_model()
    assert (
        runner._token_lookback is not None  # noqa: SLF001 -- verify 4-slot ring creation
        and len(runner._token_lookback._buffers) == LOOKBACK_RING_DEPTH  # noqa: SLF001 -- verify 4-slot ring length
    ) is (depth > 0)
    if runner._token_lookback is not None:  # noqa: SLF001 -- isolate forward_model hook
        runner._token_lookback.prepare_for_chunk = MagicMock(  # noqa: SLF001 -- isolate forward_model hook
            return_value=torch.full((16, depth), 7, dtype=torch.int32)
        )
    out, aux = runner.forward_model(
        torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    )
    assert out.shape == (2, 4) and aux is None
    assert ("lookback_token_ids" in model.last_kwargs) is (depth > 0)


def test_should_slice_chunked_lookback_via_chunk_span_in_forward_model() -> None:
    runner = _make_runner(
        num_computed_tokens=[2, 3, 6],
        prompt_tokens=[4, 4, 4],
        scheduled_tokens=[1, 1, 1],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=4,
    )
    runner.kv_cache_config = _swa_kv_config()
    runner._token_lookback = _cpu_ring(4, 3)  # noqa: SLF001 -- sync staging mode
    # Asymmetric num_tokens_no_spec across rows [2, 3, 7] discriminates batch_row=2
    # from chunk-local row=1 when preparing chunk start_index=1 with replay on req2.
    runner.input_batch.num_tokens_no_spec = np.array([2, 3, 7], dtype=np.int32)
    runner._bounded_replay.replay_starts = {"req0": 55, "req2": 77}  # noqa: SLF001 -- distinct starts on rows 0 and 2

    TPUModelRunner._prepare_inputs(runner, _scheduler_output([1, 1, 1]), 1, 3, 3)  # noqa: SLF001 -- prepare second chunk at start_index=1
    assert runner._lookback_chunk_span == (1, 2)  # noqa: SLF001 -- verify non-zero start_index span
    assert runner._attn_metadata_builder_ctx.host_schedule.replay_starts == (  # noqa: SLF001 -- verify start_index offset in host_schedule
        0,
        77,
        0,
        0,
    )

    # Activate replay match on chunk row 1 (batch_row=2, "req2") at computed=6.
    runner._bounded_replay.replay_starts = {"req0": 55, "req2": 6}  # noqa: SLF001 -- match req2 computed=6 on chunk row 1
    runner.model = _DummyLookbackModel(depth=3)
    TPUModelRunner.forward_model(
        runner,
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    lb = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(lb, torch.Tensor)
    assert lb.shape == (4, 3)
    assert lb[0].tolist() == [202, 201, 200]
    assert lb[1].tolist() == [305, 304, 303]
    assert lb[2].tolist() == [-1, -1, -1]
    assert lb[3].tolist() == [-1, -1, -1]

    # When _lookback_chunk_span is None, forward_model slices full [0..num_reqs).
    runner._lookback_chunk_span = None  # noqa: SLF001 -- verify None span fallback
    TPUModelRunner.forward_model(
        runner,
        torch.zeros(3, dtype=torch.int32),
        torch.zeros(3, dtype=torch.int32),
    )
    lb_full = runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(lb_full, torch.Tensor)
    assert lb_full[0].tolist() == [101, 100, -1]
    assert lb_full[1].tolist() == [202, 201, 200]
    assert lb_full[2].tolist() == [305, 304, 303]


@pytest.mark.tpu_test
def test_should_stage_fenced_lookback_ring_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_tpu  # noqa: F401, PLC0415 -- TPU-only runtime extension imported inside tpu_test

    dev = torch.device("tpu:0")
    runner = init_cpu_runner(monkeypatch)
    ring_arrays = [np.zeros((4, 3), dtype=np.int32) for _ in range(LOOKBACK_RING_DEPTH)]
    ring = TokenLookbackRing(
        tuple(
            SimpleNamespace(
                np=arr,
                copy_to_gpu=lambda a=arr: torch.from_numpy(a).to(
                    dev, non_blocking=True
                ),
            )
            for arr in ring_arrays
        )
    )
    runner._bounded_replay.replay_starts = {"req1": 6}  # noqa: SLF001 -- seed warm replay start
    batch = SimpleNamespace(
        req_ids=["req0", "req1"],
        num_computed_tokens_cpu=np.array([5, 6, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([4, 4, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([6, 7, 0, 0], dtype=np.int32),
        token_ids_cpu=np.array(
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [20, 21, 22, 23, 24, 25, 26, 27],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )
    outputs: list[torch.Tensor] = []
    for step in range(6):
        batch.token_ids_cpu[1, 5] = 25 + step
        outputs.append(
            ring.prepare(
                batch,
                2,
                async_scheduling=True,
                device_type="tpu",
                replay_tracker=runner._bounded_replay,  # noqa: SLF001 -- pass warm replay tracker
            )
        )
    assert all(f is not None for f in ring._fences)  # noqa: SLF001 -- verify all 4 slots fenced
    for step, lb in enumerate(outputs):
        assert lb.device.type == "tpu"
        assert lb[0].cpu().tolist() == [-1, 13, 12]
        assert lb[1].cpu().tolist() == [25 + step, 24, 23]
