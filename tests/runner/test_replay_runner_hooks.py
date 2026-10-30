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

import contextlib
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.config import CacheConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu import (
    _patch_dflash_sliding_window_retained_tokens,
    patch_registry,
    utils,
)
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.runner.tpu_runner import (
    _LOOKBACK_RING_DEPTH,
    TPUModelRunner,
)


class _DummyLookbackModel(
    torch.nn.Module, utils.TokenLookbackModel, utils.RpaSmemPageTableModel
):
    def __init__(self, *, depth: int = 3, uses_smem: bool = False) -> None:
        super().__init__()
        self.token_lookback_depth = depth
        self.uses_rpa_smem_page_table = uses_smem
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


def _init_cpu_runner(monkeypatch: pytest.MonkeyPatch) -> TPUModelRunner:
    pcfg = SimpleNamespace(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=16,
        is_moe_model=False,
        enable_expert_parallel=False,
    )
    mcfg = SimpleNamespace(
        architecture="LlamaForCausalLM",
        runner_type="generate",
        enforce_eager=True,
        quantization=None,
        get_sliding_window=lambda: None,
        get_num_layers_by_block_type=lambda _pc, _bt: 1,
        get_num_kv_heads=lambda _pc: 1,
        get_head_size=lambda: 64,
        get_vocab_size=lambda: 128,
    )
    vllm_cfg = SimpleNamespace(
        parallel_config=pcfg,
        speculative_config=None,
        kv_transfer_config=None,
        quant_config=None,
        load_config=SimpleNamespace(),
        profiler_config=SimpleNamespace(torch_profiler_dir=""),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            cudagraph_mode=None,
            compile_sizes=[16, 64],
            static_forward_context={},
        ),
        model_config=mcfg,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    def _stub_gpu_init(self: GPUModelRunner, cfg: SimpleNamespace, dev: object) -> None:
        self.vllm_config = cfg
        self.model_config = cfg.model_config
        self.cache_config = cfg.cache_config
        self.parallel_config = cfg.parallel_config
        self.scheduler_config = cfg.scheduler_config
        self.load_config = cfg.load_config
        self.device = dev
        self.dtype = torch.bfloat16
        self.max_model_len = 64
        self.max_num_reqs = 16
        self.max_num_tokens = 64
        self.block_size = 16
        self.uses_mrope = False
        self.supports_mm_inputs = False
        self.sample_from_logits = None
        self.requests = {}
        self.mesh = None
        self.drafter = None
        self.lookback_token_ids = None
        self.input_batch = SimpleNamespace(block_table=[], _tpu_spec_room_guard=True)

    monkeypatch.setattr(GPUModelRunner, "__init__", _stub_gpu_init)
    for attr in (
        "_validate_libtpu_version",
        "RunnerStageTiming",
        "get_tpu_quantization_config",
        "validate_online_fp8",
        "install_kernel_block_tables",
    ):
        monkeypatch.setattr(f"vllm_torchtpu.runner.tpu_runner.{attr}", lambda *_a: None)
    for cm in (
        "_torch_tpu_wrapper",
        "set_vllm_model_wrapper_context",
        "set_current_vllm_config",
    ):
        monkeypatch.setattr(
            f"vllm_torchtpu.runner.tpu_runner.{cm}",
            lambda *_a, **_kw: contextlib.nullcontext(),
        )
    for method in ("_update_num_xla_graphs", "_create_mesh_for_parallelism"):
        monkeypatch.setattr(TPUModelRunner, method, lambda *_a: None)
    return TPUModelRunner(
        vllm_cfg, torch.device("cpu"), profiler_rank=0, profiler_world_size=1
    )


def test_utils_model_mixins_and_attention_metadata_cpu_drop_fields() -> None:
    """Verify RpaSmemPageTableModel/TokenLookbackModel defaults and drop_fields."""
    assert not issubclass(utils.TokenLookbackModel, utils.RpaSmemPageTableModel)
    assert utils.RpaSmemPageTableModel.uses_rpa_smem_page_table is True
    assert utils.TokenLookbackModel.token_lookback_depth == 0
    assert _LOOKBACK_RING_DEPTH == 4

    bt = SimpleNamespace(
        max_num_blocks_per_req=1,
        blocks_per_kv_block=1,
        get_cpu_tensor=lambda: torch.tensor([[3], [5], [7], [9]], dtype=torch.int32),
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        block_size=8,
        max_num_reqs=4,
        most_model_len=32,
        position_ids=torch.arange(4, dtype=torch.int32),
        _block_table_stage_cpu=None,
        _free_h2d_sources={},
        _registered_h2d_sources=set(),
        _unfenced_h2d_sources=[],
        attn_groups=[],
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt]),
    )
    spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        page_size_padded=2048,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["state.0"],
        vllm_config=MagicMock(),
        device=runner.device,
        runner=runner,
        kv_cache_group_id=0,
    )
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
        query_start_loc_cpu=(0, 2, 4, 4, 4),
        seq_lens_cpu=(24, 16, 0, 0),
        prompt_lens_cpu=(20, 16, 0, 0),
        replay_starts_cpu=(8, 0, 0, 0),
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert (
        meta.query_start_loc_cpu,
        meta.seq_lens_cpu,
        meta.prompt_lens_cpu,
        meta.replay_starts_cpu,
    ) == ((0, 2, 4, 4, 4), (24, 16, 0, 0), (20, 16, 0, 0), (8, 0, 0, 0))
    leaves, treedef = jax.tree_util.tree_flatten(meta)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert (
        restored.query_start_loc_cpu,
        restored.seq_lens_cpu,
        restored.prompt_lens_cpu,
        restored.replay_starts_cpu,
    ) == (None, None, None, None)


def test_replay_starts_tracking_and_sync_lookback_gathering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _record_replay_starts, _update_states eviction, and lookback gather."""
    runner = _init_cpu_runner(monkeypatch)
    assert object.__new__(TPUModelRunner)._prefix_replay_tokens() == 0
    assert runner._replay_starts == {}
    assert runner._uses_rpa_smem_page_table is True
    assert runner._prefix_replay_tokens() == 0

    sched_out = SimpleNamespace(
        finished_req_ids={"finished_req"},
        preempted_req_ids=None,
        kv_cache_block_copies=None,
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="new_replayed",
                replay_start=16,
                num_computed_tokens=16,
                sampling_params=None,
            ),
            SimpleNamespace(
                req_id="new_plain",
                replay_start=0,
                num_computed_tokens=0,
                sampling_params=None,
            ),
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["resumed_req", "active_req", "resumed_zero"],
            resumed_req_ids={"resumed_req", "resumed_zero"},
            num_computed_tokens=[24, 32, 0],
        ),
    )
    # Handle _replay_starts=None across _update_states and _record_replay_starts.
    runner._replay_starts = None
    with patch.object(GPUModelRunner, "_update_states", return_value=False):
        assert runner._update_states(sched_out) is False
    assert runner._replay_starts is None

    runner.kv_cache_config = _swa_kv_config()
    assert runner._prefix_replay_tokens() == 32
    runner._record_replay_starts(sched_out)
    assert runner._replay_starts == {"new_replayed": 16, "resumed_req": 24}

    runner._replay_starts = {"finished_req": 16, "resumed_req": 8, "survivor": 12}
    with patch.object(GPUModelRunner, "_update_states", return_value=True):
        assert runner._update_states(sched_out) is True
    assert runner._replay_starts == {
        "survivor": 12,
        "new_replayed": 16,
        "resumed_req": 24,
    }

    buf_np = np.zeros((4, 3), dtype=np.int32)
    runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    runner.input_batch = SimpleNamespace(
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
    # Without active replay starts, tokens at/past prompt_len are masked to -1.
    runner._replay_starts = {}
    assert (runner._prepare_lookback_token_ids(0) == -1).all()
    out_prompt_only = runner._prepare_lookback_token_ids(2)
    assert out_prompt_only[0].tolist() == [-1, 13, 12]
    assert out_prompt_only[1].tolist() == [-1, -1, 23]

    # With SWA bounded replay starting at token 6 for req1, history up to
    # num_tokens_no_spec is exposed for req1 while req0 stays prompt-bounded.
    runner._replay_starts = {"req1": 6}
    out_replayed = runner._prepare_lookback_token_ids(2)
    assert out_replayed[0].tolist() == [-1, 13, 12]
    assert out_replayed[1].tolist() == [25, 24, 23]


def test_lookback_staging_ring_never_rewrites_in_flight_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 4-slot async staging ring preserves in-flight slots and fences reuse."""
    runner = _init_cpu_runner(monkeypatch)
    runner.scheduler_config.async_scheduling = True
    runner._replay_starts = {}
    runner.input_batch = SimpleNamespace(
        req_ids=["req0"],
        num_computed_tokens_cpu=np.array([3, 0, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([8, 0, 0, 0], dtype=np.int32),
        num_tokens_no_spec=np.array([8, 0, 0, 0], dtype=np.int32),
        token_ids_cpu=np.zeros((4, 8), dtype=np.int32),
    )

    # 1. CPU device path: lazy fence list allocation without TPU event creation.
    cpu_arrays = [np.zeros((4, 2), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)]
    runner._lookback_staging_ring = tuple(
        SimpleNamespace(np=arr, copy_to_gpu=lambda a=arr: torch.from_numpy(a.copy()))
        for arr in cpu_arrays
    )
    runner._lookback_staging_fences = None
    runner._lookback_staging_idx = 0
    runner.input_batch.token_ids_cpu[0, :3] = [11, 12, 13]
    out_cpu = runner._prepare_lookback_token_ids(1)
    assert out_cpu[0].tolist() == [13, 12]
    assert runner._lookback_staging_fences == [None] * _LOOKBACK_RING_DEPTH

    # 2. TPU async scheduling path (max_concurrent_batches = 2, ring depth = 4):
    # verify that across 6 consecutive dispatches, in-flight slots from prior
    # steps are never mutated before their per-slot H2D event synchronizes.
    runner.device = SimpleNamespace(type="tpu")
    ring_arrays = [
        np.zeros((4, 2), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)
    ]
    events_log: list[tuple[str, int, list[int]]] = []

    class _FakeTpuEvent:
        slot_cursor = 0

        def __init__(self) -> None:
            self.slot = _FakeTpuEvent.slot_cursor % _LOOKBACK_RING_DEPTH
            _FakeTpuEvent.slot_cursor += 1
            self.recorded_row0: list[int] = []

        def record(self) -> None:
            self.recorded_row0 = ring_arrays[self.slot][0].tolist()
            events_log.append(("record", self.slot, list(self.recorded_row0)))

        def synchronize(self) -> None:
            # At wraparound reuse, the slot's host buffer MUST still hold the
            # exact data from its previous H2D transfer before fill(-1) runs.
            current_row0 = ring_arrays[self.slot][0].tolist()
            assert current_row0 == self.recorded_row0
            events_log.append(("sync", self.slot, current_row0))

    tpu_ns = SimpleNamespace(Event=_FakeTpuEvent)
    monkeypatch.setattr(torch, "tpu", tpu_ns, raising=False)
    runner._lookback_staging_ring = tuple(
        SimpleNamespace(
            np=arr,
            copy_to_gpu=lambda slot=i: SimpleNamespace(
                device=SimpleNamespace(type="tpu"), slot=slot
            ),
        )
        for i, arr in enumerate(ring_arrays)
    )
    runner._lookback_staging_fences = [None] * _LOOKBACK_RING_DEPTH
    runner._lookback_staging_idx = 0

    for step in range(6):
        base = (step + 1) * 100
        runner.input_batch.token_ids_cpu[0, :3] = [base + 1, base + 2, base + 3]
        out = runner._prepare_lookback_token_ids(1)
        slot = step % _LOOKBACK_RING_DEPTH
        assert out.slot == slot
        assert ring_arrays[slot][0].tolist() == [base + 3, base + 2]
        if step >= 1:
            prev_slot = (step - 1) % _LOOKBACK_RING_DEPTH
            prev_base = step * 100
            # Previous in-flight async batch's staging buffer is untouched.
            assert ring_arrays[prev_slot][0].tolist() == [prev_base + 3, prev_base + 2]

    # Dispatches 0..3 only record; dispatches 4 and 5 synchronize slots 0 and 1
    # strictly before rewriting them, then record new fences.
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
    ("is_lookback", "depth", "uses_smem", "most_len", "expected_caps"),
    [
        (True, 3, False, 64, (16, 16)),
        (True, 0, True, 64, (4, 6)),
        (False, 0, True, None, (4, None)),
    ],
)
def test_token_lookback_model_load_smem_caps_and_forward(
    is_lookback: bool,
    depth: int,
    uses_smem: bool,
    most_len: int | None,
    expected_caps: tuple[int, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify TokenLookbackModel hooks in load_model, SMEM caps, and forward_model."""
    runner = _init_cpu_runner(monkeypatch)
    runner.most_model_len = most_len
    runner.max_num_blocks_per_req = 8
    runner._pp_is_last = False
    runner._attention_capacity = {"cached": (1, 1, 1)}
    runner._decode_device_cache_key = ("cached",)
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        block_table=[SimpleNamespace(max_num_blocks_per_req=8)] if is_lookback else [],
    )
    runner.block_table_cpu = torch.zeros((16, 8), dtype=torch.int32)
    runner._make_buffer = lambda rows, cols, *, dtype: SimpleNamespace(
        shape=(rows, cols), dtype=dtype
    )
    runner._prepare_lookback_token_ids = MagicMock(
        return_value=torch.full((16, max(1, depth)), 7, dtype=torch.int32)
    )
    runner._initialize_pallas_kernels = lambda: None
    model = (
        _DummyLookbackModel(depth=depth, uses_smem=uses_smem)
        if is_lookback
        else _DummyPlainModel()
    )
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
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.PallasAttentionBackend.get_max_num_seqs_for_pages",
        lambda pages: 4 if pages == 8 else 6,
    )
    monkeypatch.setattr(
        GPUModelRunner, "may_reinitialize_input_batch", lambda *_a: None
    )

    runner.load_model()
    runner.may_reinitialize_input_batch(None, [16])
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps
    out, aux = runner.forward_model(
        torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
    )
    assert out.shape == (2, 4) and aux is None
    assert (runner.lookback_token_ids is not None) is (depth > 0)
    assert (
        runner._lookback_staging_ring is not None
        and len(runner._lookback_staging_ring) == _LOOKBACK_RING_DEPTH
    ) is (depth > 0)
    assert ("lookback_token_ids" in model.last_kwargs) is (depth > 0)

    if most_len is not None:
        runner._attention_kernel_block_size = 8
        runner._attention_kv_cache_group_id = 0
        runner.most_model_len = 32
        runner.input_batch = SimpleNamespace(
            block_table=[
                SimpleNamespace(max_num_blocks_per_req=4, blocks_per_kv_block=1)
            ]
        )
        runner.kv_cache_config = None
        assert runner._attention_kernel_shapes(False) == (expected_caps[1], 4, 8)
        runner.kv_cache_config = _swa_kv_config()
        assert runner._attention_kernel_shapes(False) == (expected_caps[1], 2, 8)
        runner._attention_kv_cache_group_id = 1
        assert runner._attention_kernel_shapes(False) == (expected_caps[1], 4, 8)


def test_chunked_forward_model_uses_lookback_chunk_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify forward_model slices chunked lookback via _lookback_chunk_span."""
    cpu_runner = _init_cpu_runner(monkeypatch)
    buf_np = np.zeros((4, 3), dtype=np.int32)
    cpu_runner.lookback_token_ids = SimpleNamespace(
        np=buf_np, copy_to_gpu=lambda: torch.from_numpy(buf_np.copy())
    )
    cpu_runner._replay_starts = {"req1": 6}
    cpu_runner.input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req0", "req1"],
        num_computed_tokens_cpu=np.array([5, 6, 0, 0], dtype=np.int32),
        num_prompt_tokens=np.array([6, 4, 0, 0], dtype=np.int32),
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
    cpu_runner.model = _DummyLookbackModel(depth=3, uses_smem=True)
    cpu_runner._lookback_chunk_span = (1, 1)
    cpu_runner.forward_model(
        torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
    )
    lb = cpu_runner.model.last_kwargs["lookback_token_ids"]
    assert isinstance(lb, torch.Tensor)
    assert lb[0].tolist() == [25, 24, 23]
    assert lb[1].tolist() == [-1, -1, -1]


@pytest.mark.parametrize("pcp_size", [1, 2])
@pytest.mark.parametrize("dp_lockstep", [False, True])
@pytest.mark.parametrize("profiling", [False, True])
def test_prepare_inputs_and_dummy_run_cpu_scheduling_snapshots(
    pcp_size: int,
    dp_lockstep: bool,
    profiling: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _prepare_inputs and _dummy_run populate CPU scheduling tuples."""
    if pcp_size > 1:
        pcp_mod = "vllm_torchtpu.layers.core.pcp_sequence_layout"
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_rank", lambda: 0)
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_world_size", lambda: pcp_size)
    runner = _make_runner(
        num_computed_tokens=[4, 0],
        prompt_tokens=[8, 6],
        scheduled_tokens=[2, 6],
        token_paddings=[16],
        pcp_size=pcp_size,
        max_num_reqs=4,
    )
    runner._replay_starts = {"req0": 4}
    if dp_lockstep:
        runner.kv_cache_config = _swa_kv_config()
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([2, 6]), 0, 0, 0)
    assert runner._lookback_chunk_span == (0, 2)
    ctx = runner._attn_metadata_builder_ctx
    expected_prep = (
        (
            (0, 2, 8, 8, 8),
            (6, 6, 0, 0),
            (8, 6, 0, 0),
            (4, 0, 0, 0) if dp_lockstep else (0, 0, 0, 0),
        )
        if pcp_size == 1
        else (None, None, None, None)
    )
    assert (
        ctx.query_start_loc_cpu,
        ctx.seq_lens_cpu,
        ctx.prompt_lens_cpu,
        ctx.replay_starts_cpu,
    ) == expected_prep

    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_forward_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", lambda *_a, **_kw: None
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    runner._record_input_staging_fence = lambda: None
    runner._dp_lockstep_enabled = lambda: dp_lockstep
    runner._dp_num_tokens_across_dp = lambda _n: None
    runner.compute_selected_logits = lambda out, _idx: out
    runner._profile_inputs = {}
    runner.vocab_size = 128
    runner._pp_topk_buffer = None
    runner._profile_input = MethodType(TPUModelRunner._profile_input, runner)
    runner.forward_model = lambda *, positions, **_kw: (
        torch.zeros((positions.shape[-1], 4), dtype=torch.float32),
        None,
    )
    TPUModelRunner._dummy_run(
        runner,
        8,
        4,
        2,
        use_max_model_len=True,
        dp_lockstep=dp_lockstep,
        profile_prefix=0 if profiling else None,
    )
    assert runner._lookback_chunk_span == (0, 4)
    dummy_ctx = runner._attn_metadata_builder_ctx
    expected = (
        ((0, 0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        if dp_lockstep and not profiling
        else (None, None, None, None)
    )
    assert (
        dummy_ctx.query_start_loc_cpu,
        dummy_ctx.seq_lens_cpu,
        dummy_ctx.prompt_lens_cpu,
        dummy_ctx.replay_starts_cpu,
    ) == expected


def test_sliding_window_retained_tokens_patch_for_block_drafters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify get_kv_cache_groups retains trailing SWA blocks across configs."""
    reg_targets = {
        p.target for p in patch_registry.PATCHES if "engine_core" in p.stages
    }
    assert "vllm_torchtpu:_patch_dflash_sliding_window_retained_tokens" in reg_targets

    orig_groups = getattr(
        kv_cache_utils.get_kv_cache_groups,
        "__wrapped__",
        kv_cache_utils.get_kv_cache_groups,
    )
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", orig_groups)
    _patch_dflash_sliding_window_retained_tokens()
    patched_groups = kv_cache_utils.get_kv_cache_groups
    # Second call is an idempotent no-op.
    _patch_dflash_sliding_window_retained_tokens()
    assert kv_cache_utils.get_kv_cache_groups is patched_groups

    base_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
    )
    high_swa = SlidingWindowSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=128,
        extra_retained_tokens=10,
    )
    full_spec = FullAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    cache_cfg = CacheConfig(enable_prefix_caching=True)
    cache_cfg.kv_cache_layout = "LBHNC"
    base_vcfg = dict(
        cache_config=cache_cfg,
        model_config=SimpleNamespace(
            max_model_len=2048,
            original_max_model_len=2048,
            hf_config=SimpleNamespace(),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False, max_num_seqs=4
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=16,
    )

    # 1. No speculative decoding -> extra_retained_tokens remains 0.
    groups_plain = kv_cache_utils.get_kv_cache_groups(
        SimpleNamespace(**base_vcfg), {"attn.0": base_swa}
    )
    assert groups_plain[0].kv_cache_spec.extra_retained_tokens == 0

    # 2. Non-block-drafter method or zero lookahead -> remains 0.
    vcfg_zero_la = SimpleNamespace(
        **{
            **base_vcfg,
            "speculative_config": SimpleNamespace(
                method="dflash",
                use_eagle=lambda: False,
                use_eagle_block_drop=lambda: False,
            ),
            "num_lookahead_tokens": 0,
        }
    )
    groups_zero_la = kv_cache_utils.get_kv_cache_groups(
        vcfg_zero_la, {"attn.0": base_swa}
    )
    assert groups_zero_la[0].kv_cache_spec.extra_retained_tokens == 0

    # 3. Direct get_kv_cache_groups with mixed SlidingWindowSpec + FullAttentionSpec:
    # clamps 0 -> 6, preserves already-higher 10, and leaves FullAttentionSpec intact.
    vcfg_dspark = SimpleNamespace(
        **{
            **base_vcfg,
            "speculative_config": SimpleNamespace(
                method="dspark",
                use_eagle=lambda: False,
                use_eagle_block_drop=lambda: False,
            ),
            "num_lookahead_tokens": 5,
        }
    )
    groups_mixed = kv_cache_utils.get_kv_cache_groups(
        vcfg_dspark,
        {"swa.0": base_swa, "swa.1": high_swa, "full.0": full_spec},
    )
    by_layer = {g.layer_names[0]: g.kv_cache_spec for g in groups_mixed}
    assert by_layer["swa.0"].extra_retained_tokens == 6
    assert by_layer["swa.1"].extra_retained_tokens == 10

    # 4. End-to-end through get_kv_cache_configs (which resets extra_retained_tokens
    # to 0 right before calling get_kv_cache_groups): still produces 6!
    configs = kv_cache_utils.get_kv_cache_configs(
        vcfg_dspark, [{"attn.0": base_swa}], [64 * 1024 * 1024]
    )
    retained_spec = configs[0].kv_cache_groups[0].kv_cache_spec
    assert retained_spec.extra_retained_tokens == 6

    mgr_stub = SimpleNamespace(
        sliding_window=retained_spec.sliding_window,
        extra_retained_tokens=retained_spec.extra_retained_tokens,
    )
    skipped = SlidingWindowManager.get_num_skipped_tokens(mgr_stub, 895)
    assert skipped == 762 and skipped // 32 == 23


@pytest.mark.tpu_test
def test_replay_runner_hooks_tpu_lookback_ring_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Verify AttentionMetadata and 6-step fenced lookback ring wraparound on TPU7x."""
    import torch_tpu  # noqa: F401

    dev = torch.device("tpu:0")
    bt = SimpleNamespace(
        max_num_blocks_per_req=1,
        blocks_per_kv_block=1,
        get_cpu_tensor=lambda: torch.tensor([[3], [5], [7], [9]], dtype=torch.int32),
    )
    meta_runner = SimpleNamespace(
        device=dev,
        block_size=8,
        max_num_reqs=4,
        most_model_len=32,
        position_ids=torch.arange(4, dtype=torch.int32, device=dev),
        _block_table_stage_cpu=None,
        _free_h2d_sources={},
        _registered_h2d_sources=set(),
        _unfenced_h2d_sources=[],
        attn_groups=[],
        _unified_kv_layout=False,
        input_batch=SimpleNamespace(block_table=[bt]),
    )
    spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        page_size_padded=2048,
    )
    builder = AttentionMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["state.0"],
        vllm_config=MagicMock(),
        device=dev,
        runner=meta_runner,
        kv_cache_group_id=0,
    )
    meta_runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=2,
        start_index=0,
        use_max_model_len=False,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32, device=dev),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=dev),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32, device=dev),
        query_start_loc_cpu=(0, 2, 4, 4, 4),
        seq_lens_cpu=(24, 16, 0, 0),
        prompt_lens_cpu=(20, 16, 0, 0),
        replay_starts_cpu=(8, 0, 0, 0),
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert meta.block_tables.device.type == "tpu"
    assert meta.block_tables.cpu().tolist() == [3, 5, 0, 0]

    runner = _init_cpu_runner(monkeypatch)
    runner.device = dev
    runner.scheduler_config.async_scheduling = True
    ring_arrays = [
        np.zeros((4, 3), dtype=np.int32) for _ in range(_LOOKBACK_RING_DEPTH)
    ]
    runner._lookback_staging_ring = tuple(
        SimpleNamespace(
            np=arr,
            copy_to_gpu=lambda a=arr: torch.from_numpy(a).to(dev, non_blocking=True),
        )
        for arr in ring_arrays
    )
    runner._lookback_staging_fences = [None] * _LOOKBACK_RING_DEPTH
    runner._lookback_staging_idx = 0
    runner._replay_starts = {"req1": 6}
    runner.input_batch = SimpleNamespace(
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
        runner.input_batch.token_ids_cpu[1, 5] = 25 + step
        outputs.append(runner._prepare_lookback_token_ids(2))
    assert all(f is not None for f in runner._lookback_staging_fences)
    for step, lb in enumerate(outputs):
        assert lb.device.type == "tpu"
        assert lb[0].cpu().tolist() == [-1, 13, 12]
        assert lb[1].cpu().tolist() == [25 + step, 24, 23]
