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

import pytest
import torch
from runner_input_test_utils import _make_runner, _scheduler_output
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu import _patch_circular_buffer_block_zeroing, patch_registry
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _spec_warmup_group_num_blocks,
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


@pytest.mark.parametrize("use_max_model_len", [True, False])
@pytest.mark.parametrize("mamba_ckpt_window", [1, 2])
@pytest.mark.parametrize("is_circular", [True, False])
def test_attention_metadata_clamps_fixed_capacity_block_tables(
    use_max_model_len: bool,
    mamba_ckpt_window: int,
    is_circular: bool,
) -> None:
    """Verify _block_table_geometry clamps fixed-capacity specs to 1 block/req."""
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
    spec = (
        CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=64,
            dtype=torch.bfloat16,
            page_size_padded=2048,
        )
        if is_circular
        else MambaSpec(
            block_size=8,
            shapes=[(8, 64)],
            dtypes=[torch.bfloat16],
            page_size_padded=2048,
        )
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
        use_max_model_len=use_max_model_len,
        mamba_ckpt_window=mamba_ckpt_window,
        seq_lens=torch.tensor([24, 16, 0, 0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 2], dtype=torch.int32),
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert meta.block_tables.tolist() == [3, 5, 0, 0]


def test_spec_warmup_group_num_blocks_clamps_circular_buffer() -> None:
    """Verify _spec_warmup_group_num_blocks clamps only CircularBufferSpec."""
    vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8, cp_kv_cache_interleave_size=1
        )
    )
    full_group = KVCacheGroupSpec(
        layer_names=["attn.0"],
        kv_cache_spec=FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
        ),
    )
    circ_group = KVCacheGroupSpec(
        layer_names=["state.0"],
        kv_cache_spec=CircularBufferSpec(
            block_size=8, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
        ),
    )
    executed: list[object] = []
    runner = SimpleNamespace(
        max_model_len=128,
        max_num_reqs=4,
        max_num_tokens=64,
        vllm_config=vllm_cfg,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        kv_cache_config=SimpleNamespace(
            num_blocks=64, kv_cache_groups=[full_group, circ_group]
        ),
        execute_model=lambda so: executed.append(so) and None,
        sample_tokens=lambda _: None,
        take_draft_token_ids=lambda: None,
        _warmup_spec_decode_cleanup=lambda _rids: True,
    )
    assert _spec_warmup_group_num_blocks(runner, full_group, 40) == 3
    assert _spec_warmup_group_num_blocks(runner, full_group, 512) == 8
    assert _spec_warmup_group_num_blocks(runner, circ_group, 40) == 1
    assert TPUModelRunner._warmup_one_mtp_prefill(runner, 32) is True
    TPUModelRunner._warmup_spec_batch(runner, 1, 16, mixed=True, short=True)
    assert len(executed) >= 2
    for so in executed:
        for req in so.scheduled_new_reqs:
            assert len(req.block_ids[1]) == 1


def test_circular_buffer_block_zeroing_and_native_owner_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CircularBufferManager zeroing patch and _zero_block_ids dispatch."""
    reg_targets = {
        p.target for p in patch_registry.PATCHES if "engine_core" in p.stages
    }
    assert "vllm_torchtpu:_patch_circular_buffer_block_zeroing" in reg_targets

    monkeypatch.setattr(
        CircularBufferManager, "__init__", SingleTypeKVCacheManager.__init__
    )
    _patch_circular_buffer_block_zeroing()
    # Second invocation is an idempotent no-op.
    patched_init = CircularBufferManager.__init__
    _patch_circular_buffer_block_zeroing()
    assert CircularBufferManager.__init__ is patched_init

    comp_f32 = CircularBufferSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1024,
        dtype=torch.float32,
        page_size_padded=65536,
    )
    pool = BlockPool(num_gpu_blocks=8, enable_caching=False, hash_block_size=16)
    mgr_no_zero = CircularBufferManager(
        kv_cache_spec=comp_f32,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=16,
        needs_kv_cache_zeroing=False,
    )
    assert mgr_no_zero._record_new_block_ids is False

    mla_fp8 = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=512, dtype=torch.uint8
    )
    swa_bf16 = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=32,
    )
    idx_bf16 = MLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    kv_cfg = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[
            KVCacheTensor(
                size=16 * 65536,
                layers=["mla.0", "swa.0", "idx.0", "state.0"],
                layer_stride=65536,
                block_stride=65536,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["mla.0"], kv_cache_spec=mla_fp8),
            KVCacheGroupSpec(layer_names=["swa.0"], kv_cache_spec=swa_bf16),
            KVCacheGroupSpec(layer_names=["idx.0"], kv_cache_spec=idx_bf16),
            KVCacheGroupSpec(layer_names=["state.0"], kv_cache_spec=comp_f32),
        ],
    )
    coord = get_kv_cache_coordinator(
        kv_cache_config=kv_cfg,
        max_model_len=128,
        max_in_flight_tokens=128,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=16,
        hash_block_size=16,
        metrics_collector=None,
    )
    assert kv_cfg.needs_kv_cache_zeroing is True
    coord.allocate_new_blocks("req-0", 20, 20)
    per_mgr_new = [m.take_new_block_ids() for m in coord.single_type_managers]
    assert len(per_mgr_new) == 4
    assert all(len(ids) > 0 for ids in per_mgr_new)

    raw_runner = TPUModelRunner.__new__(TPUModelRunner)
    assert raw_runner._native_kv_cache_owner is None

    runner = _init_cpu_runner(monkeypatch)
    assert runner._native_kv_cache_owner is None
    super_zeroed: list[list[int]] = []
    with patch.object(
        GPUModelRunner,
        "_zero_block_ids",
        side_effect=lambda ids: super_zeroed.append(list(ids)),
    ):
        raw_runner._zero_block_ids([9])
        runner._zero_block_ids([])
        runner._zero_block_ids([2, 5])
    assert super_zeroed == [[9], [], [2, 5]]

    owner_calls: list[list[int]] = []
    runner._native_kv_cache_owner = SimpleNamespace(
        zero_blocks=lambda ids: owner_calls.append(list(ids)),
    )
    with patch.object(
        GPUModelRunner,
        "_zero_block_ids",
        side_effect=AssertionError("unexpected GPUModelRunner._zero_block_ids"),
    ):
        runner._zero_block_ids([])
        runner._zero_block_ids([3, 7])
    assert owner_calls == [[3, 7]]

    events: list[tuple[str, object]] = []
    runner._native_kv_cache_owner = SimpleNamespace(
        zero_blocks=lambda ids: events.append(("zero", list(ids))),
    )
    runner.kv_cache_raw_tensors = [torch.zeros((8, 128), dtype=torch.int32)]
    sched_out = SchedulerOutput.make_empty()
    sched_out.new_block_ids_to_zero = [4, 6]

    def _base_update(self_runner: GPUModelRunner, so: SchedulerOutput) -> bool:
        assert so.kv_cache_block_copies is None
        if so.new_block_ids_to_zero:
            self_runner._zero_block_ids(so.new_block_ids_to_zero)
        return False

    monkeypatch.setattr(GPUModelRunner, "_update_states", _base_update)
    runner._update_states(sched_out)
    assert events == [("zero", [4, 6])]

    sched_out.kv_cache_block_copies = [KVCacheBlockCopy(src_block_id=1, dst_block_id=4)]
    with pytest.raises(
        NotImplementedError,
        match="CoW block copies are unsupported on custom NativeKVCacheLayer pools",
    ):
        runner._update_states(sched_out)


@pytest.mark.parametrize("pcp_size", [1, 2])
def test_prepare_inputs_zeros_trailing_positions_under_all_layout(
    pcp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _prepare_inputs zeros padded tail positions in ALL layout."""
    if pcp_size > 1:
        pcp_mod = "vllm_torchtpu.layers.core.pcp_sequence_layout"
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_rank", lambda: 0)
        monkeypatch.setattr(f"{pcp_mod}._get_native_pcp_world_size", lambda: pcp_size)
    runner = _make_runner(
        num_computed_tokens=[4, 6],
        prompt_tokens=[8, 4],
        scheduled_tokens=[2, 2],
        token_paddings=[8],
        pcp_size=pcp_size,
        max_num_reqs=4,
    )
    runner.positions_cpu.fill_(999)
    TPUModelRunner._prepare_inputs(runner, _scheduler_output([2, 2]), 0, 0, 0)
    if pcp_size == 1:
        assert runner.position_ids[:4].tolist() == [4, 5, 6, 7]
        assert runner.position_ids[4:8].tolist() == [0, 0, 0, 0]
    else:
        assert runner.position_ids.shape == (8,)


def test_dummy_run_dp_ep_forward_context_and_logits_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify DPMetadata.make succeeds in warmup and idle DP logits are clamped."""
    runner = _make_runner(
        num_computed_tokens=[4, 0],
        prompt_tokens=[8, 6],
        scheduled_tokens=[2, 6],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=48,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.synchronize_tensors", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        "vllm.forward_context.current_platform.set_additional_forward_context",
        lambda **_kw: {},
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    runner._record_input_staging_fence = lambda: None
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )

    seen_non_dp: list[object] = []

    def _non_dp_forward(*, positions: torch.Tensor, **_kw: object):
        seen_non_dp.append(get_forward_context().dp_metadata)
        return torch.zeros((positions.shape[-1], 4), dtype=torch.float32), None

    runner.forward_model = _non_dp_forward
    TPUModelRunner._dummy_run(
        runner, 16, 48, 2, use_max_model_len=True, dp_lockstep=False
    )
    assert seen_non_dp == [None]

    runner.parallel_config = SimpleNamespace(
        data_parallel_size=8,
        data_parallel_size_local=8,
        data_parallel_rank=7,
        enable_expert_parallel=True,
        use_sequence_parallel_moe=False,
        is_moe_model=True,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        tensor_parallel_size=1,
    )
    runner.vllm_config.parallel_config = runner.parallel_config
    runner._dp_lockstep_enabled = MethodType(
        TPUModelRunner._dp_lockstep_enabled, runner
    )
    runner._dp_num_tokens_across_dp = MethodType(
        TPUModelRunner._dp_num_tokens_across_dp, runner
    )
    seen_dp_tokens: list[torch.Tensor] = []
    seen_logits_idx: list[torch.Tensor] = []
    runner.compute_selected_logits = lambda out, idx: (
        seen_logits_idx.append(idx),
        out,
    )[1]

    def _dp_ep_forward(*, positions: torch.Tensor, **_kw: object):
        dp_meta = get_forward_context().dp_metadata
        assert dp_meta is not None
        seen_dp_tokens.append(dp_meta.num_tokens_across_dp_cpu)
        return torch.zeros((positions.shape[-1], 4), dtype=torch.float32), None

    runner.forward_model = _dp_ep_forward
    TPUModelRunner._dummy_run(
        runner, 16, 48, 2, use_max_model_len=True, dp_lockstep=False
    )
    TPUModelRunner._dummy_run(
        runner, 16, 48, 2, use_max_model_len=True, dp_lockstep=True
    )
    assert len(seen_dp_tokens) == 2
    for t in seen_dp_tokens:
        assert torch.equal(t, torch.full((8,), 16, dtype=torch.int32))
    assert len(seen_logits_idx) == 1
    assert seen_logits_idx[0].shape == (16,)


@pytest.mark.tpu_test
def test_dp_ep_circular_runner_tpu_metadata_and_dummy_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # pragma: no cover
    """Verify CircularBufferSpec block table clamp and idle DP logits on TPU7x."""
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
    )
    meta = builder.build(0, SimpleNamespace(num_reqs=4))
    assert meta.block_tables.device.type == "tpu"
    assert meta.block_tables.cpu().tolist() == [3, 5, 0, 0]

    runner = _make_runner(
        num_computed_tokens=[4, 0],
        prompt_tokens=[8, 6],
        scheduled_tokens=[2, 6],
        token_paddings=[16],
        pcp_size=1,
        max_num_reqs=48,
    )
    runner.device = dev
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=8,
        data_parallel_size_local=8,
        data_parallel_rank=0,
        enable_expert_parallel=True,
        use_sequence_parallel_moe=False,
        is_moe_model=True,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        tensor_parallel_size=1,
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            fast_moe_cold_start=False,
            static_forward_context={},
        ),
    )
    runner._dp_lockstep_enabled = MethodType(
        TPUModelRunner._dp_lockstep_enabled, runner
    )
    runner._dp_num_tokens_across_dp = MethodType(
        TPUModelRunner._dp_num_tokens_across_dp, runner
    )
    monkeypatch.setattr(
        "vllm_torchtpu.runner.tpu_runner.set_vllm_model_wrapper_context",
        lambda *_a, **_kw: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.forward_context.current_platform.set_additional_forward_context",
        lambda **_kw: {},
    )
    runner.maybe_select_dummy_loras = lambda *_a, **_kw: contextlib.nullcontext()
    seen_tpu_idx: list[torch.Tensor] = []
    runner.compute_selected_logits = lambda out, idx: (
        seen_tpu_idx.append(idx),
        out,
    )[1]
    runner.forward_model = lambda *, positions, **_kw: (
        torch.zeros((positions.shape[-1], 4), dtype=torch.float32, device=dev),
        None,
    )
    TPUModelRunner._dummy_run(
        runner, 16, 48, 2, use_max_model_len=True, dp_lockstep=True
    )
    assert len(seen_tpu_idx) == 1
    assert seen_tpu_idx[0].device.type == "tpu"
    assert seen_tpu_idx[0].shape == (16,)
