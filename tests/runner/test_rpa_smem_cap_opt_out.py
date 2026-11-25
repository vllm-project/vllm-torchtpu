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
"""Tests for RpaSmemPageTableModel mixin and SMEM request cap opt-out."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import utils
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.vllm_v1.worker.gpu_model_runner import GPUModelRunner


class _DummySmemModel(torch.nn.Module, utils.RpaSmemPageTableModel):
    def __init__(self, *, uses_smem: bool) -> None:
        super().__init__()
        self.uses_rpa_smem_page_table = uses_smem


class _DummyPlainModel(torch.nn.Module):
    pass


def _init_cpu_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_num_reqs: int = 16,
    max_num_tokens: int = 64,
    max_model_len: int = 64,
) -> TPUModelRunner:
    pcfg = SimpleNamespace(
        data_parallel_size=1,
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
        seed=0,
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
        reasoning_config=None,
        load_config=SimpleNamespace(),
        profiler_config=SimpleNamespace(torch_profiler_dir=""),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            cudagraph_mode=None,
            compile_sizes=[min(16, max_num_tokens), max_num_tokens],
            static_forward_context={},
        ),
        model_config=mcfg,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        additional_config={},
    )

    def _stub_gpu_init(self: GPUModelRunner, cfg: SimpleNamespace, dev: object) -> None:
        self.vllm_config = cfg
        self.model_config = cfg.model_config
        self.cache_config = cfg.cache_config
        self.parallel_config = cfg.parallel_config
        self.scheduler_config = cfg.scheduler_config
        self.speculative_config = cfg.speculative_config
        self.load_config = cfg.load_config
        self.lora_config = None
        self.device = dev
        self.dtype = torch.bfloat16
        self.max_model_len = max_model_len
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.block_size = 16
        self.uses_mrope = False
        self.supports_mm_inputs = False
        self.sample_from_logits = None
        self.requests = {}
        self.mesh = None
        self.drafter = None
        self.input_batch = SimpleNamespace(
            block_table=[], _tpu_spec_room_guard=True, req_id_to_index={}
        )

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


def test_rpa_smem_page_table_mixin_defaults_to_true() -> None:
    assert utils.RpaSmemPageTableModel.uses_rpa_smem_page_table is True


@pytest.mark.parametrize(
    ("is_mixin", "uses_smem", "most_len", "max_num_tokens", "expected_caps"),
    [
        (True, False, 64, 64, (16, 16)),
        (True, False, 64, 10, (10, 10)),
        (True, True, 64, 64, (4, 6)),
        (False, True, None, 64, (4, None)),
    ],
)
def test_smem_request_cap_opt_out_persists_across_runner_lifecycle(
    is_mixin: bool,
    uses_smem: bool,
    most_len: int | None,
    max_num_tokens: int,
    expected_caps: tuple[int, int | None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _init_cpu_runner(
        monkeypatch, max_num_reqs=16, max_num_tokens=max_num_tokens
    )
    assert runner._uses_rpa_smem_page_table is True  # noqa: SLF001 -- verify initial default
    runner.most_model_len = most_len
    runner.max_num_blocks_per_req = 8
    runner._pp_is_last = False  # noqa: SLF001 -- disable greedy top-k setup
    runner._attention_capacity = {"cached": (1, 1, 1)}  # noqa: SLF001 -- verify cache clear on refresh
    runner._decode_device_cache_key = ("cached",)  # noqa: SLF001 -- verify cache clear on refresh
    runner.input_batch = SimpleNamespace(
        num_reqs=2,
        block_table=[SimpleNamespace(max_num_blocks_per_req=8)] if is_mixin else [],
    )
    runner.block_table_cpu = torch.zeros((16, 8), dtype=torch.int32)
    runner._initialize_pallas_kernels = lambda: None  # noqa: SLF001 -- stub kernel compile
    runner.kv_cache_manager = SimpleNamespace(get_kv_cache_spec=lambda: {})
    model = _DummySmemModel(uses_smem=uses_smem) if is_mixin else _DummyPlainModel()

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
    assert runner.get_kv_cache_spec() == {}
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps

    # Bare _refresh_smem_request_caps() must preserve self._uses_rpa_smem_page_table.
    runner._refresh_smem_request_caps()  # noqa: SLF001 -- verify default fallback to runner flag
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps

    runner.may_reinitialize_input_batch(None, [16])  # type: ignore[arg-type] -- None exercises pre-init fallback
    assert (
        runner.num_reqs_max_model_len,
        runner.num_reqs_most_model_len,
    ) == expected_caps

    # Explicit keyword override takes precedence over the runner flag.
    runner._refresh_smem_request_caps(uses_rpa_smem_page_table=True)  # noqa: SLF001 -- verify explicit override
    assert runner.num_reqs_max_model_len == 4
