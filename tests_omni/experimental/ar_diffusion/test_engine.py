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
"""TPU test for ARDiffusionEngine for World Models."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVConfig,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

_NUM_LAYERS = 1
_NUM_HEADS = 2
_HEAD_DIM = 16
_TOKENS_PER_FRAME = 16
_DENOISE_STEPS = 2
_BF16_RATIO = 4.0
_BF16_FLOOR = 1e-2


class _TinyARDiffusionPipeline:
    """Minimal SupportsARDiffusionPipeline calling paged_write_attn across steps."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.bound_state: ARDiffusionKVState | None = None
        self._spec = ARDiffusionKVCacheSpec(
            num_layers=_NUM_LAYERS,
            num_kv_heads=_NUM_HEADS,
            head_size=_HEAD_DIM,
            tokens_per_frame=_TOKENS_PER_FRAME,
            frames_per_block=1,
            window_frames=4,
            sink_frames=0,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            session_capacity=2,
        )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        return self._spec

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState):
        self.bound_state = state
        try:
            yield
        finally:
            self.bound_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        del session_id

    def close_ar_diffusion_session(self, session_id: str) -> None:
        del session_id

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        assert self.bound_state is not None, "AR-Diffusion state is not bound"
        extra = req.sampling_params.extra_args or {}
        q_cpu, k_cpu, v_cpu = extra["qkv"]
        q = q_cpu.to(device=self.device, dtype=self.dtype)
        k = k_cpu.to(device=self.device, dtype=self.dtype)
        v = v_cpu.to(device=self.device, dtype=self.dtype)

        scale = _HEAD_DIM**-0.5
        out = q
        for step_idx in range(_DENOISE_STEPS):
            commit = step_idx == _DENOISE_STEPS - 1
            ctxs = self.bound_state.get_kv_caches(
                "main",
                seq_len=_TOKENS_PER_FRAME,
                commit_current=commit,
            )
            ctxs[0].forward_ctx.prepare(
                device=self.device, action_len=0, query_len=_TOKENS_PER_FRAME
            )
            inputs = ctxs[0].to_layer_inputs()
            out = paged_write_attn(inputs, q + out * 0.1, k, v, None, None, scale)
        self.bound_state.commit_paged_context("main")
        return DiffusionOutput(output=out)


def _make_runner(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[ARDiffusionModelRunner, _TinyARDiffusionPipeline]:
    pipeline = _TinyARDiffusionPipeline(device=device, dtype=dtype)
    runner = object.__new__(ARDiffusionModelRunner)
    runner.vllm_config = VllmConfig()
    runner.od_config = SimpleNamespace(
        max_num_seqs=1,
        dtype=dtype,
        enforce_eager=True,
        step_execution=False,
        parallel_config=SimpleNamespace(use_hsdp=False, pipeline_parallel_size=1),
        cache_backend=None,
    )
    runner.device = device
    runner.pipeline = pipeline
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(
        enable=True,
        gpu_memory_fraction=0.1,
    )
    runner.kv_cache = None
    runner.noisy_kv_cache = None
    runner._ar_diffusion_capability = None
    runner._ar_diffusion_cell_capability = None
    runner._ar_diffusion_kv_cache_spec = None
    runner._sessions = OrderedDict()
    runner._session_capacity = 0
    runner._perf_e2e_times = []
    runner._stepwise_chunk_started = {}
    runner._active_stepwise_request_id = None
    runner.state_cache = {}
    runner.input_batch = None
    runner.cache_backend = None
    runner.kv_transfer_manager = None
    runner._kv_transfer_manager = None
    runner._kv_connector = None
    runner._kv_receive_progress = None
    runner._kv_prefetch_enabled = False
    runner._preallocate_kv_cache(available_bytes=1 << 26)
    return runner, pipeline


def _tick_request(
    session_id: str,
    chunk_index: int,
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    reset: bool = False,
) -> OmniDiffusionRequest:
    tick = ARDiffusionTickRequest(
        session_id=session_id,
        request_id=f"req-{chunk_index}",
        chunk_index=chunk_index,
        reset=reset,
    )
    extra = tick.to_extra_args()
    extra["qkv"] = qkv
    return OmniDiffusionRequest(
        prompt="world tick",
        sampling_params=OmniDiffusionSamplingParams(extra_args=extra),
        request_id=f"req-{chunk_index}",
    )


@pytest.mark.parametrize("kv_gather", ["0", "1"])
def test_ar_diffusion_engine_for_world_models(
    kv_gather: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARDiffusionEngine for World Models

    - TPU path: resolve ``ARDiffusionEngine``, preallocate ``ARDiffusionKVCache``
      on ``torch.device("tpu")``, and execute a 2-tick ``ARDiffusionTickRequest``
      rollout calling ``paged_write_attn`` in bf16 across denoise steps.
    - Setup check: ``engine_cls is ARDiffusionEngine``,
      ``runner.kv_cache._k_pools[0].device.type == "tpu"``,
      ``runner._session_capacity >= 1``, and ``completed_chunks`` advances 1 -> 2.
    - Outcome check: tick-1 output attends to tick-0's committed KV (diverges from
      ``reset=True``) and matches CPU fp32 rollout within bf16 budget.
    """
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", kv_gather)
    engine_cls = DiffusionEngine.resolve_engine_class(
        SimpleNamespace(
            engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine"
        )
    )
    assert engine_cls is ARDiffusionEngine, (
        f"Expected ARDiffusionEngine, got {engine_cls}"
    )

    torch.manual_seed(0)
    qkv0 = (
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
    )
    qkv1 = (
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
        torch.randn(_TOKENS_PER_FRAME, _NUM_HEADS, _HEAD_DIM),
    )

    tpu_runner, _ = _make_runner(torch.device("tpu"), torch.bfloat16)
    assert tpu_runner.kv_cache is not None, "TPU ARDiffusionKVCache not allocated"
    assert tpu_runner.kv_cache._k_pools[0].device.type == "tpu", (
        f"KV pool device {tpu_runner.kv_cache._k_pools[0].device} is not tpu"
    )
    assert tpu_runner._session_capacity >= 1, (
        f"Expected session_capacity >= 1, got {tpu_runner._session_capacity}"
    )

    tpu_runner.execute_model(_tick_request("s1", 0, qkv0))
    state = tpu_runner._sessions["s1"]
    assert state.adapter("main").completed_chunks == 1, (
        f"Expected 1 completed chunk after tick 0, got "
        f"{state.adapter('main').completed_chunks}"
    )
    tpu_tick1 = tpu_runner.execute_model(_tick_request("s1", 1, qkv1)).output
    assert state.adapter("main").completed_chunks == 2, (
        f"Expected 2 completed chunks after tick 1, got "
        f"{state.adapter('main').completed_chunks}"
    )
    tpu_reset = tpu_runner.execute_model(
        _tick_request("s1", 1, qkv1, reset=True)
    ).output
    assert not torch.allclose(tpu_tick1.cpu(), tpu_reset.cpu(), atol=1e-3), (
        "Tick-1 output did not attend to tick-0 committed KV (matches reset=True)"
    )

    cpu_fp32_runner, _ = _make_runner(torch.device("cpu"), torch.float32)
    cpu_fp32_runner.execute_model(_tick_request("s1", 0, qkv0))
    ref_fp32 = cpu_fp32_runner.execute_model(_tick_request("s1", 1, qkv1)).output

    cpu_bf16_runner, _ = _make_runner(torch.device("cpu"), torch.bfloat16)
    cpu_bf16_runner.execute_model(_tick_request("s1", 0, qkv0))
    ref_bf16 = cpu_bf16_runner.execute_model(_tick_request("s1", 1, qkv1)).output

    cpu_bf16_err = float(
        (ref_bf16.double() - ref_fp32.double()).norm() / ref_fp32.double().norm()
    )
    budget = max(_BF16_RATIO * cpu_bf16_err, _BF16_FLOOR)
    assert_parity(
        tpu_tick1,
        ref_fp32,
        name=f"ar_diffusion_tick1_kv_gather_{kv_gather}",
        max_rel_l2=budget,
    )
