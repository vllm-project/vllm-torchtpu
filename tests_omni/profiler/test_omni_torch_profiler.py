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
"""TPU tests for vLLM-Omni torch and CUDA profiler backends on DiffusionWorker."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import ProfilerConfig
from vllm.profiler.wrapper import CudaProfilerWrapper
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from vllm_torchtpu.omni.profiler import TpuOmniTorchProfilerWrapper

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

_BF16_BUDGET_RATIO = 2.0
_BF16_BUDGET_FLOOR = 1e-3
_EXPECTED_PROFILER_CLS = {
    "torch": TpuOmniTorchProfilerWrapper,
    "cuda": CudaProfilerWrapper,
}


@pytest.mark.parametrize(
    "profiler_backend",
    [
        pytest.param("torch", id="torch"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason=(
                    "CUDA Nsight profiler calls torch.cuda.nvtx.range, "
                    "unavailable on TPU"
                ),
            ),
        ),
    ],
)
def test_profiling(tmp_path: Path, profiler_backend: str) -> None:
    """Profiling (Torch Profiler & Nsight Systems)

    - TPU path: Runs ``DiffusionWorker._create_profiler`` and ``profile``
      for ``"torch"`` (``TpuOmniTorchProfilerWrapper``) and ``"cuda"``
      (``CudaProfilerWrapper``, xfail on TPU) around a live TPU bf16 matmul.
    - Setup check: Verifies the created profiler class and ``profiler.is_running``
      inside ``annotate_context_manager("diffusion_forward")``.
    - Outcome check: Verifies TPU matmul parity against CPU fp32, non-empty Chrome
      trace ``traceEvents`` in ``trace_rank0.json``, and non-empty ``*.xplane.pb``.
    """
    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.local_rank = 0
    profiler_cfg = (
        ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(tmp_path),
            torch_profiler_use_gzip=False,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=False,
            torch_profiler_with_flops=True,
        )
        if profiler_backend == "torch"
        else ProfilerConfig(profiler=profiler_backend)
    )
    worker.od_config = SimpleNamespace(profiler_config=profiler_cfg)

    profiler = worker._create_profiler()
    worker.profiler = profiler
    expected_cls = _EXPECTED_PROFILER_CLS[profiler_backend]
    assert isinstance(profiler, expected_cls), (
        f"backend {profiler_backend!r}: expected {expected_cls.__name__}, "
        f"got {type(profiler).__name__}"
    )

    torch.manual_seed(0)
    a_cpu = torch.randn(64, 64, dtype=torch.float32)
    b_cpu = torch.randn(64, 64, dtype=torch.float32)
    ref_fp32 = a_cpu @ b_cpu
    ref_bf16 = a_cpu.bfloat16() @ b_cpu.bfloat16()
    cpu_bf16_err = float(
        (ref_bf16.double() - ref_fp32.double()).norm() / ref_fp32.double().norm()
    )

    device = current_omni_platform.get_torch_device(0)
    a_tpu = a_cpu.bfloat16().to(device)
    b_tpu = b_cpu.bfloat16().to(device)

    worker.profile(is_start=True, profile_prefix="tpu_prof")
    with profiler.annotate_context_manager("diffusion_forward"):
        assert profiler.is_running is True, (
            f"profiler {profiler_backend!r} must be running inside annotated region"
        )
        tpu_out = a_tpu @ b_tpu
        current_omni_platform.synchronize()
    profiler.step()
    worker.profile(is_start=False)

    assert_parity(
        tpu_out,
        ref_fp32,
        name=f"profile_{profiler_backend}_matmul",
        max_rel_l2=max(_BF16_BUDGET_RATIO * cpu_bf16_err, _BF16_BUDGET_FLOOR),
    )

    trace_files = list(tmp_path.glob("*_tpu_prof/trace_rank0.json"))
    assert len(trace_files) == 1, (
        f"expected 1 trace_rank0.json under {tmp_path}, found {trace_files}"
    )
    trace_payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert len(trace_payload["traceEvents"]) > 0, (
        "exported Chrome traceEvents must be non-empty"
    )

    xplane_files = list(
        (tmp_path / "tpu_prof" / "plugins" / "profile").glob("*/*.xplane.pb")
    )
    assert len(xplane_files) >= 1, (
        f"expected at least one .xplane.pb under {tmp_path / 'tpu_prof'}"
    )
    assert xplane_files[0].stat().st_size > 0, (
        f"XSpace file {xplane_files[0]} must be non-empty"
    )
