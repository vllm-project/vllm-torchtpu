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
"""TPU tests for DiffusionWorker, WorkerProc async output, sleep, and weight reload."""

from __future__ import annotations

import functools
import queue
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from vllm.config import DeviceConfig, VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightSource
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
)
from vllm_omni.diffusion.data import (
    AsyncDiffusionOutput,
    AsyncOutputKind,
    DiffusionOutput,
    OmniDiffusionConfig,
    OmniSleepTask,
)
from vllm_omni.diffusion.ipc import unpack_diffusion_output_shm
from vllm_omni.diffusion.model_loader.weight_reload import (
    prepare_model_for_weight_reload,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QwenImagePipeline,
)
from vllm_omni.diffusion.worker.diffusion_worker import (
    CustomPipelineWorkerExtension,
    DiffusionWorker,
    WorkerProc,
)
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import (
    NIGHTLY_TIMEOUT_S,
    PR_TIMEOUT_S,
    assert_parity,
)
from tests_omni.tiny_models import tiny_diffusion_dir, tiny_qwen_image

pytestmark = [pytest.mark.omni_feature, pytest.mark.omni]

PARITY_MAX_REL_L2 = 2.0e-2
PARITY_MAX_ABS = 5.0e-2
IMAGE_MAX_REL_L2 = 5.0e-2


class _TinyReloadPipeline(nn.Module):
    supports_weight_reload = True

    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, dtype=dtype, device=device)

    def load_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        for name, tensor in weights:
            if name in params:
                with torch.no_grad():
                    params[name].copy_(
                        tensor.to(device=params[name].device, dtype=params[name].dtype)
                    )
                loaded.add(name)
        return loaded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _TinyWeightSource(WeightSource):
    def __init__(self, weight: torch.Tensor) -> None:
        self._weight = weight

    @functools.cached_property
    def metadata(self) -> list[tuple[str, torch.Size, torch.dtype]]:
        return [("proj.weight", self._weight.shape, self._weight.dtype)]

    def owned_names(self, rank: int) -> set[str]:
        del rank
        return {"proj.weight"}

    def implements_iter_full(self) -> bool:
        return True

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield ("proj.weight", self._weight)


class _FakeSyncClient:
    def __init__(self, worker_engine: IPCWeightTransferEngine) -> None:
        self._worker_engine = worker_engine

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._worker_engine.init_transfer_engine(init_info)  # type: ignore[arg-type]

    def start_weight_update(self) -> None:
        self._worker_engine.start_weight_update()

    def finish_weight_update(self) -> None:
        self._worker_engine.finish_weight_update()


class _TinySleepPipeline(nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, dtype=dtype, device=device)
        self.register_buffer("scale", torch.tensor([1.5], dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * self.scale


class _ListResultMQ:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def enqueue(self, msg: Any) -> None:
        self.items.append(msg)


class _CustomQwenImagePipelineA(QwenImagePipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.custom_forward_calls = 0

    def forward(self, req: Any) -> DiffusionOutput:
        self.custom_forward_calls += 1
        return super().forward(req)


class _CustomQwenImagePipelineB(QwenImagePipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.custom_forward_calls = 0

    def forward(self, req: Any) -> DiffusionOutput:
        self.custom_forward_calls += 1
        return super().forward(req)


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "IPCTrainerWeightTransferEngine hits torch.cuda.get_device_properties "
        "and raises 'Torch not compiled with CUDA enabled' on TPU"
    ),
)
def test_weight_transfer_for_rl_training() -> None:
    """Weight transfer for RL training.

    - TPU path: Prepare a tiny bf16 reload-enabled pipeline on TPU, initialize
      DiffusionWorker._init_weight_transfer_engine with
      WeightTransferConfig(backend="ipc"), and drive
      IPCTrainerWeightTransferEngine.trainer_init + send_weights.
    - Setup check: Assert _omni_weight_reload_prepared is True and
      worker.weight_transfer_engine is an IPCWeightTransferEngine.
    - Outcome check: Run pipeline forward on TPU after finish_weight_update and
      compare against CPU fp32 forward of the updated weights via assert_parity.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    )
    pipeline = _TinyReloadPipeline(dtype=torch.bfloat16, device=tpu_device)
    prepare_model_for_weight_reload(pipeline, od_config, tpu_device)
    assert "_omni_weight_reload_prepared" in pipeline.__dict__, (
        "prepare_model_for_weight_reload did not set _omni_weight_reload_prepared"
    )
    assert pipeline._omni_weight_reload_prepared is True, (
        "expected pipeline._omni_weight_reload_prepared is True, "
        f"got {pipeline._omni_weight_reload_prepared!r}"
    )

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="tpu"),
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    )
    worker = object.__new__(DiffusionWorker)
    worker.vllm_config = vllm_config
    worker.device = tpu_device
    worker.model_runner = SimpleNamespace(pipeline=pipeline)  # type: ignore[assignment]
    worker.weight_transfer_engine = None
    worker._init_weight_transfer_engine()
    assert isinstance(worker.weight_transfer_engine, IPCWeightTransferEngine), (
        f"Expected IPCWeightTransferEngine, got "
        f"{type(worker.weight_transfer_engine).__name__}"
    )

    new_weight_cpu = torch.randn(4, 4, dtype=torch.float32) * 0.25
    trainer = IPCTrainerWeightTransferEngine.trainer_init(
        IPCTrainerInitInfo(rank=0, packed=False),
        client=_FakeSyncClient(worker.weight_transfer_engine),  # type: ignore[arg-type]
        source=_TinyWeightSource(
            new_weight_cpu.to(dtype=torch.bfloat16, device=tpu_device)
        ),
    )
    trainer.send_weights()

    x_cpu = torch.randn(2, 4, dtype=torch.float32)
    tpu_out = pipeline(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))
    ref_out = x_cpu @ new_weight_cpu.T
    assert_parity(
        tpu_out,
        ref_out,
        name="weight_transfer_for_rl_training",
        max_rel_l2=PARITY_MAX_REL_L2,
        max_abs=PARITY_MAX_ABS,
    )


@pytest.mark.timeout(PR_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "DiffusionWorker.sleep calls CuMemAllocator.get_instance() which "
        "raises AssertionError ('cumem allocator is not available') on TPU"
    ),
)
@pytest.mark.parametrize("level", [1, 2])
def test_sleep_mode(level: int) -> None:
    """Sleep mode.

    - TPU path: Build a bare DiffusionWorker with a tiny bf16 pipeline on TPU
      and run worker.handle_sleep_task(OmniSleepTask(..., level=level)) followed
      by worker.wake_up().
    - Setup check: Assert pipeline tensors reside on TPU before sleep and
      OmniACK status == "SUCCESS" with freed_bytes > 0.
    - Outcome check: Wake the worker, assert parameters and persistent buffers
      are restored on TPU, and compare forward output via assert_parity.
    """
    torch.manual_seed(42)
    tpu_device = torch.device("tpu")
    ref_pipeline = _TinySleepPipeline(dtype=torch.float32, device=torch.device("cpu"))
    pipeline = _TinySleepPipeline(dtype=torch.bfloat16, device=tpu_device)
    with torch.no_grad():
        pipeline.proj.weight.copy_(
            ref_pipeline.proj.weight.to(dtype=torch.bfloat16, device=tpu_device)
        )
        pipeline.scale.copy_(
            ref_pipeline.scale.to(dtype=torch.bfloat16, device=tpu_device)
        )

    assert pipeline.proj.weight.device.type == "tpu", (
        f"Expected pipeline weight on tpu, got {pipeline.proj.weight.device}"
    )
    assert pipeline.scale.device.type == "tpu", (
        f"Expected pipeline buffer on tpu, got {pipeline.scale.device}"
    )

    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.device = tpu_device
    worker.od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(), enable_sleep_mode=True
    )
    worker.model_runner = SimpleNamespace(
        pipeline=pipeline,
        release_captured_graphs=lambda: None,
        refresh_diffusion_kv_block_table_layout=lambda: None,
    )  # type: ignore[assignment]
    worker._sleep_saved_buffers = {}

    ack = worker.handle_sleep_task(OmniSleepTask(task_id="sleep-test", level=level))
    assert ack is not None, "Expected non-None OmniACK on rank 0"
    assert ack.status == "SUCCESS", (
        f"Expected OmniACK status='SUCCESS', got {ack.status!r} ({ack.error_msg})"
    )
    assert ack.freed_bytes > 0, f"Expected freed_bytes > 0, got {ack.freed_bytes}"

    assert worker.wake_up() is True, "wake_up() returned False"
    assert pipeline.proj.weight.device.type == "tpu", (
        f"Expected restored weight on tpu, got {pipeline.proj.weight.device}"
    )
    assert pipeline.scale.device.type == "tpu", (
        f"Expected restored buffer on tpu, got {pipeline.scale.device}"
    )

    x_cpu = torch.randn(2, 4, dtype=torch.float32)
    tpu_out = pipeline(x_cpu.to(dtype=torch.bfloat16, device=tpu_device))
    ref_out = ref_pipeline(x_cpu)
    assert_parity(
        tpu_out,
        ref_out,
        name=f"sleep_mode_level_{level}",
        max_rel_l2=PARITY_MAX_REL_L2,
        max_abs=PARITY_MAX_ABS,
    )


@pytest.mark.timeout(PR_TIMEOUT_S)
def test_async_diffusion_output() -> None:
    """Async diffusion output.

    - TPU path: Construct a lightweight WorkerProc via object.__new__ with
      gpu_id=1 and run _async_output_loop on a background thread while
      _return_result emits a >1 MB TPU bf16 DiffusionOutput tensor.
    - Setup check: Assert result_mq receives COMPUTE_DONE then OUTPUT_READY with
      matching async_output_id and a __tensor_shm__ handle.
    - Outcome check: Unpack via unpack_diffusion_output_shm after
      drain_async_outputs() and assert exact bf16 equality against CPU tensor.
    """
    torch.manual_seed(42)
    ref_tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)
    tpu_tensor = ref_tensor.to(device=torch.device("tpu"))
    torch.accelerator.synchronize()

    proc = object.__new__(WorkerProc)
    proc.gpu_id = 1
    proc.od_config = OmniDiffusionConfig(model=tiny_diffusion_dir())
    proc.result_mq = _ListResultMQ()
    proc._result_mq_lock = threading.Lock()
    proc._async_output_queue = queue.Queue()
    proc._async_output_pending = 0
    proc._async_output_done = threading.Condition()

    thread = threading.Thread(target=proc._async_output_loop, daemon=True)
    thread.start()
    try:
        proc._return_result(DiffusionOutput(output=tpu_tensor), rpc_id="rpc-1")
        drained = proc.drain_async_outputs(timeout=30.0)
        assert drained, "drain_async_outputs timed out"
    finally:
        proc._async_output_queue.put(None)
        thread.join(timeout=10.0)

    msgs = proc.result_mq.items
    assert len(msgs) == 2, f"Expected 2 async messages, got {len(msgs)}"
    assert all(isinstance(msg, AsyncDiffusionOutput) for msg in msgs), (
        f"Unexpected message types: {[type(m).__name__ for m in msgs]}"
    )
    assert msgs[0].kind is AsyncOutputKind.COMPUTE_DONE, (
        f"Expected COMPUTE_DONE first, got {msgs[0].kind}"
    )
    assert msgs[1].kind is AsyncOutputKind.OUTPUT_READY, (
        f"Expected OUTPUT_READY second, got {msgs[1].kind}"
    )
    assert msgs[0].async_output_id == msgs[1].async_output_id, (
        f"Mismatched async_output_id: {msgs[0].async_output_id} != "
        f"{msgs[1].async_output_id}"
    )
    assert msgs[1].error is None, f"Async output failed: {msgs[1].error}"

    packed_output = msgs[1].output
    assert isinstance(packed_output, DiffusionOutput), (
        f"Expected DiffusionOutput payload, got {type(packed_output).__name__}"
    )
    assert isinstance(packed_output.output, dict), (
        f"Expected POSIX SHM dict in payload, got {type(packed_output.output).__name__}"
    )
    assert "__tensor_shm__" in packed_output.output, (
        "Expected '__tensor_shm__' key in SHM payload, "
        f"got {list(packed_output.output)!r}"
    )

    unpacked = unpack_diffusion_output_shm(packed_output)
    assert isinstance(unpacked, DiffusionOutput), (
        f"Expected unpacked DiffusionOutput, got {type(unpacked).__name__}"
    )
    assert isinstance(unpacked.output, torch.Tensor), (
        f"Expected unpacked torch.Tensor output, got {type(unpacked.output).__name__}"
    )
    assert unpacked.output.dtype is torch.bfloat16, (
        f"Expected bf16 unpacked tensor, got {unpacked.output.dtype}"
    )
    assert torch.equal(unpacked.output, ref_tensor), (
        "Unpacked async diffusion output tensor differs from CPU reference"
    )


@pytest.mark.nightly
@pytest.mark.timeout(NIGHTLY_TIMEOUT_S)
@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "custom_pipeline builds on CPU then calls model.to('tpu') before "
        "load_weights, stripping param.weight_loader on torch_tpu so "
        "QwenImageTransformer2DModel falls back to 2-arg default_weight_loader"
    ),
)
def test_custom_diffusion_pipeline_extension() -> None:
    """Custom Diffusion Pipeline Extension.

    - TPU path: Initialize inline single-device Omni on tiny_qwen_image()
      with diffusion_load_format="custom_pipeline" and custom_pipeline_args,
      run 256x256 1-step generation, and swap pipelines via collective_rpc.
    - Setup check: Assert CustomPipelineWorkerExtension,
      _CustomQwenImagePipelineA instance, custom forward call count > 0, and
      TPU transformer weights.
    - Outcome check: Compare generated image against default-loaded Omni
      baseline and verify dynamic re_init_pipeline swap works on TPU.
    """
    model_dir = tiny_qwen_image()
    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=42,
        generator_device="cpu",
    )

    baseline_omni = Omni(model=model_dir, diffusion_load_format="default")
    try:
        baseline_outputs = list(baseline_omni.generate("a red cube", [sampling_params]))
        baseline_img = np.asarray(baseline_outputs[0].images[0], dtype=np.float32)
    finally:
        baseline_omni.close()

    custom_a_qualname = f"{__name__}.{_CustomQwenImagePipelineA.__name__}"
    custom_b_qualname = f"{__name__}.{_CustomQwenImagePipelineB.__name__}"
    custom_omni = Omni(
        model=model_dir,
        diffusion_load_format="custom_pipeline",
        custom_pipeline_args={"pipeline_class": custom_a_qualname},
    )
    try:
        diffusion_engine = custom_omni.engine.stage_clients[0]._diffusion_engine
        worker = diffusion_engine.executor.driver_worker
        assert isinstance(worker, CustomPipelineWorkerExtension), (
            f"Expected CustomPipelineWorkerExtension, got {type(worker).__name__}"
        )
        pipeline_a = worker.model_runner.pipeline
        assert isinstance(pipeline_a, _CustomQwenImagePipelineA), (
            f"Expected _CustomQwenImagePipelineA, got {type(pipeline_a).__name__}"
        )
        first_param = next(pipeline_a.transformer.parameters())
        assert first_param.device.type == "tpu", (
            f"Expected transformer weights on tpu, got {first_param.device}"
        )

        custom_outputs = list(custom_omni.generate("a red cube", [sampling_params]))
        assert pipeline_a.custom_forward_calls > 0, (
            "Expected custom forward call count > 0"
        )
        custom_img = np.asarray(custom_outputs[0].images[0], dtype=np.float32)
        assert_parity(
            torch.from_numpy(custom_img),
            torch.from_numpy(baseline_img),
            name="custom_diffusion_pipeline_extension",
            max_rel_l2=IMAGE_MAX_REL_L2,
        )

        diffusion_engine.collective_rpc(
            "re_init_pipeline",
            args=({"pipeline_class": custom_b_qualname},),
        )
        pipeline_b = worker.model_runner.pipeline
        assert isinstance(pipeline_b, _CustomQwenImagePipelineB), (
            f"Expected _CustomQwenImagePipelineB after re_init_pipeline, got "
            f"{type(pipeline_b).__name__}"
        )
        swapped_outputs = list(custom_omni.generate("a red cube", [sampling_params]))
        assert pipeline_b.custom_forward_calls > 0, (
            "Expected swapped pipeline forward call count > 0"
        )
        swapped_img = np.asarray(swapped_outputs[0].images[0], dtype=np.float32)
        assert_parity(
            torch.from_numpy(swapped_img),
            torch.from_numpy(baseline_img),
            name="custom_diffusion_pipeline_extension_swapped",
            max_rel_l2=IMAGE_MAX_REL_L2,
        )
    finally:
        custom_omni.close()
