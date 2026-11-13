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
"""Unit tests for src/vllm_torchtpu/omni/worker/base.py and tpu_ar_worker.py."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import ParallelConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm_omni.config import OmniModelConfig
from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_torchtpu.omni.platform import OmniTpuPlatform
from vllm_torchtpu.omni.worker.base import OmniTPUWorkerBase
from vllm_torchtpu.omni.worker.tpu_ar_model_runner import TPUARModelRunner
from vllm_torchtpu.omni.worker.tpu_ar_worker import TPUARWorker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker

pytestmark = pytest.mark.omni


def test_tpu_ar_worker_wiring():
    assert resolve_obj_by_qualname(OmniTpuPlatform.get_omni_ar_worker_cls()) is (
        TPUARWorker
    )
    assert TPUARWorker.__bases__ == (OmniWorkerMixin, OmniTPUWorkerBase)
    assert issubclass(OmniTPUWorkerBase, TPUWorker)
    assert TPUARWorker.model_runner_cls is TPUARModelRunner


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("handle_sleep_task", ({"task_id": 1, "level": 1},)),
        ("handle_wake_task", ({"task_id": 2},)),
        ("encoder_loaded", ()),
    ],
)
def test_worker_base_refuses_unimplemented_orchestrator_hooks(method, args):
    """Refuse unimplemented hooks; replace with functional tests once implemented."""
    worker = OmniTPUWorkerBase.__new__(OmniTPUWorkerBase)
    with pytest.raises(NotImplementedError, match="TPU"):
        getattr(worker, method)(*args)


def test_tpu_ar_worker_init_device_builds_ar_model_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = OmniModelConfig.__new__(OmniModelConfig)
    for name, value in {
        "dtype": torch.bfloat16,
        "model_stage": "thinker",
        "stage_id": 0,
        "requires_full_payload_input": False,
        "custom_process_next_stage_input_func": None,
        "stage_connector_config": None,
    }.items():
        object.__setattr__(model_config, name, value)

    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=ParallelConfig(),
        compilation_config=SimpleNamespace(compile_ranges_endpoints=[]),
        profiler_config=SimpleNamespace(torch_profiler_dir=None),
        speculative_config=None,
        additional_config={},
    )

    def _fake_worker_base_init(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker=False,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

    def _fake_runner_init(self, vllm_config, device, **kwargs):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.speculative_config = vllm_config.speculative_config
        self.device = device

    monkeypatch.setattr("vllm_torchtpu.patch_registry.apply", lambda *_, **__: None)
    monkeypatch.setattr(
        "vllm_torchtpu.worker.tpu_worker.WorkerBase.__init__",
        _fake_worker_base_init,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.worker.tpu_worker.init_distributed_environment",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.worker.tpu_worker.ensure_model_parallel_initialized",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "vllm_torchtpu.worker.tpu_worker.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.worker.tpu_worker.vllm_utils.report_usage_stats",
        lambda *_: None,
    )
    monkeypatch.setattr(TPUModelRunner, "__init__", _fake_runner_init)

    worker = TPUARWorker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:29500",
        is_driver_worker=True,
    )
    worker.devices = [torch.device("cpu")]
    worker.init_device()

    assert isinstance(worker.model_runner, TPUARModelRunner)
    assert (worker.model_runner.model_stage, worker.model_runner.stage_id) == (
        "thinker",
        0,
    )
