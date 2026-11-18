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
"""TPU tests for diffusion request-level batching and admission wait window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from vllm_omni.diffusion.diffusion_engine import DiffusionExecutionMode
from vllm_omni.diffusion.sched import RequestScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

REQUEST_BATCH_MAX_REL_L2 = 5e-2


def _pil_to_tensor(image: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "Batched (2-request) Flux VAE decode fails on TPU: torch.group_norm in the "
        "decoder mid-block calls view() on a channels-last [2, 128, 32, 32] "
        "activation. Single-request decodes pass."
    ),
)
def test_diffusion_request_level_batching_and_admission_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diffusion Request-Level Batching & Admission Window

    - Runs single-device ``Omni(model=tiny_flux_schnell(), max_num_seqs=2,
      request_batch_max_wait_ms=500.0)`` on TPU for two 256x256, 2-step prompts.
    - Setup check: ``engine.execution_mode == DiffusionExecutionMode.REQUEST_BATCH``,
      ``isinstance(engine.scheduler, RequestScheduler)``,
      ``get_admission_wait_decision(now=0.0).should_wait is True``, and a spy on
      ``execute_model_batch`` records a single batch wave of size 2.
    - Outcome check: each batched image matches its single-request (batch size 1)
      output via relative L2.
    """
    model_dir = tiny_flux_schnell()
    prompts = ["a red cat", "a blue dog"]
    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=2,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )

    batch_wave_sizes: list[int] = []
    orig_execute_model_batch = DiffusionModelRunner.execute_model_batch

    def _spy_execute_model_batch(
        self_runner: DiffusionModelRunner,
        scheduler_output: Any,
        od_config: Any,
    ) -> Any:
        batch_wave_sizes.append(len(scheduler_output.scheduled_new_reqs))
        return orig_execute_model_batch(self_runner, scheduler_output, od_config)

    monkeypatch.setattr(
        DiffusionModelRunner, "execute_model_batch", _spy_execute_model_batch
    )

    omni = Omni(
        model=model_dir,
        max_num_seqs=2,
        request_batch_max_wait_ms=500.0,
    )
    try:
        engine = omni.engine.stage_clients[0]._engine
        assert engine.execution_mode == DiffusionExecutionMode.REQUEST_BATCH, (
            f"Expected REQUEST_BATCH mode, got {engine.execution_mode}"
        )
        assert isinstance(engine.scheduler, RequestScheduler), (
            f"Expected RequestScheduler, got {type(engine.scheduler)}"
        )
        decision = engine.scheduler.get_admission_wait_decision(now=0.0)
        assert decision.should_wait is True, (
            f"Expected admission wait decision should_wait=True, got {decision}"
        )

        serial_0 = omni.generate([prompts[0]], sampling_params, use_tqdm=False)[0]
        serial_1 = omni.generate([prompts[1]], sampling_params, use_tqdm=False)[0]

        batch_wave_sizes.clear()
        batched = omni.generate(prompts, sampling_params, use_tqdm=False)

        assert batch_wave_sizes == [2], (
            f"Expected a single batch wave of size 2, observed {batch_wave_sizes}"
        )
        for idx, (batched_out, serial_out) in enumerate(
            zip(batched, (serial_0, serial_1))
        ):
            assert_parity(
                _pil_to_tensor(batched_out.images[0]),
                _pil_to_tensor(serial_out.images[0]),
                name=f"request batching vs serial req {idx}",
                max_rel_l2=REQUEST_BATCH_MAX_REL_L2,
            )
    finally:
        omni.close()
