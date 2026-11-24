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
"""TPU tests for diffusion step-level execution, continuous batching, and abort."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest
import torch
from vllm_omni.diffusion.diffusion_engine import DiffusionExecutionMode
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import OutputMessage
from vllm_omni.engine.orchestrator import OrchestratorBase
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_qwen_image

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

STEP_EXECUTION_MAX_REL_L2 = 5e-2
ABORT_WAIT_TIMEOUT_S = 60.0
ABORT_CLEANUP_TIMEOUT_S = 5.0


def _pil_to_tensor(image: Any) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)


def test_diffusion_step_execution_and_continuous_batching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diffusion Step Execution & Continuous Batching

    - Runs single-device ``Omni(model=tiny_qwen_image(), step_execution=True,
      max_num_seqs=2)`` on TPU for two 256x256, 2-step equal-token-length
      prompts, plus a mid-generation abort check.
    - Setup check: ``engine.execution_mode == DiffusionExecutionMode.STEP_BATCH``,
      ``isinstance(engine.scheduler, StepScheduler)``, and a spy on
      ``execute_stepwise`` observes co-batched waves (``num_reqs == 2``).
    - Outcome check: co-batched images match monolithic ``step_execution=False``
      outputs via relative L2, and aborting one request terminates it early
      without corrupting its co-batched peer.
    """
    model_dir = tiny_qwen_image()
    prompts = ["a red cat", "a blue dog"]
    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=2,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )

    omni_mono = Omni(model=model_dir, step_execution=False)
    try:
        mono_outputs = omni_mono.generate(prompts, sampling_params, use_tqdm=False)
    finally:
        omni_mono.close()

    step_wave_sizes: list[int] = []
    orig_execute_stepwise = DiffusionModelRunner.execute_stepwise

    def _spy_execute_stepwise(
        self_runner: DiffusionModelRunner,
        scheduler_output: Any,
    ) -> Any:
        step_wave_sizes.append(len(scheduler_output.scheduled_request_ids))
        return orig_execute_stepwise(self_runner, scheduler_output)

    monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", _spy_execute_stepwise)

    orchestrators: list[OrchestratorBase] = []
    orig_create_orchestrator = AsyncOmniEngine._create_orchestrator

    def _spy_create_orchestrator(
        self_engine: AsyncOmniEngine,
        **orchestrator_kwargs: Any,
    ) -> OrchestratorBase:
        orchestrator = orig_create_orchestrator(self_engine, **orchestrator_kwargs)
        orchestrators.append(orchestrator)
        return orchestrator

    monkeypatch.setattr(
        AsyncOmniEngine,
        "_create_orchestrator",
        _spy_create_orchestrator,
    )

    omni_step = Omni(model=model_dir, step_execution=True, max_num_seqs=2)
    try:
        assert len(orchestrators) == 1, (
            f"Expected 1 orchestrator, observed {len(orchestrators)}"
        )
        orchestrator = orchestrators[0]
        engine = omni_step.engine.stage_clients[0]._engine
        assert engine.execution_mode == DiffusionExecutionMode.STEP_BATCH, (
            f"Expected STEP_BATCH mode, got {engine.execution_mode}"
        )
        assert isinstance(engine.scheduler, StepScheduler), (
            f"Expected StepScheduler, got {type(engine.scheduler)}"
        )

        step_outputs = omni_step.generate(prompts, sampling_params, use_tqdm=False)
        assert any(size == 2 for size in step_wave_sizes), (
            f"Expected co-batched step wave of size 2, observed {step_wave_sizes}"
        )

        for idx, (step_out, mono_out) in enumerate(zip(step_outputs, mono_outputs)):
            assert_parity(
                _pil_to_tensor(step_out.images[0]),
                _pil_to_tensor(mono_out.images[0]),
                name=f"step execution vs monolithic req {idx}",
                max_rel_l2=STEP_EXECUTION_MAX_REL_L2,
            )

        abort_triggered = False

        def _abort_spy(
            self_runner: DiffusionModelRunner,
            scheduler_output: Any,
        ) -> Any:
            nonlocal abort_triggered
            result = orig_execute_stepwise(self_runner, scheduler_output)
            if not abort_triggered and "req_abort" in (
                scheduler_output.scheduled_request_ids
            ):
                abort_triggered = True
                asyncio.run(
                    omni_step.engine.abort_async(
                        ["req_abort"],
                        timeout=ABORT_CLEANUP_TIMEOUT_S,
                    )
                )
            return result

        monkeypatch.setattr(DiffusionModelRunner, "execute_stepwise", _abort_spy)
        omni_step.engine.add_request(
            request_id="req_abort",
            prompt=prompts[0],
            sampling_params_list=[sampling_params],
            final_stage_id=0,
            final_output_stage_ids=[0],
        )
        omni_step.engine.add_request(
            request_id="req_keep",
            prompt=prompts[1],
            sampling_params_list=[sampling_params],
            final_stage_id=0,
            final_output_stage_ids=[0],
        )

        abort_output_seen = False
        keep_output: Any = None
        deadline = time.monotonic() + ABORT_WAIT_TIMEOUT_S
        while keep_output is None and time.monotonic() < deadline:
            msg = omni_step.engine.try_get_output()
            if msg is None:
                continue
            if isinstance(msg, OutputMessage) and msg.finished:
                if msg.request_id == "req_abort":
                    abort_output_seen = True
                elif msg.request_id == "req_keep":
                    keep_output = msg.engine_outputs

        assert abort_triggered, "Mid-generation abort was never triggered"
        assert not abort_output_seen, "Aborted request unexpectedly produced output"
        assert "req_abort" not in orchestrator.request_states, (
            "Aborted request still in orchestrator request_states after abort"
        )
        assert engine.scheduler.get_request_state("req_abort") is None, (
            "Aborted request still in StepScheduler after abort"
        )
        assert keep_output is not None, (
            "Surviving peer request produced engine_outputs=None after peer abort"
        )
        assert keep_output.images, (
            "Surviving peer request did not produce an image after peer abort"
        )
        assert_parity(
            _pil_to_tensor(keep_output.images[0]),
            _pil_to_tensor(mono_outputs[1].images[0]),
            name="surviving peer after mid-generation abort",
            max_rel_l2=STEP_EXECUTION_MAX_REL_L2,
        )
    finally:
        omni_step.close()
