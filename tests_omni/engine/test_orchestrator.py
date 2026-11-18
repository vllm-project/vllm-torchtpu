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
"""TPU engine tests for the event-driven orchestrator scheduling loop."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from vllm_omni.engine.orchestrator import _event_driven_orch_enabled
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

ORCH_PARITY_MAX_ABS = 0.0


def test_event_driven_orchestrator_scheduling_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event-Driven Orchestrator Scheduling Loop.

    - TPU path: run single-device inline ``OmniRunner`` on
      ``tiny_flux_schnell()`` (256x256, 1 step, seed=0) with
      ``VLLM_OMNI_EVENT_DRIVEN_ORCH="1"`` and ``"0"``.
    - Setup check: ``_event_driven_orch_enabled()`` matches each env setting
      and ``stage_pools[0].membership_listener`` is set under event-driven mode
      and None under legacy polling.
    - Outcome check: both runs return finished 256x256 RGB images and
      event-driven pixels match legacy polling exactly (``max_abs == 0``).
    """
    model_dir = tiny_flux_schnell()
    prompt = "A red square"
    sampling_params = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=0,
    )

    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "1")
    assert _event_driven_orch_enabled() is True, (
        "Expected _event_driven_orch_enabled() True when env is '1'"
    )
    with OmniRunner(model_dir) as runner_event:
        listener_event = runner_event.engine.stage_pools[0].membership_listener
        assert listener_event is not None, (
            "Expected membership_listener to be installed in event-driven mode"
        )
        outputs_event = list(runner_event.generate(prompt, sampling_params))

    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "0")
    assert _event_driven_orch_enabled() is False, (
        "Expected _event_driven_orch_enabled() False when env is '0'"
    )
    with OmniRunner(model_dir) as runner_poll:
        listener_poll = runner_poll.engine.stage_pools[0].membership_listener
        assert listener_poll is None, (
            "Expected membership_listener to be None in legacy polling mode"
        )
        outputs_poll = list(runner_poll.generate(prompt, sampling_params))

    assert outputs_event, "Event-driven orchestrator run returned no outputs"
    assert outputs_event[0].finished, (
        f"Event-driven orchestrator output[0].finished="
        f"{outputs_event[0].finished!r}, expected True"
    )
    assert outputs_poll, "Legacy polling orchestrator run returned no outputs"
    assert outputs_poll[0].finished, (
        f"Legacy polling orchestrator output[0].finished="
        f"{outputs_poll[0].finished!r}, expected True"
    )

    img_event = outputs_event[0].images[0]
    img_poll = outputs_poll[0].images[0]
    assert isinstance(img_event, Image.Image), (
        f"Expected PIL Image from event-driven run, got {type(img_event)}"
    )
    assert isinstance(img_poll, Image.Image), (
        f"Expected PIL Image from polling run, got {type(img_poll)}"
    )
    assert img_event.size == (256, 256), (
        f"Expected (256, 256) event-driven image, got {img_event.size}"
    )
    assert img_poll.size == (256, 256), (
        f"Expected (256, 256) polling image, got {img_poll.size}"
    )

    arr_event = torch.from_numpy(np.array(img_event.convert("RGB"), dtype=np.float32))
    arr_poll = torch.from_numpy(np.array(img_poll.convert("RGB"), dtype=np.float32))
    assert_parity(
        arr_event,
        arr_poll,
        name="event_driven_vs_polling_pixels",
        max_abs=ORCH_PARITY_MAX_ABS,
    )
