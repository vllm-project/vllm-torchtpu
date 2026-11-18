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
"""TPU tests for vLLM-Omni stage replicas and load balancing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from torch_tpu._internal.utils import hardware
from vllm_omni.engine.stage_init_utils import split_devices_for_replicas
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.multichip,
    pytest.mark.spawns_tpu_workers,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

NUM_REPLICAS = 2
IMAGE_SIZE = 256
PROMPT_A = "A lighthouse beam cutting through fog at dusk"
PROMPT_B = "A red sailboat anchored in a mountain cove"
SEED_A = 42
SEED_B = 43


def _generate_rgb(runner: OmniRunner, prompt: str, seed: int) -> np.ndarray:
    params = OmniDiffusionSamplingParams(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        num_inference_steps=1,
        guidance_scale=0.0,
        seed=seed,
    )
    outputs = list(runner.generate(prompt, params))
    img = outputs[0].images[0]
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "Second single-rank StageDiffusionProc replica aborts on "
        "/tmp/libtpu_lockfile because OmniTpuPlatform.set_device does not set "
        "ALLOW_MULTIPLE_LIBTPU_LOAD=1"
    ),
)
def test_stage_replicas_and_load_balancing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage replicas and load balancing.

    - TPU path: launch ``OmniRunner`` on ``tiny_flux_schnell()`` with
      ``num_replicas=2``, ``devices="0,1"``, and ``omni_lb_policy="round-robin"``.
    - Setup check: assert ``split_devices_for_replicas("0,1", 2, 1, 0) == ["0", "1"]``
      and ``stage_pools[0].live_num_replicas == 2``.
    - Outcome check: generate two seeded ``(256, 256, 3)`` requests across the
      replicas and assert bitwise parity against a single-replica ``mp`` engine.
    """
    device_count = hardware.get_tpu_device_count()
    assert device_count >= NUM_REPLICAS, (
        f"Test requires {NUM_REPLICAS} TPU devices, found {device_count}"
    )

    replica_devices = split_devices_for_replicas("0,1", NUM_REPLICAS, 1, 0)
    assert replica_devices == ["0", "1"], (
        f"Replica device mapping {replica_devices} != ['0', '1']"
    )

    monkeypatch.setenv(
        "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
        str(tmp_path / "tpu_compile_cache"),
    )
    model_dir = tiny_flux_schnell()

    with OmniRunner(
        model_dir,
        stage_overrides={"0": {"num_replicas": NUM_REPLICAS, "devices": "0,1"}},
        omni_lb_policy="round-robin",
        stage_init_timeout=120,
        init_timeout=240,
    ) as replica_runner:
        assert replica_runner.omni is not None, (
            "Replica OmniRunner did not initialize replica_runner.omni"
        )
        live_replicas = replica_runner.omni.engine.stage_pools[0].live_num_replicas
        assert live_replicas == NUM_REPLICAS, (
            f"Expected live_num_replicas={NUM_REPLICAS}, got {live_replicas}"
        )
        rep_img_a = _generate_rgb(replica_runner, PROMPT_A, SEED_A)
        rep_img_b = _generate_rgb(replica_runner, PROMPT_B, SEED_B)

    with OmniRunner(
        model_dir,
        distributed_executor_backend="mp",
        stage_init_timeout=120,
        init_timeout=240,
    ) as ref_runner:
        ref_img_a = _generate_rgb(ref_runner, PROMPT_A, SEED_A)
        ref_img_b = _generate_rgb(ref_runner, PROMPT_B, SEED_B)

    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert rep_img_a.shape == expected_shape, (
        f"Replica request A shape {rep_img_a.shape} != {expected_shape}"
    )
    assert rep_img_b.shape == expected_shape, (
        f"Replica request B shape {rep_img_b.shape} != {expected_shape}"
    )
    assert not np.array_equal(rep_img_a, rep_img_b), (
        "Distinct prompts and seeds produced identical images across replicas"
    )
    np.testing.assert_array_equal(
        rep_img_a,
        ref_img_a,
        err_msg="Replica pool request A diverged from single-replica mp reference",
    )
    np.testing.assert_array_equal(
        rep_img_b,
        ref_img_b,
        err_msg="Replica pool request B diverged from single-replica mp reference",
    )
