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
"""TPU engine tests for vllm_omni metrics, stage stats, and Prometheus logging."""

import numpy as np
import pytest
from prometheus_client import REGISTRY
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from tests_omni.conftest import OmniRunner
from tests_omni.omni_test_utils import NIGHTLY_TIMEOUT_S
from tests_omni.tiny_models import tiny_flux_schnell

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.nightly,
    pytest.mark.timeout(NIGHTLY_TIMEOUT_S),
]

_EXPECTED_IMAGE_PIXELS = 256 * 256
_PROMPT = {"prompt": "a blue sphere on a wooden desk"}


def _sampling_params() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=1,
        seed=0,
    )


def _prom_metric_sum(sample_name: str) -> float:
    total = 0.0
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == sample_name:
                total += float(sample.value)
    return total


def test_metrics_and_stats_logging() -> None:
    """Metrics and stats logging

    - TPU path: Runs single-device ``OmniRunner`` on ``tiny_flux_schnell()`` with
      ``log_stats=True, collect_stage_metrics=True`` against a disabled baseline.
    - Setup check: Verifies ``log_stats`` and ``collect_stage_metrics`` on both
      ``runner.omni`` and ``runner.omni.engine`` match each configuration.
    - Outcome check: Verifies ``peak_memory_mb > 0``, ``stage_0_gen_ms > 0``,
      diagnostic ``stage_metrics["0"]`` fields, Prometheus ``REGISTRY`` deltas,
      and bitwise image parity against baseline.
    """
    model_dir = tiny_flux_schnell()
    success_before_base = _prom_metric_sum("vllm_omni:requests_success_total")

    base_runner = OmniRunner(model_dir, log_stats=False, collect_stage_metrics=False)
    try:
        assert base_runner.omni.log_stats is False, (
            "baseline omni.log_stats must be False"
        )
        assert base_runner.omni.collect_stage_metrics is False, (
            "baseline omni.collect_stage_metrics must be False"
        )
        assert base_runner.omni.engine._log_stats is False, (
            "baseline engine._log_stats must be False"
        )
        assert base_runner.omni.engine._collect_stage_metrics is False, (
            "baseline engine._collect_stage_metrics must be False"
        )
        base_out = list(base_runner.generate(_PROMPT, _sampling_params()))[0]
        base_img = np.asarray(base_out.images[0])
    finally:
        base_runner.close()

    base_stage0 = (base_out.metrics or {}).get("stage_metrics", {}).get("0", {})
    assert "denoise_step_latency_ms" not in base_stage0, (
        f"diagnostic denoise_step_latency_ms present when disabled: {base_stage0}"
    )
    success_after_base = _prom_metric_sum("vllm_omni:requests_success_total")
    assert success_after_base == success_before_base, (
        f"requests_success_total incremented with log_stats=False: "
        f"{success_before_base} -> {success_after_base}"
    )

    stats_runner = OmniRunner(model_dir, log_stats=True, collect_stage_metrics=True)
    try:
        assert stats_runner.omni.log_stats is True, "stats omni.log_stats must be True"
        assert stats_runner.omni.collect_stage_metrics is True, (
            "stats omni.collect_stage_metrics must be True"
        )
        assert stats_runner.omni.engine._log_stats is True, (
            "stats engine._log_stats must be True"
        )
        assert stats_runner.omni.engine._collect_stage_metrics is True, (
            "stats engine._collect_stage_metrics must be True"
        )
        stats_out = list(stats_runner.generate(_PROMPT, _sampling_params()))[0]
        stats_img = np.asarray(stats_out.images[0])
    finally:
        stats_runner.close()

    assert stats_out.peak_memory_mb > 0.0, (
        f"expected peak_memory_mb > 0.0, got {stats_out.peak_memory_mb}"
    )
    stage_0_gen_ms = stats_out.stage_durations["stage_0_gen_ms"]
    assert stage_0_gen_ms > 0.0, f"expected stage_0_gen_ms > 0.0, got {stage_0_gen_ms}"

    stage0 = stats_out.metrics["stage_metrics"]["0"]
    assert stage0["stage_gen_time_ms"] > 0.0, (
        f"expected stage_gen_time_ms > 0.0, got {stage0['stage_gen_time_ms']}"
    )
    assert stage0["image_pixels"] == _EXPECTED_IMAGE_PIXELS, (
        f"image_pixels {stage0['image_pixels']} != {_EXPECTED_IMAGE_PIXELS}"
    )
    assert stage0["denoise_step_latency_ms"] > 0.0, (
        f"expected denoise_step_latency_ms > 0.0, "
        f"got {stage0['denoise_step_latency_ms']}"
    )

    success_after_stats = _prom_metric_sum("vllm_omni:requests_success_total")
    assert success_after_stats - success_after_base == 1.0, (
        f"expected requests_success_total delta == 1.0, "
        f"got {success_after_stats - success_after_base}"
    )
    peak_mem_prom = _prom_metric_sum("vllm_omni:peak_memory_mb")
    assert peak_mem_prom > 0.0, (
        f"expected vllm_omni:peak_memory_mb > 0.0, got {peak_mem_prom}"
    )

    assert np.array_equal(stats_img, base_img), (
        "generated image pixels diverged with log_stats=True"
    )
