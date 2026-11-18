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
"""TPU L0 test for Engine Abort Timeout (VLLM_OMNI_ABORT_TIMEOUT)."""

from __future__ import annotations

import asyncio
import runpy
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints import async_omni as async_omni_mod
from vllm_omni.entrypoints import async_omni_base
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

DEFAULT_ABORT_TIMEOUT_S = 2.0
ABORT_TIMEOUT_OVERRIDE_S = 0.05
MAX_CANCEL_ELAPSED_S = 0.5


@dataclass
class _WedgedAbortEngine:
    """Engine stub whose ``abort_async`` never resolves, exercising timeout bound."""

    submitted: asyncio.Event
    abort_timeouts: list[float | None] = field(default_factory=list)
    aborted_ids: list[str] = field(default_factory=list)
    num_stages: int = 1
    stage_configs: list[object] = field(default_factory=list)

    async def add_request_async(self, **kwargs: Any) -> None:
        del kwargs
        self.submitted.set()

    async def abort_async(
        self,
        request_ids: list[str],
        timeout: float | None = None,
    ) -> list[object]:
        self.aborted_ids.extend(request_ids)
        self.abort_timeouts.append(timeout)
        await asyncio.Event().wait()
        return []


def _make_wedged_async_omni(engine: _WedgedAbortEngine) -> AsyncOmni:
    """Construct a real ``AsyncOmni`` via ``object.__new__`` around ``engine``."""
    omni = object.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni._admitting = 0
    omni._sleeping_tags = set()
    omni._interaction_reservations = {}
    omni.engine = engine  # type: ignore[assignment]
    omni.log_stats = False
    omni.request_states = {}
    omni._final_output_handler = lambda: None  # type: ignore[method-assign]
    omni.resolve_sampling_params_list = (  # type: ignore[method-assign]
        lambda params, allow_delta_coercion: params
    )
    omni._compute_final_stage_id = lambda output_modalities: 0  # type: ignore[method-assign]
    omni._compute_final_output_stage_ids = (  # type: ignore[method-assign]
        lambda output_modalities: [0]
    )

    async def _park_results(
        request_id: str,
        metrics: Any,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
    ) -> AsyncGenerator[Any, None]:
        del request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts
        await asyncio.Event().wait()
        yield  # pragma: no cover

    omni._process_orchestrator_results = _park_results  # type: ignore[method-assign]
    omni._log_summary_and_cleanup = (  # type: ignore[method-assign]
        lambda request_id: omni.request_states.pop(request_id, None)
    )
    return omni


def test_engine_abort_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine Abort Timeout.

    - TPU path: with ``OmniTpuPlatform`` active, sets
      ``VLLM_OMNI_ABORT_TIMEOUT=0.05``, re-evaluates
      ``async_omni_base.ABORT_TIMEOUT_S``, and cancels an in-flight
      ``AsyncOmni.generate(...)`` call backed by a wedged ``abort_async``.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``async_omni_base.ABORT_TIMEOUT_S == 0.05`` (vs the 2.0 s default).
    - Outcome check: cancellation finishes in under 0.5 s (well below the 2.0 s
      default reference), forwards ``timeout=0.05`` to ``engine.abort_async``,
      and clears ``omni.request_states``.
    """
    assert current_omni_platform.device_type == "tpu", (
        "Expected OmniTpuPlatform with device_type='tpu', got "
        f"{current_omni_platform.device_type!r}"
    )

    assert async_omni_base.__file__ is not None, "async_omni_base.__file__ is None"
    monkeypatch.delenv("VLLM_OMNI_ABORT_TIMEOUT", raising=False)
    default_ns = runpy.run_path(async_omni_base.__file__)
    assert default_ns["ABORT_TIMEOUT_S"] == DEFAULT_ABORT_TIMEOUT_S, (
        f"Default ABORT_TIMEOUT_S {default_ns['ABORT_TIMEOUT_S']} != "
        f"{DEFAULT_ABORT_TIMEOUT_S}"
    )

    monkeypatch.setenv("VLLM_OMNI_ABORT_TIMEOUT", str(ABORT_TIMEOUT_OVERRIDE_S))
    override_ns = runpy.run_path(async_omni_base.__file__)
    configured_timeout = float(override_ns["ABORT_TIMEOUT_S"])
    monkeypatch.setattr(async_omni_base, "ABORT_TIMEOUT_S", configured_timeout)
    monkeypatch.setattr(async_omni_mod, "ABORT_TIMEOUT_S", configured_timeout)
    assert async_omni_base.ABORT_TIMEOUT_S == ABORT_TIMEOUT_OVERRIDE_S, (
        f"Configured ABORT_TIMEOUT_S {async_omni_base.ABORT_TIMEOUT_S} != "
        f"{ABORT_TIMEOUT_OVERRIDE_S}"
    )

    async def _exercise_cancelled_generate() -> tuple[
        _WedgedAbortEngine, AsyncOmni, float
    ]:
        submitted = asyncio.Event()
        engine = _WedgedAbortEngine(submitted=submitted)
        omni = _make_wedged_async_omni(engine)

        async def _consume() -> None:
            async for _ in omni.generate(
                prompt={"prompt": "abort-timeout-test"},
                request_id="req-abort-timeout",
                sampling_params_list=[SamplingParams()],
                output_modalities=["image"],
            ):
                pass

        task = asyncio.create_task(_consume())
        await submitted.wait()
        start_t = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        elapsed_s = time.monotonic() - start_t
        return engine, omni, elapsed_s

    engine, omni, elapsed_s = asyncio.run(_exercise_cancelled_generate())
    assert len(engine.aborted_ids) == 1, (
        f"Expected 1 aborted internal request id, got {engine.aborted_ids}"
    )
    assert engine.aborted_ids[0].startswith("req-abort-timeout-"), (
        f"Aborted internal id {engine.aborted_ids[0]!r} missing request prefix"
    )
    assert engine.abort_timeouts == [ABORT_TIMEOUT_OVERRIDE_S], (
        f"abort_async timeouts {engine.abort_timeouts} != [{ABORT_TIMEOUT_OVERRIDE_S}]"
    )
    assert elapsed_s < MAX_CANCEL_ELAPSED_S, (
        f"Cancellation took {elapsed_s:.3f}s, exceeding {MAX_CANCEL_ELAPSED_S}s "
        f"(default reference timeout is {DEFAULT_ABORT_TIMEOUT_S}s)"
    )
    assert omni.request_states == {}, (
        f"Cancelled request_states not cleaned up: {list(omni.request_states)}"
    )
