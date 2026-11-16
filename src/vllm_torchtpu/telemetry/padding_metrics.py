# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
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
"""TPU static bucket padding runner hook and Prometheus collector."""

import functools
import sys
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class _AgenticTpuPaddingCollector:
    """Prometheus collector exposing server-level TPU round padding counters."""

    _name = "agentic_tpu_padding"

    def __init__(self, service: Any = None) -> None:
        self._service = service

    def collect(self) -> Any:
        from prometheus_client.core import CounterMetricFamily

        from vllm_torchtpu.telemetry.vllm_hooks import get_telemetry_service

        svc = self._service or get_telemetry_service()
        counts = svc.sync_padding_state()
        rep = str(svc.replica_id)

        rounds_metric = CounterMetricFamily(
            "vllm:tpu_padding_rounds_total",
            "Total TPU model execution rounds by phase (prompt vs reply)",
            labels=["phase", "replica_id"],
        )
        rounds_metric.add_metric(["prompt", rep], float(counts["prompt_rounds"]))
        rounds_metric.add_metric(["reply", rep], float(counts["reply_rounds"]))
        yield rounds_metric

        tokens_metric = CounterMetricFamily(
            "vllm:tpu_padding_tokens_total",
            "Total TPU scheduled real vs static bucket filler tokens by phase",
            labels=["phase", "kind", "replica_id"],
        )
        tokens_metric.add_metric(
            ["prompt", "real", rep], float(counts["prompt_real_tokens"])
        )
        tokens_metric.add_metric(
            ["prompt", "filler", rep], float(counts["prompt_filler_tokens"])
        )
        tokens_metric.add_metric(
            ["reply", "real", rep], float(counts["reply_real_tokens"])
        )
        tokens_metric.add_metric(
            ["reply", "filler", rep], float(counts["reply_filler_tokens"])
        )
        yield tokens_metric


def _register_padding_collector_on_registry(registry: Any, service: Any = None) -> None:
    if getattr(registry, "_agentic_padding_collector_registered", False):
        return
    try:
        registry.register(_AgenticTpuPaddingCollector(service=service))
        registry._agentic_padding_collector_registered = True
    except ValueError:
        pass


def _is_padding_sync_writer(runner: Any) -> bool:
    """Return True only for TPxPP rank 0 within each DP engine."""
    return int(runner.parallel_config.rank) == 0


def _runner_dp_rank(runner: Any) -> int:
    return max(0, int(runner.parallel_config.data_parallel_rank))


def _runner_dp_size(runner: Any) -> int:
    return max(1, int(runner.parallel_config.data_parallel_size))


def _resolve_active_layout_plan(runner: Any) -> Any:
    """Return sequence layout plan or PCP sharding plan, or `None` if neither ran."""
    if runner._last_sequence_layout_plan is not None:
        return runner._last_sequence_layout_plan
    if runner.pcp_manager is not None:
        return runner.pcp_manager.sharding_plan
    return None


def _is_prompt_chunk(
    *,
    end_index: int,
    num_decode_reqs: int,
    num_windowed_reqs: int | None,
) -> bool:
    """Classify chunk as prompt vs reply (`[decode][spec-verify][prefill/mixed]`)."""
    reply_end = num_decode_reqs if num_windowed_reqs is None else num_windowed_reqs
    return end_index > reply_end


def patch_tpu_padding_metrics(
    runner_cls: Any = None,
    service: Any = None,
    *,
    patch_prometheus: bool | None = None,
    prometheus_registry: Any = None,
    multiprocess_cls: Any = None,
) -> None:
    """Hook `TPUModelRunner` and `/metrics` for TPU static bucket padding."""
    should_patch_prom = (
        (runner_cls is None) if patch_prometheus is None else patch_prometheus
    )
    if runner_cls is None:
        tpu_runner_mod = sys.modules.get("vllm_torchtpu.runner.tpu_runner")
        if tpu_runner_mod is not None:
            runner_cls = tpu_runner_mod.TPUModelRunner

    if runner_cls is not None and not runner_cls.__dict__.get(
        "_tpu_agentic_padding_patch", False
    ):
        orig_capture_model = runner_cls.capture_model
        orig_prepare_inputs = runner_cls._prepare_inputs
        runner_cls._agentic_padding_ready = False

        @functools.wraps(orig_capture_model)
        def _wrapped_capture_model(self: Any, *args: Any, **kwargs: Any) -> Any:
            self._agentic_padding_ready = False
            try:
                return orig_capture_model(self, *args, **kwargs)
            finally:
                self._agentic_padding_ready = True
                try:
                    if _is_padding_sync_writer(self):
                        from vllm_torchtpu.telemetry.vllm_hooks import (
                            get_telemetry_service,
                        )

                        svc = service or get_telemetry_service()
                        svc.init_padding_sync_writer(
                            dp_rank=_runner_dp_rank(self),
                            dp_size=_runner_dp_size(self),
                        )
                except Exception as exc:
                    logger.warning_once(
                        "Agentic telemetry padding writer init failed: %s", exc
                    )

        @functools.wraps(orig_prepare_inputs)
        def _wrapped_prepare_inputs(
            self: Any,
            scheduler_output: Any,
            start_index: int,
            num_decode_reqs: int,
            num_windowed_reqs: int | None = None,
        ) -> Any:
            res = orig_prepare_inputs(
                self,
                scheduler_output,
                start_index,
                num_decode_reqs,
                num_windowed_reqs,
            )
            if not self._agentic_padding_ready:
                return res
            try:
                plan = _resolve_active_layout_plan(self)
                if plan is None:
                    return res
                real_tokens = int(plan.global_num_tokens)
                padded_tokens = int(plan.global_padded_num_tokens)
                if real_tokens <= 0 or padded_tokens <= 0:
                    return res

                from vllm_torchtpu.telemetry.vllm_hooks import (
                    get_telemetry_service,
                )

                end_index = int(res[4])
                svc = service or get_telemetry_service()
                svc.on_tpu_round_padded(
                    real_tokens,
                    padded_tokens,
                    is_prompt_round=_is_prompt_chunk(
                        end_index=end_index,
                        num_decode_reqs=num_decode_reqs,
                        num_windowed_reqs=num_windowed_reqs,
                    ),
                    sync_ipc=_is_padding_sync_writer(self),
                    dp_rank=_runner_dp_rank(self),
                    dp_size=_runner_dp_size(self),
                )
            except Exception as exc:
                logger.warning_once(
                    "Agentic telemetry TPU round padding hook failed: %s", exc
                )
            return res

        runner_cls.capture_model = _wrapped_capture_model
        runner_cls._prepare_inputs = _wrapped_prepare_inputs
        runner_cls._tpu_agentic_padding_patch = True

    if not should_patch_prom:
        return

    reg_target = prometheus_registry
    mp_target = multiprocess_cls
    try:
        if reg_target is None or mp_target is None:
            from prometheus_client import REGISTRY, multiprocess

            if reg_target is None:
                reg_target = REGISTRY
            if mp_target is None:
                mp_target = multiprocess.MultiProcessCollector

        _register_padding_collector_on_registry(reg_target, service=service)
        if not mp_target.__dict__.get("_tpu_agentic_mp_patch", False):
            orig_mp_collect = mp_target.collect

            @functools.wraps(orig_mp_collect)
            def _wrapped_mp_collect(self: Any) -> Any:
                yield from orig_mp_collect(self)
                yield from _AgenticTpuPaddingCollector(service=service).collect()

            mp_target.collect = _wrapped_mp_collect
            mp_target._tpu_agentic_mp_patch = True
    except Exception as exc:
        logger.warning_once(
            "Skipping prometheus_client padding collector patch: %s", exc
        )
