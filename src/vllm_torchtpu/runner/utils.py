# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for TPU runners."""

import datetime
import json
import os
from enum import Enum
from typing import Any, Optional

import torch.profiler
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from vllm_torchtpu import profiler_trace
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MIN_NUM_SEQS = 8

PREFILL_HEAVY_RATIO_THRESHOLD = 0.9
DECODE_HEAVY_RATIO_THRESHOLD = 0.2
BALANCED_RATIO_THRESHOLD = (0.4, 0.6)
PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR = 15
PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP = 0
PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD = -1

# Tuning knob with no `profiler_config` equivalent, so it stays in
# `additional_config`.
PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY = (
    "phased_profiler_decode_only_kv_len_threshold")

# Diagnostics-only knob: it steers profiling and never changes the compiled
# graph, so `_patch_vllm_config_hash_ignore_diagnostics` keeps it out of the
# TPU compile cache key.
HASH_IGNORED_ADDITIONAL_CONFIG_KEYS = frozenset({
    PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
})


class InferencePhase(Enum):
    PREFILL_HEAVY = 0
    DECODE_HEAVY = 1
    BALANCED = 2
    AMBIGUOUS = 3
    PREFILL_ONLY = 4
    DECODE_ONLY = 5


def determine_phase_from_batch_composition_stats(
    batch_composition_stats: dict[str, Any], ) -> InferencePhase:
    """Determines the inference phase based on the batch composition stats."""
    num_prefill_tokens = batch_composition_stats["num_prefill_tokens"]
    total_num_scheduled_tokens = batch_composition_stats[
        "total_num_scheduled_tokens"]
    prefill_ratio_for_batch = num_prefill_tokens / total_num_scheduled_tokens
    if prefill_ratio_for_batch == 1.0:
        return InferencePhase.PREFILL_ONLY
    if prefill_ratio_for_batch == 0.0:
        return InferencePhase.DECODE_ONLY
    if prefill_ratio_for_batch >= PREFILL_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.PREFILL_HEAVY
    if prefill_ratio_for_batch <= DECODE_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.DECODE_HEAVY
    if (prefill_ratio_for_batch >= BALANCED_RATIO_THRESHOLD[0]
            and prefill_ratio_for_batch <= BALANCED_RATIO_THRESHOLD[1]):
        return InferencePhase.BALANCED

    return InferencePhase.AMBIGUOUS


def get_batch_composition_stats(
    batch_id: int,
    input_batch:
    Any,  # Use Any to avoid circular dependency or import issues for now
    total_num_scheduled_tokens: int,
    num_reqs: int,
    padded_total_num_scheduled_tokens: int,
    scheduler_output: "VllmSchedulerOutput",
) -> dict:
    """Logs batch composition stats and returns them as a dict."""
    num_prefill_tokens = 0
    num_decode_tokens = 0

    # Get the number of scheduled tokens for each request.
    num_scheduled_tokens_per_req_list = []
    # Get the number of tokens already processed for each request.
    num_computed_tokens_per_req = input_batch.num_computed_tokens_cpu[:
                                                                      num_reqs]

    scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    min_kv_len = float("inf") if num_reqs > 0 else 0
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        assert req_id is not None

        # This is the number of tokens to process in the current step for this request
        num_scheduled_for_req = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens_per_req_list.append(num_scheduled_for_req)

        # This is the number of tokens already processed for this request (before this step)
        num_already_computed = int(
            num_computed_tokens_per_req[i])  # Cast from np.int32
        min_kv_len = min(min_kv_len, num_already_computed)

        # When speculative decoding is enabled for this request, the extra
        # tokens are draft tokens being verified, not chunked prefill tokens.
        num_spec_tokens = len(scheduled_spec_decode_tokens.get(req_id, ()))

        if num_already_computed == 0:
            # Prefill
            num_prefill_tokens += num_scheduled_for_req
        # This means the request is ongoing
        else:
            if num_spec_tokens > 0:
                # Verifying draft tokens for an ongoing request — count the
                # target token plus the draft tokens as decode.
                num_decode_tokens += num_scheduled_for_req
            elif num_scheduled_for_req > 1:
                # It's a multi-token request, so it's chunked prefill
                num_prefill_tokens += num_scheduled_for_req
            else:
                # It's a single token for an ongoing request, so it's decode
                num_decode_tokens += 1

    stats = {
        "batch_id": batch_id,
        "total_num_scheduled_tokens": total_num_scheduled_tokens,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "padded_total_num_scheduled_tokens": padded_total_num_scheduled_tokens,
        "num_reqs": num_reqs,
        "min_kv_len": min_kv_len if min_kv_len != float("inf") else 0,
    }
    stats["phase"] = determine_phase_from_batch_composition_stats(stats).name
    return stats


class PhaseBasedProfiler:
    """Implements a phase-based profiler for PyTorch/XLA, profiling different inference phases."""

    def __init__(
        self,
        profile_dir: str,
        worker_rank: int = 0,
        world_size: int = 1,
        num_steps_to_profile_for:
        int = PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR,
        num_decode_steps_to_skip:
        int = PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP,
        decode_kv_len_threshold:
        int = PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD,
    ):
        self.profiling_n_steps_left: int = 0
        self.profile_dir_with_phase_suffix: Optional[str] = None
        self.num_steps_to_profile_for: int = num_steps_to_profile_for
        # Skip initial decode steps to avoid profiling XLA compilation and warmup overhead
        self.num_decode_steps_to_skip: int = num_decode_steps_to_skip
        self.decode_steps_skipped: int = 0
        # Wait for KV cache to reach a certain length before profiling decode-only phase
        # to ensure we capture traces of longer, steady-state context lengths.
        self.decode_kv_len_threshold: int = decode_kv_len_threshold
        self.profile_dir: str = profile_dir
        self.inference_phase_seen: dict[InferencePhase, bool] = {
            InferencePhase.PREFILL_ONLY: False,
            InferencePhase.PREFILL_HEAVY: False,
            InferencePhase.DECODE_ONLY: False,
            InferencePhase.DECODE_HEAVY: False,
            InferencePhase.BALANCED: False,
        }
        self.current_phase: str = ""
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.profile_context: Optional[Any] = None
        self._canonical_dst_ts: Optional[str] = None

        logger.info(
            "Phase-based profiler enabled. Traces will be saved to: %s",
            self.profile_dir,
        )
        if self.num_decode_steps_to_skip > 0:
            logger.info(
                "Will skip %d decode-heavy steps before profiling decode_heavy phase.",
                self.num_decode_steps_to_skip,
            )
        if self.decode_kv_len_threshold >= 0:
            logger.info(
                "Will skip decode-only steps until min KV len >= %d.",
                self.decode_kv_len_threshold,
            )

    def _write_batch_composition_stats_to_file_helper(
            self, batch_composition_stats: dict) -> None:
        """Writes the batch composition stats to a file."""
        if not self.profile_dir_with_phase_suffix:
            return
        now = datetime.datetime.now()
        date_string_in_profiler_format = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

        stats_file = os.path.join(
            self.profile_dir_with_phase_suffix,
            f"batch_composition_stats_{date_string_in_profiler_format}.json",
        )
        try:
            with open(stats_file, "w") as f:
                f.write(json.dumps(batch_composition_stats) + "\n")
        except Exception as e:
            logger.warning("Failed to write batch composition stats: %s", e)

    def _resolve_canonical_dst_ts(self, phase_dir: str) -> str:
        """Resolve the canonical destination timestamp for this phase."""
        return profiler_trace.resolve_canonical_dst_ts(phase_dir,
                                                       self.worker_rank)

    def _start_profiling(self, batch_composition_stats: dict) -> None:
        """Starts profiling if the current phase is unseen."""
        current_determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats)
        for phase, has_been_seen in self.inference_phase_seen.items():
            if has_been_seen or phase != current_determined_phase:
                continue

            if (phase == InferencePhase.DECODE_HEAVY and
                    self.decode_steps_skipped < self.num_decode_steps_to_skip):
                self.decode_steps_skipped += 1
                logger.debug(
                    "Skipping decode-heavy step %d/%d before profiling.",
                    self.decode_steps_skipped,
                    self.num_decode_steps_to_skip,
                )
                break

            if (phase == InferencePhase.DECODE_ONLY
                    and self.decode_kv_len_threshold >= 0):
                min_kv_len = batch_composition_stats.get("min_kv_len", 0)
                if min_kv_len < self.decode_kv_len_threshold:
                    logger.debug(
                        "Skipping decode-only step as min KV len %d < threshold %d.",
                        min_kv_len,
                        self.decode_kv_len_threshold,
                    )
                    break

            self.inference_phase_seen[phase] = True
            self.profiling_n_steps_left = self.num_steps_to_profile_for
            self.current_phase = phase.name.lower()

            logger.info(f"Starting profiling for {self.current_phase} phase")
            logger.info(f"Batch composition stats: {batch_composition_stats}")
            phase_dir = os.path.join(self.profile_dir, self.current_phase)
            os.makedirs(phase_dir, exist_ok=True)

            self._canonical_dst_ts = self._resolve_canonical_dst_ts(phase_dir)
            self.profile_dir_with_phase_suffix = profiler_trace.rank_capture_dir(
                phase_dir, self.worker_rank)
            os.makedirs(self.profile_dir_with_phase_suffix, exist_ok=True)

            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)

            # Start PyTorch/XLA profiler
            handler = torch.profiler.tensorboard_trace_handler(
                dir_name=self.profile_dir_with_phase_suffix, use_gzip=True)
            config = TpuProfilerConfig(
                run_dir=self.profile_dir_with_phase_suffix,
                host_tracer_level=2,
                device_tracer_level=1,
                python_tracer_level=1,
            )
            self.profile_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=handler,
                experimental_config=config,
            )
            try:
                self.profile_context.__enter__()
            except Exception as e:
                logger.error("Failed to start PyTorch profiler: %s", e)
                self.profile_context = None
                self.current_phase = ""
                self.profiling_n_steps_left = 0
            break

    def _step_or_stop_profiling(self, batch_composition_stats: dict) -> None:
        """Steps or stops the profiler."""
        if self.current_phase != "":
            self._write_batch_composition_stats_to_file_helper(
                batch_composition_stats)
            self.profiling_n_steps_left -= 1
            if self.profiling_n_steps_left <= 0:
                self._finish_current_phase()

    def _finish_current_phase(self) -> None:
        """Closes out the phase capture in progress, if any."""
        if self.profile_context:
            try:
                self.profile_context.__exit__(None, None, None)
                logger.info("Stopped PyTorch profiler trace.")
            except Exception as e:
                logger.error("Failed to stop PyTorch profiler: %s", e)
            self.profile_context = None

        self._merge_profile_directories()
        logger.info(f"Profiling for {self.current_phase} phase finished")
        self.current_phase = ""
        self.profiling_n_steps_left = 0

    def finish(self) -> None:
        """Force-stops a phase capture in progress.

        Called when profiling is disarmed (worker.profile(is_start=False))
        before the current phase's step budget ran out.
        """
        if self.current_phase:
            self._finish_current_phase()

    def _merge_profile_directories(self) -> None:
        """Consolidates phase trace artifacts."""
        if not self.profile_dir_with_phase_suffix or not self._canonical_dst_ts:
            return
        phase_dir = os.path.dirname(self.profile_dir_with_phase_suffix)
        profiler_trace.merge_rank_capture(
            self.profile_dir_with_phase_suffix,
            phase_dir,
            self._canonical_dst_ts,
            self.worker_rank,
            world_size=self.world_size,
        )
        if self.worker_rank == 0:
            # The phase is merged, so nothing will read this marker again.
            # Dropping it keeps a later run against the same profile_dir from
            # finding it; slower ranks have long since timed out, since a
            # phase spans num_steps_to_profile_for steps and the poll window
            # is CANONICAL_TS_POLL_TIMEOUT_S.
            profiler_trace.clear_canonical_ts_marker(phase_dir)

    def step(self, batch_composition_stats: dict) -> None:
        """Steps the profiler and logs batch composition stats."""
        have_seen_all_phases = all(self.inference_phase_seen.values())
        # A batch with no work has no phase -- the classifier divides by
        # total_num_scheduled_tokens. execute_model already returns before
        # calling us in that case, so this only keeps the invariant local.
        #
        # The threshold must stay at 0, not 1. One request in decode schedules
        # exactly one token, so `> 1` skipped every steady-state step of a
        # batch-size-1 run: decode_only could never be captured, and because
        # this guard also covers _step_or_stop_profiling, the phase already
        # capturing never counted down. That left the capture open until
        # /stop_profile, ignoring max_iterations and filing every later step
        # under the phase that opened it.
        has_scheduled_work = (
            batch_composition_stats["total_num_scheduled_tokens"] > 0)
        if has_scheduled_work and (not have_seen_all_phases
                                   or self.current_phase != ""):
            if self.profiling_n_steps_left <= 0:
                self._start_profiling(batch_composition_stats)
            else:
                self._step_or_stop_profiling(batch_composition_stats)
