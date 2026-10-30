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
"""Utility functions and classes for TPU runners."""

import datetime
import json
import os
from collections.abc import Iterable
from enum import Enum
from typing import Any

import torch
import torch.profiler
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
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
PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD = -1

# Tuning knobs with no `profiler_config` equivalent, so they stay in
# `additional_config`.
PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY = (
    "phased_profiler_decode_only_kv_len_threshold"
)
PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY = (
    "phased_profiler_prefill_only_kv_len_threshold"
)
# Narrow what gets captured. All default to off, i.e. one capture per phase.
# A list of phase names (e.g. ["decode_only", "prefill_heavy"]): only these.
PHASED_PROFILER_ONLY_PHASES_KEY = "phased_profiler_only_phases"
# A list of request counts: one capture per phase at each, keyed by the
# engine's own num_reqs (per DP engine, not the global concurrency). Under DP +
# EP lockstep, by the largest num_reqs of any engine in the step (see
# encode_dp_profile_stats).
PHASED_PROFILER_TARGET_CONCURRENCY_KEY = "phased_profiler_target_concurrency"
# Key captures on (phase, num_reqs) and name them <phase>_concur_<num_reqs>.
# Implied by a target concurrency. On its own it is unbounded: every request
# count a ramp passes through gets its own capture.
PHASED_PROFILER_TRACK_CONCURRENCY_KEY = "phased_profiler_track_concurrency"
# A list of slice-global worker ranks that take the captures; default every
# rank. Under DP + EP every rank runs each step in lockstep, so one rank's
# trace shows the whole step, and sparing the others spares the slice their
# capture start/stop pauses, which every rank waits out in the next collective.
PHASED_PROFILER_RANKS_KEY = "phased_profiler_ranks"

# Diagnostics-only knobs: they steer profiling and never change the compiled
# graph, so `_patch_vllm_config_hash_ignore_diagnostics` keeps them out of the
# TPU compile cache key.
HASH_IGNORED_ADDITIONAL_CONFIG_KEYS = frozenset(
    {
        PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
        PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,
        PHASED_PROFILER_ONLY_PHASES_KEY,
        PHASED_PROFILER_TARGET_CONCURRENCY_KEY,
        PHASED_PROFILER_TRACK_CONCURRENCY_KEY,
        PHASED_PROFILER_RANKS_KEY,
    }
)


def config_list(value: Any) -> list[str]:
    """An additional_config list knob as strings: a JSON list, or one
    comma-separated string (the shape a shell-built --additional-config
    takes most easily). None and "" are empty."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


class InferencePhase(Enum):
    PREFILL_HEAVY = 0
    DECODE_HEAVY = 1
    BALANCED = 2
    AMBIGUOUS = 3
    PREFILL_ONLY = 4
    DECODE_ONLY = 5


def determine_dp_phase(
    prefill_permille_min: int, prefill_permille_max: int
) -> InferencePhase:
    """The phase of a DP-lockstep step from the busy ranks' prefill shares (in
    1/1000): a phase holds only if it holds on every busy rank."""
    if prefill_permille_max == 0:
        return InferencePhase.DECODE_ONLY
    if prefill_permille_min == 1000:
        return InferencePhase.PREFILL_ONLY
    if prefill_permille_min >= round(PREFILL_HEAVY_RATIO_THRESHOLD * 1000):
        return InferencePhase.PREFILL_HEAVY
    if prefill_permille_max <= round(DECODE_HEAVY_RATIO_THRESHOLD * 1000):
        return InferencePhase.DECODE_HEAVY
    if prefill_permille_min >= round(
        BALANCED_RATIO_THRESHOLD[0] * 1000
    ) and prefill_permille_max <= round(BALANCED_RATIO_THRESHOLD[1] * 1000):
        return InferencePhase.BALANCED
    return InferencePhase.AMBIGUOUS


def determine_phase_from_batch_composition_stats(
    batch_composition_stats: dict[str, Any],
) -> InferencePhase:
    """Determines the inference phase based on the batch composition stats."""
    if "dp_prefill_permille_min" in batch_composition_stats:
        return determine_dp_phase(
            batch_composition_stats["dp_prefill_permille_min"],
            batch_composition_stats["dp_prefill_permille_max"],
        )
    num_prefill_tokens = batch_composition_stats["num_prefill_tokens"]
    total_num_scheduled_tokens = batch_composition_stats["total_num_scheduled_tokens"]
    prefill_ratio_for_batch = num_prefill_tokens / total_num_scheduled_tokens
    if prefill_ratio_for_batch == 1.0:
        return InferencePhase.PREFILL_ONLY
    if prefill_ratio_for_batch == 0.0:
        return InferencePhase.DECODE_ONLY
    if prefill_ratio_for_batch >= PREFILL_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.PREFILL_HEAVY
    if prefill_ratio_for_batch <= DECODE_HEAVY_RATIO_THRESHOLD:
        return InferencePhase.DECODE_HEAVY
    if (
        prefill_ratio_for_batch >= BALANCED_RATIO_THRESHOLD[0]
        and prefill_ratio_for_batch <= BALANCED_RATIO_THRESHOLD[1]
    ):
        return InferencePhase.BALANCED

    return InferencePhase.AMBIGUOUS


def get_batch_composition_stats(
    batch_id: int,
    input_batch: Any,  # Use Any to avoid circular dependency or import issues for now
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
    num_computed_tokens_per_req = input_batch.num_computed_tokens_cpu[:num_reqs]

    scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    min_kv_len = float("inf") if num_reqs > 0 else 0
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        assert req_id is not None

        # This is the number of tokens to process in the current step for this request
        num_scheduled_for_req = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens_per_req_list.append(num_scheduled_for_req)

        # This is the number of tokens already processed for this request
        # (before this step)
        num_already_computed = int(num_computed_tokens_per_req[i])  # Cast from np.int32
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

    stats: dict[str, Any] = {
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


# Under DP + EP lockstep every rank runs each step padded to the DP-wide token
# bucket, and its MoE layers carry every rank's tokens; a rank's own batch says
# little about the step it runs. TPUModelRunner._dp_coordinated_step therefore
# all-reduces these fields with the step's token count, and the phased profiler
# classifies the step from the result, identically on every rank. All fields
# reduce with MAX; minima travel negated.
DP_PROFILE_STATS_LEN = 7
_DP_IDLE_MIN_KV_LEN = 1 << 62


def _prefill_permille(num_prefill_tokens: int, total_num_tokens: int) -> int:
    """A batch's prefill share in 1/1000, exact at 0 and 1."""
    if num_prefill_tokens <= 0:
        return 0
    if num_prefill_tokens >= total_num_tokens:
        return 1000
    return min(999, max(1, num_prefill_tokens * 1000 // total_num_tokens))


def encode_dp_profile_stats(local_stats: dict | None, armed: bool) -> list[int]:
    """This rank's part of the DP-wide batch stats.

    `local_stats` is get_batch_composition_stats' result, or None on a rank
    with no scheduled work, which contributes no tokens, no prefill and no KV
    length. A rank whose profiler is not armed blocks every capture start for
    the step: arming reaches the ranks at slightly different steps, and a
    capture must start on the same step everywhere.
    """
    blocked = int(not armed)
    if local_stats is None or local_stats["total_num_scheduled_tokens"] <= 0:
        return [blocked, 0, 0, -1000, 0, 0, -_DP_IDLE_MIN_KV_LEN]
    permille = _prefill_permille(
        int(local_stats["num_prefill_tokens"]),
        int(local_stats["total_num_scheduled_tokens"]),
    )
    num_reqs = int(local_stats["num_reqs"])
    return [
        blocked,
        int(local_stats["num_prefill_tokens"]),
        permille,
        -permille,
        num_reqs,
        -num_reqs,
        -int(local_stats["min_kv_len"]),
    ]


def decode_dp_profile_stats(values: Iterable[int]) -> dict[str, Any]:
    """The element-wise MAX of every rank's encode_dp_profile_stats."""
    blocked, prefill_max, permille_max, permille_min, reqs_max, reqs_min, kv_min = (
        int(v) for v in values
    )
    kv_min = -kv_min
    return {
        "dp_trigger_blocked": bool(blocked),
        "dp_num_prefill_tokens_max": prefill_max,
        "dp_prefill_permille_max": permille_max,
        "dp_prefill_permille_min": -permille_min,
        "dp_num_reqs_max": reqs_max,
        "dp_num_reqs_min": -reqs_min,
        "dp_min_kv_len": 0 if kv_min >= _DP_IDLE_MIN_KV_LEN else kv_min,
    }


def merge_dp_profile_stats(local_stats: dict, dp_stats: dict) -> dict:
    """The stats the profiler steps on under DP lockstep: the fields it
    triggers on (phase, num_reqs, min_kv_len, token counts) are the slice's,
    and the rank's own batch is kept under "local". Each captured step's stats
    file records both."""
    stats = {"batch_id": local_stats.get("batch_id"), **dp_stats}
    stats.update(
        {
            "num_reqs": dp_stats["dp_num_reqs_max"],
            "min_kv_len": dp_stats["dp_min_kv_len"],
            "num_prefill_tokens": dp_stats["dp_num_prefill_tokens_max"],
            "total_num_scheduled_tokens": dp_stats["dp_num_tokens_max"],
            "padded_total_num_scheduled_tokens": dp_stats["dp_padded_num_tokens"],
            "local": local_stats,
        }
    )
    stats["phase"] = determine_phase_from_batch_composition_stats(stats).name
    return stats


class PhaseBasedProfiler:
    """Implements a phase-based profiler for PyTorch/XLA, profiling different
    inference phases.
    """

    def __init__(
        self,
        profile_dir: str,
        worker_rank: int = 0,
        world_size: int = 1,
        num_steps_to_profile_for: int = PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR,
        num_decode_steps_to_skip: int = PHASED_PROFILER_NUM_DECODE_STEPS_TO_SKIP,
        decode_kv_len_threshold: int = PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD,
        prefill_kv_len_threshold: int = PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD,
        standard_opts: dict[str, Any] | None = None,
        advanced_opts: dict[str, Any] | None = None,
        only_phases: Iterable[str] = (),
        target_concurrencies: Iterable[int] = (),
        track_concurrency: bool = False,
        capture_ranks: Iterable[int] = (),
    ):
        self.profiling_n_steps_left: int = 0
        self.profile_dir_with_phase_suffix: str | None = None
        self.num_steps_to_profile_for: int = num_steps_to_profile_for
        # Skip initial decode steps to avoid profiling XLA compilation
        # and warmup overhead
        self.num_decode_steps_to_skip: int = num_decode_steps_to_skip
        self.decode_steps_skipped: int = 0
        # Wait for KV cache to reach a certain length before profiling decode-only phase
        # to ensure we capture traces of longer, steady-state context lengths.
        self.decode_kv_len_threshold: int = decode_kv_len_threshold
        # Same idea for chunked prefill: without this the capture always lands
        # on batch 1, where the sequence is one chunk long.
        self.prefill_kv_len_threshold: int = prefill_kv_len_threshold
        self.standard_opts = standard_opts or {}
        self.advanced_opts = advanced_opts or {}
        self.profile_dir: str = profile_dir
        self.inference_phase_seen: dict[InferencePhase, bool] = {
            InferencePhase.PREFILL_ONLY: False,
            InferencePhase.PREFILL_HEAVY: False,
            InferencePhase.DECODE_ONLY: False,
            InferencePhase.DECODE_HEAVY: False,
            InferencePhase.BALANCED: False,
        }
        # Narrowing filters (see the *_KEY constants above). Neither consumes a
        # step: a step they skip leaves the phase eligible for a later one.
        known_phases = {phase.name.lower() for phase in self.inference_phase_seen}
        self.only_phases: frozenset[str] = frozenset(
            phase.strip().lower() for phase in only_phases
        )
        if unknown := self.only_phases - known_phases:
            raise ValueError(
                f"{PHASED_PROFILER_ONLY_PHASES_KEY}: unknown phases "
                f"{sorted(unknown)}; choose from {sorted(known_phases)}."
            )
        self.target_concurrencies: frozenset[int] = frozenset(
            int(c) for c in target_concurrencies
        )
        if any(c <= 0 for c in self.target_concurrencies):
            raise ValueError(
                f"{PHASED_PROFILER_TARGET_CONCURRENCY_KEY} must be positive "
                f"request counts, got {sorted(self.target_concurrencies)}."
            )
        self.track_concurrency: bool = track_concurrency or bool(
            self.target_concurrencies
        )
        # (phase, num_reqs) pairs captured so far, when tracking concurrency.
        self.profiled_concurrencies: set[tuple[InferencePhase, int]] = set()
        self.current_phase: str = ""
        self.worker_rank = worker_rank
        self.world_size = world_size
        # Every rank still decides (and records what it has seen), so the ranks
        # stay in agreement; only these ranks start the profiler.
        self.capture_ranks: frozenset[int] = frozenset(int(r) for r in capture_ranks)
        self.captures_here: bool = (
            not self.capture_ranks or worker_rank in self.capture_ranks
        )
        self.profile_context: Any | None = None
        self._canonical_dst_ts: str | None = None

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
        if self.prefill_kv_len_threshold >= 0:
            logger.info(
                "Will skip prefill-only steps until min KV len >= %d.",
                self.prefill_kv_len_threshold,
            )
        if self.only_phases:
            logger.info(
                "Will only profile phases: %s.", ", ".join(sorted(self.only_phases))
            )
        if self.target_concurrencies:
            logger.info(
                "Will only profile at num_reqs %s.",
                ", ".join(map(str, sorted(self.target_concurrencies))),
            )
        if self.capture_ranks:
            logger.info(
                "Captures are taken on rank(s) %s%s.",
                ", ".join(map(str, sorted(self.capture_ranks))),
                "" if self.captures_here else f", not on this rank ({worker_rank})",
            )

    def _write_batch_composition_stats_to_file_helper(
        self, batch_composition_stats: dict
    ) -> None:
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
        return profiler_trace.resolve_canonical_dst_ts(phase_dir, self.worker_rank)

    def _start_profiling(self, batch_composition_stats: dict) -> None:
        """Starts profiling if the current phase is unseen."""
        if batch_composition_stats.get("dp_trigger_blocked"):
            return
        current_determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats
        )
        concurrency = (
            int(batch_composition_stats.get("num_reqs", 0))
            if self.track_concurrency
            else 0
        )
        for phase, has_been_seen in self.inference_phase_seen.items():
            if phase != current_determined_phase:
                continue
            if self.track_concurrency:
                if (phase, concurrency) in self.profiled_concurrencies:
                    continue
            elif has_been_seen:
                continue
            if self.only_phases and phase.name.lower() not in self.only_phases:
                continue
            if (
                self.target_concurrencies
                and concurrency not in self.target_concurrencies
            ):
                continue

            if (
                phase == InferencePhase.DECODE_HEAVY
                and self.decode_steps_skipped < self.num_decode_steps_to_skip
            ):
                self.decode_steps_skipped += 1
                logger.debug(
                    "Skipping decode-heavy step %d/%d before profiling.",
                    self.decode_steps_skipped,
                    self.num_decode_steps_to_skip,
                )
                break

            # `min_kv_len` is a batch minimum, so this only makes sense for the
            # single-composition phases. PREFILL_HEAVY always carries a
            # freshly-admitted request at kv len 0 and would never arm.
            kv_len_threshold = -1
            if phase == InferencePhase.DECODE_ONLY:
                kv_len_threshold = self.decode_kv_len_threshold
            elif phase == InferencePhase.PREFILL_ONLY:
                kv_len_threshold = self.prefill_kv_len_threshold
            if kv_len_threshold >= 0:
                min_kv_len = batch_composition_stats.get("min_kv_len", 0)
                if min_kv_len < kv_len_threshold:
                    logger.debug(
                        "Skipping %s step as min KV len %d < threshold %d.",
                        phase.name.lower(),
                        min_kv_len,
                        kv_len_threshold,
                    )
                    break

            self.inference_phase_seen[phase] = True
            self.profiling_n_steps_left = self.num_steps_to_profile_for
            self.current_phase = phase.name.lower()
            if self.track_concurrency:
                self.profiled_concurrencies.add((phase, concurrency))
                self.current_phase += f"_concur_{concurrency}"
            if not self.captures_here:
                logger.info(
                    "Phase %s reached; captured on rank(s) %s, not here.",
                    self.current_phase,
                    ", ".join(map(str, sorted(self.capture_ranks))),
                )
                self.current_phase = ""
                self.profiling_n_steps_left = 0
                break

            logger.info("Starting profiling for %s phase", self.current_phase)
            logger.info("Batch composition stats: %s", batch_composition_stats)
            phase_dir = os.path.join(self.profile_dir, self.current_phase)
            os.makedirs(phase_dir, exist_ok=True)

            self._canonical_dst_ts = self._resolve_canonical_dst_ts(phase_dir)
            self.profile_dir_with_phase_suffix = profiler_trace.rank_capture_dir(
                phase_dir, self.worker_rank
            )
            os.makedirs(self.profile_dir_with_phase_suffix, exist_ok=True)

            self._write_batch_composition_stats_to_file_helper(batch_composition_stats)

            # Start PyTorch/XLA profiler
            handler = torch.profiler.tensorboard_trace_handler(
                dir_name=self.profile_dir_with_phase_suffix, use_gzip=True
            )

            config = TpuProfilerConfig(
                run_dir=self.profile_dir_with_phase_suffix,
                host_tracer_level=self.standard_opts["host_tracer_level"],
                device_tracer_level=self.standard_opts["device_tracer_level"],
                python_tracer_level=self.standard_opts["python_tracer_level"],
                experimental_options=self.advanced_opts,
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
            self._write_batch_composition_stats_to_file_helper(batch_composition_stats)
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
        logger.info("Profiling for %s phase finished", self.current_phase)
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

    def _all_captures_done(self) -> bool:
        """Whether no step can start another capture."""
        wanted = [
            phase
            for phase in self.inference_phase_seen
            if not self.only_phases or phase.name.lower() in self.only_phases
        ]
        if self.target_concurrencies:
            return all(
                (phase, c) in self.profiled_concurrencies
                for phase in wanted
                for c in self.target_concurrencies
            )
        if self.track_concurrency:
            return False
        return all(self.inference_phase_seen[phase] for phase in wanted)

    def step(self, batch_composition_stats: dict) -> None:
        """Steps the profiler and logs batch composition stats."""
        have_seen_all_phases = self._all_captures_done()
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
        has_scheduled_work = batch_composition_stats["total_num_scheduled_tokens"] > 0
        if has_scheduled_work and (
            not have_seen_all_phases or self.current_phase != ""
        ):
            if self.profiling_n_steps_left <= 0:
                self._start_profiling(batch_composition_stats)
            else:
                self._step_or_stop_profiling(batch_composition_stats)


# (dtype, shape) -> minor_to_major, for buffers whose device layout is not
# XLA's default-for-shape and so must be pinned on the compiled entry too.
_PINNED_ENTRY_LAYOUTS: dict[tuple[torch.dtype, tuple[int, ...]], list[int]] = {}
_entry_pin_installed = False


def _to_row_major(tensor: torch.Tensor) -> torch.Tensor:
    """Return `tensor` re-materialized on its device with a `{1,0}` layout.

    `LayoutContext` hooks host->device transfers only -- a device-side
    allocation inside the context still gets the default layout -- so the
    tensor makes one round trip through the host. Shape and values are
    unchanged; only the physical layout differs.
    """
    from torch_tpu._internal.compile import tpu_torch_compile as ttc
    from torch_tpu._internal.device_utils.annotations import LayoutContext, TpuLayout

    # Keep the dtype's own tiling; only the dimension order changes.
    _, tiles, elem_bits = ttc.get_default_layout(tensor.dtype, tensor.shape)
    row_major = TpuLayout(
        minor_to_major=[1, 0], tiles=tiles, element_size_in_bits=elem_bits
    )
    host = tensor.cpu()
    with LayoutContext(row_major):
        device_tensor = host.to(tensor.device)
    _PINNED_ENTRY_LAYOUTS[(tensor.dtype, tuple(tensor.shape))] = [1, 0]
    return device_tensor


def relayout_rope_caches(model: torch.nn.Module):
    """Re-materialize every rope cache with a row-major device layout."""
    num_changed = 0
    for name, module in model.named_modules():
        if not isinstance(module, RotaryEmbeddingBase):
            continue
        # The buffers on a rotary module that are indexed by position.
        caches: list[tuple[str, Any]] = [
            (buf_name, getattr(module, buf_name, None))
            for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16")
        ]
        caches = [(n, b) for n, b in caches if isinstance(b, torch.Tensor)]
        if not caches:
            continue
        if any(b.ndim != 2 for _, b in caches):
            continue
        for buf_name, buf in caches:
            setattr(module, buf_name, _to_row_major(buf))
        num_changed += 1
        logger.info(
            "Relayouted rope cache %s to {1,0} (shape %s)",
            name,
            tuple(caches[0][1].shape),
        )

    if num_changed:
        _install_entry_layout_pin()
    logger.info("Row-major relayouted %d rope caches", num_changed)


def relayout_hash_tables(model: torch.nn.Module):
    """Re-materialize every MOE hash-routing table row-major."""
    seen: set[int] = set()
    num_changed = 0
    for name, module in model.named_modules():
        # The attributes under which one hash-routing table is reachable;
        # the same tensor object is bound to all of them.
        for attr in ("tid2eid", "hash_indices_table"):
            table: torch.Tensor | None = getattr(module, attr, None)
            if not isinstance(table, torch.Tensor) or id(table) in seen:
                continue
            seen.add(id(table))
            if table.ndim != 2:
                continue
            table.data = _to_row_major(table.data)
            num_changed += 1
            logger.info(
                "Relayouted hash table %s.%s to {1,0} (shape %s)",
                name,
                attr,
                tuple(table.shape),
            )

    if num_changed:
        _install_entry_layout_pin()
    logger.info("Row-major relayouted %d hash tables", num_changed)


def _install_entry_layout_pin() -> None:
    """Make the compiler pin entry layouts for buffers in the registry."""
    global _entry_pin_installed
    if _entry_pin_installed:
        return
    from torch_tpu._internal.compile import compiler as tt_compiler

    original_call = tt_compiler.StaticCompiler.__call__

    def _call_with_pinned_layouts(self, graph_module, example_inputs, *args, **kwargs):
        # `argument_layouts` is the 5th positional parameter; len(args) < 3
        # means the caller did not pass it positionally.
        if (
            _PINNED_ENTRY_LAYOUTS
            and len(args) < 3
            and kwargs.get("argument_layouts") is None
        ):
            # One entry per *tensor* argument: `fx_to_mlir` filters the
            # non-tensors (generators, symints) out before length-checking.
            layouts: list[list[int]] = []
            pinned_any = False
            for val in example_inputs:
                if not isinstance(val, torch.Tensor):
                    continue
                pin = _PINNED_ENTRY_LAYOUTS.get((val.dtype, tuple(val.shape)))
                layouts.append(list(pin) if pin is not None else [])
                pinned_any = pinned_any or pin is not None
            if pinned_any:
                kwargs["argument_layouts"] = layouts
                logger.info(
                    "Pinned %d of %d entry layouts for this graph",
                    sum(1 for entry in layouts if entry),
                    len(layouts),
                )
        return original_call(self, graph_module, example_inputs, *args, **kwargs)

    tt_compiler.StaticCompiler.__call__ = _call_with_pinned_layouts
    _entry_pin_installed = True
