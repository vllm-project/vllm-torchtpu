# SPDX-License-Identifier: Apache-2.0
"""Utility functions and classes for TPU runners."""

import datetime
import json
import os
import types
from enum import Enum
from typing import Any

import torch
import torch.profiler
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm.model_executor.layers.rotary_embedding.base import (
    ApplyRotaryEmb,
    RotaryEmbedding,
    RotaryEmbeddingBase,
)
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

# Diagnostics-only knob: it steers profiling and never changes the compiled
# graph, so `_patch_vllm_config_hash_ignore_diagnostics` keeps it out of the
# TPU compile cache key.
HASH_IGNORED_ADDITIONAL_CONFIG_KEYS = frozenset(
    {
        PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
        PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,
    }
)


class InferencePhase(Enum):
    PREFILL_HEAVY = 0
    DECODE_HEAVY = 1
    BALANCED = 2
    AMBIGUOUS = 3
    PREFILL_ONLY = 4
    DECODE_ONLY = 5


def determine_phase_from_batch_composition_stats(
    batch_composition_stats: dict[str, Any],
) -> InferencePhase:
    """Determines the inference phase based on the batch composition stats."""
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
        self.current_phase: str = ""
        self.worker_rank = worker_rank
        self.world_size = world_size
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
        current_determined_phase = determine_phase_from_batch_composition_stats(
            batch_composition_stats
        )
        for phase, has_been_seen in self.inference_phase_seen.items():
            if has_been_seen or phase != current_determined_phase:
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
        caches = [
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


def _in_graph_rope_eligible(module: torch.nn.Module) -> bool:
    """Whether `module`'s cos/sin values depend only on its `inv_freq`.

    Holds for `RotaryEmbedding` and subclasses that override only
    `_compute_inv_freq` (e.g. Llama3, NTK, Gemma4). Excludes M-RoPE (2-D
    positions), variants whose cache scales positions or values (YaRN mscale,
    linear and dynamic-NTK scaling), and variants with their own forward
    (DeepSeek, Fourier). Those keep their `cos_sin_cache`.
    """
    cls = type(module)
    return (
        isinstance(module, RotaryEmbedding)
        and cls.forward_native is RotaryEmbedding.forward_native
        and cls._compute_cos_sin_cache is RotaryEmbeddingBase._compute_cos_sin_cache
    )


# TODO: Delete the cache comparison, the correction table, the correction limit
# and the inv_freq candidates below together with the cached RoPE path. They only
# keep the in-graph values bitwise equal to the cache; the remaining differences
# are ordinary floating-point rounding.
#
# Limit on the positions whose computed cos/sin row may differ from the cache:
# max(_IN_GRAPH_ROPE_MIN_CORRECTION_LIMIT, fraction * num_positions). Device
# sin/cos rounding differs in well under 1% of rows, but a CPU-built cache can
# also carry less accurate cos values in whole per-thread row blocks (observed
# up to 1/8 of the rows). A cache that is not cos/sin(positions * inv_freq)
# differs in nearly all rows. Corrections are applied with a
# [num_tokens, num_corrections] comparison.
_IN_GRAPH_ROPE_MAX_CORRECTION_FRACTION = 0.25
_IN_GRAPH_ROPE_MIN_CORRECTION_LIMIT = 64
# Positions per call when comparing computed cos/sin against the cache at load.
_IN_GRAPH_ROPE_VERIFY_CHUNK = 32768


def _in_graph_cos_sin(
    positions: torch.Tensor,
    inv_freq: tuple[float, ...],
    cache_dtype: torch.dtype,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns `[cos | sin](positions * inv_freq)` rows in `dtype`.

    Rounds through `cache_dtype` first, as the cached path stores the table in
    `cache_dtype` and converts it to the query dtype on use.
    """
    # A Python-constant table traces as a graph constant, not an input.
    inv = torch.tensor(inv_freq, dtype=torch.float32, device=positions.device)
    freqs = torch.outer(positions.to(torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(cache_dtype).to(dtype)


def _in_graph_rope_forward(
    self: Any,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """`RotaryEmbedding.forward_native` computing cos/sin from positions."""
    if query.dtype != self._in_graph_rope_dtype:
        raise ValueError(
            f"In-graph RoPE was prepared for {self._in_graph_rope_dtype} "
            f"queries, got {query.dtype}"
        )
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = _in_graph_cos_sin(
        positions,
        self._in_graph_rope_inv_freq,
        self._in_graph_rope_cache_dtype,
        self._in_graph_rope_dtype,
    )
    if self._in_graph_rope_fix_positions:
        # Rows where device sin/cos rounds differently from the cache.
        fix_positions = torch.tensor(
            self._in_graph_rope_fix_positions,
            dtype=positions.dtype,
            device=positions.device,
        )
        fix_rows = torch.tensor(
            self._in_graph_rope_fix_rows,
            dtype=self._in_graph_rope_dtype,
            device=positions.device,
        )
        match = positions[:, None] == fix_positions[None, :]
        fixed = fix_rows[match.to(torch.int32).argmax(dim=-1)]
        cos_sin = torch.where(match.any(dim=-1, keepdim=True), fixed, cos_sin)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = ApplyRotaryEmb.forward_static(
        query[..., : self.rotary_dim], cos, sin, self.is_neox_style
    )
    query = torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1).reshape(
        query_shape
    )
    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = ApplyRotaryEmb.forward_static(
            key[..., : self.rotary_dim], cos, sin, self.is_neox_style
        )
        key = torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1).reshape(
            key_shape
        )
    return query, key


def _in_graph_rope_mismatched_positions(
    cache: torch.Tensor, inv_freq: tuple[float, ...], dtype: torch.dtype
) -> list[int]:
    """Positions whose `_in_graph_cos_sin` row differs bitwise from `cache`.

    Both sides are compared after conversion to `dtype`, the dtype the forward
    runs in. Runs `_in_graph_cos_sin` on the cache's device (compiled on TPU),
    so the result reflects that device's sin/cos rounding.
    """
    table_fn = _in_graph_cos_sin
    if cache.device.type == "tpu":
        table_fn = torch.compile(_in_graph_cos_sin, backend="tpu", dynamic=False)
    int_dtype = {2: torch.int16, 4: torch.int32}[
        torch.empty((), dtype=dtype).element_size()
    ]
    num_positions = cache.shape[0]
    chunk = min(num_positions, _IN_GRAPH_ROPE_VERIFY_CHUNK)
    mismatched: list[int] = []
    for start in range(0, num_positions, chunk):
        # Fixed-size chunks (the last one clamped) keep a single compiled shape.
        positions = torch.arange(start, start + chunk, device=cache.device).clamp_(
            max=num_positions - 1
        )
        computed = table_fn(positions, inv_freq, cache.dtype, dtype)
        expected = cache[positions].to(dtype)
        differs = (
            (computed.view(int_dtype) != expected.view(int_dtype)).any(dim=-1).cpu()
        )
        mismatched.extend(sorted(set(positions.cpu()[differs].tolist())))
    return sorted(set(mismatched))


def _in_graph_rope_inv_freq_candidates(
    module: torch.nn.Module, device: torch.device
) -> list[tuple[float, ...]]:
    """`inv_freq` as computed on CPU and on `device`, without duplicates.

    vLLM's model loader builds modules under the target device, so the cache
    may come from a device `pow`, which can differ from CPU `pow` by an ulp.
    """
    candidates = [tuple(module._compute_inv_freq(module.base).float().tolist())]
    if device.type != "cpu":
        with torch.device(device):
            on_device = module._compute_inv_freq(module.base).float().cpu()
        candidates.append(tuple(on_device.tolist()))
    return list(dict.fromkeys(candidates))


def patch_in_graph_rotary_embedding(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """Computes RoPE cos/sin in-graph for the eligible rotary modules of `model`.

    Each eligible module drops its `cos_sin_cache` buffers and gets an
    instance-level forward that evaluates cos/sin of `positions * inv_freq` for
    `dtype` queries. The result is bitwise equal to the cached path: at patch
    time the computed rows are compared with the cache, both converted to
    `dtype`, on the cache's device, and the few rows where that device's
    sin/cos rounds differently are kept as a constant correction table. The
    vLLM rotary classes and other models are not modified. Raises if too many
    rows differ, i.e. the module's cache is not cos/sin(positions * inv_freq).

    Returns the number of patched modules.
    """
    num_patched = 0
    for name, module in model.named_modules():
        if not _in_graph_rope_eligible(module):
            continue
        cache = module.cos_sin_cache.detach()
        # Use whichever inv_freq the cache was built from.
        inv_freq_values, fix_positions = min(
            (
                (
                    candidate,
                    _in_graph_rope_mismatched_positions(cache, candidate, dtype),
                )
                for candidate in _in_graph_rope_inv_freq_candidates(
                    module, cache.device
                )
            ),
            key=lambda pair: len(pair[1]),
        )
        limit = max(
            _IN_GRAPH_ROPE_MIN_CORRECTION_LIMIT,
            int(_IN_GRAPH_ROPE_MAX_CORRECTION_FRACTION * cache.shape[0]),
        )
        if len(fix_positions) > limit:
            raise RuntimeError(
                f"Rotary module {name!r} ({type(module).__name__}): "
                f"{len(fix_positions)} of {cache.shape[0]} computed cos/sin rows "
                f"differ from the cache (limit {limit}); refusing to compute it "
                "in-graph"
            )
        fix_rows = cache[torch.tensor(fix_positions, dtype=torch.long)].to(dtype)
        module._in_graph_rope_inv_freq = inv_freq_values
        module._in_graph_rope_cache_dtype = cache.dtype
        module._in_graph_rope_dtype = dtype
        module._in_graph_rope_fix_positions = tuple(fix_positions)
        module._in_graph_rope_fix_rows = tuple(
            tuple(row) for row in fix_rows.to("cpu", torch.float32).tolist()
        )
        bound = types.MethodType(_in_graph_rope_forward, module)
        module.forward_native = bound
        module._forward_method = bound
        for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
            module._buffers.pop(buf_name, None)
            module.__dict__.pop(buf_name, None)
        logger.info(
            "Computing RoPE in-graph for %s with %d corrected positions",
            name,
            len(fix_positions),
        )
        num_patched += 1
    logger.info("Computing RoPE in-graph for %d rotary modules", num_patched)
    return num_patched


def relayout_hash_tables(model: torch.nn.Module):
    """Re-materialize every MOE hash-routing table row-major."""
    seen: set[int] = set()
    num_changed = 0
    for name, module in model.named_modules():
        # The attributes under which one hash-routing table is reachable;
        # the same tensor object is bound to all of them.
        for attr in ("tid2eid", "hash_indices_table"):
            table = getattr(module, attr, None)
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
