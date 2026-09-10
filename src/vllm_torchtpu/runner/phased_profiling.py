# SPDX-License-Identifier: Apache-2.0
from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.tracing.options import resolve_profile_dir_and_opts

logger = init_logger(__name__)


class PhasedProfilingMixin:
    """Phase-based profiler control."""

    def _init_phased_profiling(self) -> None:
        """Resolves whether USE_PHASED_PROFILER is set and where its traces go.

        The trace directory comes from `profiler_config`, the same field the
        standard torch profiler uses; the env var only selects which profiler
        consumes it. The profiler itself is armed later by
        `start_phased_profiling`/`stop_phased_profiling`, called from
        TPUWorker.profile() so both profilers share one trigger.
        """
        profiler_config = self.vllm_config.profiler_config
        self.phased_profiling_dir = (profiler_config.torch_profiler_dir
                                     if envs.USE_PHASED_PROFILER else "")
        self.phase_based_profiler = None

    def start_phased_profiling(self,
                               profile_prefix: str | None = None) -> None:
        """Arms the phase-based profiler. Called from TPUWorker.profile()."""
        if not self.phased_profiling_dir:
            logger.warning(
                "Phased profiling directory is not set. Skipping profiling.")
            return
        if self.phase_based_profiler is not None:
            logger.warning(
                "Phased profiler is already running. Ignoring start request.")
            return
        profiler_config = self.vllm_config.profiler_config
        additional_config = self.vllm_config.additional_config
        decode_kv_len_threshold = additional_config.get(
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD)
        prefill_kv_len_threshold = additional_config.get(
            runner_utils.PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,
            runner_utils.PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD)
        # Same scoping as the standard profiler: the prefix names the run, and
        # the per-phase subdirectories sit beneath it.

        profile_dir, standard_opts, advanced_opts = resolve_profile_dir_and_opts(
            self.phased_profiling_dir, profile_prefix)

        # Deliberately not read from parallel_config: its rank is TPxPP-scoped,
        # so every DP replica would call itself rank 0 and their traces would
        # overwrite each other on merge. The worker resolves the slice-global
        # rank from the TPU rank binding and passes it in.
        self.phase_based_profiler = runner_utils.PhaseBasedProfiler(
            profile_dir,
            worker_rank=self._profiler_rank,
            world_size=self._profiler_world_size,
            # max_iterations defaults to 0 ("no limit") for standard torch
            # profiling; that's meaningless for the phased profiler, so fall
            # back to its own default when unset.
            num_steps_to_profile_for=(
                profiler_config.max_iterations
                or runner_utils.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR),
            num_decode_steps_to_skip=profiler_config.delay_iterations,
            decode_kv_len_threshold=decode_kv_len_threshold,
            prefill_kv_len_threshold=prefill_kv_len_threshold,
            standard_opts=standard_opts,
            advanced_opts=advanced_opts,
        )

    def stop_phased_profiling(self) -> None:
        """Disarms the phase-based profiler. Called from TPUWorker.profile()."""
        if self.phase_based_profiler is None:
            logger.warning(
                "Phased profiler is not running. Ignoring stop request.")
            return
        self.phase_based_profiler.finish()
        self.phase_based_profiler = None
