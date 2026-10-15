# SPDX-License-Identifier: Apache-2.0
"""TPU device profiler for vLLM-Omni diffusion workers.

vLLM-Omni's ``OmniTorchProfilerWrapper`` only records CUDA-family device
activities. This subclass records the TPU (``PrivateUse1``) activity through
the torch_tpu profiler, which writes an XSpace (``*.xplane.pb``, for xprof /
TensorBoard). The XSpace and the chrome trace Omni exports land in sibling
directories under ``torch_profiler_dir``:

- chrome trace, in Omni's per-capture session directory:
  ``<torch_profiler_dir>/<YYYYmmdd-HHMMSS>_<prefix>/trace_rank<N>.json.gz``
- XSpace, in a run root shared by all ranks:
  ``<torch_profiler_dir>/<prefix>/plugins/profile/<canonical_ts>/``

The run root is keyed on the profile prefix rather than the session directory
because each rank picks its session timestamp independently, and ranks can be
a second apart; keying on the prefix lets every rank merge into one
``<canonical_ts>`` run.

Enable it the usual Omni way (``profiler_config={"profiler": "torch",
"torch_profiler_dir": ...}``) and drive it with ``start_profile()`` /
``stop_profile()``.
"""

import dataclasses
import os
from typing import Any

import torch
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper

from vllm_torchtpu import profiler_trace
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.options import resolve_profile_dir_and_opts

logger = init_logger(__name__)


class TpuOmniTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """Omni torch profiler wrapper that captures TPU device activity."""

    def __init__(
        self,
        profiler_config: Any,
        worker_name: str,
        local_rank: int,
        activities: list[Any] | None = None,
    ):
        self._tpu_python_tracer = bool(profiler_config.torch_profiler_with_stack)
        if self._tpu_python_tracer:
            logger.info(
                "torch_profiler_with_stack: recording Python stacks with "
                "the TPU python tracer, enforcing torch.profiler with_stack=False."
            )
        profiler_config = dataclasses.replace(
            profiler_config, torch_profiler_with_stack=False
        )
        super().__init__(profiler_config, worker_name, local_rank, activities)
        self._profile_session_index = 0
        self._capture_state: tuple[str, str, str, int, int, str] | None = None

    def _get_default_activities(self) -> list[Any]:
        # "TPU" is a placeholder: Omni's activity map has no TPU key, so the real
        # PrivateUse1 activity is added in _start. Two entries also keep the base
        # class's CPU-only dump_cpu_time_total off.
        # TODO: Override _resolve_memory_history_backend() once torch.tpu exposes
        # _record_memory_history/_dump_snapshot; the base (None, None) no-op is
        # correct until then.
        return ["CPU", "TPU"]

    def _create_profiler(self, profiler_config: Any, activities: list[Any]) -> None:
        logger.info(
            "TpuOmniTorchProfilerWrapper: Deferring torch.profiler construction "
            "to _start() because the TPU tracer binds its output directory."
        )
        return None

    def _start(self) -> None:
        name = self._trace_filename or self._worker_name
        run_root = (
            name if os.path.dirname(name) else os.path.join(self._trace_dir, name)
        )
        os.makedirs(run_root, exist_ok=True)

        self._ensure_session_dir()

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        session_key = (
            f"{profiler_trace.profile_session_id()}_{self._profile_session_index}"
        )
        self._profile_session_index += 1

        canonical_ts = profiler_trace.resolve_canonical_dst_ts(
            run_root, rank, session_key=session_key
        )
        capture_dir = profiler_trace.rank_capture_dir(run_root, rank)
        os.makedirs(capture_dir, exist_ok=True)

        _, standard_opts, advanced_opts = resolve_profile_dir_and_opts(capture_dir)

        if not self._tpu_python_tracer:
            standard_opts["python_tracer_level"] = 0

        cfg = self.profiler_config
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            record_shapes=cfg.torch_profiler_record_shapes,
            profile_memory=cfg.torch_profiler_with_memory,
            with_stack=False,
            with_flops=cfg.torch_profiler_with_flops,
            on_trace_ready=self._on_trace_ready,
            experimental_config=TpuProfilerConfig(
                run_dir=capture_dir,
                host_tracer_level=standard_opts["host_tracer_level"],
                device_tracer_level=standard_opts["device_tracer_level"],
                python_tracer_level=standard_opts["python_tracer_level"],
                experimental_options=advanced_opts,
            ),
        )
        self._capture_state = (
            capture_dir,
            run_root,
            canonical_ts,
            rank,
            world_size,
            session_key,
        )

        super()._start()

    def _stop(self) -> None:
        try:
            super()._stop()
        finally:
            if self._capture_state is not None:
                (
                    capture_dir,
                    run_root,
                    canonical_ts,
                    rank,
                    world_size,
                    session_key,
                ) = self._capture_state
                profiler_trace.merge_rank_capture(
                    capture_dir,
                    run_root,
                    canonical_ts,
                    rank,
                    world_size=world_size,
                )
                if rank == 0:
                    profiler_trace.clear_canonical_ts_marker(run_root, session_key)
                self._capture_state = None
