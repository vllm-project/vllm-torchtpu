# SPDX-License-Identifier: Apache-2.0
"""TPU device profiler for vLLM-Omni diffusion workers.

vLLM-Omni's ``OmniTorchProfilerWrapper`` only records CUDA-family device
activities. This subclass records the TPU (``PrivateUse1``) activity through
the torch_tpu profiler, which writes an XSpace (``*.xplane.pb``, for xprof /
TensorBoard) next to the chrome trace Omni exports, in the same per-capture
session directory.

Enable it the usual Omni way (``profiler_config={"profiler": "torch",
"torch_profiler_dir": ...}``) and drive it with ``start_profile()`` /
``stop_profile()``.
"""

from typing import Any

import torch
from torch_tpu._internal.profiler import TpuProfilerConfig
from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper

from vllm_torchtpu.tracing.options import resolve_profile_dir_and_opts


class TpuOmniTorchProfilerWrapper(OmniTorchProfilerWrapper):
    """Omni torch profiler wrapper that captures TPU device activity."""

    def _get_default_activities(self) -> list[Any]:
        # TPU activity is added in _start; Omni's activity map has no TPU key.
        return ["CPU"]

    def _create_profiler(self, profiler_config: Any, activities: list[Any]) -> None:
        # The torch_tpu profiler binds its output directory at construction,
        # but Omni names the session directory per capture (see
        # set_trace_filename), so the profiler is built in _start.
        return None

    def _start(self) -> None:
        session_dir = self._ensure_session_dir()
        _, standard_opts, advanced_opts = resolve_profile_dir_and_opts(session_dir)
        cfg = self.profiler_config
        # Run one Python tracer, the TPU one (Python calls land in the
        # XSpace). torch's with_stack tracer on top of it deadlocks profiler
        # start (inside threading.setprofile) in Omni's multi-threaded engine.
        standard_opts["python_tracer_level"] = int(cfg.torch_profiler_with_stack)
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
                run_dir=session_dir,
                worker_rank=str(self._rank()),
                experimental_options=advanced_opts,
                **standard_opts,
            ),
        )
        super()._start()
