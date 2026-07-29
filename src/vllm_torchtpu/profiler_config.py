# SPDX-License-Identifier: Apache-2.0
"""vllm-torchtpu's ProfilerConfig, adding the phased profiler kind.

The phased profiler captures one trace per inference phase (prefill-heavy,
decode-heavy, and so on) rather than one continuous trace. It is a TPU-only
profiler, so vLLM's ``ProfilerConfig.profiler`` -- a closed
``Literal["torch", "cuda"]`` -- has no value for it.

This subclass widens the field and relaxes the one validator rule that
assumes a torch profile always means ``profiler == "torch"``.
``_patch_vllm_profiler_config_class`` points ``VllmConfig.profiler_config``
at this class so vLLM builds it from the CLI.
"""

import dataclasses
from typing import Literal

from pydantic import Field, model_validator
from vllm.config.profiler import ProfilerConfig
from vllm.config.utils import config

TORCH_PHASED_PROFILER_KIND = "torch_phased"

# Kinds that record through the torch profiler and therefore share
# `torch_profiler_dir`.
TORCH_PROFILER_KINDS = ("torch", TORCH_PHASED_PROFILER_KIND)

# -1 disables the gate: capture the first decode-only batch seen.
PHASED_DECODE_ONLY_KV_LEN_THRESHOLD_DISABLED = -1


@config
class TpuProfilerConfig(ProfilerConfig):
    """ProfilerConfig with the TPU phased profiler as a selectable kind."""

    profiler: Literal["torch", "cuda", "torch_phased"] | None = None
    """Which profiler to use. Defaults to None. Options are:

    - 'torch': Use PyTorch profiler.
    - 'cuda': Use CUDA profiler.
    - 'torch_phased': Use the PyTorch profiler, but capture one trace per
      inference phase rather than a single continuous trace. Shares the
      `torch_profiler_*` settings with 'torch', `torch_profiler_dir` above
      all. Driven by TPUModelRunner, which is where batch composition -- and
      therefore the current phase -- is known."""

    phased_decode_only_kv_len_threshold: int = Field(
        default=PHASED_DECODE_ONLY_KV_LEN_THRESHOLD_DISABLED, ge=-1)
    """Minimum KV length a decode-only batch must reach before the
    'torch_phased' profiler captures the decode_only phase. The phase is
    captured the first time it is seen, which early in a run means nearly
    empty KV caches and a trace that is not representative of steady-state
    decode. Defaults to -1, meaning capture the first decode-only batch."""

    @model_validator(mode="before")
    @classmethod
    def _accept_base_profiler_config(cls, value):
        """Coerce a plain ``ProfilerConfig`` into this subclass.

        Two defaults are still built from vLLM's class: ``EngineArgs``' own
        field factory (bound when that module is imported) and the
        ``--profiler-config`` argparse default (built from the field's
        unrepointed factory). Both reach ``VllmConfig`` explicitly, where the
        repointed field would otherwise reject them and break every run that
        does not profile.
        """
        if isinstance(value, ProfilerConfig) and not isinstance(value, cls):
            return {
                f.name: getattr(value, f.name)
                for f in dataclasses.fields(value)
            }
        return value

    @model_validator(mode="after")
    def _validate_profiler_config(self):
        # Overrides the base validator by name; pydantic collects model
        # validators into __pydantic_decorators__ keyed by name, so this
        # replaces rather than augments the inherited one.
        profiler_dir = self.torch_profiler_dir
        if profiler_dir and self.profiler not in TORCH_PROFILER_KINDS:
            raise ValueError(
                "torch_profiler_dir is only applicable when profiler is set "
                f"to one of {TORCH_PROFILER_KINDS}")
        if self.profiler in TORCH_PROFILER_KINDS and not profiler_dir:
            raise ValueError("torch_profiler_dir must be set when profiler "
                             f"is {self.profiler!r}")

        if self.profiler != TORCH_PHASED_PROFILER_KIND:
            return ProfilerConfig._validate_profiler_config(self)

        # Delegate the rest (dir abspath/URI handling, ...) to the base
        # validator by presenting a kind it accepts, so upstream changes to
        # that logic are inherited rather than forked. ignore_frontend rides
        # along because AsyncLLM starts its frontend CPU profiler only for
        # `profiler == "torch"` (v1/engine/async_llm.py), so a phased run has
        # no frontend to profile -- without this the base validator warns
        # about frontend overhead that cannot occur.
        real_ignore_frontend = self.ignore_frontend
        self.profiler = "torch"
        self.ignore_frontend = True
        try:
            return ProfilerConfig._validate_profiler_config(self)
        finally:
            self.profiler = TORCH_PHASED_PROFILER_KIND
            self.ignore_frontend = real_ignore_frontend
