# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Megakernel configuration, whatever the model.

* :class:`MegakernelEngineConfig` is what vLLM's configuration fixes for a
  program, as plain values (:meth:`MegakernelEngineConfig.from_vllm_config`).
* The kernel config: each model's megakernels have their own kernel settings
  (K3: ``kimi_k3.config.KernelConfig``), baked into every op's trace. The
  model records its config here when it installs it
  (:func:`set_active_kernel_config`), so the model-agnostic layers
  (``bridge``'s export-cache keys) can key on it without knowing the model.
  One model runs per process, so there is one active config.

No vLLM or torch imports: model kernel configs import this module, and they
stay free of both.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from vllm_torchtpu.megakernel import backend


@dataclasses.dataclass(frozen=True)
class MegakernelEngineConfig:
    """What vLLM's configuration fixes for a program, as plain values (the
    same on every rank). Serving builds it with :meth:`from_vllm_config`;
    the standalone tools construct it directly."""

    role: str  # "prefill" or "decode"
    tensor_parallel_size: int
    decode_context_parallel_size: int
    block_size: int
    max_num_seqs: int
    max_model_len: int
    max_num_batched_tokens: int
    num_speculative_tokens: int = 0
    # vLLM's per-batch-size draft count: (min_batch, max_batch, drafts)
    # ranges (None: num_speculative_tokens at every batch size)
    num_speculative_tokens_per_batch_size: tuple[tuple[int, int, int], ...] | None = (
        None
    )
    async_scheduling: bool = True
    synthetic_acceptance_rates: tuple[float, ...] | None = None
    # (draft width, rates) pairs: each width's own synthetic rates (None:
    # every width takes the leading synthetic_acceptance_rates)
    synthetic_acceptance_rates_by_width: (
        tuple[tuple[int, tuple[float, ...]], ...] | None
    ) = None
    kv_cache_dtype: str = "auto"
    mamba_ssm_cache_dtype: str = "auto"
    cp_kv_cache_interleave_size: int | None = None
    # Decoder attention data parallelism: groups of TP / attention_dp ranks,
    # each decoding its own requests with DCP over the group.
    attention_dp: int = 1

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> MegakernelEngineConfig:
        """The configuration of the role ``vllm_config`` selects (raises when
        it does not select the megakernel backend)."""
        role = backend.role(vllm_config)
        if role is None:
            raise ValueError(
                "the megakernel backend is not selected (--load-format megakernel)"
            )
        parallel, scheduler = vllm_config.parallel_config, vllm_config.scheduler_config
        spec = vllm_config.speculative_config
        schedule = (
            spec.num_speculative_tokens_per_batch_size if spec is not None else None
        )
        return cls(
            role=role,
            tensor_parallel_size=int(parallel.tensor_parallel_size),
            decode_context_parallel_size=int(parallel.decode_context_parallel_size),
            block_size=int(vllm_config.cache_config.block_size),
            kv_cache_dtype=str(vllm_config.cache_config.cache_dtype),
            mamba_ssm_cache_dtype=str(vllm_config.cache_config.mamba_ssm_cache_dtype),
            cp_kv_cache_interleave_size=int(parallel.cp_kv_cache_interleave_size),
            max_num_seqs=int(scheduler.max_num_seqs),
            max_model_len=int(vllm_config.model_config.max_model_len),
            max_num_batched_tokens=int(scheduler.max_num_batched_tokens),
            num_speculative_tokens=int(spec.num_speculative_tokens)
            if spec is not None
            else 0,
            num_speculative_tokens_per_batch_size=tuple(
                (int(lo), int(hi), int(k)) for lo, hi, k in schedule
            )
            if schedule
            else None,
            async_scheduling=bool(scheduler.async_scheduling),
            synthetic_acceptance_rates=tuple(spec.synthetic_acceptance_rates)
            if spec is not None and spec.rejection_sample_method == "synthetic"
            else None,
            synthetic_acceptance_rates_by_width=_rates_by_width(
                spec, backend.synthetic_acceptance_lengths(vllm_config)
            ),
            attention_dp=backend.attention_dp(vllm_config),
        )


def _rates_by_width(
    spec: Any, lengths: dict[int, float] | None
) -> tuple[tuple[int, tuple[float, ...]], ...] | None:
    """Each draft width's synthetic rates from its mean acceptance length,
    by vLLM's own rule (``SpeculativeConfig._acceptance_length_to_rates``)."""
    if lengths is None:
        return None
    if spec is None or spec.rejection_sample_method != "synthetic":
        raise ValueError(
            "synthetic_acceptance_lengths needs --speculative-config "
            "rejection_sample_method synthetic"
        )
    for width, length in lengths.items():
        if width < 1 or not 1.0 <= length <= width + 1:
            raise ValueError(
                f"synthetic_acceptance_lengths: {width} drafts take a mean "
                f"acceptance length in [1, {width + 1}], not {length}"
            )
    to_rates = type(spec)._acceptance_length_to_rates
    return tuple(
        (width, tuple(to_rates(length, width)))
        for width, length in sorted(lengths.items())
    )


class KernelConfig(Protocol):
    """A model's kernel settings."""

    def key(self) -> str:
        """A stable digest of the settings, for caches of compiled kernels."""
        ...


_ACTIVE: KernelConfig | None = None


def set_active_kernel_config(config: KernelConfig) -> None:
    """Record the process's kernel config (the model's install does it)."""
    global _ACTIVE
    _ACTIVE = config


def get_active_kernel_config() -> KernelConfig | None:
    """The active model's kernel config (None: none is installed)."""
    return _ACTIVE
