# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/config.py
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
from typing import Protocol

from . import backend


@dataclasses.dataclass(frozen=True)
class MegakernelEngineConfig:
    """Engine parameters that vLLM's configuration fixes for a megakernel program.

    Serving builds this config with :meth:`from_vllm_config`; standalone tools
    construct it directly.
    """

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
    def from_vllm_config(cls, vllm_config: object) -> MegakernelEngineConfig:
        """Build the engine configuration for the role selected by ``vllm_config``.

        Args:
            vllm_config: Active vLLM engine configuration object.

        Returns:
            Populated :class:`MegakernelEngineConfig` instance.

        Raises:
            ValueError: If ``vllm_config`` does not select the megakernel backend.
        """
        selected_role = backend.role(vllm_config)
        if selected_role is None:
            msg = "the megakernel backend is not selected (--load-format megakernel)"
            raise ValueError(msg)
        vllm_attrs = vars(vllm_config)
        parallel = vllm_attrs["parallel_config"]
        scheduler = vllm_attrs["scheduler_config"]
        cache_config = vllm_attrs["cache_config"]
        model_config = vllm_attrs["model_config"]
        spec = vllm_attrs.get("speculative_config")
        schedule = (
            spec.num_speculative_tokens_per_batch_size if spec is not None else None
        )
        return cls(
            role=selected_role,
            tensor_parallel_size=int(parallel.tensor_parallel_size),
            decode_context_parallel_size=int(parallel.decode_context_parallel_size),
            block_size=int(cache_config.block_size),
            kv_cache_dtype=str(cache_config.cache_dtype),
            mamba_ssm_cache_dtype=str(cache_config.mamba_ssm_cache_dtype),
            cp_kv_cache_interleave_size=int(parallel.cp_kv_cache_interleave_size),
            max_num_seqs=int(scheduler.max_num_seqs),
            max_model_len=int(model_config.max_model_len),
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
    spec: object,
    lengths: dict[int, float] | None,
) -> tuple[tuple[int, tuple[float, ...]], ...] | None:
    """Compute per-draft-width synthetic acceptance rates from mean lengths.

    Uses ``SpeculativeConfig._acceptance_length_to_rates`` to convert each width's
    target mean acceptance length into per-position rates.

    Args:
        spec: vLLM ``SpeculativeConfig`` object, or ``None``.
        lengths: Mapping from draft width to mean acceptance length, or ``None``.

    Returns:
        Sorted tuple of ``(width, rates_tuple)`` pairs, or ``None``.

    Raises:
        ValueError: If synthetic rejection sampling is not enabled or a mean
            acceptance length falls outside ``[1, width + 1]``.
    """
    if lengths is None:
        return None
    if spec is None or vars(spec).get("rejection_sample_method") != "synthetic":
        msg = (
            "synthetic_acceptance_lengths needs --speculative-config "
            "rejection_sample_method synthetic"
        )
        raise ValueError(msg)
    for width, length in lengths.items():
        if width < 1 or not 1.0 <= length <= width + 1:
            msg = (
                f"synthetic_acceptance_lengths: {width} drafts take a mean "
                f"acceptance length in [1, {width + 1}], not {length}"
            )
            raise ValueError(msg)
    to_rates = vars(type(spec))["_acceptance_length_to_rates"]
    return tuple(
        (width, tuple(to_rates(length, width)))
        for width, length in sorted(lengths.items())
    )


class KernelConfig(Protocol):
    """Protocol for a model's compile-time kernel settings."""

    def key(self) -> str:
        """Return a stable digest of the settings for compiled kernel caches."""
        ...


_ACTIVE: KernelConfig | None = None


def set_active_kernel_config(config: KernelConfig) -> None:
    """Record the process-wide active kernel configuration.

    Args:
        config: Model ``KernelConfig`` instance being installed.
    """
    globals()["_ACTIVE"] = config


def get_active_kernel_config() -> KernelConfig | None:
    """Return the active model's kernel config, or ``None`` if none is installed.

    Returns:
        Installed :class:`KernelConfig` instance, or ``None``.
    """
    return _ACTIVE
