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
# src/vllm_torchtpu/megakernel/backend.py
r"""Select the Kimi-K3 megakernel backend from vLLM configuration.

Example CLI invocation::

    --load-format megakernel \
    --model-loader-extra-config '{"presharded": "/path/to/kimi-k3-tp32"}'

``presharded`` is the program's TP32 pre-shard directory. A KV producer runs
the prefill program, a KV consumer the decode program; the worker then builds
``MegakernelModelRunner`` and loads the weightless model through the
megakernel loader (not in this package yet). Cheap to import: the platform
and scheduler
patches call it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.paths import require_local_path

if TYPE_CHECKING:
    from pathlib import Path

    from vllm.config import VllmConfig

    from .program import Capabilities


LOAD_FORMAT = "megakernel"
# Model architectures with a megakernel program.
ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiLinearForCausalLM")
# --model-loader-extra-config keys, by role.
_LOADER_KEYS = {
    "prefill": {"presharded", "draft"},
    "decode": {
        "presharded",
        "attention_dp",
        "synthetic_acceptance_lengths",
    },
}


def role(vllm_config: VllmConfig) -> str | None:
    """Return ``"prefill"`` or ``"decode"`` when ``--load-format megakernel`` is set.

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        ``"prefill"`` for a KV producer, ``"decode"`` for a KV consumer, or
        ``None`` when ``--load-format megakernel`` is not selected.

    Raises:
        TypeError: If ``model_loader_extra_config`` is not a ``dict``.
        ValueError: If the KV role, model architecture, or loader config is invalid.
    """
    load_config = vllm_config.load_config
    if load_config is None or load_config.load_format != LOAD_FORMAT:
        return None
    kv = vllm_config.kv_transfer_config
    if kv is None or kv.is_kv_producer == kv.is_kv_consumer:
        msg = (
            "the megakernel backend serves one role of a P/D pair: "
            "a KV transfer config with "
            "kv_role kv_producer (prefill) or kv_consumer (decode)"
        )
        raise ValueError(msg)
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config if model_config is not None else None
    architectures = (hf_config.architectures if hf_config is not None else None) or []
    if model_config is not None and not any(
        arch in ARCHITECTURES for arch in architectures
    ):
        msg = (
            f"no megakernel program for {architectures}; "
            f"supported: {list(ARCHITECTURES)}"
        )
        raise ValueError(msg)
    selected = "prefill" if kv.is_kv_producer else "decode"
    _loader_config(vllm_config, selected)
    return selected


def _loader_config(vllm_config: VllmConfig, selected: str) -> dict[str, object]:
    """Return the validated ``--model-loader-extra-config`` dictionary for ``selected``.

    Args:
        vllm_config: Active vLLM engine configuration object.
        selected: Role name (``"prefill"`` or ``"decode"``).

    Returns:
        Validated loader configuration mapping for ``selected``.

    Raises:
        TypeError: If ``model_loader_extra_config`` is not a ``dict``.
        ValueError: If unknown keys are present or ``"presharded"`` is missing.
    """
    load_config = vllm_config.load_config
    value = (
        load_config.model_loader_extra_config if load_config is not None else None
    ) or {}
    if not isinstance(value, dict):
        msg = (
            "--load-format megakernel takes --model-loader-extra-config as a "
            'dict, e.g. {"presharded": "<dir>"}'
        )
        raise TypeError(msg)
    unknown = set(value) - _LOADER_KEYS[selected]
    if unknown:
        hint = (
            " (the decoder drafts with --speculative-config's model and imports "
            "the prefill's rings whenever that is DSpark)"
            if "draft" in unknown
            else ""
        )
        msg = (
            f"--model-loader-extra-config for the megakernel {selected} role "
            f"takes {sorted(_LOADER_KEYS[selected])}, not {sorted(unknown)}{hint}"
        )
        raise ValueError(msg)
    if not value.get("presharded"):
        msg = (
            "--load-format megakernel needs --model-loader-extra-config "
            '\'{"presharded": "<dir>"}\': the TP32 pre-shard directory '
            "(e.g. /path/to/kimi-k3-tp32)"
        )
        raise ValueError(msg)
    return value


def presharded(vllm_config: VllmConfig) -> Path:
    """Return the TP32 pre-sharded weight directory path from ``vllm_config``.

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        Filesystem ``Path`` to the pre-sharded weight directory.

    Raises:
        ValueError: If the megakernel backend is not selected.
    """
    selected = role(vllm_config)
    if selected is None:
        msg = "the megakernel backend is not selected (--load-format)"
        raise ValueError(msg)
    return require_local_path(
        str(_loader_config(vllm_config, selected)["presharded"]),
        label="presharded",
    )


def draft(vllm_config: VllmConfig) -> Path | None:
    """Return the DSpark speculator checkpoint path for the prefill role, if set.

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        ``Path`` to the ``draft`` checkpoint on the prefill role, or ``None``.
    """
    if role(vllm_config) != "prefill":
        return None
    path = _loader_config(vllm_config, "prefill").get("draft")
    return require_local_path(str(path), label="draft") if path else None


def attention_dp(vllm_config: VllmConfig) -> int:
    """Return the decoder attention data-parallel degree (default ``1``).

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        Attention data-parallel degree (``1`` when not on the decode role).

    Raises:
        TypeError: If ``attention_dp`` is a boolean or non-integer type.
        ValueError: If ``attention_dp`` is not a positive integer.
    """
    if role(vllm_config) != "decode":
        return 1
    raw_dp = _loader_config(vllm_config, "decode").get("attention_dp", 1)
    if isinstance(raw_dp, bool) or not isinstance(raw_dp, (int, str)):
        msg = f"attention_dp must be a positive integer, not {raw_dp!r}"
        raise TypeError(msg)
    dp = int(raw_dp)
    if dp <= 0:
        msg = f"attention_dp must be positive, not {dp}"
        raise ValueError(msg)
    return dp


def synthetic_acceptance_lengths(
    vllm_config: VllmConfig,
) -> dict[int, float] | None:
    """Return the decoder's synthetic mean acceptance length per draft width.

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        Mapping from draft width to mean acceptance length, or ``None``.

    Raises:
        TypeError: If ``synthetic_acceptance_lengths`` is not a ``dict``.
        ValueError: If ``synthetic_acceptance_lengths`` is empty.
    """
    if role(vllm_config) != "decode":
        return None
    value = _loader_config(vllm_config, "decode").get("synthetic_acceptance_lengths")
    if value is None:
        return None
    if not isinstance(value, dict):
        msg = (
            "synthetic_acceptance_lengths maps draft widths to mean acceptance "
            'lengths, e.g. {"3": 2.87, "7": 3.78}'
        )
        raise TypeError(msg)
    if not value:
        msg = (
            "synthetic_acceptance_lengths maps draft widths to mean acceptance "
            'lengths, e.g. {"3": 2.87, "7": 3.78}'
        )
        raise ValueError(msg)
    return {int(width): float(length) for width, length in value.items()}


def capabilities(vllm_config: VllmConfig) -> Capabilities | None:
    """Return the selected program's :class:`Capabilities`, or ``None`` if inactive.

    Args:
        vllm_config: Active vLLM engine configuration object.

    Returns:
        Program :class:`Capabilities` when the megakernel backend is active,
        else ``None``.
    """
    if role(vllm_config) is None:
        return None
    # Deferred imports break the backend -> program -> config -> backend import cycle.
    from vllm_torchtpu.models.kimi_k3.megakernel import program
    from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config import (
        MegakernelEngineConfig,
    )

    return program.capabilities(MegakernelEngineConfig.from_vllm_config(vllm_config))
