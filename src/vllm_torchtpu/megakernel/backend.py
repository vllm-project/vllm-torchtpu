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
"""Selecting the megakernel backend from vLLM's configuration::

    --load-format megakernel \\
    --model-loader-extra-config '{"presharded": "/path/to/kimi-k3-tp32"}'

``presharded`` is the program's TP32 pre-shard directory. A KV producer runs
the prefill program, a KV consumer the decode program; the worker then builds
``MegakernelModelRunner`` and loads the weightless model through
``megakernel.loader``. Cheap to import: the platform and the scheduler
patches call it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

LOAD_FORMAT = "megakernel"
# Model architectures with a megakernel program: the package under
# ``vllm_torchtpu.megakernel`` whose ``program`` module checks, describes and
# builds it (``LOADER_KEYS``, the ``--model-loader-extra-config`` keys by role;
# ``capabilities``; ``from_vllm``).
PROGRAMS = {
    "KimiK3ForConditionalGeneration": "kimi_k3",
    "KimiLinearForCausalLM": "kimi_k3",
    "Qwen3_5MoeForCausalLM": "qwen38",
}


def program(vllm_config: Any) -> ModuleType:
    """The ``program`` module of the model ``vllm_config`` serves."""
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    for architecture in architectures:
        if architecture in PROGRAMS:
            return importlib.import_module(
                f"vllm_torchtpu.megakernel.{PROGRAMS[architecture]}.program"
            )
    raise ValueError(
        f"no megakernel program for {architectures}; supported: {list(PROGRAMS)}"
    )


def role(vllm_config: Any) -> str | None:
    """``"prefill"`` or ``"decode"`` when the backend is selected
    (``--load-format megakernel``), else None. Raises on a configuration the
    backend can't serve."""
    load_config = getattr(vllm_config, "load_config", None)
    if getattr(load_config, "load_format", None) != LOAD_FORMAT:
        return None
    kv = getattr(vllm_config, "kv_transfer_config", None)
    if kv is None or kv.is_kv_producer == kv.is_kv_consumer:
        raise ValueError(
            "the megakernel backend serves one role of a P/D pair: "
            "a KV transfer config with "
            "kv_role kv_producer (prefill) or kv_consumer (decode)"
        )
    selected = "prefill" if kv.is_kv_producer else "decode"
    _loader_config(vllm_config, selected)
    return selected


def _loader_config(vllm_config: Any, selected: str) -> dict:
    """The validated ``--model-loader-extra-config`` of role ``selected``."""
    value = vllm_config.load_config.model_loader_extra_config or {}
    if not isinstance(value, dict):
        raise ValueError(
            "--load-format megakernel takes --model-loader-extra-config as a "
            'dict, e.g. {"presharded": "<dir>"}'
        )
    keys = program(vllm_config).LOADER_KEYS[selected]
    unknown = set(value) - keys
    if unknown:
        hint = (
            " (the decoder drafts with --speculative-config's model and imports "
            "the prefill's rings whenever that is DSpark)"
            if "draft" in unknown
            else ""
        )
        raise ValueError(
            f"--model-loader-extra-config for the megakernel {selected} role "
            f"takes {sorted(keys)}, not {sorted(unknown)}{hint}"
        )
    if (
        "presharded" in keys
        and not value.get("presharded")
        and value.get("synthetic_seed") is None
    ):
        raise ValueError(
            "--load-format megakernel needs --model-loader-extra-config "
            '\'{"presharded": "<dir>"}\': the pre-shard directory of the '
            "program (e.g. /path/to/kimi-k3-tp32)"
        )
    return value


def presharded(vllm_config: Any) -> Path | None:
    """The program's pre-shard directory (None: synthetic weights)."""
    selected = role(vllm_config)
    if selected is None:
        raise ValueError("the megakernel backend is not selected (--load-format)")
    path = _loader_config(vllm_config, selected).get("presharded")
    return Path(path) if path else None


def loader_option(vllm_config: Any, key: str) -> Any:
    """``key`` of the selected role's loader config (None if unset)."""
    selected = role(vllm_config)
    if selected is None:
        return None
    return _loader_config(vllm_config, selected).get(key)


def synthetic_seed(vllm_config: Any) -> int | None:
    """The seed of random weights in place of a pre-shard (``synthetic_seed``
    in the loader config of a program that takes it), for timing runs."""
    selected = role(vllm_config)
    if selected is None:
        return None
    seed = _loader_config(vllm_config, selected).get("synthetic_seed")
    return None if seed is None else int(seed)


def draft(vllm_config: Any) -> Path | None:
    """The DSpark speculator checkpoint whose ring the prefill role computes
    (``draft`` in its loader config; None on the decode role)."""
    if role(vllm_config) != "prefill":
        return None
    path = _loader_config(vllm_config, "prefill").get("draft")
    return Path(path) if path else None


def mla_draft(vllm_config: Any) -> Path | None:
    """The MLA DSpark draft checkpoint (``model_type`` ``k3_dspark``,
    ``Inferact/Kimi-K3-DSpark``) of either role -- the prefill's ``draft``,
    the decoder's speculative model -- or None. Its latent layers are KV cache
    layers of both roles (``megakernel.runner``), so the transfer carries them."""
    import json

    selected = role(vllm_config)
    if selected == "prefill":
        path = draft(vllm_config)
    elif selected == "decode":
        spec = vllm_config.speculative_config
        path = (
            Path(str(spec.model))
            if spec is not None and spec.method == "dspark" and spec.model
            else None
        )
    else:
        return None
    if path is None or not (path / "config.json").is_file():
        return None
    config = json.loads((path / "config.json").read_text())
    return path if config.get("model_type") == "k3_dspark" else None


def attention_dp(vllm_config: Any) -> int:
    """The decoder's attention data parallelism (``attention_dp`` in the
    decode role's loader config, default 1): the decode role then runs
    ``--decode-context-parallel-size`` TP / attention_dp."""
    if role(vllm_config) != "decode":
        return 1
    return int(_loader_config(vllm_config, "decode").get("attention_dp", 1))


def shared_blocks(vllm_config: Any) -> int | None:
    """The decode role's shared KV block region under attention DP
    (``shared_blocks`` in its loader config; None: the default size,
    ``megakernel.dp_blocks``)."""
    if role(vllm_config) != "decode":
        return None
    value = _loader_config(vllm_config, "decode").get("shared_blocks")
    return None if value is None else int(value)


def synthetic_acceptance_lengths(vllm_config: Any) -> dict[int, float] | None:
    """The decoder's synthetic mean acceptance length per draft width
    (``synthetic_acceptance_lengths`` in its loader config, e.g.
    ``{"3": 2.87, "7": 3.78}``; None: vLLM's one rate vector, truncated to
    each width)."""
    if role(vllm_config) != "decode":
        return None
    value = _loader_config(vllm_config, "decode").get("synthetic_acceptance_lengths")
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "synthetic_acceptance_lengths maps draft widths to mean acceptance "
            'lengths, e.g. {"3": 2.87, "7": 3.78}'
        )
    return {int(width): float(length) for width, length in value.items()}


def draft_ring_rows(vllm_config: Any) -> int:
    """KDA state-block rows the KV transfer carries for the DSpark draft ring
    (0 without one): on the prefill role when it computes rings (``draft``),
    on the decode role when it speculates with DSpark. The peers' pool
    manifests, and so their layout fingerprints, differ unless both agree."""
    selected = role(vllm_config)
    if selected == "prefill":
        carries = draft(vllm_config) is not None
    elif selected == "decode":
        spec = vllm_config.speculative_config
        carries = spec is not None and spec.method == "dspark"
    else:
        return 0
    if not carries or mla_draft(vllm_config) is not None:
        return 0  # (the MLA draft's context is in its own KV cache layers)
    from vllm_torchtpu.megakernel.kimi_k3.layout import LAYOUT

    return LAYOUT.region("draft").rows


def capabilities(vllm_config: Any):
    """The selected program's :class:`Capabilities` (None: backend not
    selected). Raises on a pipeline the program cannot be a stage of."""
    if role(vllm_config) is None:
        return None
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig

    selected = program(vllm_config).capabilities(
        MegakernelEngineConfig.from_vllm_config(vllm_config)
    )
    pipeline = vllm_config.parallel_config.pipeline_parallel_size > 1
    if pipeline and not selected.pipeline:
        raise ValueError(
            "this megakernel program does not run as a pipeline stage; "
            "use --pipeline-parallel-size 1"
        )
    return selected
