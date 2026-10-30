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
"""Runtime-facing views for megakernel model folders.

The model contract itself (`TpuMonoSpec`, the `MonoProgram` protocol,
`Bucket`, `RunnerShape`, `WarmupCase`, `Decline`, `LayoutDecline`,
`StepDecision`, `EligibilityResult`, the feature / parallelism / `requires`
vocabularies) lives in the framework-free `megakernel.runtime.contract` and is
re-exported here. What stays here are the runtime's own facts about the
runner: `RunnerView` (startup / bind) and `ForwardView` (per step), and the
shorthand table of built-in templates.

Stdlib only at import (no torch / jax / vLLM).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

from vllm_torchtpu.megakernel.runtime.contract import (
    API_VERSION,
    FEATURES,
    NUMERICS,
    PARALLELISM,
    REQUIRES_KEYS,
    Bucket,
    Decline,
    EligibilityResult,
    LayoutDecline,
    MonoProgram,
    RunnerShape,
    StepDecision,
    StepFn,
    TpuMonoSpec,
    WarmupCase,
)


class Shorthand(NamedTuple):
    """A built-in template selectable by name or by the checkpoint's HF arch."""

    module: str
    architectures: tuple[str, ...]
    model_types: tuple[str, ...]
    legacy_keys: frozenset[str]


# Built-in templates by shorthand name. Imported by string only, so the
# runtime never imports a template at module load.
SHORTHANDS: Mapping[str, Shorthand] = {
    "qwen35": Shorthand(
        module="vllm_torchtpu.models.qwen3_5.megakernel",
        architectures=(
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLM",
        ),
        model_types=("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text"),
        legacy_keys=frozenset(
            {
                "profile",
                "mode",
                "buckets",
                "verify_buckets",
                "fused_pool_io",
                "fused_pool_io_max_b",
                "fused_pool_io_min_b_hi",
                "pool_gather",
                "compile_hlo_check",
            }
        ),
    ),
}


def _int_or_none(x: Any) -> int | None:
    return int(x) if x is not None else None


class RunnerView:
    """Read-only, named facts about the runner a program may use.

    Each property reads one runner attribute; none runs a collective. The raw
    vLLM `model` is passed through: a program may mutate it only after
    agreement #1 (in `load`), and the stock forward must stay valid.
    """

    __slots__ = ("_runner", "_tp_rank")

    def __init__(self, runner: Any, *, tp_rank: int | None = None) -> None:
        self._runner = runner
        self._tp_rank = tp_rank

    @property
    def vllm_config(self) -> Any:
        return getattr(self._runner, "vllm_config", None)

    @property
    def model(self) -> Any:
        return getattr(self._runner, "model", None)

    @property
    def device(self) -> Any:
        return getattr(self._runner, "device", None)

    @property
    def hf_config(self) -> Any:
        r = self._runner
        model_cfg = getattr(r, "model_config", None) or getattr(
            getattr(r, "vllm_config", None), "model_config", None
        )
        return getattr(model_cfg, "hf_config", None) if model_cfg is not None else None

    @property
    def tp_size(self) -> int:
        p_cfg = getattr(self._runner, "parallel_config", None)
        tp_size = int(getattr(p_cfg, "tensor_parallel_size", 0) or 0)
        if tp_size > 0:
            return tp_size
        mesh = getattr(self._runner, "mesh", None)
        devs = getattr(mesh, "devices", None) if mesh is not None else None
        return int(devs.size) if devs is not None else 0

    @property
    def tp_rank(self) -> int:
        if self._tp_rank is not None:
            return int(self._tp_rank)
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    @property
    def speculative_tokens(self) -> int | None:
        """`num_speculative_tokens` (0 without a draft); `None` if unknown."""
        if not hasattr(self._runner, "speculative_config"):
            return None
        spec_cfg = getattr(self._runner, "speculative_config", None)
        return int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)

    @property
    def cache_block_size(self) -> int | None:
        r = self._runner
        for cfg in (
            getattr(getattr(r, "vllm_config", None), "cache_config", None),
            getattr(r, "cache_config", None),
        ):
            bs = getattr(cfg, "block_size", None)
            if bs:
                return int(bs)
        return None

    @property
    def token_paddings(self) -> tuple[int, ...]:
        return tuple(int(x) for x in getattr(self._runner, "num_tokens_paddings", ()))

    @property
    def req_block_pairs(self) -> tuple[tuple[int, int], ...]:
        """`(max reqs, blocks per req)` per attention-kernel shape class."""
        return tuple(_req_block_pairs(self._runner))

    @property
    def ckpt_window(self) -> int:
        spec_cfg = getattr(self._runner, "speculative_config", None)
        k_spec = int(spec_cfg.num_speculative_tokens) if spec_cfg is not None else 0
        return max(1, int(getattr(self._runner, "_mamba_ckpt_window", k_spec + 1)))

    @property
    def mamba_slot_read_offsets(self) -> Any:
        return getattr(self._runner, "mamba_slot_read_offsets", None)

    @property
    def uses_mrope(self) -> bool:
        return bool(getattr(self._runner, "uses_mrope", False))

    @property
    def enforce_eager(self) -> bool:
        return bool(getattr(self._runner, "enforce_eager", False))

    @property
    def unified_kv_layout(self) -> bool:
        return bool(getattr(self._runner, "_unified_kv_layout", False))

    @property
    def max_pages_per_seq(self) -> int:
        attn_shapes_fn = getattr(self._runner, "_attention_kernel_shapes", None)
        if callable(attn_shapes_fn):
            return int(attn_shapes_fn(True)[1])
        return int(getattr(self._runner, "max_num_blocks_per_req", 2))

    @property
    def device_kind(self) -> str:
        """The local accelerator's `device_kind` (e.g. `"TPU7x"`); no collective."""
        dev = self.device
        if getattr(dev, "type", None) == "tpu":
            import torch

            attrs = torch.get_device_module("tpu")._get_local_device_attributes()
            return str(attrs.get("device_kind") or "")
        import jax

        return str(getattr(jax.local_devices()[0], "device_kind", "") or "")


def _req_block_pairs(runner: Any) -> Iterator[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if runner is None:
        return iter(pairs)
    attn_shapes_fn = getattr(runner, "_attention_kernel_shapes", None)
    if callable(attn_shapes_fn):
        r_max, b_max, _ = attn_shapes_fn(True)
        pairs.append((int(r_max), int(b_max)))
        if getattr(runner, "most_model_len", None) is not None:
            r_most, b_most, _ = attn_shapes_fn(False)
            if (int(r_most), int(b_most)) not in pairs:
                pairs.append((int(r_most), int(b_most)))
        return iter(pairs)
    r_max = _int_or_none(getattr(runner, "num_reqs_max_model_len", None))
    b_max = _int_or_none(getattr(runner, "max_num_blocks_per_req", None))
    if r_max is not None and b_max is not None:
        pairs.append((r_max, b_max))
    if getattr(runner, "most_model_len", None) is not None:
        r_most = _int_or_none(getattr(runner, "num_reqs_most_model_len", None))
        b_most = _int_or_none(getattr(runner, "num_blocks_per_most_len_req", None))
        if r_most is not None and b_most is not None and (r_most, b_most) not in pairs:
            pairs.append((r_most, b_most))
    return iter(pairs)


@dataclasses.dataclass(slots=True)
class ForwardView:
    """What a program sees of one forward step besides its `StepContext`.

    `attn_metadata` is the forward context's per-layer metadata dict,
    `mrope_positions` the runner's host M-RoPE positions (read-only, or
    `None`), `steps` the compiled `StepFn` per bucket key and `num_tokens`
    the row count the stock model would return. `batch_shape(attn_md, m0)`
    is the runtime's host-only batch inspection; the program passes its
    attention and first recurrent layer metadata, used only when the runner
    can't describe the batch itself. Built once per step by the runtime;
    programs treat it as read-only.
    """

    attn_metadata: Mapping[str, Any] | None
    mrope_positions: Any
    steps: Mapping[tuple[int, int], StepFn]
    num_tokens: int | None
    inspect: Callable[[Any, Any], Any]

    def batch_shape(self, attn_md: Any = None, m0: Any = None) -> Any:
        return self.inspect(attn_md, m0)


__all__ = [
    "API_VERSION",
    "FEATURES",
    "NUMERICS",
    "PARALLELISM",
    "REQUIRES_KEYS",
    "SHORTHANDS",
    "Bucket",
    "Decline",
    "EligibilityResult",
    "ForwardView",
    "LayoutDecline",
    "MonoProgram",
    "RunnerShape",
    "RunnerView",
    "Shorthand",
    "StepDecision",
    "StepFn",
    "TpuMonoSpec",
    "WarmupCase",
]
