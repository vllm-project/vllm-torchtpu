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
from collections.abc import Callable, Mapping
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
    hf_fields: tuple[tuple[str, Any], ...] = ()


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
    "qwen38_2_4t": Shorthand(
        module="vllm_torchtpu.models.qwen38_2_4t.megakernel",
        architectures=(
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLM",
        ),
        model_types=("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text"),
        legacy_keys=frozenset(),
        hf_fields=(("num_hidden_layers", 92), ("hidden_size", 8192)),
    ),
    "kimi_k3": Shorthand(
        module="vllm_torchtpu.models.kimi_k3.megakernel",
        architectures=("KimiK3ForConditionalGeneration",),
        model_types=("kimi_k3",),
        legacy_keys=frozenset(),
        hf_fields=(),
    ),
}


class RunnerView:
    """Read-only, named facts about the runner a program may use.

    Each property reads attributes the TPU model runner always sets (as does
    `testing.fake_runner`); none runs a collective, and a missing attribute
    raises rather than falling back to a guess. The raw vLLM `model` is passed
    through: a program may mutate it only after all TP ranks have agreed to
    load the program (in `load`), and the stock forward must stay valid.
    """

    __slots__ = ("_runner", "_tp_rank")

    def __init__(self, runner: Any, *, tp_rank: int | None = None) -> None:
        self._runner = runner
        self._tp_rank = tp_rank

    @property
    def vllm_config(self) -> Any:
        return self._runner.vllm_config

    @property
    def model(self) -> Any:
        return self._runner.model

    @property
    def device(self) -> Any:
        return self._runner.device

    @property
    def hf_config(self) -> Any:
        return self._runner.model_config.hf_config

    @property
    def tp_size(self) -> int:
        """`parallel_config.tensor_parallel_size` (the runner's own mesh may
        hold fewer devices than the TP group)."""
        tp_size = getattr(self._runner.parallel_config, "tensor_parallel_size", None)
        if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size <= 0:
            raise RuntimeError(
                "RunnerView.tp_size: runner.parallel_config.tensor_parallel_size "
                f"is {tp_size!r}; expected a positive int"
            )
        return tp_size

    @property
    def tp_rank(self) -> int:
        if self._tp_rank is not None:
            return int(self._tp_rank)
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    @property
    def speculative_tokens(self) -> int:
        """`num_speculative_tokens` (0 without a draft)."""
        spec_cfg = self._runner.speculative_config
        return int(spec_cfg.num_speculative_tokens) if spec_cfg is not None else 0

    @property
    def cache_block_size(self) -> int:
        """`cache_config.block_size`: the KV / state cache block size (the
        attention kernel's page size can differ; see `max_pages_per_seq`)."""
        return int(self._runner.cache_config.block_size)

    @property
    def token_paddings(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._runner.num_tokens_paddings)

    @property
    def req_block_pairs(self) -> tuple[tuple[int, int], ...]:
        """`(max reqs, blocks per req)` per attention-kernel shape class: the
        max-model-len class, then the most-model-len class if the runner has
        one and it differs."""
        shapes = self._runner._attention_kernel_shapes
        r_max, b_max, _ = shapes(True)
        pairs = [(int(r_max), int(b_max))]
        if self._runner.most_model_len is not None:
            r_most, b_most, _ = shapes(False)
            if (int(r_most), int(b_most)) not in pairs:
                pairs.append((int(r_most), int(b_most)))
        return tuple(pairs)

    @property
    def ckpt_window(self) -> int:
        return max(1, int(self._runner._mamba_ckpt_window))

    @property
    def mamba_slot_read_offsets(self) -> Any:
        return self._runner.mamba_slot_read_offsets

    @property
    def uses_mrope(self) -> bool:
        return bool(self._runner.uses_mrope)

    @property
    def enforce_eager(self) -> bool:
        return bool(self._runner.enforce_eager)

    @property
    def unified_kv_layout(self) -> bool:
        return bool(self._runner._unified_kv_layout)

    @property
    def max_pages_per_seq(self) -> int:
        """Pages per sequence the attention kernel's max-model-len shape uses."""
        return int(self._runner._attention_kernel_shapes(True)[1])

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
