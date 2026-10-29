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
"""The megakernel plugin interface: what a megakernel folder implements.

A folder exports `PLUGIN = MegakernelPlugin(...)`; its `create` builds one
`MegakernelProgram` per TP rank, and the runtime (`ProgramDriver`) does the
rest: startup checks, rank agreement, compilation, warmup and step routing.

Stdlib only at import (no torch / jax / vLLM), so the plugin, the config
parser and the compile-cache salt can import it anywhere. The library types
`EligibilityResult`, `StepDecision` and `LayoutDecline` are exported lazily
(module `__getattr__`), since importing `megakernel.*` imports jax.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from .errors import MegakernelConfigError

if TYPE_CHECKING:  # exported lazily, see `__getattr__`
    from vllm_torchtpu.megakernel.errors import EligibilityResult
    from vllm_torchtpu.megakernel.runtime.eligibility import StepDecision
    from vllm_torchtpu.megakernel_runtime.declines import LayoutDecline

API_VERSION = 1

# --- Capabilities: framework features a plugin declares it can serve under.
# A feature that is on but undeclared declines the megakernel at startup with
# a named code (see `ProgramDriver`).
CAP_PIPELINE_PARALLEL = "pipeline_parallel"
CAP_LORA = "lora"
CAP_MULTIMODAL = "multimodal"
CAP_DCP = "dcp"
CAP_PCP = "pcp"
CAP_AUX_HIDDEN_STATES = "aux_hidden_states"
CAP_SPEC_DECODE = "spec_decode"
CAP_MROPE = "mrope"
CAP_PREFIX_CACHING = "prefix_caching"
CAP_KV_OFFLOAD = "kv_offload"
CAP_KV_TRANSFER = "kv_transfer"
CAP_EPLB_DYNAMIC = "eplb_dynamic"
CAP_SLEEP_MODE = "sleep_mode"
CAPABILITIES: frozenset[str] = frozenset(
    {
        CAP_PIPELINE_PARALLEL,
        CAP_LORA,
        CAP_MULTIMODAL,
        CAP_DCP,
        CAP_PCP,
        CAP_AUX_HIDDEN_STATES,
        CAP_SPEC_DECODE,
        CAP_MROPE,
        CAP_PREFIX_CACHING,
        CAP_KV_OFFLOAD,
        CAP_KV_TRANSFER,
        CAP_EPLB_DYNAMIC,
        CAP_SLEEP_MODE,
    }
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
        module="vllm_torchtpu.megakernel_runtime.models.qwen35",
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


@dataclasses.dataclass(frozen=True)
class MegakernelPlugin:
    """What a megakernel folder exports as `PLUGIN`.

    `create` is a callable or a `"module:attr"` string (`".adapter:create"` is
    relative to the plugin package), imported lazily, so importing the plugin
    imports nothing heavy. `create(view, options)` returns a program or a
    `Decline`; it runs no collective and loads no weights.
    """

    name: str
    version: str
    create: str | Callable[..., Any]
    capabilities: frozenset[str] = frozenset()
    layout_request: str | Callable[..., Any] | None = None
    decline_codes: tuple[str, ...] = ()
    source_root: str | None = None
    api_version: int = API_VERSION

    def __post_init__(self) -> None:
        caps = frozenset(str(c) for c in self.capabilities)
        object.__setattr__(self, "capabilities", caps)
        object.__setattr__(self, "decline_codes", tuple(self.decline_codes))
        if not isinstance(self.name, str) or not self.name:
            raise MegakernelConfigError("MegakernelPlugin.name must be non-empty")
        if unknown := sorted(caps - CAPABILITIES):
            raise MegakernelConfigError(
                f"MegakernelPlugin {self.name!r}: unknown capabilities {unknown}; "
                f"known: {sorted(CAPABILITIES)}"
            )


@dataclasses.dataclass(frozen=True)
class Bucket:
    """One compiled step shape: `T` tokens per sequence, `B` rows.

    `out_rows=None` lets the runtime pick the runner's padded token count.
    """

    T: int
    B: int
    out_rows: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.T), int(self.B))


@dataclasses.dataclass(frozen=True)
class RunnerShape:
    """One runner input shape a bucket is warmed up with."""

    tok_pad: int
    req_pad: int
    blk_pad: int
    ckpt_cols: int


@dataclasses.dataclass(frozen=True)
class WarmupCase:
    """Positional args for a bucket's `StepFn`, and an optional post-call check."""

    args: tuple[Any, ...]
    check: Callable[[Any], None] | None = None


StepFn = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class Decline:
    """A named, expected reason not to serve (`create`'s other result)."""

    code: str
    message: str = ""
    level: str = "warning"

    @property
    def result(self) -> Any:
        """This decline as a library `EligibilityResult`."""
        from vllm_torchtpu.megakernel.errors import EligibilityResult

        return EligibilityResult(False, self.code, self.message, self.level)


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


@dataclasses.dataclass(frozen=True)
class ForwardView:
    """What a program sees of one forward step besides its `StepContext`.

    `attn_metadata` is the forward context's per-layer metadata dict,
    `mrope_positions` the runner's host M-RoPE positions (read-only, or
    `None`), `steps` the compiled `StepFn` per bucket key and `num_tokens`
    the row count the stock model would return. `batch_shape(attn_md, m0)`
    is the runtime's host-only batch inspection; the program passes its
    attention and first recurrent layer metadata, used only when the runner
    can't describe the batch itself.
    """

    attn_metadata: Mapping[str, Any] | None
    mrope_positions: Any
    steps: Mapping[tuple[int, int], StepFn]
    num_tokens: int | None
    inspect: Callable[[Any, Any], Any]

    def batch_shape(self, attn_md: Any = None, m0: Any = None) -> Any:
        return self.inspect(attn_md, m0)


class MegakernelProgram(Protocol):
    """One megakernel instance on one TP rank (returned by `create`).

    Optional methods (looked up with `getattr`): `warmup_session()` (a
    context manager around warmup), `check_compiled_ir(debug, bucket)`,
    `attach_draft(drafter)` / `detach_draft()`, `on_warmup_done()`.
    """

    display_name: str

    def buckets(self, token_paddings: tuple[int, ...]) -> Sequence[Bucket]: ...

    def load(self, mesh: Any) -> None: ...

    def bind(self, view: RunnerView) -> Any: ...

    def build(self, bucket: Bucket) -> StepFn: ...

    def warmup_inputs(
        self, bucket: Bucket, shape: RunnerShape
    ) -> WarmupCase | None: ...

    def decide(self, ctx: Any, fwd: ForwardView) -> Any: ...

    def forward(self, decision: Any, ctx: Any, fwd: ForwardView) -> Any: ...


_LAZY_LIBRARY_EXPORTS = {
    "EligibilityResult": ("vllm_torchtpu.megakernel.errors", "EligibilityResult"),
    "StepDecision": ("vllm_torchtpu.megakernel.runtime.eligibility", "StepDecision"),
    "LayoutDecline": ("vllm_torchtpu.megakernel_runtime.declines", "LayoutDecline"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_LIBRARY_EXPORTS:
        import importlib

        mod_name, attr = _LAZY_LIBRARY_EXPORTS[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "API_VERSION",
    "CAPABILITIES",
    "CAP_AUX_HIDDEN_STATES",
    "CAP_DCP",
    "CAP_EPLB_DYNAMIC",
    "CAP_KV_OFFLOAD",
    "CAP_KV_TRANSFER",
    "CAP_LORA",
    "CAP_MROPE",
    "CAP_MULTIMODAL",
    "CAP_PCP",
    "CAP_PIPELINE_PARALLEL",
    "CAP_PREFIX_CACHING",
    "CAP_SLEEP_MODE",
    "CAP_SPEC_DECODE",
    "SHORTHANDS",
    "Bucket",
    "Decline",
    "EligibilityResult",
    "ForwardView",
    "LayoutDecline",
    "MegakernelConfigError",
    "MegakernelPlugin",
    "MegakernelProgram",
    "RunnerShape",
    "RunnerView",
    "Shorthand",
    "StepDecision",
    "StepFn",
    "WarmupCase",
]
