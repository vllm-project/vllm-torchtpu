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
vocabularies) lives in the framework-free `megakernel.runtime.contract`;
import it from there (its only import path). Here are the runtime's own
facts about the runner, `RunnerView` (startup / bind) and `ForwardView` (per
step), and the shorthands: the built-in table and the model folders
discovered by their `SPEC` (`shorthands()`).

Stdlib only at import (no torch / jax / vLLM).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import os
import pkgutil
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.core.runtime import contract as _contract

if TYPE_CHECKING:
    from vllm_torchtpu.megakernel.core.runtime.contract import StepFn

logger = init_logger(__name__)

# Where model folders live: `<MODELS_PACKAGE>.<model>.megakernel`.
MODELS_PACKAGE = "vllm_torchtpu.models"


class Shorthand(NamedTuple):
    """A template selectable by name or by the checkpoint's HF arch.

    `architectures` / `model_types` match the HF config's `architectures`
    entries and `model_type` (also of its `text_config`); `hf_fields`
    (`(field, value)` pairs, read from the HF config or its `text_config`)
    narrows a match, and an entry with `hf_fields` is preferred. The
    template's own settings live in its `SPEC`, not here.
    """

    module: str
    architectures: tuple[str, ...]
    model_types: tuple[str, ...]
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
    ),
}


@functools.cache
def discovered_shorthands() -> Mapping[str, Shorthand]:
    """Model folders under `MODELS_PACKAGE` whose `SPEC` declares the
    checkpoints it serves (`architectures` / `model_types`), by `SPEC.name`
    (lowercased).

    Found once per process: a package `<MODELS_PACKAGE>.<model>` with a
    `megakernel/__init__.py` on disk is imported, which runs that package's
    `__init__` and the folder's stdlib-only `__init__` (`SPEC`), never its
    `create` target. Packages without a `megakernel` folder aren't imported.
    A folder that fails to import is skipped with a warning. On a name clash
    the first package in name order wins.
    """
    try:
        models = importlib.import_module(MODELS_PACKAGE)
    except Exception as exc:  # no model folders installed: nothing to find
        logger.debug("megakernel: no %s (%s)", MODELS_PACKAGE, exc)
        return MappingProxyType({})
    found: dict[str, Shorthand] = {}
    infos = pkgutil.iter_modules(getattr(models, "__path__", ()))
    for info in sorted(infos, key=lambda i: i.name):
        root = getattr(info.module_finder, "path", None)
        if not info.ispkg or root is None:
            continue
        if not os.path.isfile(
            os.path.join(root, info.name, "megakernel", "__init__.py")
        ):
            continue
        module = f"{MODELS_PACKAGE}.{info.name}.megakernel"
        try:
            spec = getattr(importlib.import_module(module), "SPEC", None)
        except Exception as exc:  # a broken folder: say so, and skip it
            logger.warning(
                "Megakernel: model folder %s failed to import (%s: %s); "
                "skipping it, so the megakernel stays off for its models.",
                module,
                type(exc).__name__,
                exc,
            )
            continue
        if not isinstance(spec, _contract.TpuMonoSpec):
            continue
        name = spec.name.lower()
        if not (spec.architectures or spec.model_types) or name in found:
            continue
        found[name] = Shorthand(
            module=module,
            architectures=spec.architectures,
            model_types=spec.model_types,
            hf_fields=spec.hf_fields,
        )
    return MappingProxyType(found)


def shorthands() -> dict[str, Shorthand]:
    """Every shorthand: `SHORTHANDS`, then the discovered model folders whose
    name it doesn't already use (an explicit entry wins)."""
    merged = dict(SHORTHANDS)
    for name, entry in discovered_shorthands().items():
        merged.setdefault(name, entry)
    return merged


class RunnerView:
    """Read-only, named facts about the runner a program may use.

    Each property reads attributes the TPU model runner always sets (as does
    `testing.fake_runner`); none runs a collective, and a missing attribute
    raises rather than falling back to a guess. The raw vLLM `model` is passed
    through: a program may mutate it only after all TP ranks have agreed to
    load the program (in `load`), and the stock forward must stay valid.

    `cache_block_size`, `req_block_pairs`, `max_pages_per_seq` and
    `mamba_slot_read_offsets` describe the KV cache: they raise until the
    runner's KV cache is initialized (`runner.kv_cache_config` is set by
    `initialize_kv_cache`), so `create` can't read them, and `bind` can.
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
        attention kernel's page size can differ; see `max_pages_per_seq`).
        vLLM finalizes it after `load_model`, so it needs the KV cache
        initialized."""
        self._require_kv_cache("cache_block_size")
        return int(self._runner.cache_config.block_size)

    @property
    def token_paddings(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self._runner.num_tokens_paddings)

    def _require_kv_cache(self, name: str) -> None:
        if self._runner.kv_cache_config is None:
            raise RuntimeError(
                f"RunnerView.{name} describes the KV cache, which the runner "
                "hasn't initialized yet (runner.kv_cache_config is None); read "
                "it in bind, not in create"
            )

    @property
    def req_block_pairs(self) -> tuple[tuple[int, int], ...]:
        """`(max reqs, blocks per req)` per attention-kernel shape class: the
        max-model-len class, then the most-model-len class if the runner has
        one and it differs. Needs the KV cache initialized."""
        self._require_kv_cache("req_block_pairs")
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
        """The runner's per-slot recurrent-state read offsets (`None` without
        a hybrid model and a draft). Needs the KV cache initialized."""
        self._require_kv_cache("mamba_slot_read_offsets")
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
        """Pages per sequence the attention kernel's max-model-len shape uses
        (`req_block_pairs[0][1]`). Needs the KV cache initialized."""
        self._require_kv_cache("max_pages_per_seq")
        return self.req_block_pairs[0][1]

    @property
    def device_kind(self) -> str:
        """The local TPU's `device_kind` (e.g. `"TPU7x"`, from
        `utils.local_device_kind`); for a runner on another device (a CPU
        test), that device's type. No collective."""
        dev_type = str(self.device.type)
        if dev_type != "tpu":
            return dev_type
        from vllm_torchtpu.utils import local_device_kind

        return local_device_kind()


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
    "MODELS_PACKAGE",
    "SHORTHANDS",
    "ForwardView",
    "RunnerView",
    "Shorthand",
    "discovered_shorthands",
    "shorthands",
]
