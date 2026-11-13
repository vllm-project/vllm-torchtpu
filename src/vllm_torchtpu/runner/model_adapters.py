# SPDX-License-Identifier: Apache-2.0
"""Optional model-owned runner extension, selected once before weight loading.

A model directory registers an adapter factory for its architectures; the
runner asks for one in `load_model`. With no registration (the default) the
selection returns None and the ordinary runner and cache paths are unchanged.

An adapter owns the input snapshot of each step, the dummy/profile forward and
its KV cache lifecycle; the model forward then receives explicit kwargs rather
than consulting runner state. Optional extras, looked up with `getattr`:
`bind_draft(drafter)`, `precompile_fused_inputs(num_tokens_paddings,
rows_options)` and `kv_cache_memory_budget() -> int | None`.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from vllm_torchtpu.kv_cache_materializer import LayerKVCache


@dataclass(frozen=True)
class ModelInputSnapshot:
    """What the runner hands an adapter for one forward chunk."""

    token_store: torch.Tensor
    row_start: int
    num_requests: int
    num_tokens: int
    attention_metadata: dict[str, Any]
    input_ids: torch.Tensor


class ModelAdapter(Protocol):
    def bind_model(self, model): ...

    def begin_step(self) -> torch.Tensor: ...

    def prepare_inputs(self, snapshot: ModelInputSnapshot) -> dict[str, Any]: ...

    def dummy_run(
        self, num_tokens, num_reqs, profile_prefix=None, *, dp_lockstep=False
    ) -> float | None: ...

    def get_kv_cache_spec(self) -> dict[str, Any]: ...

    def initialize_kv_cache(self, config) -> dict[str, LayerKVCache] | None: ...


ModelAdapterFactory = Callable[[Any], ModelAdapter | None]

_FACTORIES: dict[str, ModelAdapterFactory] = {}


def register_model_adapter(
    architectures: Iterable[str], factory: ModelAdapterFactory
) -> None:
    """Serve `architectures` with the adapter `factory(runner)` returns.

    Called from a model directory's registration hook. A later registration
    for the same architecture replaces the earlier one.
    """
    for arch in architectures:
        _FACTORIES[arch] = factory


def unregister_model_adapter(architectures: Iterable[str]) -> None:
    for arch in architectures:
        _FACTORIES.pop(arch, None)


def select_model_adapter(runner) -> ModelAdapter | None:
    """The registered adapter for the runner's architecture, else None."""
    factory = _FACTORIES.get(runner.model_config.architecture)
    return None if factory is None else factory(runner)
