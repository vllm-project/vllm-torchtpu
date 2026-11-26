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
"""Optional model-owned runner extension, selected once before weight loading.

A model directory registers an adapter factory for its architectures; the
runner asks for one in `load_model`. With no registration (the default) the
selection returns None and the ordinary runner and cache paths are unchanged.

An adapter subclasses `ModelAdapter`. It owns the input snapshot of each step,
the dummy/profile forward and its KV cache lifecycle; the model forward then
receives explicit kwargs rather than consulting runner state. The optional
hooks (`bind_draft`, `precompile_fused_inputs`, `kv_cache_memory_budget`) are
declared here with no-op defaults, so the runner calls them directly.
"""

import abc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.kv_cache_materializer import LayerKVCache
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class ModelInputSnapshot:
    """What the runner hands an adapter for one forward chunk."""

    token_store: torch.Tensor
    row_start: int
    num_requests: int
    num_tokens_padded: int
    attention_metadata: dict[str, Any]
    input_ids: torch.Tensor


class ModelAdapter(abc.ABC):
    """Model-owned runner extension; subclass it to opt in."""

    @abc.abstractmethod
    def bind_model(self, model: torch.nn.Module) -> None:
        """Called once the target model is loaded."""

    @abc.abstractmethod
    def begin_step(self) -> torch.Tensor:
        """Start a step; returns the token store the step's snapshots carry."""

    @abc.abstractmethod
    def prepare_inputs(self, snapshot: ModelInputSnapshot) -> dict[str, Any]:
        """The model kwargs for one forward chunk."""

    @abc.abstractmethod
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int,
        profile_prefix: int | None = None,
        *,
        dp_lockstep: bool = False,
    ) -> float | None:
        """The dummy/profile forward; returns the wall time in ms when profiling."""

    @abc.abstractmethod
    def get_kv_cache_spec(self) -> dict[str, Any]:
        """The KV cache spec of every layer that needs a cache."""

    @abc.abstractmethod
    def initialize_kv_cache(self, config: Any) -> dict[str, LayerKVCache] | None:
        """Allocate the cache for `config`; returns its live named buffers.

        The runner's ordinary cache bookkeeping is skipped on this path, so
        the adapter must set what the runner's step reads: `kv_cache_config`
        (to `config`), `attn_groups` and `kv_caches` are checked right after
        this call; the input batch's block tables and any mamba slot state
        are the adapter's too. Returning None is only valid without a KV
        connector.
        """

    def bind_draft(self, drafter: Any) -> None:  # noqa: B027
        """Called after the draft model loads; binds the draft's own layers."""

    def precompile_fused_inputs(  # noqa: B027
        self,
        num_tokens_paddings: Sequence[int],
        rows_options: Sequence[int | None],
    ) -> None:
        """Precompile the adapter's input programs (a row option may be None)."""

    def kv_cache_memory_budget(self) -> int | None:
        """HBM bytes the KV cache takes, if the adapter sizes it itself.

        When not None, the worker uses this number as the KV budget in place
        of the measured one; the adapter is responsible for leaving room for
        anything its configuration needs.
        """
        return None


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
        previous = _FACTORIES.get(arch)
        if previous is not None and previous is not factory:
            logger.warning(
                "Model adapter for %s: %r replaces %r", arch, factory, previous
            )
        _FACTORIES[arch] = factory


def select_model_adapter(runner: Any) -> ModelAdapter | None:
    """The registered adapter for the runner's architecture, else None."""
    factory = _FACTORIES.get(runner.model_config.architecture)
    if factory is None:
        return None
    adapter = factory(runner)
    if adapter is not None and not isinstance(adapter, ModelAdapter):
        raise TypeError(
            f"model adapter factory {factory!r} returned "
            f"{type(adapter).__name__}, which does not subclass ModelAdapter"
        )
    return adapter
