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
"""Optional model-owned EAGLE/MTP draft proposer, keyed on the draft architecture.

A model directory that keeps its own copy of the draft proposer registers a
factory for its draft model's architectures; the runner asks for one where it
builds the EAGLE/MTP proposer. With no registration (the default) the selection
returns None and the runner builds the shared `Eagle3Proposer` as before.

The factory is called as `factory(runner, vllm_config)` and must return a
drop-in replacement for `Eagle3Proposer`: the runner type-checks its drafter,
so the replacement subclasses it (checked at startup). Beyond the methods,
shared code reads or writes these `Eagle3Proposer` attributes, so a
replacement must keep them with the same meaning:

- `draft_vllm_config` (set by `load_model`): its
  `compilation_config.compile_sizes` are the draft's compiled buckets
  (`TPUModelRunner` sets them in `capture_model` and checks the chunk
  buckets against them);
- `draft_model` (set by `load_model`): the draft module, whose
  `model.layers` hold the draft's attention layers. The runner relayouts its
  tables, patches its rope caches, prebuilds its ops and arms the MoE TP
  decode kernel on it; `KVCacheManager` checks the KV sharing of its layers;
  `layers/adapter/fused_moe_decode.py` arms the fused MoE decode kernel on
  it. Several of these reads use `getattr(..., None)` (they also serve
  proposers without a draft module), so a replacement without it skips
  them silently;
- `_draft_attn_layer_names`: the draft's attention layers, which the
  runner's `_initialize_attention_kernels`, the TPU KV connector and the
  runner's KV transfer include;
- `draft_chunks` (written by the runner): the per-chunk draft inputs of the
  step, set before the forward and cleared after the spec-decode warmup.
"""

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from vllm.config import VllmConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

DraftProposerFactory = Callable[["TPUModelRunner", VllmConfig], Eagle3Proposer]

_FACTORIES: dict[str, DraftProposerFactory] = {}


def register_draft_proposer(
    draft_architectures: Iterable[str], factory: DraftProposerFactory
) -> None:
    """Build the draft proposer of `draft_architectures` with `factory`.

    Called from a model directory's registration hook. A later registration
    for the same architecture replaces the earlier one.

    Raises:
        ValueError: `draft_architectures` is a bare string or empty.
    """
    if isinstance(draft_architectures, str):
        raise ValueError(
            "draft_architectures must be an iterable of architecture names, "
            f"not the string {draft_architectures!r}"
        )
    draft_architectures = list(draft_architectures)
    if not draft_architectures:
        raise ValueError("draft_architectures must name at least one architecture")
    for arch in draft_architectures:
        previous = _FACTORIES.get(arch)
        if previous is not None and previous is not factory:
            logger.warning(
                "Draft proposer for %s: %r replaces %r", arch, factory, previous
            )
        _FACTORIES[arch] = factory


def select_draft_proposer(vllm_config: VllmConfig) -> DraftProposerFactory | None:
    """The factory registered for the draft model's architecture, else None.

    Only called for EAGLE/MTP, whose speculative config always has a draft
    model config; a missing one fails loudly.
    """
    if not _FACTORIES:
        return None
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None, (
        "EAGLE/MTP speculative decoding needs a draft model config"
    )
    # vLLM resolves the draft model to one architecture; key on that one, as
    # the block size utils do, so the proposer matches the loaded model.
    return _FACTORIES.get(spec.draft_model_config.architecture)


def build_draft_proposer(
    runner: "TPUModelRunner", vllm_config: VllmConfig
) -> Eagle3Proposer:
    """Build the EAGLE/MTP proposer: the registered one, else `Eagle3Proposer`.

    The runner gates draft compile sizes, draft-bucket checks and the KV
    connector's draft layers on `isinstance(drafter, Eagle3Proposer)`, so a
    factory that returns anything else is rejected here, at startup.
    """
    factory = select_draft_proposer(vllm_config)
    if factory is None:
        return Eagle3Proposer(runner, vllm_config)
    drafter = factory(runner, vllm_config)
    if not isinstance(drafter, Eagle3Proposer):
        raise TypeError(
            f"draft proposer factory {factory!r} returned "
            f"{type(drafter).__name__}, which does not subclass Eagle3Proposer"
        )
    logger.info(
        "Draft proposer: %s.%s (registered)",
        type(drafter).__module__,
        type(drafter).__qualname__,
    )
    return drafter
