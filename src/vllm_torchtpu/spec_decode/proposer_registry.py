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
drop-in replacement for `Eagle3Proposer` (the runner type-checks its drafter,
so the replacement subclasses it).
"""

from collections.abc import Callable, Iterable
from typing import Any

DraftProposerFactory = Callable[[Any, Any], Any]

_FACTORIES: dict[str, DraftProposerFactory] = {}


def register_draft_proposer(
    draft_architectures: Iterable[str], factory: DraftProposerFactory
) -> None:
    """Build the draft proposer of `draft_architectures` with `factory`.

    Called from a model directory's registration hook. A later registration
    for the same architecture replaces the earlier one.
    """
    for arch in draft_architectures:
        _FACTORIES[arch] = factory


def unregister_draft_proposer(draft_architectures: Iterable[str]) -> None:
    for arch in draft_architectures:
        _FACTORIES.pop(arch, None)


def _draft_architectures(vllm_config) -> list[str]:
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    if draft is None:
        return []
    archs = getattr(draft, "architectures", None)
    if archs is None:
        archs = getattr(getattr(draft, "hf_config", None), "architectures", None)
    return list(archs or [])


def select_draft_proposer(vllm_config) -> DraftProposerFactory | None:
    """The factory registered for the draft model's architecture, else None."""
    if not _FACTORIES:
        return None
    for arch in _draft_architectures(vllm_config):
        factory = _FACTORIES.get(arch)
        if factory is not None:
            return factory
    return None
