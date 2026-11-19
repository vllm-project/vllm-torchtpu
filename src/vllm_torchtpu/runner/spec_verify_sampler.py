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
"""Optional model-supplied sampler for the non-greedy spec-decode verify.

A target model may set the attribute ``spec_verify_sampler``
(`SPEC_VERIFY_SAMPLER_ATTR`) to an object with the `SpecVerifySampler`
methods. The runner then hands it the non-greedy, grammar-free verify of each
chunk (bonus sampling and rejection) instead of running its own programs. The
runner still builds the per-request sampling parameters and takes the random
draws, in the order its own path takes them, so a sampler that computes the
same math returns the same tokens.

With no attribute (the default, every model on main) the runner's own verify
runs unchanged.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

# Target-model attribute; None or unset keeps the runner's verify.
SPEC_VERIFY_SAMPLER_ATTR = "spec_verify_sampler"


@dataclass(frozen=True, kw_only=True)
class SpecVerifyInputs:
    """One verify chunk, as the runner's own non-greedy path would see it.

    `temperatures`, `top_k`, `top_p` are per request ([num_reqs, 1], from
    `_build_padded_sampling_params`). `bonus_u` [num_reqs, vocab], `accept_u`
    (shaped like `draft_token_ids`) and `recover_u` [num_tokens, vocab] are
    the draws the runner's path takes, in its order. `topk_candidates` and
    `apply_topk_topp` are the runner's per-step sampler choices;
    `any_top_k` is False when no running request sets top-k.
    """

    logits: torch.Tensor
    bonus_logits_indices: torch.Tensor
    target_logits_indices: torch.Tensor
    draft_token_ids: torch.Tensor
    num_draft_tokens: torch.Tensor
    segment_ids: torch.Tensor
    group_indices: torch.Tensor
    temperatures: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    bonus_u: torch.Tensor
    accept_u: torch.Tensor
    recover_u: torch.Tensor
    synthetic_conditional_rates: torch.Tensor | None
    synthetic_mode: bool
    max_draft_tokens: int
    topk_candidates: int
    apply_topk_topp: bool
    any_top_k: bool


@runtime_checkable
class SpecVerifySampler(Protocol):
    def sample(self, inputs: SpecVerifyInputs) -> torch.Tensor:
        """The chunk's [num_reqs, max_draft_tokens + 1] tokens, as the
        runner's rejection sampler returns them."""
        ...

    def precompile(
        self,
        inputs: SpecVerifyInputs,
        sampler_variants: Sequence[tuple[int, bool]],
    ) -> None:
        """Compile every program `sample` can pick for these input shapes.

        `sampler_variants` lists the runner's (topk_candidates,
        apply_topk_topp) choices; `inputs` holds zero-filled dummies.
        """
        ...


@runtime_checkable
class SupportsSpecVerifySampler(Protocol):
    """Capability of a target model that supplies its own verify sampler:
    the attribute ``spec_verify_sampler`` (`SPEC_VERIFY_SAMPLER_ATTR`), a
    `SpecVerifySampler` or None for the runner's verify."""

    spec_verify_sampler: SpecVerifySampler | None


def model_spec_verify_sampler(model: Any) -> SpecVerifySampler | None:
    """`model.spec_verify_sampler`, or None when the model does not declare
    the capability or sets None."""
    if not isinstance(model, SupportsSpecVerifySampler):
        return None
    sampler = model.spec_verify_sampler
    if sampler is not None and not isinstance(sampler, SpecVerifySampler):
        raise TypeError(
            f"{SPEC_VERIFY_SAMPLER_ATTR} must provide sample() and precompile(), "
            f"got {type(sampler).__name__}"
        )
    return sampler
