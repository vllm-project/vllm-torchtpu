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
"""Stop charging DFlash/DSpark drafting slots against the step token budget.

vLLM's scheduler charges every scheduled request
``SpeculativeConfig.max_num_new_slots_for_drafting`` tokens on top of its
scheduled tokens against ``max_num_batched_tokens``: K for DFlash, K - 1 for
DSpark. That room is for drafters whose first pass appends those query slots
to the target's tokens in one forward, such as upstream's generic
parallel-drafting path (``LLMBaseProposer.set_inputs_first_pass``).

The TPU DFlash and DSpark proposers do not. Target hidden states go through a
KV-update pass at the target's token bucket, and the draft forward is
``num_reqs * block_size`` tokens (``DFlashProposer._prepare_dflash_inputs``).
On TPU the charge only takes prefill tokens out of each step, so for these
methods the property returns 0. The scheduled-token cap, and with it the
largest target bucket, stays ``max_num_batched_tokens``. Other methods keep
the upstream value.

The charge was also what kept ``num_reqs * block_size`` within
``max_num_batched_tokens``: without it, a request scheduled fewer than
``block_size`` tokens still drafts a full block. The proposer pads the draft
forward to its own buckets, which extend past the target's to a full chunk
of requests (``DFlashProposer._draft_token_paddings``).
"""

from functools import wraps

# Methods tpu_runner serves with DFlashProposer or its DSpark subclass.
_TPU_BLOCK_DRAFT_METHODS = ("dflash", "dspark")

_PATCH_MARKER = "_tpu_drafting_slots_patch"


def patch_drafting_slot_budget() -> None:
    # TODO: let the platform declare the per-request drafting slots upstream
    # instead of overriding the SpeculativeConfig property here.
    from vllm.config.speculative import SpeculativeConfig

    upstream = SpeculativeConfig.__dict__["max_num_new_slots_for_drafting"]
    assert isinstance(upstream, property) and upstream.fget is not None
    if getattr(upstream.fget, _PATCH_MARKER, False):
        return
    upstream_fget = upstream.fget

    @wraps(upstream_fget)
    def max_num_new_slots_for_drafting(self) -> int:
        if self.method in _TPU_BLOCK_DRAFT_METHODS:
            return 0
        return upstream_fget(self)

    setattr(max_num_new_slots_for_drafting, _PATCH_MARKER, True)
    SpeculativeConfig.max_num_new_slots_for_drafting = property(
        max_num_new_slots_for_drafting
    )
