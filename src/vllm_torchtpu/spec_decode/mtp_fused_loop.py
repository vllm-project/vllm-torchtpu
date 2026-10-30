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
"""An MTP draft's loop steps 1..K-1 as one compiled program.

Each loop step used to be two host dispatches -- the draft forward and the
lm-head + argmax -- plus a metadata build, so K=3 paid for four programs and
two builds between step 0 and the verify. The steps differ only in values the
device can derive itself: positions and attention seq_lens advance by one, and
the input ids are the previous step's argmax. This program unrolls the steps
and advances both on device, so the host builds step 1's metadata once and
dispatches once.

Built with ``propose_step0=True``, the program also takes over step 0's
lm-head + argmax: it is handed step 0's gathered hidden carry instead of step
0's token, proposes that token first, and returns the whole ``[p, K]`` draft
block stacked, so nothing but the post-step-0 bridge runs between the step-0
forward and this program.

seq_lens reaches the attention kernel through the forward context, not as an
argument, so the program is handed the same tensor the step-1 metadata holds
and bumps it in place between steps; the traced attention then reads the
bumped value. The caller must pass a tensor built for this call alone.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from typing import Any, Protocol

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors


class MtpDraftModel(Protocol):
    """What the loop calls on the MTP draft. The draft is an nn.Module, but
    nn.Module types its attributes as Tensor | Module, so name the calls."""

    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
        greedy_head_owner: Any = None,
        propose_step0: bool = False,
    ) -> None:
        """`greedy_head_owner`: a module with `logits_processor` and `lm_head`
        (greedy_top_tokens.resolve_head_owner, or eagle3's draft head proxy)
        whose `get_top_tokens` replaces compute_logits + argmax; None keeps
        compute_logits + argmax. `propose_step0`: see the module docstring.
        """
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.greedy_head_owner = greedy_head_owner
        self.propose_step0 = propose_step0

    def _propose(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The greedy draft token of each row, [p] int32."""
        owner = self.greedy_head_owner
        if owner is not None:
            return owner.logits_processor.get_top_tokens(
                owner.lm_head, hidden_states
            ).to(torch.int32)
        return self.mtp.compute_logits(hidden_states).argmax(dim=-1).to(torch.int32)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """`positions` and `seq_lens` are step 1's and `hidden_states` is step
        0's carry [p, H].

        Without `propose_step0`, `input_ids` is step 0's token and the result
        is the K-1 draft tokens, one [p] int32 tensor per step. With it,
        `input_ids` must be None, step 0's token is proposed from
        `hidden_states`, and the result is the stacked [p, K] int32 drafts.
        """
        tokens = []
        if self.propose_step0:
            assert input_ids is None, "propose_step0 proposes step 0's token"
            input_ids = self._propose(hidden_states)
            tokens.append(input_ids)
        assert input_ids is not None
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            out = type(self.mtp).forward(
                self.mtp,
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=i + 1,
            )
            # Same normalisation as EagleProposer._unwrap_model_out.
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if isinstance(out, (list, tuple)):
                last_hidden, hidden_states = out
            else:
                last_hidden = hidden_states = out
            input_ids = self._propose(last_hidden)
            tokens.append(input_ids)
        if self.propose_step0:
            return torch.stack(tokens, dim=1)
        return tuple(tokens)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    greedy_head_owner: Any = None,
    propose_step0: bool = False,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way. vLLM's compile wrapper skips a None
    argument's dims, so `input_ids=None` under `propose_step0` is fine.
    """
    base = _MtpFusedLoop
    cls = support_torch_compile(dynamic_arg_dims=dict(dynamic_arg_dims))(
        type("MtpFusedLoop", (base,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        greedy_head_owner=greedy_head_owner,
        propose_step0=propose_step0,
    )
