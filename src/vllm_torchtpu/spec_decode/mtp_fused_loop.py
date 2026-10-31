# SPDX-License-Identifier: Apache-2.0
"""An MTP draft's loop steps 1..K-1 as one compiled program.

Each loop step used to be two host dispatches -- the draft forward and the
lm-head + argmax -- plus a metadata build, so K=3 paid for four programs and
two builds between step 0 and the verify. The steps differ only in values the
device can derive itself: positions and attention seq_lens advance by one, and
the input ids are the previous step's argmax. This program unrolls the steps
and advances both on device, so the host builds step 1's metadata once and
dispatches once.

seq_lens reaches the attention kernel through the forward context, not as an
argument, so the program is handed the same tensor the step-1 metadata holds
and bumps it in place between steps; the traced attention then reads the
bumped value. The caller must pass a tensor built for this call alone.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

import inspect
from typing import Any, Protocol

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.runner import greedy_top_tokens


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
        head_owner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        # Keep out of nn.Module._modules when head_owner is `mtp` itself.
        object.__setattr__(self, "_head_owner", head_owner)
        # VLLM_TPU_MTP_LOOP_REBIND_SEQ_LENS: advance seq_lens by rebinding the
        # draft metadata's field to `seq_lens + i` in the graph instead of
        # `seq_lens.add_(1)`, so the program mutates no input and AOT emits no
        # copy-back epilogue for it.
        self.rebind_seq_lens = envs.VLLM_TPU_MTP_LOOP_REBIND_SEQ_LENS

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """`positions` and `seq_lens` are step 1's. Returns the K-1 draft
        tokens, one [p] int32 tensor per step."""
        tokens = []
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                if self.rebind_seq_lens:
                    stepped = seq_lens + i
                    for md in get_forward_context().attn_metadata.values():
                        md.seq_lens = stepped
                else:
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
            if self._head_owner is not None:
                input_ids = greedy_top_tokens.draft_top_token_ids(
                    self._head_owner, last_hidden
                )
            else:
                input_ids = (
                    self.mtp.compute_logits(last_hidden).argmax(dim=-1).to(torch.int32)
                )
            tokens.append(input_ids)
        return tuple(tokens)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    head_owner: nn.Module | None = None,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way."""
    # Keep only the draft dims the loop's forward takes: a draft-only
    # argument (e.g. Qwen4ExpMTP's `draft_rows`) makes support_torch_compile
    # reject the loop, which then silently falls back to per-step dispatch.
    loop_args = inspect.signature(_MtpFusedLoop.forward).parameters
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in loop_args}
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type("MtpFusedLoop", (_MtpFusedLoop,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        head_owner=head_owner,
    )
