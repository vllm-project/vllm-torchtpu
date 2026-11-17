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
"""An MTP draft's loop steps 1..K-1 (or step 0..K-1) as one compiled program.

Each loop step used to be two host dispatches -- the draft forward and the
lm-head + argmax -- plus a metadata build, so K=3 paid for four programs and
two builds between step 0 and the verify. The steps differ only in values the
device can derive itself: positions and attention seq_lens advance by one, and
the input ids are the previous step's argmax. This program unrolls the steps
and advances both on device, so the host builds step 1's metadata once and
dispatches once.

Built with ``include_step0=True`` (`TPU_DRAFT_STEP0_IN_LOOP=1`), `_MtpFullDraftLoop`
also fuses step 0's forward, the post-step-0 gather bridge, step 0's token
proposal, and loop steps 1..K-1 into one compiled program, returning the
stacked ``[p, K]`` draft block directly.

seq_lens reaches the attention kernel through the forward context, not as an
argument, so the program is handed the same tensor the step-1 metadata holds
and bumps it in place between steps; the traced attention then reads the
bumped value. The caller must pass a tensor built for this call alone.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read.
"""

from typing import Any, Protocol, cast

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


class _MtpFusedLoopBase(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps

    def _run_loop(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        for step_idx in range(self.num_steps):
            if step_idx > 0:
                positions = positions + 1
                seq_lens.add_(1)
            out = type(self.mtp).forward(
                self.mtp,
                input_ids,
                positions,
                hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=step_idx + 1,
            )
            # Same normalisation as EagleProposer._unwrap_model_out.
            if isinstance(out, (list, tuple)) and len(out) == 1:
                out = out[0]
            if isinstance(out, (list, tuple)):
                last_hidden, hidden_states = out
            else:
                last_hidden = hidden_states = out
            input_ids = (
                self.mtp.compute_logits(last_hidden).argmax(dim=-1).to(torch.int32)
            )
            tokens.append(input_ids)
        return tokens


class _MtpFusedLoop(_MtpFusedLoopBase):
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
        tokens = self._run_loop(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            seq_lens=seq_lens,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return tuple(tokens)


class _MtpFullDraftLoop(_MtpFusedLoopBase):
    """Merged MTP draft step 0 + post-step-0 bridge + loop steps 1..K-1 in one
    compiled program (`TPU_DRAFT_STEP0_IN_LOOP=1`)."""

    def forward(
        self,
        step0_input_ids: torch.Tensor,
        step0_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        step0_gather_indices: torch.Tensor,
        step0_seq_lens: torch.Tensor,
        step0_query_start_loc: torch.Tensor,
        step0_request_distribution: torch.Tensor,
        num_rejected_dev: torch.Tensor,
        loop_query_start_loc: torch.Tensor,
        loop_request_distribution: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        attn_metadata = forward_ctx.attn_metadata
        saved_fields: list[tuple[Any, Any, Any, Any]] = []
        seen_ids: set[int] = set()
        if isinstance(attn_metadata, dict):
            # The concrete metadata classes carry these fields; the base
            # AttentionMetadata type does not declare them.
            for md_obj in cast(dict[str, Any], attn_metadata).values():
                md_id = id(md_obj)
                if md_id not in seen_ids:
                    seen_ids.add(md_id)
                    saved_fields.append(
                        (
                            md_obj,
                            md_obj.seq_lens,
                            md_obj.query_start_loc,
                            md_obj.request_distribution,
                        )
                    )
                    md_obj.seq_lens = step0_seq_lens
                    md_obj.query_start_loc = step0_query_start_loc
                    md_obj.request_distribution = step0_request_distribution
        try:
            out0 = type(self.mtp).forward(
                self.mtp,
                step0_input_ids,
                step0_positions,
                target_hidden_states,
                intermediate_tensors,
                inputs_embeds,
                spec_step_idx=0,
            )
            if isinstance(out0, (list, tuple)) and len(out0) == 1:
                out0 = out0[0]
            if isinstance(out0, (list, tuple)):
                last_hidden0, hidden0 = out0
                last_hidden_carry = torch.index_select(
                    last_hidden0, 0, step0_gather_indices
                )
                hidden_carry = torch.index_select(hidden0, 0, step0_gather_indices)
            else:
                hidden_carry = torch.index_select(out0, 0, step0_gather_indices)
                last_hidden_carry = hidden_carry

            pos_carry = (
                torch.index_select(step0_positions, 1, step0_gather_indices)
                if step0_positions.ndim == 2
                else torch.index_select(step0_positions, 0, step0_gather_indices)
            )
            pos_step1 = pos_carry + 1
            seq_lens_step1 = (step0_seq_lens + 1 - num_rejected_dev).clone()

            for md_obj, _orig_sl, _orig_qsl, _orig_rd in saved_fields:
                md_obj.seq_lens = seq_lens_step1
                md_obj.query_start_loc = loop_query_start_loc
                md_obj.request_distribution = loop_request_distribution

            tok0 = (
                self.mtp.compute_logits(last_hidden_carry)
                .argmax(dim=-1)
                .to(torch.int32)
            )
            loop_tokens = self._run_loop(
                input_ids=tok0,
                positions=pos_step1,
                hidden_states=hidden_carry,
                seq_lens=seq_lens_step1,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            )
            return torch.stack([tok0, *loop_tokens], dim=1)
        finally:
            for md_obj, orig_sl, orig_qsl, orig_rd in saved_fields:
                md_obj.seq_lens = orig_sl
                md_obj.query_start_loc = orig_qsl
                md_obj.request_distribution = orig_rd


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    include_step0: bool = False,
) -> nn.Module:
    """Compile `_MtpFusedLoop` (or `_MtpFullDraftLoop` when `include_step0` is
    True) with the draft's own dynamic dims."""
    if include_step0:
        base: type[_MtpFusedLoopBase] = _MtpFullDraftLoop
        cls_name = "MtpFullDraftLoop"
    else:
        base = _MtpFusedLoop
        cls_name = "MtpFusedLoop"
    cls = support_torch_compile(dynamic_arg_dims=dict(dynamic_arg_dims))(
        type(cls_name, (base,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
    )
