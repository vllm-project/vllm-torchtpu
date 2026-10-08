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

IndexShare (GLM-5.2 / DeepSeek-V3.2 DSA drafts): draft step 0 computes the
sparse top-k and steps 1..K-1 reuse it, read from the draft's one shared
``topk_indices_buffer``. The next chunk's step-0 indexer overwrites that
buffer, so the per-step loop puts each chunk's saved rows back from the host
before every step. Here the rows come in as ``shared_topk_indices`` and the
program stores them into the buffer itself before each unrolled step -- the
same in-graph store the indexer already makes into it -- so the K-1 steps
still cost one dispatch. Like every vLLM program this one is compiled once
with its guards dropped, so its first trace must run with ``skip_topk`` True.

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


# `shared_topk_indices` is [p, index_topk]: one saved top-k row per request of
# the loop bucket p, so its dim 0 is the same token dimension the draft marks
# dynamic on every other argument. Left static, the first trace's p would be
# baked into the store, and with the guards dropped every other bucket would
# be handed a program built for the wrong row count.
SHARED_TOPK_DYNAMIC_ARG_DIMS = {"shared_topk_indices": 0}


class _MtpFusedLoop(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: MtpDraftModel,
        num_steps: int,
        topk_indices_buffer: torch.Tensor | None = None,
        vocab_parallel_argmax_info: Any = None,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        # IndexShare only: the draft's top-k table -- the very tensor its MLA
        # layer reads while skip_topk is set. A plain attribute, not a
        # registered buffer: the draft owns it, and this module must not
        # save, move or cast it.
        self.topk_indices_buffer = topk_indices_buffer
        self.vocab_parallel_argmax_info = vocab_parallel_argmax_info

    def _install_shared_topk(self, step: int, rows: torch.Tensor) -> None:
        """Put the chunk's step-0 top-k rows at the front of the table before
        loop step `step` reads it: the in-graph twin of
        `Eagle3Proposer._mtp_restore_topk`."""
        assert self.topk_indices_buffer is not None
        self.topk_indices_buffer[: rows.shape[0]] = rows

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        shared_topk_indices: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """`positions` and `seq_lens` are step 1's. `shared_topk_indices` is
        the chunk's compacted step-0 top-k rows, [p, index_topk]; it is
        required exactly when the program was built with a top-k buffer.
        Returns the K-1 draft tokens, one [p] int32 tensor per step."""
        if (shared_topk_indices is None) != (self.topk_indices_buffer is None):
            raise ValueError(
                "shared_topk_indices must be passed exactly when the fused MTP "
                "loop was built with a topk_indices_buffer (IndexShare)."
            )
        from vllm_torchtpu.spec_decode.utils import vocab_parallel_argmax

        tokens = []
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            if shared_topk_indices is not None:
                self._install_shared_topk(i + 1, shared_topk_indices)
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
            logits = self.mtp.compute_logits(last_hidden)
            if self.vocab_parallel_argmax_info is not None:
                input_ids = vocab_parallel_argmax(
                    logits, self.vocab_parallel_argmax_info
                ).clone()
            else:
                input_ids = logits.argmax(dim=-1).to(torch.int32).clone()
            tokens.append(input_ids)
        return tuple(tokens)


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    topk_indices_buffer: torch.Tensor | None = None,
    vocab_parallel_argmax_info: Any = None,
) -> nn.Module:
    """Compile `_MtpFusedLoop` with the draft's own dynamic dims (Qwen3.5's
    MTP marks positions -1, DeepSeek's 0), so both programs specialise the
    token dimension the same way. With `topk_indices_buffer` (IndexShare) the
    program also takes the chunk's top-k rows, dynamic on the same dim."""
    dims = dict(dynamic_arg_dims)
    if topk_indices_buffer is not None:
        dims.update(SHARED_TOPK_DYNAMIC_ARG_DIMS)
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type("MtpFusedLoop", (_MtpFusedLoop,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        num_steps=num_steps,
        topk_indices_buffer=topk_indices_buffer,
        vocab_parallel_argmax_info=vocab_parallel_argmax_info,
    )
