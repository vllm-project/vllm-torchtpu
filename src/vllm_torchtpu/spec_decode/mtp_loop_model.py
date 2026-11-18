# SPDX-License-Identifier: Apache-2.0
"""A second compiled program for an MTP draft's IndexShare loop steps.

vLLM compiles a model once and then drops the guards that would recompile it
(``TorchCompileWithNoGuardsWrapper``), so anything the forward decides from a
plain Python value is fixed by the first trace. A DSA MTP layer's attention
decides from ``skip_topk`` whether to run its indexer or reuse the top-k table,
and IndexShare needs both: the indexer at draft step 0, the table at steps 1+.
One compiled program cannot do both, so steps 1+ run this one: the same draft
forward over the same weights, KV caches and top-k table, first traced while
``skip_topk`` is True.

This lives outside eagle3.py because that module uses postponed annotations,
which ``support_torch_compile`` cannot read. The dims are spelled out below
anyway, so nothing here depends on annotation style.
"""

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

# What `support_torch_compile` infers for `DeepSeekMTP.forward`: dim 0, the
# token dimension, of every tensor argument. The proposer checks the draft's
# own dims against this before building the program.
MTP_DYNAMIC_ARG_DIMS: dict[str, int | list[int] | dict[int, str]] = {
    "input_ids": 0,
    "positions": 0,
    "hidden_states": 0,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
}


@support_torch_compile(dynamic_arg_dims=MTP_DYNAMIC_ARG_DIMS)
class MtpLoopStepModel(nn.Module):
    """Runs an MTP draft's own ``forward`` as a separately compiled program."""

    def __init__(
        self, *, vllm_config: VllmConfig, prefix: str = "", mtp: nn.Module
    ) -> None:
        # vllm_config is unused here: the compile decorator reads it and sets
        # self.vllm_config. Declaring it lets type checkers see the keyword the
        # decorator's wrapper accepts.
        super().__init__()
        # Read by the plugin's compile_prefix patch (vllm_torchtpu/__init__.py):
        # this program gets its own compile-cache folder, with the same name on
        # every start, instead of sharing the step-0 program's.
        self.prefix = prefix
        self.mtp = mtp

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # The draft class's forward, called unbound: this program runs exactly
        # what the step-0 program runs, and Dynamo traces it inline instead of
        # going through the draft's own compiled wrapper.
        return type(self.mtp).forward(
            self.mtp,
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )


def decode_rows_count(tokens, tokens_per_request):
    """Rows MTP draft step 0 evaluates for a ``tokens``-token bucket
    (VLLM_TPU_MTP_DRAFT_DECODE_ROWS): the bucket's request bound in a pure
    decode step, ``tokens / tokens_per_request`` (K + 1), rounded up to a
    multiple of 16. Takes an int or the traced token count."""
    step = 16 * tokens_per_request
    return (tokens + step - 1) // step * 16


class MtpStep0RowsModel(nn.Module):
    """An MTP draft's step 0 at selected rows as a separately compiled
    program (VLLM_TPU_MTP_DRAFT_DECODE_ROWS, VLLM_TPU_MTP_DRAFT_CHECK).

    Each kind has its own ``forward``: vLLM keys a module's AOT artifact by the
    forward's qualified name and line, not by the instance.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        mtp: nn.Module,
        tokens_per_request: int = 0,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.tokens_per_request = tokens_per_request

    def _step0(self, input_ids, positions, hidden_states, draft_rows):
        return type(self.mtp).forward(
            self.mtp,
            input_ids,
            positions,
            hidden_states,
            None,
            None,
            0,
            draft_rows,
        )


class MtpStep0DecodeRows(MtpStep0RowsModel):
    """``draft_rows`` is token-length (dynamic, one entry per token of the
    bucket, -1 past the requests); the layer evaluates its first
    ``decode_rows_count(T, tokens_per_request)`` entries."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_rows: torch.Tensor,
    ) -> torch.Tensor:
        rows = draft_rows[
            : decode_rows_count(draft_rows.shape[0], self.tokens_per_request)
        ]
        return self._step0(input_ids, positions, hidden_states, rows)


class MtpStep0CapacityRows(MtpStep0RowsModel):
    """``draft_rows`` at one static length, as the draft's own program takes
    it (the VLLM_TPU_MTP_DRAFT_CHECK reference)."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_rows: torch.Tensor,
    ) -> torch.Tensor:
        return self._step0(input_ids, positions, hidden_states, draft_rows)


def build_mtp_step0_rows_model(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: nn.Module,
    dynamic_arg_dims: dict,
    tokens_per_request: int,
) -> nn.Module:
    """Compile ``MtpStep0DecodeRows`` (``tokens_per_request`` > 0) or
    ``MtpStep0CapacityRows`` with the draft's own dynamic dims, plus
    ``draft_rows`` dim 0 when it is token-length."""
    base: type[MtpStep0RowsModel] = MtpStep0CapacityRows
    dims = {
        k: v
        for k, v in dict(dynamic_arg_dims).items()
        if k in ("input_ids", "positions", "hidden_states")
    }
    if tokens_per_request:
        base = MtpStep0DecodeRows
        dims["draft_rows"] = 0
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type(base.__name__, (base,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
        tokens_per_request=tokens_per_request,
    )
