# SPDX-License-Identifier: Apache-2.0
"""A second compiled program for an MTP draft's loop steps 1+.

vLLM compiles a model once and then drops the guards that would recompile it
(``TorchCompileWithNoGuardsWrapper``), so anything the forward decides from a
plain Python value is fixed by the first trace. A DSA MTP layer's attention
decides from ``skip_topk`` whether to run its indexer or reuse the top-k table,
and IndexShare needs both: the indexer at draft step 0, the table at steps 1+.
One compiled program cannot do both, so steps 1+ run this one: the same draft
forward over the same weights, KV caches and top-k table, first traced while
``skip_topk`` is True.

The MTP step-0 post-attention gather (``TPU_MTP_STEP0_POST_ATTN_GATHER``) has
the same shape of problem: step 0 traces with the gather, steps 1+ without.
``build_mtp_loop_step_model`` compiles this program with the draft's own
dynamic dims (Qwen3.5 marks ``positions`` -1 for MRoPE), which the fixed
``MTP_DYNAMIC_ARG_DIMS`` of the IndexShare program do not cover.

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
            spec_step_idx=spec_step_idx,
        )


def build_mtp_loop_step_model(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: nn.Module,
    dynamic_arg_dims: dict,
) -> nn.Module:
    """Compile a separate loop-step program with the draft's own dynamic dims
    (e.g. Qwen3.5 marks `positions` dim -1 for MRoPE, DeepSeek marks dim 0)."""
    cls = support_torch_compile(dynamic_arg_dims=dict(dynamic_arg_dims))(
        type("MtpLoopStepModel", (MtpLoopStepModel,), {})
    )
    return cls(
        vllm_config=vllm_config,
        prefix=prefix,
        mtp=mtp,
    )
