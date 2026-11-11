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
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs


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
        use_mk: bool = False,
    ) -> None:
        super().__init__()
        # Own compile-cache folder, as in MtpLoopStepModel.
        self.prefix = prefix
        self.mtp = mtp
        self.num_steps = num_steps
        self.use_mk = use_mk

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
        # VLLM_TPU_MK_DRAFT: each step is one kernel call returning the carry and the
        # greedy token (attached before the first trace, so a static choice here).
        # The MK choice is a construction-time attribute, never a branch on the token
        # count: vLLM traces one graph for every compile range, so a shape branch would
        # freeze at the first traced bucket (za9g ShapeSpecializationError). The MK
        # variant is a separate program compiled only at the kernel's rows
        # (`mk_draft_compile_config`); the caller picks it per step in Python.
        mk_step = getattr(self.mtp, "mk_draft_step", None) if self.use_mk else None
        for i in range(self.num_steps):
            if i > 0:
                positions = positions + 1
                seq_lens.add_(1)
            if mk_step is not None:
                hidden_states, input_ids = mk_step(input_ids, positions, hidden_states, 1)
                tokens.append(input_ids)
                continue
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
            input_ids = (
                self.mtp.compute_logits(last_hidden).argmax(dim=-1).to(torch.int32)
            )
            tokens.append(input_ids)
        return tuple(tokens)


def mk_draft_rows_ok(inner: Any, rows: int, rows_per_request: int) -> bool:
    """VLLM_TPU_MK_DRAFT: the draft kernel takes this traced row count (`max_rows` of
    the attached MkDraft per T); larger compile ranges use the stock draft step."""
    mk = getattr(inner, "mk_draft", None)
    if mk is None:
        return False
    cap = getattr(mk, "max_rows", {}).get(rows_per_request, 0)
    return 0 < rows <= cap


def mk_draft_compile_config(vllm_config: VllmConfig, inner: Any, rows_per_request: int):
    """`(config, sizes)`: the draft config with compile sizes cut to the draft
    kernel's rows (`MkDraft.max_rows`), so the MK draft programs are never built at a
    prefill-sized range (VMEM: 1872 MiB at 8192 rows, za9e j-c6b8b17f). Empty sizes:
    no MK program."""
    import copy

    cap = getattr(getattr(inner, "mk_draft", None), "max_rows", {}).get(rows_per_request, 0)
    sizes = [s for s in (vllm_config.compilation_config.compile_sizes or []) if 0 < s <= cap]
    cfg = copy.copy(vllm_config)
    cc = copy.copy(vllm_config.compilation_config)
    cc.compile_sizes = list(sizes)
    # No open range either: vLLM appends max_num_batched_tokens to the range
    # endpoints, and an MK program must never be traced past the kernel's rows.
    cc.compile_ranges_endpoints = []
    # Built after the stock draft programs compiled: by then the shared inductor config
    # carries the backend's pre-grad pass, which a fresh backend asserts absent
    # (vllm backends.configure_post_pass; exos j-87ff00d1). Give the MK programs their
    # own copy without it.
    cc.inductor_compile_config = {
        k: v for k, v in dict(cc.inductor_compile_config or {}).items()
        if k != "pre_grad_custom_pass"
    }
    object.__setattr__(cfg, "compilation_config", cc)
    return cfg, sizes


def build_mtp_fused_loop(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    num_steps: int,
    dynamic_arg_dims: dict,
    use_mk: bool = False,
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
    with set_current_vllm_config(vllm_config):
        return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps,
                   use_mk=use_mk)


class _MkDraftStep0(nn.Module):
    """VLLM_TPU_MK_DRAFT: draft step 0 as one kernel call over the verify rows (4 per
    request) plus the per-request carry: `(hidden carry [p], token 0 [p] int32,
    positions carry [p])` at the last accepted row of each request (`rows`). Replaces
    the draft forward, `_draft_rows_carry_and_token`'s position gather and its
    lm-head + argmax. Only for uniform verify steps (the whole-step's), where every
    request has the same 4 rows."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", mtp: MtpDraftModel,
                 rows_per_request: int) -> None:
        super().__init__()
        self.prefix = prefix
        self.mtp = mtp
        self.rows_per_request = rows_per_request

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_rows: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        carry, tokens = self.mtp.mk_draft_step(  # type: ignore[attr-defined]
            input_ids, positions, hidden_states, self.rows_per_request
        )
        rows = draft_rows.clamp_min(0)
        return (
            torch.index_select(carry, 0, rows),
            torch.index_select(tokens, 0, rows),
            torch.index_select(positions, positions.ndim - 1, rows),
        )


def build_mk_draft_step0(
    *,
    vllm_config: VllmConfig,
    prefix: str,
    mtp: MtpDraftModel,
    rows_per_request: int,
    dynamic_arg_dims: dict,
) -> nn.Module:
    """Compile `_MkDraftStep0` with the draft's own dynamic dims (as the fused loop)."""
    step_args = inspect.signature(_MkDraftStep0.forward).parameters
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in step_args}
    cls = support_torch_compile(dynamic_arg_dims=dims)(
        type("MkDraftStep0", (_MkDraftStep0,), {})
    )
    # The compile wrapper builds its backend from the *current* config, not the
    # vllm_config argument: pin it here so the caller's context cannot widen the
    # compile sizes (za9j j-d4040629).
    with set_current_vllm_config(vllm_config):
        return cls(vllm_config=vllm_config, prefix=prefix, mtp=mtp,
                   rows_per_request=rows_per_request)
