
try:
    import jax
except ImportError:
    pass
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
"""Torch bridge for the sort-free MoE router top-k.

On TPU the selection runs as the Pallas kernel in ``kernels/router_topk.py``,
elsewhere as the identical plain-torch algorithm in
``layers/core/rowmax_topk.py``.
"""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.router_topk import (
    SCORING_NAME_TO_CODE,
    router_fused_gate_grouped_topk_with_bias,
    router_grouped_topk,
    router_grouped_topk_with_bias,
    router_topk,
    router_topk_fused,
    router_topk_fused_with_bias,
    router_topk_with_bias,
    router_topk_with_raw,
)
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk

# Grid-chunked Pallas router top-k handles up to 64K rows in VMEM.
PALLAS_MAX_ROWS = 65536


def _apply_scoring_torch(scoring_fn: str, logits: torch.Tensor) -> torch.Tensor:
    scores = logits.float()
    if scoring_fn == "none":
        return scores
    if scoring_fn == "softmax":
        return scores.softmax(dim=-1)
    if scoring_fn == "sigmoid":
        return scores.sigmoid()
    if scoring_fn == "sqrtsoftplus":
        import torch.nn.functional as F

        return torch.sqrt(F.softplus(scores))
    raise NotImplementedError(
        f"FusedMoE does not support {scoring_fn} scoring function for TPU."
    )


def _apply_epilogue_torch(
    topk_weights: torch.Tensor,
    renormalize: bool,
    routed_scaling_factor: float,
) -> torch.Tensor:
    if renormalize:
        topk_weights = topk_weights / torch.clamp(
            topk_weights.sum(dim=-1, keepdim=True), min=1e-20
        )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights


def _fake_router_topk(scores: torch.Tensor, topk: int):
    rows = scores.shape[0]
    return (
        torch.empty((rows, topk), dtype=torch.float32, device=scores.device),
        torch.empty((rows, topk), dtype=torch.int32, device=scores.device),
    )


def _fake_router_topk_with_raw(
    scores: torch.Tensor, raw_scores: torch.Tensor, topk: int
):
    rows = scores.shape[0]
    return (
        torch.empty((rows, topk), dtype=torch.float32, device=scores.device),
        torch.empty((rows, topk), dtype=torch.int32, device=scores.device),
    )


def _fake_router_topk_with_bias(
    raw_scores: torch.Tensor, bias: torch.Tensor, topk: int
):
    del bias
    rows = raw_scores.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=raw_scores.device
        ),
        torch.empty((rows, topk), dtype=torch.int32, device=raw_scores.device),
    )


def _fake_router_topk_fused(
    router_logits: torch.Tensor,
    topk: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    del scoring_code, renormalize, routed_scaling_factor
    rows = router_logits.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=router_logits.device
        ),
        torch.empty(
            (rows, topk), dtype=torch.int32, device=router_logits.device
        ),
    )


def _fake_router_topk_fused_with_bias(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    del bias, scoring_code, renormalize, routed_scaling_factor
    rows = router_logits.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=router_logits.device
        ),
        torch.empty(
            (rows, topk), dtype=torch.int32, device=router_logits.device
        ),
    )


def _fake_router_grouped_topk(
    router_logits: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    del num_expert_group, topk_group, scoring_code, renormalize, routed_scaling_factor
    rows = router_logits.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=router_logits.device
        ),
        torch.empty(
            (rows, topk), dtype=torch.int32, device=router_logits.device
        ),
    )


def _fake_router_grouped_topk_with_bias(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    del (
        bias,
        num_expert_group,
        topk_group,
        scoring_code,
        renormalize,
        routed_scaling_factor,
    )
    rows = router_logits.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=router_logits.device
        ),
        torch.empty(
            (rows, topk), dtype=torch.int32, device=router_logits.device
        ),
    )


def _fake_router_fused_gate_grouped_topk_with_bias(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
):
    del (
        gate_weight,
        bias,
        num_expert_group,
        topk_group,
        scoring_code,
        renormalize,
        routed_scaling_factor,
    )
    rows = hidden_states.shape[0]
    return (
        torch.empty(
            (rows, topk), dtype=torch.float32, device=hidden_states.device
        ),
        torch.empty(
            (rows, topk), dtype=torch.int32, device=hidden_states.device
        ),
    )


# Registered at import, not lazily: this op is called from inside the MoE
# forward, and Dynamo cannot trace a lock-guarded registration there.
_op = pallas.jax_op("pallas::moe_router_topk", router_topk)
_op.register_fake(_fake_router_topk)

_op_with_raw = pallas.jax_op(
    "pallas::moe_router_topk_with_raw", router_topk_with_raw
)
_op_with_raw.register_fake(_fake_router_topk_with_raw)

_op_with_bias = pallas.jax_op(
    "pallas::moe_router_topk_with_bias", router_topk_with_bias
)
_op_with_bias.register_fake(_fake_router_topk_with_bias)

_op_fused = pallas.jax_op("pallas::moe_router_topk_fused", router_topk_fused)
_op_fused.register_fake(_fake_router_topk_fused)

_op_fused_with_bias = pallas.jax_op(
    "pallas::moe_router_topk_fused_with_bias", router_topk_fused_with_bias
)
_op_fused_with_bias.register_fake(_fake_router_topk_fused_with_bias)

_op_grouped = pallas.jax_op(
    "pallas::moe_router_grouped_topk", router_grouped_topk
)
_op_grouped.register_fake(_fake_router_grouped_topk)

_op_grouped_with_bias = pallas.jax_op(
    "pallas::moe_router_grouped_topk_with_bias", router_grouped_topk_with_bias
)
_op_grouped_with_bias.register_fake(_fake_router_grouped_topk_with_bias)

_op_fused_gate_grouped_with_bias = pallas.jax_op(
    "pallas::moe_router_fused_gate_grouped_topk_with_bias",
    router_fused_gate_grouped_topk_with_bias,
)
_op_fused_gate_grouped_with_bias.register_fake(
    _fake_router_fused_gate_grouped_topk_with_bias
)


def _pad_rows_to_8(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    orig_rows = x.shape[0]
    pad_r = (-orig_rows) & 7
    if pad_r:
        x = torch.cat((x, torch.zeros((pad_r, *x.shape[1:]), dtype=x.dtype, device=x.device)), dim=0)
    return x, orig_rows, pad_r


def rowmax_select(
    scores: torch.Tensor,
    topk: int,
    raw_scores: torch.Tensor | None = None,
    *,
    bias: torch.Tensor | None = None,
    scoring_fn: str = "none",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores or logits."""
    is_fused = (
        scoring_fn != "none" or renormalize or routed_scaling_factor != 1.0
    )
    if scores.device.type != "tpu":
        s = _apply_scoring_torch(scoring_fn, scores)
        if bias is not None:
            w, i = rowmax_topk(s + bias.float().view(1, -1), topk, raw_scores=s)
        else:
            w, i = rowmax_topk(s, topk, raw_scores=raw_scores)
        return _apply_epilogue_torch(w, renormalize, routed_scaling_factor), i
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    if scores.shape[0] > PALLAS_MAX_ROWS:
        s = _apply_scoring_torch(scoring_fn, scores)
        raw_f32 = raw_scores.float() if raw_scores is not None else None
        if bias is not None:
            w, i = rowmax_topk(s + bias.float().view(1, -1), topk, raw_scores=s)
        else:
            w, i = rowmax_topk(s, topk, raw_scores=raw_f32)
        return _apply_epilogue_torch(w, renormalize, routed_scaling_factor), i
    if is_fused and raw_scores is None:
        logits_in = (
            scores
            if scores.dtype in (torch.bfloat16, torch.float32)
            else scores.float()
        )
        logits_pad, orig_rows, pad_r = _pad_rows_to_8(logits_in)
        scoring_code = SCORING_NAME_TO_CODE[scoring_fn]
        scale = float(routed_scaling_factor)
        if bias is not None:
            w, i = _op_fused_with_bias(
                logits_pad,
                bias.float().view(1, -1),
                topk,
                scoring_code,
                bool(renormalize),
                scale,
            )
        else:
            w, i = _op_fused(
                logits_pad, topk, scoring_code, bool(renormalize), scale
            )
        return (w[:orig_rows], i[:orig_rows]) if pad_r else (w, i)
    scores_pad, orig_rows, pad_r = _pad_rows_to_8(scores.float())
    raw_f32 = raw_scores.float() if raw_scores is not None else None
    if bias is not None:
        w, i = _op_with_bias(scores_pad, bias.float().view(1, -1), topk)
    elif raw_f32 is not None:
        raw_pad, _, _ = _pad_rows_to_8(raw_f32)
        w, i = _op_with_raw(scores_pad, raw_pad, topk)
    else:
        w, i = _op(scores_pad, topk)
    return (w[:orig_rows], i[:orig_rows]) if pad_r else (w, i)


def grouped_rowmax_select(
    router_logits: torch.Tensor,
    *,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-VMEM grouped top-k router selection on TPU."""
    if router_logits.device.type != "tpu" or router_logits.shape[0] > PALLAS_MAX_ROWS:
        from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
            grouped_topk,
        )

        return grouped_topk(
            hidden_states=router_logits,
            gating_output=router_logits,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_fn,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=bias,
        )
    if router_logits.dim() != 2:
        raise ValueError(
            f"grouped router top-k kernel expects [rows, experts], got {router_logits.shape}"
        )
    logits_in = (
        router_logits
        if router_logits.dtype in (torch.bfloat16, torch.float32)
        else router_logits.float()
    )
    logits_pad, orig_rows, pad_r = _pad_rows_to_8(logits_in)
    scoring_code = SCORING_NAME_TO_CODE[scoring_fn]
    scale = float(routed_scaling_factor)
    if bias is not None:
        w, i = _op_grouped_with_bias(
            logits_pad,
            bias.float().view(1, -1),
            int(topk),
            int(num_expert_group),
            int(topk_group),
            scoring_code,
            bool(renormalize),
            scale,
        )
    else:
        w, i = _op_grouped(
            logits_pad,
            int(topk),
            int(num_expert_group),
            int(topk_group),
            scoring_code,
            bool(renormalize),
            scale,
        )
    return (w[:orig_rows], i[:orig_rows]) if pad_r else (w, i)


def fused_gate_grouped_rowmax_select(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused in-VMEM ``gate(hidden_states) + scoring_fn + bias + grouped_topk``."""
    gate_w = (
        gate_weight
        if gate_weight.shape[0] == hidden_states.shape[-1]
        else gate_weight.T
    )
    if hidden_states.device.type != "tpu" or hidden_states.shape[0] > PALLAS_MAX_ROWS:
        logits = torch.matmul(hidden_states, gate_w)
        return grouped_rowmax_select(
            logits,
            topk=topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            bias=bias,
        )
    hidden_pad, orig_rows, pad_r = _pad_rows_to_8(hidden_states)
    scoring_code = SCORING_NAME_TO_CODE[scoring_fn]
    w, i = _op_fused_gate_grouped_with_bias(
        hidden_pad,
        gate_w,
        bias.float().view(1, -1),
        int(topk),
        int(num_expert_group),
        int(topk_group),
        scoring_code,
        bool(renormalize),
        float(routed_scaling_factor),
    )
    return (w[:orig_rows], i[:orig_rows]) if pad_r else (w, i)


