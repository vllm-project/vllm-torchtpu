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
"""DeepSeek-V4.1 Single-Pass Delayed mHC custom ops on TPU."""

import functools
from typing import Any, NamedTuple

import torch

from vllm_torchtpu.kernels.deepseek_v4.mhc.post_kernel import mhc_post
from vllm_torchtpu.kernels.deepseek_v41.mhc import (
    engram_gate_pallas,
    mhc_final_collapse_norm_v41,
    mhc_fused_post_engram_pre_quant_v41,
    mhc_fused_post_engram_pre_v41,
    mhc_fused_post_pre_quant_v41,
    mhc_fused_post_pre_v41,
    mhc_post_collapse_norm_v41,
    mhc_pre_quant_v41,
    mhc_pre_v41,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (
    _fake_mhc_post,
    _fake_pre_outputs,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_compressor import (  # noqa: E501
    _get_or_register_op,
    _name_float,
)


class MHCOpsV41(NamedTuple):
    pre: Any
    fused: Any
    post: Any
    pre_quant: Any = None
    fused_quant: Any = None
    engram_gate: Any = None
    fused_engram: Any = None
    fused_engram_quant: Any = None
    fused_quant_fp8_only: Any = None


def _fake_mhc_pre_v41(residual, *args):
    next_pre_mix = residual.new_empty(residual.shape[:-1], dtype=torch.float32)
    return (*_fake_pre_outputs(residual), next_pre_mix)


def _fake_mhc_pre_quant_v41(residual, *args):
    *outer_shape, _, hidden_size = residual.shape
    normed_fp8 = residual.new_empty(
        (*outer_shape, hidden_size), dtype=torch.float8_e4m3fn
    )
    normed_scale = residual.new_empty((*outer_shape, 1), dtype=torch.float32)
    return (*_fake_mhc_pre_v41(residual), normed_fp8, normed_scale)


def _fake_mhc_fused_post_pre_v41(x, residual, *args):
    return (torch.empty_like(residual), *_fake_mhc_pre_v41(residual))


def _fake_mhc_fused_post_pre_quant_v41(x, residual, *args):
    return (torch.empty_like(residual), *_fake_mhc_pre_quant_v41(residual))


def _fake_mhc_fused_post_pre_quant_fp8_only_v41(x, residual, *args):
    *outer_shape, _, hidden_size = residual.shape
    normed_w = hidden_size
    post_mix, comb_mix, _, next_pre_mix, normed_fp8, normed_scale = (
        _fake_mhc_pre_quant_v41(residual)
    )
    normed_stub = residual.new_empty((*outer_shape, normed_w), dtype=residual.dtype)
    return (
        torch.empty_like(residual),
        post_mix,
        comb_mix,
        normed_stub,
        next_pre_mix,
        normed_fp8,
        normed_scale,
    )


def _fake_engram_gate_v41(residual, kv, qk_weight, *args):
    return torch.empty_like(residual)


def _fake_mhc_post_collapse_norm_v41(x, residual, *args):
    return (torch.empty_like(residual), torch.empty_like(x))


def _fake_mhc_final_collapse_norm_v41(x, residual, *args):
    return torch.empty_like(x)


def get_engram_gate_v41_op(rms_eps: float = 1e-6):
    return _get_or_register_op(
        f"pallas::deepseek_v41_engram_gate_e{_name_float(rms_eps)}",
        functools.partial(engram_gate_pallas, eps=rms_eps),
        3,
        _fake_engram_gate_v41,
    )


def get_mhc_post_v41_op():
    return _get_or_register_op(
        "pallas::deepseek_v41_mhc_post",
        mhc_post,
        4,
        _fake_mhc_post,
    )


def get_mhc_post_collapse_norm_v41_op(rms_eps: float = 1e-6):
    return _get_or_register_op(
        f"pallas::deepseek_v41_mhc_post_collapse_norm_e{_name_float(rms_eps)}",
        functools.partial(mhc_post_collapse_norm_v41, rms_eps=rms_eps),
        6,
        _fake_mhc_post_collapse_norm_v41,
    )


def get_mhc_final_collapse_norm_v41_op(rms_eps: float = 1e-6):
    return _get_or_register_op(
        f"pallas::deepseek_v41_mhc_final_collapse_norm_e{_name_float(rms_eps)}",
        functools.partial(mhc_final_collapse_norm_v41, rms_eps=rms_eps),
        6,
        _fake_mhc_final_collapse_norm_v41,
    )


def mhc_collapse_v41(
    hidden_states: torch.Tensor, pre_mix: torch.Tensor
) -> torch.Tensor:
    """Collapse the final hc_mult streams using the last sublayer's delayed pre_mix."""
    return torch.sum(pre_mix.unsqueeze(-1).float() * hidden_states.float(), dim=-2).to(
        hidden_states.dtype
    )


def get_mhc_ops_v41(
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
) -> MHCOpsV41:
    kw = dict(
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    suffix = (
        f"_e{_name_float(rms_eps)}_pe{_name_float(hc_pre_eps)}"
        f"_se{_name_float(hc_sinkhorn_eps)}"
        f"_pm{_name_float(hc_post_mult_value)}_sr{sinkhorn_repeat}"
    )
    return MHCOpsV41(
        pre=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_pre_s{int(use_stream0)}{suffix}",
            functools.partial(mhc_pre_v41, **kw, use_stream0=use_stream0),
            6,
            _fake_mhc_pre_v41,
        ),
        fused=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_fused_post_pre{suffix}",
            functools.partial(mhc_fused_post_pre_v41, **kw),
            9,
            _fake_mhc_fused_post_pre_v41,
        ),
        post=get_mhc_post_v41_op(),
        pre_quant=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_pre_quant_s{int(use_stream0)}{suffix}",
            functools.partial(mhc_pre_quant_v41, **kw, use_stream0=use_stream0),
            6,
            _fake_mhc_pre_quant_v41,
        ),
        fused_quant=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_fused_post_pre_quant{suffix}",
            functools.partial(mhc_fused_post_pre_quant_v41, **kw, write_bf16=True),
            9,
            _fake_mhc_fused_post_pre_quant_v41,
        ),
        engram_gate=get_engram_gate_v41_op(rms_eps),
        fused_engram=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_fused_post_engram_pre{suffix}",
            functools.partial(mhc_fused_post_engram_pre_v41, **kw),
            11,
            _fake_mhc_fused_post_pre_v41,
        ),
        fused_engram_quant=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_fused_post_engram_pre_quant{suffix}",
            functools.partial(mhc_fused_post_engram_pre_quant_v41, **kw),
            11,
            _fake_mhc_fused_post_pre_quant_v41,
        ),
        fused_quant_fp8_only=_get_or_register_op(
            f"pallas::deepseek_v41_mhc_fused_post_pre_quant_fp8_only{suffix}",
            functools.partial(mhc_fused_post_pre_quant_v41, **kw, write_bf16=False),
            9,
            _fake_mhc_fused_post_pre_quant_fp8_only_v41,
        ),
    )
