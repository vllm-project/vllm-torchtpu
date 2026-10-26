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
"""DeepSeek-V4 mHC Pallas kernels.

Drop-in replacements for vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``, plus
the fused seam op that runs one layer's post together with the next layer's
pre as a single kernel.
"""

import functools
from typing import Any, NamedTuple

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.deepseek_v4.mhc import (
    fused_post_pre_kernel,
    post_head_kernel,
    post_kernel,
    pre_kernel,
    utils,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

# Every mHC tensor is token-major, so its leading axis is the attention-DP
# token axis. As in the other DeepSeek-V4 ops, the mesh carries no such axis
# today, so it resolves to None.
ATTN_DATA_AXIS = None

_mhc_op_cache: dict[str, Any] = {}


def _name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _gate_suffix(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    fuse_norm: bool,
) -> str:
    """The gate constants are baked into the traced op, so they name it."""
    return (
        f"_r{_name_float(rms_eps)}_p{_name_float(hc_pre_eps)}"
        f"_s{_name_float(hc_sinkhorn_eps)}"
        f"_a{_name_float(hc_post_mult_value)}_i{sinkhorn_repeat}"
        + ("_gk" if gates_in_kernel else "")
        + ("_n" if fuse_norm else "")
    )


# Module-level so `pallas.jax_op` can trace and register them as torch ops.
# jax_op derives the torch schema from these signatures: every tensor must be
# positional and annotated, so the norm variants are separate functions.
def _mhc_pre_jax(
    residual: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mix GEMM + stream collapse in Pallas, gates/Sinkhorn in XLA (or in
    the kernel, with ``gates_in_kernel``)."""
    return pre_kernel.mhc_pre(
        residual,
        utils.SplitFn(fn_hi, fn_mid, fn_lo),
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
    )


def _mhc_pre_norm_jax(
    residual: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``_mhc_pre_jax`` emitting the layer input RMS-normalized with
    ``norm_weight`` (the sublayer's input norm, which shares ``rms_eps``)."""
    return pre_kernel.mhc_pre(
        residual,
        utils.SplitFn(fn_hi, fn_mid, fn_lo),
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
        norm_weight=norm_weight,
        norm_eps=rms_eps,
    )


def _mhc_post_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """Recombine the sublayer output into the mHC residual streams."""
    return post_kernel.mhc_post(x, residual, post_layer_mix, comb_res_mix)


def _mhc_fused_post_pre_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One layer's post and the next layer's pre as a single kernel."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        utils.SplitFn(fn_hi, fn_mid, fn_lo),
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
        **({"fn_hbm": True} if fn_hbm else {}),
    )


def _mhc_fused_post_pre_norm_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_hi: jax.Array,
    fn_mid: jax.Array,
    fn_lo: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``_mhc_fused_post_pre_jax`` emitting the RMS-normalized layer input."""
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        utils.SplitFn(fn_hi, fn_mid, fn_lo),
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
        **({"fn_hbm": True} if fn_hbm else {}),
        norm_weight=norm_weight,
        norm_eps=rms_eps,
    )


def _mhc_pre_cat_jax(
    residual: jax.Array,
    fn_cat: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``_mhc_pre_jax`` taking fn as its stacked (3 * M3, K) bf16 chunks."""
    return _pre_call(
        residual, utils.CatFn(fn_cat), hc_scale, hc_base, None,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
    )


def _mhc_pre_cat_norm_jax(
    residual: jax.Array,
    fn_cat: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``_mhc_pre_cat_jax`` emitting the RMS-normalized layer input."""
    return _pre_call(
        residual, utils.CatFn(fn_cat), hc_scale, hc_base, norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
    )


def _kernel_kw(gates_max_tokens: int, token_block_size: int) -> dict:
    kw: dict = {}
    if gates_max_tokens >= 0:
        kw["gates_max_tokens"] = gates_max_tokens
    if token_block_size:
        kw["token_block_size"] = token_block_size
    return kw


def _pre_call(residual, fn, hc_scale, hc_base, norm_weight, *, rms_eps, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, **kernel_opts):
    norm = {} if norm_weight is None else dict(norm_weight=norm_weight, norm_eps=rms_eps)
    return pre_kernel.mhc_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, **norm, **kernel_opts,
    )


def _fused_call(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
                norm_weight, *, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat, **kernel_opts):
    norm = {} if norm_weight is None else dict(norm_weight=norm_weight, norm_eps=rms_eps)
    return fused_post_pre_kernel.mhc_fused_post_pre(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        **norm, **kernel_opts,
    )


def _mhc_fused_post_pre_cat_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_cat: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``_mhc_fused_post_pre_jax`` taking fn as its stacked bf16 chunks."""
    return _fused_call(
        x, residual, post_layer_mix, comb_res_mix, utils.CatFn(fn_cat),
        hc_scale, hc_base, None,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
        **({"fn_hbm": True} if fn_hbm else {}),
    )


def _mhc_fused_post_pre_cat_norm_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn_cat: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    norm_weight: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    gates_in_kernel: bool,
    gates_max_tokens: int = -1,
    token_block_size: int = 0,
    fn_hbm: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """``_mhc_fused_post_pre_cat_jax`` emitting the RMS-normalized input."""
    return _fused_call(
        x, residual, post_layer_mix, comb_res_mix, utils.CatFn(fn_cat),
        hc_scale, hc_base, norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        gates_in_kernel=gates_in_kernel,
        **_kernel_kw(gates_max_tokens, token_block_size),
        **({"fn_hbm": True} if fn_hbm else {}),
    )


def _fake_pre_outputs(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(post_mix, comb_mix, layer_input) placeholders for Dynamo tracing."""
    outer_shape = residual.shape[:-2]
    hc_mult, hidden_size = residual.shape[-2:]
    return (
        torch.empty(
            (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
        ),
        torch.empty(
            (*outer_shape, hc_mult, hc_mult),
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            (*outer_shape, hidden_size), dtype=residual.dtype, device=residual.device
        ),
    )


def _fake_mhc_pre(residual, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return _fake_pre_outputs(residual)


def _fake_mhc_post(x, residual, post_layer_mix, comb_res_mix, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty_like(residual)


def _fake_mhc_fused_post_pre(x, residual, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return (torch.empty_like(residual), *_fake_pre_outputs(residual))


class KernelOpts(NamedTuple):
    """Static mHC kernel options beyond the gate constants (all name the op).

    stacked_fn: the ops take fn as one stacked (3 * M3, K) bf16 array
        (``utils.CatFn``, one MXU pass) instead of three chunks.
    token_block: fused seam token block (0: the kernel default, 32).
    fn_hbm: the fused seam reads fn from HBM in its own pipeline instead of
        letting XLA prefetch it into VMEM (VLLM_TPU_DSV4_MHC_SEAM_FN_HBM).
    gates_max_tokens: in-kernel gates up to this many tokens (-1: the kernel
        default, 512; 0: no cap).
    """

    gates_in_kernel: bool = False
    fuse_norm: bool = False
    stacked_fn: bool = False
    token_block: int = 0
    gates_max_tokens: int = -1
    fn_hbm: bool = False

    def suffix(self) -> str:
        return (
            ("_gk" if self.gates_in_kernel else "")
            + ("_n" if self.fuse_norm else "")
            + ("_cat" if self.stacked_fn else "")
            + (f"_tb{self.token_block}" if self.token_block else "")
            + (f"_gm{self.gates_max_tokens}" if self.gates_max_tokens >= 0 else "")
            + ("_fh" if self.fn_hbm else "")
        )

    def kernel_kwargs(self, *, fused: bool) -> dict:
        kw = {
            "gates_in_kernel": self.gates_in_kernel,
            "gates_max_tokens": self.gates_max_tokens,
            "token_block_size": self.token_block if fused else 0,
        }
        if fused and self.fn_hbm:
            kw["fn_hbm"] = True
        return kw

    def fn_specs(self) -> tuple:
        return (P(),) if self.stacked_fn else (P(), P(), P())


def _mhc_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    opts: KernelOpts,
):
    op_name = (
        "pallas::deepseek_v4_mhc_pre_v2"
        + _gate_suffix(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            False,
            False,
        )
        + opts.suffix()
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        fn = {
            (False, False): _mhc_pre_jax,
            (False, True): _mhc_pre_norm_jax,
            (True, False): _mhc_pre_cat_jax,
            (True, True): _mhc_pre_cat_norm_jax,
        }[(opts.stacked_fn, opts.fuse_norm)]
        op = pallas.jax_op(
            op_name,
            functools.partial(
                fn,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                **opts.kernel_kwargs(fused=False),
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None, None),  # residual
                *opts.fn_specs(),  # fn chunks (or the stacked chunks)
                P(),  # hc_scale
                P(),  # hc_base
                *([P()] if opts.fuse_norm else []),  # norm weight
            ),
        )
        op.register_fake(_fake_mhc_pre)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_post_op():
    """The ``post`` op alone, for callers that settle a deferred post.

    ``post`` takes no gate constants, so this needs none. Same
    outside-the-trace requirement as ``get_mhc_ops``.
    """
    op_name = "pallas::deepseek_v4_mhc_post_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_post_jax,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
            ),
        )
        op.register_fake(_fake_mhc_post)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_fused_post_pre_op(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    opts: KernelOpts,
):
    op_name = (
        "pallas::deepseek_v4_mhc_fused_post_pre_v2"
        + _gate_suffix(
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            False,
            False,
        )
        + opts.suffix()
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        fn = {
            (False, False): _mhc_fused_post_pre_jax,
            (False, True): _mhc_fused_post_pre_norm_jax,
            (True, False): _mhc_fused_post_pre_cat_jax,
            (True, True): _mhc_fused_post_pre_cat_norm_jax,
        }[(opts.stacked_fn, opts.fuse_norm)]
        op = pallas.jax_op(
            op_name,
            functools.partial(
                fn,
                rms_eps=rms_eps,
                hc_pre_eps=hc_pre_eps,
                hc_sinkhorn_eps=hc_sinkhorn_eps,
                hc_post_mult_value=hc_post_mult_value,
                sinkhorn_repeat=sinkhorn_repeat,
                **opts.kernel_kwargs(fused=True),
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                *opts.fn_specs(),  # fn chunks (or the stacked chunks)
                P(),  # hc_scale
                P(),  # hc_base
                *([P()] if opts.fuse_norm else []),  # norm weight
            ),
        )
        op.register_fake(_fake_mhc_fused_post_pre)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_split_fn_jax(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return tuple(utils.split_fn3(fn))


def _mhc_stack_fn_jax(fn: jax.Array) -> jax.Array:
    return utils.cat_fn3(fn).cat


def get_mhc_stack_fn_op():
    """fn (f32) -> its three bf16 chunks stacked (3 * M3, K), for
    ``KernelOpts.stacked_fn`` ops. Run once per weight at load."""
    op_name = "pallas::deepseek_v4_mhc_stack_fn_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_stack_fn_jax,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(P(),),
        )

        def _fake(fn):
            return torch.empty(
                (3 * fn.shape[0], fn.shape[1]), dtype=torch.bfloat16, device=fn.device
            )

        op.register_fake(_fake)
        _mhc_op_cache[op_name] = op
    return op


def get_mhc_split_fn_op():
    """fn (f32) -> its three bf16 chunks, as the ``pre``/``fused`` ops take it.

    Run once per weight at load, outside the compiled region.
    """
    op_name = "pallas::deepseek_v4_mhc_split_fn_v1"
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            _mhc_split_fn_jax,
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(P(),),
        )

        def _fake(fn):
            return tuple(torch.empty_like(fn, dtype=torch.bfloat16) for _ in range(3))

        op.register_fake(_fake)
        _mhc_op_cache[op_name] = op
    return op


def _mhc_post_head_jax(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_eps: float,
    want_head: bool,
    want_mean: bool,
) -> tuple[jax.Array, ...]:
    """Last post + hc head collapse and/or the stream mean, one kernel."""
    return post_head_kernel.mhc_post_head(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        want_head=want_head,
        want_mean=want_mean,
    )


def get_mhc_post_head_op(
    rms_eps: float, hc_eps: float, *, want_head: bool, want_mean: bool
):
    """``post`` fused with what consumes its streams at the end of the stack:
    the hc head collapse (``want_head``) and/or DSpark's aux stream mean
    (``want_mean``). Returns a tuple: (head,), (mean,) or (head, mean)."""
    assert want_head or want_mean
    op_name = (
        "pallas::deepseek_v4_mhc_post_head_v1"
        + f"_r{_name_float(rms_eps)}_e{_name_float(hc_eps)}"
        + ("_h" if want_head else "")
        + ("_m" if want_mean else "")
    )
    op = _mhc_op_cache.get(op_name)
    if op is None:
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _mhc_post_head_jax,
                rms_eps=rms_eps,
                hc_eps=hc_eps,
                want_head=want_head,
                want_mean=want_mean,
            ),
            mesh=get_pallas_op_mesh(),
            input_partition_specs=(
                P(ATTN_DATA_AXIS, None),  # x
                P(ATTN_DATA_AXIS, None, None),  # residual
                P(ATTN_DATA_AXIS, None, None),  # post_layer_mix
                P(ATTN_DATA_AXIS, None, None),  # comb_res_mix
                P(),  # hc_fn
                P(),  # hc_scale
                P(),  # hc_base
            ),
        )
        n_out = int(want_head) + int(want_mean)

        def _fake(x, residual, *args, **kwargs):
            return tuple(torch.empty_like(x) for _ in range(n_out))

        op.register_fake(_fake)
        _mhc_op_cache[op_name] = op
    return op


class MHCOps(NamedTuple):
    """The three registered mHC ops for one set of gate constants.

    ``pre`` and ``post`` match vLLM's ``mhc_pre_torch`` / ``mhc_post_torch``
    contracts; ``fused`` is the seam op, equivalent to ``post`` followed by
    ``pre`` on its output but sharing one pass over the residual streams.

    ``fn3`` below is fn's three bf16 chunks from ``get_mhc_split_fn_op``.

    ``pre(residual, *fn3, hc_scale, hc_base)``
        residual (..., hc_mult, hidden_size) bf16 in; post_mix
        (..., hc_mult, 1) f32, comb_mix (..., hc_mult, hc_mult) f32,
        layer_input (..., hidden_size) bf16 out.
    ``post(x, residual, post_layer_mix, comb_res_mix)``
        (..., hc_mult, hidden_size) bf16 out.
    ``fused(x, residual, post_layer_mix, comb_res_mix, *fn3, hc_scale, hc_base)``
        residual_cur -- the ``post`` output -- then ``pre``'s three outputs
        computed from it.

    With ``fuse_norm`` both ``pre`` and ``fused`` take one more trailing
    argument, the sublayer's input-norm weight ((hidden_size,) or, to avoid a
    per-call relayout, (1, hidden_size)), and return layer_input already
    RMS-normalized with it. With ``KernelOpts.stacked_fn`` ``fn3`` is the
    single stacked array from ``get_mhc_stack_fn_op``.
    """

    pre: Any
    fused: Any
    post: Any


def get_mhc_ops(
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    gates_in_kernel: bool = False,
    fuse_norm: bool = False,
    opts: KernelOpts | None = None,
) -> MHCOps:
    """Build (or fetch) every mHC op for these gate constants.

    Must be called outside the compiled region: the builders construct jax
    ``PartitionSpec``s, which Dynamo cannot trace, so a cold cache inside a
    compiled forward is a hard error rather than a graph break. Callers hold
    the returned ops and invoke them directly, keeping the op lookup out of
    the traced graph entirely.
    """
    if opts is None:
        opts = KernelOpts(gates_in_kernel=gates_in_kernel, fuse_norm=fuse_norm)
    consts = (
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        opts,
    )
    return MHCOps(
        pre=_mhc_pre_op(*consts),
        fused=_mhc_fused_post_pre_op(*consts),
        post=get_mhc_post_op(),
    )
