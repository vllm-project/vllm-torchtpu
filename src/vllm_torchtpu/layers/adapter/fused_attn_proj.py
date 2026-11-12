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
"""Torch bridge for kernels/fused_attn_proj (TPU_FUSED_ATTN_PROJ).

With TPU_FUSED_ATTN_PROJ=1, prepare_model gives each eligible layer one torch
op that replaces input_layernorm through o_proj:

  full attention: fused_qkv -> batched RPA -> fused_out_proj
  GDN:            fused_gdn, then the GDN layer's own op from Part 2 on.

The op reads its weights from the layer's vLLM modules. Other layers, and
steps the op can't take, run the unfused path.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_wrapper
from vllm_torchtpu.kernels.fused_attn_proj.gdn import kernel as gdn_kernel
from vllm_torchtpu.kernels.fused_attn_proj.gdn import weights as gdn_weights
from vllm_torchtpu.kernels.fused_attn_proj.gqa import kernel as gqa_kernel
from vllm_torchtpu.kernels.fused_attn_proj.gqa import weights as gqa_weights
from vllm_torchtpu.kernels.fused_attn_proj.out_proj import kernel as out_proj_kernel
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.core.sequence_layout import (
    is_pcp_streaming_attention_metadata,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

_ATTR = "_fused_attn_proj"


class Ineligible(Exception):
    pass


def _own_buffer(x: torch.Tensor | None) -> torch.Tensor | None:
    """x, or a copy if x is a view: torch_tpu re-materializes a view argument
    as an as_strided program on every step (see fresh_buffer)."""
    return fresh_buffer(x) if x is not None and x._is_view() else x


def _norm(norm: torch.nn.Module | None) -> tuple[str, torch.Tensor | None]:
    """(kernel Norm value, (1, dim) weight or None) of a vLLM norm."""
    if norm is None:
        return "none", None
    if isinstance(norm, GemmaRMSNorm):  # (1 + w); also Qwen3.5's norms.
        kind = "rms_one_plus_w"
    elif isinstance(norm, RMSNorm):
        kind = "rms" if norm.has_weight else "rms_noscale"
    else:
        raise Ineligible(f"unsupported norm {type(norm).__name__}")
    w = getattr(norm, "weight", None)
    return kind, None if w is None else _own_buffer(w.reshape(1, -1))


def _linear(linear: torch.nn.Module, num_cols: int | None = None) -> tuple:
    """(w (K, N), w_scale (1, N) or None, bias (1, N) or None)."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import (  # Import cycle.
        VllmFp8LinearMethodTPU,
    )
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedLinearMethod,
    )

    method = linear.quant_method
    if not isinstance(method, (VllmFp8LinearMethodTPU, VllmUnquantizedLinearMethod)):
        raise Ineligible(f"quant method {type(method).__name__}")
    w, scale, bias = linear.weight, getattr(linear, "weight_scale", None), linear.bias
    if not getattr(linear, WEIGHT_FLIPPED_ATTR, False):
        raise Ineligible("weight not stored (K, N)")
    if w.dtype == torch.float8_e4m3fn and (scale is None or scale.ndim != 1):
        raise Ineligible("fp8 weight not per-channel (REQUANTIZE_BLOCK_SIZE set)")
    if scale is not None:
        scale = scale.reshape(1, -1)
    if bias is not None:
        bias = bias.reshape(1, -1)
    if num_cols is not None:
        w, scale, bias = (
            None if x is None else x[:, :num_cols] for x in (w, scale, bias)
        )
    return _own_buffer(w), _own_buffer(scale), _own_buffer(bias)


def _act_quant_block(hidden_size: int) -> int:
    """Largest multiple of 256 up to 1024 dividing hidden_size (768 for 5376)."""
    return next(b for b in range(1024, 0, -256) if hidden_size % b == 0)


_load_ops: dict[str, Callable] = {}


def _load_op(name: str, fn: Callable) -> Callable:
    """A torch op for a load-time weight transform; name must determine fn."""
    if name not in _load_ops:
        _load_ops[name] = pallas.jax_op(f"pallas::fused_attn_proj_{name}", fn)
    return _load_ops[name]


_ops: dict[tuple, Callable] = {}


def _shared_op(
    kind: str, key: tuple, fn: Callable, fake: Callable, donate: bool
) -> Callable:
    """The op for (kind, key), built once; key must cover all fn and fake capture.

    Layers with the same config share an op: a jax_op caches its export per
    instance, so one op per layer would re-export the same kernel for every
    layer at every compile bucket (see gdn_attention_op._get_shared_gdn_op).
    """
    op = _ops.get((kind, key))
    if op is None:
        donate_argnums = (0,) if donate and not envs.TPU_KERNEL_ITER_MODE else None
        name = f"pallas::fused_attn_proj_{kind}_{len(_ops)}"
        op = pallas.jax_op(name, fn, donate_argnums=donate_argnums)
        op.register_fake(fake)
        _ops[(kind, key)] = op
    return op


def _pad_tokens(multiple: int, *xs: jax.Array | None) -> list[jax.Array | None]:
    """Pads dim 0 to a multiple of multiple: the kernels take T in whole fp8
    sublane tiles (32), fused_gdn in whole lanes (128)."""
    return [
        None if x is None else jnp.pad(x, ((0, -x.shape[0] % multiple), (0, 0)))
        for x in xs
    ]


def _cos_sin(rotary_emb: torch.nn.Module, positions: torch.Tensor, n_pairs: int):
    """(cos, sin), each (T, n_pairs) bf16, as vLLM's RoPE reads them."""
    cos, sin = rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
    if positions.ndim == 2:  # Interleaved MRoPE (Qwen3.5).
        cos = apply_interleaved_rope(cos, rotary_emb.mrope_section)
        sin = apply_interleaved_rope(sin, rotary_emb.mrope_section)
    return cos[..., :n_pairs].to(torch.bfloat16), sin[..., :n_pairs].to(torch.bfloat16)


# Full attention.


def _gqa_block(
    kv_cache: jax.Array,
    hidden: jax.Array,
    residual: jax.Array | None,
    w: jax.Array,
    w_scale: jax.Array | None,
    bias: jax.Array | None,
    q_norm_w: jax.Array | None,
    k_norm_w: jax.Array | None,
    input_norm_w: jax.Array | None,
    out_w: jax.Array,
    out_w_scale: jax.Array | None,
    out_bias: jax.Array | None,
    cos: jax.Array,
    sin: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    *,
    qkv: tuple,
    rpa: tuple,
    out: tuple,
) -> tuple[jax.Array, ...]:
    """fused_qkv -> batched RPA -> fused_out_proj; returns (kv_cache, out[, res]).

    qkv, rpa and out are keyword arguments as tuples of items (jit statics).
    """
    qkv_kw, rpa_kw, out_kw = dict(qkv), dict(rpa), dict(out)
    t = hidden.shape[0]
    hidden, residual, cos, sin = _pad_tokens(32, hidden, residual, cos, sin)
    sal = rpa_kw["kv_layout"] == rpa_configs.KVLayout.SEQ_ALONG_LANE
    q, kv, gate, new_residual = gqa_kernel.fused_qkv(
        hidden, residual, w, w_scale, bias, q_norm_w, k_norm_w, input_norm_w,
        cos, sin, page_size=kv_cache.shape[4 if sal else 1], **qkv_kw,
    )  # fmt: skip
    # Shapes only (prepared_inputs replaces them); RPA is jitted, so these must be
    # arrays, not ShapeDtypeStructs. XLA drops them as unused.
    t_rpa, d = q.shape[1], out_kw["head_dim"]
    queries = jnp.zeros((t_rpa, out_kw["num_q_heads"], d), q.dtype)
    kvs = jnp.zeros((t_rpa, out_kw["num_kv_heads"], d), kv.dtype)
    o_raw, kv_cache = rpa_wrapper.ragged_paged_attention(
        queries, kvs, kvs, kv_cache, seq_lens, block_tables, query_start_loc,
        distribution, prepared_inputs=(q, kv), return_raw_output=True, **rpa_kw,
    )  # fmt: skip
    o = out_proj_kernel.fused_out_proj(
        o_raw, out_w, out_w_scale, out_bias, gate, num_tokens=hidden.shape[0], **out_kw
    )
    o = o[:t]
    return (kv_cache, o) if residual is None else (kv_cache, o, new_residual[:t])


@dataclasses.dataclass
class FusedGqa:
    attn: torch.nn.Module  # The decoder layer's self_attn.
    kwargs: dict[str, Any]  # fused_qkv's model keyword arguments.
    op: Callable
    weights: tuple
    n_pairs: int
    has_residual: bool

    @classmethod
    def build(cls, attn, input_norm, *, has_residual: bool, v_from_k: bool):
        from vllm_torchtpu.layers.adapter.attention import (  # Import cycle.
            PallasBatchedRPAAttentionBackendImpl,
        )

        impl = attn.attn.impl
        if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
            raise Ineligible("not batched RPA (e.g. head_dim 64)")
        if attn.head_dim == 64 or impl.dcp_world_size > 1:
            raise Ineligible("head_dim 64 or DCP")
        if impl.kv_sharing_target_layer_name or getattr(attn.attn, "sinks", None):
            raise Ineligible("KV sharing or attention sinks")
        gate = bool(getattr(attn, "attn_output_gate", False))
        num_qkv_cols = (2 if gate else 1) * attn.q_size + attn.kv_size
        w, w_scale, bias = _linear(attn.qkv_proj, num_qkv_cols if v_from_k else None)
        block = _act_quant_block(w.shape[0])
        norms = {
            name: _norm(getattr(attn, f"{name}_norm", None)) for name in ("q", "k", "v")
        }
        input_norm_kind, input_norm_w = _norm(input_norm)
        if w.dtype == torch.float8_e4m3fn and input_norm_w is not None:
            input_norm_w = _load_op(
                f"input_norm_{input_norm_kind}_{block}",
                functools.partial(
                    gqa_weights.prepare_input_norm_w,
                    fp8=True,
                    input_norm=input_norm_kind,
                    act_quant_block=block,
                ),
            )(input_norm_w)
        rotary = attn.rotary_emb
        rope_angles = getattr(rotary, "rope_angles", None)  # Gemma 4 proportional.
        n_pairs = rope_angles or rotary.rotary_dim // 2
        kwargs = dict(
            num_q_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            gate=gate,
            v_from_k=v_from_k,
            input_norm=input_norm_kind,
            q_norm=norms["q"][0],
            k_norm=norms["k"][0],
            v_norm=norms["v"][0],
            rope_offset=attn.head_dim // 2 if rope_angles else None,
            eps=input_norm.variance_epsilon,
            k_scale=attn.attn._k_scale_float,
            v_scale=attn.attn._v_scale_float,
            act_quant_block=block,
        )
        weights = (
            w, w_scale, bias, norms["q"][1], norms["k"][1], input_norm_w,
            *_linear(attn.o_proj),
        )  # fmt: skip
        layout = rpa_configs.KVLayout(impl.kv_layout)
        rpa = dict(
            sm_scale=impl.scale,
            sliding_window=impl.sliding_window,
            soft_cap=impl.logits_soft_cap,
            k_scale=kwargs["k_scale"],
            v_scale=kwargs["v_scale"],
            use_causal_mask=impl.use_causal_mask,
            kv_layout=layout,
            decode_query_size=impl.decode_query_size,
        )
        statics = dict(
            qkv=tuple(dict(kwargs, kv_layout=layout).items()),
            rpa=tuple(rpa.items()),
            out=tuple(
                (k, kwargs[k]) for k in ("num_q_heads", "num_kv_heads", "head_dim")
            ),
        )
        fn = functools.partial(_gqa_block, **statics)

        def fake(kv_cache, hidden, residual, *args):
            outs = [torch.empty_like(kv_cache), torch.empty_like(hidden)]
            return outs if residual is None else [*outs, torch.empty_like(hidden)]

        op = _shared_op("gqa", tuple(statics.values()), fn, fake, donate=True)
        return cls(attn, kwargs, op, weights, n_pairs, has_residual)

    def __call__(self, hidden, residual, positions):
        attn_md, _, kv_cache, _ = get_attention_context(self.attn.attn.layer_name)
        if (
            (residual is None) == self.has_residual  # First decoder layer.
            or attn_md is None
            or kv_cache.numel() == 0  # Memory profiling.
            or kv_cache.ndim != 5  # Shared 4-D or block-major 6-D pools.
            or get_vllm_model_wrapper_context().kv_cache_bundle is not None
            or is_pcp_streaming_attention_metadata(attn_md)
        ):
            return None
        cos, sin = _cos_sin(self.attn.rotary_emb, positions, self.n_pairs)
        outs = self.op(
            kv_cache,
            hidden,
            residual if self.has_residual else None,
            *self.weights,
            cos,
            sin,
            attn_md.seq_lens,
            attn_md.block_tables,
            attn_md.query_start_loc,
            attn_md.request_distribution,
        )
        kv_cache.copy_(outs[0])
        if not torch.compiler.is_compiling():  # As the RPA backend does.
            synchronize_tensors(kv_cache, wait=False)
        out = outs[1]
        if getattr(self.attn.o_proj, "reduce_results", True):
            out = tensor_model_parallel_all_reduce(out)
        return out, (outs[2] if self.has_residual else hidden)


# GDN.


def _gdn_block(
    hidden: jax.Array,
    residual: jax.Array,
    w: jax.Array,
    w_scale: jax.Array,
    ba_w: jax.Array,
    norm_w: jax.Array | None,
    *,
    kwargs: tuple,
    qkv_cols: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """fused_gdn, outputs in the GDN layer's shapes and dtypes."""
    gdn_kw = dict(kwargs)  # kwargs is a tuple of items (a jit static).
    t, nv, dtype = hidden.shape[0], gdn_kw["num_v_heads"], hidden.dtype
    mixed_qkv, z, b, a, new_residual = gdn_kernel.fused_gdn(
        *_pad_tokens(128, hidden, residual), w, w_scale, ba_w, norm_w, **gdn_kw
    )
    return (
        mixed_qkv[:t].reshape(t, qkv_cols).astype(dtype),
        z[:t],
        b[:t, 0, :nv].astype(dtype),
        a[:t, 0, :nv].astype(dtype),
        new_residual[:t],
    )


@dataclasses.dataclass
class FusedGdn:
    gdn: torch.nn.Module  # The decoder layer's linear_attn.
    op: Callable
    weights: tuple

    @classmethod
    def build(cls, gdn, input_norm):
        if not hasattr(gdn, "in_proj_qkvz") or getattr(
            gdn, "_pcp_streaming_configured", False
        ):
            raise Ineligible("separate in_proj_qkv / in_proj_z, or PCP")
        w, w_scale, _ = _linear(gdn.in_proj_qkvz)
        if w_scale is None:
            raise Ineligible("in_proj_qkvz is not fp8")
        ba, _, _ = _linear(gdn.in_proj_ba)
        norm_kind, norm_w = _norm(input_norm)
        kwargs = dict(
            num_k_heads=gdn.num_k_heads // gdn.tp_size,
            num_v_heads=gdn.num_v_heads // gdn.tp_size,
            head_dim=gdn.head_k_dim,
            input_norm=norm_kind,
            eps=input_norm.variance_epsilon,
            norm_dtype=jnp.float32,  # vLLM's precision.
            act_quant_block=_act_quant_block(w.shape[0]),
        )
        prep = {
            k: kwargs[k]
            for k in ("num_v_heads", "input_norm", "norm_dtype", "act_quant_block")
        }
        ba_w, norm_w = _load_op(
            "gdn_weights_{num_v_heads}_{input_norm}_{act_quant_block}".format(**prep),
            functools.partial(gdn_weights.prepare_weights, **prep),
        )(ba, norm_w)
        qkv_cols = gdn.tp_head_geometry.local_conv_dim(gdn.head_k_dim, gdn.head_v_dim)
        nv, d = kwargs["num_v_heads"], kwargs["head_dim"]

        def fake(hidden, residual, *weights):
            t = hidden.shape[0]
            return (
                hidden.new_empty(t, qkv_cols),
                hidden.new_empty(t, nv, d),
                hidden.new_empty(t, nv),
                hidden.new_empty(t, nv),
                torch.empty_like(hidden),
            )

        key = (tuple(kwargs.items()), qkv_cols)
        fn = functools.partial(_gdn_block, kwargs=key[0], qkv_cols=qkv_cols)
        op = _shared_op("gdn", key, fn, fake, donate=False)
        return cls(gdn, op, (w, w_scale, ba_w, norm_w))

    def __call__(self, hidden, residual, positions):
        if residual is None:  # First decoder layer.
            return None
        mixed_qkv, z, b, a, new_residual = self.op(hidden, residual, *self.weights)
        return self.gdn.forward_projected(
            hidden.shape[0], mixed_qkv, z, b, a
        ), new_residual


# Entry points.


def prepare_model(model: torch.nn.Module, hf_config: Any) -> None:
    """Builds the fused op of each eligible decoder layer; call after loading."""
    fused = 0
    k_eq_v = bool(getattr(hf_config, "attention_k_eq_v", False))
    for name, layer in model.named_modules():
        input_norm = getattr(layer, "input_layernorm", None)
        attn, gdn = (
            getattr(layer, "self_attn", None),
            getattr(layer, "linear_attn", None),
        )
        if input_norm is None or (attn is None and gdn is None):
            continue
        if gdn is None and not hasattr(attn, "attn"):  # E.g. a vision tower.
            continue
        try:
            if getattr(layer, "use_attn_reduce_scatter_for_moe", False):
                raise Ineligible("sequence-parallel attention input")
            if getattr(layer, "layer_scale", False):
                raise Ineligible("layer_scale")
            if gdn is not None:
                setattr(gdn, _ATTR, FusedGdn.build(gdn, input_norm))
            else:
                # Gemma 4 adds the residual after post_attention_layernorm, so
                # its input norm takes no residual.
                is_sliding = getattr(attn, "is_sliding", None)  # Gemma 4.
                op = FusedGqa.build(
                    attn,
                    input_norm,
                    has_residual=is_sliding is None,
                    v_from_k=k_eq_v and is_sliding is False,
                )
                setattr(attn, _ATTR, op)
            fused += 1
        except Ineligible as e:
            logger.info("fused_attn_proj: %s unfused: %s", name, e)
    logger.info("fused_attn_proj: fused %d decoder layers.", fused)


def attention(module, hidden, residual, positions):
    """(output, new residual) of input_layernorm + module, or None if unfused.

    module is the decoder layer's self_attn or linear_attn; hidden is the
    layer's input, before input_layernorm.
    """
    fused = getattr(module, _ATTR, None)
    return None if fused is None else fused(hidden, residual, positions)
