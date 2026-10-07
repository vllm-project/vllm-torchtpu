"""Single-pass delayed mHC with compact TPU Sinkhorn and upstream residual flow.

Shared by the v1 and v2 DeepSeek V4.1 implementations.
"""

import functools

import jax
import jax.numpy as jnp
import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_delayed_torch

from vllm_torchtpu.layers.adapter.normalization import rms_norm


def _pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    V4.1 keeps Sinkhorn in a compact device loop.
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)

    def normalize(_, value):
        value = value / (jnp.sum(value, axis=-1, keepdims=True) + hc_sinkhorn_eps)
        return value / (jnp.sum(value, axis=-2, keepdims=True) + hc_sinkhorn_eps)

    comb_mix = jax.lax.fori_loop(0, sinkhorn_repeat - 1, normalize, comb_mix)
    return pre_mix, post_mix, comb_mix


_pre_op = None


def delayed_pre_jax(
    residual: jax.Array,
    x: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    mult, dim = residual.shape[-2:]
    if fn.dtype == jnp.bfloat16:
        # Load-time packed weight (see `pack_mix_weight`).
        mixes = _packed_mixes(x, fn, mult * (mult + 2))
    else:
        mixes = jnp.matmul(
            x.astype(jnp.float32), fn.T, precision=jax.lax.Precision.HIGHEST
        )
    x = x.astype(jnp.float32)
    # Upstream's first-layer broadcast optimization folds x/fn to hidden_size.
    # Gate normalization must still use the actual projection input width.
    squares = jnp.mean(jnp.square(x), -1, keepdims=True) * (mult * dim)
    new_pre, post, comb = _pre_gates(
        mixes,
        squares,
        mult,
        dim,
        scale,
        base,
        rms_eps,
        pre_eps,
        sinkhorn_eps,
        post_multiplier,
        repeats,
    )
    value = (previous_pre[..., None] * residual.astype(jnp.float32)).sum(1)
    return post[..., None], comb, value.astype(residual.dtype), new_pre


def _truncate_bf16(values: jax.Array) -> jax.Array:
    # A mask, not an f32->bf16->f32 round trip: XLA's excess-precision
    # rewrite may drop the round trip and zero the residual parts.
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _pack_mix_weight(fn: jax.Array) -> jax.Array:
    """FP32 ``fn [n, K]`` as BF16 ``[K, 128]`` columns ``hi | mid | lo | 0``.

    ``hi + mid + lo == fn`` exactly, so one BF16 matmul against BF16 inputs
    matches the FP32 projection up to accumulation order.
    """
    high = _truncate_bf16(fn)
    middle = _truncate_bf16(fn - high)
    low = fn - high - middle
    parts = jnp.concatenate([high, middle, low]).astype(jnp.bfloat16)
    return jnp.pad(parts.T, ((0, 0), (0, 128 - parts.shape[0])))


def pack_mix_weight(fn: torch.Tensor) -> torch.Tensor:
    """Torch twin of `_pack_mix_weight`, applied once after weight loading."""
    weight = fn.detach().cpu().float()

    def truncate(values):
        return (values.view(torch.int32) & -65536).view(torch.float32)

    high = truncate(weight)
    middle = truncate(weight - high)
    low = weight - high - middle
    parts = torch.cat([high, middle, low]).to(torch.bfloat16)
    packed = torch.nn.functional.pad(parts.T, (0, 128 - parts.shape[0]))
    return packed.contiguous().to(fn.device)


def pack_mix_weights(layers) -> None:
    """Pack each decoder layer's FP32 mix weights after weight loading.

    Call after upstream ``finalize_mhc_broadcast_weights``: its first-layer
    broadcast weight reads the FP32 ``hc_attn_fn``. Packed weights are BF16,
    so a second call is a no-op.
    """
    for layer in layers:
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name)
            if fn.device.type == "tpu" and fn.dtype == torch.float32:
                packed = torch.nn.Parameter(pack_mix_weight(fn), requires_grad=False)
                setattr(layer, name, packed)


def _packed_mixes(x: jax.Array, weight: jax.Array, mixes: int) -> jax.Array:
    """Mix logits from a packed weight, summed in `_flat_pre_sums` order."""
    parts = jnp.dot(x, weight, preferred_element_type=jnp.float32)
    return (
        parts[:, :mixes] + parts[:, mixes : 2 * mixes] + parts[:, 2 * mixes : 3 * mixes]
    )


def _flat_pre_sums(flat, weight, previous_pre, mixes, tile):
    """One pass over ``flat [T, M*D]``: mix logits, square sums, collapse."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(x_ref, w_ref, pre_ref, mix_ref, square_ref, value_ref):
        pre = pre_ref[...]
        terms, sums = [], []
        for m in range(mult):
            x = x_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            terms.append(pre[:, m : m + 1] * x)
            sums.append(jnp.sum(x * x, axis=1, keepdims=True))
        value_ref[...] = functools.reduce(jnp.add, terms).astype(value_ref.dtype)
        square_ref[...] = functools.reduce(jnp.add, sums)
        parts = jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
    )(flat, weight, previous_pre)


def delayed_pre_flat_jax(
    flat: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """`delayed_pre_jax` with ``x = residual = flat [T, M*D]``.

    Prefill-sized batches use one Pallas pass; viewing the flat residual as
    ``[T, M, D]`` costs XLA a relayout copy plus an FP32 copy for the
    projection. Small (decode) batches keep the XLA path.
    """
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    if tokens < 64 or tokens % 64 or 3 * mixes > 128:
        return delayed_pre_jax(
            flat.reshape(tokens, mult, dim),
            flat,
            fn,
            scale,
            base,
            previous_pre,
            *constants,
        )
    tile = 128 if tokens % 128 == 0 else 64
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    mixes, square, value = _flat_pre_sums(flat, weight, previous_pre, mixes, tile)
    new_pre, post, comb = _pre_gates(mixes, square, mult, dim, scale, base, *constants)
    return post[..., None], comb, value, new_pre


def _post_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
) -> jax.Array:
    """XLA 2D stream recombine matching `_post_kernel` accumulation order."""
    mult = comb_res_mix.shape[-1]
    dim = flat.shape[-1] // mult
    x_f32 = x.reshape(-1, dim).astype(flat.dtype).astype(jnp.float32)
    post = post_layer_mix.reshape(-1, mult)
    comb = comb_res_mix.reshape(-1, mult, mult)
    streams = [
        flat[:, i * dim : (i + 1) * dim].astype(jnp.float32) for i in range(mult)
    ]
    out_streams = []
    for j in range(mult):
        acc = post[:, j : j + 1] * x_f32
        for i in range(mult):
            acc = acc + comb[:, i, j : j + 1] * streams[i]
        out_streams.append(acc.astype(flat.dtype))
    return jnp.concatenate(out_streams, axis=1)


def _flat_post_pre_sums(
    sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre, mixes, tile
):
    """One Pallas pass fusing `mhc_post_2d` with `_flat_pre_sums`."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult

    def kernel(
        sub_x_ref,
        res_ref,
        post_ref,
        comb_ref,
        w_ref,
        pre_ref,
        out_res_ref,
        mix_ref,
        square_ref,
        value_ref,
    ):
        sub_x_f32 = sub_x_ref[...].astype(jnp.float32)
        post = post_ref[...]
        comb = comb_ref[...]
        old_bf = [res_ref[:, i * dim : (i + 1) * dim] for i in range(mult)]
        for j in range(mult):
            acc = post[:, j : j + 1] * sub_x_f32
            for i in range(mult):
                k = i * mult + j
                acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
            out_res_ref[:, j * dim : (j + 1) * dim] = acc.astype(out_res_ref.dtype)
        pre = pre_ref[...]
        terms, sums = [], []
        for m in range(mult):
            x = out_res_ref[:, m * dim : (m + 1) * dim].astype(jnp.float32)
            terms.append(pre[:, m : m + 1] * x)
            sums.append(jnp.sum(x * x, axis=1, keepdims=True))
        value_ref[...] = functools.reduce(jnp.add, terms).astype(value_ref.dtype)
        square_ref[...] = functools.reduce(jnp.add, sums)
        parts = jnp.dot(
            out_res_ref[...], w_ref[...], preferred_element_type=jnp.float32
        )
        mix_ref[...] = (
            parts[:, :mixes]
            + parts[:, mixes : 2 * mixes]
            + parts[:, 2 * mixes : 3 * mixes]
        )

    def row(i):
        return i, 0

    return pl.pallas_call(
        kernel,
        grid=(tokens // tile,),
        in_specs=[
            pl.BlockSpec((tile, dim), row),
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mult), row),
            pl.BlockSpec((tile, mult * mult), row),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, mult), row),
        ],
        out_specs=[
            pl.BlockSpec((tile, width), row),
            pl.BlockSpec((tile, mixes), row),
            pl.BlockSpec((tile, 1), row),
            pl.BlockSpec((tile, dim), row),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((tokens, width), flat.dtype),
            jax.ShapeDtypeStruct((tokens, mixes), jnp.float32),
            jax.ShapeDtypeStruct((tokens, 1), jnp.float32),
            jax.ShapeDtypeStruct((tokens, dim), flat.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=48 * 1024 * 1024
        ),
    )(sub_x, flat, post_layer_mix, comb_res_mix, weight, previous_pre)


def delayed_post_pre_flat_jax(
    x: jax.Array,
    flat: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    fn: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    previous_pre: jax.Array,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    repeats: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused `post` + `delayed_pre_flat_jax` seam for `shifted_post_pre`."""
    tokens, width = flat.shape
    mult = previous_pre.shape[1]
    dim = width // mult
    constants = (rms_eps, pre_eps, sinkhorn_eps, post_multiplier, repeats)
    mixes = mult * (mult + 2)
    if tokens < 64 or tokens % 64 or 3 * mixes > 128:
        flat_new = jax.lax.optimization_barrier(
            _post_jax(
                x,
                flat,
                post_layer_mix,
                comb_res_mix,
            )
        )
        post, comb, value, new_pre = delayed_pre_jax(
            flat_new.reshape(tokens, mult, dim),
            flat_new,
            fn,
            scale,
            base,
            previous_pre,
            *constants,
        )
        return flat_new, post, comb, value, new_pre
    tile = 64
    weight = fn if fn.dtype == jnp.bfloat16 else _pack_mix_weight(fn)
    flat_new, mixes_out, square, value = _flat_post_pre_sums(
        x.reshape(tokens, dim).astype(flat.dtype),
        flat,
        post_layer_mix.reshape(tokens, mult),
        comb_res_mix.reshape(tokens, mult * mult),
        weight,
        previous_pre,
        mixes,
        tile,
    )
    new_pre, post, comb = _pre_gates(
        mixes_out, square, mult, dim, scale, base, *constants
    )
    return flat_new, post[..., None], comb, value, new_pre


def initialize_delayed_mhc():
    global _pre_op
    if _pre_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_vllm_model_wrapper_context,
    )

    _pre_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_pre",
        delayed_pre_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            None,
            None,
            None,
            None,
            None,
        ),
    )

    def fake(residual, x, fn, scale, base, previous_pre, *args):
        tokens, mult, dim = residual.shape
        return (
            fn.new_empty((tokens, mult, 1)),
            fn.new_empty((tokens, mult, mult)),
            residual.new_empty((tokens, dim)),
            fn.new_empty((tokens, mult)),
        )

    _pre_op.register_fake(fake)

    global _pre_flat_op
    _pre_flat_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_pre_flat",
        delayed_pre_flat_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 5 + (None,) * 5,
    )

    def fake_flat(flat, fn, scale, base, previous_pre, *args):
        tokens, mult = previous_pre.shape
        return (
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    _pre_flat_op.register_fake(fake_flat)

    global _post_pre_flat_op
    _post_pre_flat_op = pallas.jax_op(
        "pallas::deepseek_v41_delayed_mhc_post_pre_flat",
        delayed_post_pre_flat_jax,
        mesh=get_vllm_model_wrapper_context().mesh,
        input_partition_specs=(P(),) * 8 + (None,) * 5,
    )

    def fake_post_pre_flat(
        x, flat, post_layer_mix, comb_res_mix, fn, scale, base, previous_pre, *args
    ):
        tokens, mult = previous_pre.shape
        return (
            torch.empty_like(flat),
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    _post_pre_flat_op.register_fake(fake_post_pre_flat)


_pre_flat_op = None
_post_pre_flat_op = None


def delayed_pre(residual, x, fn, scale, base, previous_pre, *constants):
    assert _pre_op is not None and _pre_flat_op is not None
    if x is None:
        return _pre_flat_op(
            residual.flatten(1), fn, scale, base, previous_pre, *constants
        )
    return _pre_op(residual, x, fn, scale, base, previous_pre, *constants)


def delayed_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    scale,
    base,
    previous_pre,
    *constants,
):
    assert _post_pre_flat_op is not None
    flat_new, post, comb, value, next_pre = _post_pre_flat_op(
        x,
        residual.flatten(1),
        post_layer_mix,
        comb_res_mix,
        fn,
        scale,
        base,
        previous_pre,
        *constants,
    )
    return flat_new.view_as(residual), post, comb, value, next_pre


_post_op = None


def initialize_mhc_post():
    global _post_op
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (
        get_mhc_post_op,
    )

    _post_op = get_mhc_post_op()


def post(x, residual, post_layer_mix, comb_res_mix):
    if residual.device.type == "tpu":
        assert _post_op is not None
        return _post_op(x, residual, post_layer_mix, comb_res_mix)
    return mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)


def collapse(x, pre_mix):
    return (x.float() * pre_mix.unsqueeze(-1)).sum(-2).to(x.dtype)


def _default_pre_mix(residual: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # First layer uses residual[:, 0], not the newly computed gate.
    # Avoid `new_zeros(residual.shape[:2])` and slice-scatter so SP
    # profile_run does not pass a SymIntArrayRef `[s0 // tp, 4]` to
    # `aten.new_zeros` in fx_to_mlir.
    first_stream = (torch.arange(residual.shape[1], device=residual.device) == 0).to(
        dtype=dtype
    )
    return torch.zeros_like(residual[:, :, 0], dtype=dtype) + first_stream


def pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    x=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    if residual.device.type == "tpu":
        if pre_mix is None:
            pre_mix = _default_pre_mix(residual, hc_scale.dtype)
        post, comb, value, next_pre = delayed_pre(
            residual,
            x,
            fn,
            hc_scale,
            hc_base,
            pre_mix,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        post, comb, value, next_pre = mhc_pre_delayed_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            x=x,
        )
    if norm_weight is not None:
        value = rms_norm(value, norm_weight, norm_eps)
    return post, comb, value, next_pre


def shifted_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    pre_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
    capture_aux=False,
    *,
    stream=None,
    reduce_results=False,
):
    if stream is not None:
        raise ValueError("TPU mHC does not use CUDA streams")
    if reduce_results:
        x = tensor_model_parallel_all_reduce(x)
    if residual.device.type == "tpu" and _post_pre_flat_op is not None:
        if pre_mix is None:
            pre_mix = _default_pre_mix(residual, hc_scale.dtype)
        residual, post_mix, comb_mix, value, next_pre = delayed_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            pre_mix,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            value = rms_norm(value, norm_weight, norm_eps)
        aux = residual.mean(1) if capture_aux else x.new_empty(0, x.shape[-1])
        return residual, post_mix, comb_mix, value, next_pre, aux
    residual = post(x, residual, post_layer_mix, comb_res_mix)
    aux = residual.mean(1) if capture_aux else x.new_empty(0, x.shape[-1])
    return (
        residual,
        *pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=pre_mix,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        ),
        aux,
    )
