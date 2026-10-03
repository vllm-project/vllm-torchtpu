"""Single-pass delayed mHC with compact TPU Sinkhorn and upstream residual flow."""

import jax
import jax.numpy as jnp
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
    x = x.astype(jnp.float32)
    mixes = jnp.matmul(x, fn.T, precision=jax.lax.Precision.HIGHEST)
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


def delayed_pre(residual, x, fn, scale, base, previous_pre, *constants):
    assert _pre_op is not None
    return _pre_op(residual, x, fn, scale, base, previous_pre, *constants)


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
        if x is None:
            x = residual.flatten(1)
        if pre_mix is None:
            # First layer uses residual[:, 0], not the newly computed gate.
            pre_mix = residual.new_zeros(residual.shape[:2], dtype=fn.dtype)
            pre_mix[:, 0] = 1
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
