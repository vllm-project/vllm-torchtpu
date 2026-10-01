"""Hardware checks for the V4.1 adapter to the native FP8 CSA kernel."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compute
from vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather import csa_gather
from vllm_torchtpu.models.vllm.deepseek_v41 import cache, csa


def _require_tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("Requires a physical TPU with SparseCore")


def test_native_packer_parity():
    _require_tpu()
    rng = np.random.default_rng(129)
    values = rng.normal(size=(32, 512)).astype(np.float32)
    values[0] = 0
    values[1] *= 1e-6
    values[2] *= 1000
    x = jnp.asarray(values, jnp.bfloat16)

    def kernel(x_ref, out_ref):
        values = x_ref[...].astype(jnp.float32).reshape(8, 4, 128)
        q, scales = compute.quantize_fp8_lane_periodic(values, 64, 448)
        slabs = compute.pack_nope_lane_periodic(q, scales, 448, 512, 128)
        out_ref.bitcast(jnp.uint8).reshape(8, 4, 128)[...] = slabs

    native = jax.jit(
        pl.pallas_call(
            kernel,
            grid=(4,),
            in_specs=[pl.BlockSpec((8, 512), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((32, 128), jnp.int32),
        )
    )(x)
    actual, _ = jax.jit(csa.pack_csa_values)(x)
    np.testing.assert_array_equal(actual, native)


@pytest.mark.parametrize("shared_pool", [False, True])
@pytest.mark.parametrize("separate", [False, True])
def test_independent_rope_gather_and_legacy_default(shared_pool, separate):
    _require_tpu()
    rng = np.random.default_rng(131)
    words = rng.integers(-(2**31), 2**31 - 1, (256, 1, 128), dtype=np.int32)
    nope = rng.integers(0, 256, 3 * 128, dtype=np.int32)
    rope = rng.integers(0, 1024, 3 * 128, dtype=np.int32)
    ri = rope if separate else nope

    def gather(first, second):
        return csa_gather(
            first,
            second,
            jnp.asarray(nope),
            top_k=128,
            rope_indices=jnp.asarray(ri) if separate else None,
            shared_pool=shared_pool,
        )

    if shared_pool:
        # One operand feeding both arguments reproduces XLA deduplication.
        gathered, rotary = jax.jit(lambda pool: gather(pool, pool))(jnp.asarray(words))
    else:
        gathered, rotary = gather(jnp.asarray(words), jnp.asarray(words))
    np.testing.assert_array_equal(gathered, words.reshape(-1, 128)[nope])
    selected = words.reshape(-1, 32)[ri].view(np.uint32)
    bits = np.concatenate((selected & 65535, selected >> 16), axis=-1)
    halves = bits.reshape(-1, 2, 64, 64)
    rows = np.concatenate((halves[:, 0], halves[:, 1]), axis=-1).reshape(-1, 2, 128)
    expected = (rows[:, 0] | (rows[:, 1] << 16)).view(np.int32)
    np.testing.assert_array_equal(rotary, expected)


@pytest.mark.parametrize("tokens,topk", [(8, 71), (8, 1024), (128, 2048)])
def test_shared_pool_csa_with_swa_and_holey_selection(tokens, topk):
    _require_tpu()
    rng = np.random.default_rng(132)
    count, dim, heads = 128, 512, 8
    values = torch.from_numpy(
        rng.normal(size=(count, dim)).astype(np.float32)
    ).bfloat16()
    original = torch.full((320, 128), 0x7FC07FC0, dtype=torch.int32)
    owner = csa.PackedCSACache(
        original.clone(), ((16, 8, 640), torch.uint8, 14 * 512, 3 * 512)
    )
    slots = torch.arange(count, dtype=torch.int32)
    owner.write(slots, values)
    nope = owner.row_indices(slots).numpy()
    rope = owner.rope_indices(slots).numpy()
    # Device publication must preserve both adjacent records and unrelated bytes.
    written = jax.jit(csa.write_csa_values)(
        jnp.asarray(original.numpy()),
        jnp.asarray(nope),
        jnp.asarray(rope),
        jnp.asarray(values.float().numpy(), jnp.bfloat16),
    )
    np.testing.assert_array_equal(written, owner.raw.numpy())
    decoded = owner[slots // 8, slots % 8].float().numpy()
    q = jnp.asarray(rng.normal(size=(tokens, heads, dim)), jnp.bfloat16)
    swa_values = jnp.asarray(rng.normal(size=(count, dim)), jnp.bfloat16)
    swa_pool = jax.jit(cache.scatter_cache_values)(
        jnp.zeros((count * 2, 128), jnp.int32),
        jnp.arange(count, dtype=jnp.int32),
        swa_values,
    )
    selection = rng.integers(0, count, (tokens, topk), dtype=np.int32)
    valid = rng.uniform(size=selection.shape) > 0.3
    valid[0] = False
    sr = rng.integers(0, count, (tokens, 64), dtype=np.int32)
    sr[0] = -1  # Sink-only query; another query has SWA with no compressed KV.
    if tokens > 1:
        valid[1] = False
    nr = np.where(valid, nope[selection], -1)
    rr = np.where(valid, rope[selection], -1)
    sink = rng.normal(size=(heads,)).astype(np.float32)
    scale = dim**-0.5
    actual = jax.jit(functools.partial(csa.csa_attention_jax, scale=scale))(
        q,
        swa_pool,
        written,
        jnp.asarray(sr),
        jnp.asarray(nr),
        jnp.asarray(rr),
        jnp.asarray(sink),
    )
    actual = np.asarray(actual, dtype=np.float32)
    q_np, swa_np = (
        np.asarray(q, dtype=np.float32),
        np.asarray(swa_values, dtype=np.float32),
    )
    expected = np.zeros_like(actual)
    for t in range(tokens):
        kv = np.concatenate(
            (swa_np[sr[t][sr[t] >= 0]], decoded[selection[t][valid[t]]])
        )
        scores = q_np[t] @ kv.T * scale
        maximum = np.maximum(scores.max(axis=-1, initial=-np.inf), sink)
        p = np.exp(scores - maximum[:, None])
        expected[t] = p @ kv / (p.sum(-1) + np.exp(sink - maximum))[:, None]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=0.008, rtol=0.03)
