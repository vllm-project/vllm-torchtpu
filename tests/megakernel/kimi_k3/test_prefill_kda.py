# SPDX-License-Identifier: Apache-2.0
"""The prefill KDA layer's pieces on CPU: unit-RMS window rows, the input
norm in the KDA tiles, and the chunked delta rule's sub-block factors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

pytestmark = pytest.mark.cpu_test

HBM = pl.BlockSpec(memory_space=pltpu.HBM)
E4M3 = jnp.float8_e4m3fn


def _interpret():
    return pltpu.InterpretParams(detect_races=True)


def _mixture_inputs(rng, tokens, hidden):
    prefix = rng.standard_normal((tokens, hidden)) * 3
    blocks = rng.standard_normal((8, tokens, hidden)) * 2
    folded = rng.standard_normal((1, hidden)) * 0.05
    norm = 1 + 0.2 * rng.standard_normal((1, hidden))
    return (
        jnp.asarray(prefix, jnp.bfloat16),
        jnp.asarray(blocks, jnp.bfloat16),
        jnp.asarray(folded, jnp.float32),
        jnp.asarray(norm, jnp.bfloat16),
    )


def _unit_rows(prefix, blocks, folded, active, dtype, tile):
    """``mixture_norm_tiled_body`` without a norm weight, as the KDA layer
    runs it."""
    tokens, hidden = prefix.shape

    def body(active_ref, prefix_ref, blocks_ref, folded_ref, out_ref, *scratch):
        prefix_bufs, block_buf, out_bufs, numerator, row_max, row_sum, folded_buf = (
            scratch[:7]
        )
        kernels.mixture_norm_tiled_body(
            active_ref,
            prefix_ref,
            blocks_ref,
            folded_ref,
            None,
            out_ref,
            prefix_bufs,
            block_buf,
            out_bufs,
            numerator,
            row_max,
            row_sum,
            folded_buf,
            None,
            *scratch[7:],
            tile=tile,
            eps=1e-5,
        )

    scratch = (
        pltpu.VMEM((2, tile, hidden), jnp.bfloat16),
        pltpu.VMEM((2, tile, hidden), jnp.bfloat16),
        pltpu.VMEM((2, tile, hidden), dtype),
        pltpu.VMEM((tile, hidden), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((tile, 1), jnp.float32),
        pltpu.VMEM((1, hidden), jnp.float32),
        *(pltpu.SemaphoreType.DMA((2,)),) * 4,
    )
    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[HBM] * 3,
            out_specs=HBM,
            scratch_shapes=scratch,
        ),
        interpret=_interpret(),
    )
    return call(active, prefix, blocks, folded)


@pytest.mark.parametrize("dtype", [E4M3, jnp.bfloat16])
@pytest.mark.parametrize("active", [0, 3, 8])
def test_unit_rows_times_norm_match_the_normalized_rows(dtype, active):
    rng = np.random.default_rng(active)
    tokens, hidden, tile = 64, 1024, 32
    prefix, blocks, folded, norm = _mixture_inputs(rng, tokens, hidden)
    count = jnp.asarray([active], jnp.int32)
    weighted = kernels.make_mixture_norm_tiled(
        seq_len=tokens, tile=tile, interpret=_interpret()
    )(prefix, blocks, folded, norm, count)
    unit = _unit_rows(prefix, blocks, folded, count, dtype, tile)
    assert unit.dtype == dtype

    reference = np.asarray(
        kernels.residual_mixture(blocks, prefix, folded, active), np.float32
    )
    reference /= np.sqrt(np.mean(reference * reference, axis=1, keepdims=True) + 1e-5)
    assert np.abs(reference).max() <= np.sqrt(hidden)
    # Round to nearest: E4M3FN keeps 3 mantissa bits (BF16 7), its subnormal
    # step is 2^-9; the kernel's BF16 mixture may round one step apart.
    bits = 3 if dtype == E4M3 else 7
    relative = 2.0 ** -(bits + 1) + 2.0**-7
    unit32 = np.asarray(unit, np.float32)
    np.testing.assert_array_less(
        np.abs(unit32 - reference), relative * np.abs(reference) + 2.0**-9
    )
    # The KDA tile's product with the norm weight, against the BF16 rows.
    rows = np.asarray(unit.astype(jnp.bfloat16) * norm, np.float32)
    expected = np.asarray(weighted, np.float32)
    scale = np.abs(np.asarray(norm, np.float32))
    np.testing.assert_array_less(
        np.abs(rows - expected), relative * np.abs(expected) + 2.0**-8 * scale
    )


def _kda_weights(rng, hidden):
    def normal(shape, scale, dtype=jnp.bfloat16):
        return jnp.asarray(rng.standard_normal(shape) * scale, dtype)

    return (
        normal((hidden, 128), hidden**-0.5),  # gate_w
        normal((128, 384), 128**-0.5),  # fb_w
        normal((hidden, 1664), hidden**-0.5),  # proj_w
        normal((4, 3, 3, 128), 0.5, jnp.float32),  # conv_w
        jnp.asarray(np.log(rng.uniform(1, 16, (3, 1))), jnp.float32),  # A_log
        normal((3, 128), 1.0, jnp.float32),  # dt_bias
        jnp.asarray(1 + 0.1 * rng.standard_normal((8, 128)), jnp.float32),  # norm
    )


def _kda_unit_rows(hidden_rows, weights, state, input_norm, length):
    """``kda_prefill_tiled_body`` over unit-RMS rows with ``input_norm``."""
    tokens, hidden = hidden_rows.shape

    def body(length_ref, hidden_ref, *refs):
        weight_refs, (norm_ref, out_ref, rec_ref, hist_ref) = refs[:9], refs[9:13]
        *scratch, norm_vmem = refs[13:]
        kernels.kda_prefill_tiled_body(
            length_ref,
            hidden_ref,
            *weight_refs,
            out_ref,
            rec_ref,
            hist_ref,
            *scratch,
            tile=128,
            decay_lower_bound=-5.0,
            epsilon=1e-5,
            input_norm=(norm_ref, norm_vmem),
        )

    scratch = (
        *kernels.kda_prefill_tiled_scratch_shapes(
            hidden, 128, *weights, hidden_dtype=hidden_rows.dtype
        ),
        pltpu.VMEM((1, hidden), jnp.bfloat16),
    )
    call = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, 384), jnp.bfloat16),
            jax.ShapeDtypeStruct(state[0].shape, jnp.float32),
            jax.ShapeDtypeStruct(state[1].shape, jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[HBM] * 11,
            out_specs=[HBM] * 3,
            scratch_shapes=scratch,
        ),
        interpret=_interpret(),
    )
    return call(length, hidden_rows, *weights, *state, input_norm)


@pytest.mark.parametrize("length", [256, 200])
def test_kda_tiles_apply_the_input_norm_to_fp8_unit_rows(length):
    rng = np.random.default_rng(length)
    tokens, hidden = 256, 1024
    x = rng.standard_normal((tokens, hidden)) * rng.uniform(0.1, 4, (tokens, 1))
    unit = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-5)
    norm = jnp.asarray(1 + 0.2 * rng.standard_normal((1, hidden)), jnp.bfloat16)
    weights = _kda_weights(rng, hidden)
    state = (
        jnp.asarray(0.05 * rng.standard_normal((3, 128, 128)), jnp.float32),
        jnp.asarray(0.5 * rng.standard_normal((3, 3, 3, 128)), jnp.bfloat16),
    )
    count = jnp.asarray([length], jnp.int32)
    unit_fp8 = jnp.asarray(unit, E4M3)
    fused = _kda_unit_rows(unit_fp8, weights, state, norm, count)

    # The same rows with the norm weight applied outside: bit-exact.
    tiled = kernels.make_kda_tiled(seq_len=tokens, interpret=_interpret())
    normalized = unit_fp8.astype(jnp.bfloat16) * norm
    outside = tiled(normalized, *weights, *state, count)
    for got, want in zip(fused, outside, strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    # Against BF16 rows of the unrounded unit rows: E4M3FN rounding noise
    # (3 mantissa bits), where a missing norm weight would be 0.2.
    bf16_rows = jnp.asarray(unit * np.asarray(norm, np.float32), jnp.bfloat16)
    attended, rec, _ = tiled(bf16_rows, *weights, *state, count)
    got = np.asarray(fused[0], np.float32)
    want = np.asarray(attended, np.float32)
    assert np.abs(got - want).max() <= 0.12 * np.abs(want).max()
    assert np.sqrt(np.mean((got - want) ** 2)) <= 0.08 * np.sqrt(np.mean(want**2))
    np.testing.assert_array_equal(got[length:], 0)
    rec_err = np.abs(np.asarray(fused[1]) - np.asarray(rec)).max()
    assert rec_err <= 0.1 * np.abs(np.asarray(rec)).max()


def _delta_rule(q, k, v, gate, beta, state):
    """The Kimi delta rule token by token in FP64."""
    q, k, v, gate, beta = (np.asarray(x, np.float64) for x in (q, k, v, gate, beta))
    s = np.asarray(state, np.float64)
    outputs = []
    for r in range(q.shape[1]):
        s = s * np.exp(gate[:, r, :, None])
        delta = beta[:, r, None] * (v[:, r] - np.einsum("hk,hkv->hv", k[:, r], s))
        s = s + np.einsum("hk,hv->hkv", k[:, r], delta)
        outputs.append(np.einsum("hk,hkv->hv", q[:, r], s))
    return np.stack(outputs, axis=1), s


def _gates(kind, rng, shape):
    if kind == "realistic":  # lower bound * sigmoid, as kda_decay
        return -5 / (1 + np.exp(-(2 * rng.standard_normal(shape) + 1)))
    if kind == "mixed":
        return rng.choice([0.0, -0.01, -1.0, -5.0], size=shape)
    gate = np.full(shape, -5.0)  # every row at the bound: the widest factors
    if kind == "boundary":  # but no decay between sub-blocks
        gate[:, 32::32] = 0.0
    return gate


@pytest.mark.parametrize("kind", ["realistic", "mixed", "saturated", "boundary"])
@pytest.mark.parametrize("fast", [True, False])
def test_chunk_sub_blocks_against_the_sequential_delta_rule(kind, fast):
    assert kernels._KDA_SUB_BLOCK == 32  # the kernel's default
    rng = np.random.default_rng(7)
    heads, chunks, channels = 3, 3, 128
    shape = (heads, 64, channels)
    state = 0.1 * rng.standard_normal((heads, channels, channels))
    initial = state
    outputs = {16: [], 32: []}
    states = {16: state, 32: state}
    references = []
    for _ in range(chunks):
        q = rng.standard_normal(shape)
        q *= channels**-0.5 / np.linalg.norm(q, axis=-1, keepdims=True)
        k = rng.standard_normal(shape)
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        v = rng.standard_normal(shape)
        gate = _gates(kind, rng, shape)
        beta = 1 / (1 + np.exp(-rng.standard_normal(shape[:2])))
        reference, initial = _delta_rule(q, k, v, gate, beta, initial)
        references.append(reference)
        args = [jnp.asarray(x, jnp.float32) for x in (q, k, v, gate, beta)]
        for sub_block in (16, 32):
            invert = None if fast else kernels._invert_delta_rule_two_level
            out, states[sub_block] = kernels.kda_chunk(
                *args,
                jnp.asarray(states[sub_block], jnp.float32),
                sub_block=sub_block,
                invert=invert,
            )
            outputs[sub_block].append(np.asarray(out))
    reference = np.stack(references)
    scale = np.abs(reference).max()
    errors = {
        sub_block: np.abs(np.stack(outputs[sub_block]) - reference).max() / scale
        for sub_block in (16, 32)
    }
    for sub_block in (16, 32):
        state_error = np.abs(np.asarray(states[sub_block]) - initial).max()
        assert state_error <= 5e-5 * np.abs(initial).max()
    assert errors[16] <= 2e-5
    # Rows at the decay bound reach the dropped-term bound of 32-row
    # sub-blocks (``_kda_chunk_core``); other gates agree with 16 rows.
    saturated = kind in ("saturated", "boundary")
    assert errors[32] <= (1e-3 if saturated else 2e-5)
