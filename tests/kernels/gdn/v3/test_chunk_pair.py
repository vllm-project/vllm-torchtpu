# SPDX-License-Identifier: Apache-2.0
"""TPU_GDN_CHUNK_PAIR: two chunks per grid step in the chunked prefill give
bit-identical outputs and states to one chunk per step, including a sequence
whose last pair holds a single chunk and a prefill mixed with decodes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N_KQ, N_V, DIM, WIDTH = 2, 6, 128, 2 * 2 * 128 + 6 * 128


def _inputs(tokens, seed, num_seqs=1):
    rng = np.random.default_rng(seed)

    def normal(shape, dtype=jnp.bfloat16, scale=0.5):
        return jnp.asarray(rng.normal(size=shape) * scale, dtype)

    return dict(
        qkv=normal((tokens, WIDTH)),
        b=normal((tokens, N_V)),
        a=normal((tokens, N_V)),
        # Slot 0 is the null block; sequence i uses slot i + 1.
        conv=normal((num_seqs + 1, 3, WIDTH), jnp.float32, 0.1),
        ssm=normal((num_seqs + 1, N_V, DIM, DIM), jnp.float32, 0.05),
        weight=normal((WIDTH, 1, 4), scale=0.2),
        bias=normal((WIDTH,), scale=0.1),
        a_log=normal((N_V,), jnp.float32),
        dt_bias=normal((N_V,), jnp.float32) - 1,
    )


def _run(x, query_start_loc, indices, distribution, seq_lens):
    from vllm_torchtpu.kernels.gdn.v3 import wrapper

    jax.clear_caches()  # fused_conv1d_gdn's jit cache is keyed on args, not env
    # The states are donated: give each run its own copies.
    conv, ssm = jnp.array(x["conv"], copy=True), jnp.array(x["ssm"], copy=True)
    return wrapper.fused_conv1d_gdn(
        x["qkv"], x["b"], x["a"], conv, ssm, x["weight"], x["bias"],
        x["a_log"], x["dt_bias"], query_start_loc, indices, distribution, seq_lens,
        n_kq=N_KQ, n_v=N_V, d_k=DIM, d_v=DIM, kernel_size=4,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "lens",
    [
        (2048,),  # whole pairs
        (2176,),  # odd last chunk
        (1, 1, 1000),  # two decodes then a prefill with a ragged tail
        (1000,),
        (1, 1, 2048),
        (3, 1000),  # a short prefill ahead of a long one
        (5, 300, 1, 700),  # a 1-token sequence inside the PER_SEQ segment
    ],
)
def test_chunk_pair_bit_identical(lens, monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", True)
    tokens = sum(lens)
    n = len(lens)
    x = _inputs(tokens, seed=tokens, num_seqs=n)
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]), jnp.int32)
    indices = jnp.arange(1, n + 1, dtype=jnp.int32)
    # Only the leading 1-token sequences form the batched decode segment.
    decodes = next((i for i, t in enumerate(lens) if t != 1), n)
    distribution = jnp.asarray([decodes, n, n], jnp.int32)
    seq_lens = jnp.asarray([256 + t for t in lens], jnp.int32)

    out = {}
    for pair in (False, True):
        monkeypatch.setattr(envs, "TPU_GDN_CHUNK_PAIR", pair)
        (conv, ssm), o = _run(x, query_start_loc, indices, distribution, seq_lens)
        out[pair] = (np.asarray(o, np.float32), np.asarray(conv), np.asarray(ssm, np.float32))
    for got, want in zip(out[True], out[False]):
        np.testing.assert_array_equal(got, want)
