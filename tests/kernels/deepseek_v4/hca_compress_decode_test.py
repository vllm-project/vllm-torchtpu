# SPDX-License-Identifier: Apache-2.0
"""Independent full-cache checks for HCA compressor/attention composition."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import (  # noqa: E501
    _attention_hca,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (  # noqa: E501
    _compressor_jax_hca,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.hca_compress_attention import (
    attention_compressor,
    attention_compressor_shared,
)

pytestmark = pytest.mark.tpu_test


@pytest.fixture(scope="module")
def inputs():
    rng = np.random.default_rng(632)
    w = jnp.asarray(rng.normal(0, 0.03, (4096, 1024)), jnp.bfloat16)
    ape = jnp.asarray(rng.normal(0, 0.1, (128, 512)), jnp.float32)
    norm = jnp.asarray(rng.uniform(0.8, 1.2, 512), jnp.bfloat16)
    phase = np.arange(66000)[:, None] * np.linspace(0.001, 0.01, 32)[None, :]
    cs = jnp.asarray(np.concatenate((np.cos(phase), np.sin(phase)), 1), jnp.float32)
    return rng, w, ape, norm, cs


@pytest.mark.parametrize("shared_cache", (False, True))
@pytest.mark.parametrize("word_cache", (False, True))
@pytest.mark.parametrize("seqs", (1, 256))
@pytest.mark.parametrize("start,decode", ((120, True), (65528, True), (120, False)))
def test_compressor_attention_continuous_bytes(
    inputs, seqs, start, decode, word_cache, shared_cache
):
    rng, w, ape, norm, cs = inputs
    pps = (start + 16 + 1023) // 1024
    sp = jnp.asarray(np.tile(np.arange((start + 16 + 127) // 128) % 3, seqs), jnp.int32)
    mp = jnp.asarray(np.tile(rng.permutation(pps), seqs), jnp.int32)
    st = jnp.asarray(
        np.tile(np.arange((start + 16 + 31) // 32)[::-1] % 8, seqs), jnp.int32
    )
    cu = jnp.minimum(jnp.arange(seqs + 1), 1).astype(jnp.int32)
    dist = jnp.array([1, 1, 1] if decode else [0, 0, 1], jnp.int32)
    main = mla_swa.prepare_kv_inputs(
        jnp.asarray(rng.normal(size=(pps * 8, 512)), jnp.bfloat16)
    ).reshape(pps, 16, 4, 128)
    sw = mla_swa.prepare_kv_inputs(
        jnp.asarray(rng.normal(size=(3 * 128, 512)), jnp.bfloat16)
    ).reshape(3, 256, 4, 128)
    history = jnp.asarray(
        rng.normal(0, 0.1, (11 if shared_cache else 8, 256, 128))
        .astype(np.float32)
        .view(np.int32)
    )
    if word_cache:
        sw = csa_cache_layout.slabs_to_words(sw)
    else:
        history = csa_cache_layout.words_to_slabs(history)
    if shared_cache:
        history = history.at[8:].set(sw)
        sw = history
        sp = sp + 8
    sink = jnp.asarray(rng.normal(size=64), jnp.float32)
    opts = dict(
        eps=1e-6,
        sm_scale=512**-0.5,
        sliding_window=128,
        logical_page_size=128,
        swa_only=False,
        two_caches_same_buffer=False,
    )

    def ref(
        q,
        kv,
        sw,
        sl,
        sp,
        cu,
        dist,
        main,
        ml,
        att,
        mp,
        mcu,
        mdist,
        sink,
        pos,
        cs,
        h,
        w,
        ape,
        norm,
        state,
        st,
        ccu,
        kt,
        cdist,
    ):
        main, state = _compressor_jax_hca(
            h,
            w,
            ape,
            norm,
            cs,
            pos,
            st,
            ccu,
            kt,
            cdist,
            main,
            state,
            state_block_size=32,
            head_dim=512,
            compress_ratio=128,
            overlap=False,
            rms_eps=1e-6,
            quant_block=0,
        )
        q, sw = _attention_hca(
            q,
            kv,
            state if shared_cache else sw,
            sl,
            sp,
            cu,
            dist,
            main,
            ml,
            att,
            mp,
            mcu,
            mdist,
            sink,
            pos,
            cs,
            **opts,
        )
        return q, sw if shared_cache else state, main, sw

    def selected(*args, **opts):
        if not shared_cache:
            return attention_compressor(*args, **opts)
        output, main, shared = attention_compressor_shared(
            *args[:20], *args[21:], **opts
        )
        return output, shared, main, shared

    funcs = [
        jax.jit(ref),
        jax.jit(
            functools.partial(
                selected,
                **opts,
                state_block_size=32,
                compress_eps=1e-6,
                quant_block=0,
            )
        ),
    ]
    q = jnp.asarray(rng.normal(size=(1, 64, 512)), jnp.bfloat16)
    carry = [(q, history, main, sw), (q, history, main, sw)]
    for step in range(16):
        pos = jnp.array([start + step], jnp.int32)
        sl = jnp.zeros(seqs, jnp.int32).at[0].set(start + step + 1)
        ml = sl // 128
        kv = jnp.asarray(rng.normal(size=(1, 512)), jnp.bfloat16)
        h = jnp.asarray(rng.normal(0, 0.2, (1, 4096)), jnp.bfloat16)
        carry = [
            fn(
                c[0],
                kv,
                c[3],
                sl,
                sp,
                cu,
                dist,
                c[2],
                ml,
                ml[:1],
                mp,
                cu,
                dist,
                sink,
                pos,
                cs,
                h,
                w,
                ape,
                norm,
                c[1],
                st,
                cu,
                mp,
                dist,
            )
            for fn, c in zip(funcs, carry)
        ]
        for a, b in zip(*carry):
            np.testing.assert_array_equal(
                np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)
            )
