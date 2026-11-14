# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_CONV_HISTORY_FAST keeps the request-owned PLE conv bit-identical."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen38_flash.experimental.kernels import ple


@pytest.mark.parametrize("lengths", [[5, 0, 9, 1], [1, 1, 1, 1], [12, 3, 0, 0]])
@pytest.mark.parametrize("dilation", [1, 2])
def test_fast_history_is_bit_identical(monkeypatch, lengths, dilation):
    rng = np.random.default_rng(len(lengths) * 7 + dilation + sum(lengths))
    kernel, channels, padded = 4, 16, 32
    state_len = (kernel - 1) * dilation
    requests = len(lengths)
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    x = jnp.asarray(rng.normal(size=(padded, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, kernel)), jnp.bfloat16)
    history = jnp.asarray(
        rng.normal(size=(requests, state_len, channels)), jnp.bfloat16
    )
    active = np.asarray(lengths) > 0
    active[-1] = False if requests > 2 else active[-1]
    slots = jnp.asarray(np.where(active, np.arange(requests), -1), jnp.int32)
    initial = jnp.asarray(rng.random(requests) > 0.3)

    def run(flag):
        monkeypatch.setenv("VLLM_TPU_PLE_CONV_HISTORY_FAST", flag)
        out, upd = ple.short_conv(
            x,
            weight,
            history,
            starts,
            slots,
            initial,
            dilation=dilation,
            request_owned_history=True,
        )
        return np.asarray(out, np.float32), np.asarray(upd, np.float32)

    base_out, base_upd = run("0")
    fast_out, fast_upd = run("1")
    np.testing.assert_array_equal(fast_out, base_out)
    np.testing.assert_array_equal(fast_upd, base_upd)


def test_fp8_decode_matches_rounded_widening():
    rng = np.random.default_rng(0)
    raw = jnp.asarray(rng.integers(0, 255, size=(8, 64)), jnp.uint8)
    raw = jax.lax.bitcast_convert_type(raw, jnp.int8)
    values = jax.lax.bitcast_convert_type(raw, jnp.float8_e4m3fn)
    scale = jnp.float32(0.37)
    ref = (
        values.astype(jnp.float32).astype(jnp.bfloat16) * scale.astype(jnp.bfloat16)
    ).astype(jnp.bfloat16)
    out = ple.decode_embeddings(raw, scale)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref, np.float32)
    )
