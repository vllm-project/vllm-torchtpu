# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_CONV_HISTORY_FAST keeps the request-owned PLE conv bit-identical."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import ple


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
            x, weight, history, starts, slots, initial,
            dilation=dilation, request_owned_history=True,
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
    ref = (values.astype(jnp.float32).astype(jnp.bfloat16) * scale.astype(
        jnp.bfloat16
    )).astype(jnp.bfloat16)
    out = ple.decode_embeddings(raw, scale)
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(ref, np.float32)
    )


@pytest.mark.parametrize(
    ("padded", "expected_shapes"),
    [
        # All 3 non-zero taps (distances 9, 6, 3) satisfy R * d < T for R=4, T=64.
        (64, [(4, 9), (4, 6), (4, 3), (4, 9)]),
        # Mixed regime (R=4, T=32): tap 0 takes else (T=32), taps 1&2 take fast_history.
        (32, [(32,), (4, 6), (4, 3), (4, 9)]),
    ],
)
def test_fast_history_routes_through_row_kernel(monkeypatch, padded, expected_shapes):
    """Every short_conv history gather (incl. fast_history) uses gather_history_rows."""
    rng = np.random.default_rng(padded)
    kernel, dilation, channels = 4, 3, 16
    state_len = (kernel - 1) * dilation
    lengths = [4, 4, 4, 12]
    requests = len(lengths)
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
    x = jnp.asarray(rng.normal(size=(padded, channels)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(channels, kernel)), jnp.bfloat16)
    history = jnp.asarray(
        rng.normal(size=(requests, state_len, channels)), jnp.bfloat16
    )
    slots = jnp.arange(requests, dtype=jnp.int32)
    initial = jnp.asarray([True, True, True, False])

    monkeypatch.setenv("VLLM_TPU_PLE_CONV_HISTORY_FAST", "0")
    monkeypatch.setenv("VLLM_TPU_PLE_HISTORY_ROW_KERNEL", "0")
    ref_out, ref_upd = ple.short_conv(
        x, weight, history, starts, slots, initial,
        dilation=dilation, request_owned_history=True,
    )

    seen_shapes = []

    def spy_gather(hist, rows):
        seen_shapes.append(tuple(rows.shape))
        flat = hist.reshape(hist.shape[0] * hist.shape[1], hist.shape[2])
        assert int(jnp.min(rows)) >= 0
        assert int(jnp.max(rows)) < flat.shape[0]
        return flat[rows]

    monkeypatch.setattr(ple, "gather_history_rows", spy_gather)
    monkeypatch.setenv("VLLM_TPU_PLE_CONV_HISTORY_FAST", "1")
    monkeypatch.setenv("VLLM_TPU_PLE_HISTORY_ROW_KERNEL", "1")
    out, upd = ple.short_conv(
        x, weight, history, starts, slots, initial,
        dilation=dilation, request_owned_history=True,
    )
    assert seen_shapes == expected_shapes
    np.testing.assert_array_equal(
        np.asarray(out).view(np.uint16), np.asarray(ref_out).view(np.uint16)
    )
    np.testing.assert_array_equal(
        np.asarray(upd).view(np.uint16), np.asarray(ref_upd).view(np.uint16)
    )
