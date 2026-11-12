# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for 48-value-head GDN v3 tiling and recurrent scan fusion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, tiling


@pytest.mark.parametrize(
    ("num_v_heads", "expected"),
    [(6, 16), (8, 16), (12, 8), (16, 8), (24, 4), (32, 4), (48, 2), (64, 2)],
)
def test_triangular_block_size_is_power_of_two(num_v_heads, expected):
    cfg = config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.bfloat16),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.bfloat16),
            conv_state=jnp.dtype(jnp.float32),
        ),
        batch_size=64,
        dim_size=2048,
        kernel_size=4,
        tile_size=64,
        num_kq_heads=max(1, num_v_heads // 3),
        num_v_heads=num_v_heads,
        kq_head_dim=128,
        v_head_dim=128,
    )
    assert cfg.triangular_block_size == expected
    assert 64 % cfg.triangular_block_size == 0


def test_get_tile_sizes_48v_and_bf16_budget():
    vmem_192m = 192 * 1024 * 1024
    limit_w1 = int(config.GDNConfig.DEFAULT_VMEM_FRACTION * vmem_192m)
    limit_w4 = int(config.GDNConfig.WINDOWED_VMEM_FRACTION * vmem_192m)

    # Non-power-of-two head counts pick full 128 prefill chunks at 192 MiB.
    dec, mix = tiling.get_tile_sizes(
        batch_size=2048,
        num_seqs=1,
        padded_batch_size=2048,
        n_kq=16,
        n_v=48,
        d_k=128,
        d_v=128,
        kernel_size=4,
        conv_state_dim_size=2 * 16 * 128 + 48 * 128,
        act_out_dtype=jnp.dtype(jnp.bfloat16),
        recurrent_state_dtype=jnp.dtype(jnp.bfloat16),
        num_lanes=128,
        decode_vmem_limit_bytes=limit_w1,
        mixed_vmem_limit_bytes=limit_w1,
        window_size=1,
    )
    assert (dec, mix) == (1, 128)

    # bf16 1-token decode bumps step budget for non-power-of-two head counts,
    # whereas window_size > 1 keeps uninflated pipeline step counts.
    assert tiling.decode_tile_target(
        4, 12, window_size=1, rec_state_bytes=2
    ) > tiling.decode_tile_target(4, 12, window_size=4, rec_state_bytes=2)
    dec_w4, _ = tiling.get_tile_sizes(
        batch_size=32,
        num_seqs=8,
        padded_batch_size=32,
        n_kq=4,
        n_v=12,
        d_k=128,
        d_v=128,
        kernel_size=4,
        conv_state_dim_size=2 * 4 * 128 + 12 * 128,
        act_out_dtype=jnp.dtype(jnp.bfloat16),
        recurrent_state_dtype=jnp.dtype(jnp.bfloat16),
        num_lanes=128,
        decode_vmem_limit_bytes=limit_w4,
        mixed_vmem_limit_bytes=limit_w1,
        window_size=4,
    )
    assert dec_w4 == 4


@pytest.mark.parametrize(
    ("seq_tile_size", "n_kq", "n_v", "window_size"),
    [
        (4, 4, 12, 1),  # 1-token decode: vectorized [k, q]-fused path
        (4, 4, 12, 4),  # verify decode <= 96 head states: vectorized [k, q]-fused path
        (16, 2, 8, 4),  # verify decode > 96 head states: per-seq VMEM safeguard path
    ],
)
def test_recurrent_gdn_matches_per_seq_reference(seq_tile_size, n_kq, n_v, window_size):
    d_k, d_v = 16, 16
    dim_size = 2 * n_kq * d_k + n_v * d_v
    cfg = config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=jnp.dtype(jnp.float32),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.float32),
        ),
        batch_size=seq_tile_size * window_size,
        dim_size=dim_size,
        kernel_size=4,
        tile_size=seq_tile_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        window_size=window_size,
    )
    rng = np.random.default_rng(0)
    real_sizes = jnp.asarray(
        rng.integers(1, window_size + 1, size=(seq_tile_size,), dtype=np.int32)
    )
    q = jnp.asarray(
        rng.normal(scale=0.3, size=(seq_tile_size, n_kq, window_size, 1, d_k)),
        dtype=jnp.float32,
    )
    k = jnp.asarray(
        rng.normal(scale=0.3, size=(seq_tile_size, n_kq, window_size, 1, d_k)),
        dtype=jnp.float32,
    )
    v = jnp.asarray(
        rng.normal(scale=0.3, size=(seq_tile_size, n_v, window_size, 1, d_v)),
        dtype=jnp.float32,
    )
    b = jnp.asarray(
        rng.normal(scale=0.3, size=(seq_tile_size, 1, window_size, 1, 128)),
        dtype=jnp.float32,
    )
    a = jnp.asarray(
        rng.normal(scale=0.3, size=(seq_tile_size, 1, window_size, 1, 128)),
        dtype=jnp.float32,
    )
    state_prev = jnp.asarray(
        rng.normal(scale=0.2, size=(seq_tile_size, n_v, d_k, d_v)),
        dtype=jnp.float32,
    )
    a_log = jnp.asarray(rng.normal(scale=0.2, size=(128,)), dtype=jnp.float32)
    dt_bias = jnp.asarray(rng.normal(scale=0.2, size=(128,)), dtype=jnp.float32)

    out, rec_state = compute_gdn.recurrent_gdn(
        real_sizes=real_sizes,
        q_compact=q,
        k_compact=k,
        v_compact=v,
        b_compact=b,
        a_compact=a,
        state_prev=state_prev,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
    )

    # Build reference using compute_gdn.recurrent_gdn_per_seq directly.
    mask = jax.lax.broadcasted_iota(
        jnp.float32, (seq_tile_size, 1, window_size, 1, 1), 2
    ) < real_sizes.reshape(-1, 1, 1, 1, 1).astype(jnp.float32)
    q_m = compute_gdn.l2_norm(jnp.where(mask, q, 0)) * (d_k**-0.5)
    k_m = compute_gdn.l2_norm(jnp.where(mask, k, 0))
    kt_m = compute_gdn.fused_transpose_broadcast(k_m, src_dim=4, dst_dim=3)
    v_m = jnp.where(mask, v, 0)
    beta_m = compute_gdn.fused_transpose_broadcast(
        jnp.where(mask, jax.nn.sigmoid(b), 0), src_dim=4, dst_dim=1
    )[:, :n_v]
    gate_m = jnp.exp(
        jnp.where(
            mask,
            -jnp.exp(a_log.reshape(1, 1, 1, 1, -1))
            * jax.nn.softplus(a + dt_bias.reshape(1, 1, 1, 1, -1)),
            0,
        )
    )
    gate_m = compute_gdn.fused_transpose_broadcast(gate_m, src_dim=4, dst_dim=1)[
        :, :n_v
    ]

    ref_outs, ref_states = [], []
    for i in range(seq_tile_size):
        ro, rs = compute_gdn.recurrent_gdn_per_seq(
            q_m[i], k_m[i], kt_m[i], v_m[i], gate_m[i], beta_m[i], state_prev[i], cfg
        )
        ref_outs.append(ro)
        ref_states.append(rs)

    np.testing.assert_allclose(out, jnp.stack(ref_outs, axis=0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        rec_state, jnp.stack(ref_states, axis=0), rtol=1e-5, atol=1e-5
    )
