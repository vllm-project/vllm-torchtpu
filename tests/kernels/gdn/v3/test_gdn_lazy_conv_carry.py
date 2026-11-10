# SPDX-License-Identifier: Apache-2.0
"""Bug 2 (PCPL7B): a lazily-converted GDN tile that is partial but not its sequence's
last (PCP: a request starting mid communication chunk) must carry the conv rows ending at
its last real token. Runs `_lazy_conv_state` itself in Pallas interpret mode on CPU."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, wrapper


def _cfg():
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(act_in=jnp.float32.dtype, act_out=jnp.bfloat16.dtype,
                             compute=jnp.float32.dtype, recurrent_state=jnp.float32.dtype,
                             conv_state=jnp.float32.dtype),
        batch_size=256, dim_size=128, kernel_size=4, tile_size=64,
        num_kq_heads=1, num_v_heads=1, kq_head_dim=128, v_head_dim=128)


def _run(lhs, real_size, is_last):
    cfg = _cfg()
    kprev = cfg.prev_kernel_size

    def kernel(lhs_ref, rs_ref, last_ref, conv_out_ref, carry_ref):
        conv_out_ref[...] = jnp.full(conv_out_ref.shape, -7.0, conv_out_ref.dtype)
        carry_ref[...] = jnp.full(carry_ref.shape, -9.0, carry_ref.dtype)
        md = types.SimpleNamespace(p_id_is_last_tile=last_ref)
        wrapper._lazy_conv_state(lhs=lhs_ref[...], real_sizes=rs_ref, conv_state_out_slot_ref=conv_out_ref,
                                 carry_conv_scratch_ref=carry_ref, metadata_ref=md, p_id=0, cfg=cfg)

    shape = lhs.shape
    conv_out, carry = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((1, 1, kprev) + shape[1:], lhs.dtype),
                   jax.ShapeDtypeStruct((1, kprev) + shape[1:], lhs.dtype)),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.SMEM),
                  pl.BlockSpec(memory_space=pltpu.SMEM)],
        interpret=True,
    )(lhs, jnp.asarray([real_size], jnp.int32), jnp.asarray([[is_last]], jnp.bool_))
    return np.asarray(conv_out)[0, 0], np.asarray(carry)[0]


@pytest.mark.parametrize("real_size", [1, 17, 63])
def test_partial_first_tile_carries_rows_ending_at_last_real_token(real_size):
    cfg = _cfg()
    chunk, kprev = cfg.chunk_size, cfg.prev_kernel_size
    lhs = jnp.asarray(np.random.default_rng(real_size).standard_normal((kprev + chunk, 1, cfg.dim_size)), jnp.float32)
    _, carry = _run(lhs, real_size, is_last=False)        # partial, NOT the sequence's last tile (PCP)
    np.testing.assert_array_equal(carry, np.asarray(lhs)[real_size: real_size + kprev])


def test_partial_last_tile_state_and_carry_agree():
    cfg = _cfg()
    chunk, kprev = cfg.chunk_size, cfg.prev_kernel_size
    lhs = jnp.asarray(np.random.default_rng(5).standard_normal((kprev + chunk, 1, cfg.dim_size)), jnp.float32)
    conv_out, carry = _run(lhs, 23, is_last=True)
    np.testing.assert_array_equal(conv_out, np.asarray(lhs)[23: 23 + kprev])
    np.testing.assert_array_equal(carry, conv_out)


def test_full_tile_unchanged_static_tail():
    cfg = _cfg()
    chunk, kprev = cfg.chunk_size, cfg.prev_kernel_size
    lhs = jnp.asarray(np.random.default_rng(9).standard_normal((kprev + chunk, 1, cfg.dim_size)), jnp.float32)
    for is_last in (False, True):
        conv_out, carry = _run(lhs, chunk, is_last=is_last)
        np.testing.assert_array_equal(carry, np.asarray(lhs)[chunk: chunk + kprev])
        if is_last:
            np.testing.assert_array_equal(conv_out, np.asarray(lhs)[chunk: chunk + kprev])
