# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The lazy conv state of the Qwen3.5 GDN v3 copy (`wrapper._lazy_conv_state`).

A PER_SEQ chunked tile can be partial without being its sequence's last: an
internal prefill checkpoint ends a tile at its checkpoint row, and PCP splits
tiles at rank-slot boundaries. Such a tile must hand the next tile the conv
rows that end at its last real token. Every tile also leaves its final conv
state in window position 0, which the output ref copies to a prefill
checkpoint block. Runs `_lazy_conv_state` itself in Pallas interpret mode on
CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.qwen3_5.kernels.gdn import compute_conv1d, config, wrapper

_UNWRITTEN = -7.0


def _cfg() -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.float32),
        ),
        batch_size=256,
        dim_size=128,
        kernel_size=4,
        tile_size=64,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )


def _lhs(seed: int) -> jax.Array:
    cfg = _cfg()
    rows = cfg.prev_kernel_size + cfg.chunk_size
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((rows, 1, cfg.dim_size)), jnp.float32)


def _run(lhs: jax.Array, real_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (window position 0 of the conv state out, the carried rows)."""
    cfg = _cfg()
    kprev = cfg.prev_kernel_size

    def kernel(lhs_ref, rs_ref, conv_out_ref, carry_ref):
        conv_out_ref[...] = jnp.full(conv_out_ref.shape, _UNWRITTEN, jnp.float32)
        carry_ref[...] = jnp.full(carry_ref.shape, -9.0, jnp.float32)
        wrapper._lazy_conv_state(
            lhs=lhs_ref[...],
            real_sizes=rs_ref,
            conv_state_out_slot_ref=conv_out_ref,
            carry_conv_scratch_ref=carry_ref,
            cfg=cfg,
        )

    state_shape = lhs.shape[1:]
    conv_out, carry = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, 1, kprev, *state_shape), jnp.float32),
            jax.ShapeDtypeStruct((1, kprev, *state_shape), jnp.float32),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
        ],
        interpret=True,
    )(lhs, jnp.asarray([real_size], jnp.int32))
    return np.asarray(conv_out)[0, 0], np.asarray(carry)[0]


@pytest.mark.parametrize("real_size", [1, 17, 63])
def test_a_partial_tile_carries_the_rows_ending_at_its_last_real_token(real_size):
    """The next tile of the sequence resumes from the rows after its last real
    token, whether or not this tile was the sequence's last."""
    kprev = _cfg().prev_kernel_size
    lhs = _lhs(real_size)
    conv_out, carry = _run(lhs, real_size)
    expected = np.asarray(lhs)[real_size : real_size + kprev]
    np.testing.assert_array_equal(carry, expected)
    np.testing.assert_array_equal(conv_out, expected)


def test_a_full_tile_stores_and_carries_the_static_tail():
    """A full tile, last or not, leaves the static tail both in window
    position 0 (a prefill checkpoint block may receive it) and in the carry."""
    cfg = _cfg()
    chunk, kprev = cfg.chunk_size, cfg.prev_kernel_size
    lhs = _lhs(9)
    conv_out, carry = _run(lhs, chunk)
    tail = np.asarray(lhs)[chunk : chunk + kprev]
    np.testing.assert_array_equal(carry, tail)
    np.testing.assert_array_equal(conv_out, tail)


@pytest.mark.parametrize("real_size", [1, 23, 64])
def test_the_lazy_state_equals_the_eager_select(real_size):
    """The lazy state matches the masked select causal_conv1d makes on every
    tile, for partial and full tiles alike."""
    cfg = _cfg()
    lhs = _lhs(100 + real_size)
    conv_out, carry = _run(lhs, real_size)
    _, eager_state = compute_conv1d.causal_conv1d(
        real_sizes=jnp.asarray([real_size], jnp.int32),
        lhs=lhs[None],
        conv_weight=jnp.ones((cfg.kernel_size, 1, cfg.dim_size), jnp.float32),
        conv_bias=None,
        cfg=cfg,
    )
    eager = np.asarray(eager_state)[0, -1]
    np.testing.assert_array_equal(conv_out, eager)
    np.testing.assert_array_equal(carry, eager)
