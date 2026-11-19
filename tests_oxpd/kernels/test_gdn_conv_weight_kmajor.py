# SPDX-License-Identifier: Apache-2.0
"""A load-time `[kernel_size, 1, dim]` float32 conv weight is bitwise neutral.

`VLLM_TPU_GDN_CONV_WEIGHT_KMAJOR` hands the GDN v3 kernel the weight the
wrapper would otherwise derive on every forward, so both forms must produce
identical outputs and states. Runs in CPU interpret mode.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import wrapper


def _run(conv_weight, *, kmajor, inputs):
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    pltpu.reset_tpu_interpret_mode_state()
    try:
        with (
            use_abstract_mesh(mesh),
            mock.patch.object(envs, "TPU_ENABLE_GDN_DYNAMIC_TILING", False),
            pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(
                    dma_execution_mode="eager", skip_floating_point_ops=False
                )
            ),
        ):
            # The wrapper donates the state operands; give each run its own.
            states = {k: jnp.copy(inputs[k]) for k in ("conv_state", "recurrent_state")}
            result = wrapper.fused_conv1d_gdn(
                conv_weight=conv_weight,
                conv_weight_kmajor=kmajor,
                decode_tile_size=2,
                # CPU interpret mode needs C = 1 tiles on the prefill path.
                mixed_tile_size=1,
                **{**inputs, **states},
            )
            return jax.block_until_ready(result)
    finally:
        pltpu.reset_tpu_interpret_mode_state()


def test_kmajor_conv_weight_matches_runtime_swap():
    n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
    dim = 2 * n_kq * d_k + n_v * d_v
    num_decode, num_seqs = 4, 6
    rngs = iter(jax.random.split(jax.random.key(0), 9))
    conv_w = jax.random.normal(next(rngs), (dim, 1, kernel_size), dtype=jnp.bfloat16)
    inputs = dict(
        qkv=jax.random.normal(next(rngs), (num_seqs, dim), dtype=jnp.bfloat16),
        b=jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16),
        a=jax.random.normal(next(rngs), (num_seqs, n_v), dtype=jnp.bfloat16),
        conv_bias=jax.random.normal(next(rngs), (dim,), dtype=jnp.bfloat16),
        a_log=jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
        dt_bias=jax.random.normal(next(rngs), (n_v,), dtype=jnp.float32),
        conv_state=jax.random.normal(
            next(rngs), (num_seqs, kernel_size - 1, dim), dtype=jnp.bfloat16
        ),
        recurrent_state=jax.random.normal(
            next(rngs), (num_seqs, n_v, d_k, d_v), dtype=jnp.float32
        ),
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32),
        state_indices=jnp.arange(num_seqs, dtype=jnp.int32),
        distribution=jnp.array([num_decode, num_seqs, num_seqs], dtype=jnp.int32),
        seq_lens=jnp.full((num_seqs,), 11, dtype=jnp.int32),
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
    )

    expected = _run(conv_w, kmajor=False, inputs=inputs)
    got = _run(conv_w.swapaxes(0, 2).astype(jnp.float32), kmajor=True, inputs=inputs)

    for e, g in zip(jax.tree.leaves(expected), jax.tree.leaves(got)):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e))
