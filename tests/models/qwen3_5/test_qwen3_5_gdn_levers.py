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
"""The GDN v3 levers of the Qwen3.5 kernel copy, checked on CPU against main's
kernel in kernels/gdn/v3: the prefill shared k k^T (equal up to rounding, one
k k^T dot per kq head)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn as main_compute_gdn
from vllm_torchtpu.kernels.gdn.v3 import config as main_config
from vllm_torchtpu.models.qwen3_5.kernels.gdn import compute_gdn, config

N_KQ, N_V, D_K, D_V = 2, 8, 128, 128


@pytest.fixture(autouse=True)
def _tpu7x_info():
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _cfg(cfg_module, mode, *, tile_size, window_size=1):
    return cfg_module.GDNConfig(
        mode=mode,
        dtypes=cfg_module.Dtypes(
            act_in=jnp.dtype(jnp.float32),
            act_out=jnp.dtype(jnp.bfloat16),
            compute=jnp.dtype(jnp.float32),
            recurrent_state=jnp.dtype(jnp.float32),
            conv_state=jnp.dtype(jnp.float32),
        ),
        batch_size=256,
        dim_size=2 * N_KQ * D_K + N_V * D_V,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=N_KQ,
        num_v_heads=N_V,
        kq_head_dim=D_K,
        v_head_dim=D_V,
        window_size=window_size,
    )


def _dot_shapes(fn, *args):
    """(lhs shape, rhs shape, contracting dims) of every dot_general in fn."""
    jaxpr = jax.make_jaxpr(fn)(*args).jaxpr
    shapes = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            contract, _ = eqn.params["dimension_numbers"]
            shapes.append(
                (
                    tuple(eqn.invars[0].aval.shape),
                    tuple(eqn.invars[1].aval.shape),
                    tuple(map(tuple, contract)),
                )
            )
    return shapes


# --- prefill: shared k k^T -----------------------------------------------------


def _chunked_inputs(chunk, seed=0):
    rngs = iter(jax.random.split(jax.random.key(seed), 8))
    aligned_v = 128
    q = jax.random.normal(next(rngs), (N_KQ, chunk, D_K), jnp.float32)
    k = jax.random.normal(next(rngs), (N_KQ, chunk, D_K), jnp.float32)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * D_K**-0.5
    v = jax.random.normal(next(rngs), (N_V, chunk, D_V), jnp.float32)
    gating_log = -jax.nn.softplus(
        jax.random.normal(next(rngs), (1, chunk, aligned_v), jnp.float32)
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(next(rngs), (1, chunk, aligned_v), jnp.float32)
    )
    state = 0.1 * jax.random.normal(next(rngs), (N_V, D_K, D_V), jnp.float32)
    return q, k, v, gating_log, beta, state


@pytest.mark.parametrize("chunk", [16, 64])
def test_shared_kkt_chunk_matches_main_up_to_rounding(chunk):
    args = _chunked_inputs(chunk, seed=chunk)
    out, state = compute_gdn.chunked_gdn_per_seq(
        *args, _cfg(config, config.GDNMode.PER_SEQ, tile_size=chunk)
    )
    ref_out, ref_state = main_compute_gdn.chunked_gdn_per_seq(
        *args, _cfg(main_config, main_config.GDNMode.PER_SEQ, tile_size=chunk)
    )
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state, ref_state, rtol=1e-4, atol=1e-4)


def test_shared_kkt_computes_one_kkt_per_kq_head():
    """k k^T is a dot over the kq heads; no per-value-head (beta k) k^T dot."""
    chunk = 16
    cfg = _cfg(config, config.GDNMode.PER_SEQ, tile_size=chunk)
    args = _chunked_inputs(chunk)
    shapes = _dot_shapes(lambda *a: compute_gdn.chunked_gdn_per_seq(*a, cfg), *args)
    per_v_head = ((N_V, chunk, D_K), (N_V, chunk, D_K), ((2,), (2,)))
    per_kq_head = ((N_KQ, chunk, D_K), (N_KQ, chunk, D_K), ((2,), (2,)))
    assert per_v_head not in shapes
    # k k^T and q k^T.
    assert shapes.count(per_kq_head) == 2
