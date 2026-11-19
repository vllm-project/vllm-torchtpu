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
k k^T dot per kq head, only at the Qwen3.5-397B prefill shape; every other
shape computes main's chunk bit for bit)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.gdn.v3 import compute_gdn as main_compute_gdn
from vllm_torchtpu.kernels.gdn.v3 import config as main_config
from vllm_torchtpu.models.qwen3_5.kernels.gdn import compute_gdn, config

N_KQ, N_V, D_K, D_V = 2, 8, 128, 128
# The Qwen3.5-397B prefill legs' per-rank shape (TP1), where the shared k k^T
# was measured.
SHARED_N_KQ, SHARED_N_V = 16, 64


@pytest.fixture(autouse=True)
def _tpu7x_info():
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _cfg(cfg_module, mode, *, tile_size, window_size=1, n_kq=N_KQ, n_v=N_V):
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
        dim_size=2 * n_kq * D_K + n_v * D_V,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
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


def _chunked_inputs(chunk, seed=0, n_kq=N_KQ, n_v=N_V):
    rngs = iter(jax.random.split(jax.random.key(seed), 8))
    aligned_v = 128
    q = jax.random.normal(next(rngs), (n_kq, chunk, D_K), jnp.float32)
    k = jax.random.normal(next(rngs), (n_kq, chunk, D_K), jnp.float32)
    k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True) * D_K**-0.5
    v = jax.random.normal(next(rngs), (n_v, chunk, D_V), jnp.float32)
    gating_log = -jax.nn.softplus(
        jax.random.normal(next(rngs), (1, chunk, aligned_v), jnp.float32)
    )
    beta = jax.nn.sigmoid(
        jax.random.normal(next(rngs), (1, chunk, aligned_v), jnp.float32)
    )
    state = 0.1 * jax.random.normal(next(rngs), (n_v, D_K, D_V), jnp.float32)
    return q, k, v, gating_log, beta, state


def _per_seq_cfgs(chunk, n_kq, n_v):
    return (
        _cfg(config, config.GDNMode.PER_SEQ, tile_size=chunk, n_kq=n_kq, n_v=n_v),
        _cfg(
            main_config,
            main_config.GDNMode.PER_SEQ,
            tile_size=chunk,
            n_kq=n_kq,
            n_v=n_v,
        ),
    )


def test_shared_kkt_runs_only_at_the_397b_prefill_shape():
    on, _ = _per_seq_cfgs(64, SHARED_N_KQ, SHARED_N_V)
    assert on.shared_kkt
    # Qwen3.5-35B-A3B at TP4 and TP1 (the CI model), and the 397B decode legs'
    # TP2 and TP8 shapes, where the 397B decode configurations ran main's
    # (beta k) k^T.
    for n_kq, n_v in ((4, 8), (16, 32), (8, 32), (2, 8)):
        off, _ = _per_seq_cfgs(64, n_kq, n_v)
        assert not off.shared_kkt, (n_kq, n_v)


@pytest.mark.parametrize("chunk", [16, 64])
def test_shared_kkt_chunk_matches_main_up_to_rounding(chunk):
    args = _chunked_inputs(chunk, seed=chunk, n_kq=SHARED_N_KQ, n_v=SHARED_N_V)
    cfg, ref_cfg = _per_seq_cfgs(chunk, SHARED_N_KQ, SHARED_N_V)
    out, state = compute_gdn.chunked_gdn_per_seq(*args, cfg)
    ref_out, ref_state = main_compute_gdn.chunked_gdn_per_seq(*args, ref_cfg)
    np.testing.assert_allclose(out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(state, ref_state, rtol=1e-4, atol=1e-4)


def test_shared_kkt_computes_one_kkt_per_kq_head():
    """k k^T is a dot over the kq heads; no per-value-head (beta k) k^T dot."""
    chunk = 16
    cfg, _ = _per_seq_cfgs(chunk, SHARED_N_KQ, SHARED_N_V)
    args = _chunked_inputs(chunk, n_kq=SHARED_N_KQ, n_v=SHARED_N_V)
    shapes = _dot_shapes(lambda *a: compute_gdn.chunked_gdn_per_seq(*a, cfg), *args)
    per_v_head = ((SHARED_N_V, chunk, D_K), (SHARED_N_V, chunk, D_K), ((2,), (2,)))
    per_kq_head = ((SHARED_N_KQ, chunk, D_K), (SHARED_N_KQ, chunk, D_K), ((2,), (2,)))
    assert per_v_head not in shapes
    # k k^T and q k^T.
    assert shapes.count(per_kq_head) == 2


@pytest.mark.parametrize(("n_kq", "n_v"), [(4, 8), (16, 32), (8, 32), (2, 8)])
@pytest.mark.parametrize("chunk", [16, 64])
def test_other_shapes_compute_mains_chunk_bit_for_bit(chunk, n_kq, n_v):
    """Off the measured shape the chunk takes main's (beta k) k^T, one dot per
    value head, and equals main's chunk exactly."""
    args = _chunked_inputs(chunk, seed=chunk + n_v, n_kq=n_kq, n_v=n_v)
    cfg, ref_cfg = _per_seq_cfgs(chunk, n_kq, n_v)
    out, state = jax.jit(lambda *a: compute_gdn.chunked_gdn_per_seq(*a, cfg))(*args)
    ref_out, ref_state = jax.jit(
        lambda *a: main_compute_gdn.chunked_gdn_per_seq(*a, ref_cfg)
    )(*args)
    np.testing.assert_array_equal(out, ref_out)
    np.testing.assert_array_equal(state, ref_state)
    shapes = _dot_shapes(lambda *a: compute_gdn.chunked_gdn_per_seq(*a, cfg), *args)
    assert ((n_v, chunk, D_K), (n_v, chunk, D_K), ((2,), (2,))) in shapes
