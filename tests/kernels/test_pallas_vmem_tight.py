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
"""TPU_PALLAS_VMEM_TIGHT: working-set accounting and the limit it sets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pallas_vmem_tight as vt

_MIB = 1 << 20


def _toy(limit, name="toy_vmem_tight"):
    def kernel(x_ref, o_ref, s_ref):
        def body(t_ref):
            t_ref[...] = x_ref[...] * 2
            s_ref[...] = t_ref[...] + 1
            o_ref[...] = s_ref[...] - x_ref[...]

        pl.run_scoped(body, pltpu.VMEM((256, 512), jnp.float32))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1024, 512), jnp.float32),
        grid=(4,),
        in_specs=[pl.BlockSpec((256, 512), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((256, 512), lambda i: (i, 0)),
        scratch_shapes=[pltpu.VMEM((256, 512), jnp.float32)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=limit),
        name=name,
    )


def test_working_set_counts_blocks_scratch_and_run_scoped():
    x = jax.ShapeDtypeStruct((1024, 512), jnp.float32)
    eqn = vt._find_pallas_eqn(jax.make_jaxpr(_toy(100 * _MIB))(x).jaxpr)
    blk = 256 * 512 * 4
    # in + out double-buffered, one scratch, one run_scoped buffer, and the body's
    # array values (x*2, t+1, s-x: three loads and three results)
    assert vt._working_set(eqn) >= 2 * 2 * blk + blk + blk + 3 * blk
    assert vt._nbytes(jax.ShapeDtypeStruct((32, 4), jnp.float32)) == 32 * 128 * 4
    assert vt._nbytes(jax.ShapeDtypeStruct((8, 7168), jnp.bfloat16)) == 16 * 7168 * 2


def test_tight_limit_bounds(monkeypatch):
    for k in ("MIN_MIB", "SLACK", "MARGIN_MIB"):
        monkeypatch.delenv(f"TPU_PALLAS_VMEM_TIGHT_{k}", raising=False)
    assert vt.tight_limit(0, 64 * _MIB) == 8 * _MIB  # floor
    assert vt.tight_limit(20 * _MIB, 64 * _MIB) == 38 * _MIB  # 1.5 * 20 + 8
    assert vt.tight_limit(80 * _MIB, 64 * _MIB) == 64 * _MIB  # never raises


def _limit_in_jaxpr(f, x):
    eqn = vt._find_pallas_eqn(jax.make_jaxpr(f)(x).jaxpr)
    cp = eqn.params["compiler_params"]
    if isinstance(cp, dict):
        cp = cp.get("mosaic_tpu") or next(iter(cp.values()))
    return cp.vmem_limit_bytes


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setenv("TPU_PALLAS_VMEM_TIGHT", "1")
    monkeypatch.setattr(vt, "_mode", lambda: 1)  # the suite may run with =2 set
    vt._memo.clear()
    orig = pl.pallas_call
    vt.install_pallas_vmem_tight()
    yield
    pl.pallas_call = orig


def test_patch_sets_the_tight_limit_and_skips(patched, monkeypatch):
    x = jax.ShapeDtypeStruct((1024, 512), jnp.float32)
    ws = vt._working_set(vt._find_pallas_eqn(jax.make_jaxpr(_toy(100 * _MIB, name="t0"))(x).jaxpr))
    assert _limit_in_jaxpr(_toy(100 * _MIB), x) == vt.tight_limit(ws, 64 * _MIB - 64 * 1024) < 64 * _MIB
    assert _limit_in_jaxpr(_toy(4 * _MIB), x) == 4 * _MIB  # never raised
    # the default skip list leaves the fused EP MoE family alone
    assert _limit_in_jaxpr(_toy(100 * _MIB, name="fused_ep_moe_v2_x"), x) == 100 * _MIB


def test_bit_identical_on_tpu(monkeypatch):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("TPU only")
    x = jnp.asarray(np.random.default_rng(0).standard_normal((1024, 512)), jnp.float32)
    ref = jax.jit(_toy(100 * _MIB))(x)
    monkeypatch.setenv("TPU_PALLAS_VMEM_TIGHT", "1")
    orig = pl.pallas_call
    vt.install_pallas_vmem_tight()
    try:
        out = jax.jit(_toy(100 * _MIB, name="toy_vmem_tight2"))(x)
    finally:
        pl.pallas_call = orig
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(out))


def test_probe_mode_on_tpu(monkeypatch):
    """TPU_PALLAS_VMEM_TIGHT=2 compiles the kernel at ladder rungs and keeps the lowest that fits."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("TPU only")
    x = jnp.asarray(np.random.default_rng(1).standard_normal((1024, 512)), jnp.float32)
    ref = jax.jit(_toy(100 * _MIB, name="toy_probe_ref"))(x)
    monkeypatch.setenv("TPU_PALLAS_VMEM_TIGHT", "2")
    monkeypatch.setattr(vt, "_mode", lambda: 2)
    vt._memo.clear()
    orig = pl.pallas_call
    vt.install_pallas_vmem_tight()
    try:
        f = _toy(100 * _MIB, name="toy_probe")
        lim = _limit_in_jaxpr(f, jax.ShapeDtypeStruct(x.shape, x.dtype))
        out = jax.jit(f)(x)
    finally:
        pl.pallas_call = orig
    assert lim <= 16 * _MIB, lim  # a ~3 MiB working set: low rung + headroom
    np.testing.assert_array_equal(np.asarray(ref), np.asarray(out))
