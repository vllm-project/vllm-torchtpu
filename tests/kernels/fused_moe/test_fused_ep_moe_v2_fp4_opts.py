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
"""Device tests for the fused EP layer's fp4 options (eight-way slice).

- fp4_body="dequant" tracks the dense reference within the fp8 bands, at
  tile heights 128 and 256, and the two heights agree bit for bit (a row's
  arithmetic does not depend on the tile it sits in).
- A 256-row tile on the blockscale body agrees bit for bit with 128.
- local_dispatch hands the kernel the same rows as the flat gather: the
  layer output is bit-identical.
- shared_partial equals the layer plus a separate replica-group psum of the
  partial, to bf16 rounding of the two summation orders.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.kernels.fused_moe.test_fused_ep_moe_v2 import (
    AXIS,
    FP8_RELATIVE_BOUND,
    FP8_TOKEN_BOUND,
    _dense_reference,
    _make_fp4_weights,
    _make_inputs,
    _mesh,
    _relative_l2,
    _worst_token_relative_l2,
)
from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2, kernel, layer

pytestmark = pytest.mark.multichip

SHAPE = dict(hidden=1024, inter=512, e_total=32, topk=4, tokens=4096)
QB = 128
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7))


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    if jax.device_count() < 8:
        pytest.skip("needs the eight-way slice")


@pytest.fixture(scope="module")
def operands():
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh, 51, hidden=SHAPE["hidden"], inter=SHAPE["inter"],
        e_total=SHAPE["e_total"], rhs_qb=QB,
    )
    x, gating = _make_inputs(
        mesh, 52, tokens=SHAPE["tokens"], hidden=SHAPE["hidden"], e_total=SHAPE["e_total"]
    )
    return mesh, weights, scales, decoded, x, gating


def _run(mesh, weights, scales, x, gating, **kw):
    kernel._BUILD_CACHE.clear()
    layer._LAYER_SM_CACHE.clear()
    common = dict(topk=SHAPE["topk"], renormalize=True, mesh=mesh,
                  weight_format=WeightFormat.FP4, rhs_qb=QB)
    common.update(kw)
    common.setdefault("capacity", 128)
    return np.asarray(fused_ep_moe_v2(x, *weights, *scales, gating, **common).astype(jnp.float32))


def test_dequant_body_tracks_reference_and_tile_heights_agree(operands):
    mesh, weights, scales, decoded, x, gating = operands
    ref = np.asarray(_dense_reference(mesh, x, *decoded, None, None, gating, topk=SHAPE["topk"]))
    outs = {}
    for cap in (128, 256):
        outs[cap] = _run(mesh, weights, scales, x, gating, capacity=cap, fp4_body="dequant")
        err, worst = _relative_l2(outs[cap], ref), _worst_token_relative_l2(outs[cap], ref)
        print(f"dequant cap={cap} relative_l2={err} worst={worst}")
        assert err < FP8_RELATIVE_BOUND and worst < FP8_TOKEN_BOUND
    np.testing.assert_array_equal(outs[256], outs[128])
    base = _run(mesh, weights, scales, x, gating)
    # Not the same arithmetic as the fp8 body, but the same layer.
    assert _relative_l2(outs[128], base) < FP8_RELATIVE_BOUND


def test_blockscale_256_row_tiles_are_bit_exact(operands):
    mesh, weights, scales, _, x, gating = operands
    np.testing.assert_array_equal(
        _run(mesh, weights, scales, x, gating, capacity=256),
        _run(mesh, weights, scales, x, gating, capacity=128),
    )


def _replicated(mesh, x, gating):
    """Inputs replicated inside each of GROUPS (TP-replicated MoE input)."""
    t = x.shape[0] // 8
    idx = np.concatenate([np.arange(t) + (r // 4) * 4 * t for r in range(8)])
    sh = NamedSharding(mesh, P(AXIS))
    return jax.device_put(x[idx], sh), jax.device_put(gating[idx], sh)


def test_local_dispatch_is_bit_exact(operands):
    mesh, weights, scales, _, x, gating = operands
    xr, gr = _replicated(mesh, x, gating)
    for body in ("blockscale", "dequant"):
        np.testing.assert_array_equal(
            _run(mesh, weights, scales, xr, gr, token_replica_groups=GROUPS,
                 local_dispatch=True, fp4_body=body),
            _run(mesh, weights, scales, xr, gr, token_replica_groups=GROUPS, fp4_body=body),
        )


def test_shared_fold_equals_separate_group_sum(operands):
    mesh, weights, scales, _, x, gating = operands
    xr, gr = _replicated(mesh, x, gating)
    sh = NamedSharding(mesh, P(AXIS))
    partial = jax.device_put(
        (jax.random.normal(jax.random.key(9), xr.shape, jnp.float32) / 8).astype(jnp.bfloat16), sh
    )
    folded = _run(mesh, weights, scales, xr, gr, token_replica_groups=GROUPS,
                  shared_partial=partial, fp4_body="dequant")
    plain = _run(mesh, weights, scales, xr, gr, token_replica_groups=GROUPS, fp4_body="dequant")
    summed = jax.jit(jax.shard_map(
        lambda p: lax.psum(p, AXIS, axis_index_groups=GROUPS), mesh=mesh,
        in_specs=P(AXIS), out_specs=P(AXIS), check_vma=False))(partial)
    want = (jnp.asarray(plain, jnp.bfloat16) + summed).astype(jnp.float32)
    np.testing.assert_allclose(folded, np.asarray(want), rtol=2e-2, atol=2e-2)
    assert _relative_l2(folded, np.asarray(want)) < 5e-3
