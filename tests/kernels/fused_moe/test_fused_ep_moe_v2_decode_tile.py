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
"""MOE_FUSED_EP_V2_DECODE_TILE_M: decode builds at a 32-row single-rung tile.

The tile height changes how rows are grouped into tiles, never a row's own
arithmetic, so the layer's output has to be bit-identical to the 128-row
ladder build -- including an expert whose rows overflow one short tile (the
skewed case runs several tiles per visit) and the decode glue + shared-in-op
fold path the served model takes. A build that did not change (env ignored)
would pass those trivially, so each case also asserts that the knob reached
the kernel builder (the build saw tile height 32) and that the prefill-sized
call above the token gate kept 128.
"""

import numpy as np
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2
from vllm_torchtpu.kernels.fused_moe.v2 import kernel, layer

from .test_fused_ep_moe_v2 import (
    _dense_reference,
    _make_fp4_weights,
    _make_inputs,
    _mesh,
    _relative_l2,
    FP8_RELATIVE_BOUND,
)
from .test_fused_ep_moe_v2_shared_in_op import (  # noqa: F401
    _TP4_PERMUTED,
    _by_rank,
    _inputs,
    _layer,
    mesh,
)

pytestmark = pytest.mark.multichip

SHAPE = dict(hidden=1024, inter=512, e_total=32, topk=4)
QB = 256


@pytest.fixture
def builds(monkeypatch):
    """Clears both layer caches and records each kernel build's tile height."""
    seen = []
    real = kernel.tile_height_rungs

    def spy(tile_m, *args, **kwargs):
        seen.append(tile_m)
        return real(tile_m, *args, **kwargs)

    monkeypatch.setattr(kernel, "tile_height_rungs", spy)

    def reset(tile):
        monkeypatch.setattr(envs, "MOE_FUSED_EP_V2_DECODE_TILE_M", tile, raising=False)
        monkeypatch.setattr(
            envs, "MOE_FUSED_EP_V2_DECODE_TILE_MAX_LOCAL_TOKENS", 16, raising=False
        )
        kernel._BUILD_CACHE.clear()
        layer._LAYER_SM_CACHE.clear()
        seen.clear()
        return seen

    yield reset
    kernel._BUILD_CACHE.clear()
    layer._LAYER_SM_CACHE.clear()


def test_single_rung_at_the_short_tile():
    assert kernel.tile_height_rungs(32, 4096, 24, 16) == (32,)
    assert kernel.tile_height_rungs(16, 4096, 24, 16) == (16,)


@pytest.mark.parametrize("tokens", [8, 16, 64, 128, 256])
@pytest.mark.parametrize("skew", [False, True], ids=["uniform", "one-expert"])
def test_fp4_decode_tile_is_bit_exact(builds, tokens, skew):
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(mesh, 71, rhs_qb=QB, **{
        k: SHAPE[k] for k in ("hidden", "inter", "e_total")})
    x, gating = _make_inputs(mesh, 72, tokens=tokens, hidden=SHAPE["hidden"],
                             e_total=SHAPE["e_total"])
    if skew:
        # Every token's first choice is expert 5: at 128 tokens that expert
        # holds 128 rows, four 32-row tiles in one visit.
        gating = gating.at[:, 5].add(50.0)
    common = dict(topk=SHAPE["topk"], renormalize=True, mesh=mesh, capacity=128,
                  weight_format=WeightFormat.FP4, rhs_qb=QB, sharded_plan=True)
    outs = {}
    for tile in (0, 32):
        seen = builds(tile)
        outs[tile] = np.asarray(fused_ep_moe_v2(x, *weights, *scales, gating, **common))
        # 256 tokens = t_local 32 on eight cores: above the gate, keeps 128.
        want = 32 if tile and tokens // mesh.devices.size <= 16 else 128
        assert seen and set(seen) == {want}, (tile, tokens, seen)
    np.testing.assert_array_equal(outs[32], outs[0])
    ref = _dense_reference(mesh, x, *decoded, None, None, gating, topk=SHAPE["topk"])
    assert _relative_l2(outs[32], ref) < FP8_RELATIVE_BOUND


@pytest.mark.parametrize("rows", [8, 16, 32])
def test_decode_glue_fold_path_is_bit_exact(builds, mesh, rows):  # noqa: F811
    groups = _TP4_PERMUTED
    rank, inputs, shared = _inputs(mesh, groups, rows)
    common = dict(decode_combine=True, pallas_restore=True, shared_in_op=True,
                  shared_out_scale=2.5, shared_rs_min_rows=rows + 1,
                  shared_small_ar_max_rows=rows, shared_fold_restore=2)
    outs = {}
    for tile in (0, 32):
        seen = builds(tile)
        outs[tile] = _by_rank(_layer(mesh, rank, inputs, groups, shared, **common), rows)
        assert seen and set(seen) == {32 if tile else 128}, (tile, rows, seen)
    for r in outs[0]:
        assert np.isfinite(outs[32][r]).all()
        np.testing.assert_array_equal(outs[32][r], outs[0][r])
