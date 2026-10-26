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
"""shared_fold_restore=2 (MOE_FUSED_EP_V2_FOLD_ANY_DECODE): the fold at a
decode bucket where the Pallas restore is not in effect (TP4, 32 rows,
t_local 8), with the small all-reduce's default and chip-pairs forms
(TPU_SMALL_ALL_REDUCE_PAIRS). Held, like the t_local <= 4 fold, to the
unfolded path's own error vs an exact f32 sum, with identical replicas."""

import numpy as np
import pytest

from vllm_torchtpu import envs

from .test_fused_ep_moe_v2_shared_in_op import (  # noqa: F401
    _by_rank,
    _exact,
    _inputs,
    _layer,
    mesh,
)

pytestmark = pytest.mark.multichip

# Mesh-index pairs (0,1)/(4,5) and (2,3)/(6,7): group_form takes the pairs form.
_TP4_PAIRS = ((0, 1, 4, 5), (2, 3, 6, 7))
_DECODE_GLUE = dict(decode_combine=True, pallas_restore=True)


@pytest.mark.parametrize("pairs", [0, 1, 2])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_fold_any_decode_tp4_r32(mesh, monkeypatch, pairs, scale):
    rows = 32
    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE_PAIRS", pairs, raising=False)
    groups = _TP4_PAIRS
    rank, inputs, shared = _inputs(mesh, groups, rows)
    routed = _layer(mesh, rank, inputs, groups, **_DECODE_GLUE)
    exact = _by_rank(_exact(mesh, routed, shared, groups, 1.0 * scale), rows)
    common = dict(shared_in_op=True, shared_out_scale=scale,
                  shared_rs_min_rows=rows + 1, shared_small_ar_max_rows=rows,
                  **_DECODE_GLUE)
    base = _by_rank(_layer(mesh, rank, inputs, groups, shared, **common), rows)
    old = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_fold_restore=1, **common), rows
    )
    fold = _by_rank(
        _layer(mesh, rank, inputs, groups, shared, shared_fold_restore=2, **common), rows
    )
    # fold=1 does not fire at t_local 8 (no Pallas restore): bits of the base.
    assert all(np.array_equal(old[r], base[r]) for r in base)
    same = all(np.array_equal(fold[r], base[r]) for r in base)
    for r in base:
        assert np.isfinite(fold[r]).all()
        norm = max(np.linalg.norm(exact[r]), 1e-20)
        err_fold = float(np.linalg.norm(fold[r] - exact[r]) / norm)
        err_base = float(np.linalg.norm(base[r] - exact[r]) / norm)
        print(f"pairs={pairs} rank {r}: rel-L2 vs exact fold {err_fold:.2e} "
              f"base {err_base:.2e}")
        assert err_fold < max(2.5 * err_base, 2.0**-9), (r, err_fold, err_base)
    assert not same, "fold path did not run (bits identical to the unfolded path)"
    for g in groups:
        for m in g[1:]:
            np.testing.assert_array_equal(fold[m], fold[g[0]])
            np.testing.assert_array_equal(base[m], base[g[0]])
    zero_routed = _by_rank(_exact(mesh, routed * 0, shared, groups, scale), rows)
    assert any(
        not np.allclose(zero_routed[r], exact[r], rtol=1e-2, atol=1e-2) for r in exact
    )
