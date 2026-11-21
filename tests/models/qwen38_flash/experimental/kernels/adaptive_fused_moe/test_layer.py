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
"""The model's adaptive fused MoE layer copy on the eight-way slice.

Its one change is the optional leading-axis all-gather of the FP8 activations,
which must be bit-exact against the token-axis gather and the replicated
routing plan. Reuses the shared kernel test's fixtures and helpers.
"""

import numpy as np
import pytest

from tests.kernels.experimental.adaptive_fused_moe.test_kernel import (
    CAPACITY,
    SMALL,
    _make_inputs,
    _make_weights,
    _mesh,
    _require_eight_devices,  # noqa: F401  (module fixture: skips off the slice)
)
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import WeightFormat
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.adaptive_fused_moe.layer import (  # noqa: E501
    adaptive_fused_moe,
)

pytestmark = pytest.mark.multichip


@pytest.mark.parametrize("ag_lead_axis", [False, True])
def test_leading_axis_gather_matches_the_replicated_plan(ag_lead_axis):
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        20,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        21,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    from vllm_torchtpu.kernels.fused_moe.v2 import fused_ep_moe_v2

    replicated = fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, sharded_plan=False, **common
    )
    sharded = adaptive_fused_moe(
        x, w1, w2, w1_scale, w2_scale, gating, ag_lead_axis=ag_lead_axis, **common
    )
    np.testing.assert_array_equal(np.asarray(sharded), np.asarray(replicated))
