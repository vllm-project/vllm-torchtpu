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
"""Which MoE calls take the split-SparseCore preprocessing, at 8 and 16 devices.

`split_sc_plan` is the decision `fused_ep_moe_v2` makes before tracing. The
DSV4.1 decode step must take the split path at 16 devices with XLA's scatter
(the SC scatter does not fit there and was not measured where it would), and
larger prefill buckets inside a decode server must keep the native path.
"""

import pytest

from vllm_torchtpu.kernels.fused_moe.v2.preprocessing import (
    XLA_GATHER_MAX_LOCAL_TOKENS,
    PreprocessTopology,
    split_sc_plan,
    xla_gather_topology,
)

HIDDEN = 5120
EIGHT = PreprocessTopology((1, 2, 4), (0, 1, 2, 4))
SIXTEEN = xla_gather_topology()


def plan(topology, *, t_local=288, fused=True, fp4=True, fits=True):
    calls = []

    def scatter_fits():
        calls.append(1)
        return fits

    result = split_sc_plan(
        topology,
        fused=fused,
        fp4=fp4,
        t_local=t_local,
        hidden=HIDDEN,
        scatter_fits=scatter_fits,
    )
    return result, len(calls)


@pytest.mark.cpu_test
def test_sixteen_device_decode_takes_split_path_without_sc_scatter():
    (topology, sc_scatter, split), _ = plan(SIXTEEN, t_local=288, fits=False)
    assert topology is SIXTEEN
    assert split is True
    assert sc_scatter is False


@pytest.mark.cpu_test
def test_sixteen_device_keeps_xla_scatter_even_when_the_sc_scatter_fits():
    (_, sc_scatter, split), calls = plan(SIXTEEN, t_local=64, fits=True)
    assert split is True
    assert sc_scatter is False
    assert calls == 0


@pytest.mark.cpu_test
@pytest.mark.parametrize("t_local", [XLA_GATHER_MAX_LOCAL_TOKENS + 1, 1024, 2048])
def test_sixteen_device_prefill_buckets_keep_native(t_local):
    (topology, sc_scatter, split), calls = plan(SIXTEEN, t_local=t_local)
    assert (topology, sc_scatter, split) == (None, False, False)
    assert calls == 0


@pytest.mark.cpu_test
def test_sixteen_device_bound_is_inclusive():
    (_, _, split), _ = plan(SIXTEEN, t_local=XLA_GATHER_MAX_LOCAL_TOKENS)
    assert split is True


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs",
    [{"fused": False}, {"fp4": False}],
    ids=["route_or_compute_stage", "not_fp4"],
)
@pytest.mark.parametrize("topology", [EIGHT, SIXTEEN], ids=["eight", "sixteen"])
def test_other_stages_and_formats_keep_native(topology, kwargs):
    (result_topology, sc_scatter, split), _ = plan(topology, **kwargs)
    assert (result_topology, sc_scatter, split) == (None, False, False)


@pytest.mark.cpu_test
def test_no_topology_keeps_native():
    assert plan(None)[0] == (None, False, False)


@pytest.mark.cpu_test
def test_eight_device_kernel_still_requires_the_sc_scatter():
    (topology, sc_scatter, split), _ = plan(EIGHT, t_local=288, fits=True)
    assert topology is EIGHT
    assert (sc_scatter, split) == (True, True)
    (topology, sc_scatter, split), _ = plan(EIGHT, t_local=288, fits=False)
    assert (topology, sc_scatter, split) == (None, False, False)


@pytest.mark.cpu_test
def test_eight_device_kernel_scratch_bound_is_unchanged():
    # supports_activation_gather: 4 * local bytes within 32 MiB, 8 KiB aligned.
    (topology, _, split), calls = plan(EIGHT, t_local=2048 * 2)
    assert (topology, split) == (None, False)
    assert calls == 0
