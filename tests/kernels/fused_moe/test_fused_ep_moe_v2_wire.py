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
"""The fused EP MoE result-wire override: a bf16 wire for fp8 weights."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host

GLM = dict(g_local=16, capacity=128, hidden=6144, inter=2048)


class _FakeChip:
    num_lanes = 128
    generation = host.MIN_GENERATION
    vmem_capacity_bytes = 128 * 2**20

    def get_sublane_tiling(self, dtype):
        return 32 // jnp.dtype(dtype).itemsize


def test_default_wire_follows_the_format():
    assert host.weight_form(host.WeightFormat.FP8).wire_dtype is host.FP8
    assert host.weight_form(host.WeightFormat.BF16).wire_dtype is host.BF16


@pytest.mark.parametrize("fmt", list(host.WeightFormat))
def test_bf16_override_changes_only_the_wire(fmt):
    base = host.weight_form(fmt)
    form = host.weight_form(fmt, jnp.bfloat16)
    assert form.wire_dtype is host.BF16
    assert form._replace(wire_dtype=base.wire_dtype) == base


@pytest.mark.parametrize("wire", [jnp.float8_e4m3fn, jnp.float32])
def test_other_wires_are_refused(wire):
    with pytest.raises(NotImplementedError, match="bfloat16"):
        host.weight_form(host.WeightFormat.FP8, wire)


def _arrays(**kw):
    scratch = host.vmem_scratch_arrays(**GLM, nbuf=1, **kw)
    body = host.vmem_tile_body_arrays(
        GLM["capacity"], GLM["hidden"], GLM["inter"], **kw
    )
    return {name: (shape, jnp.dtype(dtype)) for name, shape, dtype in scratch + body}


def test_bf16_wire_stages_bf16_rows_and_drops_the_quantized_copy():
    fp8, bf16 = _arrays(), _arrays(wire_dtype=jnp.bfloat16)
    assert fp8["out_vm"][1] == jnp.float8_e4m3fn
    assert bf16["out_vm"] == (fp8["out_vm"][0], jnp.dtype(jnp.bfloat16))
    assert "wire_rows" in fp8 and "wire_rows" not in bf16
    # The activations stay fp8: only the result wire widens.
    assert bf16["lhs_vm"] == fp8["lhs_vm"]


def test_bf16_wire_is_counted_in_the_weight_slot_fit():
    chip = _FakeChip()
    fp8 = host.fit_weight_slots(**GLM, info=chip)
    bf16 = host.fit_weight_slots(**GLM, info=chip, wire_dtype=jnp.bfloat16)
    a, b = _arrays(), _arrays(wire_dtype=jnp.bfloat16)

    def cost(arrays, name):
        return host.array_vmem_bytes(*arrays[name], chip)

    delta = cost(b, "out_vm") - cost(a, "out_vm") - cost(a, "wire_rows")
    assert (fp8.nbuf, fp8.distance) == (bf16.nbuf, bf16.distance)
    assert bf16.vmem_bytes - fp8.vmem_bytes == delta
