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
"""The TP all-reduce's carried operands: how the kernel takes them out of its
refs, and that the call lowers with them pinned to HBM (XLA otherwise moves
them into VMEM ahead of the call). Lowering only; the copies' bytes are
checked by pallas_all_reduce's arming self-test on the device."""

import json
import re

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel

N = 8
_TPU7X = jax.sharding.AbstractMesh(
    (N,),
    ("x",),
    axis_types=(jax.sharding.AxisType.Auto,),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    ),
)


def test_carrying_hands_the_body_its_refs_and_one_copy_per_operand(monkeypatch):
    monkeypatch.setattr(
        kernel.pltpu, "make_async_copy", lambda src, dst, sem: (src, dst, sem)
    )

    class Sems:
        def __init__(self):
            self.at = {i: f"sem{i}" for i in range(2)}

    seen = []
    sems = Sems()
    refs = ("rank", "x", "src0", "src1", "out", "dst0", "dst1", "s0", "s1", sems)
    kernel._carrying(lambda *a: seen.append(a), 2, 2)(*refs)
    *plain, copies = seen[0]
    assert plain == ["rank", "x", "out", "s0", "s1"]
    assert copies == [("src0", "dst0", "sem0"), ("src1", "dst1", "sem1")]

    seen.clear()
    kernel._carrying(lambda *a: seen.append(a), 2, 0)("rank", "x", "out", "s0")
    assert seen == [("rank", "x", "out", "s0", [])]


@pytest.mark.parametrize("rows", [8, 24])
def test_carried_operands_lower_pinned_to_hbm(rows):
    def body(x, rank, norm, weight):
        total, copies = kernel.all_reduce_carrying(x, rank, N, (norm, weight))
        return total, *copies

    fn = jax.shard_map(
        body,
        mesh=_TPU7X,
        in_specs=(P("x"),) * 4,
        out_specs=(P("x"),) * 3,
        check_vma=False,
    )
    args = (
        jax.ShapeDtypeStruct((N * rows, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((N,), jnp.int32),
        jax.ShapeDtypeStruct((N * 6144,), jnp.float32),
        jax.ShapeDtypeStruct((N * 6144, 1536), jnp.float8_e4m3fn),
    )
    with jax.sharding.use_abstract_mesh(_TPU7X):
        text = jax.export.export(jax.jit(fn), platforms=["tpu"])(*args).mlir_module()
    configs = re.findall(r'backend_config = "(\{.*?\})", kernel_name', text)
    assert len(configs) == 1
    config = json.loads(configs[0].replace('\\"', '"').replace("\\22", '"'))
    colors = config["custom_call_config"]["input_memory_space_colors"]
    # operands: rank (scalar prefetch), x, then the two carried ones; color 0
    # is HBM
    assert sorted((c["operand_index"], c["color"]) for c in colors) == [(2, 0), (3, 0)]
