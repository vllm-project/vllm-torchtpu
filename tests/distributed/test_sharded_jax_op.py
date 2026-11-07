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

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

# More than one character, so a spec torch_tpu read as a sequence of axis
# names would fail on the mesh lookup rather than pass by accident.
_AXIS = "sp"


def _one(x: jax.Array) -> jax.Array:
    return x


def _two(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return x, x


@pytest.mark.parametrize(
    "fn, out_specs, expected",
    [
        (_one, P(_AXIS, None), (P(_AXIS, None),)),
        (_two, (P(_AXIS, None), P()), (P(_AXIS, None), P())),
    ],
)
def test_gives_torch_tpu_one_output_spec_per_result(fn, out_specs, expected):
    mesh = Mesh(np.asarray(jax.devices()[:1]), (_AXIS,))
    op = sharded_jax_op(
        f"test_sharded_jax_op::specs_{fn.__name__.strip('_')}",
        fn,
        mesh=mesh,
        input_partition_specs=(P(_AXIS, None),),
        output_partition_specs=out_specs,
    )

    specs = op._init_fn.output_partition_specs
    assert specs == expected
    # As torch_tpu reads them: output i is sized from spec i.
    normalized = pallas_impl._normalize_output_partition_specs(specs, len(expected))
    assert list(normalized) == list(expected)


def _allocated_output_shapes(op, mesh, *args):
    """The output shapes ``JaxCallable.__call__`` allocates for ``args``.

    Reproduces how it builds its output placeholders, which unlike running the
    op needs no TPU.
    """
    callable_ = op._init_fn
    jax_args = pallas_impl.jax_placeholders(
        args, mesh=mesh, partition_specs=callable_.input_partition_specs
    )
    with jax._src.config.export_ignore_forward_compatibility(True):
        lowered = callable_.exported(*jax_args)
    n = len(lowered.out_avals)
    specs = (
        pallas_impl._normalize_output_partition_specs(
            callable_.output_partition_specs, n
        )
        or [None] * n
    )
    return [
        tuple(
            pallas_impl.torch_placeholder(
                aval, mesh=mesh, partition_spec=spec, device="meta"
            ).shape
        )
        for aval, spec in zip(lowered.out_avals, specs, strict=True)
    ]


@pytest.mark.multichip
def test_sharded_outputs_are_sized_for_this_ranks_shard(monkeypatch):
    if len(jax.devices()) != 8:
        pytest.skip("requires eight devices; CPU virtual devices are sufficient")
    mesh = Mesh(np.asarray(jax.devices()), (_AXIS,))

    def double_and_gather(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            lambda part: (part * 2, jax.lax.all_gather(part, _AXIS, tiled=True)),
            mesh=mesh,
            in_specs=P(_AXIS, None),
            out_specs=(P(_AXIS, None), P()),
            check_vma=False,
        )(x)

    op = sharded_jax_op(
        "test_sharded_jax_op::double_and_gather",
        double_and_gather,
        mesh=mesh,
        input_partition_specs=(P(_AXIS, None),),
        output_partition_specs=(P(_AXIS, None), P()),
    )
    # This rank's shard of a (16, 4) array split eight ways.
    local = torch.empty((2, 4))
    # The doubled shard stays sharded; the gather is the whole array.
    expected = [(2, 4), (16, 4)]

    assert _allocated_output_shapes(op, mesh, local) == expected

    # The default fake sizes its outputs the same way. It allocates on "tpu",
    # which a CPU-only host does not have.
    real_placeholder = pallas_impl.torch_placeholder
    monkeypatch.setattr(
        pallas_impl,
        "torch_placeholder",
        lambda *args, **kwargs: real_placeholder(*args, **kwargs, device="meta"),
    )
    assert [tuple(t.shape) for t in op._abstract_fn(local)] == expected
