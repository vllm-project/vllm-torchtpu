# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import jax
import pytest
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import pallas

from vllm_torchtpu.distributed import pallas_shapes


@pytest.mark.parametrize(
    "spec,expected",
    [
        (P(), (3, 8, 16)),
        (P(None, "tp"), (3, 16, 16)),
        (P(None, ("tp", "pcp")), (3, 64, 16)),
        (P("pcp", "tp"), (12, 16, 16)),
    ],
)
def test_native_placeholder_local_global_roundtrip(monkeypatch, spec, expected):
    # The abstract Pallas devices do not require a TPU runtime.
    devices = [
        pallas._PallasDevice(id=i, process_index=i, device_kind="TPU v7")
        for i in range(8)
    ]
    import numpy as np

    mesh = jax.sharding.Mesh(np.array(devices).reshape(4, 2), ("pcp", "tp"))
    monkeypatch.setattr(pallas, "get_global_shape", pallas_shapes.get_global_shape)
    monkeypatch.setattr(pallas, "get_local_shape", pallas_shapes.get_local_shape)
    value = pallas.jax_placeholder(torch.zeros(3, 8, 16), mesh, spec)
    assert value.shape == expected
    # Inspect native output allocation without creating an actual TPU tensor.
    allocation = {}

    def empty(shape, **kwargs):
        allocation["shape"] = tuple(shape)
        return allocation

    monkeypatch.setattr(torch, "empty", empty)
    pallas.torch_placeholder(
        jax.core.ShapedArray(expected, value.dtype), mesh=mesh, partition_spec=spec
    )
    assert allocation["shape"] == (3, 8, 16)


def test_compound_axis_requires_divisible_output():
    with pytest.raises(ValueError, match="not divisible"):
        pallas_shapes.get_local_shape(
            (63,), SimpleNamespace(shape={"pcp": 4, "tp": 2}), P(("tp", "pcp"))
        )
