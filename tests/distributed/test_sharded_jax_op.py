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
"""Unit tests for sharded_jax_op integration with torch_tpu JaxCallable."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, PartitionSpec
from torch._subclasses.fake_tensor import FakeTensorMode
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed.sharded_jax_op import (
    _flat_output_partition_specs,
    sharded_jax_op,
)

_SUBPROC_ENV = "_SHARDED_JAX_OP_SUBPROC"


def _rerun_on_two_cpu_devices(test_name: str) -> bool:
    """Re-runs `test_name` in a 2-device CPU subprocess when only 1 device exists.

    Returns True if the caller should return (the subprocess ran the test).
    """
    if len(jax.devices()) >= 2:
        return False
    assert os.environ.get(_SUBPROC_ENV) != "1", (
        "Expected at least 2 JAX devices under --xla_force_host_platform_device_count=2"
    )
    env = os.environ.copy()
    xla_flags = env.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in xla_flags:
        env["XLA_FLAGS"] = (
            f"{xla_flags} --xla_force_host_platform_device_count=2".strip()
        )
    env[_SUBPROC_ENV] = "1"
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"{__file__}::{test_name}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, (
        f"2-device CPU subprocess failed (rc={res.returncode}):\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
    return True


def _two_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:2])
    assert len(devices) == 2
    return Mesh(devices, ("tp",))


def test_flat_output_partition_specs() -> None:
    """Output specs are flattened to one entry per output leaf, None kept."""
    p0, p1, p2 = PartitionSpec("tp", None), PartitionSpec(None, "tp"), PartitionSpec()
    assert _flat_output_partition_specs(p0) == [p0]
    assert _flat_output_partition_specs((p0, p1)) == [p0, p1]
    assert _flat_output_partition_specs((p0, (p1, p2))) == [p0, p1, p2]
    assert _flat_output_partition_specs((p0, None, p1)) == [p0, None, p1]


def test_sharded_jax_op_stock_call_sizes_local_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stock JaxCallable.__call__ sizes each output to this rank's shard.

    Runs the installed `JaxCallable.__call__` up to its output placeholders on a
    2-device CPU mesh, for a bare spec (which stock cannot read without
    `_flat_output_partition_specs`) and a tuple of per-output specs.
    """
    if _rerun_on_two_cpu_devices("test_sharded_jax_op_stock_call_sizes_local_shards"):
        return
    mesh = _two_device_mesh()

    # CPU-only host: register no TPU kernel and allocate placeholders on meta.
    real_placeholder = pallas_impl.torch_placeholder
    monkeypatch.setattr(
        pallas_impl,
        "torch_placeholder",
        lambda array, mesh=None, partition_spec=None, device="tpu": real_placeholder(
            array, mesh, partition_spec, device="meta"
        ),
    )
    monkeypatch.setattr(
        pallas_impl.tpu_torch_pallas, "register_custom_kernel", lambda *a, **k: None
    )
    monkeypatch.setattr(
        pallas_impl.tpu_torch_pallas, "lookup_custom_kernel", lambda *a, **k: False
    )

    def single(x: jax.Array) -> jax.Array:
        return x * 2

    def multi(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return x, x[:, :4], x.T

    cases = (
        ("single", single, PartitionSpec("tp", None), [(2, 8)]),
        (
            "multi",
            multi,
            (PartitionSpec("tp", None), PartitionSpec(None, "tp"), PartitionSpec()),
            # Global (4, 8), (4, 4), (8, 4): only the sharded dims halve.
            [(2, 8), (4, 2), (8, 4)],
        ),
    )
    for tag, fn, out_specs, expected in cases:
        op = sharded_jax_op(
            f"test_sharded_jax_op::stock_{tag}",
            fn,
            mesh=mesh,
            input_partition_specs=(PartitionSpec(None, None),),
            output_partition_specs=out_specs,
        )
        wrapped = op._init_fn
        assert type(wrapped) is pallas_impl.JaxCallable
        # Dispatch needs a TPU; the placeholders are built and cached before it.
        with pytest.raises(Exception):  # noqa: B017
            wrapped(torch.zeros((4, 8), dtype=torch.float32))
        ((output_shapes, _),) = wrapped.output_shapes.values()
        assert [tuple(t.shape) for t in output_shapes] == expected, tag


def test_sharded_jax_op_fake_fn_with_static_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fake_fn returns local per-rank shard shapes when called with static kwargs."""
    if _rerun_on_two_cpu_devices("test_sharded_jax_op_fake_fn_with_static_kwargs"):
        return
    mesh = _two_device_mesh()

    # Avoid allocating a real TPU tensor inside _placeholder on CPU-only hosts.
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.sharded_jax_op._placeholder",
        lambda aval, spec, m: torch.empty(
            tuple(pallas_impl.get_local_shape(aval.shape, m, spec))
            if spec is not None
            else tuple(aval.shape),
            dtype=pallas_impl.JAX_TO_TORCH_DTYPE_MAP[aval.dtype],
            device="meta",
        ),
    )

    def scale_fn(
        x: jax.Array,
        factor: float = 2.0,
        scales: Sequence[int] = (2, 1),
    ) -> jax.Array:
        if factor > 0.0:
            return (x * jnp.float32(factor * scales[0] + scales[1])).astype(x.dtype)
        return x

    # 1. Replicated input (4, 8) -> sharded output ("tp", None) across 2 ranks:
    #    global output is (4, 8), so per-rank local output must be (2, 8).
    op_shard_out = sharded_jax_op(
        "test_sharded_jax_op::scale_static_kw",
        scale_fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec(None, None),),
        output_partition_specs=PartitionSpec("tp", None),
    )

    # 2. Sharded input (2, 8) along dim 0 -> global (4, 8) -> re-sharded output
    #    (None, "tp") along dim 1: local output (4, 4) differs from both global
    #    (4, 8) and input shard (2, 8).
    op_reshard = sharded_jax_op(
        "test_sharded_jax_op::reshard_static_kw",
        scale_fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec("tp", None),),
        output_partition_specs=PartitionSpec(None, "tp"),
    )

    with FakeTensorMode() as fake_mode:
        x_rep = fake_mode.from_tensor(torch.zeros((4, 8), dtype=torch.float32))
        out = op_shard_out(x_rep, factor=3.0, scales=[3, 1])
        assert out.shape == (2, 8)
        assert out.dtype == torch.float32

        x_shard = fake_mode.from_tensor(torch.zeros((2, 8), dtype=torch.float32))
        out_reshard = op_reshard(x_shard, factor=3.0, scales=[3, 1])
        assert out_reshard.shape == (4, 4)
        assert out_reshard.dtype == torch.float32
