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
from typing import Any

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


class _StopAtDispatch(Exception):
    """Sentinel raised at `call_custom_kernel` on CPU-only test hosts."""


def _two_device_mesh() -> Mesh:
    devices = np.asarray(jax.devices()[:2])
    assert len(devices) == 2
    return Mesh(devices, ("tp",))


def _patch_cpu_pallas_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[tuple[int, ...]]]:
    """Allocates placeholders on `meta` and stops cleanly at kernel dispatch."""
    captured_shapes: list[list[tuple[int, ...]]] = []
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

    def _stop_at_dispatch(*args: Any, **kwargs: Any) -> Any:
        output_shapes = kwargs.get("output_shapes")
        if output_shapes is None and len(args) >= 4:
            output_shapes = args[3]
        assert output_shapes is not None
        captured_shapes.append([tuple(t.shape) for t in output_shapes])
        raise _StopAtDispatch

    monkeypatch.setattr(
        pallas_impl.tpu_torch_pallas, "call_custom_kernel", _stop_at_dispatch
    )
    return captured_shapes


def test_flat_output_partition_specs() -> None:
    """Output specs flatten to one entry per leaf, broadcasting bare specs."""
    p0, p1, p2 = PartitionSpec("tp", None), PartitionSpec(None, "tp"), PartitionSpec()
    assert _flat_output_partition_specs(p0) == [p0]
    assert _flat_output_partition_specs(p0, 3) == [p0, p0, p0]
    assert _flat_output_partition_specs(None, 2) == [None, None]
    assert _flat_output_partition_specs((p0, p1), 2) == [p0, p1]
    assert _flat_output_partition_specs((p0, (p1, p2)), 3) == [p0, p1, p2]
    assert _flat_output_partition_specs((p0, None, p1), 3) == [p0, None, p1]
    with pytest.raises(ValueError, match="has 2 leaves, but fn returned 3"):
        _flat_output_partition_specs((p0, p1), 3)


def _run_two_cpu_device_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _two_device_mesh()
    captured = _patch_cpu_pallas_dispatch(monkeypatch)

    def single(x: jax.Array) -> jax.Array:
        return x * 2

    def multi(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return x, x[:, :4], x.T

    cases: tuple[tuple[str, Any, Any, list[tuple[int, ...]]], ...] = (
        ("single", single, PartitionSpec("tp", None), [(2, 8)]),
        (
            "multi_tuple",
            multi,
            (PartitionSpec("tp", None), PartitionSpec(None, "tp"), PartitionSpec()),
            [(2, 8), (4, 2), (8, 4)],
        ),
        (
            "multi_bare_broadcast",
            multi,
            PartitionSpec("tp", None),
            [(2, 8), (2, 4), (4, 4)],
        ),
    )
    for tag, fn, out_specs, expected in cases:
        captured.clear()
        op = sharded_jax_op(
            f"test_sharded_jax_op::stock_{tag}",
            fn,
            mesh=mesh,
            input_partition_specs=(PartitionSpec(None, None),),
            output_partition_specs=out_specs,
        )
        wrapped = op._init_fn
        assert type(wrapped) is pallas_impl.JaxCallable
        with pytest.raises(_StopAtDispatch):
            wrapped(torch.zeros((4, 8), dtype=torch.float32))
        assert captured == [expected], tag
        ((output_shapes, _),) = wrapped.output_shapes.values()
        assert [tuple(t.shape) for t in output_shapes] == expected, tag

        with FakeTensorMode() as fake_mode:
            fake_in = fake_mode.from_tensor(torch.zeros((4, 8), dtype=torch.float32))
            fake_out = op(fake_in)
            flat_fake = jax.tree.leaves(fake_out)
            assert [tuple(t.shape) for t in flat_fake] == expected, tag

    # Leaf count mismatch raises ValueError cleanly in both eager and FakeTensorMode.
    op_mismatch = sharded_jax_op(
        "test_sharded_jax_op::mismatch",
        multi,
        mesh=mesh,
        input_partition_specs=(PartitionSpec(None, None),),
        output_partition_specs=(PartitionSpec("tp", None), PartitionSpec(None, "tp")),
    )
    with pytest.raises(ValueError, match="out_shardings|leaves"):
        op_mismatch._init_fn(torch.zeros((4, 8), dtype=torch.float32))
    with (
        FakeTensorMode() as fake_mode,
        pytest.raises(ValueError, match="out_shardings|leaves"),
    ):
        op_mismatch(fake_mode.from_tensor(torch.zeros((4, 8), dtype=torch.float32)))

    # Static kwargs with non-hashable Python list values work under FakeTensorMode.
    def scale_fn(
        x: jax.Array,
        factor: float = 2.0,
        scales: Sequence[int] = (2, 1),
    ) -> jax.Array:
        if factor > 0.0:
            return (x * jnp.float32(factor * scales[0] + scales[1])).astype(x.dtype)
        return x

    op_shard_out = sharded_jax_op(
        "test_sharded_jax_op::scale_static_kw",
        scale_fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec(None, None),),
        output_partition_specs=PartitionSpec("tp", None),
    )
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


def test_sharded_jax_op_two_cpu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs all 2-CPU-device shard-sizing checks in a single subprocess."""
    if len(jax.devices()) >= 2:
        _run_two_cpu_device_checks(monkeypatch)
        return
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
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{__file__}::test_sharded_jax_op_two_cpu_devices",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert res.returncode == 0, (
        f"2-device CPU subprocess failed (rc={res.returncode}):\n"
        f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    )
