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
"""CPU unit tests for sharded_jax_op configure_callable and partition specs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed import sharded_jax_op as sharded_jax_op_mod
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _cpu_mesh(*axes: str) -> Mesh:
    """Build a 1-device CPU mesh matching the requested axis names."""
    devices = np.asarray(jax.devices("cpu")[:1]).reshape((1,) * len(axes))
    return Mesh(devices, axes)


def _identity_single(x: jax.Array) -> jax.Array:
    """Return single sharded array unchanged."""
    return x


def _identity_pair(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return two arrays to verify multi-output PartitionSpec flattening."""
    return x, x


def test_sharded_jax_op_configure_callable_runs_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run configure_callable before custom_op and keep its changes on _init_fn."""
    mesh = _cpu_mesh("tp")
    spec = PartitionSpec("tp")
    events: list[tuple[str, object]] = []
    export_calls: list[int] = []
    real_custom_op = torch.library.custom_op

    def spy_custom_op(
        name: str,
        fn: Callable[..., object] | None = None,
        /,
        *,
        mutates_args: str | Iterable[str] = (),
    ) -> object:
        """Record registration ordering while delegating to real custom_op."""
        events.append(("register", fn))
        return real_custom_op(name, fn, mutates_args=mutates_args)

    monkeypatch.setattr(torch.library, "custom_op", spy_custom_op)
    configured_subclasses: list[type] = []

    def configure(callable_op: pallas_impl.JaxCallable) -> None:
        """Wrap exported and swap __class__ before custom_op registers it."""
        events.append(("configure", callable_op))
        base_exported = callable_op.exported

        def recording_exported(*args: object, **kwargs: object) -> object:
            """Count invocations routed through the hook-wrapped exported."""
            export_calls.append(1)
            return base_exported(*args, **kwargs)

        class _ConfiguredCallable(type(callable_op)):
            """Marker subclass verifying __class__ swaps survive registration."""

        callable_op.exported = recording_exported
        callable_op.__class__ = _ConfiguredCallable
        configured_subclasses.append(_ConfiguredCallable)

    op = sharded_jax_op(
        "test_sharded_jax_op::configured",
        _identity_single,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=spec,
        configure_callable=configure,
    )

    assert [kind for kind, _ in events] == ["configure", "register"]
    configured_op = vars(op)["_init_fn"]
    assert events[0][1] is configured_op
    assert events[1][1] is configured_op
    assert isinstance(configured_op, pallas_impl.JaxCallable)
    assert type(configured_op) is configured_subclasses[0]
    assert configured_op.output_partition_specs == [spec]

    sample = torch.empty((2, 4), dtype=torch.float32)
    jax_args = pallas_impl.jax_placeholders(
        (sample,), mesh=mesh, partition_specs=(spec,)
    )
    lowered = configured_op.exported(*jax_args)
    assert export_calls == [1]
    assert len(lowered.out_avals) == 1


def test_sharded_jax_op_leaves_callable_unchanged_without_hook(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Leave JaxCallable unmutated and emit no warnings when hook is absent."""
    mesh = _cpu_mesh("tp")
    spec = PartitionSpec("tp")
    test_logger = logging.getLogger("test_sharded_jax_op")
    monkeypatch.setattr(sharded_jax_op_mod, "_SHARDED_CALLABLE_CLS", None)
    monkeypatch.setattr(sharded_jax_op_mod, "logger", test_logger)

    with caplog.at_level(logging.DEBUG, logger="test_sharded_jax_op"):
        op_omitted = sharded_jax_op(
            "test_sharded_jax_op::plain_omitted",
            _identity_single,
            mesh=mesh,
            input_partition_specs=(spec,),
            output_partition_specs=spec,
        )
        op_none = sharded_jax_op(
            "test_sharded_jax_op::plain_none",
            _identity_single,
            mesh=mesh,
            input_partition_specs=(spec,),
            output_partition_specs=spec,
            configure_callable=None,
        )

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    fn_omitted = vars(op_omitted)["_init_fn"]
    fn_none = vars(op_none)["_init_fn"]
    assert isinstance(fn_omitted, pallas_impl.JaxCallable)
    assert type(fn_omitted) is type(fn_none)
    assert type(fn_omitted) is vars(sharded_jax_op_mod)["_SHARDED_CALLABLE_CLS"]
    if type(fn_omitted) is pallas_impl.JaxCallable:
        assert [r.levelno for r in caplog.records] == [logging.DEBUG]
    assert sorted(vars(fn_omitted)) == sorted(vars(fn_none))
    assert fn_omitted.output_partition_specs == [spec]
    assert fn_none.output_partition_specs == [spec]
    assert fn_omitted.exported.__code__ is fn_none.exported.__code__
    assert fn_omitted.exported.__code__.co_filename != __file__


@pytest.mark.parametrize(
    ("case_name", "axes", "body", "io_specs", "expected"),
    [
        (
            "single_sharded",
            ("tp",),
            _identity_single,
            (PartitionSpec("tp"), PartitionSpec("tp")),
            [PartitionSpec("tp")],
        ),
        (
            "single_replicated_empty",
            ("tp",),
            _identity_single,
            (PartitionSpec("tp"), PartitionSpec()),
            [PartitionSpec()],
        ),
        (
            "compound_axis",
            ("dp", "tp"),
            _identity_single,
            (PartitionSpec(("dp", "tp")), PartitionSpec(("dp", "tp"))),
            [PartitionSpec(("dp", "tp"))],
        ),
        (
            "multi_output_mixed",
            ("tp",),
            _identity_pair,
            (PartitionSpec("tp"), (PartitionSpec("tp"), PartitionSpec())),
            [PartitionSpec("tp"), PartitionSpec()],
        ),
    ],
)
def test_sharded_jax_op_flattens_output_partition_specs(
    case_name: str,
    axes: tuple[str, ...],
    body: Callable[..., object],
    io_specs: tuple[PartitionSpec, PartitionSpec | tuple[PartitionSpec, ...]],
    expected: list[PartitionSpec],
) -> None:
    """Flatten PartitionSpec trees into export leaf order across valid schemas."""
    in_spec, out_specs = io_specs
    op = sharded_jax_op(
        f"test_sharded_jax_op::{case_name}",
        body,
        mesh=_cpu_mesh(*axes),
        input_partition_specs=(in_spec,),
        output_partition_specs=out_specs,
    )
    callable_op = vars(op)["_init_fn"]
    assert isinstance(callable_op, pallas_impl.JaxCallable)
    assert callable_op.output_partition_specs == expected
