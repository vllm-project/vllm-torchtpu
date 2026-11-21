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
"""CPU unit tests for Kimi-K3 runtime_shim compilation, bridge, and export cache."""

from __future__ import annotations

import contextlib
import enum
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.experimental.topologies as jax_topologies
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from vllm_torchtpu.distributed import pallas_shapes
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    bridge,
    compilation,
    export_cache,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import pytest


class _PrivateOnlyEagerMode(enum.Enum):
    """Stand-in matching torch_tpu 0.2.0 EagerMode attribute names."""

    _INTERNAL_COMPILE_FX_GRAPH = "private_fx_graph"


@dataclass
class _ReadyHandle:
    """Completed async compile handle double."""

    ready: bool = True

    def is_ready(self) -> bool:
        """Report handle readiness."""
        return self.ready

    def get(self) -> None:
        """Consume completed compilation."""


def test_compile_ops_falls_back_to_private_fx_graph_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve _INTERNAL_COMPILE_FX_GRAPH when the public enum name is absent."""
    recorded_modes: list[object] = []
    recorded_options: list[dict[str, str]] = []
    active_options: dict[str, str] = {}

    @contextlib.contextmanager
    def fake_set_eager_mode(mode: object) -> Iterator[None]:
        """Record active eager mode during placeholder tracing."""
        recorded_modes.append(mode)
        yield

    @contextlib.contextmanager
    def fake_compiler_options(opts: dict[str, str]) -> Iterator[None]:
        """Capture custom compiler options around async compile."""
        active_options.clear()
        active_options.update(opts)
        yield

    def fake_placeholder_like(value: int) -> str:
        """Build synthetic placeholder value."""
        return f"ph:{value}"

    def fake_async_compile(
        outputs: list[str],
        inputs: list[str],
        *,
        fast_compile: bool,
    ) -> _ReadyHandle:
        """Verify placeholder wiring and record active compiler options."""
        assert fast_compile is True
        assert inputs == ["ph:7"]
        assert outputs == ["out:ph:7"]
        recorded_options.append(dict(active_options))
        return _ReadyHandle()

    exec_mod = types.ModuleType("torch_tpu._internal.execution_mode")
    exec_mod.EagerMode = _PrivateOnlyEagerMode
    exec_mod.set_eager_mode = fake_set_eager_mode

    compile_pkg = types.ModuleType("torch_tpu._internal.compile")
    compile_mod = types.ModuleType("torch_tpu._internal.compile.tpu_torch_compile")
    compile_mod.placeholder_like = fake_placeholder_like
    compile_mod.async_traverse_and_compile = fake_async_compile
    compile_pkg.tpu_torch_compile = compile_mod

    opts_mod = types.ModuleType("torch_tpu._internal.compiler_options")
    opts_mod.custom_compiler_options = fake_compiler_options

    internal_pkg = types.ModuleType("torch_tpu._internal")
    internal_pkg.execution_mode = exec_mod
    internal_pkg.compile = compile_pkg
    internal_pkg.compiler_options = opts_mod

    root_pkg = types.ModuleType("torch_tpu")
    root_pkg._internal = internal_pkg

    for mod_name, mod_obj in (
        ("torch_tpu", root_pkg),
        ("torch_tpu._internal", internal_pkg),
        ("torch_tpu._internal.execution_mode", exec_mod),
        ("torch_tpu._internal.compile", compile_pkg),
        ("torch_tpu._internal.compile.tpu_torch_compile", compile_mod),
        ("torch_tpu._internal.compiler_options", opts_mod),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)

    def sample_op(ph_val: str) -> tuple[str]:
        """Return traced output tuple."""
        return (f"out:{ph_val}",)

    call = compilation.CompileCall(
        op=sample_op,
        operands=(7,),
        options={"xla_flag": "1"},
    )
    compilation.compile_ops([call])

    expected_mode = _PrivateOnlyEagerMode._INTERNAL_COMPILE_FX_GRAPH
    assert recorded_modes == [expected_mode]
    assert recorded_options == [{"xla_flag": "1"}]


@dataclass
class _FakeLowered:
    """Minimal lowered export payload for _ExportCache round-trip testing."""

    out_avals: tuple[jax.core.ShapedArray, ...]
    out_tree: object
    mlir_module_serialized: bytes
    _out_named_shardings: tuple[NamedSharding, ...] = ()


class _FakeJaxCallable:
    """Minimal JaxCallable double exercising export_cache.hook store and reload."""

    def __init__(
        self,
        name: str,
        mesh: Mesh,
        in_specs: tuple[PartitionSpec, ...],
    ) -> None:
        """Initialize callable metadata."""
        self.name = name
        self.mesh = mesh
        self.input_partition_specs = in_specs
        self.output_partition_specs: list[PartitionSpec] | None = None
        self.donate_argnums: tuple[int, ...] = ()
        self.static_argnums: tuple[int, ...] = ()
        self.trace_key = name
        self.output_shapes: dict[str, object] = {}
        self.kernel_key_to_mlir_fingerprint: dict[str, str] = {}
        self.export_calls = 0

    def exported(self, *args: jax.Array) -> _FakeLowered:
        """Record export invocation and return serialized payload."""
        self.export_calls += 1
        aval = jax.core.ShapedArray(args[0].shape, args[0].dtype)
        _, out_tree = jax.tree.flatten((aval,))
        spec = (
            self.output_partition_specs[0]
            if self.output_partition_specs
            else PartitionSpec()
        )
        return _FakeLowered(
            out_avals=(aval,),
            out_tree=out_tree,
            mlir_module_serialized=b"MLIRBC_TEST",
            _out_named_shardings=(NamedSharding(self.mesh, spec),),
        )

    def __call__(self, *args: jax.Array) -> tuple[str, object]:
        """Populate fingerprint on cold miss or return cached output shapes."""
        key = f"k:{args[0].shape}"
        if key not in self.output_shapes:
            self.exported(*args)
            self.kernel_key_to_mlir_fingerprint[key] = "fp123"
            self.output_shapes[key] = ("cold", None)
        return ("ok", self.output_shapes[key])


def test_register_op_forwards_specs_and_export_cache_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Attach flattened PartitionSpecs and reload persisted MLIR from k3_export."""
    monkeypatch.setenv("VLLM_XLA_CACHE_PATH", str(tmp_path))
    registered_kernels: list[tuple[str, str, bytes]] = []

    def fake_register_sharded_op(
        name: str,
        _fn: object,
        *,
        mesh: Mesh,
        **kwargs: object,
    ) -> object:
        """Construct wrapper carrying _init_fn like torch_tpu CustomOpDef."""
        raw_in_specs = kwargs["input_partition_specs"]
        assert isinstance(raw_in_specs, tuple)
        configure_callable = kwargs.get("configure_callable")
        raw_out_specs = kwargs.get("output_partition_specs", ())
        callable_op = _FakeJaxCallable(name, mesh, raw_in_specs)
        callable_op.output_partition_specs = list(raw_out_specs)
        if callable(configure_callable):
            configure_callable(callable_op)
        return types.SimpleNamespace(_init_fn=callable_op)

    monkeypatch.setattr(bridge, "_register_sharded_op", fake_register_sharded_op)
    monkeypatch.setattr(
        "vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.config._ACTIVE",
        None,
    )
    monkeypatch.setattr(
        export_cache.pallas_impl,
        "_get_kernel_invocation_key",
        lambda _trace_key, args, _kwargs, _static_argnums: f"k:{args[0].shape}",
    )
    monkeypatch.setattr(
        export_cache.pallas_impl.tpu_torch_pallas,
        "register_custom_kernel",
        lambda name, fp, serialized_mlir_module: registered_kernels.append(
            (name, fp, serialized_mlir_module)
        ),
    )
    monkeypatch.setattr(
        export_cache.pallas_impl,
        "torch_placeholder",
        lambda aval, *, mesh=None, partition_spec=None: (
            aval.shape,
            str(aval.dtype),
            partition_spec,
        ),
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    spec = PartitionSpec("tp")
    op_wrapper = bridge.register_op(
        "kimi_k3::shim_unit_op",
        mesh,
        lambda x: (x,),
        in_specs=(spec,),
        out_specs=(spec,),
        export_key="unit_v1",
    )
    callable_op = op_wrapper._init_fn
    assert callable_op.output_partition_specs == [spec]

    sample = jax.numpy.zeros((4,), dtype=jax.numpy.float32)
    callable_op(sample)
    assert callable_op.export_calls == 1
    assert list((tmp_path / "k3_export").glob("module-fp123.mlirbc"))

    reloaded_op = _FakeJaxCallable("kimi_k3::shim_unit_op", mesh, (spec,))
    reloaded_op.output_partition_specs = [spec]
    hook_fn = export_cache.hook("unit_v1")
    assert hook_fn is not None
    hook_fn(reloaded_op)
    status, cached_entry = reloaded_op(sample)
    assert status == "ok"
    assert reloaded_op.export_calls == 0
    assert registered_kernels == [("kimi_k3::shim_unit_op", "fp123", b"MLIRBC_TEST")]
    loaded_outputs, _ = cached_entry
    assert loaded_outputs == [((4,), "float32", spec)]


def test_register_sharded_op_with_sharded_and_replicated_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real bridge._register_sharded_op on a multi-device topology mesh."""
    topology = jax_topologies.get_topology_desc("tpu7x_2x2x4")
    devices = list(topology.devices)
    world_size = len(devices)
    mesh = Mesh(np.array(devices), ("tp",))
    in_spec = PartitionSpec("tp", None)
    out_specs = (PartitionSpec("tp", None), PartitionSpec())

    monkeypatch.setattr(
        bridge.pallas_impl, "get_global_shape", pallas_shapes.get_global_shape
    )
    monkeypatch.setattr(
        bridge.pallas_impl, "get_local_shape", pallas_shapes.get_local_shape
    )

    events: list[str] = []
    registered: dict[str, Any] = {}
    recorded_call_shapes: list[tuple[int, ...]] = []

    def fake_custom_op(name: str, fn: Any, *, mutates_args: tuple[Any, ...]) -> Any:
        """Record custom_op registration and capture registered fake_fn."""
        assert mutates_args == ()
        events.append("custom_op")
        registered["name"] = name
        registered["fn"] = fn

        def register_fake(fake_fn: Any) -> Any:
            registered["fake_fn"] = fake_fn
            return fake_fn

        return types.SimpleNamespace(_init_fn=fn, register_fake=register_fake)

    def stub_empty(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype | None = None,
        **_kwargs: object,
    ) -> torch.Tensor:
        """Allocate CPU placeholders instead of device buffers."""
        return torch.zeros(shape, dtype=dtype or torch.float32)

    def stub_call_custom_kernel(
        _name: str,
        _fingerprint: str,
        *,
        output_shapes: list[torch.Tensor],
        **_kwargs: object,
    ) -> list[torch.Tensor]:
        """Record per-rank placeholder shapes passed by JaxCallable.__call__."""
        recorded_call_shapes.extend(tuple(out.shape) for out in output_shapes)
        return output_shapes

    monkeypatch.setattr(torch.library, "custom_op", fake_custom_op)
    monkeypatch.setattr(torch, "empty", stub_empty)
    monkeypatch.setattr(
        bridge.pallas_impl.tpu_torch_pallas,
        "register_custom_kernel",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bridge.pallas_impl.tpu_torch_pallas,
        "call_custom_kernel",
        stub_call_custom_kernel,
    )

    def configure_hook(callable_obj: Any) -> None:
        """Verify hook runs before custom_op registration."""
        events.append("configure")
        assert "name" not in registered
        assert type(callable_obj) is bridge.pallas_impl.JaxCallable

    def body(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return one sharded output and one replicated output."""
        return x, x

    previous_ignore_fc = bool(jax.config.jax_export_ignore_forward_compatibility)
    jax.config.update("jax_export_ignore_forward_compatibility", val=True)
    try:
        op_wrapper = bridge._register_sharded_op(
            "kimi_k3::real_helper_unit_op",
            body,
            mesh=mesh,
            input_partition_specs=(in_spec,),
            output_partition_specs=out_specs,
            configure_callable=configure_hook,
        )
        assert events == ["configure", "custom_op"]
        callable_op = op_wrapper._init_fn
        assert type(callable_op) is bridge.pallas_impl.JaxCallable
        assert callable_op.output_partition_specs == out_specs

        local_rows = 4
        cols = 16
        sample = torch.zeros((local_rows, cols), dtype=torch.float32)

        jax_arg = bridge.pallas_impl.jax_placeholder(sample, mesh, in_spec)
        lowered = callable_op.exported(jax_arg)
        named_shardings = getattr(lowered, "_out_named_shardings", None)
        assert named_shardings is not None
        assert tuple(s.spec for s in named_shardings) == out_specs

        expected_local = [(local_rows, cols), (local_rows * world_size, cols)]
        fake_outputs = registered["fake_fn"](sample)
        assert [tuple(out.shape) for out in fake_outputs] == expected_local

        callable_op(sample)
        assert recorded_call_shapes == expected_local
    finally:
        jax.config.update(
            "jax_export_ignore_forward_compatibility", val=previous_ignore_fc
        )
        jax.clear_caches()
