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
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

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
    vars(exec_mod)["EagerMode"] = _PrivateOnlyEagerMode
    vars(exec_mod)["set_eager_mode"] = fake_set_eager_mode

    compile_pkg = types.ModuleType("torch_tpu._internal.compile")
    compile_mod = types.ModuleType("torch_tpu._internal.compile.tpu_torch_compile")
    vars(compile_mod)["placeholder_like"] = fake_placeholder_like
    vars(compile_mod)["async_traverse_and_compile"] = fake_async_compile
    vars(compile_pkg)["tpu_torch_compile"] = compile_mod

    opts_mod = types.ModuleType("torch_tpu._internal.compiler_options")
    vars(opts_mod)["custom_compiler_options"] = fake_compiler_options

    internal_pkg = types.ModuleType("torch_tpu._internal")
    vars(internal_pkg)["execution_mode"] = exec_mod
    vars(internal_pkg)["compile"] = compile_pkg
    vars(internal_pkg)["compiler_options"] = opts_mod

    root_pkg = types.ModuleType("torch_tpu")
    vars(root_pkg)["_internal"] = internal_pkg

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

    expected_mode = _PrivateOnlyEagerMode.__members__["_INTERNAL_COMPILE_FX_GRAPH"]
    assert recorded_modes == [expected_mode]
    assert recorded_options == [{"xla_flag": "1"}]


@dataclass
class _FakeLowered:
    """Minimal lowered export payload for _ExportCache round-trip testing."""

    out_avals: tuple[jax.core.ShapedArray, ...]
    out_tree: object
    mlir_module_serialized: bytes


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
        return _FakeLowered(
            out_avals=(aval,),
            out_tree=out_tree,
            mlir_module_serialized=b"MLIRBC_TEST",
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
    import vllm.envs as vllm_envs
    monkeypatch.setattr(vllm_envs, "VLLM_XLA_CACHE_PATH", str(tmp_path))
    orig_is_set = vllm_envs.is_set
    monkeypatch.setattr(vllm_envs, "is_set", lambda name: True if name == "VLLM_XLA_CACHE_PATH" else orig_is_set(name))
    registered_kernels: list[tuple[str, str, bytes]] = []

    def fake_sharded_jax_op(
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
        callable_op = _FakeJaxCallable(name, mesh, raw_in_specs)
        if callable(configure_callable):
            configure_callable(callable_op)
        return types.SimpleNamespace(_init_fn=callable_op)

    monkeypatch.setattr(bridge, "_SHARDED_JAX_OP_ACCEPTS_CONFIGURE", True)
    monkeypatch.setattr(bridge, "sharded_jax_op", fake_sharded_jax_op)
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
        export_cache,
        "_placeholder",
        lambda aval, spec, _mesh: (aval.shape, str(aval.dtype), spec),
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
    callable_op = vars(op_wrapper)["_init_fn"]
    assert callable_op.output_partition_specs == [spec]

    sample = jax.numpy.zeros((4,), dtype=jax.numpy.float32)
    callable_op(sample)
    assert callable_op.export_calls == 1
    assert list((tmp_path / "k3_export").glob("module-fp123.mlirbc"))

    reloaded_op = _FakeJaxCallable("kimi_k3::shim_unit_op", mesh, (spec,))
    vars(reloaded_op)["output_partition_specs"] = [spec]
    hook_fn = export_cache.hook("unit_v1")
    assert hook_fn is not None
    hook_fn(reloaded_op)
    status, cached_entry = reloaded_op(sample)
    assert status == "ok"
    assert reloaded_op.export_calls == 0
    assert registered_kernels == [("kimi_k3::shim_unit_op", "fp123", b"MLIRBC_TEST")]
    loaded_outputs, _ = cached_entry
    assert loaded_outputs == [((4,), "float32", spec)]
