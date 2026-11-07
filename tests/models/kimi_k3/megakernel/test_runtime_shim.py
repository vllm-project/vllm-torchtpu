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
"""CPU unit tests for Kimi-K3 runtime_shim, vmem_pool, and collectives32."""

from __future__ import annotations

import builtins
import contextlib
import enum
import inspect
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import pallas as pl
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.megakernel import pool_alias as shared_pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel import common
from vllm_torchtpu.models.kimi_k3.megakernel.common import (
    collectives32,
    vmem_pool,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    bridge,
    compilation,
    export_cache,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    config as runtime_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

_TP32_WORLD_SIZE = 32
_TP32_PEER_COUNT = 31
_TEST_RANK = 13


class _MonkeyPatch(Protocol):
    """Subset of pytest.MonkeyPatch used by runtime_shim unit tests."""

    def setattr(
        self,
        target: object,
        name: str,
        value: object = ...,
        *,
        raising: bool = ...,
    ) -> None:
        """Patch attribute on target object or dotted import path."""

    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        """Set environment variable for the duration of the test."""

    def setitem(self, dic: Mapping[str, object], name: str, value: object) -> None:
        """Set dictionary entry for the duration of the test."""


class _PrivateOnlyEagerMode(enum.Enum):
    """Stand-in matching torch_tpu 0.2.0 EagerMode attribute names."""

    _INTERNAL_COMPILE_FX_GRAPH = "private_fx_graph"


@dataclass
class _ReadyHandle:
    """Completed async compile handle double."""

    ready: bool = True

    def is_ready(self) -> bool:
        """Report handle readiness.

        Returns:
            True when the compilation handle has completed.
        """
        return self.ready

    def get(self) -> None:
        """Consume completed compilation."""


def test_compile_ops_falls_back_to_private_fx_graph_mode(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Resolve _INTERNAL_COMPILE_FX_GRAPH when the public enum name is absent."""
    recorded_modes: list[object] = []
    recorded_options: list[dict[str, str]] = []
    active_options: dict[str, str] = {}

    @contextlib.contextmanager
    def fake_set_eager_mode(mode: object) -> Iterator[None]:
        """Record active eager mode during placeholder tracing.

        Yields:
            Control to the caller while the eager mode is active.
        """
        recorded_modes.append(mode)
        yield

    @contextlib.contextmanager
    def fake_compiler_options(opts: dict[str, str]) -> Iterator[None]:
        """Capture custom compiler options around async compile.

        Yields:
            Control to the caller while custom compiler options are active.
        """
        active_options.clear()
        active_options.update(opts)
        yield

    def fake_placeholder_like(value: int) -> str:
        """Build synthetic placeholder value.

        Returns:
            Formatted placeholder identifier string.
        """
        return f"ph:{value}"

    def fake_async_compile(
        outputs: list[str],
        inputs: list[str],
        *,
        fast_compile: bool,
    ) -> _ReadyHandle:
        """Verify placeholder wiring and record active compiler options.

        Returns:
            Completed compilation handle double.
        """
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
        """Return traced output tuple.

        Returns:
            Single-element tuple containing the tagged placeholder string.
        """
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
        self.output_shapes: dict[str, tuple[object, object]] = {}
        self.kernel_key_to_mlir_fingerprint: dict[str, str] = {}
        self.export_calls = 0

    def exported(self, *args: jax.Array) -> _FakeLowered:
        """Record export invocation and return serialized payload.

        Returns:
            Synthetic lowered export object carrying output avals and MLIR bytes.
        """
        self.export_calls += 1
        aval = jax.core.ShapedArray(args[0].shape, args[0].dtype)
        _, out_tree = jax.tree.flatten((aval,))
        return _FakeLowered(
            out_avals=(aval,),
            out_tree=out_tree,
            mlir_module_serialized=b"MLIRBC_TEST",
        )

    def __call__(self, *args: jax.Array) -> tuple[str, tuple[object, object]]:
        """Populate fingerprint on cold miss or return cached output shapes.

        Returns:
            Status tag paired with the recorded output shape entry.
        """
        key = f"k:{args[0].shape}"
        if key not in self.output_shapes:
            self.exported(*args)
            self.kernel_key_to_mlir_fingerprint[key] = "fp123"
            self.output_shapes[key] = ("cold", None)
        return ("ok", self.output_shapes[key])


def test_register_op_forwards_specs_and_export_cache_hook(
    monkeypatch: _MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Attach flattened PartitionSpecs and reload persisted MLIR from k3_export."""
    monkeypatch.setenv("VLLM_XLA_CACHE_PATH", str(tmp_path))
    registered_kernels: list[tuple[str, str, bytes]] = []

    def fake_sharded_jax_op(
        name: str,
        _fn: object,
        *,
        mesh: Mesh,
        **kwargs: object,
    ) -> object:
        """Construct wrapper carrying _init_fn like torch_tpu CustomOpDef.

        Returns:
            Namespace wrapping the configured _FakeJaxCallable instance.
        """
        raw_in_specs = kwargs["input_partition_specs"]
        assert isinstance(raw_in_specs, tuple)
        configure_callable = kwargs.get("configure_callable")
        callable_op = _FakeJaxCallable(name, mesh, raw_in_specs)
        if callable(configure_callable):
            configure_callable(callable_op)
        return types.SimpleNamespace(_init_fn=callable_op)

    monkeypatch.setattr(bridge, "_SHARDED_JAX_OP_ACCEPTS_CONFIGURE", True)
    monkeypatch.setattr(bridge, "sharded_jax_op", fake_sharded_jax_op)
    monkeypatch.setattr(runtime_config, "_ACTIVE", None)
    monkeypatch.setattr(
        export_cache.pallas_impl,
        "_get_kernel_invocation_key",
        lambda _trace_key, args, _kwargs, _static_argnums: f"k:{args[0].shape}",
    )
    monkeypatch.setattr(
        export_cache.pallas_impl.tpu_torch_pallas,
        "register_custom_kernel",
        lambda name, fp, serialized_mlir_module: registered_kernels.append(
            (name, fp, serialized_mlir_module),
        ),
    )
    monkeypatch.setattr(
        export_cache,
        "_create_torch_placeholder",
        lambda aval, spec, _mesh: (aval.shape, str(aval.dtype), spec),
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    spec = PartitionSpec("tp")
    op_wrapper = bridge.register_op(
        "kimi_k3::shim_unit_op",
        mesh,
        lambda x: (x,),
        bridge.OpRegistrationSpec(
            in_specs=(spec,),
            out_specs=(spec,),
            export_key="unit_v1",
        ),
    )
    callable_op = vars(op_wrapper)["_init_fn"]
    assert callable_op.output_partition_specs == [spec]

    sample = jnp.zeros((4,), dtype=jnp.float32)
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


class _FakeArenaSlice:
    """Minimal VMEM arena slice double recording bitcast and transform views."""

    def __init__(self, row_slice: object, col_slice: object) -> None:
        """Store sliced arena coordinates."""
        self.row_slice = row_slice
        self.col_slice = col_slice


class _FakeArena:
    """Minimal flat uint8 VMEM arena double indexed by 128-byte rows."""

    @property
    def at(self) -> _FakeArena:
        """Return indexable ``.at`` view matching Pallas ref indexing."""
        return self

    def __getitem__(self, indices: tuple[object, object]) -> _FakeArenaSlice:
        """Slice arena at requested row range and column slice.

        Returns:
            Synthetic arena slice recording the row and column slices.
        """
        return _FakeArenaSlice(indices[0], indices[1])


def test_vmem_pool_delegates_to_shared_dma_pool_alias(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Verify VmemPool delegates to vllm_torchtpu.megakernel.pool_alias."""
    assert vmem_pool.pool_alias is shared_pool_alias
    assert common.pool_alias is shared_pool_alias
    common_dir = Path(common.__file__).resolve().parent
    assert not (common_dir / "pool_alias.py").exists()

    recorded_views: list[tuple[object, tuple[int, ...], jnp.dtype]] = []

    def fake_view(
        ref: object,
        shape: tuple[int, ...],
        dtype: jnp.dtype | type,
    ) -> tuple[str, object, tuple[int, ...], jnp.dtype]:
        """Record delegated pool_alias.view arguments.

        Returns:
            Tagged tuple carrying the sliced ref, target shape, and dtype.
        """
        resolved_dtype = jnp.dtype(dtype)
        recorded_views.append((ref, tuple(shape), resolved_dtype))
        return ("aliased_view", ref, tuple(shape), resolved_dtype)

    monkeypatch.setattr(shared_pool_alias, "view", fake_view)
    arena = _FakeArena()
    pool = vmem_pool.VmemPool(arena, capacity=16384)
    offset = pool.allocate("stage_buf", 8192)
    assert offset == 0
    assert pool.offset("stage_buf") == 0

    aliased = pool.view(offset, (8, 128), jnp.bfloat16)
    assert isinstance(aliased, tuple)
    assert aliased[0] == "aliased_view"
    assert len(recorded_views) == 1
    assert recorded_views[0][1] == (8, 128)
    assert recorded_views[0][2] == jnp.dtype(jnp.bfloat16)
    pool.release("stage_buf")


def test_bridge_typed_signature_registers_without_exec(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Build typed inspect.Signature on bridge ops without calling builtins.exec."""
    captured_ops: dict[str, Callable[..., object]] = {}
    captured_specs: dict[str, bridge.OpRegistrationSpec] = {}

    def forbid_exec(*_args: object, **_kwargs: object) -> None:
        """Fail if dynamic code execution is invoked during op registration.

        Raises:
            AssertionError: Always, because bridge registration must avoid exec.
        """
        msg = "builtins.exec must not be called by bridge op registration"
        raise AssertionError(msg)

    def fake_sharded_op(
        name: str,
        fn: Callable[..., object],
        *,
        mesh: Mesh,
        spec: bridge.OpRegistrationSpec,
    ) -> Callable[..., object]:
        """Capture the typed wrapper function passed to _sharded_op.

        Args:
            name: Custom op registration name.
            fn: Typed JAX wrapper callable.
            mesh: Global JAX device mesh.
            spec: Resolved custom op registration specification.

        Returns:
            The wrapper callable registered for ``name``.
        """
        assert mesh.axis_names == ("tp",)
        captured_ops[name] = fn
        captured_specs[name] = spec
        return fn

    monkeypatch.setattr(builtins, "exec", forbid_exec)
    monkeypatch.setattr(bridge, "_sharded_op", fake_sharded_op)
    monkeypatch.setattr(runtime_config, "_ACTIVE", None)

    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    spec = PartitionSpec("tp")

    bridge.register_op(
        "kimi_k3::typed_local_op",
        mesh,
        lambda x, y: (x + y,),
        (spec, spec),
        (spec,),
    )
    local_fn = captured_ops["kimi_k3::typed_local_op"]
    local_sig = inspect.signature(local_fn)
    assert tuple(local_sig.parameters) == ("arg_0", "arg_1")
    for param in local_sig.parameters.values():
        assert param.annotation is jax.Array
    assert local_sig.return_annotation == tuple[jax.Array]
    assert local_fn.__annotations__ == {
        "arg_0": jax.Array,
        "arg_1": jax.Array,
        "return": tuple[jax.Array],
    }

    bridge.register_global_op(
        "kimi_k3::typed_global_op",
        mesh,
        lambda x: (x, x),
        bridge.OpRegistrationSpec(
            in_specs=(spec,),
            out_specs=(spec, spec),
            donate_argnums=(0,),
            export_key="global_v1",
        ),
    )
    assert captured_specs["kimi_k3::typed_global_op"].donate_argnums == (0,)
    assert captured_specs["kimi_k3::typed_global_op"].export_key == "global_v1"
    global_fn = captured_ops["kimi_k3::typed_global_op"]
    global_sig = inspect.signature(global_fn)
    assert tuple(global_sig.parameters) == ("arg_0",)
    assert global_sig.parameters["arg_0"].annotation is jax.Array
    assert global_sig.return_annotation == tuple[jax.Array, jax.Array]
    assert global_fn.__annotations__ == {
        "arg_0": jax.Array,
        "return": tuple[jax.Array, jax.Array],
    }


def test_bridge_resolve_op_spec_rejects_invalid_out_specs() -> None:
    """Fail closed with TypeError on missing or duplicate out_specs."""
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    spec = PartitionSpec("tp")
    missing_error = ""
    try:
        bridge.register_op("kimi_k3::missing_out", mesh, lambda x: (x,), (spec,))
    except TypeError as exc:
        missing_error = str(exc)
    assert "out_specs is required" in missing_error

    duplicate_error = ""
    try:
        bridge.register_op(
            "kimi_k3::extra_out",
            mesh,
            lambda x: (x,),
            bridge.OpRegistrationSpec(in_specs=(spec,), out_specs=(spec,)),
            (spec,),
        )
    except TypeError as exc:
        duplicate_error = str(exc)
    assert "out_specs must be omitted" in duplicate_error


def test_export_cache_placeholder_roundtrip(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Size local torch output placeholders without importing private _placeholder."""
    assert "_placeholder" not in vars(export_cache)
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    spec = PartitionSpec("tp", None)
    aval = jax.core.ShapedArray((8, 16), jnp.bfloat16)

    monkeypatch.setattr(
        export_cache.pallas_impl,
        "get_local_shape",
        lambda global_shape, _mesh, _spec: (global_shape[0] // 2, global_shape[1]),
    )
    monkeypatch.setattr(
        export_cache.pallas_impl,
        "JAX_TO_TORCH_DTYPE_MAP",
        {jnp.dtype(jnp.bfloat16): torch.bfloat16},
    )
    monkeypatch.setattr(
        export_cache.pallas_impl,
        "torch_placeholder",
        lambda target_aval, **_kwargs: torch.empty(
            target_aval.shape,
            dtype=torch.float32,
        ),
    )

    create_placeholder: Callable[..., torch.Tensor] = vars(export_cache)[
        "_create_torch_placeholder"
    ]
    sharded_tensor = create_placeholder(aval, spec, mesh)
    assert isinstance(sharded_tensor, torch.Tensor)
    assert sharded_tensor.shape == (4, 16)
    assert sharded_tensor.dtype == torch.bfloat16

    fallback_tensor = create_placeholder(aval, None, mesh)
    assert isinstance(fallback_tensor, torch.Tensor)
    assert fallback_tensor.shape == (8, 16)
    assert fallback_tensor.dtype == torch.float32


def test_collectives32_barrier_signals_all_xor_peers(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Signal all 31 XOR peer devices across the 32-rank tp axis and wait for 31."""
    signaled_peers: list[int] = []
    waited_counts: list[int] = []
    sentinel_sem = object()

    def record_signal(
        sem: object,
        inc: int,
        *,
        device_id: tuple[int, ...],
        device_id_type: object,
    ) -> None:
        """Record peer device ID when semaphore and mesh id type match."""
        is_valid = (
            sem is sentinel_sem and inc == 1 and device_id_type == pl.DeviceIdType.MESH
        )
        signaled_peers.append(device_id[0] if is_valid else -1)

    monkeypatch.setattr(jax.lax, "axis_index", lambda _axis: _TEST_RANK)
    monkeypatch.setattr(
        collectives32.tpu,
        "get_barrier_semaphore",
        lambda: sentinel_sem,
    )
    monkeypatch.setattr(pl, "semaphore_signal", record_signal)
    monkeypatch.setattr(
        pl,
        "semaphore_wait",
        lambda sem, count: waited_counts.append(
            count if sem is sentinel_sem else -1,
        ),
    )

    collectives32.barrier()
    assert len(signaled_peers) == _TP32_PEER_COUNT
    assert set(signaled_peers) == set(range(_TP32_WORLD_SIZE)) - {_TEST_RANK}
    assert waited_counts == [_TP32_PEER_COUNT]
