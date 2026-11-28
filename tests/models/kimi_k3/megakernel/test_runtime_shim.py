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
"""CPU unit tests for Kimi-K3 runtime_shim, vmem_pool, pool_alias, and collectives32."""

from __future__ import annotations

import base64
import contextlib
import enum
import gc
import re
import sys
import threading
import types
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.experimental.topologies as jax_topologies
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental import pallas as pl
from jax.experimental.mosaic.dialects import tpu as mosaic_tpu
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxlib.mlir import ir as mlir_ir
from vllm.config import SpeculativeConfig

from vllm_torchtpu.distributed import pallas_shapes
from vllm_torchtpu.models.kimi_k3.megakernel import common
from vllm_torchtpu.models.kimi_k3.megakernel import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel import layout as k3_layout
from vllm_torchtpu.models.kimi_k3.megakernel.common import (
    collectives32,
    pool_alias,
    vmem_pool,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    backend,
    bridge,
    compilation,
    export_cache,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim import (
    config as runtime_config,
)
from vllm_torchtpu.models.kimi_k3.megakernel.runtime_shim.bridge import (
    _register_sharded_op,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from vllm.config import VllmConfig

_TP_SIZE = 32
_TP_PEERS = 31
_MIB_BYTES = 1 << 20
_GIB_BYTES = 1 << 30
_TEST_RANK = 13
_KEY_HEX_LEN = 16
_COERCED_INT = 128
_COERCED_FLOAT = 6.5
_UNALIGNED_CAPACITY_BYTES = vmem_pool.ALIGN + 1
_TEST_ARENA_MIB = 4


def _clear_active_kernel_config(monkeypatch: _MonkeyPatch) -> None:
    """Reset the active Kimi-K3 runtime kernel config state for unit tests."""
    monkeypatch.setattr(runtime_config, "_ACTIVE", None)


class _PallasRefProtocol(Protocol):
    """Minimal Pallas memory-reference protocol for pool_alias lowering tests."""

    @property
    def at(self) -> object:
        """Return sliceable reference indexer."""


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

    def delenv(self, name: str, *, raising: bool = ...) -> None:
        """Remove environment variable for the duration of the test."""

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


@dataclass
class _CustomOpCapture:
    """Captured registration events and placeholder shapes from stubbed custom_op."""

    events: list[str]
    registered: dict[str, object]
    recorded_call_shapes: list[tuple[int, ...]]


def _install_custom_op_stubs(monkeypatch: _MonkeyPatch) -> _CustomOpCapture:
    """Install CPU-only torch custom_op and Pallas shape stubs for bridge tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Capture bundle recording hook ordering, fake_fn, and call output shapes.
    """
    capture = _CustomOpCapture(events=[], registered={}, recorded_call_shapes=[])
    monkeypatch.setattr(
        bridge.pallas_impl, "get_global_shape", pallas_shapes.get_global_shape
    )
    monkeypatch.setattr(
        bridge.pallas_impl, "get_local_shape", pallas_shapes.get_local_shape
    )

    def fake_custom_op(
        name: str, fn: object, *, mutates_args: tuple[object, ...]
    ) -> object:
        """Record custom_op registration and capture registered fake_fn."""
        assert mutates_args == ()
        capture.events.append("custom_op")
        capture.registered["name"] = name
        capture.registered["fn"] = fn

        def register_fake(fake_fn: Callable[..., object]) -> Callable[..., object]:
            """Capture registered fake_fn."""
            capture.registered["fake_fn"] = fake_fn
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
        capture.recorded_call_shapes.extend(tuple(out.shape) for out in output_shapes)
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
    return capture


class _FakeArenaSlice:
    """Minimal arena slice double recording bitcast and transform views."""

    def __init__(
        self,
        row_slice: object,
        col_slice: object,
        dtype: jnp.dtype | None = None,
    ) -> None:
        """Store sliced arena coordinates and active element dtype."""
        self.row_slice = row_slice
        self.col_slice = col_slice
        self.dtype = jnp.dtype(jnp.uint8) if dtype is None else dtype
        self.ref = self
        self.transforms: tuple[object, ...] = ()

    def bitcast(self, dtype: object) -> _FakeArenaSlice:
        """Return a slice view reinterpreted as ``dtype``."""
        return _FakeArenaSlice(self.row_slice, self.col_slice, jnp.dtype(dtype))


class _FakeArena:
    """Minimal flat uint8 arena double indexed by 128-byte rows."""

    shape: tuple[int, ...] = (128, 128)

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


def test_vmem_pool_uses_local_common_pool_alias() -> None:
    """Verify VmemPool uses Kimi-K3's local pool_alias."""
    assert vmem_pool.pool_alias is pool_alias
    common_dir = Path(common.__file__).resolve().parent
    assert (common_dir / "pool_alias.py").is_file()

    arena = _FakeArena()
    pool = vmem_pool.VmemPool(arena, capacity=16384)
    offset = pool.allocate("stage_buf", 8192)
    assert offset == 0
    assert pool.offset("stage_buf") == 0

    aliased = pool.view(offset, (8, 128), jnp.dtype(jnp.bfloat16))
    assert isinstance(aliased, pool_alias.types.TransformedRef)
    assert aliased.transforms[-1] == pool_alias.Reinterpret((8, 128))
    pool.release("stage_buf")


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
    assert len(signaled_peers) == _TP_PEERS
    assert set(signaled_peers) == set(range(_TP_SIZE)) - {_TEST_RANK}
    assert waited_counts == [_TP_PEERS]


def test_kernel_config_validates_invariants_and_coerces_cli_overrides() -> None:
    """Verify KernelConfig validates tile/dtype invariants and coerces CLI strings."""
    base = k3_config.KernelConfig()
    default_ktile = k3_config.KernelConfig.mla_decode_ktile
    experimental_ktile = 2 * default_ktile
    experimental_dp = 2
    experimental_dcp = _TP_SIZE // experimental_dp
    assert base.mla_decode_ktile == default_ktile

    coerced = base.replace(
        mla_decode_ktile=str(experimental_ktile),
        attention_dp=str(experimental_dp),
        dcp_group=str(experimental_dcp),
        vmem_spill_reserve_mb=str(_COERCED_FLOAT),
        vmem_pad_kb=str(_COERCED_INT),
        compact_only="true",
        weight_streams='{"dense_gu": [0, 2]}',
    )
    assert coerced.mla_decode_ktile == experimental_ktile
    assert coerced.attention_dp == experimental_dp
    assert coerced.dcp_group == experimental_dcp
    assert coerced.vmem_spill_reserve_mb == _COERCED_FLOAT
    assert coerced.vmem_pad_kb == _COERCED_INT
    assert coerced.compact_only is True
    assert coerced.weight_streams == (("dense_gu", (0, 2)),)
    assert len(coerced.key()) == _KEY_HEX_LEN

    with pytest.raises(ValueError, match="moe_reduce_dtype"):
        base.replace(moe_reduce_dtype="float16")
    with pytest.raises(ValueError, match="mla_decode_ktile supports"):
        base.replace(mla_decode_ktile=default_ktile // 2)
    with pytest.raises(ValueError, match="mla_decode_ktile=2048 requires"):
        base.replace(
            mla_decode_ktile=experimental_ktile, attention_dp=1, dcp_group=_TP_SIZE
        )


def test_cache_layout_selects_bf16_and_fp8_contracts_and_rejects_unknown() -> None:
    """Verify cache_layout maps BF16 and FP8 dtypes and rejects unsupported names."""
    bf16_auto = k3_layout.cache_layout("auto")
    bf16_explicit = k3_layout.cache_layout("torch.bfloat16")
    assert bf16_auto == bf16_explicit
    assert bf16_auto == k3_layout.LAYOUT
    assert bf16_auto.tokens_per_block == k3_layout.LAYOUT.tokens_per_block
    assert bf16_auto.tokens_per_row == k3_layout.LAYOUT.tokens_per_row

    fp8 = k3_layout.cache_layout("fp8_e4m3")
    assert fp8 == k3_layout.cache_layout("float8_e4m3fn")
    assert fp8 == k3_layout.FP8_LAYOUT
    assert fp8.tokens_per_block == 2 * bf16_auto.tokens_per_block
    assert fp8.tokens_per_row == 2 * bf16_auto.tokens_per_row

    with pytest.raises(NotImplementedError, match="unsupported"):
        k3_layout.cache_layout("float32")


def test_vmem_pool_enforces_alignment_lifecycle_and_loop_state_invariants() -> None:
    """Verify VmemPool rejects unaligned, over-capacity, and drifted loops."""
    assert _GIB_BYTES == 1024 * _MIB_BYTES
    with pytest.raises(vmem_pool.PoolError, match="multiple of 4 KiB"):
        vmem_pool.VmemPool(None, _UNALIGNED_CAPACITY_BYTES)

    pool = vmem_pool.VmemPool(None, _TEST_ARENA_MIB * _MIB_BYTES)
    with pytest.raises(vmem_pool.PoolError, match="MiB arena"):
        pool.allocate("oversized", (_TEST_ARENA_MIB + 1) * _MIB_BYTES)

    pool.request("tile_a", _MIB_BYTES, start=lambda _off, _idx: None)
    with pytest.raises(vmem_pool.PoolError, match="already live"):
        pool.request("tile_a", _MIB_BYTES, start=lambda _off, _idx: None)
    with pytest.raises(vmem_pool.PoolError, match="never waited on"):
        pool.release("tile_a")
    pool.acquire("tile_a")
    pool.release("tile_a")
    with pytest.raises(vmem_pool.PoolError, match="is not live"):
        pool.release("tile_a")

    with pytest.raises(vmem_pool.PoolError, match="arena state it began in"):

        @pool.loop(0, 1)
        def _drifted_body(_idx: int) -> None:
            """Traced loop body to verify drift detection."""
            pool.allocate("leaked_buf", _MIB_BYTES)


def test_pool_alias_unpadded_expert_scale_views() -> None:
    """Verify local pool_alias.view aliases unpadded scale and tile shapes."""
    # (32, 128) uint8 tiles: 4 * ceil(112/32)=4 * (1536/128)=12 * 4096 B.
    gus_shape = (4, 112, 1536)
    gus_unpadded = 4 * 112 * 1536
    assert (
        pool_alias.storage_size(gus_shape, jnp.dtype(jnp.uint8))
        == 4 * 4 * 12 * vmem_pool.ALIGN
    )
    gus_raw = _FakeArenaSlice(pl.ds(0, gus_unpadded // 128), slice(None))
    gus_view = pool_alias.view(gus_raw, gus_shape, jnp.uint8)
    assert isinstance(gus_view, pool_alias.types.TransformedRef)
    assert gus_view.transforms[-1] == pool_alias.Reinterpret(gus_shape)

    # (32, 128) uint8 tiles: 4 * ceil(24/32)=1 * (3584/128)=28 * 4096 B.
    ds_shape = (4, 24, 3584)
    ds_unpadded = 4 * 24 * 3584
    assert (
        pool_alias.storage_size(ds_shape, jnp.dtype(jnp.uint8))
        == 4 * 28 * vmem_pool.ALIGN
    )
    ds_raw = _FakeArenaSlice(pl.ds(0, ds_unpadded // 128), slice(None))
    ds_view = pool_alias.view(ds_raw, ds_shape, jnp.uint8)
    assert isinstance(ds_view, pool_alias.types.TransformedRef)
    assert ds_view.transforms[-1] == pool_alias.Reinterpret(ds_shape)

    # (16, 128) bf16 tiles: ceil(8/16)=1 * (128/128)=1 * 4096 B.
    bf16_shape = (8, 128)
    bf16_unpadded = 8 * 128 * 2
    assert (
        pool_alias.storage_size(bf16_shape, jnp.dtype(jnp.bfloat16)) == vmem_pool.ALIGN
    )
    bf16_raw = _FakeArenaSlice(pl.ds(0, bf16_unpadded // 128), slice(None))
    bf16_view = pool_alias.view(bf16_raw, bf16_shape, jnp.bfloat16)
    assert isinstance(bf16_view, pool_alias.types.TransformedRef)
    assert bf16_view.transforms[-1] == pool_alias.Reinterpret(bf16_shape)

    arena_rows = pool_alias.storage_size(gus_shape, jnp.dtype(jnp.uint8)) // 128

    def _alias_copy_kernel(
        src_ref: _PallasRefProtocol,
        dst_ref: _PallasRefProtocol,
        arena_ref: _PallasRefProtocol,
        sem_ref: object,
    ) -> None:
        """Copy an unpadded expert-scale tensor through a tiled arena view."""
        aliased = pool_alias.view(
            arena_ref.at[pl.ds(0, gus_unpadded // 128), :],
            gus_shape,
            jnp.uint8,
        )
        for src, dst in ((src_ref, aliased), (aliased, dst_ref)):
            copy = pltpu.make_async_copy(src, dst, sem_ref)
            copy.start()
            copy.wait()

    pallas_fn = pl.pallas_call(
        _alias_copy_kernel,
        out_shape=jax.ShapeDtypeStruct(gus_shape, jnp.uint8),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((arena_rows, 128), jnp.uint8),
            pltpu.SemaphoreType.DMA(()),
        ],
    )
    devices = jax_topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    sharding = NamedSharding(Mesh(devices[:1], ("tp",)), PartitionSpec())
    try:
        hlo_text = (
            jax.jit(pallas_fn, in_shardings=sharding, out_shardings=sharding)
            .lower(jax.ShapeDtypeStruct(gus_shape, jnp.uint8))
            .as_text()
        )
    finally:
        jax.clear_caches()
    body_match = re.search(r"\\22body\\22:\s*\\22([A-Za-z0-9+/=]+)\\22", hlo_text)
    assert body_match is not None
    with mlir_ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        mosaic_tpu.register_dialect(ctx)
        mosaic_text = str(mlir_ir.Module.parse(base64.b64decode(body_match.group(1))))
    assert "tpu.reinterpret_cast" in mosaic_text
    assert "#tpu.tiled<(32,128)(4,1),[48,12,1]>" in mosaic_text


@dataclass
class _StubLoadConfig:
    """Minimal load_config stand-in for backend and engine config tests."""

    load_format: object = "megakernel"
    model_loader_extra_config: object = None


@dataclass
class _StubKvTransferConfig:
    """Minimal kv_transfer_config stand-in selecting prefill or decode role."""

    is_kv_producer: bool = False
    is_kv_consumer: bool = True


@dataclass
class _StubHfConfig:
    """Minimal hf_config stand-in exposing supported model architectures."""

    architectures: tuple[str, ...] | None = ("KimiK3ForConditionalGeneration",)


@dataclass
class _StubModelConfig:
    """Minimal model_config stand-in carrying hf_config and max_model_len."""

    hf_config: _StubHfConfig | None = None
    max_model_len: int = 32768


@dataclass
class _StubParallelConfig:
    """Minimal parallel_config stand-in for TP and DCP settings."""

    tensor_parallel_size: int = _TP_SIZE
    decode_context_parallel_size: int = _TP_SIZE
    cp_kv_cache_interleave_size: int = 1


@dataclass
class _StubSchedulerConfig:
    """Minimal scheduler_config stand-in for batch and token limits."""

    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    async_scheduling: bool = True


@dataclass
class _StubCacheConfig:
    """Minimal cache_config stand-in for block size and dtypes."""

    block_size: int = 256
    cache_dtype: str = "fp8"
    mamba_ssm_cache_dtype: str = "auto"


class _StubSpeculativeConfig(SpeculativeConfig):
    """Minimal speculative_config stand-in inheriting real rate conversion."""

    def __init__(
        self,
        *,
        num_speculative_tokens: int = 7,
        num_speculative_tokens_per_batch_size: tuple[tuple[int, int, int], ...]
        | None = (
            (1, 8, 7),
            (9, 64, 3),
        ),
        rejection_sample_method: str | None = "synthetic",
        synthetic_acceptance_rates: tuple[float, ...] = (0.9, 0.8, 0.7),
    ) -> None:
        """Initialize stub fields without invoking vLLM's Pydantic validator."""
        self.num_speculative_tokens = num_speculative_tokens
        self.num_speculative_tokens_per_batch_size = (
            num_speculative_tokens_per_batch_size
        )
        self.rejection_sample_method = rejection_sample_method
        self.synthetic_acceptance_rates = synthetic_acceptance_rates


@dataclass
class _StubVllmConfig:
    """Minimal VllmConfig stand-in for runtime_shim unit tests."""

    load_config: _StubLoadConfig | None
    kv_transfer_config: _StubKvTransferConfig | None
    model_config: _StubModelConfig | None
    parallel_config: _StubParallelConfig
    scheduler_config: _StubSchedulerConfig
    cache_config: _StubCacheConfig
    speculative_config: _StubSpeculativeConfig | None = None


def _make_vllm_config(
    *,
    is_producer: bool = False,
    is_consumer: bool = True,
    extra: object = None,
    spec: _StubSpeculativeConfig | None = None,
    load_format: str = "megakernel",
) -> VllmConfig:
    """Construct a synthetic VllmConfig for runtime_shim unit tests."""
    resolved_extra = {"presharded": "/mnt/k3_shards"} if extra is None else extra
    return cast(
        "VllmConfig",
        _StubVllmConfig(
            load_config=_StubLoadConfig(
                load_format=load_format,
                model_loader_extra_config=resolved_extra,
            ),
            kv_transfer_config=_StubKvTransferConfig(
                is_kv_producer=is_producer,
                is_kv_consumer=is_consumer,
            ),
            model_config=_StubModelConfig(hf_config=_StubHfConfig()),
            parallel_config=_StubParallelConfig(),
            scheduler_config=_StubSchedulerConfig(),
            cache_config=_StubCacheConfig(),
            speculative_config=spec,
        ),
    )


def test_backend_role_and_loader_config_contracts() -> None:
    """Verify backend role selection and loader validation."""
    inactive = _make_vllm_config(load_format="safetensors")
    assert backend.role(inactive) is None
    assert backend.synthetic_acceptance_lengths(inactive) is None

    prefill_cfg = _make_vllm_config(
        is_producer=True,
        is_consumer=False,
        extra={"presharded": "/mnt/shards", "draft": "/mnt/draft"},
    )
    assert backend.role(prefill_cfg) == "prefill"
    assert backend.presharded(prefill_cfg) == Path("/mnt/shards")
    assert backend.draft(prefill_cfg) == Path("/mnt/draft")
    assert backend.synthetic_acceptance_lengths(prefill_cfg) is None

    expected_dp = 2
    decode_cfg = _make_vllm_config(
        extra={
            "presharded": "/mnt/shards",
            "attention_dp": expected_dp,
            "synthetic_acceptance_lengths": {"3": 2.87, "7": 3.78},
        },
    )
    assert backend.role(decode_cfg) == "decode"
    assert backend.draft(decode_cfg) is None
    assert backend.attention_dp(decode_cfg) == expected_dp
    assert backend.synthetic_acceptance_lengths(decode_cfg) == {3: 2.87, 7: 3.78}

    with pytest.raises(ValueError, match="takes --model-loader-extra-config as a dict"):
        backend.role(_make_vllm_config(extra=["bad"]))
    with pytest.raises(ValueError, match="megakernel decode role takes"):
        backend.role(_make_vllm_config(extra={"presharded": "/x", "draft": "/d"}))
    with pytest.raises(ValueError, match="presharded"):
        backend.role(_make_vllm_config(extra={}))
    with pytest.raises(
        ValueError, match="maps draft widths to mean acceptance lengths"
    ):
        backend.synthetic_acceptance_lengths(
            _make_vllm_config(
                extra={"presharded": "/x", "synthetic_acceptance_lengths": [1, 2]}
            )
        )
    with pytest.raises(
        ValueError, match="maps draft widths to mean acceptance lengths"
    ):
        backend.synthetic_acceptance_lengths(
            _make_vllm_config(
                extra={"presharded": "/x", "synthetic_acceptance_lengths": {}}
            )
        )


def test_engine_config_from_vllm_config_acceptance_rates(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Verify KernelConfig protocol and MegakernelEngineConfig.from_vllm_config."""
    assert isinstance(runtime_config.KernelConfig, type)
    _clear_active_kernel_config(monkeypatch)
    installed: runtime_config.KernelConfig = k3_config.KernelConfig()
    runtime_config.set_active_kernel_config(installed)
    assert runtime_config.get_active_kernel_config() is installed

    expected_dp = 2
    spec = _StubSpeculativeConfig()
    vllm_cfg = _make_vllm_config(
        extra={
            "presharded": "/mnt/shards",
            "attention_dp": expected_dp,
            "synthetic_acceptance_lengths": {"3": 2.5, "7": 3.25},
        },
        spec=spec,
    )
    engine_cfg = runtime_config.MegakernelEngineConfig.from_vllm_config(vllm_cfg)
    assert engine_cfg.role == "decode"
    assert engine_cfg.attention_dp == expected_dp
    assert engine_cfg.synthetic_acceptance_rates == (0.9, 0.8, 0.7)
    assert engine_cfg.synthetic_acceptance_rates_by_width is not None
    by_width = dict(engine_cfg.synthetic_acceptance_rates_by_width)
    assert by_width[3] == pytest.approx((1.0, 0.5, 0.0))
    assert by_width[7] == pytest.approx((1.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0))

    bad_method_cfg = _make_vllm_config(
        extra={"presharded": "/mnt/shards", "synthetic_acceptance_lengths": {"3": 2.0}},
        spec=_StubSpeculativeConfig(rejection_sample_method="rejection_sampler"),
    )
    with pytest.raises(ValueError, match="rejection_sample_method synthetic"):
        runtime_config.MegakernelEngineConfig.from_vllm_config(bad_method_cfg)

    out_of_range_cfg = _make_vllm_config(
        extra={"presharded": "/mnt/shards", "synthetic_acceptance_lengths": {"3": 5.0}},
        spec=spec,
    )
    with pytest.raises(ValueError, match="mean acceptance length"):
        runtime_config.MegakernelEngineConfig.from_vllm_config(out_of_range_cfg)


def _install_real_jax_callable_spies(
    monkeypatch: _MonkeyPatch,
) -> tuple[dict[tuple[str, str], bytes], list[tuple[str, str]]]:
    """Install in-memory kernel registry and call spies for real ``JaxCallable``."""
    registry: dict[tuple[str, str], bytes] = {}
    calls_seen: list[tuple[str, str]] = []

    def fake_register_kernel(
        name: str,
        fingerprint: str,
        *,
        serialized_mlir_module: bytes,
    ) -> None:
        """Record registered serialized MLIR module in memory."""
        registry[(name, fingerprint)] = serialized_mlir_module

    def fake_lookup_kernel(name: str, fingerprint: str) -> bool:
        """Report whether ``(name, fingerprint)`` was registered."""
        return (name, fingerprint) in registry

    def fake_call_kernel(
        name: str,
        fingerprint: str,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        """Record custom-kernel invocation and return placeholder outputs."""
        calls_seen.append((name, fingerprint))
        output_shapes = cast("list[torch.Tensor]", kwargs["output_shapes"])
        return list(output_shapes)

    _install_custom_op_stubs(monkeypatch)
    monkeypatch.setattr(
        bridge.pallas_impl.tpu_torch_pallas,
        "register_custom_kernel",
        fake_register_kernel,
    )
    monkeypatch.setattr(
        bridge.pallas_impl.tpu_torch_pallas,
        "lookup_custom_kernel",
        fake_lookup_kernel,
    )
    monkeypatch.setattr(
        bridge.pallas_impl.tpu_torch_pallas,
        "call_custom_kernel",
        fake_call_kernel,
    )
    monkeypatch.setattr(
        bridge.pallas_impl,
        "_validate_pallas_mesh_world_size",
        lambda **_kwargs: None,
    )
    return registry, calls_seen


def test_export_cache_roundtrip_through_real_jax_callable(
    monkeypatch: _MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Store and reload exported MLIR through real ``JaxCallable.__call__``."""
    topology = jax_topologies.get_topology_desc("tpu7x_2x2x4")
    mesh = Mesh(np.array(list(topology.devices)), ("tp",))
    in_spec = PartitionSpec("tp", None)
    out_specs = (PartitionSpec("tp", None), PartitionSpec())

    monkeypatch.setenv("VLLM_XLA_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(export_cache, "_process_rank", lambda: 0)
    registry, calls_seen = _install_real_jax_callable_spies(monkeypatch)

    def body(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return sharded and replicated outputs for real export lowering."""
        return x, x

    sample = torch.zeros((4, 16), dtype=torch.float32)
    op_name = "kimi_k3::real_cache_roundtrip_op"

    previous_ignore_forward_compatibility = bool(
        jax.config.jax_export_ignore_forward_compatibility
    )
    jax.config.update("jax_export_ignore_forward_compatibility", val=True)
    try:
        cold_wrapper = _register_sharded_op(
            op_name,
            body,
            mesh=mesh,
            input_partition_specs=(in_spec,),
            output_partition_specs=out_specs,
            configure_callable=export_cache.hook("real_jax_callable_roundtrip"),
        )
        cold_callable = getattr(cold_wrapper, "_init_fn", None)
        assert isinstance(cold_callable, bridge.pallas_impl.JaxCallable)
        cold_outputs = cold_callable(sample)
        assert isinstance(cold_outputs, (list, tuple))
        assert len(cold_outputs) == len(out_specs)
        assert len(calls_seen) == 1
        cold_fingerprint = calls_seen[0][1]

        registry.clear()
        warm_wrapper = _register_sharded_op(
            op_name,
            body,
            mesh=mesh,
            input_partition_specs=(in_spec,),
            output_partition_specs=out_specs,
            configure_callable=export_cache.hook("real_jax_callable_roundtrip"),
        )
        warm_callable = getattr(warm_wrapper, "_init_fn", None)
        assert isinstance(warm_callable, bridge.pallas_impl.JaxCallable)

        def fail_if_exported(*_args: object, **_kwargs: object) -> object:
            """Fail test if warm cache hit triggers fresh JAX export."""
            msg = "warm reload must not re-export"
            raise AssertionError(msg)

        monkeypatch.setattr(warm_callable, "exported", fail_if_exported)
        warm_outputs = warm_callable(sample)
        assert isinstance(warm_outputs, (list, tuple))
        assert [tuple(t.shape) for t in warm_outputs] == [
            tuple(t.shape) for t in cold_outputs
        ]
        assert calls_seen[-1] == (op_name, cold_fingerprint)
        assert (op_name, cold_fingerprint) in registry
    finally:
        jax.config.update(
            "jax_export_ignore_forward_compatibility",
            val=previous_ignore_forward_compatibility,
        )
        jax.clear_caches()


def test_stream_to_device_releases_host_array_before_next_item(
    monkeypatch: _MonkeyPatch,
) -> None:
    """Verify stream_to_device drops references to host arrays before reading next."""

    class _TrackableArray(np.ndarray):
        """Weakref-capable ndarray subclass to verify timely release."""

    uploaded_first = threading.Event()
    weak_refs: list[weakref.ref[_TrackableArray]] = []
    first_released_before_second = False

    def fake_upload(array: np.ndarray) -> torch.Tensor:
        """Consume the host array and return a tiny CPU placeholder tensor."""
        out = torch.tensor([float(array.flat[0])], dtype=torch.float32)
        uploaded_first.set()
        return out

    def source_iter() -> types.GeneratorType[tuple[str, str, np.ndarray], None, None]:
        """Yield two arrays and check that the first was freed while waiting."""
        nonlocal first_released_before_second
        first = np.ones((16,), dtype=np.float32).view(_TrackableArray)
        weak_refs.append(weakref.ref(first))
        yield ("decode", "w0", first)
        del first
        assert uploaded_first.wait(timeout=5.0)
        # Allow the worker future to clear its closure reference.
        for _ in range(50):
            gc.collect()
            if weak_refs[0]() is None:
                break
            threading.Event().wait(0.01)
        first_released_before_second = weak_refs[0]() is None
        second = np.full((16,), 2.0, dtype=np.float32).view(_TrackableArray)
        yield ("decode", "w1", second)

    monkeypatch.setattr(bridge, "upload", fake_upload)
    collected: dict[str, dict[str, torch.Tensor]] = {"decode": {}}

    def store(group: str, name: str, tensor: torch.Tensor) -> None:
        """Record uploaded tensor into ``collected``."""
        collected[group][name] = tensor

    stats = bridge.stream_to_device(source_iter(), store)
    assert stats["gib"] > 0.0
    assert set(collected["decode"]) == {"w0", "w1"}
    assert first_released_before_second
