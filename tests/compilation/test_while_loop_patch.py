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
"""MLIR AST, FX pass, and lowering unit tests for while_loop.patch."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import importlib
import operator
import pickle
import sys
import textwrap
import threading
import types
from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from jax._src.interpreters.mlir import make_ir_context
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
from torch._higher_order_ops.while_loop import while_loop_op as _raw_while_loop_op
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from vllm_torchtpu import patch_registry
from vllm_torchtpu.compilation.while_loop import (
    _mlir as wl_mlir,
)
from vllm_torchtpu.compilation.while_loop import (
    ops as wl_ops,
)
from vllm_torchtpu.compilation.while_loop import (
    tpu_while_loop,
)

while_loop_op: Any = _raw_while_loop_op

_MLIR_DTYPES: dict[torch.dtype, str] = {
    torch.bool: "i1",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.uint64: "ui64",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}


def _tensor_mlir_ty(t: torch.Tensor) -> str:
    elem = _MLIR_DTYPES[t.dtype]
    dims = "x".join(str(int(d)) for d in t.shape)
    return f"tensor<{dims}x{elem}>" if dims else f"tensor<{elem}>"


def _parse_and_verify_mlir(payload: str | bytes) -> tuple[ir.Module, str]:
    with make_ir_context():
        mod = ir.Module.parse(payload)
        assert mod.operation.verify()
        return mod, str(mod.operation.get_asm(enable_debug_info=False))


def _build_while_loop_mlir(
    cond_mlir_text: str,
    body_mlir_text: str,
    *,
    num_carries: int,
    num_additional_inputs: int = 0,
    has_rng_state: bool = False,
    num_partitions: int = 1,
    allowed_donor_indices: set[int] | None = None,
) -> wl_mlir._WhileLoopMlirArtifact:
    with wl_mlir._parse_while_subgraphs(
        cond_mlir_text, body_mlir_text, num_carries=num_carries
    ) as parsed:
        return wl_mlir._assemble_while_loop_mlir(
            parsed,
            num_carries=num_carries,
            num_additional_inputs=num_additional_inputs,
            has_rng_state=has_rng_state,
            num_partitions=num_partitions,
            allowed_donor_indices=allowed_donor_indices,
        )


class _MockTpuTensor(torch.Tensor):
    """Lightweight PrivateUse1 tensor wrapper backed by a CPU tensor."""

    _cpu: torch.Tensor
    _dev_idx: int

    @staticmethod
    def __new__(cls, cpu: torch.Tensor, dev_idx: int = 0) -> _MockTpuTensor:
        wrapped = torch.Tensor._make_wrapper_subclass(
            cls,
            cpu.shape,
            strides=cpu.stride(),
            storage_offset=0,
            dtype=cpu.dtype,
            device=torch.device("privateuseone", dev_idx),
        )
        wrapped._cpu = cpu
        wrapped._dev_idx = dev_idx
        return wrapped

    def __repr__(self) -> str:
        return f"_MockTpuTensor(shape={tuple(self.shape)}, dev={self._dev_idx})"

    def data_ptr(self) -> int:
        return self._cpu.data_ptr()

    def storage_offset(self) -> int:  # type: ignore[override]
        return int(self._cpu.storage_offset())

    def is_contiguous(  # type: ignore[override]
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> bool:
        return bool(self._cpu.is_contiguous(memory_format=memory_format))

    def untyped_storage(self) -> torch.UntypedStorage:
        return self._cpu.untyped_storage()

    def __getitem__(self, idx: Any) -> _MockTpuTensor:  # type: ignore[override]
        return _MockTpuTensor(self._cpu[idx], self._dev_idx)

    def t(self) -> _MockTpuTensor:
        return _MockTpuTensor(self._cpu.t(), self._dev_idx)

    def contiguous(
        self, memory_format: torch.memory_format = torch.contiguous_format
    ) -> _MockTpuTensor:
        return (
            self
            if self._cpu.is_contiguous(memory_format=memory_format)
            else _MockTpuTensor(
                self._cpu.contiguous(memory_format=memory_format), self._dev_idx
            )
        )

    def clone(  # type: ignore[override]
        self, *, memory_format: torch.memory_format | None = torch.preserve_format
    ) -> _MockTpuTensor:
        fmt = torch.preserve_format if memory_format is None else memory_format
        return _MockTpuTensor(self._cpu.clone(memory_format=fmt), self._dev_idx)

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]
        cls, func_op: Any, types_tuple: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        del types_tuple
        kwargs = kwargs or {}
        if func_op is while_loop_op:
            return while_loop_op.dispatch(
                torch._C.DispatchKey.PrivateUse1, *args, **kwargs
            )
        leaves = [
            x._dev_idx
            for x in pytree.tree_leaves((args, kwargs))
            if isinstance(x, _MockTpuTensor)
        ]
        dev_idx = leaves[0] if leaves else 0
        u_args = pytree.tree_map(
            lambda v: v._cpu if isinstance(v, _MockTpuTensor) else v, args
        )
        u_kw = pytree.tree_map(
            lambda v: v._cpu if isinstance(v, _MockTpuTensor) else v, kwargs
        )
        out = func_op(*u_args, **u_kw)
        if func_op is torch.ops.aten._local_scalar_dense.default:
            return out
        return pytree.tree_map(
            lambda v: _MockTpuTensor(v, dev_idx) if isinstance(v, torch.Tensor) else v,
            out,
        )


@dataclasses.dataclass
class _MockGenerator:
    state: _MockTpuTensor

    @classmethod
    def create(cls, seed: int, dev_idx: int = 0) -> _MockGenerator:
        return cls(_MockTpuTensor(torch.tensor([seed, 0], dtype=torch.uint64), dev_idx))


def _default_subgraph_mlir(
    args: Sequence[torch.Tensor], res: Sequence[torch.Tensor]
) -> str:
    arg_sig = ", ".join(f"%a{i}: {_tensor_mlir_ty(t)}" for i, t in enumerate(args))
    res_tys = [_tensor_mlir_ty(t) for t in res]
    if len(res) == 1 and res[0].dtype == torch.bool and res[0].ndim == 0:
        body = (
            "    %c = stablehlo.constant dense<true> : tensor<i1>\n"
            "    return %c : tensor<i1>"
        )
        ret_sig = "tensor<i1>"
    else:
        donor_set = {
            i
            for i, a in enumerate(args[: len(res)])
            if getattr(a, "_donor_hint", False)
        }
        arg_parts = [
            f"%a{i}: {_tensor_mlir_ty(t)}"
            + (" {jax.buffer_donor = true}" if i in donor_set else "")
            for i, t in enumerate(args)
        ]
        arg_sig = ", ".join(arg_parts)
        vals = ", ".join(f"%a{i}" for i in range(len(res)))
        ret_sig = f"({', '.join(res_tys)})" if len(res_tys) > 1 else res_tys[0]
        body = f"    return {vals} : {', '.join(res_tys)}"
    return (
        f"module {{\n  func.func public @main({arg_sig}) -> {ret_sig} {{\n"
        f"{body}\n  }}\n}}\n"
    )


@dataclasses.dataclass
class _MockTorchTpuRig:
    registered_kernels: dict[tuple[str, str], str | bytes] = dataclasses.field(
        default_factory=dict
    )
    custom_kernel_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    build_mlir_calls: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    force_strides_calls: list[tuple[int, ...]] = dataclasses.field(default_factory=list)
    mlir_queue: list[str] = dataclasses.field(default_factory=list)
    default_device_assignment: list[int] = dataclasses.field(
        default_factory=lambda: [0, 1, 4, 5, 6, 7, 2, 3]
    )
    default_generators: list[_MockGenerator] = dataclasses.field(default_factory=list)
    export_override_generators: dict[int, Any] = dataclasses.field(default_factory=dict)
    tpu_torch_compile: Any = dataclasses.field(init=False)
    tpu_torch_pallas: Any = dataclasses.field(init=False)
    execution_mode: Any = dataclasses.field(init=False)
    device_mesh: Any = dataclasses.field(init=False)

    def active_default_generator(self, dev_idx: int = 0) -> Any:
        if dev_idx in self.export_override_generators:
            return self.export_override_generators[dev_idx]
        if (
            self.execution_mode.eager_mode
            == self.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            raise RuntimeError("No export-mode generator override active")
        return self.default_generators[dev_idx]

    def advance_default_rng(self, dev_idx: int = 0) -> _MockTpuTensor:
        gen = self.active_default_generator(dev_idx)
        s0, s1 = (int(x) for x in gen.state._cpu.view(torch.int64).tolist())
        gen.state = _MockTpuTensor(
            torch.tensor([s0, s1 + 1], dtype=torch.uint64), dev_idx
        )
        return gen.state


@pytest.fixture
def while_loop_dispatch_guard():
    orig_k = dict(while_loop_op.py_kernels)
    orig_nf = while_loop_op.non_fallthrough_keys
    orig_py = dict(while_loop_op.python_key_table)
    try:
        yield
    finally:
        while_loop_op.py_kernels.clear()
        while_loop_op.py_kernels.update(orig_k)
        while_loop_op.non_fallthrough_keys = orig_nf
        while_loop_op.python_key_table.clear()
        while_loop_op.python_key_table.update(orig_py)
        while_loop_op._dispatch_cache.clear()


@pytest.fixture
def installed_while_loop_patch(
    monkeypatch, while_loop_dispatch_guard
) -> tuple[Any, _MockTorchTpuRig]:
    rig = _MockTorchTpuRig(default_generators=[_MockGenerator.create(42, 0)])

    tpu_ns = types.SimpleNamespace()

    class _GenProxy:
        def __getitem__(self, idx: int) -> Any:
            return rig.active_default_generator(idx)

    tpu_ns.default_generators = _GenProxy()
    monkeypatch.setattr(torch, "tpu", tpu_ns, raising=False)

    exec_mod: Any = types.ModuleType("torch_tpu._internal.execution_mode")

    class EagerMode(enum.Enum):
        DEFER_AND_FUSE = 1
        _INTERNAL_COMPILE_FX_GRAPH = 2

    exec_mod.EagerMode = EagerMode
    exec_mod.eager_mode = EagerMode.DEFER_AND_FUSE

    @contextlib.contextmanager
    def set_eager_mode(mode: EagerMode):
        prev = exec_mod.eager_mode
        exec_mod.eager_mode = mode
        try:
            yield
        finally:
            exec_mod.eager_mode = prev

    exec_mod.set_eager_mode = set_eager_mode
    rig.execution_mode = exec_mod

    ttc: Any = types.ModuleType("torch_tpu._internal.compile.tpu_torch_compile")
    ttc.placeholder_like = lambda t, *_a: _MockTpuTensor(
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype),
        getattr(t, "_dev_idx", 0),
    )

    def force_strides(
        t: _MockTpuTensor, strides: list[int], storage_offset: int = 0
    ) -> _MockTpuTensor:
        rig.force_strides_calls.append(tuple(int(s) for s in strides))
        need = (
            1
            + sum(max(0, d - 1) * abs(s) for d, s in zip(t.shape, strides, strict=True))
            + storage_offset
        )
        buf = torch.zeros((max(1, need),), dtype=t.dtype)
        res = _MockTpuTensor(
            torch.as_strided(buf, tuple(t.shape), tuple(strides), storage_offset),
            t._dev_idx,
        )
        res._donor_root = getattr(t, "_donor_root", t)
        return res

    ttc.force_strides = force_strides
    ttc.get_device_state_tensor = lambda g: g.state
    ttc.set_device_state_tensor = lambda g, s: setattr(g, "state", s)
    ttc.make_export_mode_device_generator = lambda g: _MockGenerator(g.state.clone())
    ttc.set_export_mode_default_device_generator = lambda g, idx: (
        rig.export_override_generators.pop(idx, None)
        if g is None
        else rig.export_override_generators.__setitem__(idx, g)
    )
    ttc.build_mlir = (
        lambda *, result_tensors, argument_tensors, use_stablehlo_bounds=False: (
            rig.build_mlir_calls.append(
                {
                    "argument_tensors": list(argument_tensors),
                    "result_tensors": list(result_tensors),
                }
            ),
            rig.mlir_queue.pop(0)
            if rig.mlir_queue
            else _default_subgraph_mlir(argument_tensors, result_tensors),
        )[1]
    )
    ttc.serialize_mlir_text = lambda mod, enable_debug_info=True: str(mod)
    rig.tpu_torch_compile = ttc

    compiler_mod: Any = types.ModuleType("torch_tpu._internal.compile.compiler")
    compiler_mod.is_pinned_host_tensor = lambda _t: False

    export_mod: Any = types.ModuleType("torch_tpu._internal.export.export")
    export_mod.EagerLikeFxInterpreter = fx.Interpreter

    ttp: Any = types.ModuleType("torch_tpu._internal.pallas.tpu_torch_pallas")
    ttp.lookup_custom_kernel = lambda n, k: (n, k) in rig.registered_kernels
    ttp.register_custom_kernel = lambda n, k, *, serialized_mlir_module: (
        rig.registered_kernels.__setitem__((n, k), serialized_mlir_module)
    )

    def call_custom_kernel(
        name: str,
        kernel_key: str,
        *,
        inputs: Sequence[torch.Tensor],
        output_shapes: Sequence[torch.Tensor],
        donate_argnums: Sequence[int] = (),
        device_assignment: Sequence[int] = (),
    ) -> list[_MockTpuTensor]:
        for idx in donate_argnums:
            if 0 <= int(idx) < len(inputs):
                inp = inputs[int(idx)]
                getattr(inp, "_donor_root", inp)._donor_hint = True
        rig.custom_kernel_calls.append(
            {
                "name": name,
                "kernel_key": kernel_key,
                "donate_argnums": [int(i) for i in donate_argnums],
                "device_assignment": [int(d) for d in device_assignment],
            }
        )
        return [
            _MockTpuTensor(
                torch.zeros(tuple(o.shape), dtype=o.dtype), getattr(o, "_dev_idx", 0)
            )
            for o in output_shapes
        ]

    ttp.call_custom_kernel = call_custom_kernel
    rig.tpu_torch_pallas = ttp

    dm_mod: Any = types.ModuleType("torch_tpu._internal.distributed.device_mesh")
    dm_mod.get_device_assignment = lambda _pg=None: list(rig.default_device_assignment)
    rig.device_mesh = dm_mod

    cmrp_mod: Any = types.ModuleType(
        "torch_tpu._internal.compile.fx_passes.clone_mutated_returned_placeholders"
    )
    cmrp_mod._is_returned_compute_placeholder = lambda n, ret: n in ret
    sc_mod: Any = types.ModuleType("torch_tpu._internal.compile.split_compiler")

    class SplitCompiler:
        def _compile_graph(self, gm: Any, *_a: Any, **_kw: Any) -> Any:
            return types.SimpleNamespace(_split_gm=gm)

    sc_mod.SplitCompiler = SplitCompiler
    sc_mod._SplitCompiledExecutable = lambda target: types.SimpleNamespace(
        _split_gm=target
    )

    root_mod: Any = types.ModuleType("torch_tpu")
    internal_mod: Any = types.ModuleType("torch_tpu._internal")
    compile_pkg: Any = types.ModuleType("torch_tpu._internal.compile")
    fx_pkg: Any = types.ModuleType("torch_tpu._internal.compile.fx_passes")
    export_pkg: Any = types.ModuleType("torch_tpu._internal.export")
    pallas_pkg: Any = types.ModuleType("torch_tpu._internal.pallas")
    dist_pkg: Any = types.ModuleType("torch_tpu._internal.distributed")

    fx_pkg.clone_mutated_returned_placeholders = cmrp_mod
    compile_pkg.tpu_torch_compile = ttc
    compile_pkg.compiler = compiler_mod
    compile_pkg.split_compiler = sc_mod
    compile_pkg.fx_passes = fx_pkg
    export_pkg.export = export_mod
    pallas_pkg.tpu_torch_pallas = ttp
    dist_pkg.device_mesh = dm_mod
    internal_mod.execution_mode = exec_mod
    internal_mod.compile = compile_pkg
    internal_mod.export = export_pkg
    internal_mod.pallas = pallas_pkg
    internal_mod.distributed = dist_pkg
    root_mod._internal = internal_mod

    for m_name, m_obj in {
        "torch_tpu": root_mod,
        "torch_tpu._internal": internal_mod,
        "torch_tpu._internal.execution_mode": exec_mod,
        "torch_tpu._internal.compile": compile_pkg,
        "torch_tpu._internal.compile.tpu_torch_compile": ttc,
        "torch_tpu._internal.compile.compiler": compiler_mod,
        "torch_tpu._internal.compile.split_compiler": sc_mod,
        "torch_tpu._internal.compile.fx_passes": fx_pkg,
        (
            "torch_tpu._internal.compile.fx_passes.clone_mutated_returned_placeholders"
        ): cmrp_mod,
        "torch_tpu._internal.export": export_pkg,
        "torch_tpu._internal.export.export": export_mod,
        "torch_tpu._internal.pallas": pallas_pkg,
        "torch_tpu._internal.pallas.tpu_torch_pallas": ttp,
        "torch_tpu._internal.distributed": dist_pkg,
        "torch_tpu._internal.distributed.device_mesh": dm_mod,
    }.items():
        monkeypatch.setitem(sys.modules, m_name, m_obj)

    monkeypatch.setattr(patch_registry, "_applied", set())
    monkeypatch.setattr(patch_registry, "_applying", set())
    monkeypatch.setattr(patch_registry, "_active_stages", set())
    monkeypatch.setattr(
        patch_registry,
        "PATCHES",
        tuple(
            p
            for p in patch_registry.PATCHES
            if p.target.endswith(":install_while_loop_patch")
        ),
    )
    patch_registry.apply("worker_init")
    while_loop_op.py_impl(_MockTpuTensor)(
        lambda *a, **kw: while_loop_op.dispatch(
            torch._C.DispatchKey.PrivateUse1, *a, **kw
        )
    )
    patch_mod = importlib.import_module("vllm_torchtpu.compilation.while_loop.patch")
    return patch_mod, rig


_OPAQUE_STORE: dict[int, tuple[Any, fx.GraphModule]] = {}


def _restore_opaque(key: int) -> Any:
    cls, gm = _OPAQUE_STORE[key]
    return cls(gm, key=key)


class TestPatchLifecycleAndRealTorchTpuPasses:
    def test_cpu_only_noop_and_idempotent_worker_init(
        self, monkeypatch, while_loop_dispatch_guard
    ) -> None:
        for name in [
            k for k in sys.modules if k == "torch_tpu" or k.startswith("torch_tpu.")
        ]:
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.setitem(sys.modules, "torch_tpu", None)
        monkeypatch.setattr(patch_registry, "_applied", set())
        monkeypatch.setattr(patch_registry, "_applying", set())
        monkeypatch.setattr(patch_registry, "_active_stages", set())
        monkeypatch.setattr(
            patch_registry,
            "PATCHES",
            tuple(
                p
                for p in patch_registry.PATCHES
                if p.target.endswith(":install_while_loop_patch")
            ),
        )
        patch_registry.apply("worker_init")
        assert torch._C.DispatchKey.PrivateUse1 not in while_loop_op.py_kernels

    def test_real_clone_mutated_returned_placeholders_pass(self, monkeypatch) -> None:
        from torch_tpu._internal.compile.fx_passes import (
            clone_mutated_returned_placeholders as cmrp,
        )

        orig = getattr(
            cmrp._is_returned_compute_placeholder,
            "__wrapped__",
            cmrp._is_returned_compute_placeholder,
        )
        monkeypatch.setattr(cmrp, "_is_returned_compute_placeholder", orig)
        patch_mod = importlib.import_module(
            "vllm_torchtpu.compilation.while_loop.patch"
        )

        def make_gm(*, shared_mm: bool = False, mut: str = "") -> fx.GraphModule:
            g = fx.Graph()
            s, kv, w, dirty = (
                g.placeholder("s"),
                g.placeholder("kv"),
                g.placeholder("w"),
                g.placeholder("dirty"),
            )
            for ph in (s, kv, w, dirty):
                ph.meta["val"] = torch.zeros((4, 4))
            wn = g.call_function(
                while_loop_op,
                args=(
                    fx.GraphModule({}, fx.Graph()),
                    fx.GraphModule({}, fx.Graph()),
                    (s, kv),
                    (w,),
                ),
                kwargs={"mutated_arg_indices": mut} if mut else {},
            )
            o0 = g.call_function(operator.getitem, args=(wn, 0))
            o1 = g.call_function(operator.getitem, args=(wn, 1))
            c0 = g.call_function(torch.ops.aten.clamp_.default, args=(o0, 0.0, 1.0))
            g.call_function(torch.ops.aten.add_.Tensor, args=(dirty, 1.0))
            outs = (
                (
                    o1,
                    c0,
                    g.call_function(torch.ops.aten.mm.default, args=(kv, w)),
                    kv,
                    w,
                    dirty,
                )
                if shared_mm
                else (o1, c0, kv, w, dirty)
            )
            g.output(outs)
            return fx.GraphModule({}, g)

        def cloned(gm: fx.GraphModule) -> list[str]:
            cmrp.apply(gm)
            return [
                n.args[0].name
                for n in gm.graph.nodes
                if n.op == "call_function" and n.target is torch.ops.aten.clone.default
            ]

        assert cloned(make_gm()) == ["kv", "w", "dirty"]
        patch_mod._patch_clone_mutated_returned_placeholders(cmrp)
        assert cloned(make_gm()) == ["dirty"]
        assert cloned(make_gm(mut="0")) == ["kv", "w", "dirty"]
        assert cloned(make_gm(shared_mm=True)) == ["kv", "w", "dirty"]

    def test_real_split_compiler_carry_validation_pickle_and_collectives(
        self, monkeypatch
    ) -> None:
        from functorch.compile import make_boxed_func
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch_tpu._internal.compile import compiler as compiler_mod
        from torch_tpu._internal.compile import split_compiler as sc

        orig = getattr(
            sc.SplitCompiler._compile_graph,
            "__wrapped__",
            sc.SplitCompiler._compile_graph,
        )
        monkeypatch.setattr(sc.SplitCompiler, "_compile_graph", orig)
        patch_mod = importlib.import_module(
            "vllm_torchtpu.compilation.while_loop.patch"
        )
        patch_mod._patch_split_compiler_while_carry_validation(sc)

        class _Art(compiler_mod.CompiledArtifact):
            def __init__(self, gm: fx.GraphModule, key: int | None = None) -> None:
                super().__init__()
                self._gm = gm
                if key is None:
                    key = len(_OPAQUE_STORE) + 1
                    _OPAQUE_STORE[key] = (_Art, gm)
                self._key = key

            def __call__(self, *args: Any) -> Any:
                return self._gm(*args)

            @property
            def graph_module_debug_strs(self) -> Sequence[str]:
                return ()

            @property
            def mlir_texts(self) -> Sequence[str]:
                return ()

            def updates_default_generator_state(self) -> bool:
                return False

            def resolve(self) -> None:
                return None

            @property
            def is_resolved(self) -> bool:
                return True

            def fingerprint(self) -> str:
                return f"k_{self._key}"

            def __reduce__(self) -> tuple[Any, ...]:
                return (_restore_opaque, (self._key,))

        class _Base(compiler_mod.Compiler):
            def __call__(
                self,
                gm: fx.GraphModule,
                example_inputs: Sequence[Any],
                is_fwd: bool = True,
                module_name: str | None = None,
            ) -> compiler_mod.CompiledArtifact:
                del example_inputs, is_fwd, module_name
                val_op = torch.ops.vllm_torchtpu._validate_while_carries.default
                assert not any(
                    n.op == "call_function" and n.target is val_op
                    for m in gm.modules()
                    if isinstance(m, fx.GraphModule)
                    for n in m.graph.nodes
                )
                return _Art(gm)

        split_comp = sc.SplitCompiler(_Base())

        def backend(gm: fx.GraphModule, inps: list[Any]) -> Callable[..., Any]:
            return aot_module_simplified(
                gm,
                inps,
                fw_compiler=lambda fg, fi: make_boxed_func(
                    pickle.loads(
                        pickle.dumps(
                            split_comp._compile_graph(
                                fg, fi, materialize_collectives=False
                            )
                        )
                    )
                ),
            )

        s0 = torch.tensor(0, dtype=torch.int32)
        pool = torch.ones((8, 4), dtype=torch.float32)

        @torch.compile(backend=backend, fullgraph=True)
        def topk_and_chained(
            st: torch.Tensor, logits: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            vals, idxs = torch.topk(logits, k=2, dim=-1)
            s1, v1, i1 = tpu_while_loop(
                lambda s, _v, _i: s < 2,
                lambda s, v, i: (s + 1, v + 0.5, i + 1),
                (st, vals, idxs),
            )
            return tpu_while_loop(
                lambda s, _v: s < 4, lambda s, v: (s + 1, v * 2.0), (s1, v1)
            ) + (i1,)

        cs, cv, ci = topk_and_chained(s0, torch.tensor([[1.0, 4.0, 2.0, 3.0]]))
        assert int(cs.item()) == 4 and cv.shape == (1, 2) and ci.shape == (1, 2)

        @torch.compile(backend=backend, fullgraph=True)
        def split_overlap(st: torch.Tensor, buf: torch.Tensor):
            a, b = torch.split(buf, 4, dim=0)
            return tpu_while_loop(
                lambda s, _a, _b: s < 2,
                lambda s, x, y: (s + 1, x + 1, y + 1),
                (st, a, b),
            )

        with pytest.raises(ValueError, match="storage"):
            split_overlap(s0, pool)

        @torch.compile(backend=backend, fullgraph=True)
        def nested_cond_hoisted(
            flag: torch.Tensor, x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            def loop_br(ax: torch.Tensor, ay: torch.Tensor) -> torch.Tensor:
                _, u, v = tpu_while_loop(
                    lambda k, _a, _b: k < 2,
                    lambda k, a, b: (k + 1, a + 1.0, b + 2.0),
                    (torch.zeros((), dtype=torch.int32), ax, ay),
                )
                return (u + v).sum()

            return torch.cond(flag, loop_br, lambda ax, ay: (ax + ay).sum(), (x, y))

        torch.testing.assert_close(
            nested_cond_hoisted(
                torch.tensor(True), torch.ones((4, 4)), torch.full((4, 4), 2.0)
            ),
            torch.tensor(9.0 * 16),
        )
        for flg in (True, False):
            with pytest.raises(ValueError, match="storage"):
                nested_cond_hoisted(torch.tensor(flg), pool[0:4], pool[4:8])

    def test_fx_subgraph_collective_classification(self) -> None:
        import torch_tpu  # noqa: F401

        patch_mod = importlib.import_module(
            "vllm_torchtpu.compilation.while_loop.patch"
        )
        ragged_op = torch.ops.tpu._ragged_all_to_all.default
        for op_fn, args, expected in (
            (
                torch.ops._c10d_functional.all_reduce.default,
                ("sum", list(range(8))),
                (False, frozenset({8})),
            ),
            (torch.ops._c10d_functional.wait_tensor.default, (), (False, frozenset())),
            (
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                (8, "0"),
                (True, frozenset()),
            ),
            (
                torch.ops._c10d_functional.isend.default,
                (0, "0", 1),
                (True, frozenset()),
            ),
        ):
            g = fx.Graph()
            x = g.placeholder("x")
            g.output(g.call_function(op_fn, args=(x, *args)))
            assert (
                patch_mod._inspect_fx_subgraph_collectives(fx.GraphModule({}, g))
                == expected
            )

        g_rag = fx.Graph()
        r_phs = [g_rag.placeholder(f"p{i}") for i in range(6)]
        g_rag.output(g_rag.call_function(ragged_op, args=(*r_phs, "0")))
        assert patch_mod._inspect_fx_subgraph_collectives(
            fx.GraphModule({}, g_rag)
        ) == (True, frozenset())


class TestMlirAstBuilder:
    def test_while_assembly_donation_rng_and_helpers(self) -> None:
        cond_m = textwrap.dedent(
            """
            module {
              func.func private @helper(%x: tensor<i32>) -> tensor<i32> {
                return %x : tensor<i32>
              }
              func.func public @main(
                  %s: tensor<i32>, %kv: tensor<4xf32>, %lim: tensor<i32>
              ) -> tensor<i1> {
                %c = func.call @helper(%s) : (tensor<i32>) -> tensor<i32>
                %lt = stablehlo.compare LT, %c, %lim, SIGNED
                    : (tensor<i32>, tensor<i32>) -> tensor<i1>
                return %lt : tensor<i1>
              }
            }
            """
        )
        body_m = textwrap.dedent(
            """
            module {
              func.func private @helper(%x: tensor<i32>) -> tensor<i32> {
                %one = stablehlo.constant dense<1> : tensor<i32>
                %y = stablehlo.add %x, %one : tensor<i32>
                return %y : tensor<i32>
              }
              func.func public @main(
                  %s: tensor<i32>,
                  %kv: tensor<4xf32> {jax.buffer_donor = true},
                  %rng: tensor<2xui64>,
                  %lim: tensor<i32>
              ) -> (tensor<i32>, tensor<4xf32>, tensor<2xui64>) {
                %ns = func.call @helper(%s) : (tensor<i32>) -> tensor<i32>
                return %ns, %kv, %rng : tensor<i32>, tensor<4xf32>, tensor<2xui64>
              }
            }
            """
        )
        art = _build_while_loop_mlir(
            cond_m,
            body_m,
            num_carries=2,
            num_additional_inputs=1,
            has_rng_state=True,
            num_partitions=8,
        )
        assert art.donated_carry_indices == (1,)
        mod, text = _parse_and_verify_mlir(art.normalized_mlir_text)
        assert text.count("sdy.manual_computation") == 1
        assert "tf.aliasing_output = 1 : i32" in text
        funcs = [
            op.sym_name.value
            for op in mod.body.operations
            if isinstance(op, func.FuncOp)
        ]
        assert len([f for f in funcs if f != "main"]) == 2

        art_filtered = _build_while_loop_mlir(
            cond_m,
            body_m,
            num_carries=2,
            num_additional_inputs=1,
            has_rng_state=True,
            allowed_donor_indices=set(),
        )
        assert (
            art_filtered.donated_carry_indices == ()
            and art_filtered.kernel_key != art.kernel_key
        )

    def test_stale_sharding_stripped_and_gspmd_custom_call_rejected(
        self, installed_while_loop_patch
    ) -> None:
        _, rig = installed_while_loop_patch
        sharded_cond = textwrap.dedent(
            """
            module attributes {mhlo.num_partitions = 8 : i32} {
              sdy.mesh @mesh = <["tp"=8]>
              func.func public @main(
                  %s: tensor<i32> {mhlo.sharding = "{replicated}"}
              ) -> tensor<i1> {
                %lim = stablehlo.constant dense<4> : tensor<i32>
                %c = stablehlo.compare LT, %s, %lim, SIGNED
                    : (tensor<i32>, tensor<i32>) -> tensor<i1>
                return %c : tensor<i1>
              }
            }
            """
        )
        sharded_body = textwrap.dedent(
            """
            module attributes {mhlo.num_partitions = 8 : i32} {
              sdy.mesh @mesh = <["tp"=8]>
              func.func public @main(
                  %s: tensor<i32> {sdy.sharding = #sdy.sharding<@mesh, []>}
              ) -> tensor<i32> {
                %r = sdy.reshard %s <@mesh, []> : tensor<i32>
                return %r : tensor<i32>
              }
            }
            """
        )
        rig.mlir_queue.extend([sharded_cond, sharded_body])
        s_in = _MockTpuTensor(torch.tensor(0, dtype=torch.int32))
        with rig.execution_mode.set_eager_mode(
            rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            while_loop_op(lambda s: s < 4, lambda s: (s + 1,), (s_in,), ())
        assert rig.custom_kernel_calls[-1]["device_assignment"] == []

        gspmd_body = textwrap.dedent(
            """
            module {
              func.func public @main(%s: tensor<i32>) -> tensor<i32> {
                %x = stablehlo.custom_call @Sharding(%s) {
                    mhlo.sharding = "{replicated}"
                } : (tensor<i32>) -> tensor<i32>
                return %x : tensor<i32>
              }
            }
            """
        )
        with pytest.raises(ValueError, match="Unsupported GSPMD sharding custom_call"):
            _build_while_loop_mlir(sharded_cond, gspmd_body, num_carries=1)


class TestDispatchLoweringContracts:
    def test_strides_donation_and_rng_splicing(
        self, installed_while_loop_patch
    ) -> None:
        patch_mod, rig = installed_while_loop_patch
        outer_gen = _MockGenerator.create(777, 0)
        rig.tpu_torch_compile.set_export_mode_default_device_generator(outer_gen, 0)

        step0 = _MockTpuTensor(torch.tensor(0, dtype=torch.int32))
        h_strided = _MockTpuTensor(
            torch.arange(64, dtype=torch.float32).reshape(16, 4)
        ).t()
        w_strided = _MockTpuTensor(torch.ones((16, 16), dtype=torch.float32)).t()

        seen_strides: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        with rig.execution_mode.set_eager_mode(
            rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            _, out_h = while_loop_op(
                lambda s, h, w: (seen_strides.append((h.stride(), w.stride())), s < 2)[
                    1
                ],
                lambda s, h, w: (
                    s + 1,
                    (h @ w + h.t().view(-1).mean()).t().contiguous().t(),
                ),
                (step0, h_strided),
                (w_strided,),
            )
        assert seen_strides == [((1, 4), (1, 16))]
        assert out_h.stride() == (1, 4)

        # Eager TPU loop clones non-zero-offset views of inputs.
        sm_tpu = _MockTpuTensor(torch.zeros((4,), dtype=torch.float32))
        big_tpu = _MockTpuTensor(torch.arange(8, dtype=torch.float32))
        _, out_sm, _ = while_loop_op(
            lambda s, _sm, _bg: s < 1,
            lambda s, _sm, bg: (s + 1, bg[4:8], bg + 1.0),
            (step0, sm_tpu, big_tpu),
            (),
        )
        assert (
            out_sm.untyped_storage().data_ptr() != big_tpu.untyped_storage().data_ptr()
        )

        rig.force_strides_calls.clear()
        kv_init = _MockTpuTensor(torch.zeros((4, 8), dtype=torch.bfloat16))

        def donating_body(s, kv):
            (nkv,) = rig.tpu_torch_pallas.call_custom_kernel(
                "kv_op", "k1", inputs=[kv], output_shapes=[kv], donate_argnums=[0]
            )
            return s + 1, nkv

        with rig.execution_mode.set_eager_mode(
            rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            while_loop_op(lambda s, _k: s < 2, donating_body, (step0, kv_init), ())
        assert rig.force_strides_calls == []
        assert rig.custom_kernel_calls[-1]["donate_argnums"] == [1]

        pool = _MockTpuTensor(torch.zeros((8, 4), dtype=torch.float32))
        for bad_kv in (pool.t(), pool[:4]):
            with (
                rig.execution_mode.set_eager_mode(
                    rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
                ),
                pytest.raises(ValueError, match="Donated while_loop"),
            ):
                while_loop_op(lambda s, _k: s < 2, donating_body, (step0, bad_kv), ())

        with FakeTensorMode(shape_env=ShapeEnv()) as fm:
            fa = fm.from_tensor(torch.zeros((4,)))
            assert wl_ops._tensor_storage_base_ptr(fa) is None
            assert patch_mod._compute_allowed_donor_indices(1, [fa]) == set()

        prev_state = outer_gen.state
        with rig.execution_mode.set_eager_mode(
            rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            while_loop_op(
                lambda s: s < 2,
                lambda s: (rig.advance_default_rng(0), s + 1)[1:],
                (step0,),
                (),
            )
        assert outer_gen.state.data_ptr() != prev_state.data_ptr()
        with pytest.raises(RuntimeError, match="cond_fn"):
            while_loop_op(
                lambda s: (rig.advance_default_rng(0), s < 2)[1],
                lambda s: (s + 1,),
                (step0,),
                (),
            )

    def test_ep_mesh_reconciliation_and_thread_safe_capture(
        self, installed_while_loop_patch
    ) -> None:
        patch_mod, rig = installed_while_loop_patch
        step0 = _MockTpuTensor(torch.tensor(0, dtype=torch.int32))
        ep_order = [0, 1, 2, 3, 4, 5, 6, 7]
        cond_i1 = textwrap.dedent(
            """
            module {
              func.func public @main(%s: tensor<i32>) -> tensor<i1> {
                %c = stablehlo.constant dense<true> : tensor<i1>
                return %c : tensor<i1>
              }
            }
            """
        )
        body_world_ar = textwrap.dedent(
            """
            module {
              func.func public @main(%s: tensor<i32>) -> tensor<i32> {
                %r = "stablehlo.all_reduce"(%s) ({
                ^bb0(%a: tensor<i32>, %b: tensor<i32>):
                  %sum = stablehlo.add %a, %b : tensor<i32>
                  stablehlo.return %sum : tensor<i32>
                }) {
                    replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>
                } : (tensor<i32>) -> tensor<i32>
                return %r : tensor<i32>
              }
            }
            """
        )
        body_pid = textwrap.dedent(
            """
            module {
              func.func public @main(%s: tensor<i32>) -> tensor<i32> {
                %p = stablehlo.partition_id : tensor<ui32>
                %pi = stablehlo.convert %p : (tensor<ui32>) -> tensor<i32>
                %o = stablehlo.add %s, %pi : tensor<i32>
                return %o : tensor<i32>
              }
            }
            """
        )

        def ep_body(s):
            (s1,) = rig.tpu_torch_pallas.call_custom_kernel(
                "ep_moe",
                "k_ep",
                inputs=[s],
                output_shapes=[s],
                device_assignment=ep_order,
            )
            return (s1 + 1,)

        rig.mlir_queue.extend([cond_i1, body_world_ar])
        with rig.execution_mode.set_eager_mode(
            rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
        ):
            while_loop_op(lambda s: s < 2, ep_body, (step0,), ())
        assert rig.custom_kernel_calls[-1]["device_assignment"] == ep_order

        rig.mlir_queue.extend([cond_i1, body_pid])
        with (
            rig.execution_mode.set_eager_mode(
                rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
            ),
            pytest.raises(ValueError, match="Mismatched device_assignment"),
        ):
            while_loop_op(lambda s: s < 2, ep_body, (step0,), ())

        # Unresolvable or empty default device_assignment fails closed when
        # multi-device ops are present instead of falling back to 1 partition.
        rig.default_device_assignment = []
        rig.mlir_queue.extend([cond_i1, body_world_ar])
        with (
            rig.execution_mode.set_eager_mode(
                rig.execution_mode.EagerMode._INTERNAL_COMPILE_FX_GRAPH
            ),
            pytest.raises(ValueError, match="device_assignment"),
        ):
            while_loop_op(lambda s: s < 2, lambda s: (s + 1,), (step0,), ())
        rig.default_device_assignment = [0, 1, 4, 5, 6, 7, 2, 3]

        captured_by_thread: dict[str, list[list[int]]] = {}

        def _worker(tag: str, da: list[int]) -> None:
            with patch_mod._capture_inner_device_assignments(
                rig.tpu_torch_pallas
            ) as cap:
                rig.tpu_torch_pallas.call_custom_kernel(
                    tag,
                    tag,
                    inputs=[step0],
                    output_shapes=[step0],
                    device_assignment=da,
                )
                captured_by_thread[tag] = list(cap)

        threads = [
            threading.Thread(target=_worker, args=("ep", ep_order)),
            threading.Thread(
                target=_worker, args=("pg", rig.default_device_assignment)
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert captured_by_thread == {
            "ep": [ep_order],
            "pg": [rig.default_device_assignment],
        }
