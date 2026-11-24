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
"""Bridge between `megakernel` (`make_flat_pooled_step`) and PyTorch/vLLM.

Provides:
- `build_pooled_step_op`: wraps `cache.make_flat_pooled_step` into a
  `sharded_jax_callable` + `@torch.library.custom_op` (taking `list[Tensor]`
  for `weights` and `pools` to stay under the PyTorch dispatcher's 64-argument
  schema limit), with automatic CPU `interpret=True` execution when called with
  CPU `torch.Tensor` operands in unit tests.
- `execute_pooled_step_inplace`: invokes the custom op and copies the donated
  pool outputs back into `pools` via `dst.copy_(src)`, preserving `dst.data_ptr()`
  (Raiden pointer identity) and folding into an XLA `input_output_alias` /
  `tf.aliasing_output` under `torch.compile(backend="tpu")`.
- `verify_pooled_step_hlo`: lowers (or compiles when `mesh` holds real
  `jax.Device` instances on TPU) the JAX pooled step and validates pool buffer
  donation (`input_output_alias`) and absence of full-pool copies via
  `cache.check_pooled_step_hlo`. Under `torch_tpu`'s `_PallasDevice` mesh,
  `StaticCompiler` owns XLA compilation, so `verify_pooled_step_hlo` checks the
  lowered HLO while `verify_torchtpu_compiled_step_ir` checks the outer
  `torch.compile(backend="tpu")` graph.
- `verify_torchtpu_compiled_step_ir`: inspects the compiled `torch_tpu` forward
  graph's StableHLO MLIR / HLO (`TpuCompileDebug.stablehlo_forward_text`) to
  verify that `dst.copy_(src)` on all pools folded into `jax.buffer_donor = true` /
  `tf.aliasing_output` / `input_output_alias` with zero full-pool copies.
- `export_pooled_step`: offline inspection/serialization helper that exports the
  StableHLO module and aliasing manifest (`PooledStepExport`) via `jax.export`
  (serving traces directly through `build_pooled_step_op` / `JaxCallable`).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import cache
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.pool.config import cache_exec
from vllm_torchtpu.megakernel.spec import ModelSpec


def _leaf_output_specs(output_partition_specs: Any) -> list[P]:
    """Flatten `output_partition_specs` to one `PartitionSpec` per output leaf."""
    return jax.tree.leaves(output_partition_specs, is_leaf=lambda s: isinstance(s, P))


def sharded_jax_callable(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[P],
    output_partition_specs: Any,
    donate_argnums: Sequence[int] | None = None,
) -> Any:
    """Wrap an SPMD JAX function over `mesh` into a `torch_tpu` `JaxCallable`.

    Self-contained so `exported_bridge` does not depend on any `vllm` or
    `vllm_torchtpu.distributed` modules when extracted to a standalone repo.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name {name} must be 'namespace::op', with one '::'.")

    init_sig = inspect.signature(pallas_impl.JaxCallable.__init__)
    if "output_partition_specs" not in init_sig.parameters:
        raise RuntimeError(
            "torch_tpu.JaxCallable.__init__ does not support 'output_partition_specs'; "
            "upgrade torch_tpu to use sharded_jax_callable."
        )

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, P),
    )

    jit_fn = jax.jit(
        fn,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
        out_shardings=out_shardings,
        keep_unused=True,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}",
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(output_partition_specs),
        },
    )
    return pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=tuple(_leaf_output_specs(output_partition_specs)),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )


@dataclass(frozen=True)
class PooledStepExport:
    """Offline serialized StableHLO export and aliasing manifest for a pooled step."""

    op_name: str
    exported: Any
    mlir_module_serialized: bytes
    in_specs: tuple[P, ...]
    out_specs: tuple[P, ...]
    donate_argnums: tuple[int, ...]
    input_output_aliases: dict[int, int]


_OP_COUNTER = 0


def _torch_dtype_to_jax(dt: torch.dtype) -> Any:
    if dt == torch.float8_e4m3fn:
        return jnp.float8_e4m3fn
    if dt == torch.bfloat16:
        return jnp.bfloat16
    if dt == torch.float32:
        return jnp.float32
    if dt == torch.int32:
        return jnp.int32
    if dt == torch.int64:
        return jnp.int64
    raise TypeError(f"Unsupported torch.dtype for megakernel bridge: {dt}")


def _torch_cpu_to_jax(
    t: torch.Tensor, sharding: NamedSharding | None = None
) -> jax.Array:
    """Convert a CPU `torch.Tensor` to a `jax.Array` without precision loss."""
    t_c = t.detach().cpu().contiguous()
    if t_c.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t_c.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t_c.dtype == torch.bfloat16:
        arr = jnp.asarray(t_c.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t_c.numpy())
    if sharding is not None:
        arr = jax.device_put(arr, sharding)
    return arr


def _jax_to_torch_cpu(a: jax.Array, like: torch.Tensor) -> torch.Tensor:
    """Convert a `jax.Array` to a CPU `torch.Tensor` matching `like.dtype`."""
    np_arr = np.asarray(a)
    if like.dtype == torch.float8_e4m3fn:
        out = torch.from_numpy(np_arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    elif like.dtype == torch.bfloat16:
        out = torch.from_numpy(np_arr.view(np.uint16).copy()).view(torch.bfloat16)
    else:
        out = torch.from_numpy(np_arr.copy()).to(like.dtype)
    return out


def make_typed_step_wrapper(
    fn_name: str,
    step_fn: Callable[..., Any],
    num_inputs: int,
    num_outputs: int,
) -> Callable[..., Any]:
    """Synthesize a fixed-arity `jax.Array`-annotated wrapper for `JaxCallable`.

    `torch_tpu`'s `JaxCallable` inspects the function's explicit positional
    parameters and rejects `*args` signatures.
    """
    args = [f"x{i}" for i in range(num_inputs)]
    ret = "tuple[" + ", ".join(["jax.Array"] * num_outputs) + "]"
    src = (
        f"def {fn_name}({', '.join(a + ': jax.Array' for a in args)}) -> {ret}:\n"
        f"    return _step_fn({', '.join(args)})\n"
    )
    ns: dict[str, Any] = {"jax": jax, "_step_fn": step_fn}
    exec(compile(src, f"<{fn_name}>", "exec"), ns)  # noqa: S102
    return ns[fn_name]


def _build_global_avals(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
) -> list[jax.ShapeDtypeStruct]:
    """Construct global `jax.ShapeDtypeStruct` descriptors for HLO lowering/export."""
    r_cnt = int(mesh.devices.size)
    g: Any = spec.config
    b_tok = ex.B
    t_tok = ex.T
    k_ckpt = int(ckpt_cols) if ckpt_cols is not None else t_tok
    nseq = ex.NS
    m_rows = ex.M
    g_per_att = max(1, int(getattr(g, "att_every", 4)) - 1)

    def _global_sds(t: torch.Tensor) -> jax.ShapeDtypeStruct:
        dt = _torch_dtype_to_jax(t.dtype)
        return jax.ShapeDtypeStruct((r_cnt * int(t.shape[0]),) + tuple(t.shape[1:]), dt)

    return [
        jax.ShapeDtypeStruct((b_tok,), jnp.int32),
        jax.ShapeDtypeStruct((r_cnt * m_rows, int(g.H)), jnp.bfloat16),
        *(_global_sds(w) for w in w_flat),
        *(_global_sds(p) for p in raw_pools),
        jax.ShapeDtypeStruct((nseq, int(max_pages_per_seq)), jnp.int32),
        jax.ShapeDtypeStruct((g_per_att, nseq, k_ckpt), jnp.int32),
        jax.ShapeDtypeStruct((g_per_att, nseq), jnp.int32),
    ]


def verify_pooled_step_hlo(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    compile_hlo: bool | None = None,
) -> tuple[bool, str]:
    """Lower or compile `make_flat_pooled_step` and verify pool donation in HLO.

    When `compile_hlo` is `None` (default), compiles post-optimization HLO via
    JAX when `mesh` contains real `jax.Device` instances on TPU
    (`jax.default_backend() == "tpu" and not ex.interpret`), and lowers pre-XLA
    HLO when `mesh` uses `torch_tpu`'s `_PallasDevice` mock devices (where XLA
    compilation is owned by `torch_tpu` `StaticCompiler` and verified via
    `verify_torchtpu_compiled_step_ir`). Passing `compile_hlo=True` with a
    non-`jax.Device` mesh raises `MegakernelConfigError`.
    """
    has_real_jax_devices = bool(
        mesh.devices.size > 0 and isinstance(mesh.devices.flat[0], jax.Device)
    )
    if compile_hlo is True and not has_real_jax_devices:
        dev_type = (
            type(mesh.devices.flat[0]).__name__ if mesh.devices.size > 0 else "empty"
        )
        raise MegakernelConfigError(
            f"compile_hlo=True requires a Mesh of real jax.Device instances "
            f"(got {dev_type}); under torch_tpu's _PallasDevice mesh, XLA "
            f"compilation is performed by StaticCompiler and verified via "
            f"verify_torchtpu_compiled_step_ir."
        )
    if compile_hlo is None:
        compile_hlo = bool(
            has_real_jax_devices and jax.default_backend() == "tpu" and not ex.interpret
        )
    else:
        compile_hlo = bool(compile_hlo)
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    avals = _build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    j_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    lowered = j_fn.lower(*avals)
    hlo_text = lowered.compile().as_text() if compile_hlo else lowered.as_text("hlo")
    num_pages = int(raw_pools[0].shape[0])
    page_size = int(raw_pools[0].shape[-1])
    g: Any = spec.config
    kvh = int(g.nkv(ex.R)) if hasattr(g, "nkv") else 1
    hd = int(getattr(g, "HD", 256))
    pb = spec.cache_backend
    k_names = (
        pb.kernel_names
        if pb is not None
        else ("gather_gdn_states", "scatter_pool_states")
    )
    return cache.check_pooled_step_hlo(
        hlo_text,
        page_size=page_size,
        kvh=kvh,
        hd=hd,
        min_pool_pages=max(16, num_pages),
        fused_pool_io=cache_exec(ex).FUSED,
        kernel_names=k_names,
    )


def verify_torchtpu_compiled_step_ir(
    ir_text: str,
    *,
    num_pages: int,
    page_size: int = 128,
    kvh: int = 1,
    hd: int = 256,
    num_pools: int = 15,
    hlo_text: str | None = None,
) -> tuple[bool, str]:
    """Verify that a compiled `torch_tpu` forward step donates and aliases all pools.

    Inspects the StableHLO MLIR (`TpuCompileDebug.stablehlo_forward_text`) or
    HLO text produced by `torch.compile(backend="tpu")` for the outer step graph
    that invokes `execute_pooled_step_inplace` (`op(...)` + `dst.copy_(src)`).

    Note on `torch_tpu` lowering: `JaxCallable` passes `donate_argnums` to C++
    `PyCallCustomKernel`, which records `DeferredOpParams::donated_inputs`.
    During `fx_to_mlir`, `Traversal::BuildMlirModule` collects the `mlir::Value`
    of every custom-kernel donated input and calls `AnnotateBufferDonations` on
    matching `@main` parameters (independent of `StaticCompiler`'s explicit
    `donated_inputs` argument). Same-shape contiguous `dst.copy_(src)` calls are
    lowered by `CopyTpuToTpu` as buffer-handle rebinds (`GetBuffer(src)`) rather
    than MLIR copy ops.

    Checks performed:
    1. Confirms a custom Pallas kernel call op (`stablehlo.custom_call`,
       `mhlo.custom_call`, or `custom-call(`) is present.
    2. For StableHLO MLIR, inspects the outer `@main` signature and confirms at
       least `num_pools` pool arguments carry `jax.buffer_donor = true` (or
       `tf.aliasing_output = <idx>`), ensuring outer pool inputs are donated
       rather than passing vacuously on an inlined submodule.
    3. For HLO text (`ir_text` when it is an `HloModule`, or `hlo_text` when
       supplied alongside MLIR), confirms `HloModule ... input_output_alias`
       contains at least `num_pools` alias entries.
    4. Confirms no `copy`, `copy-start`, `dynamic_update_slice`, or
       `as_strided_copy_inverse` materializes a full-pool buffer.
    """
    if not ir_text or not ir_text.strip():
        return False, "empty torch_tpu compiled IR"

    has_custom_call = (
        "stablehlo.custom_call" in ir_text
        or "mhlo.custom_call" in ir_text
        or "custom-call(" in ir_text
    )
    if not has_custom_call:
        return False, "missing custom kernel call in torch_tpu compiled IR"

    has_mlir_main = "func.func" in ir_text
    has_hlo_module = "HloModule" in ir_text
    if not has_mlir_main and not has_hlo_module:
        return False, "unrecognized torch_tpu IR format (missing func.func / HloModule)"

    if has_mlir_main:
        main_match = re.search(
            r"func\.func\b[^{]*@main\s*\((.*?)\)\s*->", ir_text, re.DOTALL
        )
        if main_match is None:
            return False, "missing func.func @main signature in torch_tpu MLIR"
        main_params = main_match.group(1)
        pool_shape_str = f"{num_pages}x{2 * kvh}x{hd // 4}x4x{page_size}xf8E4M3FN"
        pool_arg_pat = re.compile(
            rf"%arg\d+\s*:\s*tensor<{pool_shape_str}>(?:\s*(\{{[^}}]*\}}))?"
        )
        n_donated_main = 0
        for m in pool_arg_pat.finditer(main_params):
            attr = m.group(1) or ""
            if "jax.buffer_donor = true" in attr or re.search(
                r"tf\.aliasing_output\s*=\s*\d+", attr
            ):
                n_donated_main += 1
        if n_donated_main < num_pools:
            return (
                False,
                f"expected >= {num_pools} donated pool arguments "
                f"(jax.buffer_donor = true) on @main in torch_tpu IR, "
                f"found {n_donated_main}",
            )

    hlo_sources: list[str] = []
    if has_hlo_module:
        hlo_sources.append(ir_text)
    if hlo_text and hlo_text.strip():
        hlo_sources.append(hlo_text)

    for hlo_src in hlo_sources:
        hlo_header = hlo_src.split("ENTRY", 1)[0]
        if "input_output_alias" not in hlo_header:
            return False, "missing input_output_alias in torch_tpu HLO header"
        alias_entries = re.findall(r"(?:\{\d*\}\s*:|\{\{\d*\}\s*,\s*\d+)", hlo_header)
        if len(alias_entries) < num_pools:
            return (
                False,
                f"expected >= {num_pools} entries in HloModule input_output_alias, "
                f"found {len(alias_entries)}",
            )

    mlir_pool_pat = re.compile(rf"\b(\d+)x{2 * kvh}x{hd // 4}x4x{page_size}xf8E4M3FN\b")
    hlo_pool_suffix = f",{2 * kvh},{hd // 4},4,{page_size}]"
    copy_ops = (
        "stablehlo.copy",
        "mhlo.copy",
        "copy-start(",
        "dynamic_update_slice",
        "dynamic-update-slice(",
        "as_strided_copy_inverse",
    )
    texts_to_scan = [ir_text] + ([hlo_text] if hlo_text and hlo_text.strip() else [])
    for txt in texts_to_scan:
        for line in txt.splitlines():
            if "func.func" in line or "ENTRY" in line or "custom_call" in line:
                continue
            is_copy_line = any(op in line for op in copy_ops) or (
                "= f8e4m3fn[" in line and "copy(" in line
            )
            if not is_copy_line:
                continue
            for m in mlir_pool_pat.finditer(line):
                if int(m.group(1)) >= num_pages:
                    return (
                        False,
                        f"unexpected full-pool copy in torch_tpu IR: {line.strip()}",
                    )
            if hlo_pool_suffix in line:
                prefix = line.split(hlo_pool_suffix, 1)[0]
                dim_str = prefix.rsplit("[", 1)[-1].rsplit(",", 1)[-1].strip()
                if not dim_str.isdigit() or int(dim_str) >= num_pages:
                    return (
                        False,
                        f"unexpected full-pool copy in torch_tpu HLO: {line.strip()}",
                    )
    return True, "ok"


def export_pooled_step(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    op_name: str = "megakernel_runtime::exported_pooled_step",
    platforms: Sequence[str] = ("tpu",),
) -> PooledStepExport:
    """Export `make_flat_pooled_step` to a serialized StableHLO module.

    Note: At serving time, `build_pooled_step_op` traces through
    `sharded_jax_callable` (`JaxCallable`). This helper is provided for offline
    StableHLO inspection and serialization tests.
    """
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=True
    )
    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    avals = _build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    j_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )
    try:
        with jax._src.config.export_ignore_forward_compatibility(True):
            exported: Any = jax.export.export(
                j_fn,
                platforms=list(platforms),
                disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
            )(*avals)
        mlir_bytes = exported.mlir_module_serialized
    except NotImplementedError:
        if not ex.interpret:
            raise
        lowered = j_fn.lower(*avals)
        exported = lowered
        mlir_bytes = lowered.as_text("stablehlo").encode("utf-8")
    n_w = len(w_flat)
    n_p = len(raw_pools)
    aliases = {2 + n_w + i: 1 + i for i in range(n_p)}
    return PooledStepExport(
        op_name=op_name,
        exported=exported,
        mlir_module_serialized=mlir_bytes,
        in_specs=in_specs,
        out_specs=out_specs,
        donate_argnums=donate_argnums,
        input_output_aliases=aliases,
    )


def build_pooled_step_op(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    *,
    op_name: str | None = None,
    donate_pools: bool = True,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """Build a PyTorch custom op wrapping `cache.make_flat_pooled_step`.

    Returns `(op, in_specs, out_specs, donate_argnums)` where `op` takes
    `(pos, hidden, page_table, ckpt_indices, read_offsets, weights, pools)`.
    """
    global _OP_COUNTER
    step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=donate_pools
    )
    _OP_COUNTER += 1
    if op_name is None:
        n_pages = cache_exec(ex).num_pages
        op_name = f"megakernel_runtime::step_t{ex.T}_b{ex.B}_p{n_pages}_{_OP_COUNTER}"
    elif "::" not in op_name:
        op_name = f"megakernel_runtime::{op_name}"

    fn_name = op_name.replace("::", "_")
    typed_fn = make_typed_step_wrapper(
        fn_name, step_fn, num_inputs=len(in_specs), num_outputs=len(out_specs)
    )
    try:
        flat_callable: Any = sharded_jax_callable(
            op_name,
            typed_fn,
            mesh=mesh,
            input_partition_specs=in_specs,
            output_partition_specs=out_specs,
            donate_argnums=donate_argnums,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("torch_tpu"):
            flat_callable = None
        else:
            raise

    in_sh = tuple(NamedSharding(mesh, s) for s in in_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in out_specs)
    cpu_jit_fn = jax.jit(
        step_fn,
        in_shardings=in_sh,
        out_shardings=out_sh,
        donate_argnums=donate_argnums,
        keep_unused=True,
    )

    @torch.library.custom_op(op_name, mutates_args=())
    def op(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        all_tensors = [
            pos,
            hidden,
            *weights,
            *pools,
            page_table,
            ckpt_indices,
            read_offsets,
        ]
        if pos.device.type == "cpu":
            # CPU interpret execution path for unit tests without a TPU host.
            jax_inputs = [
                _torch_cpu_to_jax(t, sh)
                for t, sh in zip(all_tensors, in_sh, strict=True)
            ]
            jax_outs = cpu_jit_fn(*jax_inputs)
            like_tensors = [hidden, *pools]
            return [
                _jax_to_torch_cpu(o, lk)
                for o, lk in zip(jax_outs, like_tensors, strict=True)
            ]
        if flat_callable is None:
            raise RuntimeError(
                "torch_tpu is required to execute build_pooled_step_op on TPU"
            )
        return list(flat_callable(*all_tensors))

    @op.register_fake
    def _(
        pos: torch.Tensor,
        hidden: torch.Tensor,
        page_table: torch.Tensor,
        ckpt_indices: torch.Tensor,
        read_offsets: torch.Tensor,
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [torch.empty_like(hidden)] + [torch.empty_like(p) for p in pools]

    return op, in_specs, out_specs, donate_argnums


def execute_pooled_step_inplace(
    op: Callable[..., list[torch.Tensor]],
    pos: torch.Tensor,
    hidden: torch.Tensor,
    page_table: torch.Tensor,
    ckpt_indices: torch.Tensor,
    read_offsets: torch.Tensor,
    weights: Sequence[torch.Tensor],
    pools: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Execute `op`, copy donated outputs into `pools` in-place, and return `hid_out`.

    Preserves `pool.data_ptr()` and `pool.storage().data_ptr()` on every pool
    tensor so `raiden`-pinned buffers remain valid without re-registration.
    """
    outs = op(
        pos,
        hidden,
        page_table,
        ckpt_indices,
        read_offsets,
        list(weights),
        list(pools),
    )
    n_p = len(pools)
    for dst, src in zip(pools, outs[1 : 1 + n_p]):
        dst.copy_(src)
    return outs[0]
