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
- `export_pooled_step`: offline inspection/serialization helper that exports the
  StableHLO module and aliasing manifest (`PooledStepExport`) via `jax.export`
  (serving traces directly through `build_pooled_step_op` / `JaxCallable`).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import cache
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.pool.config import cache_exec
from vllm_torchtpu.megakernel.runtime import build_global_avals
from vllm_torchtpu.megakernel.spec import ModelSpec


def _leaf_output_specs(output_partition_specs: Any) -> list[P]:
    """Flatten `output_partition_specs` to one `PartitionSpec` per output leaf."""
    return jax.tree.leaves(output_partition_specs, is_leaf=lambda s: isinstance(s, P))


def _find_matching_delim(s: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    """Return matching `close_ch` index for `s[open_idx]`, skipping strings."""
    depth = 0
    in_str = False
    i = open_idx
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError(f"Unmatched {open_ch!r} at index {open_idx}")


def _find_func_body_open(s: str, after_params_idx: int) -> int:
    """Return index of `{` opening a `func.func` body after its parameter list `)`."""
    in_str = False
    i = after_params_idx
    n = len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "(":
            i = _find_matching_delim(s, i, "(", ")")
        elif ch == "{":
            prefix = s[after_params_idx:i].rstrip()
            if prefix.endswith("attributes"):
                i = _find_matching_delim(s, i, "{", "}")
            else:
                return i
        i += 1
    raise ValueError(f"Missing func.func body '{{' after index {after_params_idx}")


def _rewrite_calls_to_callees(
    body: str,
    *,
    callees: set[str],
    fn_rank_ssa: str,
    global_rank_ty: str,
) -> str:
    """Append `fn_rank_ssa` (`global_rank_ty`) to calls targeting `callees`."""
    if not callees:
        return body
    callee_alt = "|".join(re.escape(c) for c in sorted(callees))
    pat = re.compile(rf"\b(?:func\.)?call\s+@({callee_alt})\s*\(")
    cursor = 0
    while True:
        m = pat.search(body, cursor)
        if m is None:
            break
        args_open = m.end() - 1
        args_close = _find_matching_delim(body, args_open, "(", ")")
        colon_idx = body.index(":", args_close)
        ty_open = body.index("(", colon_idx)
        ty_close = _find_matching_delim(body, ty_open, "(", ")")
        ty_sep = ", " if body[ty_open + 1 : ty_close].strip() else ""
        arg_sep = ", " if body[args_open + 1 : args_close].strip() else ""
        ty_ins = f"{ty_sep}{global_rank_ty}"
        arg_ins = f"{arg_sep}{fn_rank_ssa}"
        body = body[:ty_close] + ty_ins + body[ty_close:]
        body = body[:args_close] + arg_ins + body[args_close:]
        cursor = ty_close + len(ty_ins) + len(arg_ins) + 1
    return body


def _rewrite_sdy_manual_blocks(
    body: str,
    *,
    fn_rank_ssa: str,
    global_rank_ty: str,
    mesh_sym: str,
) -> str:
    """Thread `fn_rank_ssa` into every `sdy.manual_computation` using `partition_id`."""
    needle = "sdy.manual_computation"
    cursor = 0
    uid = 0
    while True:
        idx = body.find(needle, cursor)
        if idx < 0:
            break
        op_open = body.index("(", idx + len(needle))
        op_close = _find_matching_delim(body, op_open, "(", ")")
        in_sh_key = body.index("in_shardings=", op_close)
        in_sh_open = body.index("[", in_sh_key)
        in_sh_close = _find_matching_delim(body, in_sh_open, "[", "]")
        man_key = body.index("manual_axes=", in_sh_close)
        man_open = body.index("{", man_key)
        man_close = _find_matching_delim(body, man_open, "{", "}")
        blk_open = body.index("(", man_close)
        blk_close = _find_matching_delim(body, blk_open, "(", ")")
        reg_open = body.index("{", blk_close)
        reg_close = _find_matching_delim(body, reg_open, "{", "}")

        reg_text = body[reg_open : reg_close + 1]
        if "stablehlo.partition_id" not in reg_text:
            cursor = idx + len(needle)
            continue

        k = reg_close + 1
        while k < len(body) and body[k].isspace():
            k += 1
        if k >= len(body) or body[k] != ":":
            raise ValueError(
                "Expected functional type ':' after sdy.manual_computation body"
            )
        ty_open = body.index("(", k)
        ty_close = _find_matching_delim(body, ty_open, "(", ")")

        axes = re.findall(r'"([^"]+)"', body[man_open : man_close + 1])
        if not axes:
            axes = ["tp"]
        axes_inner = ", ".join(f'"{a}"' for a in axes)
        rank_sharding = f"<@{mesh_sym}, [{{{axes_inner}}}]>"
        local_arg = f"%arg_tp_rank_local_{uid}"

        def _sub_pid(m: re.Match[str], arg_name: str = local_arg) -> str:
            nonlocal uid
            uid += 1
            indent, lhs = m.group(1), m.group(2)
            tmp = f"%tp_rank_i32_{uid}"
            return (
                f"{indent}{tmp} = stablehlo.reshape {arg_name} : "
                f"(tensor<1xi32>) -> tensor<i32>\n"
                f"{indent}{lhs} = stablehlo.convert {tmp} : "
                f"(tensor<i32>) -> tensor<ui32>"
            )

        new_reg_text = re.sub(
            r"([ \t]*)(%[\w\d_.$]+)\s*=\s*stablehlo\.partition_id\s*:\s*tensor<ui32>"
            r"(?:\s*loc\([^\n]*\))?",
            _sub_pid,
            reg_text,
        )

        ty_ins = (
            f", {global_rank_ty}"
            if body[ty_open + 1 : ty_close].strip()
            else global_rank_ty
        )
        blk_ins = (
            f", {local_arg}: tensor<1xi32>"
            if body[blk_open + 1 : blk_close].strip()
            else f"{local_arg}: tensor<1xi32>"
        )
        sh_ins = (
            f", {rank_sharding}"
            if body[in_sh_open + 1 : in_sh_close].strip()
            else rank_sharding
        )
        op_ins = (
            f", {fn_rank_ssa}" if body[op_open + 1 : op_close].strip() else fn_rank_ssa
        )

        end_after_splice = (
            ty_close
            + len(ty_ins)
            + (len(new_reg_text) - len(reg_text))
            + len(blk_ins)
            + len(sh_ins)
            + len(op_ins)
        )
        body = body[:ty_close] + ty_ins + body[ty_close:]
        body = body[:reg_open] + new_reg_text + body[reg_close + 1 :]
        body = body[:blk_close] + blk_ins + body[blk_close:]
        body = body[:in_sh_close] + sh_ins + body[in_sh_close:]
        body = body[:op_close] + op_ins + body[op_close:]
        cursor = end_after_splice
    return body


def _rewrite_partition_id_with_rank_arg(mlir_text: str) -> str:
    """Replace `stablehlo.partition_id` using `@main`'s trailing `tensor<Rxi32>` arg.

    `torch_tpu` lowers `JaxCallable` modules via `PartitionGlobalToLocal`, which
    runs XLA `SpmdPartitioner` over the exported StableHLO and rejects
    `stablehlo.partition_id`. When `mesh.devices.size > 1`, `@main` is traced
    with a trailing `P("tp")`-sharded `int32[R]` rank vector (`[r]` per device),
    and this pass threads that argument into `sdy.manual_computation` blocks in
    place of `stablehlo.partition_id`.
    """
    if "stablehlo.partition_id" not in mlir_text:
        return mlir_text

    mesh_m = re.search(r"sdy\.mesh\s+@([\w\d_.$]+)\s*=", mlir_text)
    mesh_sym = mesh_m.group(1) if mesh_m is not None else "mesh"

    fn_pat = re.compile(
        r"(?m)^[ \t]*func\.func\s+(?:(?:public|private)\s+)?@([\w\d_.$]+)\s*\("
    )
    funcs: list[tuple[str, int, int, int, int]] = []
    prev_end = 0
    for m in fn_pat.finditer(mlir_text):
        if m.start() < prev_end:
            continue
        fn_name = m.group(1)
        p_open = m.end() - 1
        p_close = _find_matching_delim(mlir_text, p_open, "(", ")")
        b_open = _find_func_body_open(mlir_text, p_close + 1)
        b_close = _find_matching_delim(mlir_text, b_open, "{", "}")
        funcs.append((fn_name, p_open, p_close, b_open, b_close))
        prev_end = b_close + 1

    main_entry = next((f for f in funcs if f[0] == "main"), None)
    if main_entry is None:
        raise ValueError("Missing @main in exported StableHLO module")
    main_params = mlir_text[main_entry[1] + 1 : main_entry[2]]
    param_matches = list(re.finditer(r"(%arg\d+)\s*:\s*(tensor<[^>]+>)", main_params))
    if not param_matches or not re.fullmatch(
        r"tensor<\d+xi32>", param_matches[-1].group(2)
    ):
        raise ValueError("Expected trailing tensor<Rxi32> rank parameter on @main")
    main_rank_ssa = param_matches[-1].group(1)
    global_rank_ty = param_matches[-1].group(2)

    bodies = {
        name: mlir_text[b_open : b_close + 1] for name, _, _, b_open, b_close in funcs
    }
    direct_pid_fns = {
        name for name, b_txt in bodies.items() if "stablehlo.partition_id" in b_txt
    }
    needs_rank = set(direct_pid_fns)
    changed = True
    while changed:
        changed = False
        for name, b_txt in bodies.items():
            if name in needs_rank:
                continue
            for target in list(needs_rank):
                if re.search(rf"\b(?:func\.)?call\s+@{re.escape(target)}\s*\(", b_txt):
                    needs_rank.add(name)
                    changed = True
                    break

    callees = needs_rank - {"main"}
    for name, p_open, p_close, b_open, b_close in reversed(funcs):
        if name not in needs_rank:
            continue
        fn_rank_ssa = main_rank_ssa if name == "main" else "%arg_tp_rank"
        new_body = bodies[name]
        if callees:
            new_body = _rewrite_calls_to_callees(
                new_body,
                callees=callees,
                fn_rank_ssa=fn_rank_ssa,
                global_rank_ty=global_rank_ty,
            )
        if name in direct_pid_fns:
            new_body = _rewrite_sdy_manual_blocks(
                new_body,
                fn_rank_ssa=fn_rank_ssa,
                global_rank_ty=global_rank_ty,
                mesh_sym=mesh_sym,
            )
        mlir_text = mlir_text[:b_open] + new_body + mlir_text[b_close + 1 :]
        if name != "main":
            p_ins = (
                f", %arg_tp_rank: {global_rank_ty}"
                if mlir_text[p_open + 1 : p_close].strip()
                else f"%arg_tp_rank: {global_rank_ty}"
            )
            mlir_text = mlir_text[:p_close] + p_ins + mlir_text[p_close:]

    if "stablehlo.partition_id" in mlir_text:
        raise RuntimeError(
            "Failed to eliminate all stablehlo.partition_id ops from StableHLO"
        )
    return mlir_text


def _sanitize_exported_partition_id(exported: Any) -> Any:
    """Rewrite `stablehlo.partition_id` in `jax.export.Exported` and re-serialize."""
    mlir_text = exported.mlir_module()
    if "stablehlo.partition_id" not in mlir_text:
        return exported
    from jax._src.export import _export as jax_export_internal
    from jax._src.interpreters import mlir as jax_mlir

    rewritten = _rewrite_partition_id_with_rank_arg(mlir_text)
    with jax_mlir.make_ir_context():
        ir_mod = jax_mlir.ir.Module.parse(rewritten)
        ir_mod.operation.verify()
        new_bytes = jax_export_internal._module_to_bytecode(ir_mod)
    return replace(exported, mlir_module_serialized=new_bytes)


def _rank_partition_spec(mesh: jax.sharding.Mesh) -> P:
    """Return 1D `PartitionSpec` sharding a length-`R` rank vector 1-per-device."""
    if len(mesh.axis_names) == 1:
        return P(mesh.axis_names[0])
    return P(tuple(mesh.axis_names))


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
    callable_obj = pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        output_partition_specs=tuple(_leaf_output_specs(output_partition_specs)),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )
    if int(mesh.devices.size) > 1:
        orig_exported = callable_obj.exported

        def _sanitized_exported(*a: Any, **kw: Any) -> Any:
            return _sanitize_exported_partition_id(orig_exported(*a, **kw))

        callable_obj.exported = _sanitized_exported
    return callable_obj


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
    if dt == torch.uint8:
        return jnp.uint8
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
    avals = build_global_avals(
        mesh,
        spec,
        ex,
        w_flat,
        raw_pools,
        max_pages_per_seq=max_pages_per_seq,
        ckpt_cols=ckpt_cols,
    )
    has_multi_rank = int(mesh.devices.size) > 1
    if has_multi_rank:

        def _step_for_export(*args: Any) -> Any:
            return step_fn(*args[:-1])

        export_fn: Callable[..., Any] = _step_for_export
        export_in_sh = (*in_sh, NamedSharding(mesh, _rank_partition_spec(mesh)))
        export_avals = [
            *avals,
            jax.ShapeDtypeStruct((int(mesh.devices.size),), jnp.int32),
        ]
    else:
        export_fn = step_fn
        export_in_sh = in_sh
        export_avals = avals
    j_fn = jax.jit(
        export_fn,
        in_shardings=export_in_sh,
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
            )(*export_avals)
        if has_multi_rank:
            exported = _sanitize_exported_partition_id(exported)
        mlir_bytes = exported.mlir_module_serialized
    except NotImplementedError:
        if not ex.interpret:
            raise
        lowered = j_fn.lower(*export_avals)
        mlir_str = lowered.as_text("stablehlo")
        if has_multi_rank:
            mlir_str = _rewrite_partition_id_with_rank_arg(mlir_str)
        exported = lowered
        mlir_bytes = mlir_str.encode("utf-8")
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
    tp_rank: int = 0,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """Build a PyTorch custom op wrapping `cache.make_flat_pooled_step`.

    Returns `(op, in_specs, out_specs, donate_argnums)` where `op` takes
    `(pos, hidden, page_table, ckpt_indices, read_offsets, weights, pools)`
    plus an optional `tp_rank_1d` 1-element `int32` device tensor when
    `mesh.devices.size > 1`.
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

    has_multi_rank = int(mesh.devices.size) > 1
    if has_multi_rank:

        def _step_for_tpu(*args: Any) -> Any:
            return step_fn(*args[:-1])

        tpu_step_fn: Callable[..., Any] = _step_for_tpu
        tpu_in_specs: tuple[P, ...] = (*in_specs, _rank_partition_spec(mesh))
    else:
        tpu_step_fn = step_fn
        tpu_in_specs = in_specs

    fn_name = op_name.replace("::", "_")
    typed_fn = make_typed_step_wrapper(
        fn_name,
        tpu_step_fn,
        num_inputs=len(tpu_in_specs),
        num_outputs=len(out_specs),
    )
    try:
        flat_callable: Any = sharded_jax_callable(
            op_name,
            typed_fn,
            mesh=mesh,
            input_partition_specs=tpu_in_specs,
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
        tp_rank_1d: torch.Tensor | None = None,
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
        if has_multi_rank:
            rank_arg = (
                tp_rank_1d
                if tp_rank_1d is not None
                else torch.tensor([int(tp_rank)], dtype=torch.int32, device=pos.device)
            )
            return list(flat_callable(*all_tensors, rank_arg))
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
        tp_rank_1d: torch.Tensor | None = None,
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
    tp_rank_1d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute `op`, copy donated outputs into `pools` in-place, and return `hid_out`.

    Preserves `pool.data_ptr()` and `pool.storage().data_ptr()` on every pool
    tensor so `raiden`-pinned buffers remain valid without re-registration.
    """
    if tp_rank_1d is None:
        outs = op(
            pos,
            hidden,
            page_table,
            ckpt_indices,
            read_offsets,
            list(weights),
            list(pools),
        )
    else:
        outs = op(
            pos,
            hidden,
            page_table,
            ckpt_indices,
            read_offsets,
            list(weights),
            list(pools),
            tp_rank_1d,
        )
    n_p = len(pools)
    for dst, src in zip(pools, outs[1 : 1 + n_p]):
        dst.copy_(src)
    return outs[0]
