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
"""Generic torch <-> jax step bridge (no `megakernel` library imports).

- `sharded_jax_callable`: an SPMD JAX function over a mesh as a `torch_tpu`
  `JaxCallable`, built by torch_tpu's own `pallas.custom_jax_kernel`, for
  callers that wrap it in their own custom op (e.g. to take `list[Tensor]`
  arguments past the 64-argument schema limit of a `torch.library` op).
- `make_typed_step_wrapper`: a fixed-arity, `jax.Array`-annotated wrapper of a
  `*args` step function (`JaxCallable` rejects `*args` signatures).
- `torch_cpu_to_jax` / `jax_to_torch_cpu`: lossless CPU conversions for the
  CPU interpret path of unit tests (a 64-bit dtype needs `jax_enable_x64`;
  without it the conversion raises rather than narrowing).
- `build_jax_step_op`: a JAX step function as a torch callable (a
  `torch.library` custom op over a `list[Tensor]` of operands):
  `sharded_jax_callable` with TPU tensors, the function run by JAX with CPU
  tensors, with donated outputs copied back into their input tensors in
  place.

These ops are for megakernel steps. The driver compiles each step with
`torch.compile(..., options={"serializable": False})`, so no compile cache
persists them, and it runs them from `warmup` and `try_forward`, outside the
stock model's cached executables. That is why this module is not in the TPU
compile cache key (`tpu_compiler._RUNTIME_CACHE_KEY_PATHS`). A custom op used
inside the stock model belongs in that list instead.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed.pallas_shapes import get_global_shape, get_local_shape


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
    """`fn`, an SPMD JAX function over `mesh`, as a `torch_tpu` `JaxCallable`.

    Called with this rank's shards of the operands (`input_partition_specs`
    describe the global arrays), it returns this rank's shards of the
    outputs (`output_partition_specs`). Every parameter of `fn` must be an
    array (no static arguments). No custom op is registered.
    """
    from torch_tpu._internal.pallas import pallas as pallas_impl

    if name.count("::") != 1:
        raise ValueError(f"op name {name!r} must be 'namespace::op', with one '::'")
    return pallas_impl.custom_jax_kernel(
        fn,
        name,
        donate_argnums=list(donate_argnums) if donate_argnums else None,
        mesh=mesh,
        input_partition_specs=tuple(input_partition_specs),
        output_partition_specs=tuple(_leaf_output_specs(output_partition_specs)),
        warn_deprecated=False,
    )


def torch_cpu_to_jax(
    t: torch.Tensor, sharding: NamedSharding | None = None
) -> jax.Array:
    """Convert a CPU `torch.Tensor` to a `jax.Array` without precision loss.

    Raises `ValueError` for a 64-bit tensor (int64 / float64 / ...) while
    `jax_enable_x64` is off: JAX would narrow it to 32 bits silently.
    """
    t_c = t.detach().cpu().contiguous()
    if t_c.dtype.itemsize == 8 and not jax.config.jax_enable_x64:
        raise ValueError(
            f"torch_cpu_to_jax: a {t_c.dtype} tensor needs jax_enable_x64; "
            "without it JAX narrows it to 32 bits. Pass a 32-bit tensor or "
            "enable x64"
        )
    if t_c.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t_c.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t_c.dtype == torch.bfloat16:
        arr = jnp.asarray(t_c.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t_c.numpy())
    if sharding is not None:
        arr = jax.device_put(arr, sharding)
    return arr


def jax_to_torch_cpu(a: jax.Array, like: torch.Tensor) -> torch.Tensor:
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


_OP_IDS = itertools.count()

_TORCH_TO_JAX_DTYPE: dict[torch.dtype, Any] = {
    torch.float32: jnp.float32,
    torch.float16: jnp.float16,
    torch.bfloat16: jnp.bfloat16,
    torch.float8_e4m3fn: jnp.float8_e4m3fn,
    torch.int64: jnp.int64,
    torch.int32: jnp.int32,
    torch.int16: jnp.int16,
    torch.int8: jnp.int8,
    torch.uint8: jnp.uint8,
    torch.bool: jnp.bool_,
}
_JAX_TO_TORCH_DTYPE = {jnp.dtype(j): t for t, j in _TORCH_TO_JAX_DTYPE.items()}


def _jax_to_torch(a: jax.Array) -> torch.Tensor:
    """A CPU `torch.Tensor` of `a`'s own dtype (bit-exact for bf16 / fp8)."""
    dtype = _JAX_TO_TORCH_DTYPE[jnp.dtype(a.dtype)]
    return jax_to_torch_cpu(a, torch.empty((), dtype=dtype))


def _out_structs(
    fn: Callable[..., Any],
    operands: Sequence[torch.Tensor],
    n_out: int,
    *,
    mesh: jax.sharding.Mesh | None = None,
    in_specs: Sequence[P] = (),
    out_specs: Sequence[P] = (),
) -> list[jax.ShapeDtypeStruct]:
    """The outputs' shapes and dtypes for `operands`.

    With `mesh`, the operands are this rank's shards: they are scaled to the
    global shapes by `in_specs`, `fn`'s global outputs are shrunk to this
    rank's shards by `out_specs` (what torch_tpu's `JaxCallable` returns).
    Without, the operands and outputs are the global arrays.
    """
    structs = []
    for i, t in enumerate(operands):
        shape = tuple(t.shape)
        if mesh is not None:
            shape = get_global_shape(shape, mesh, in_specs[i])
        structs.append(jax.ShapeDtypeStruct(shape, _TORCH_TO_JAX_DTYPE[t.dtype]))
    outs = jax.tree.leaves(jax.eval_shape(fn, *structs))
    if len(outs) != n_out:
        raise ValueError(f"step returned {len(outs)} outputs, out_specs has {n_out}")
    if mesh is None:
        return outs
    return [
        jax.ShapeDtypeStruct(get_local_shape(o.shape, mesh, spec), o.dtype)
        for o, spec in zip(outs, out_specs, strict=True)
    ]


def build_jax_step_op(
    fn: Callable[..., Any],
    *,
    mesh: jax.sharding.Mesh,
    in_specs: Sequence[P],
    out_specs: P | Sequence[P],
    donate: Mapping[int, int] | None = None,
    name: str | None = None,
    interpret_on_cpu: bool = True,
) -> Callable[..., Any]:
    """A JAX step function as a torch callable over positional tensors.

    `fn(*arrays)` is an SPMD JAX function over `mesh` whose (global) operands
    are sharded by `in_specs` and outputs by `out_specs` (one
    `PartitionSpec`: the callable returns one tensor; a sequence: a tuple).
    `donate` maps an output index to the input it updates: that input's
    buffer is donated on TPU, and the output is copied back into the input
    tensor in place (`data_ptr()` is preserved) and returned as that tensor.

    The callable runs a `torch.library` custom op, named `name` (default
    `megakernel::jax_step`; a name without a namespace gets `megakernel::`)
    plus a per-op `_<n>` suffix, so every build registers a new op.
    - With TPU tensors it calls `sharded_jax_callable` (built on first use):
      each rank passes its shards of the operands and gets its shards of the
      outputs.
    - With CPU tensors (unit tests) JAX runs `fn` in this process over the
      whole mesh: the operands and outputs are the global arrays
      (`interpret_on_cpu=False` makes that an error). Kernels that need
      Pallas interpret mode on CPU choose it inside `fn`.
    The op's fake kernel (from `jax.eval_shape`) gives the outputs those
    shapes, so `torch.compile` can trace it.
    """
    single = isinstance(out_specs, P)
    outs_specs: tuple[P, ...] = (out_specs,) if single else tuple(out_specs)
    ins_specs = tuple(in_specs)
    n_in, n_out = len(ins_specs), len(outs_specs)
    aliases = dict(donate or {})
    for o, i in aliases.items():
        if not (0 <= o < n_out and 0 <= i < n_in):
            raise ValueError(
                f"donate maps output {o} to input {i}: out of range for "
                f"{n_out} outputs and {n_in} inputs"
            )
    if len(set(aliases.values())) != len(aliases):
        raise ValueError(f"donate maps two outputs to one input: {aliases}")
    op_id = next(_OP_IDS)
    base = name or "jax_step"
    if "::" not in base:
        base = f"megakernel::{base}"
    name = f"{base}_{op_id}"

    def tuple_fn(*args: Any) -> tuple[Any, ...]:
        out = fn(*args)
        return (out,) if single else tuple(out)

    in_sh = tuple(NamedSharding(mesh, s) for s in ins_specs)
    out_sh = tuple(NamedSharding(mesh, s) for s in outs_specs)
    cpu_fn = jax.jit(tuple_fn, in_shardings=in_sh, out_shardings=out_sh)
    tpu_fn: list[Any] = []

    def tpu_callable() -> Any:
        if not tpu_fn:
            typed = make_typed_step_wrapper(
                name.replace("::", "_"), tuple_fn, n_in, n_out
            )
            tpu_fn.append(
                sharded_jax_callable(
                    name,
                    typed,
                    mesh=mesh,
                    input_partition_specs=ins_specs,
                    output_partition_specs=outs_specs,
                    donate_argnums=tuple(sorted(aliases.values())) or None,
                )
            )
        return tpu_fn[0]

    @torch.library.custom_op(name, mutates_args=())
    def op(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(operands) != n_in:
            raise ValueError(f"{name}: {len(operands)} operands, in_specs has {n_in}")
        if operands[0].device.type == "cpu":
            if not interpret_on_cpu:
                raise RuntimeError(f"{name}: CPU operands with interpret_on_cpu=False")
            arrays = [
                torch_cpu_to_jax(t, sh) for t, sh in zip(operands, in_sh, strict=True)
            ]
            return [_jax_to_torch(a) for a in cpu_fn(*arrays)]
        return list(tpu_callable()(*operands))

    @op.register_fake
    def _(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        dev = operands[0].device
        sharded = dev.type != "cpu"
        structs = _out_structs(
            tuple_fn,
            operands,
            n_out,
            mesh=mesh if sharded else None,
            in_specs=ins_specs,
            out_specs=outs_specs,
        )
        return [
            torch.empty(
                s.shape, dtype=_JAX_TO_TORCH_DTYPE[jnp.dtype(s.dtype)], device=dev
            )
            for s in structs
        ]

    def step(*operands: torch.Tensor) -> Any:
        outs = op(list(operands))
        res = []
        for o, out in enumerate(outs):
            i = aliases.get(o)
            if i is None:
                res.append(out)
            else:
                operands[i].copy_(out)
                res.append(operands[i])
        return res[0] if single else tuple(res)

    step.op = op  # type: ignore[attr-defined]
    step.op_name = name  # type: ignore[attr-defined]
    return step


__all__ = [
    "build_jax_step_op",
    "jax_to_torch_cpu",
    "make_typed_step_wrapper",
    "sharded_jax_callable",
    "torch_cpu_to_jax",
]
