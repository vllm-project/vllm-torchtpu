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

- `sharded_jax_callable`: an SPMD JAX function over a mesh as a callable over
  torch tensors, registered with torch_tpu's `pallas.jax_op`. The operands
  are passed as `list[Tensor]` arguments, one per partition spec (and per
  donated or not), so any number of operands fits the 64-argument schema
  limit of a `torch.library` op.
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

import inspect
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
) -> Callable[..., list[torch.Tensor]]:
    """`fn`, an SPMD JAX function over `mesh`, as a callable over torch tensors.

    Called with this rank's shards of the operands (`input_partition_specs`
    describe the global arrays), it returns this rank's shards of the
    outputs (`output_partition_specs`, one per output) as a list. Every
    parameter of `fn` must be an array (no static arguments), and every
    `donate_argnums` operand is donated.

    It registers `fn` with torch_tpu's `pallas.jax_op` as the op `name`, which
    applies one partition spec and one donation flag to a whole `list[Tensor]`
    argument: the operands are grouped into one list per `(spec, donated)`, in
    order of first use, and `fn` gets them back in their own order.
    """
    from torch_tpu._internal import pallas

    if name.count("::") != 1:
        raise ValueError(f"op name {name!r} must be 'namespace::op', with one '::'")
    in_specs = tuple(input_partition_specs)
    donated = set(donate_argnums or ())
    group_of: dict[tuple[P, bool], int] = {}
    members: list[list[int]] = []
    slots: list[tuple[int, int]] = []
    for i, spec in enumerate(in_specs):
        g = group_of.setdefault((spec, i in donated), len(members))
        if g == len(members):
            members.append([])
        slots.append((g, len(members[g])))
        members[g].append(i)
    group_keys = sorted(group_of, key=group_of.__getitem__)

    def jax_step(*groups: list[jax.Array]) -> list[jax.Array]:
        return list(jax.tree.leaves(fn(*(groups[g][j] for g, j in slots))))

    # jax_op reads one `list[jax.Array]` parameter per group from the signature.
    jax_step.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                f"g{g}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=list[jax.Array],
            )
            for g in range(len(members))
        ],
        return_annotation=list[jax.Array],
    )
    # torch_tpu annotates the partition specs as tuples of axis-name tuples, but
    # jax_op takes PartitionSpecs at runtime, as other jax_op call sites pass them.
    op: Any = pallas.jax_op(
        name,
        jax_step,
        mesh=mesh,
        input_partition_specs=tuple(spec for spec, _ in group_keys),
        output_partition_specs=tuple(_leaf_output_specs(output_partition_specs)),
        donate_argnums=tuple(g for g, (_, d) in enumerate(group_keys) if d) or None,
    )

    def call(*operands: torch.Tensor) -> list[torch.Tensor]:
        return list(op(*([operands[i] for i in m] for m in members)))

    return call


def torch_cpu_to_jax(
    t: torch.Tensor, sharding: NamedSharding | None = None
) -> jax.Array:
    """Convert a CPU `torch.Tensor` to a `jax.Array` without precision loss.

    Raises `ValueError` for an int64 / uint64 / float64 tensor while
    `jax_enable_x64` is off: JAX would narrow it to 32 bits silently.
    (complex64 is two 32-bit floats, which JAX keeps without x64.)
    """
    t_c = t.detach().cpu().contiguous()
    if t_c.dtype in _X64_DTYPES and not jax.config.jax_enable_x64:
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
    """Convert a `jax.Array` to a CPU `torch.Tensor` of dtype `like.dtype`.

    Raises `ValueError` when `a.dtype` is not the JAX dtype of `like.dtype`:
    the conversion reinterprets bytes, it does not cast.
    """
    want = _TORCH_TO_JAX_DTYPE.get(like.dtype)
    if want is None:  # a dtype numpy has, e.g. complex64
        want = torch.empty(0, dtype=like.dtype).numpy().dtype
    if jnp.dtype(a.dtype) != jnp.dtype(want):
        raise ValueError(
            f"jax_to_torch_cpu: a {jnp.dtype(a.dtype)} array can't be read as "
            f"{like.dtype} (expected {jnp.dtype(want)})"
        )
    np_arr = np.asarray(a)
    if like.dtype == torch.float8_e4m3fn:
        out = torch.from_numpy(np_arr.view(np.uint8).copy()).view(torch.float8_e4m3fn)
    elif like.dtype == torch.bfloat16:
        out = torch.from_numpy(np_arr.view(np.uint16).copy()).view(torch.bfloat16)
    else:
        out = torch.from_numpy(np_arr.copy())
    return out


_OP_IDS = itertools.count()

_TORCH_TO_JAX_DTYPE: dict[torch.dtype, Any] = {
    torch.float64: jnp.float64,
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
# The dtypes JAX narrows to 32 bits unless `jax_enable_x64` is on.
_X64_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


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
    rank's shards by `out_specs` (what `sharded_jax_callable` returns).
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
    shapes, so `torch.compile` can trace it. The op and its fake kernel check
    the operand count and that every operand is on one device. A donated
    output must have its input's shape and dtype, and a step that donates
    inputs must run under `torch.compile` when given TPU tensors.
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
            tpu_fn.append(
                sharded_jax_callable(
                    f"{name}_jax",
                    tuple_fn,
                    mesh=mesh,
                    input_partition_specs=ins_specs,
                    output_partition_specs=outs_specs,
                    donate_argnums=tuple(sorted(aliases.values())) or None,
                )
            )
        return tpu_fn[0]

    def check_operands(operands: Sequence[torch.Tensor]) -> torch.device:
        """The operands' one device, after the operand count and device checks."""
        if len(operands) != n_in:
            raise ValueError(f"{name}: {len(operands)} operands, in_specs has {n_in}")
        devices = {t.device for t in operands}
        if len(devices) != 1:
            # A CPU operand among TPU ones would take the CPU path, which
            # copies every operand (TPU pools included) to the host.
            raise ValueError(f"{name}: operands on more than one device: {devices}")
        dev = devices.pop()
        if dev.type == "cpu" and not interpret_on_cpu:
            raise RuntimeError(f"{name}: CPU operands with interpret_on_cpu=False")
        return dev

    @torch.library.custom_op(name, mutates_args=())
    def op(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        if check_operands(operands).type == "cpu":
            arrays = [
                torch_cpu_to_jax(t, sh) for t, sh in zip(operands, in_sh, strict=True)
            ]
            return [_jax_to_torch(a) for a in cpu_fn(*arrays)]
        return list(tpu_callable()(*operands))

    @op.register_fake
    def _(operands: list[torch.Tensor]) -> list[torch.Tensor]:
        dev = check_operands(operands)
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
        dev = check_operands(operands)
        if aliases and dev.type != "cpu" and not torch.compiler.is_compiling():
            # Eagerly, the donated input is consumed by the call and the result
            # is copied back into it: a pool-sized copy every step. Under
            # `torch.compile` the copy-back aliases the donated buffer instead.
            raise RuntimeError(
                f"{name}: a step that donates inputs must run under "
                "torch.compile with TPU tensors (eagerly it copies each donated "
                "input every call)"
            )
        outs = op(list(operands))
        res = []
        for o, out in enumerate(outs):
            i = aliases.get(o)
            if i is None:
                res.append(out)
                continue
            src = operands[i]
            if out.shape != src.shape or out.dtype != src.dtype:
                raise ValueError(
                    f"{name}: donated output {o} is {out.dtype}{list(out.shape)} "
                    f"but its input {i} is {src.dtype}{list(src.shape)}"
                )
            src.copy_(out)
            res.append(src)
        return res[0] if single else tuple(res)

    step.op = op  # type: ignore[attr-defined]
    step.op_name = name  # type: ignore[attr-defined]
    return step


__all__ = [
    "build_jax_step_op",
    "jax_to_torch_cpu",
    "sharded_jax_callable",
    "torch_cpu_to_jax",
]
