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

- `sharded_jax_callable`: wraps an SPMD JAX function over a mesh into a
  `torch_tpu` `JaxCallable` (operands as positional tensors; callers pass
  variable-length groups such as weights / pools as `list[Tensor]` custom-op
  arguments and flatten them).
- `make_typed_step_wrapper`: a fixed-arity, `jax.Array`-annotated wrapper of a
  `*args` step function (`JaxCallable` rejects `*args` signatures).
- `torch_cpu_to_jax` / `jax_to_torch_cpu`: lossless CPU conversions for the
  CPU interpret path of unit tests.
- `build_jax_step_op`: a JAX step function as a torch callable (a
  `torch.library` custom op): `sharded_jax_callable` on TPU, the function run
  by JAX on CPU otherwise, with donated outputs copied back into their input
  tensors in place.
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

    Self-contained: it depends on no `vllm`, `vllm_torchtpu.distributed` or
    `megakernel` module.
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


def torch_cpu_to_jax(
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


def _eval_shapes(
    fn: Callable[..., Any], operands: Sequence[torch.Tensor], n_out: int
) -> list[jax.ShapeDtypeStruct]:
    structs = [
        jax.ShapeDtypeStruct(tuple(t.shape), _TORCH_TO_JAX_DTYPE[t.dtype])
        for t in operands
    ]
    outs = jax.tree.leaves(jax.eval_shape(fn, *structs))
    if len(outs) != n_out:
        raise ValueError(f"step returned {len(outs)} outputs, out_specs has {n_out}")
    return outs


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

    `fn(*arrays)` is an SPMD JAX function over `mesh` whose operands are
    sharded by `in_specs` and outputs by `out_specs` (one `PartitionSpec`:
    the callable returns one tensor; a sequence: a tuple). `donate` maps an
    output index to the input it updates: that input's buffer is donated on
    TPU, and the output is copied back into the input tensor in place
    (`data_ptr()` is preserved) and returned as that tensor.

    The callable runs a `torch.library` custom op (a fake kernel from
    `jax.eval_shape` keeps it traceable by `torch.compile`): with TPU tensors
    it calls `sharded_jax_callable`, built on first use; with CPU tensors JAX
    runs `fn` on its CPU devices (`interpret_on_cpu=False` makes that an
    error). Kernels that need Pallas interpret mode on CPU choose it inside
    `fn`.
    """
    single = isinstance(out_specs, P)
    outs_specs: tuple[P, ...] = (out_specs,) if single else tuple(out_specs)
    ins_specs = tuple(in_specs)
    n_in, n_out = len(ins_specs), len(outs_specs)
    aliases = dict(donate or {})
    for o, i in aliases.items():
        if not (0 <= o < n_out and 0 <= i < n_in):
            raise ValueError(f"donate maps output {o} to input {i}: out of range")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("donate maps two outputs to one input")
    op_id = next(_OP_IDS)
    if name is None:
        name = f"megakernel_runtime::jax_step_{op_id}"
    elif "::" not in name:
        name = f"megakernel_runtime::{name}_{op_id}"

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
        return [
            torch.empty(
                s.shape, dtype=_JAX_TO_TORCH_DTYPE[jnp.dtype(s.dtype)], device=dev
            )
            for s in _eval_shapes(tuple_fn, operands, n_out)
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
    return step


__all__ = [
    "build_jax_step_op",
    "jax_to_torch_cpu",
    "make_typed_step_wrapper",
    "sharded_jax_callable",
    "torch_cpu_to_jax",
]
