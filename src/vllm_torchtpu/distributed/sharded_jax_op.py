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
"""Register a torch custom op backed by a multi-device JAX/Pallas function.

``torch_tpu``'s ``pallas.jax_op`` accepts ``mesh`` and
``input_partition_specs`` and exports a real multi-device SPMD module, but it
sizes the torch output tensors from ``lowered.out_avals``, where ``jax.export``
records ``shard_map`` results as *replicated*. A caller whose outputs are
genuinely sharded therefore gets tensors sized for the global array instead of
this rank's shard -- for an EP MoE at 2048 tokens a rank, an output claiming
16384 rows.

``pcp_streaming_rpa``'s adapter solved that by reimplementing ``__call__``, and
that copy has since fallen behind -- stock has gained an eager wrapper-tensor
branch and MLIR-fingerprint-based kernel caching that it does not have. So
instead of forking the method, this derives it from the installed one and
changes the single line that builds the output placeholders. If upstream
reworks that line the derivation no-ops and the caller falls back to stock,
which is wrong only in the way described above and fails loudly on a shape
mismatch rather than silently.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_STOCK_PLACEHOLDER_LINE = """      output_shapes = [
          torch_placeholder(aval, mesh=self.mesh) for aval in lowered.out_avals
      ]"""
_SHARDED_PLACEHOLDER_LINE = """      output_shapes = _sharded_output_placeholders(lowered, self.mesh)"""


def _sharded_output_placeholders(lowered, mesh):
    """Torch placeholders sized for this rank's shard of each output.

    ``jax.export`` keeps the real output shardings on ``_out_named_shardings``
    even though it reports ``out_avals`` as replicated; where it does not, fall
    back to the aval, which is what stock would have used anyway.
    """
    shardings = getattr(lowered, "_out_named_shardings", None)
    if shardings is None or len(shardings) != len(lowered.out_avals):
        shardings = (None, ) * len(lowered.out_avals)
    return [
        _placeholder(aval, getattr(sharding, "spec", None), mesh)
        for aval, sharding in zip(lowered.out_avals, shardings)
    ]


def _placeholder(aval, spec, mesh):
    """One output tensor, shaped by ``spec`` rather than by the aval.

    The aval carries a replicated spec that cannot be overridden in place --
    ``ShapedArray.update(sharding=...)`` drops a concrete mesh's spec on the
    floor -- so this reproduces what ``torch_placeholder`` would do, reading the
    partitioning from ``spec``. Outputs ``jax.export`` did not label, and
    non-array outputs, go through the installed path unchanged.
    """
    if spec is None or not isinstance(aval, jax.core.ShapedArray):
        return pallas_impl.torch_placeholder(aval, mesh=mesh)
    torch_dtype = pallas_impl.JAX_TO_TORCH_DTYPE_MAP.get(aval.dtype)
    if torch_dtype is None:
        raise NotImplementedError(
            f"Unsupported dtype for pallas kernels: {aval.dtype}")
    shape = tuple(pallas_impl.get_local_shape(aval.shape, mesh, spec))
    if torch_dtype == torch.float4_e2m1fn_x2 and shape:
        if shape[-1] % 2:
            raise ValueError(
                "expected the last dimension of the JAX local shape to be even "
                f"to pack into {torch_dtype}, got {shape[-1]} (local={shape}, "
                f"global={aval.shape}, spec={spec})")
        shape = shape[:-1] + (shape[-1] // 2, )
    return torch.empty(shape, dtype=torch_dtype, device="tpu")


def _build_sharded_callable_cls():
    """A ``JaxCallable`` whose ``__call__`` is the installed one, minus the
    assumption that a sharded output is replicated.

    The method source is recompiled inside a throwaway class so its original
    indentation -- and therefore the needle below -- survives verbatim.
    """
    src = inspect.getsource(pallas_impl.JaxCallable.__call__)
    if _STOCK_PLACEHOLDER_LINE not in src:
        logger.warning(
            "torch_tpu's JaxCallable.__call__ no longer builds its output "
            "placeholders the way this adapter expects; sharded outputs will "
            "be sized globally. Sharded multi-device ops will fail on a shape "
            "mismatch until this is re-derived.")
        return pallas_impl.JaxCallable
    patched = "class _Holder:\n" + src.replace(_STOCK_PLACEHOLDER_LINE,
                                               _SHARDED_PLACEHOLDER_LINE, 1)
    namespace: dict = {}
    exec(  # noqa: S102 - deriving from the installed source is the point
        compile(patched, pallas_impl.__file__, "exec"),
        {
            **pallas_impl.__dict__,
            "_sharded_output_placeholders": _sharded_output_placeholders,
        },
        namespace,
    )
    return type("_ShardedJaxCallable", (pallas_impl.JaxCallable, ),
                {"__call__": namespace["_Holder"].__call__})


_SHARDED_CALLABLE_CLS = None


def sharded_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Any,
    donate_argnums: Sequence[int] | None = None,
):
    """A torch custom op for ``fn`` run as an SPMD program over ``mesh``.

    ``input_partition_specs`` describe the *global* arrays the op is called
    with; the torch tensors passed in are this rank's shards and
    ``pallas.jax_op`` reconstitutes the global shapes from the mesh.
    ``output_partition_specs`` is required, not optional -- getting it wrong is
    how a sharded output silently becomes a replicated one -- and mirrors the
    shape of ``fn``'s result: one spec for one array.

    Prefer returning a single array from ``fn`` over a one-tuple. Both work
    here, but a one-tuple gives the op the schema ``-> ((Tensor))``, and torch
    ``guard_int``s every symbolic input dimension of an op with that schema, so
    a caller under ``torch.compile`` loses its dynamic batch dimension.
    """
    global _SHARDED_CALLABLE_CLS
    if _SHARDED_CALLABLE_CLS is None:
        _SHARDED_CALLABLE_CLS = _build_sharded_callable_cls()
    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(
            f"Op name {name} must be 'namespace::op', with one '::'.")

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, PartitionSpec))

    # `out_shardings` is what puts the real shardings on the export's
    # `_out_named_shardings`, which is the only place they survive; the rest of
    # the jit config mirrors `pallas.custom_jax_kernel`, `keep_unused` included
    # so the jitted arity keeps matching the torch operands.
    jit_fn = jax.jit(fn,
                     static_argnums=static_argnums,
                     donate_argnums=donate_argnums,
                     out_shardings=out_shardings,
                     keep_unused=True)
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{name}_{id(fn)}", [], {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums,
            "output_partition_specs": str(output_partition_specs),
        })
    wrapped_fn = _SHARDED_CALLABLE_CLS(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

    op = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args, **kwargs):
        jax_args = pallas_impl.jax_placeholders(
            args, mesh=mesh, partition_specs=input_partition_specs)
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            _sharded_output_placeholders(lowered, mesh))

    op.register_fake(fake_fn)
    return op
