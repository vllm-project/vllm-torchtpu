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
from typing import Any, cast

import jax
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl


def _sharded_output_placeholders(
    lowered: Any,
    mesh: jax.sharding.Mesh,
    fallback_specs: Sequence[PartitionSpec],
) -> list[torch.Tensor | None]:
    shardings = getattr(lowered, "_out_named_shardings", None)
    if shardings is not None and len(shardings) == len(lowered.out_avals):
        specs: Sequence[Any] = [getattr(s, "spec", None) for s in shardings]
    else:
        specs = fallback_specs
    return [
        pallas_impl.torch_placeholder(aval, mesh=mesh, partition_spec=spec)  # type: ignore[arg-type]
        for aval, spec in zip(lowered.out_avals, specs)
    ]


def sharded_jax_op(
    name: str,
    fn: Callable[..., object],
    *,
    mesh: jax.sharding.Mesh,
    input_partition_specs: Sequence[PartitionSpec],
    output_partition_specs: Any,
    donate_argnums: Sequence[int] | None = None,
    fake: Callable[..., object] | None = None,
):
    """Register ``fn`` as an SPMD custom op over ``mesh`` with explicit output specs."""
    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name {name} must be 'namespace::op', with one '::'.")

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)

    def is_spec(s: Any) -> bool:
        return isinstance(s, PartitionSpec)

    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s), output_partition_specs, is_leaf=is_spec
    )
    out_specs = jax.tree.leaves(output_partition_specs, is_leaf=is_spec)

    # `out_shardings` is what puts the real shardings on the export's
    # `_out_named_shardings`, which is the only place they survive; the rest of
    # the jit config mirrors `pallas.custom_jax_kernel`, `keep_unused` included
    # so the jitted arity keeps matching the torch operands.
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
    wrapped_fn = pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=cast(Any, input_partition_specs),
        output_partition_specs=cast(Any, out_specs),
        static_argnums=static_argnums,
        donate_argnums=cast(Any, donate_argnums),
    )

    op = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args: Any, **kwargs: Any) -> Any:
        jax_args = pallas_impl.jax_placeholders(
            args, mesh=mesh, partition_specs=input_partition_specs
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            _sharded_output_placeholders(lowered, mesh, out_specs)
        )

    op.register_fake(fake if fake is not None else fake_fn)
    return op
