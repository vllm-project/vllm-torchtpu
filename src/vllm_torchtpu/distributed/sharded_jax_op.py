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

``torch_tpu``'s ``JaxCallable`` sizes each torch output tensor from the
partition spec it is given for that output, and without one from
``lowered.out_avals``, where ``jax.export`` records ``shard_map`` results as
*replicated*. A caller whose outputs are genuinely sharded but that passes no
output specs therefore gets tensors sized for the global array instead of this
rank's shard -- for an EP MoE at 2048 tokens a rank, an output claiming 16384
rows. So ``sharded_jax_op`` requires the output specs, pins the export's output
shardings to them, and sizes both the real and the fake outputs from them.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl


def _is_spec(s: Any) -> bool:
    return isinstance(s, PartitionSpec)


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
    if "::" not in name or len(name.split("::")) != 2:
        raise ValueError(f"Op name {name} must be 'namespace::op', with one '::'.")

    signature = inspect.signature(inspect.unwrap(fn), eval_str=True)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    input_partition_specs = tuple(input_partition_specs)
    out_shardings = jax.tree.map(
        lambda s: NamedSharding(mesh, s), output_partition_specs, is_leaf=_is_spec
    )
    # One spec per output, in the order the export flattens them: torch_tpu
    # sizes output i from spec i.
    output_specs = tuple(jax.tree.leaves(output_partition_specs, is_leaf=_is_spec))

    # `out_shardings` pins the export's output shardings to the same specs;
    # the rest of the jit config mirrors `pallas.custom_jax_kernel`,
    # `keep_unused` included so the jitted arity keeps matching the torch
    # operands.
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
        input_partition_specs=input_partition_specs,
        output_partition_specs=output_specs,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

    op = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args, **kwargs):
        jax_args = pallas_impl.jax_placeholders(
            args, mesh=mesh, partition_specs=input_partition_specs
        )
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **kwargs)
        return lowered.out_tree.unflatten(
            pallas_impl.torch_placeholder(aval, mesh=mesh, partition_spec=spec)
            for aval, spec in zip(lowered.out_avals, output_specs, strict=True)
        )

    op.register_fake(fake_fn)
    return op
