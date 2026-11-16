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

``torch_tpu==0.2.0``'s ``JaxCallable`` takes ``output_partition_specs`` and
sizes each output with ``torch_placeholder(..., partition_spec=spec)``, but it
expects one spec per output leaf: given a bare ``PartitionSpec`` it zips the
spec's own entries against the outputs and fails. So this resolves the caller's
``output_partition_specs`` against ``len(lowered.out_avals)`` (broadcasting a
bare ``PartitionSpec`` across all output leaves, matching ``jax.jit``'s
``out_shardings`` semantics) and hands the per-leaf list to both the stock
``JaxCallable`` and ``fake_fn``.
"""

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import jax
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl


def _is_spec_leaf(spec: Any) -> bool:
    return spec is None or isinstance(spec, PartitionSpec)


def _flat_output_partition_specs(
    output_partition_specs: Any,
    num_outputs: int | None = None,
) -> list[PartitionSpec | None]:
    """One spec per output leaf, in ``lowered.out_avals`` order.

    Matches ``jax.jit``'s ``out_shardings`` tree-prefix rule for a bare
    ``PartitionSpec`` or ``None``: when ``num_outputs`` is known, a single leaf
    spec broadcasts across all ``num_outputs`` outputs; structured PyTrees must
    have exactly ``num_outputs`` leaves.
    """
    if _is_spec_leaf(output_partition_specs):
        count = 1 if num_outputs is None else num_outputs
        return [output_partition_specs] * count
    flat = jax.tree.leaves(output_partition_specs, is_leaf=_is_spec_leaf)
    if num_outputs is not None and len(flat) != num_outputs:
        raise ValueError(
            f"output_partition_specs has {len(flat)} leaves, "
            f"but fn returned {num_outputs} outputs."
        )
    return flat


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
    shape of ``fn``'s result (or passes a single ``PartitionSpec`` to broadcast
    across all outputs).

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
        lambda s: NamedSharding(mesh, s),
        output_partition_specs,
        is_leaf=lambda s: isinstance(s, PartitionSpec),
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
    wrapped_fn = pallas_impl.JaxCallable(
        name=name,
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs,
        # Annotated as tuple[tuple[str, ...], ...], but each entry is handed to
        # torch_placeholder(partition_spec=...), which takes a PartitionSpec.
        # pyrefly: ignore[bad-argument-type]
        output_partition_specs=_flat_output_partition_specs(output_partition_specs),
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

    orig_exported = wrapped_fn.exported
    resolved_specs: list[PartitionSpec | None] | None = None

    def _exported_with_resolved_specs(*jax_args: Any, **jax_kwargs: Any) -> Any:
        nonlocal resolved_specs
        lowered = orig_exported(*jax_args, **jax_kwargs)
        n_out = len(lowered.out_avals)
        if resolved_specs is None or len(resolved_specs) != n_out:
            resolved_specs = _flat_output_partition_specs(output_partition_specs, n_out)
            # pyrefly: ignore[bad-assignment]
            wrapped_fn.output_partition_specs = resolved_specs
        return lowered

    wrapped_fn.exported = _exported_with_resolved_specs
    op = torch.library.custom_op(name, wrapped_fn, mutates_args=())

    def fake_fn(*args: Any, **kwargs: Any) -> Any:
        norm_args, norm_kwargs = wrapped_fn._normalize_invocation_args(args, kwargs)
        jax_args, jax_kwargs = wrapped_fn._get_jax_placeholders(norm_args, norm_kwargs)
        with jax._src.config.export_ignore_forward_compatibility(True):
            lowered = wrapped_fn.exported(*jax_args, **jax_kwargs)
        assert resolved_specs is not None
        placeholders = [
            # pyrefly: ignore[bad-argument-type]
            pallas_impl.torch_placeholder(aval, mesh=mesh, partition_spec=spec)
            for aval, spec in zip(lowered.out_avals, resolved_specs)
        ]
        return lowered.out_tree.unflatten(placeholders)

    op.register_fake(fake_fn)
    return op
