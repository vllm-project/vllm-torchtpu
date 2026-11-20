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

The generic torch <-> jax pieces (`sharded_jax_callable`,
`make_typed_step_wrapper`, the CPU conversions) are in `jax_bridge`; this
module adds the `cache` / `ModelSpec`-bound pooled-step op. Provides:
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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.adapter.jax_bridge import (
    jax_to_torch_cpu as _jax_to_torch_cpu,
)
from vllm_torchtpu.megakernel.adapter.jax_bridge import (
    make_typed_step_wrapper,
    sharded_jax_callable,
)
from vllm_torchtpu.megakernel.adapter.jax_bridge import (
    torch_cpu_to_jax as _torch_cpu_to_jax,
)
from vllm_torchtpu.megakernel.core import cache
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.pool.config import cache_exec
from vllm_torchtpu.megakernel.core.runtime import build_global_avals
from vllm_torchtpu.megakernel.core.spec import ModelSpec
from vllm_torchtpu.megakernel.core.token_io import token_io


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


def export_pooled_step(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    w_flat: Sequence[torch.Tensor],
    raw_pools: Sequence[torch.Tensor],
    *,
    max_pages_per_seq: int,
    ckpt_cols: int | None = None,
    op_name: str = "megakernel::exported_pooled_step",
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
    single_ex: ExecSpec | None = None,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """Build a PyTorch custom op wrapping `cache.make_flat_pooled_step`.

    Returns `(op, in_specs, out_specs, donate_argnums)` where `op` takes
    `(pos, hidden, page_table, step_meta, weights, pools)`: `step_meta` is the
    flat step's arguments after the page table (`ckpt_indices`, `read_offsets`
    for recurrent pools, then the module extra inputs). Under token step I/O
    (`ExecSpec.tokens`) the first three are the runner's `input_ids`,
    `positions` and `block_tables`, `step_meta` ends with its `seq_lens`, and
    the first output is the `[B]` int32 greedy tokens instead of the hidden
    states. With `single_ex` (a token verify step `ex`, and the one-token
    step of its sequences), `step_meta` ends with `has_drafts`, see
    `_skip_empty_verify`.
    """
    tokens = token_io(ex) is not None
    global _OP_COUNTER
    if single_ex is not None:
        step_fn, in_specs, out_specs, donate_argnums = _skip_empty_verify(
            mesh, spec, ex, single_ex, donate_pools
        )
    else:
        step_fn, in_specs, out_specs, donate_argnums = cache.make_flat_pooled_step(
            mesh, spec, ex, donate_pools=donate_pools
        )
    _OP_COUNTER += 1
    if op_name is None:
        n_pages = cache_exec(ex).num_pages
        op_name = f"megakernel::step_t{ex.T}_b{ex.B}_p{n_pages}_{_OP_COUNTER}"
    elif "::" not in op_name:
        op_name = f"megakernel::{op_name}"

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
        step_meta: list[torch.Tensor],
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        all_tensors = [
            pos,
            hidden,
            *weights,
            *pools,
            page_table,
            *step_meta,
        ]
        if pos.device.type == "cpu":
            # CPU interpret execution path for unit tests without a TPU host.
            jax_inputs = [
                _torch_cpu_to_jax(t, sh)
                for t, sh in zip(all_tensors, in_sh, strict=True)
            ]
            jax_outs = cpu_jit_fn(*jax_inputs)
            like_tensors = [_first_out(hidden), *pools]
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
        step_meta: list[torch.Tensor],
        weights: list[torch.Tensor],
        pools: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [_first_out(hidden)] + [torch.empty_like(p) for p in pools]

    def _first_out(hidden: torch.Tensor) -> torch.Tensor:
        if tokens:
            return hidden.new_empty((ex.B,), dtype=torch.int32)
        return torch.empty_like(hidden)

    return op, in_specs, out_specs, donate_argnums


def _skip_empty_verify(
    mesh: jax.sharding.Mesh,
    spec: ModelSpec,
    ex: ExecSpec,
    one: ExecSpec,
    donate_pools: bool,
) -> tuple[Any, tuple[P, ...], tuple[P, ...], tuple[int, ...]]:
    """The token verify step of `ex` that runs the one-token step `one` when
    no sequence has a draft to verify, chosen on the device from the extra
    last input `has_drafts [1]` (0: none): such a step is a decode step, so it
    costs one (2460 vs 2573 us at T=8). Row `s * T` then holds sequence `s`'s
    token and the other rows -1, which no draft matches; the K/V of the first
    token of each sequence is written, the only one a step without drafts
    keeps. Both branches alias the pools in place."""
    if token_io(ex) is None or ex.T < 2 or (one.B, one.T) != (ex.NS, 1):
        raise ValueError(f"not a token verify step and its one-token step: {ex}, {one}")
    full, in_specs, out_specs, donate = cache.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=donate_pools
    )
    single, single_in, _, _ = cache.make_flat_pooled_step(
        mesh, spec, one, donate_pools=donate_pools
    )
    if tuple(single_in) != tuple(in_specs):
        raise ValueError("the one-token step takes other inputs than the verify step")
    T, NS, B = ex.T, ex.NS, ex.B
    first = slice(0, NS * T, T)

    def run_single(args: tuple[Any, ...]) -> tuple[Any, ...]:
        # The runner's step tensors: ids, positions, ..., seq_lens last.
        ids, positions, *rest, seq_lens = args
        ids = jnp.zeros_like(ids).at[:NS].set(ids[first])
        positions = jnp.zeros_like(positions).at[..., :NS].set(positions[..., first])
        tok, *pools = single(ids, positions, *rest, seq_lens - (T - 1))
        return (jnp.full((B,), -1, tok.dtype).at[first].set(tok[:NS]), *pools)

    def step(*args: Any) -> tuple[Any, ...]:
        *args, has_drafts = args
        return jax.lax.cond(
            has_drafts[0] > 0,
            lambda a: tuple(full(*a)),
            run_single,
            tuple(args),
        )

    return step, (*in_specs, P()), out_specs, donate


def execute_pooled_step_inplace(
    op: Callable[..., list[torch.Tensor]],
    pos: torch.Tensor,
    hidden: torch.Tensor,
    page_table: torch.Tensor,
    step_meta: Sequence[torch.Tensor],
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
        list(step_meta),
        list(weights),
        list(pools),
    )
    n_p = len(pools)
    for dst, src in zip(pools, outs[1 : 1 + n_p]):
        dst.copy_(src)
    return outs[0]
