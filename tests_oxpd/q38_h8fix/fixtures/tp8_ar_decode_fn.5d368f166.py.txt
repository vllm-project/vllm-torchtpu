# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX: the JAX function of the decode TP8 all-reduce op (pure JAX; the torch wiring is
``layers/adapter/tp_ar_decode.py``). One member's [rows, hidden] bf16 partial -> the sum over the 8 members of the
mesh axis: H8 (``tp8_hier_all_reduce``) when it takes the shape, else ``lax.psum`` in the same program. The row
count is static inside the op, so the choice costs nothing at run time and no torch branch depends on the bucket.
"""
from __future__ import annotations

import functools
import os

N_MEMBERS = 8


ENV = "VLLM_TPU_DECODE_TP_AR"
MAX_ROWS_ENV = "VLLM_TPU_DECODE_TP_AR_MAX_ROWS"
MODES = {"h8": "bf16", "h8x": "f32"}  # mode -> H8 wire



def mode() -> str:
    """'' (off), 'h8' or 'h8x'. Anything else raises: a typo must not silently serve stock."""
    value = os.environ.get(ENV, "").strip()
    if value and value not in MODES:
        raise ValueError(f"{ENV}={value!r}: expected one of '', {', '.join(sorted(MODES))}")
    return value


def max_rows() -> int:
    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8

    raw = os.environ.get(MAX_ROWS_ENV, "").strip()
    value = int(raw) if raw else h8.MAX_ROWS
    if not (16 <= value <= h8.MAX_ROWS) or value % 16:
        raise ValueError(f"{MAX_ROWS_ENV}={raw!r}: a multiple of 16 in [16, {h8.MAX_ROWS}]")
    return value


def compile_factors() -> dict[str, str]:
    """The compile-key factors of this lever: empty when off (so every key is unchanged)."""
    m = mode()
    if not m:
        return {}
    return {ENV: m, MAX_ROWS_ENV: str(max_rows())}


def local_body(x, *, axis: str, pairs, wire: str, max_rows_: int, interpret=False):
    """One member's all-reduce of its [rows, hidden] partial (inside shard_map). Rows are static here.
    ``interpret``: the Pallas TPU interpreter (CPU tests only)."""
    from jax import lax

    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8

    rows, hidden = x.shape
    if h8.supports(rows, hidden, x.dtype, max_rows_):
        return h8.tp8_hier_all_reduce_local(x, axis=axis, pairs=pairs, ag="fwd", wire=wire, interpret=interpret)
    return lax.psum(x, axis)


def global_fn(mesh, *, axis: str, pairs, wire: str, max_rows_: int, interpret=False):
    """The op's JAX function: x [N * rows, hidden] sharded on axis 0 over ``mesh`` -> same layout, each member's
    block replaced by the sum of all members' blocks."""
    import jax
    from jax.sharding import PartitionSpec

    spec = PartitionSpec(axis)
    body = functools.partial(local_body, axis=axis, pairs=pairs, wire=wire, max_rows_=max_rows_, interpret=interpret)

    def fn(x: jax.Array) -> jax.Array:
        return jax.shard_map(body, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False)(x)

    return fn


def pairs_from_device_ids(device_ids) -> tuple[tuple[int, int], ...]:
    """H8's (core 0, core 1) mesh indices per chip from global TPU device ids in mesh order (two TensorCores per
    v7x chip, numbered 2 * chip + core: M, T39 device list). A wrong guess costs speed only, never the sum."""
    chips: dict[int, dict[int, int]] = {}
    for i, d in enumerate(device_ids):
        chips.setdefault(int(d) // 2, {})[int(d) % 2] = i
    if len(chips) != 4 or any(sorted(v) != [0, 1] for v in chips.values()):
        raise RuntimeError(f"{ENV}: EP mesh devices {tuple(device_ids)} are not 4 chips x 2 cores")
    return tuple((chips[c][0], chips[c][1]) for c in sorted(chips))
