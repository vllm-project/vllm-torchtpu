# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX: the JAX function of the decode TP8 all-reduce op (pure JAX; the torch wiring is
``layers/adapter/tp_ar_decode.py``). One member's [rows, hidden] bf16 partial -> the sum over the 8 members of the
mesh axis: H8 (``tp8_hier_all_reduce``) when it takes the shape, else ``lax.psum`` in the same program. The row
count is static inside the op, so the choice costs nothing at run time and no torch branch depends on the bucket.
"""
from __future__ import annotations

import functools
import os

# q38 H8-FIX: module scope, not local to global_fn. `from __future__ import annotations` stores fn's annotations as
# the strings 'jax.Array'; torch_tpu jax_op evaluates them (inspect.signature(..., eval_str=True)) in this module's
# globals, so a local import raised NameError at arm() on TPU (j-db896a2a, 5d368f166).
import jax

N_MEMBERS = 8


ENV = "VLLM_TPU_DECODE_TP_AR"
MAX_ROWS_ENV = "VLLM_TPU_DECODE_TP_AR_MAX_ROWS"
# q38 L3 D68: pad row counts that are not a multiple of 16 up to the next multiple of 16 so H8 still takes them
# (the zero rows add nothing to any member's sum; the real rows are sliced back). Unset / empty = off (the old
# lax.psum fallback at those rows). Only "1" turns it on; anything else raises, so a typo cannot serve stock.
PAD16_ENV = "VLLM_TPU_DECODE_TP_AR_PAD16"
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


def pad16() -> bool:
    """D68: True when VLLM_TPU_DECODE_TP_AR_PAD16=1; False when unset or empty. Anything else raises."""
    value = os.environ.get(PAD16_ENV, "").strip()
    if value not in ("", "1"):
        raise ValueError(f"{PAD16_ENV}={value!r}: expected '' (off) or '1'")
    return value == "1"


def padded_rows(rows: int, hidden: int, dtype, max_rows_: int, pad16_: bool) -> int:
    """The row count H8 runs for a [rows, hidden] partial: ``rows`` itself when H8 takes it; with ``pad16_``, the
    next multiple of 16 when H8 takes that; else 0 (the lax.psum fallback)."""
    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8

    if h8.supports(rows, hidden, dtype, max_rows_):
        return rows
    if pad16_ and rows > 0 and rows % 16:
        up = -(-rows // 16) * 16
        if h8.supports(up, hidden, dtype, max_rows_):
            return up
    return 0


def compile_factors() -> dict[str, str]:
    """The compile-key factors of this lever: empty when off (so every key is unchanged). PAD16 joins the key only
    when it is set, so keys without it are unchanged."""
    m = mode()
    if not m:
        return {}
    factors = {ENV: m, MAX_ROWS_ENV: str(max_rows())}
    if pad16():
        factors[PAD16_ENV] = "1"
    return factors


def local_body(x, *, axis: str, pairs, wire: str, max_rows_: int, interpret=False, pad16_: bool = False):
    """One member's all-reduce of its [rows, hidden] partial (inside shard_map). Rows are static here.
    ``interpret``: the Pallas TPU interpreter (CPU tests only). ``pad16_`` (D68): rows that are not a multiple of 16
    are zero-padded to the next multiple of 16 for H8 and sliced back, when H8 takes the padded count."""
    import jax.numpy as jnp
    from jax import lax

    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8

    rows, hidden = x.shape
    run = padded_rows(rows, hidden, x.dtype, max_rows_, pad16_)
    if run == rows:
        return h8.tp8_hier_all_reduce_local(x, axis=axis, pairs=pairs, ag="fwd", wire=wire, interpret=interpret)
    if run:
        xp = jnp.pad(x, ((0, run - rows), (0, 0)))
        out = h8.tp8_hier_all_reduce_local(xp, axis=axis, pairs=pairs, ag="fwd", wire=wire, interpret=interpret)
        return out[:rows]
    return lax.psum(x, axis)


def global_fn(mesh, *, axis: str, pairs, wire: str, max_rows_: int, interpret=False, pad16_: bool = False):
    """The op's JAX function: x [N * rows, hidden] sharded on axis 0 over ``mesh`` -> same layout, each member's
    block replaced by the sum of all members' blocks."""
    from jax.sharding import PartitionSpec

    spec = PartitionSpec(axis)
    body = functools.partial(
        local_body, axis=axis, pairs=pairs, wire=wire, max_rows_=max_rows_, interpret=interpret, pad16_=pad16_
    )

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
