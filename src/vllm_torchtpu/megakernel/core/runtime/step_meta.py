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
"""Framework-free step metadata packing and position resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


@dataclass(frozen=True)
class StepMeta:
    """Packed step metadata arrays/tensors `(ids_b, pos_b, pt_b, ckpt_b, roff_b)`."""

    ids_b: Any
    pos_b: Any
    pt_b: Any
    ckpt_b: Any
    roff_b: Any

    def as_tuple(self) -> tuple[Any, Any, Any, Any, Any]:
        """Return `(ids_b, pos_b, pt_b, ckpt_b, roff_b)`."""
        return (self.ids_b, self.pos_b, self.pt_b, self.ckpt_b, self.roff_b)


@runtime_checkable
class StepMetaOps(Protocol):
    """Minimal array/tensor primitive surface for `pack_step_meta`."""

    def shape(self, x: Any) -> tuple[int, ...]: ...
    def ndim(self, x: Any) -> int: ...
    def numel(self, x: Any) -> int: ...
    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any: ...
    def slice_or_pad_1d(self, x: Any, target_len: int) -> Any: ...
    def slice_or_pad_2d(self, x: Any, target_rows: int, target_cols: int) -> Any: ...
    def zeros_i32(self, shape: tuple[int, ...], *, like: Any) -> Any: ...
    def zeros_like_i32(self, x: Any) -> Any: ...
    def arange_i32(self, start: int, stop: int, *, like: Any) -> Any: ...
    def clip(self, x: Any, lo: int, hi: int | None = None) -> Any: ...
    def take_1d(self, table_1d: Any, idx_1d: Any) -> Any: ...
    def take_col_2d(self, table_2d: Any, col_idx_1d: Any) -> Any: ...
    def expand_1d_to_2d(
        self, x_1d: Any, cols: int, *, contiguous: bool = True
    ) -> Any: ...
    def concat_cols(self, col0: Any, col1: Any) -> Any: ...
    def stack_dim0(self, xs: Sequence[Any]) -> Any: ...
    def where(self, cond: Any, x: Any, y: Any) -> Any: ...
    def gt_zero(self, x: Any) -> Any: ...
    def lt_zero(self, x: Any) -> Any: ...
    def logical_and(self, a: Any, b: Any) -> Any: ...


class JaxStepMetaOps:
    """`StepMetaOps` implementation over JAX / NumPy arrays."""

    def shape(self, x: Any) -> tuple[int, ...]:
        return tuple(int(d) for d in x.shape)

    def ndim(self, x: Any) -> int:
        return int(x.ndim)

    def numel(self, x: Any) -> int:
        return int(x.size)

    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        return jnp.asarray(x).reshape(shape)

    def slice_or_pad_1d(self, x: Any, target_len: int) -> Any:
        x1 = jnp.asarray(x, dtype=jnp.int32).reshape(-1)
        n = int(x1.shape[0])
        if n >= target_len:
            return x1[:target_len]
        return jnp.pad(x1, (0, target_len - n), constant_values=0)

    def slice_or_pad_2d(self, x: Any, target_rows: int, target_cols: int) -> Any:
        arr = jnp.asarray(x, dtype=jnp.int32)
        r, c = int(arr.shape[0]), int(arr.shape[1])
        out = arr[: min(r, target_rows), : min(c, target_cols)]
        pad_r, pad_c = max(0, target_rows - r), max(0, target_cols - c)
        if pad_r > 0 or pad_c > 0:
            out = jnp.pad(out, ((0, pad_r), (0, pad_c)), constant_values=0)
        return out

    def zeros_i32(self, shape: tuple[int, ...], *, like: Any) -> Any:
        del like
        return jnp.zeros(shape, dtype=jnp.int32)

    def zeros_like_i32(self, x: Any) -> Any:
        return jnp.zeros_like(jnp.asarray(x, dtype=jnp.int32))

    def arange_i32(self, start: int, stop: int, *, like: Any) -> Any:
        del like
        return jnp.arange(start, stop, dtype=jnp.int32)

    def clip(self, x: Any, lo: int, hi: int | None = None) -> Any:
        arr = jnp.asarray(x, dtype=jnp.int32)
        return jnp.maximum(arr, lo) if hi is None else jnp.clip(arr, lo, hi)

    def take_1d(self, table_1d: Any, idx_1d: Any) -> Any:
        t_arr = jnp.asarray(table_1d, dtype=jnp.int32).reshape(-1)
        return t_arr[jnp.asarray(idx_1d, dtype=jnp.int32)]

    def take_col_2d(self, table_2d: Any, col_idx_1d: Any) -> Any:
        t_arr = jnp.asarray(table_2d, dtype=jnp.int32)
        idx = jnp.asarray(col_idx_1d, dtype=jnp.int32)[:, None]
        return jnp.take_along_axis(t_arr, idx, axis=1)

    def expand_1d_to_2d(self, x_1d: Any, cols: int, *, contiguous: bool = True) -> Any:
        del contiguous
        arr = jnp.asarray(x_1d).reshape(-1, 1)
        return jnp.broadcast_to(arr, (int(arr.shape[0]), int(cols)))

    def concat_cols(self, col0: Any, col1: Any) -> Any:
        return jnp.concatenate(
            [jnp.asarray(col0, dtype=jnp.int32), jnp.asarray(col1, dtype=jnp.int32)],
            axis=1,
        )

    def stack_dim0(self, xs: Sequence[Any]) -> Any:
        return jnp.stack([jnp.asarray(x, dtype=jnp.int32) for x in xs], axis=0)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return jnp.where(
            jnp.asarray(cond, dtype=jnp.bool_),
            jnp.asarray(x, dtype=jnp.int32),
            jnp.asarray(y, dtype=jnp.int32),
        )

    def gt_zero(self, x: Any) -> Any:
        return jnp.asarray(x, dtype=jnp.int32) > 0

    def lt_zero(self, x: Any) -> Any:
        return jnp.asarray(x, dtype=jnp.int32) < 0

    def logical_and(self, a: Any, b: Any) -> Any:
        return jnp.asarray(a, dtype=jnp.bool_) & jnp.asarray(b, dtype=jnp.bool_)


_JAX_STEP_META_OPS = JaxStepMetaOps()


def resolve_step_positions(
    positions: Any | None,
    seq_lens: Any | None,
    *,
    num_reqs: int,
    tokens_per_seq: int,
    like: Any,
    ops: StepMetaOps | None = None,
) -> Any:
    """Resolve flat per-token step positions from `positions` or `seq_lens`."""
    eff_ops = _JAX_STEP_META_OPS if ops is None else ops
    T = int(tokens_per_seq)
    active_rows = int(num_reqs) * T
    fb_1d = (
        eff_ops.zeros_i32((max(1, int(num_reqs)),), like=like)
        if seq_lens is None
        else seq_lens
    )
    if positions is not None:
        pos_flat = (
            positions[0]
            if eff_ops.ndim(positions) == 2
            else eff_ops.reshape(positions, (-1,))
        )
        if eff_ops.numel(pos_flat) >= active_rows:
            return pos_flat[:active_rows]
    seq_l = eff_ops.slice_or_pad_1d(fb_1d, int(num_reqs))
    if T > 1:
        offsets = eff_ops.reshape(eff_ops.arange_i32(1 - T, 1, like=like), (1, T))
        return eff_ops.reshape(
            eff_ops.clip(eff_ops.reshape(seq_l, (-1, 1)) + offsets, 0),
            (-1,),
        )
    return eff_ops.clip(seq_l - 1, 0)


def check_ckpt_write_cols_distinct(ckpt_2d: Any, tokens_per_seq: int) -> None:
    """Raise `ValueError` if any active row lacks `T` distinct positive slots."""
    T = int(tokens_per_seq)
    if T <= 1:
        return
    rows = (
        ckpt_2d.tolist() if hasattr(ckpt_2d, "tolist") else np.asarray(ckpt_2d).tolist()
    )
    for row in rows:
        r_t = [int(x) for x in row[:T]]
        if r_t[0] > 0 and (any(x <= 0 for x in r_t) or len(set(r_t)) < T):
            raise ValueError(
                f"pack_step_meta requires {T} distinct positive checkpoint "
                f"slots per active sequence when tokens_per_seq={T} > 1 "
                f"(got row {r_t})"
            )


def _assert_distinct_ckpt_cols_if_host(
    ckpt_g: Any, T: int, eff_ops: Any, *, validate: bool
) -> None:
    chk = getattr(eff_ops, "check_distinct_cols_2d", None)
    if chk is not None:
        chk(ckpt_g, T)
        return
    if not (validate or pl.enable_debug_checks.value):
        return
    if isinstance(ckpt_g, jax.core.Tracer) or not isinstance(
        ckpt_g, (np.ndarray, jax.Array)
    ):
        return
    check_ckpt_write_cols_distinct(np.asarray(ckpt_g), T)


def pack_step_meta(
    input_ids: Any,
    positions: Any,
    block_tables: Any,
    recurrent_groups: Sequence[tuple[Any, Any, Any]],
    *,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
    ops: StepMetaOps | None = None,
    validate: bool = False,
) -> StepMeta:
    """Pack `(ids_b, pos_b, pt_b, ckpt_b, roff_b)` for a bucket of `bucket_t` tokens."""
    eff_ops = _JAX_STEP_META_OPS if ops is None else ops
    T, B = int(tokens_per_seq), int(bucket_t)
    if B % T != 0:
        raise ValueError(f"bucket_t={B} must be divisible by tokens_per_seq={T}")
    nseq_bkt = B // T

    bt = block_tables
    if eff_ops.ndim(bt) == 1:
        n_rows_bt = eff_ops.shape(recurrent_groups[0][0])[0] if recurrent_groups else 0
        total_bt = eff_ops.numel(bt)
        if n_rows_bt > 0 and total_bt % n_rows_bt == 0:
            bt = eff_ops.reshape(bt, (n_rows_bt, total_bt // n_rows_bt))
        elif max_pages_per_seq > 0 and total_bt % max_pages_per_seq == 0:
            bt = eff_ops.reshape(bt, (total_bt // max_pages_per_seq, max_pages_per_seq))
        else:
            raise ValueError(
                f"1-D block_tables of length {total_bt} is not divisible by "
                f"s0.shape[0]={n_rows_bt} or max_pages_per_seq={max_pages_per_seq}"
            )
    pt_b = eff_ops.slice_or_pad_2d(bt, nseq_bkt, max_pages_per_seq)

    ckpt_rows: list[Any] = []
    roff_rows: list[Any] = []
    for s_raw, c_raw, r_raw in recurrent_groups:
        s_src = s_raw[:, 0] if eff_ops.ndim(s_raw) == 2 else s_raw
        s_b = eff_ops.slice_or_pad_1d(s_src, nseq_bkt)
        if has_roff:
            r_1d = eff_ops.reshape(r_raw, (-1,))
            s_idx = eff_ops.clip(s_b, 0, eff_ops.numel(r_1d) - 1)
            roff_g = eff_ops.take_1d(r_1d, s_idx)
        else:
            roff_g = eff_ops.zeros_i32((nseq_bkt,), like=s_b)
        if T > 1:
            if (
                not use_ckpt_2d
                or eff_ops.ndim(c_raw) != 2
                or eff_ops.shape(c_raw)[1] < T
            ):
                raise ValueError(
                    f"pack_step_meta requires 2-D ckpt_indices with >= T={T} "
                    f"columns when tokens_per_seq={T} > 1 (got shape "
                    f"{eff_ops.shape(c_raw)}, use_ckpt_2d={use_ckpt_2d})"
                )
            ckpt_g = eff_ops.slice_or_pad_2d(c_raw, nseq_bkt, T)
        else:
            if (
                use_ckpt_2d
                and has_roff
                and eff_ops.ndim(c_raw) == 2
                and eff_ops.shape(c_raw)[1] > 1
            ):
                c_cols = eff_ops.shape(c_raw)[1]
                c_b = eff_ops.slice_or_pad_2d(c_raw, nseq_bkt, c_cols)
                r_idx = eff_ops.clip(roff_g, 0, c_cols - 1)
                read_col = eff_ops.take_col_2d(c_b, r_idx)
            else:
                read_col = eff_ops.reshape(s_b, (nseq_bkt, 1))
            write_col = eff_ops.reshape(s_b, (nseq_bkt, 1))
            ckpt_g = eff_ops.concat_cols(read_col, write_col)
            roff_g = eff_ops.where(
                eff_ops.lt_zero(roff_g), roff_g, eff_ops.zeros_like_i32(roff_g)
            )

        act_g = eff_ops.logical_and(eff_ops.gt_zero(s_b), eff_ops.gt_zero(ckpt_g[:, 0]))
        ckpt_g = eff_ops.where(
            eff_ops.reshape(act_g, (nseq_bkt, 1)),
            ckpt_g,
            eff_ops.zeros_like_i32(ckpt_g),
        )
        if T > 1:
            _assert_distinct_ckpt_cols_if_host(
                ckpt_g, T, eff_ops, validate=bool(validate)
            )
        roff_g = eff_ops.where(act_g, roff_g, eff_ops.zeros_like_i32(roff_g))
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    pos_1d = positions[0] if eff_ops.ndim(positions) == 2 else positions
    ids_padded = eff_ops.slice_or_pad_1d(input_ids, B)
    pos_padded = eff_ops.slice_or_pad_1d(pos_1d, B)
    if ckpt_rows:
        ckpt_b = eff_ops.stack_dim0(ckpt_rows)
        roff_b = eff_ops.stack_dim0(roff_rows)
        act_seq = eff_ops.gt_zero(ckpt_b[0, :, 0])
        act_tok = eff_ops.reshape(
            eff_ops.expand_1d_to_2d(act_seq, T, contiguous=False), (B,)
        )
        zero_b = eff_ops.zeros_i32((B,), like=ids_padded)
        ids_b = eff_ops.where(act_tok, ids_padded, zero_b)
        pos_b = eff_ops.where(act_tok, pos_padded, zero_b)
    else:
        ckpt_cols = 2 if T == 1 else T
        ckpt_b = eff_ops.zeros_i32((0, nseq_bkt, ckpt_cols), like=input_ids)
        roff_b = eff_ops.zeros_i32((0, nseq_bkt), like=input_ids)
        ids_b, pos_b = ids_padded, pos_padded
    return StepMeta(ids_b=ids_b, pos_b=pos_b, pt_b=pt_b, ckpt_b=ckpt_b, roff_b=roff_b)
