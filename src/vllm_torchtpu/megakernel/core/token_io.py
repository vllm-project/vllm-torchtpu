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
"""Token step I/O (`ExecSpec.tokens`): the step is the whole decode step.

With `TokenIO` set, the kernel takes the runner's step tensors as they are
(`input_ids`, `positions`, `block_tables`, `seq_lens`, int32, any padding) and
returns each row's greedy token id instead of the final hidden state:

- staging: DMAs the rows' ids, RoPE positions (row 0 of M-RoPE positions),
  sequence lengths and page-table rows into SMEM; the KV index of row `t` of
  sequence `s` is `seq_lens[s] - T + t`;
- embedding: the rank that owns a row's id DMAs its embedding row from its
  vocabulary shard, the others contribute zeros, and the in-kernel all-reduce
  sums them (exact: one nonzero bf16 term);
- LM head: after the final norm, the vocabulary shard streams through the
  dense ring as `n_tiles` tiles of `tile` columns (`TwoRing.add_tail_segment`);
  each tile's logits are rounded to bf16 and the running `(max, index)` keeps
  the first maximum; the ranks then exchange their `(max, index)` and the
  first rank holding the maximum wins, so ties go to the lowest token id, as
  `argmax` over the gathered bf16 logits.

The LM head weight is packed `[n_tiles, H, tile]` (pad columns are never
selected); the embedding is the runner's own `[shard, H]` table.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .errors import MegakernelConfigError
from .ops import BF16, F32, I32, round_bf16, round_up

if TYPE_CHECKING:
    from .exec_spec import ExecSpec

LANES = 128
EMBED_WEIGHT = "embed"
HEAD_WEIGHT = "lm_head"
HEAD_KIND = HEAD_WEIGHT
# The runner's RoPE cos / sin cache `[rope_rows, rope_cols]` (`cos | sin`
# halves per position), a replicated extra input of the module that rotates.
ROPE_CACHE_INPUT = "rope_cache"


@dataclass(frozen=True)
class TokenIO:
    """Build-time settings of the token step I/O (`ExecSpec.tokens`)."""

    vocab: int  # real vocabulary size (ids at or past it are never chosen)
    shard: int  # embedding / LM head vocabulary rows per rank
    tile: int  # LM head vocabulary columns per streamed tile (multiple of 128)
    rope_rows: int = 0  # rows of the RoPE cos / sin cache input (0: none)
    rope_cols: int = 0  # its columns (the rotated dims)
    rope_dtype: Any = F32  # its dtype

    def __post_init__(self) -> None:
        if self.tile <= 0 or self.tile % LANES:
            raise MegakernelConfigError(
                f"TokenIO.tile={self.tile} must be a positive multiple of {LANES}"
            )
        if self.shard <= 0 or self.vocab <= 0:
            raise MegakernelConfigError(f"TokenIO needs vocab and shard > 0: {self}")

    @property
    def n_tiles(self) -> int:
        return -(-self.shard // self.tile)


def head_tile(H: int, shard: int, bank: int, itemsize: int = 2) -> int:
    """LM head tile width: the fewest `(H, tile)` tiles that fit a dense-ring
    bank, balanced so the last one pads least (18992 x 5120 bf16 in 8 MiB
    banks: 25 tiles of 768)."""
    widest = (bank // (H * itemsize)) // LANES * LANES
    if widest <= 0:
        raise MegakernelConfigError(
            f"an LM head column block of H={H} does not fit a {bank}-byte bank"
        )
    n = -(-shard // widest)
    return round_up(-(-shard // n), LANES)


def token_io(ex: ExecSpec) -> TokenIO | None:
    """The `TokenIO` of `ex` (`None`: the step maps hidden states)."""
    tio = ex.tokens
    if tio is not None and not isinstance(tio, TokenIO):
        raise TypeError(f"ExecSpec.tokens must be a TokenIO, got {type(tio).__name__}")
    return tio


def with_tokens(ex: ExecSpec, tio: TokenIO | None) -> ExecSpec:
    return dataclasses.replace(ex, tokens=tio)


def weight_shapes(tio: TokenIO, H: int) -> dict[str, Any]:
    """Per-rank shapes of the embedding table and the packed LM head tiles."""
    return {
        EMBED_WEIGHT: ((tio.shard, H), BF16),
        HEAD_WEIGHT: ((tio.n_tiles, H, tio.tile), BF16),
    }


def pack_head(w: Any, tio: TokenIO) -> Any:
    """`[shard, H]` LM head rows (any array API with `.T` / `pad` via jnp) ->
    `[n_tiles, H, tile]` tiles, zero-padded columns."""
    shard, H = w.shape
    wt = jnp.pad(jnp.asarray(w).T, ((0, 0), (0, tio.n_tiles * tio.tile - shard)))
    return wt.reshape(H, tio.n_tiles, tio.tile).transpose(1, 0, 2)


def scratch(ex: ExecSpec, H: int, max_pages: int) -> dict[str, Any]:
    """SMEM staging of the step inputs, the embedding row window and the
    `(max, index)` exchange buffers."""
    B, NS, MB, R = ex.B, ex.NS, ex.MB, ex.R
    tio = token_io(ex)
    assert tio is not None
    rope = (
        {
            "tio_rope": pltpu.VMEM(
                (_window(tio.rope_dtype), tio.rope_cols), tio.rope_dtype
            )
        }
        if tio.rope_rows
        else {}
    )
    return {
        **rope,
        "tio_ids": pltpu.SMEM((B,), I32),
        "tio_rpos": pltpu.SMEM((B,), I32),
        "tio_seq": pltpu.SMEM((NS,), I32),
        "tio_kv": pltpu.SMEM((B,), I32),
        "tio_pt": pltpu.SMEM((NS, max_pages), I32),
        "tio_sem": pltpu.SemaphoreType.DMA((4,)),
        "tio_emb": pltpu.VMEM((_window(BF16), H), BF16),
        "tio_best": pltpu.VMEM((MB, LANES), F32),
        "tio_recv": pltpu.VMEM((R, MB, LANES), F32),
        "tio_rs": pltpu.SMEM((R, MB, LANES), F32),
        "tio_ssem": pltpu.SemaphoreType.DMA((max(R - 1, 1),)),
        "tio_rsem": pltpu.SemaphoreType.DMA((max(R - 1, 1),)),
    }


def _window(dtype: Any) -> int:
    """Rows of one `(8, 128)` 32-bit tile of `dtype` (16 for bf16)."""
    return 8 * (4 // jnp.dtype(dtype).itemsize)


def rope_rows(ctx: Any, cache: Any, rcs: Any) -> None:
    """`rcs [2, MR, half]` = the cos / sin of each row's RoPE position from the
    runner's cache, rounded to bf16 as the runner rotates with it."""
    ex = ctx.ex
    half = rcs.shape[-1]
    shape = rcs.shape[1:]
    rows = lax.broadcasted_iota(I32, shape, 0)
    cos = jnp.zeros(shape, F32)
    sin = jnp.zeros(shape, F32)
    for b in range(ex.B):
        row = round_bf16(gather_row(cache, ctx.rope_pos[b], ctx.scr["tio_rope"]))
        cos = jnp.where(rows == b, row[:, :half], cos)
        sin = jnp.where(rows == b, row[:, half : 2 * half], sin)
    rcs[0] = cos
    rcs[1] = sin


def _row_window(n_rows: int, row: Any, w: int) -> tuple[Any, Any]:
    """Start and offset of the `w`-row tile window holding `row` (`n_rows >= w`)."""
    a = jnp.minimum(lax.div(row, w) * w, n_rows - w)
    return pl.multiple_of(a, 8), row - a


def gather_row(table: Any, row: Any, stage: Any) -> Any:
    """Row `row` of HBM `table [N, C]` as a `(1, C)` f32 value, through the
    whole-tile window `stage [W, C]` (a single-row DMA is not tile-aligned)."""
    w = stage.shape[0]
    if table.shape[0] < w:
        raise MegakernelConfigError(
            f"row gather needs >= {w} table rows: {table.shape}"
        )
    a, off = _row_window(table.shape[0], row, w)
    pltpu.sync_copy(table.at[pl.ds(a, w)], stage)
    rows = lax.broadcasted_iota(I32, stage.shape, 0)
    return jnp.sum(
        jnp.where(rows == off, stage[...].astype(F32), 0.0), axis=0, keepdims=True
    )


def stage_inputs(
    ctx: Any, ids: Any, positions: Any, tables: Any, seq_lens: Any
) -> None:
    """DMA the step tensors' used entries into SMEM; derive the KV indices.

    `positions` is `[P]` or `[3, P]` (M-RoPE: row 0 is the text position);
    `tables` is `[S, W]` or flat `[S * W]` with `S = seq_lens.shape[0]`.
    """
    ex, scr = ctx.ex, ctx.scr
    B, NS, T = ex.B, ex.NS, ex.T
    sem = scr["tio_sem"]
    pt = scr["tio_pt"]
    max_pages = pt.shape[1]
    S = seq_lens.shape[0]
    width = math.prod(tables.shape) // S
    n_pt = min(width, max_pages)
    pos_row = positions.at[0] if positions.ndim == 2 else positions

    def pt_row(s: int) -> Any:
        if tables.ndim == 2:
            return tables.at[s, pl.ds(0, n_pt)]
        return tables.at[pl.ds(s * width, n_pt)]

    cps = [
        pltpu.make_async_copy(ids.at[pl.ds(0, B)], scr["tio_ids"], sem.at[0]),
        pltpu.make_async_copy(pos_row.at[pl.ds(0, B)], scr["tio_rpos"], sem.at[1]),
        pltpu.make_async_copy(seq_lens.at[pl.ds(0, NS)], scr["tio_seq"], sem.at[2]),
    ]
    rows = [
        pltpu.make_async_copy(pt_row(s), pt.at[s, pl.ds(0, n_pt)], sem.at[3])
        for s in range(NS)
    ]
    for c in cps + rows:
        c.start()
    for c in cps + rows:
        c.wait()
    for s in range(NS):
        if n_pt < max_pages:

            def zero(j: Any, c: Any, s: int = s) -> Any:
                pt[s, j] = jnp.int32(0)
                return c

            lax.fori_loop(n_pt, max_pages, zero, 0)
        for t in range(T):
            scr["tio_kv"][s * T + t] = scr["tio_seq"][s] - (T - t)


def embed(ctx: Any, table: Any, ar_slot: int) -> None:
    """`res` rows = the embeddings of the staged ids (owner rank's row, summed
    over ranks by the all-reduce on `ar_slot`, whose sends it drains)."""
    ex, scr, comm = ctx.ex, ctx.scr, ctx.comm
    tio = token_io(ex)
    assert tio is not None
    shard = table.shape[0]
    MB, H = ex.MB, table.shape[1]
    x = jnp.zeros((MB, H), F32)
    rows = lax.broadcasted_iota(I32, (MB, H), 0)
    for b in range(ex.B):
        local = scr["tio_ids"][b] - ctx.me * shard
        mine = jnp.logical_and(local >= 0, local < shard)
        row = gather_row(
            table, jnp.maximum(jnp.minimum(local, shard - 1), 0), scr["tio_emb"]
        )
        x = jnp.where(jnp.logical_and(rows == b, mine), row, x)
    scr["res"][...] = jnp.zeros(scr["res"].shape, F32)
    comm.add_residual(comm.allreduce(ar_slot, x, True))
    if comm.comm:
        for rnd_i in comm.cp.AR_ROUNDS:
            for d in comm.ar_copies(ar_slot, rnd_i):
                d.wait_send()


def head(ctx: Any, x: Any, out_ref: Any) -> None:
    """Greedy token of each row of the normed hidden `x [MB, H]` bf16 into
    `out_ref [B]` int32 (see the module docstring)."""
    ex, scr, stream = ctx.ex, ctx.scr, ctx.stream
    tio = token_io(ex)
    assert tio is not None
    MB, R, me = ex.MB, ex.R, ctx.me
    TN = tio.tile
    stream.end_layers()
    g0 = stream.tail_start()
    cols = lax.broadcasted_iota(I32, (MB, TN), 1)
    big = 2**30

    def tile(j: Any, carry: Any) -> Any:
        best_v, best_i = carry
        bank = lax.rem(g0 + j, stream.sp.NBD)
        w = stream.dview(HEAD_KIND, bank)
        pltpu.make_async_copy(ctx.W[HEAD_WEIGHT].at[0], w, scr["dsem"].at[bank]).wait()
        logits = round_bf16(stream.mm(x, w[...]))
        gid = cols + j * TN
        logits = jnp.where(gid < tio.shard, logits, -jnp.inf)
        m = jnp.max(logits, axis=1, keepdims=True)
        i = jnp.min(jnp.where(logits == m, gid, big), axis=1, keepdims=True)
        stream.dgive(g0 + j)
        better = m > best_v
        return jnp.where(better, m, best_v), jnp.where(better, i, best_i)

    init = (jnp.full((MB, 1), -jnp.inf, F32), jnp.zeros((MB, 1), I32))
    best_v, best_i = lax.fori_loop(0, tio.n_tiles, tile, init)
    gidx = best_i + me * tio.shard
    best_v = jnp.where(gidx < tio.vocab, best_v, -jnp.inf)
    lane = lax.broadcasted_iota(I32, (MB, LANES), 1)
    scr["tio_best"][...] = jnp.where(
        lane == 0, best_v, jnp.where(lane == 1, gidx.astype(F32), 0.0)
    )
    recv = scr["tio_recv"]
    recv[me] = scr["tio_best"][...]
    if R > 1 and "comm" not in ctx.skip:
        descs = [
            pltpu.make_async_remote_copy(
                src_ref=scr["tio_best"],
                dst_ref=recv.at[me],
                send_sem=scr["tio_ssem"].at[off - 1],
                recv_sem=scr["tio_rsem"].at[off - 1],
                device_id=(me ^ off,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            for off in range(1, R)
        ]
        for d in descs:
            d.start()
        for d in descs:
            d.wait_recv()
        for d in descs:
            d.wait_send()
    rs = scr["tio_rs"]
    pltpu.sync_copy(recv, rs)
    exchanged = R > 1 and "comm" not in ctx.skip
    for b in range(ex.B):
        r0 = 0 if exchanged else me
        v, i = rs[r0, b, 0], rs[r0, b, 1]
        for r in range(1, R if exchanged else 1):
            better = rs[r, b, 0] > v
            v = jnp.where(better, rs[r, b, 0], v)
            i = jnp.where(better, rs[r, b, 1], i)
        out_ref[b] = i.astype(I32)


__all__ = [
    "EMBED_WEIGHT",
    "HEAD_KIND",
    "HEAD_WEIGHT",
    "ROPE_CACHE_INPUT",
    "TokenIO",
    "embed",
    "gather_row",
    "head",
    "head_tile",
    "pack_head",
    "rope_rows",
    "scratch",
    "stage_inputs",
    "token_io",
    "weight_shapes",
    "with_tokens",
]
