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
"""W_UV and o_proj of an MLA layer on a rank's own rows, every head.

Under tensor parallelism rank `r` holds o_proj's input rows `[r * kb, (r +
1) * kb)` (its heads x v_head_dim) as per-channel fp8 with its own scale,
and W_UV for the same heads. A rank whose attention produced every head of
its own rows (token-parallel attention on sequence-parallel rows) can finish
the layer with every rank's shard instead of an all-to-all of outputs and a
reduce-scatter of partial sums. The shards do not depend on the step, so a
layer's gather can be issued while the layer before it runs.

A rank's shards travel as one buffer (`PackedOProj`): o_proj's fp8 rows,
then W_UV and the scale. The product keeps each shard's arithmetic
(`rows_oproj`): W_UV of the shard's heads, the activation quantized per row
exactly as each rank quantizes its slice for its partial, and the partial
scaled by that shard's weight scale. Only the sum over shards moves, from a
bf16 reduce-scatter into an f32 accumulator.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FP8 = jnp.float8_e4m3fn
_FP8_MAX = float(jnp.finfo(FP8).max)
# Row multiple of a packed buffer: the fp8 sublane tile.
_ROW_ALIGN = 32


@dataclasses.dataclass(frozen=True)
class PackedOProj:
    """Layout of one rank's packed shards, fp8 `[rows, n_out]`: o_proj's
    `[k_shard, n_out]` rows, then the bytes of W_UV `[heads, kv_lora_rank,
    v_head_dim]` bf16 and of the o_proj scale `[n_out]` f32. Every byte
    travels as fp8: an all-gather of u8 is not offloaded to the SparseCore."""

    shards: int
    k_shard: int
    n_out: int
    heads: int
    kv_lora_rank: int
    v_head_dim: int

    def __post_init__(self):
        if self.k_shard != self.heads * self.v_head_dim:
            raise ValueError(f"o_proj rows per shard != heads x v_head_dim: {self}")
        if self.k_shard % _ROW_ALIGN:
            raise ValueError(f"k_shard must be a multiple of {_ROW_ALIGN}: {self}")

    @property
    def uv_bytes(self) -> int:
        return self.heads * self.kv_lora_rank * self.v_head_dim * 2

    @property
    def scale_bytes(self) -> int:
        return self.n_out * 4

    @property
    def rows(self) -> int:
        tail = -(-(self.uv_bytes + self.scale_bytes) // self.n_out)
        return self.k_shard + -(-tail // _ROW_ALIGN) * _ROW_ALIGN


def pack_shard(
    geom: PackedOProj, w_o: jax.Array, w_uv: jax.Array, w_scale: jax.Array
) -> jax.Array:
    """One rank's shards as `geom`'s fp8 `[rows, n_out]` buffer."""
    if w_o.shape != (geom.k_shard, geom.n_out) or w_o.dtype != FP8:
        raise ValueError(f"o_proj shard {w_o.shape} {w_o.dtype} does not fit {geom}")
    uv = (geom.heads, geom.kv_lora_rank, geom.v_head_dim)
    if w_uv.shape != uv or w_uv.dtype != jnp.bfloat16:
        raise ValueError(f"W_UV shard {w_uv.shape} {w_uv.dtype} does not fit {geom}")
    if w_scale.shape != (geom.n_out,):
        raise ValueError(f"o_proj scale {w_scale.shape} does not fit {geom}")
    tail = jnp.concatenate(
        [
            lax.bitcast_convert_type(w_uv, jnp.uint8).reshape(-1),
            lax.bitcast_convert_type(w_scale.astype(jnp.float32), jnp.uint8).reshape(
                -1
            ),
        ]
    )
    tail_rows = geom.rows - geom.k_shard
    tail = jnp.pad(tail, (0, tail_rows * geom.n_out - tail.shape[0]))
    tail = lax.bitcast_convert_type(tail.reshape(tail_rows, geom.n_out), FP8)
    return jnp.concatenate([w_o, tail])


def ring_order(devices) -> tuple[int, ...]:
    """Mesh indices of `devices` (a 1-D mesh's order) along a ring whose
    neighbours share a chip or an ICI link: the chips of the slice in Gray
    code order, both TensorCores of a chip next to each other."""
    chips = sorted({tuple(d.coords) for d in devices})
    dims = [sorted({c[i] for c in chips}) for i in range(len(chips[0]))]
    if any(len(v) > 2 for v in dims) or len(chips) != 2 ** len(dims):
        raise NotImplementedError(f"ring order for chip grid {dims}")
    order = []
    for g in range(len(chips)):
        gray = g ^ (g >> 1)
        order.append(tuple(v[(gray >> i) & 1] for i, v in enumerate(dims)))
    index = {(tuple(d.coords), d.core_on_chip): i for i, d in enumerate(devices)}
    cores = sorted({d.core_on_chip for d in devices})
    return tuple(index[(c, k)] for c in order for k in cores)


def _ring_all_gather(
    packed: jax.Array, axis_name: str, ring: tuple[int, ...]
) -> jax.Array:
    """`lax.all_gather(packed, axis_name)` by collective-permutes around
    `ring`, both ways: the TensorCores' DMA engines move the shards and the
    SparseCores stay free."""
    n = len(ring)
    pos = {d: i for i, d in enumerate(ring)}
    to_next = [(d, ring[(pos[d] + 1) % n]) for d in ring]
    to_prev = [(d, ring[(pos[d] - 1) % n]) for d in ring]
    rank = lax.all_to_all(jnp.arange(n, dtype=jnp.int32), axis_name, 0, 0, tiled=True)
    me = jnp.asarray([pos[i] for i in range(n)], jnp.int32)[rank[0]]
    ring_arr = jnp.asarray(ring, jnp.int32)
    out = jnp.zeros((n, *packed.shape), packed.dtype)
    out = lax.dynamic_update_index_in_dim(out, packed, ring_arr[me], 0)
    fwd = bwd = packed
    for k in range(1, n // 2 + 1):
        fwd = lax.ppermute(fwd, axis_name, to_next)
        out = lax.dynamic_update_index_in_dim(out, fwd, ring_arr[(me - k) % n], 0)
    for k in range(1, (n + 1) // 2):
        bwd = lax.ppermute(bwd, axis_name, to_prev)
        out = lax.dynamic_update_index_in_dim(out, bwd, ring_arr[(me + k) % n], 0)
    return out


def gather_shards(
    geom: PackedOProj,
    w_o: jax.Array,
    w_uv: jax.Array,
    w_scale: jax.Array,
    after: jax.Array,
    axis_name: str,
    ring: tuple[int, ...] | None = None,
) -> jax.Array:
    """Every rank's packed shards, `[shards, rows, n_out]` (inside a
    `shard_map`): one all-gather, or with `ring` (`ring_order`)
    collective-permutes around it. Packing and gather start no earlier than
    `after` exists: a gather that reads only weights would otherwise be
    hoisted to the start of the step, every layer's at once."""
    w_o, w_uv, w_scale, _ = lax.optimization_barrier((w_o, w_uv, w_scale, after))
    packed = pack_shard(geom, w_o, w_uv, w_scale)
    if ring is None:
        return lax.all_gather(packed, axis_name)
    return _ring_all_gather(packed, axis_name, ring)


def unpack(
    geom: PackedOProj, gathered: jax.Array, after: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`(w_o, w_uv, w_scale)` of every rank from `gathered`: w_o is the
    whole buffer (`rows_oproj` reads each shard's first `k_shard` rows in
    place), w_uv `[shards * heads, kv_lora_rank, v_head_dim]` bf16 and
    w_scale `[shards, n_out]` f32. The W_UV / scale copies wait for `after`
    (the attention output they are applied to): scheduled any earlier, they
    would make the gather finish early too."""
    n = geom.shards
    if gathered.shape != (n, geom.rows, geom.n_out) or gathered.dtype != FP8:
        raise ValueError(f"gathered {gathered.shape} {gathered.dtype} != {geom}")
    gathered, _ = lax.optimization_barrier((gathered, after))
    tail = lax.bitcast_convert_type(gathered[:, geom.k_shard :], jnp.uint8)
    tail = tail.reshape(n, -1)
    uv = tail[:, : geom.uv_bytes].reshape(
        n, geom.heads, geom.kv_lora_rank, geom.v_head_dim, 2
    )
    w_uv = lax.bitcast_convert_type(uv, jnp.bfloat16).reshape(
        n * geom.heads, geom.kv_lora_rank, geom.v_head_dim
    )
    sc = tail[:, geom.uv_bytes : geom.uv_bytes + geom.scale_bytes]
    w_scale = lax.bitcast_convert_type(sc.reshape(n, geom.n_out, 4), jnp.float32)
    return gathered, w_uv, w_scale


def _quantize_rows(v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`quantize_tensor(v, FP8)` along the last axis, op for op: an f32 scale
    `max|v| / 448` (1 for an all-zero row), then `v / scale` clipped and cast
    once. Returns (fp8 values, f32 scale `[rows, 1]`)."""
    amax = jnp.max(jnp.abs(v), axis=-1, keepdims=True).astype(jnp.float32)
    scale = amax / _FP8_MAX
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(v.astype(jnp.float32) / scale, -_FP8_MAX, _FP8_MAX).astype(FP8)
    return q, scale


def _shard_partial(lat_rows, uv, heads):
    """W_UV of one shard's heads, in the activation dtype as the per-rank
    bmm: `lat_rows(h)` gives head h's rows `[r, kv_lora_rank]`, `uv`
    `[heads, kv_lora_rank, v_head_dim]`; returns `[r, heads * v_head_dim]`."""
    return jnp.concatenate(
        [
            jnp.dot(lat_rows(h), uv[h], preferred_element_type=jnp.float32).astype(
                uv.dtype
            )
            for h in range(heads)
        ],
        axis=1,
    )


def _rows_oproj_kernel(
    lat_ref, uv_ref, w_ref, ws_ref, out_ref, acc_ref, *, heads, chunks
):
    s = pl.program_id(1)
    head_tile = lat_ref.shape[1]
    n_out = out_ref.shape[1]
    width = n_out // chunks

    def shard(base: int):
        def lat_rows(h):
            return lat_ref[:, base + h, :]

        xq, xs = _quantize_rows(_shard_partial(lat_rows, uv_ref, heads))
        for c in range(chunks):
            cols = slice(c * width, (c + 1) * width)
            part = jnp.dot(xq, w_ref[:, cols], preferred_element_type=jnp.float32)
            # Same order as the per-rank product: (dot * x_scale) * w_scale.
            part = part * xs * ws_ref[pl.ds(s, 1), cols]

            @pl.when(s == 0)
            def _():
                acc_ref[:, cols] = part

            @pl.when(s > 0)
            def _():
                acc_ref[:, cols] += part

    # The block holds a whole sublane tile of heads: this shard's start at a
    # static offset in it, one branch per offset.
    per_tile = head_tile // heads
    for i in range(per_tile):
        pl.when(s % per_tile == i)(functools.partial(shard, i * heads))

    @pl.when(s == pl.num_programs(1) - 1)
    def _():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


# Heads per latent block: a whole sublane tile of the [R, H, kv_lora_rank]
# latent, so the kernel reads the attention output in its own layout.
_HEAD_TILE = 8


@functools.partial(
    jax.jit, static_argnames=("k_shard", "block_rows", "chunks", "interpret")
)
def rows_oproj(
    out_latent: jax.Array,
    w_uv: jax.Array,
    w_o: jax.Array,
    w_scale: jax.Array,
    *,
    k_shard: int,
    block_rows: int = 256,
    chunks: int = 4,
    interpret: bool = False,
) -> jax.Array:
    """W_UV and o_proj of every head of a rank's rows, one kernel:
    `out_latent` `[R, H, kv_lora_rank]` -> `[R, n_out]` in its dtype.

    w_uv `[H, kv_lora_rank, v_head_dim]`, w_o `[shards, rows >= k_shard,
    n_out]` fp8 (shard s's o_proj in its first `k_shard` rows, e.g. `unpack`'s
    view of a gathered buffer), w_scale `[shards, n_out]` f32. For shard s
    (heads `[s * H / shards, (s + 1) * H / shards)`) the kernel runs what rank
    s runs for its partial (W_UV of those heads, per-row fp8 quantization,
    the fp8 product scaled by row and channel) and sums the partials in f32.
    """
    rows, heads_all, kv_lora_rank = out_latent.shape
    shards, w_rows, n_out = w_o.shape
    heads = heads_all // shards
    v_head_dim = w_uv.shape[-1]
    head_tile = max(heads, _HEAD_TILE)
    if (
        heads * shards != heads_all
        or head_tile % heads
        or heads_all % head_tile
        or w_uv.shape != (heads_all, kv_lora_rank, v_head_dim)
        or k_shard != heads * v_head_dim
        or w_rows < k_shard
        or w_scale.shape != (shards, n_out)
    ):
        raise ValueError(
            f"rows_oproj shapes: latent {out_latent.shape} w_uv {w_uv.shape} "
            f"w_o {w_o.shape} w_scale {w_scale.shape} k_shard {k_shard}"
        )
    block_rows = min(block_rows, rows)
    if rows % block_rows or n_out % (chunks * 128):
        raise ValueError(f"rows {rows} / block_rows {block_rows} / n_out {n_out}")
    per_tile = head_tile // heads
    itemsize = out_latent.dtype.itemsize
    # Double-buffered inputs and output, the f32 accumulator, slack.
    vmem = 2 * (
        block_rows * head_tile * kv_lora_rank * itemsize
        + heads * kv_lora_rank * v_head_dim * itemsize
        + k_shard * n_out
        + shards * n_out * 4
        + block_rows * n_out * itemsize
    )
    vmem += block_rows * n_out * 4 + (8 << 20)
    kernel = functools.partial(_rows_oproj_kernel, heads=heads, chunks=chunks)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(rows // block_rows, shards),
            in_specs=[
                pl.BlockSpec(
                    (block_rows, head_tile, kv_lora_rank),
                    lambda r, s: (r, s // per_tile, 0),
                ),
                pl.BlockSpec((heads, kv_lora_rank, v_head_dim), lambda r, s: (s, 0, 0)),
                pl.BlockSpec((pl.squeezed, k_shard, n_out), lambda r, s: (s, 0, 0)),
                pl.BlockSpec((shards, n_out), lambda r, s: (0, 0)),
            ],
            out_specs=pl.BlockSpec((block_rows, n_out), lambda r, s: (r, 0)),
            scratch_shapes=[pltpu.VMEM((block_rows, n_out), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((rows, n_out), out_latent.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=int(min(vmem, 60 << 20)),
        ),
        name=f"rows_oproj_s{shards}_r{block_rows}",
        interpret=interpret,
    )(out_latent, w_uv, w_o, w_scale.astype(jnp.float32))


def rows_oproj_ref(
    out_latent: jax.Array,
    w_uv: jax.Array,
    w_o: jax.Array,
    w_scale: jax.Array,
    *,
    k_shard: int,
) -> jax.Array:
    """XLA `rows_oproj` (same arithmetic; f32 out, before the final cast)."""
    heads_all = out_latent.shape[1]
    shards = w_o.shape[0]
    heads = heads_all // shards
    acc = None
    for s in range(shards):
        base = s * heads
        v = _shard_partial(
            lambda h, base=base: out_latent[:, base + h],
            w_uv[base : base + heads],
            heads,
        )
        xq, xs = _quantize_rows(v)
        part = jnp.dot(xq, w_o[s, :k_shard], preferred_element_type=jnp.float32)
        part = part * xs * w_scale[s][None, :]
        acc = part if acc is None else acc + part
    return acc
