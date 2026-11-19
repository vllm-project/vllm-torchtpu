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
"""Lossy TP all-reduce of large bf16 [T, H] activations with an fp8 (e4m3)
wire (TPU_TP_FP8_ALL_REDUCE; distributed/pallas_all_reduce.py routes to it).

A bf16 all-reduce of a prefill chunk is bytes-bound (SparseCore offloaded,
TensorCore blocked), so it is replaced by collectives that move e4m3 values
with dynamic f32 scales, about half the bytes. Per rank, inside shard_map
over a 1-D mesh axis of n ranks:

"a2a"   quantize -> all-to-all (rank j receives every rank's row block j)
        -> dequantize + f32 sum of the n partials -> requantize
        -> all-gather. Two fp8 roundings; ~1/2 of the bf16 bytes.
"rs"    bf16 reduce-scatter -> quantize -> all-gather. One rounding;
        ~3/4 of the bytes.
"a2a16" fp8 all-to-all + f32 sum -> bf16 all-gather. One rounding (of the
        partials); ~3/4 of the bytes.

Measured on tpu7x 2x2x1 TP8, [8192, 6144] bf16, c48 served LIBTPU flags
(scripts/benchmark_fp8_all_reduce.py): SparseCore all-reduce 1.03 ms per
call, "rs" 0.98-0.99, bf16 reduce-scatter + all-gather 1.15, "a2a" 1.98-2.17
and "a2a16" 1.88-2.33 (the all-to-all is the slow part, SparseCore offload or
not). Without SparseCore all-gather offload (c64 flags) "rs" takes 1.23.

Scales: "row" (one per row) or "g128" (one per 128 columns of a row), from
the amax of the finite values, so a NaN still propagates through its value.
The e4m3 values and the scales' bytes travel as one int8 payload
[rows, H + pad] (a uint8 payload hangs the SparseCore-offloaded all-gather;
int8 does not); pad rounds the scale bytes up to 128 lanes.

Every rank gets the same bits (the gathered values are dequantized the same
way everywhere). Rows must be a multiple of n (`applies`).
"""

import jax
import jax.numpy as jnp
from jax import lax

FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
MODES = ("a2a", "rs", "a2a16")
GRANULARITIES = ("row", "g128")
_GROUP = 128
_LANES = 128


def num_groups(hidden: int, granularity: str) -> int:
    if granularity == "row":
        return 1
    if granularity == "g128":
        if hidden % _GROUP:
            raise ValueError(f"g128 scales need hidden % 128 == 0, got {hidden}")
        return hidden // _GROUP
    raise ValueError(f"granularity must be one of {GRANULARITIES}: {granularity!r}")


def scale_bytes(hidden: int, granularity: str) -> int:
    """int8 lanes the scales take in a payload row (rounded up to 128)."""
    raw = 4 * num_groups(hidden, granularity)
    return -(-raw // _LANES) * _LANES


def wire_bytes_per_row(hidden: int, granularity: str) -> int:
    return hidden + scale_bytes(hidden, granularity)


def applies(rows: int, hidden: int, n: int, min_tokens: int, granularity: str) -> bool:
    """Whether a [rows, hidden] all-reduce over n ranks takes the fp8 path."""
    if min_tokens <= 0 or n <= 1 or rows < min_tokens or rows % n:
        return False
    return granularity == "row" or hidden % _GROUP == 0


def quantize(x: jax.Array, granularity: str) -> tuple[jax.Array, jax.Array]:
    """[T, H] -> (e4m3 [T, H], f32 scale [T, G]); scale = amax / 448."""
    t, h = x.shape
    g = num_groups(h, granularity)
    xf = x.astype(jnp.float32).reshape(t, g, h // g)
    amax = jnp.max(jnp.abs(jnp.where(jnp.isfinite(xf), xf, 0.0)), axis=-1)
    scale = jnp.where(amax > 0.0, amax / FP8_MAX, 1.0)
    q = jnp.clip(xf * (1.0 / scale)[..., None], -FP8_MAX, FP8_MAX).astype(FP8)
    return q.reshape(t, h), scale


def dequantize(q: jax.Array, scale: jax.Array) -> jax.Array:
    """(e4m3 [T, H], f32 [T, G]) -> f32 [T, H]."""
    t, h = q.shape
    g = scale.shape[-1]
    xf = q.astype(jnp.float32).reshape(t, g, h // g) * scale[..., None]
    return xf.reshape(t, h)


def pack(q: jax.Array, scale: jax.Array, granularity: str) -> jax.Array:
    """(e4m3 [T, H], f32 [T, G]) -> int8 [T, H + scale_bytes]."""
    t, h = q.shape
    sb = lax.bitcast_convert_type(scale, jnp.int8).reshape(t, -1)  # [T, 4G]
    pad = scale_bytes(h, granularity) - sb.shape[1]
    parts = [lax.bitcast_convert_type(q, jnp.int8), sb]
    if pad:
        parts.append(jnp.zeros((t, pad), jnp.int8))
    return jnp.concatenate(parts, axis=1)


def unpack(p: jax.Array, hidden: int, granularity: str) -> tuple[jax.Array, jax.Array]:
    """Inverse of `pack`."""
    t = p.shape[0]
    g = num_groups(hidden, granularity)
    q = lax.bitcast_convert_type(p[:, :hidden], FP8)
    sb = p[:, hidden : hidden + 4 * g].reshape(t, g, 4)
    return q, lax.bitcast_convert_type(sb, jnp.float32)


def _to_wire(x, granularity):
    q, s = quantize(x, granularity)
    return pack(q, s, granularity)


def _from_wire(p, hidden, granularity):
    return dequantize(*unpack(p, hidden, granularity))


def _a2a_sum(x, axis, n, granularity):
    """fp8 all-to-all of x's n row blocks, then the f32 sum of the n partials
    of this rank's block: f32 [T / n, H]."""
    t, h = x.shape
    recv = lax.all_to_all(
        _to_wire(x, granularity), axis, split_axis=0, concat_axis=0, tiled=True
    )
    return _from_wire(recv, h, granularity).reshape(n, t // n, h).sum(0)


def _ag8(block, axis, granularity, dtype):
    h = block.shape[1]
    got = lax.all_gather(_to_wire(block, granularity), axis, axis=0, tiled=True)
    return _from_wire(got, h, granularity).astype(dtype)


def all_reduce(
    x: jax.Array, axis: str, n: int, mode: str, granularity: str
) -> jax.Array:
    """Per-rank body: the (lossy) sum of x over the n ranks of `axis`."""
    if mode == "a2a":
        return _ag8(_a2a_sum(x, axis, n, granularity), axis, granularity, x.dtype)
    if mode == "a2a16":
        block = _a2a_sum(x, axis, n, granularity).astype(x.dtype)
        return lax.all_gather(block, axis, axis=0, tiled=True)
    if mode == "rs":
        block = lax.psum_scatter(x, axis, scatter_dimension=0, tiled=True)
        return _ag8(block, axis, granularity, x.dtype)
    raise ValueError(f"mode must be one of {MODES}: {mode!r}")
