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
"""In-kernel post/comb gates (sigmoid, softmax, Sinkhorn) of the mHC pre.

A mix kernel stages each grid step's gate logits (``stage_logits``) and, on
its last step, gates all its tokens at once (``gate_staged``) into the
row-major (T, hc_mult) / (T, hc_mult**2) post and comb outputs the next seam
reads.

The gates run with every logit entry k in vregs of its own, tokens dense on
sublanes and lanes (token 128 * t + l at row t, lane l): the softmax and
Sinkhorn sums are then adds across vregs, and the 20 serial Sinkhorn steps
have a whole call's tokens of independent work each, where one token block
per step left the chain's latency exposed.

The arithmetic follows what XLA compiles ``utils.mhc_pre_gates`` to: each
Sinkhorn step divides the softmax output by the running product of all
normalizers so far (XLA rewrites ``(a / b) / c`` as ``a / (b * c)``), the
row sums add (x0 + x2) + (x1 + x3), and the column sums add in order.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128


class GateConsts(NamedTuple):
    """The static constants of the post/comb gates (``utils.mhc_pre_gates``)."""

    hc_sinkhorn_eps: float
    hc_post_mult_value: float
    sinkhorn_repeat: int


class MixOutputs(NamedTuple):
    """How a mix kernel lays out its two f32 outputs (``mix_outputs``)."""

    specs: list[pl.BlockSpec]
    shapes: list[jax.ShapeDtypeStruct]
    scratch: list[pltpu.VMEM]
    semantics: str


def mix_outputs(
    tb: int, padded_tokens: int, hc_mult: int, gates: GateConsts | None
) -> MixOutputs:
    """A mix kernel's two f32 outputs: per token block, the raw mixes and
    squared sums for the XLA gates; or, with ``gates``, the post and comb
    gates, whole-array blocks written on the last step from every step's
    staged logits, so the grid runs in order."""
    hc_mult3 = hc_mult * (hc_mult + 2)
    if gates is None:
        widths = (hc_mult3, 1)
        return MixOutputs(
            specs=[pl.BlockSpec((tb, w), lambda i: (i, 0)) for w in widths],
            shapes=[
                jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32) for w in widths
            ],
            scratch=[],
            semantics="parallel",
        )
    widths = (hc_mult, hc_mult * hc_mult)
    tiles = (padded_tokens + LANE - 1) // LANE
    return MixOutputs(
        specs=[pl.BlockSpec((padded_tokens, w), lambda i: (0, 0)) for w in widths],
        shapes=[jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32) for w in widths],
        # The logits staging buffer (token-major, whole lane tiles) and the
        # entry-major buffer the gates run from (``gate_staged``).
        scratch=[
            pltpu.VMEM((tiles * LANE, LANE), jnp.float32),
            pltpu.VMEM((hc_mult3 * tiles, LANE), jnp.float32),
        ],
        semantics="arbitrary",
    )


def gate_logits(scaled, sc_ref, hb_ref, hc_mult):
    """(tb, hc_mult3) scaled mixes -> gate logits: each pre / post / comb
    entry times its ``hc_scale`` plus its ``hc_base``."""
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, scaled.shape[1]), 1)
    scale = jnp.where(
        lane < hc_mult,
        sc_ref[0, 0],
        jnp.where(lane < 2 * hc_mult, sc_ref[0, 1], sc_ref[0, 2]),
    )
    return scaled * scale + hb_ref[...]


def stage_logits(
    logits_scr, acc, sqr, sc_ref, hb_ref, *, hc_mult, hidden_size, rms_eps
):
    """Stage this grid step's gate logits, from its raw mixes ``acc``
    (tb, hc_mult3) and squared sums ``sqr`` (tb, 1)."""
    m, h = hc_mult, hidden_size
    logits = gate_logits(
        acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps), sc_ref, hb_ref, m
    )
    tb, width = logits.shape
    row0 = pl.multiple_of(pl.program_id(0) * tb, tb)
    logits_scr[pl.ds(row0, tb), 0:width] = logits


def _sum_j(x):
    """A Sinkhorn row's sum, in XLA's order for a 4-entry reduce."""
    return (x[0] + x[2]) + (x[1] + x[3])


def _sum_i(x):
    """A Sinkhorn column's sum, in order."""
    total = x[0]
    for v in x[1:]:
        total = total + v
    return total


def _gates_dense(logits, consts: GateConsts, hc_mult: int):
    """Gates on entry-major logits: ``logits[k]`` holds entry k (pre 0:m,
    post m:2m, comb 2m: row-major (i, j)) for every token.

    Returns post (m arrays) and comb (m * m arrays, row-major (i, j)).
    """
    m = hc_mult
    eps = consts.hc_sinkhorn_eps
    post = [
        (1.0 / (jnp.exp(-logits[m + j]) + 1.0)) * consts.hc_post_mult_value
        for j in range(m)
    ]
    comb = []
    for i in range(m):
        row = logits[2 * m + i * m : 2 * m + (i + 1) * m]
        top = jnp.maximum(jnp.maximum(row[0], row[2]), jnp.maximum(row[1], row[3]))
        e = [jnp.exp(x - top) for x in row]
        total = _sum_j(e)
        comb.append([x / total + eps for x in e])

    def col_norms(cur):
        return [_sum_i([cur[i][j] for i in range(m)]) + eps for j in range(m)]

    n = col_norms(comb)
    d = [[n[j] for j in range(m)] for _ in range(m)]
    for _ in range(consts.sinkhorn_repeat - 1):
        for i in range(m):
            r = _sum_j([comb[i][j] / d[i][j] for j in range(m)]) + eps
            d[i] = [d[i][j] * r for j in range(m)]
        n = col_norms([[comb[i][j] / d[i][j] for j in range(m)] for i in range(m)])
        d = [[d[i][j] * n[j] for j in range(m)] for i in range(m)]
    return post, [comb[i][j] / d[i][j] for i in range(m) for j in range(m)]


def gate_staged(logits_scr, dense_scr, post_ref, comb_ref, *, hc_mult, consts):
    """Gate every staged token into ``post_ref`` (T, hc_mult) and
    ``comb_ref`` (T, hc_mult**2), row-major (i, j)."""
    assert hc_mult == 4, hc_mult  # _sum_j's order is for 4 entries
    m = hc_mult
    width = m * (m + 2)
    tiles = logits_scr.shape[0] // LANE
    num_tokens = post_ref.shape[0]

    # Token-major tiles -> entry-major rows: tile t's entry k lands in row
    # k * tiles + t.
    for t in range(tiles):
        tile_t = logits_scr[t * LANE : (t + 1) * LANE, :].T
        dense_scr[pl.ds(t, width, stride=tiles), :] = tile_t[0:width]
    logits = [dense_scr[k * tiles : (k + 1) * tiles, :] for k in range(width)]

    post, comb = _gates_dense(logits, consts, m)
    # Back to token-major, through the same rows: comb entries first, then
    # post, so each tile's (m * m + m) outputs come out in its output order.
    for k, v in enumerate(comb + post):
        dense_scr[k * tiles : (k + 1) * tiles, :] = v
    outs = m * m + m
    for t in range(tiles):
        rows = dense_scr[pl.ds(t, outs, stride=tiles), :]
        tile = jnp.concatenate([rows, jnp.zeros((LANE - outs, LANE), jnp.float32)]).T
        n = min(LANE, num_tokens - t * LANE)
        comb_ref[t * LANE : t * LANE + n, :] = tile[:n, 0 : m * m]
        post_ref[t * LANE : t * LANE + n, :] = tile[:n, m * m : outs]
