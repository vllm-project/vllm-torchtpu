# SPDX-License-Identifier: Apache-2.0
"""Fused prefill PLE layer after the table lookup (P-DENSE D1).

The served layer (``models/vllm/qwen4_exp/ple.py`` gated_values, three
``GroupedGemmaRMSNorm`` and ``kernels/qwen4_exp/ple.py`` short_conv) runs as
many XLA passes over [T, hc * hidden] tensors, several of them in f32, and the
conv's history taps as four [T, C] gathers from a [R, state_len, C] table.
Here the work after the key/value projections is two token-tiled Pallas
passes plus a small fix-up of the rows that read conv history:

  gate pass  (h, key, value) -> gated [T, C], conv input [T, C]
             norm_query(h), norm_key(key), the per-stream gate
             sigmoid(signed_sqrt(<k, q> / sqrt(hidden))) * value, norm_conv.
  conv pass  (conv input, gated) -> out [T, C]
             dilated causal depthwise conv over the token stream (taps read
             earlier rows of the step, with a 16-row halo from the previous
             tile), silu, + gated, masks.
  fix-up     rows whose taps reach back past their request's chunk start
             (at most state_len rows per request) are recomputed in XLA with
             the request's history and scattered over the conv pass's rows.

Math follows the served torch/JAX path operation by operation, including its
bf16 rounding points (PyTorch opmath: bf16 elementwise ops compute in f32 and
round once). Reductions (the RMS variances and the gate's dot product) sum in
a different order, so the result is not bitwise; it is judged by the float64
rule and an accuracy gate.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HALO = 16  # rows of the previous tile kept for the conv taps (>= state_len)


def _rne(x):
    """Round finite f32 to the nearest bf16 (ties to even), staying in f32.

    The same value as x.astype(bf16).astype(f32) for finite x, without bf16
    vectors of one lane (per-row gate columns) or a pack/unpack round trip.
    """
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    bits = bits + jnp.uint32(0x7FFF) + ((bits >> 16) & jnp.uint32(1))
    return jax.lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)


NORM_MODES = ("sum", "mean", "recip", "newton")
# "torch": PyTorch's roundings (value projection to bf16, gated = bf16(sig * value) feeds the conv
# norm). "xla": the value projection stays f32 and the conv norm reads the f32 product sig * value
# (gated is still stored bf16 for the residual add). On TPU, XLA computes the served-math JAX port
# that way (excess precision; mb1 j-19f04582 base errors equal this mode's to 4 digits on CPU), so
# "xla" is the float64-rule-matched mode (P-DENSE mb6b).
PRECISIONS = ("torch", "xla")


def _proj_dims(weights_kn):
    """dot_general dims for embeddings [T, E] against a projection weight stored as torch Linear's [out, in]
    (``weights_kn`` False) or as the TPU linear method stores it after loading, [in, out]
    (``weights_kn`` True; layers/adapter/quantization/unquantized.py flips dense weights)."""
    return (((1,), (0,)), ((), ())) if weights_kn else (((1,), (1,)), ((), ()))


def _norm(x, w1p, *, eps, mode="sum"):
    """GroupedGemmaRMSNorm on one f32 group: (x * rsqrt(mean(x^2) + eps)) * (1 + w).

    ``mode`` (the variance and scale arithmetic; mb6 picks one): sum = sum / N
    (the measured default), mean = jnp.mean (as hc_norm / gdn_gated_norm),
    recip = sum * (1 / N), newton = sum / N with one Newton step on the rsqrt.
    """
    if mode == "mean":
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
    elif mode == "recip":
        variance = jnp.sum(x * x, axis=-1, keepdims=True) * jnp.float32(1.0 / x.shape[-1])
    else:
        variance = jnp.sum(x * x, axis=-1, keepdims=True) / jnp.float32(x.shape[-1])
    a = variance + eps
    r = jax.lax.rsqrt(a)
    if mode == "newton":
        r = r * (1.5 - 0.5 * a * r * r)
    return ((x * r) * w1p).astype(jnp.bfloat16)


def _gate_stream(h_ref, key_ref, value, wq_ref, wk_ref, wc_ref, lo, hi, *, hidden, eps, mutant,
                 norm_mode="sum", f32_gated=False):
    """One stream's (gated f32 holding bf16 values, conv input bf16) for a row tile."""
    root = jnp.float32(math.sqrt(hidden))
    floor = jnp.float32(1e-6)
    qn = _norm(h_ref[:, lo:hi].astype(jnp.float32), wq_ref[:, lo:hi], eps=eps, mode=norm_mode)
    kn = _norm(key_ref[:, lo:hi].astype(jnp.float32), wk_ref[:, lo:hi], eps=eps, mode=norm_mode)
    # torch: (key * query).float().sum(-1).to(bf16); bf16 products are exact in f32.
    prod = _rne(kn.astype(jnp.float32) * qn.astype(jnp.float32))
    dot = _rne(jnp.sum(prod, axis=-1, keepdims=True))
    gate = _rne(dot / root)
    mag = _rne(jnp.sqrt(_rne(jnp.maximum(jnp.abs(gate), floor))))
    sign = jnp.where(gate > 0, 1.0, jnp.where(gate < 0, -1.0, 0.0)).astype(jnp.float32)
    if mutant:
        sign = -sign
    sig = _rne(jax.nn.sigmoid(_rne(mag * sign)))
    if f32_gated:   # precision "xla": the conv norm reads the unrounded product
        product = sig * value
        return _rne(product), _norm(product, wc_ref[:, lo:hi], eps=eps, mode=norm_mode)
    gated = _rne(sig * value)
    return gated, _norm(gated, wc_ref[:, lo:hi], eps=eps, mode=norm_mode)


def _gate_kernel(
    h_ref, key_ref, value_ref, wq_ref, wk_ref, wc_ref, gated_ref, conv_ref,
    *, streams, hidden, eps, mutant, norm_mode="sum", f32_gated=False,
):
    value = value_ref[...].astype(jnp.float32)
    for s in range(streams):
        lo, hi = s * hidden, (s + 1) * hidden
        gated, conv_in = _gate_stream(
            h_ref, key_ref, value, wq_ref, wk_ref, wc_ref, lo, hi,
            hidden=hidden, eps=eps, mutant=mutant, norm_mode=norm_mode, f32_gated=f32_gated,
        )
        gated_ref[:, lo:hi] = gated.astype(jnp.bfloat16)
        conv_ref[:, lo:hi] = conv_in


@functools.cache
def ple_gate_fn(
    streams: int, hidden: int, token_tile: int, eps: float,
    mutant: bool = False, interpret: bool = False, norm_mode: str = "sum",
    precision: str = "torch",
):
    """(h [T,C] bf16, key [T,C] bf16, value [T,hidden] bf16 (f32 under precision
    "xla"), 1+w_q, 1+w_k, 1+w_conv [1,C] f32) -> (gated [T,C] bf16, conv input
    [T,C] bf16). T % token_tile == 0."""
    width = streams * hidden
    if norm_mode not in NORM_MODES:
        raise ValueError(f"unknown PLE norm mode {norm_mode!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown PLE precision {precision!r}")
    value_dtype = jnp.float32 if precision == "xla" else jnp.bfloat16
    if hidden % 128 or token_tile % 16:
        raise ValueError("PLE gate needs 128-aligned groups and 16-row tiles")

    def run(h, key, value, wq1, wk1, wc1):
        tokens = h.shape[0]
        if (
            tokens % token_tile
            or h.shape != (tokens, width)
            or key.shape != (tokens, width)
            or value.shape != (tokens, hidden)
            or h.dtype != jnp.bfloat16
            or key.dtype != jnp.bfloat16
            or value.dtype != value_dtype
        ):
            raise ValueError("PLE gate operands do not match its geometry")
        row = pl.BlockSpec((token_tile, width), lambda i: (i, 0))
        weight = pl.BlockSpec((1, width), lambda i: (0, 0))
        return pl.pallas_call(
            functools.partial(
                _gate_kernel, streams=streams, hidden=hidden, eps=eps, mutant=mutant,
                norm_mode=norm_mode, f32_gated=precision == "xla",
            ),
            grid=(tokens // token_tile,),
            in_specs=(row, row, pl.BlockSpec((token_tile, hidden), lambda i: (i, 0)),
                      weight, weight, weight),
            out_specs=(row, row),
            out_shape=(
                jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
                jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",), vmem_limit_bytes=60 << 20
            ),
            interpret=interpret,
            name=f"ple_gate_s{streams}_h{hidden}_m{token_tile}{'_mut' if mutant else ''}"
            f"{'' if norm_mode == 'sum' else '_n' + norm_mode}{'_xp' if precision == 'xla' else ''}",
        )(h, key, value, wq1, wk1, wc1)

    return run


def _conv_kernel(
    x_ref, halo_ref, gated_ref, w_ref, mask_ref, out_ref, ext_ref,
    *, taps, dilation, state_len, tap_mode, halo,
):
    tile = x_ref.shape[0]
    # ext rows [0, halo) = the previous rows of the step, [halo, halo + tile) = this tile.
    ext_ref[0:halo, :] = halo_ref[...].astype(ext_ref.dtype)
    ext_ref[halo:, :] = x_ref[...].astype(ext_ref.dtype)
    accum = jnp.zeros(x_ref.shape, jnp.float32)
    if tap_mode in ("roll", "mxu"):
        ext = ext_ref[...]
    if tap_mode == "mxu":
        out_row = jax.lax.broadcasted_iota(jnp.int32, (tile, tile + halo), 0)
        src_row = jax.lax.broadcasted_iota(jnp.int32, (tile, tile + halo), 1)
    for tap in range(taps):
        back = state_len - tap * dilation
        if tap_mode == "slice":
            rows = ext_ref[pl.ds(halo - back, tile), :]
        elif tap_mode == "roll":
            rows = (pltpu.roll(ext, back, 0) if back else ext)[halo:, :]
        else:
            # One-hot shift on the MXU: row i takes ext row i + halo - back exactly
            # (bf16 values times 1.0, f32 accumulation of zeros).
            shift = (src_row == out_row + (halo - back)).astype(jnp.bfloat16)
            rows = jnp.dot(shift, ext, preferred_element_type=jnp.float32)
        # The tap value is an exact f32 copy of a bf16 value, as in short_conv.
        accum = accum + rows.astype(jnp.float32) * w_ref[tap : tap + 1, :]
    conv = _rne(jax.nn.silu(_rne(accum)))
    mask = mask_ref[...]
    conv = jnp.where(mask[:, 1:2] != 0, conv, 0.0)
    total = _rne(gated_ref[...].astype(jnp.float32) + conv)
    out_ref[...] = jnp.where(mask[:, 0:1] != 0, total, 0.0).astype(out_ref.dtype)


@functools.cache
def ple_conv_fn(
    width: int, taps: int, dilation: int, token_tile: int,
    tap_mode: str = "slice", interpret: bool = False,
):
    """(conv input [T,C] bf16, gated [T,C] bf16, taps [8,C] f32 (row k = tap k),
    masks [T,2] int32 (col 0 output valid, col 1 conv valid)) -> out [T,C] bf16.

    Every tap reads the step's own earlier rows; rows whose taps cross their
    request's start are wrong here and must be overwritten by ``conv_fixup``.
    tap_mode: how a tap reads the rows ``back`` above: "slice" (offset loads
    from an f32 scratch), "roll" (sublane rotate of the f32 scratch) or "mxu"
    (one-hot shift matmul on a bf16 scratch, with a 128-row halo so that the
    contraction is 128-aligned). All three are exact copies.
    """
    state_len = (taps - 1) * dilation
    halo = 128 if tap_mode == "mxu" else _HALO
    if state_len > _HALO or token_tile % halo or width % 128 or taps > 8:
        raise ValueError("PLE conv geometry unsupported")
    if tap_mode not in ("slice", "roll", "mxu"):
        raise ValueError(f"unknown PLE conv tap mode {tap_mode!r}")
    per_tile = token_tile // halo
    scratch = jnp.bfloat16 if tap_mode == "mxu" else jnp.float32

    def run(x, gated, weights, masks):
        tokens = x.shape[0]
        if (
            tokens % token_tile
            or x.shape != (tokens, width)
            or gated.shape != (tokens, width)
            or weights.shape != (8, width)
            or masks.shape != (tokens, 2)
        ):
            raise ValueError("PLE conv operands do not match its geometry")
        row = pl.BlockSpec((token_tile, width), lambda i: (i, 0))
        return pl.pallas_call(
            functools.partial(
                _conv_kernel, taps=taps, dilation=dilation, state_len=state_len,
                tap_mode=tap_mode, halo=halo,
            ),
            grid=(tokens // token_tile,),
            in_specs=(
                row,
                pl.BlockSpec(
                    (halo, width), lambda i: (jnp.maximum(i * per_tile - 1, 0), 0)
                ),
                row,
                pl.BlockSpec((8, width), lambda i: (0, 0)),
                pl.BlockSpec((token_tile, 2), lambda i: (i, 0)),
            ),
            out_specs=row,
            out_shape=jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
            scratch_shapes=(pltpu.VMEM((token_tile + halo, width), scratch),),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",), vmem_limit_bytes=60 << 20
            ),
            interpret=interpret,
            name=f"ple_conv_w{width}_k{taps}_d{dilation}_m{token_tile}_{tap_mode}",
        )(x, x, gated, weights, masks)

    return run


def request_rows(starts, ids, valid_tokens, tokens):
    """Per-row masks as the served path computes them.

    Returns int32 [T, 2]: col 0 = forward_with_history's output mask
    (row < valid_tokens and its request's pool id >= 0, request from
    token_requests' cumsum form), col 1 = short_conv's ``valid`` (row <
    starts[-1] and the request is active: id > 0 and a non-empty chunk).
    """
    starts = starts.astype(jnp.int32)
    requests = ids.shape[0]
    t = jnp.arange(tokens, dtype=jnp.int32)
    # token_requests: marks at interior starts (clamped to T), cumsum, clamp to R - 1.
    inner = jnp.minimum(starts[1:requests], tokens)
    marks = jnp.zeros((tokens + 1,), jnp.int32).at[inner].add(1)
    owner = jnp.minimum(jnp.cumsum(marks[:tokens]), requests - 1)
    out_valid = (t < valid_tokens) & (ids[owner] >= 0)
    # short_conv's request map and active slots (pooled_short_conv).
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), requests - 1)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    conv_valid = (t < starts[-1]) & active[req]
    return jnp.stack([out_valid, conv_valid], axis=1).astype(jnp.int32)


def conv_fixup(out, conv_in, gated, history, weight, starts, ids, initial, masks,
               *, dilation):
    """Recompute the rows whose taps reach back past their chunk start.

    For request r, rows starts[r] + j, j < state_len, inside the chunk: the
    same tap order, f32 accumulation and rounding as short_conv, reading
    ``history[r]`` (zero without initial state) where the tap precedes the
    chunk. history: [R, state_len, C] bf16 (pooled_short_conv's read_region).
    """
    tokens, channels = out.shape
    requests, state_len, _ = history.shape
    taps = weight.shape[-1]
    starts = starts.astype(jnp.int32)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    j = jnp.arange(state_len, dtype=jnp.int32)
    rows = starts[:-1, None] + j[None, :]
    inside = (rows < starts[1:, None]) & active[:, None] & (rows < tokens)
    safe = jnp.clip(rows, 0, tokens - 1)
    accum = jnp.zeros((requests, state_len, channels), jnp.float32)
    for tap in range(taps):
        back = state_len - tap * dilation
        position = j - back
        past = history[:, jnp.clip(position + state_len, 0, state_len - 1)]
        past = jnp.where(initial[:, None, None], past, jnp.zeros((), past.dtype))
        current = conv_in[jnp.clip(rows - back, 0, tokens - 1)]
        value = jnp.where((position >= 0)[None, :, None], current, past)
        accum = accum + value.astype(jnp.float32) * weight[:, tap].astype(jnp.float32)
    pre = accum.astype(jnp.bfloat16)
    conv = jax.nn.silu(pre.astype(jnp.float32)).astype(jnp.bfloat16)
    row_masks = masks[safe]
    conv = jnp.where(row_masks[..., 1:2] != 0, conv, jnp.zeros((), conv.dtype))
    total = (gated[safe].astype(jnp.float32) + conv.astype(jnp.float32)).astype(jnp.bfloat16)
    total = jnp.where(row_masks[..., 0:1] != 0, total, jnp.zeros((), total.dtype))
    dest = jnp.where(inside, rows, tokens).reshape(-1)
    return out.at[dest].set(total.reshape(-1, channels), mode="drop")


def new_windows(conv_in, history, starts, initial):
    """Each request's last state_len conv inputs (short_conv's ``updated``)."""
    tokens = conv_in.shape[0]
    state_len = history.shape[1]
    starts = starts.astype(jnp.int32)
    lengths = starts[1:] - starts[:-1]
    positions = lengths[:, None] - state_len + jnp.arange(state_len)[None, :]
    past = history[
        jnp.arange(history.shape[0])[:, None],
        jnp.clip(positions + state_len, 0, state_len - 1),
    ]
    past = jnp.where(initial[:, None, None], past, jnp.zeros((), past.dtype))
    current = conv_in[jnp.clip(starts[:-1, None] + positions, 0, tokens - 1)]
    return jnp.where((positions >= 0)[..., None], current, past)


def ple_fused_after_lookup(
    h, embeddings, key_weight, value_weight, wq1, wk1, wc1, conv_weight,
    history, starts, ids, initial, valid_tokens,
    *, streams, hidden, eps, dilation, token_tile=128, tap_mode="slice",
    mutant=False, interpret=False, norm_mode="sum", precision="torch", weights_kn=False,
):
    """The PLE layer from decoded embeddings to the residual addend.

    key_weight [C, E] and value_weight [hidden, E] as torch Linear stores
    them; wq1/wk1/wc1 = 1 + norm weight in f32, [1, C]; conv_weight [C, taps].
    Returns (addend [T, C] bf16, new windows [R, state_len, C] bf16).
    """
    tokens, width = h.shape
    taps = conv_weight.shape[-1]
    key = jax.lax.dot_general(
        embeddings, key_weight, _proj_dims(weights_kn),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    value = jax.lax.dot_general(
        embeddings, value_weight, _proj_dims(weights_kn),
        preferred_element_type=jnp.float32,
    )
    if precision != "xla":
        value = value.astype(jnp.bfloat16)
    gated, conv_in = ple_gate_fn(
        streams, hidden, token_tile, eps, mutant, interpret, norm_mode, precision
    )(h, key, value, wq1, wk1, wc1)
    masks = request_rows(starts, ids, valid_tokens, tokens)
    weights = jnp.zeros((8, width), jnp.float32).at[:taps].set(
        conv_weight.astype(jnp.float32).T
    )
    out = ple_conv_fn(width, taps, dilation, token_tile, tap_mode, interpret)(
        conv_in, gated, weights, masks
    )
    out = conv_fixup(out, conv_in, gated, history, conv_weight, starts, ids, initial,
                     masks, dilation=dilation)
    return out, new_windows(conv_in, history, starts, initial)


def pooled_fused_ple(
    h: jax.Array,
    embeddings: jax.Array,
    key_weight: jax.Array,
    value_weight: jax.Array,
    wq: jax.Array,
    wk: jax.Array,
    wc: jax.Array,
    conv_weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    valid: jax.Array,
    *,
    streams: int,
    hidden: int,
    eps: float,
    dilation: int,
    state_len: int,
    token_tile: int = 128,
    tap_mode: str = "slice",
    interpret: bool = False,
    norm_mode: str = "sum",
    precision: str = "torch",
    weights_kn: bool = False,
) -> tuple[jax.Array, jax.Array]:
    # Every parameter is annotated: torch_tpu's pallas.jax_op (build_pooled_fused_ple_op)
    # refuses an unannotated JAX function (srv1-F j-4dfa116b failed engine init on it).
    """pooled_ple's prefill layer after the lookup on the unified pool.

    Norm weights are the checkpoint's bf16 [C] (the op adds 1 in f32);
    conv_weight is [C, taps] or the layer's [C, 1, taps] (sliced here); valid has one element
    (query_start_loc[-1:]).
    History is read and written like pooled_short_conv: idle or empty
    requests leave their pages untouched. Returns (pool, addend [T, C]).
    """
    if conv_weight.ndim == 3:   # the conv1d layer weight [C, 1, taps]
        conv_weight = conv_weight[:, 0]
    from vllm_torchtpu.kernels.qwen4_exp.pool import read_region, write_region

    channels = h.shape[-1]
    history = read_region(
        pool, ids, elements=state_len * channels, dtype=jnp.bfloat16
    ).reshape(len(ids), state_len, channels)

    def one(w):
        return (1.0 + w.astype(jnp.float32)).reshape(1, channels)

    args = (h, embeddings, key_weight, value_weight, one(wq), one(wk), one(wc), conv_weight,
            history, starts, ids, initial, valid.reshape(()))
    kw = dict(streams=streams, hidden=hidden, eps=eps, dilation=dilation,
              token_tile=token_tile, interpret=interpret, norm_mode=norm_mode, precision=precision,
              weights_kn=weights_kn)
    if tap_mode in ("single", "single_mxu"):
        out, windows = ple_single_after_lookup(
            *args, **kw, tap_mode="mxu" if tap_mode == "single_mxu" else "slice"
        )
    else:
        out, windows = ple_fused_after_lookup(*args, **kw, tap_mode=tap_mode)
    active = (ids > 0) & (starts[1:] > starts[:-1])
    pool = write_region(pool, windows.reshape(len(ids), -1), jnp.where(active, ids, 0))
    return pool, out


_HIST_ROWS = 128  # history one-hot contraction, 128-aligned (requests * state_len <= 128)


def pooled_fused_ple_kq(
    h: jax.Array,
    embeddings: jax.Array,
    key_weight: jax.Array,
    value_weight: jax.Array,
    wk: jax.Array,
    wq: jax.Array,
    wc: jax.Array,
    conv_weight: jax.Array,
    pool: jax.Array,
    starts: jax.Array,
    ids: jax.Array,
    initial: jax.Array,
    valid: jax.Array,
    *,
    streams: int,
    hidden: int,
    eps: float,
    dilation: int,
    state_len: int,
    token_tile: int = 128,
    tap_mode: str = "slice",
    interpret: bool = False,
    norm_mode: str = "sum",
    precision: str = "torch",
    weights_kn: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """pooled_fused_ple with the norm weights taken key-first (norm_key, then norm_query): the order in which
    the served PLE step first uses them. Every parameter is annotated (pallas.jax_op). Same values: wq and wk
    keep their roles in pooled_fused_ple.

    Why: shape_variants hands a compiled token bucket between the fused and the served trace whenever their
    subgraph inputs match by (dtype, shape) in order, and the handed runnable is called positionally. The two
    [C] norm weights have one shape, so if the fused trace lifted norm_query first while the served trace lifts
    norm_key first, a handed bucket would silently receive them swapped (P-MOE-6 fix 4)."""
    return pooled_fused_ple(
        h, embeddings, key_weight, value_weight, wq, wk, wc, conv_weight, pool, starts, ids, initial, valid,
        streams=streams, hidden=hidden, eps=eps, dilation=dilation, state_len=state_len, token_tile=token_tile,
        tap_mode=tap_mode, interpret=interpret, norm_mode=norm_mode, precision=precision, weights_kn=weights_kn,
    )


def _single_kernel(
    h_ref, key_ref, value_ref, wq_ref, wk_ref, wc_ref, w_ref, hist_ref, rows_ref,
    slots_ref, out_ref, win_ref, halo_ref, ext_ref, gated_ref,
    *, streams, hidden, eps, taps, dilation, state_len, mutant, tap_mode, halo, norm_mode="sum",
    f32_gated=False,
):
    """One pass per row tile: gate, norm_conv, conv (history taps by an exact
    one-hot matmul against the resident history rows), silu, + gated, masks;
    the new windows gathered by a one-hot matmul. Tiles run in order: the
    previous tile's last 16 conv inputs are carried in ``halo_ref``."""
    i = pl.program_id(0)
    tile = h_ref.shape[0]

    @pl.when(i == 0)
    def _init():
        halo_ref[...] = jnp.zeros(halo_ref.shape, halo_ref.dtype)
        # The window block stays resident across the (ordered) grid: it is the
        # accumulator. Each slot receives at most one exact value, so bf16 is exact.
        win_ref[...] = jnp.zeros(win_ref.shape, win_ref.dtype)

    value = value_ref[...].astype(jnp.float32)
    ext_ref[0:halo, :] = halo_ref[...]
    for s in range(streams):
        lo, hi = s * hidden, (s + 1) * hidden
        gated, conv_in = _gate_stream(
            h_ref, key_ref, value, wq_ref, wk_ref, wc_ref, lo, hi,
            hidden=hidden, eps=eps, mutant=mutant, norm_mode=norm_mode, f32_gated=f32_gated,
        )
        gated_ref[:, lo:hi] = gated.astype(jnp.bfloat16)   # bf16-exact values
        ext_ref[halo:, lo:hi] = conv_in.astype(ext_ref.dtype)

    rows = rows_ref[...]
    position, base = rows[:, 2:3], rows[:, 3:4]
    col = jax.lax.broadcasted_iota(jnp.int32, (tile, _HIST_ROWS), 1)
    hist = hist_ref[...]
    accum = jnp.zeros((tile, ext_ref.shape[1]), jnp.float32)
    if tap_mode == "mxu":
        ext = ext_ref[...]
        out_row = jax.lax.broadcasted_iota(jnp.int32, (tile, tile + halo), 0)
        src_row = jax.lax.broadcasted_iota(jnp.int32, (tile, tile + halo), 1)
    for tap in range(taps):
        back = state_len - tap * dilation
        if tap_mode == "mxu":
            shift = (src_row == out_row + (halo - back)).astype(jnp.bfloat16)
            current = jnp.dot(shift, ext, preferred_element_type=jnp.float32)
        else:
            current = ext_ref[pl.ds(halo - back, tile), :]
        before = position < back
        pick = ((col == base - back) & before).astype(jnp.bfloat16)
        past = jnp.dot(pick, hist, preferred_element_type=jnp.float32)
        value_tap = jnp.where(before, past, current)
        accum = accum + value_tap * w_ref[tap : tap + 1, :]
    conv = _rne(jax.nn.silu(_rne(accum)))
    conv = jnp.where(rows[:, 1:2] != 0, conv, 0.0)
    total = _rne(gated_ref[...].astype(jnp.float32) + conv)
    out_ref[...] = jnp.where(rows[:, 0:1] != 0, total, 0.0).astype(out_ref.dtype)

    # New windows: slot j takes global row slots_ref[j] (-1: none) of the conv input.
    tile_row = i * tile + jax.lax.broadcasted_iota(jnp.int32, (_HIST_ROWS, tile), 1)
    take = (tile_row == slots_ref[...]).astype(jnp.bfloat16)
    conv_tile = ext_ref[halo:, :].astype(jnp.bfloat16)
    win_ref[...] += jnp.dot(take, conv_tile, preferred_element_type=jnp.float32).astype(win_ref.dtype)
    halo_ref[...] = ext_ref[tile : tile + halo, :]


@functools.cache
def ple_single_fn(
    streams: int, hidden: int, token_tile: int, eps: float, taps: int, dilation: int,
    mutant: bool = False, interpret: bool = False, tap_mode: str = "slice", norm_mode: str = "sum",
    precision: str = "torch",
):
    """The whole PLE layer after the projections in one token-tiled pass.
    tap_mode "slice" (offset loads from an f32 tile buffer, 16-row halo) or
    "mxu" (one-hot shift matmuls on a bf16 buffer, 128-row halo).

    (h, key, value, 1+w_q, 1+w_k, 1+w_conv, taps [8, C] f32, history rows
    [128, C] bf16 (request r's window at rows r*state_len.., zero without
    initial state), rows [T, 4] int32 (out mask, conv mask, chunk position,
    r*state_len + position + state_len), window sources [128, 1] int32)
    -> (out [T, C] bf16, window rows [128, C] bf16 (in-step sources only)).
    """
    width = streams * hidden
    state_len = (taps - 1) * dilation
    halo = 128 if tap_mode == "mxu" else _HALO
    ext_dtype = jnp.bfloat16 if tap_mode == "mxu" else jnp.float32
    if tap_mode not in ("slice", "mxu"):
        raise ValueError(f"unknown PLE single-pass tap mode {tap_mode!r}")
    if state_len > _HALO or token_tile % halo or hidden % 128 or taps > 8:
        raise ValueError("PLE single-pass geometry unsupported")
    if precision not in PRECISIONS:
        raise ValueError(f"unknown PLE precision {precision!r}")
    value_dtype = jnp.float32 if precision == "xla" else jnp.bfloat16

    def run(h, key, value, wq1, wk1, wc1, weights, hist, rows, slots):
        tokens = h.shape[0]
        if (tokens % token_tile or hist.shape != (_HIST_ROWS, width) or rows.shape != (tokens, 4)
                or value.dtype != value_dtype):
            raise ValueError("PLE single-pass operands do not match its geometry")
        row = pl.BlockSpec((token_tile, width), lambda i: (i, 0))
        whole = lambda shape: pl.BlockSpec(shape, lambda i: (0,) * len(shape))  # noqa: E731
        return pl.pallas_call(
            functools.partial(
                _single_kernel, streams=streams, hidden=hidden, eps=eps, taps=taps,
                dilation=dilation, state_len=state_len, mutant=mutant, tap_mode=tap_mode,
                halo=halo, norm_mode=norm_mode, f32_gated=precision == "xla",
            ),
            grid=(tokens // token_tile,),
            in_specs=(
                row, row, pl.BlockSpec((token_tile, hidden), lambda i: (i, 0)),
                whole((1, width)), whole((1, width)), whole((1, width)), whole((8, width)),
                whole((_HIST_ROWS, width)), pl.BlockSpec((token_tile, 4), lambda i: (i, 0)),
                whole((_HIST_ROWS, 1)),
            ),
            out_specs=(row, whole((_HIST_ROWS, width))),
            out_shape=(
                jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16),
                jax.ShapeDtypeStruct((_HIST_ROWS, width), jnp.bfloat16),
            ),
            scratch_shapes=(
                pltpu.VMEM((halo, width), ext_dtype),
                pltpu.VMEM((token_tile + halo, width), ext_dtype),
                pltpu.VMEM((token_tile, width), jnp.bfloat16),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=60 << 20
            ),
            interpret=interpret,
            name=f"ple_single_s{streams}_h{hidden}_m{token_tile}{'_mxu' if tap_mode == 'mxu' else ''}"
            f"{'_mut' if mutant else ''}{'' if norm_mode == 'sum' else '_n' + norm_mode}"
            f"{'_xp' if precision == 'xla' else ''}",
        )(h, key, value, wq1, wk1, wc1, weights, hist, rows, slots)

    return run


def ple_single_after_lookup(
    h, embeddings, key_weight, value_weight, wq1, wk1, wc1, conv_weight,
    history, starts, ids, initial, valid_tokens,
    *, streams, hidden, eps, dilation, token_tile=128, mutant=False, interpret=False,
    tap_mode="slice", norm_mode="sum", precision="torch", weights_kn=False,
):
    """ple_fused_after_lookup's contract in one Pallas pass (no fix-up, no
    gated / conv-input round trip through HBM)."""
    tokens, width = h.shape
    taps = conv_weight.shape[-1]
    requests, state_len, _ = history.shape
    if requests * state_len > _HIST_ROWS:
        raise ValueError("PLE single pass holds at most 128 history rows")
    key = jax.lax.dot_general(
        embeddings, key_weight, _proj_dims(weights_kn), preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    value = jax.lax.dot_general(
        embeddings, value_weight, _proj_dims(weights_kn), preferred_element_type=jnp.float32,
    )
    if precision != "xla":
        value = value.astype(jnp.bfloat16)
    starts = starts.astype(jnp.int32)
    masks = request_rows(starts, ids, valid_tokens, tokens)
    # short_conv's request map: the chunk position and its history row base.
    t = jnp.arange(tokens, dtype=jnp.int32)
    req = jnp.minimum(jnp.sum(t[:, None] >= starts[None, 1:], axis=1), requests - 1)
    position = t - starts[req]
    rows = jnp.concatenate(
        [masks, position[:, None], (req * state_len + position + state_len)[:, None]], axis=1
    )
    owned = jnp.where(initial[:, None, None], history, jnp.zeros((), history.dtype))
    hist = jnp.zeros((_HIST_ROWS, width), jnp.bfloat16).at[: requests * state_len].set(
        owned.reshape(-1, width)
    )
    lengths = starts[1:] - starts[:-1]
    offsets = lengths[:, None] - state_len + jnp.arange(state_len)[None, :]
    sources = jnp.where(offsets >= 0, starts[1:, None] - state_len + jnp.arange(state_len)[None, :], -1)
    slots = jnp.full((_HIST_ROWS, 1), -1, jnp.int32).at[: requests * state_len, 0].set(sources.reshape(-1))
    weights = jnp.zeros((8, width), jnp.float32).at[:taps].set(conv_weight.astype(jnp.float32).T)
    out, win = ple_single_fn(
        streams, hidden, token_tile, eps, taps, dilation, mutant, interpret, tap_mode, norm_mode,
        precision,
    )(
        h, key, value, wq1, wk1, wc1, weights, hist, rows, slots
    )
    past = history[
        jnp.arange(requests)[:, None], jnp.clip(offsets + state_len, 0, state_len - 1)
    ]
    past = jnp.where(initial[:, None, None], past, jnp.zeros((), past.dtype))
    win = win[: requests * state_len].reshape(requests, state_len, width)
    return out, jnp.where((offsets >= 0)[..., None], win, past)
