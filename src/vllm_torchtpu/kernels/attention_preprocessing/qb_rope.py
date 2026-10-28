# SPDX-License-Identifier: Apache-2.0
"""Q-B projection and interleaved RoPE for integration into the model.

Same pipeline and numerics as ``qb_overlap3``. RoPE input is the per-step
``[tokens, rotary_dim]`` row block from ``rope_gather.gather_rows`` (packed
``[cos | sin]``), shared by every layer of the forward step.

Each token tile's rows are read and lane-expanded once, in the first head
window, alongside the qr quantization; later windows reuse the expanded
coefficients from VMEM.

Every block size comes from a ``QBRopeConfig`` (tuning table or
``default_config``). The FP8 qr encoding and RoPE coefficients of a whole
token segment stay resident in VMEM, so long inputs run as several segments,
each writing its rows of one aliased output.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.attention_preprocessing import rope_gather
from vllm_torchtpu.kernels.attention_preprocessing.serial import (
    _LANE,
    _lane_gather,
    _quantize_rows,
)
from vllm_torchtpu.kernels.attention_preprocessing.tuned_block_sizes import (
    QB_ROPE_TUNING,
    QBRopeConfig,
)

# The resident FP8 qr encoding has (32, 128) tiles: every dynamic row offset
# stored into it (token tiles, quantization fragments) must be a multiple of 32.
_ROW_ALIGN = 32


def _qb_rope_kernel(
    qr_ref,
    weight_hbm,
    scale_ref,
    rope_rows_ref,
    *refs,
    head_dim,
    tb,
    pairs,
    quant_fragment,
    post_fragment,
    token_offset,
    aliased,
):
    if aliased:
        refs = refs[1:]  # the aliased input is read and written as the output
    (
        out_hbm,
        encoded_ref,
        row_scale_ref,
        coeff_ref,
        even_ref,
        odd_ref,
        obuf_ref,
        sems,
        weight_ref,
        weight_sems,
    ) = refs
    window, tile = pl.program_id(0), pl.program_id(1)
    windows, tiles = pl.num_programs(0), pl.num_programs(1)
    slots = obuf_ref.shape[0]
    step = window * tiles + tile
    last = windows * tiles * pairs - 1
    tail_start = head_dim - _LANE
    out_words = out_hbm.bitcast(jnp.uint32)  # [tokens, heads/2, head_dim]
    token_start = pl.multiple_of(tile * tb, tb)  # within this segment
    products = (even_ref, odd_ref)

    def weight_copy(w, pair):
        """Head pair ``pair`` of window ``w`` into half ``w % 2`` of the buffer."""
        columns = pl.ds(
            pl.multiple_of((w * pairs + pair) * 2 * head_dim, 2 * head_dim),
            2 * head_dim,
        )
        return pltpu.make_async_copy(
            weight_hbm.at[:, columns],
            weight_ref.at[w % 2, pair],
            weight_sems.at[w % 2, pair],
        )

    # Weights stream per head pair on their own DMA priority, so row loads and
    # output copies (priority 0) never queue behind them. Window 0 is issued
    # up front; window w+1 is issued pair by pair during window w's first tile.
    @pl.when(step == 0)
    def prologue():
        def issue(pair, _):
            weight_copy(0, pair).start(priority=1)

        jax.lax.fori_loop(0, pairs, issue, None)

    @pl.when(window == 0)
    def prepare_rows():
        def quantize_fragment(fragment, _):
            start = pl.multiple_of(fragment * quant_fragment, quant_fragment)
            rows = pl.ds(start, quant_fragment)
            cached = pl.ds(token_start + start, quant_fragment)
            encoded, scale = _quantize_rows(qr_ref[rows, :])
            encoded_ref[cached, :] = encoded
            row_scale_ref[cached, :] = scale

        jax.lax.fori_loop(0, tb // quant_fragment, quantize_fragment, None)
        cos_lanes, sin_lanes = rope_gather.coefficients(rope_rows_ref[...])
        coeff_ref[0, pl.ds(token_start, tb), :] = cos_lanes
        coeff_ref[1, pl.ds(token_start, tb), :] = sin_lanes

    def copy(g):
        """Output DMA of global pair g (dst address only matters for start)."""
        t, q = g // pairs, g % pairs
        rows = pl.ds(pl.multiple_of(token_offset + (t % tiles) * tb, _ROW_ALIGN), tb)
        return pltpu.make_async_copy(
            obuf_ref.at[g % slots],
            out_words.at[rows, (t // tiles) * pairs + q, :],
            sems.at[g % slots],
        )

    def reclaim(*targets):
        """Wait for the DMAs that last used the slots of pairs ``targets``."""
        for g in targets:

            @pl.when(g - slots >= 0)
            def _(g=g):
                copy(g - slots).wait()

    def matmul(pair, parity):
        """FP32 products of this step's head pair into its parity buffer."""

        # Later tiles of a window find its weights already resident.
        @pl.when(tile == 0)
        def _():
            @pl.when(window + 1 < windows)
            def _():
                weight_copy(window + 1, pair).start(priority=1)

            weight_copy(window, pair).wait()

        tokens = pl.ds(token_start, tb)
        weights = weight_ref.at[window % 2, pair]
        for half in range(2):
            products[parity][half] = jnp.dot(
                encoded_ref[tokens, :],
                weights[:, pl.ds(half * head_dim, head_dim)],
                preferred_element_type=jnp.float32,
            )

    def post(g, parity):
        """Scale, round, rotate and pack global pair g, then start its DMA."""
        valid = g >= 0
        g = jnp.maximum(g, 0)  # step 0's "previous pair": compute on safe rows
        t, q = g // pairs, g % pairs
        rows0 = pl.multiple_of((t % tiles) * tb, _ROW_ALIGN)
        head0 = ((t // tiles) * pairs + q) * 2
        slot = g % slots
        source = products[parity]
        for f in range(tb // post_fragment):
            local = pl.ds(f * post_fragment, post_fragment)
            tokens = pl.ds(rows0 + f * post_fragment, post_fragment)
            halves = []
            for half in range(2):
                columns = pl.ds(
                    pl.multiple_of((head0 + half) * head_dim, head_dim), head_dim
                )
                projected = source[half, local, :] * row_scale_ref[tokens, :]
                projected = (projected * scale_ref[columns][None, :]).astype(
                    jnp.bfloat16
                )
                tail = projected[:, -_LANE:].astype(jnp.float32)
                partner = _lane_gather(tail, jnp.arange(_LANE) ^ 1)
                rotated = (
                    tail * coeff_ref[0, tokens, :] + partner * coeff_ref[1, tokens, :]
                ).astype(jnp.bfloat16)
                halves.append(
                    (
                        projected[:, :-_LANE].astype(jnp.float32),
                        rotated.astype(jnp.float32),
                    )
                )
            (low_plain, low_tail), (high_plain, high_tail) = halves
            # Values are already BF16-rounded, so the FP32->BF16 pack is exact.
            obuf_ref[slot, local, pl.ds(0, tail_start)] = pltpu.pack_elementwise(
                [low_plain, high_plain], packed_dtype=jnp.bfloat16
            )
            obuf_ref[slot, local, pl.ds(tail_start, _LANE)] = pltpu.pack_elementwise(
                [low_tail, high_tail], packed_dtype=jnp.bfloat16
            )

        @pl.when(valid)
        def _():
            copy(g).start()

    first = step * pairs  # global index of this step's pair 0

    # Prologue: pair 0's matmul beside the previous step's last (odd) pair.
    reclaim(first - 1)
    matmul(0, 0)
    post(first - 1, 1)

    def body(k, _):
        odd = 2 * k + 1
        reclaim(first + odd - 1, first + odd)
        matmul(odd, 1)  # beside the even pair before it
        post(first + odd - 1, 0)
        matmul(odd + 1, 0)  # beside the odd pair
        post(first + odd, 1)

    jax.lax.fori_loop(0, (pairs - 2) // 2, body, None)
    reclaim(first + pairs - 2)
    matmul(pairs - 1, 1)
    post(first + pairs - 2, 0)

    @pl.when(step == windows * tiles - 1)
    def finish():
        reclaim(last)
        post(last, 1)
        for back in range(slots):
            copy(last - back).wait()


def vmem_bytes(config: QBRopeConfig, segment, rank, heads, head_dim=512):
    """Upper estimate of one segment call's VMEM (lane/sublane padding included)."""
    tb, hb = config.token_block, config.head_block
    resident = segment * (rank + 4 * _LANE + 2 * 4 * _LANE)  # FP8 qr, row scale, coeff
    weights = 2 * rank * hb * head_dim  # FP8 window, double-buffered
    qr_tiles = 2 * tb * rank * 2
    products = 2 * 2 * tb * head_dim * 4
    out_slots = config.output_slots * tb * head_dim * 4
    rope_rows = 2 * tb * _LANE * 4  # double-buffered input tile
    scales = heads * head_dim * 4
    return resident + weights + qr_tiles + products + out_slots + rope_rows + scales


def _largest_divisor(n, cap, multiple):
    for d in range(min(n, cap) // multiple * multiple, 0, -multiple):
        if n % d == 0:
            return d
    return None


def default_config(tokens: int, rank: int, heads: int) -> QBRopeConfig:
    """Shape-derived block sizes; ``tokens`` must already be a multiple of 32."""
    tb = tokens if tokens <= 512 else _largest_divisor(tokens, 512, _ROW_ALIGN)
    quant_fragment = _largest_divisor(tb or 0, 64, _ROW_ALIGN)
    assert tb is not None and quant_fragment is not None  # 32 always divides
    hb = next(h for h in (16, 8, 4) if heads % h == 0)
    while hb > 4 and heads // hb * (tokens // tb) < 2:
        hb //= 2
    pairs_total = heads // 2 * (tokens // tb)
    return QBRopeConfig(
        token_block=tb,
        head_block=hb,
        output_slots=max(2, min(8, pairs_total)),
        quant_fragment=quant_fragment,
        post_fragment=32,
    )


def select_config(tokens: int, rank: int, width: int, heads: int) -> QBRopeConfig:
    from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import (
        get_tpu_version,
    )

    tuned = QB_ROPE_TUNING.get((get_tpu_version(), tokens, rank, width))
    return tuned or default_config(tokens, rank, heads)


def check_config(config: QBRopeConfig, segment: int, heads: int) -> None:
    tb, hb = config.token_block, config.head_block
    problems = []
    if tb <= 0 or tb % _ROW_ALIGN or segment % tb:
        problems.append(f"token_block {tb} must be a multiple of 32 dividing {segment}")
    if config.post_fragment % 8 or tb % config.post_fragment:
        problems.append("post_fragment must be a multiple of 8 dividing token_block")
    if config.quant_fragment % _ROW_ALIGN or tb % config.quant_fragment:
        problems.append("quant_fragment must be a multiple of 32 dividing token_block")
    if hb <= 0 or heads % hb or hb % 4:
        problems.append(f"head_block {hb} must be a multiple of 4 dividing {heads}")
    steps = heads // max(hb, 1) * (segment // max(tb, 1))
    if steps < 2:
        problems.append("the grid needs at least two steps")
    if not 2 <= config.output_slots <= steps * hb // 2:
        problems.append("output_slots must cover 2..total head pairs")
    if problems:
        raise ValueError("Invalid QBRopeConfig: " + "; ".join(problems))


def _segment_tokens(config, tokens, rank, heads, head_dim):
    if config.segment_tokens is not None:
        if tokens % config.segment_tokens or config.segment_tokens % config.token_block:
            raise ValueError(
                "segment_tokens must divide tokens and be tiled by token_block"
            )
        return config.segment_tokens
    segment = tokens
    while vmem_bytes(config, segment, rank, heads, head_dim) > config.vmem_limit_bytes:
        smaller = _largest_divisor(
            tokens, segment - config.token_block, config.token_block
        )
        if smaller is None:
            raise ValueError("No token segment fits the VMEM limit; lower token_block")
        segment = smaller
    return segment


def _segment_call(
    qr,
    weight,
    weight_scale,
    rope_rows,
    out,
    *,
    config,
    offset,
    segment,
    head_dim,
):
    tokens, rank = qr.shape
    heads = weight.shape[1] // head_dim
    tb, hb = config.token_block, config.head_block
    tiles, first_tile = segment // tb, offset // tb
    last_tile = first_tile + tiles - 1
    rotary_dim = rope_rows.shape[1]

    # Row inputs are only read in the first window; keep their block index
    # fixed afterwards so the pipeline does not fetch them again.
    def rows(h, t, *_):
        return (jnp.where(h == 0, first_tile + t, last_tile), 0)

    once = pl.Buffered(buffer_count=1)
    in_specs = [
        pl.BlockSpec((tb, rank), rows),
        pl.BlockSpec(memory_space=pltpu.HBM),  # streamed per head pair
        # The previous step's last pair may belong to the previous window.
        pl.BlockSpec(weight_scale.shape, lambda h, t, *_: (0,), once),
        pl.BlockSpec((tb, rotary_dim), rows),
    ]
    operands = [qr, weight, weight_scale, rope_rows]
    aliases = {}
    if out is not None:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
        operands.append(out)
        aliases = {len(operands) - 1: 0}
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(heads // hb, tiles),
        in_specs=in_specs,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM((segment, rank), jnp.float8_e4m3fn),
            pltpu.VMEM((segment, 1), jnp.float32),
            pltpu.VMEM((2, segment, _LANE), jnp.float32),
            pltpu.VMEM((2, tb, head_dim), jnp.float32),
            pltpu.VMEM((2, tb, head_dim), jnp.float32),
            pltpu.VMEM((config.output_slots, tb, head_dim), jnp.uint32),
            pltpu.SemaphoreType.DMA((config.output_slots,)),
            pltpu.VMEM((2, hb // 2, rank, 2 * head_dim), weight.dtype),
            pltpu.SemaphoreType.DMA((2, hb // 2)),
        ),
    )
    kernel = functools.partial(
        _qb_rope_kernel,
        head_dim=head_dim,
        tb=tb,
        pairs=hb // 2,
        quant_fragment=config.quant_fragment,
        post_fragment=config.post_fragment,
        token_offset=offset,
        aliased=out is not None,
    )
    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((tokens, heads, head_dim), jnp.bfloat16),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=config.vmem_limit_bytes,
        ),
        name=f"qb_rope_t{tb}_h{hb}_s{config.output_slots}",
    )(*operands)


def fused_qb_rope(
    qr: jax.Array,
    weight: jax.Array,
    weight_scale: jax.Array,
    rope_rows: jax.Array,
    *,
    head_dim: int = 512,
    config: QBRopeConfig | None = None,
) -> jax.Array:
    """Return BF16 Q[tokens, heads, head_dim] for normalized BF16 ``qr``.

    ``weight`` is FP8 ``[rank, heads*head_dim]`` in the canonical (k, n)
    layout with one FP32 scale per column. ``rope_rows`` is the per-step
    ``[tokens, rotary_dim]`` FP32 ``rope_gather.gather_rows`` of the model's
    packed cos/sin cache. Token counts that are not a multiple of 32 are
    zero-padded internally.
    """
    _check_projection_shapes(qr, weight, weight_scale)
    if head_dim <= 0 or weight.shape[1] % head_dim or head_dim % _LANE:
        raise ValueError("Q-B output width must be a multiple of a 128-lane head")
    tokens, rank = qr.shape
    rope_gather.check_rope_rows(rope_rows, tokens, head_dim)
    heads = weight.shape[1] // head_dim
    padded = -(-tokens // _ROW_ALIGN) * _ROW_ALIGN
    if padded != tokens:
        qr = jnp.pad(qr, ((0, padded - tokens), (0, 0)))
        rope_rows = jnp.pad(rope_rows, ((0, padded - tokens), (0, 0)))
    if config is None:
        config = select_config(padded, rank, weight.shape[1], heads)
    segment = _segment_tokens(config, padded, rank, heads, head_dim)
    check_config(config, segment, heads)
    weight_scale = weight_scale.reshape(-1)
    out = None
    for offset in range(0, padded, segment):
        out = _segment_call(
            qr,
            weight,
            weight_scale,
            rope_rows,
            out,
            config=config,
            offset=offset,
            segment=segment,
            head_dim=head_dim,
        )
    assert out is not None
    return out if padded == tokens else out[:tokens]


def _check_projection_shapes(qr, weight, scale):
    """``serial._check_projection`` without its fixed 256-row token block."""
    if qr.ndim != 2 or weight.ndim != 2 or qr.shape[0] == 0:
        raise ValueError("Use non-empty rank-2 activations and weights")
    if qr.dtype != jnp.bfloat16 or weight.dtype != jnp.float8_e4m3fn:
        raise ValueError("Use BF16 activations and FP8 E4M3FN weights")
    if qr.shape[1] != weight.shape[0]:
        raise ValueError("Weight input dimension must match the activation width")
    if scale.shape not in ((weight.shape[1],), (1, weight.shape[1])):
        raise ValueError("Use one weight scale per output column")
    if scale.dtype != jnp.float32:
        raise ValueError("Weight scales must be FP32")


def supported(qr, weight, weight_scale, rope_rows, head_dim=512) -> bool:
    """Whether ``fused_qb_rope`` accepts these concrete lowering inputs."""
    try:
        _check_projection_shapes(qr, weight, weight_scale)
        rope_gather.check_rope_rows(rope_rows, qr.shape[0], head_dim)
    except ValueError:
        return False
    return weight.shape[1] % head_dim == 0 and (weight.shape[1] // head_dim) % 4 == 0
