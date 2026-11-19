# SPDX-License-Identifier: Apache-2.0
"""Decode-sized HyperConnection mix as one Pallas call per call site.

At decode token counts the HC math (per-branch RMS norm, the low-rank
down/SiLU/up gate, the sigmoid-weighted branch mean and the combine's
injection gate) is a chain of ~15 tiny XLA fusions per call that are
dominated by launch and HBM round trips. This kernel streams the layer's
HC weights into VMEM once per call (one DMA per branch slice, overlapped
with the normalization) and keeps every intermediate on chip.

It reproduces the TorchTPU eager path's rounding boundaries (see
``GatedResidual.mix``/``combine`` in models/vllm/qwen4_exp/hyperconnection.py):

* norm = bf16(f32(x) * rsqrt(mean(x^2) + eps) * (1 + f32(w))) per branch
* gate = bf16(silu(f32(bf16(norm @ down^T) / hc)))
* mixed = bf16(sum_b(sigmoid(f32(bf16(gate @ up_b^T))) * f32(norm_b)) / hc)
* injection = 2 * sigmoid(f32(bf16(norm @ inject^T)) / hc), kept in f32

so it only differs from XLA by the matmul accumulation order and the
transcendental implementations (ulp level).

With ``prologue=True`` the kernel first applies the previous HC's combine,
h = bf16(f32(residual) + f32(block) * injection[:, b]), writes h (the next
residual) and mixes it; this fuses attn-combine with mlp-mix.

With ``up_t=True`` (TPU_QWEN4_HC_DECODE_UP_T) the kernel reads the up weight
as ``up.T``, ``[lowrank, hc * hidden]``. The served programs keep the bf16
``[hc * hidden, lowrank]`` up weight in the ``{0,1}`` tiled layout, i.e. as a
row-major ``[lowrank, hc * hidden]`` array, while a Mosaic operand must be
``{1,0}``: taking ``up`` as is costs an XLA relayout copy of the whole 13 MB
matrix before every call (xprof ``copy.weight_ref`` / ``copy.up_hbm``, about
6.5 us), and ``up.T`` is then a bitcast. Each branch is a 128-lane-aligned
column slab of ``up.T``. The products and the BF16/FP32 boundaries are the
same; only the MXU accumulation order may differ.
"""

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_TOKENS = 512
_SINGLE_TILE_MAX = 128
_ROW_CHUNK = 32


def decode_token_tile(tokens: int) -> int | None:
    """Token tile for ``tokens`` rows, or None when the kernel does not apply."""
    if tokens < 1 or tokens > MAX_TOKENS:
        return None
    if tokens <= _SINGLE_TILE_MAX:
        return tokens
    for tile in (128, 64, 32, 16):
        if tokens % tile == 0:
            return tile
    return None


# q38 PORT-BL HC: TPU_QWEN4_HC_DECODE_TILE_ROWS (default 0 = off), ported from Benliang Wang
# <benliangw@google.com>, commit 28419c4a5 on benliangw/q38fh-integ-pin-5b44702aea. Only the
# kernel's row tile changes: decode_token_tile above stays the torch-side gate (its `tokens <= 128`
# test is HCGATE's guard-neutral bound, so it is not changed here), and decode_kernel_tile is only
# evaluated on concrete shapes inside the jitted kernel wrapper, never on a traced token count.
_TILE_ROWS_MULTIPLE = 16


def decode_tile_rows(value: int) -> int:
    """Validated TPU_QWEN4_HC_DECODE_TILE_ROWS: 0 (off) or a multiple of 16 in [128, MAX_TOKENS].

    A smaller limit could not be honoured at 128 rows (the 128-row tile is kept), so it is
    refused rather than silently ignored."""
    rows = int(value)
    if rows == 0:
        return 0
    if (
        rows < _SINGLE_TILE_MAX
        or rows > MAX_TOKENS
        or rows % _TILE_ROWS_MULTIPLE
    ):
        raise ValueError(
            f"TPU_QWEN4_HC_DECODE_TILE_ROWS={rows}: must be 0 (off) or a multiple of "
            f"{_TILE_ROWS_MULTIPLE} in [{_SINGLE_TILE_MAX}, {MAX_TOKENS}]"
        )
    return rows


def decode_kernel_tile(tokens: int, max_tile: int = 0) -> int | None:
    """Row tile the kernel runs for a concrete ``tokens``, or None.

    ``max_tile`` 0 takes ``decode_token_tile``. Otherwise ``tokens`` of at
    least 128 split into the fewest equal tiles of at most ``max_tile`` rows
    that are a multiple of 16 (192 rows: one tile, 448: two of 224). Every
    tile streams the layer's low-rank weights through the MXU again, so a
    64-row tile costs nearly as much as a 128-row one. Accepts every count
    ``decode_token_tile`` accepts (it falls back to that tile), so the
    torch-side gate stays ``decode_token_tile``.
    """
    tile = decode_token_tile(tokens)
    if tile is None or not max_tile or tokens < _SINGLE_TILE_MAX:
        return tile
    for count in range(-(-tokens // max_tile), tokens // tile):
        if tokens % count == 0 and (tokens // count) % _TILE_ROWS_MULTIPLE == 0:
            return tokens // count
    return tile


# HCX (default off; port of q38mm 20e6fcf23, MM DSTOCK HCX): HC mixes of <= N rows
# take GatedResidual.mix's XLA path, the one non-multiple-of-128 buckets (96, 224)
# already run, instead of the prefill-tile norm / up-epilogue Pallas ops. Those ops
# cost a relayout copy of the 13 MB up weight per call (xprof copy.weight_ref, about
# 6.5 us). Unset or empty = off: every path exactly as before.
# compilation/tpu_compiler.py keys the compile cache on the value when it is set (and
# only then); it is deliberately not registered in envs.py, which would key every
# compile hash on it, set or not.
DECODE_XLA_MAX_ROWS_ENV = "VLLM_TPU_HC_XLA_MAX_ROWS"


_HCX_LIMITS = (128, 256, 512)   # q38 L3-HCGATE: the guard-neutral HCX limits (see decode_xla_applies)


def decode_xla_max_rows() -> int:
    """The HCX row limit: 0 (off) when unset, else the env value (128, 256 or 512)."""
    value = os.environ.get(DECODE_XLA_MAX_ROWS_ENV, "").strip()
    if not value:
        return 0
    try:
        rows = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value '{value}' for {DECODE_XLA_MAX_ROWS_ENV}: "
            "must be an integer."
        ) from exc
    if rows < 1:
        raise ValueError(f"{DECODE_XLA_MAX_ROWS_ENV} must be >= 1, got {rows}")
    if rows not in _HCX_LIMITS:
        raise ValueError(
            f"{DECODE_XLA_MAX_ROWS_ENV}={rows}: only 128, 256 and 512 are accepted (HCX may add "
            "no bound on the traced token count; see decode_xla_applies)"
        )
    return rows


def decode_xla_applies(tokens: int, limit: int | None = None) -> bool:
    """True when an HC mix of ``tokens`` rows takes the XLA path under HCX.

    ``limit`` is ``decode_xla_max_rows()`` (read here when None). Off returns
    False before the row count is compared, so an unset env leaves no
    token-count guard behind.

    q38 L3-HCGATE (T95u generalization): decided by divisibility, `tokens % (2 * limit) != 0`,
    not `tokens <= limit`, so it puts no bound on the traced token count (an upper bound at
    the limit can meet another module's lower bound and pin the symbol). HCX only changes HC
    mixes that have prefill-tile ops, i.e. rows that are multiples of the 128-row tile; over
    the decode bucket ladder (multiples of 16 up to 224, then 256, 512, 1024, 2048) a tile row
    is <= limit exactly when it is not a multiple of 2 x limit, for each accepted limit."""
    if limit is None:
        limit = decode_xla_max_rows()
    return limit > 0 and tokens % (2 * limit) != 0


# HCGATE (default off; q38 L3-HCGATE): with TPU_QWEN4_HC_DECODE_FUSED armed, the fused
# decode HC kernel runs only where the row gate allows it. A disallowed input takes the
# path it takes with TPU_QWEN4_HC_DECODE_FUSED off: GatedResidual.mix's XLA path at
# buckets that are not a multiple of the prefill tiles (112 / 160 / 192 / 224 at FINAL's
# 128 / 128 / 256 tiles), the prefill-tile ops at 128 / 256 unless HCX
# (VLLM_TPU_HC_XLA_MAX_ROWS) routes them to XLA too. Reason (M, J1' layer v9, b1 + b2):
# HC fused costs +2.859 ms per step at 208 rows (bucket 224, 32-row tiles) and is flat at
# <= 96 rows.
#   N     fused only for rows <= N
#   N:M   fused for rows <= N and for rows >= M (M >= N + 16, M <= MAX_TOKENS): the rows
#         strictly between N and M take the unfused path. Keeps the 256 route a parameter
#         ("128" sends 256 to the unfused path, "128:256" keeps it fused).
# Guard neutrality (T95u fix; L2 localization, M: the old `tokens >= M` put a lower bound of
# 256 on the traced token SymInt, which met an upper bound of 256 elsewhere in the graph, so
# the 256-bucket retrace pinned the symbol, dropped two SymInt placeholders and was refused
# by the shape-variant session): the gate may only test what decode_token_tile already tests.
# So only N = 128 (decode_token_tile's own `tokens <= 128`) and M = 256 are accepted, and
# `rows >= 256` is decided as `tokens % 128 == 0` (above 128 rows), which is decode_token_tile's
# own first tile test and puts no bound on the symbol's value range. Over the bucket ladder
# (16 ... 64 by 16, 80 / 96 / 112 / 160 / 192 / 224, and the powers of two) this gives the
# same decision as `rows >= 256` at every bucket: 256 / 512 fused, 160 / 192 / 224 not, and
# rows above 512 have no decode tile either way. Any other N or M raises at build.
# Floor (lead ruling, cycle 4): N < 128 needs HCX >= 128, else the 128 bucket would fall
# to the slow prefill-tile path; such a configuration raises at build.
# Unset or empty = off: every path exactly as before. compilation/tpu_compiler.py keys
# the compile cache on the value when it is set (and only then); it is deliberately not
# registered in envs.py, which would key every compile hash on it, set or not.
DECODE_FUSED_MAX_ROWS_ENV = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"
DECODE_FUSED_FLOOR_ROWS = 128   # FINAL's HC prefill norm / up tile: the first bucket with a tile path
_BAND_MIN_GAP = 16
_GUARD_NEUTRAL_LO = _SINGLE_TILE_MAX          # 128: decode_token_tile's own single-tile bound
_GUARD_NEUTRAL_HI = (0, 2 * _SINGLE_TILE_MAX)  # off, or 256 = "a multiple of the 128-row tile above 128"


def _rows_int(value: str, part: str) -> int:
    try:
        return int(part)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value '{value}' for {DECODE_FUSED_MAX_ROWS_ENV}: "
            "must be N or N:M with integers N and M."
        ) from exc


def decode_fused_max_rows() -> tuple[int, int]:
    """The HCGATE limits ``(N, M)``: ``(0, 0)`` (off) when unset, ``(N, 0)`` for ``N``,
    ``(N, M)`` for ``N:M``. Raises on a malformed value and on N < 128 without
    HCX >= 128 (the floor)."""
    value = os.environ.get(DECODE_FUSED_MAX_ROWS_ENV, "").strip()
    if not value:
        return 0, 0
    parts = value.split(":")
    if len(parts) > 2:
        raise ValueError(f"Invalid value '{value}' for {DECODE_FUSED_MAX_ROWS_ENV}: at most one ':'.")
    lo = _rows_int(value, parts[0])
    hi = _rows_int(value, parts[1]) if len(parts) == 2 else 0
    if lo < 1:
        raise ValueError(f"{DECODE_FUSED_MAX_ROWS_ENV} must be >= 1, got {lo}")
    if len(parts) == 2 and not (lo + _BAND_MIN_GAP <= hi <= MAX_TOKENS):
        raise ValueError(
            f"{DECODE_FUSED_MAX_ROWS_ENV}={value}: M must satisfy "
            f"N + {_BAND_MIN_GAP} <= M <= {MAX_TOKENS}"
        )
    if lo < DECODE_FUSED_FLOOR_ROWS and decode_xla_max_rows() < DECODE_FUSED_FLOOR_ROWS:
        raise ValueError(
            f"{DECODE_FUSED_MAX_ROWS_ENV}={value}: N < {DECODE_FUSED_FLOOR_ROWS} needs "
            f"{DECODE_XLA_MAX_ROWS_ENV} >= {DECODE_FUSED_FLOOR_ROWS} (the "
            f"{DECODE_FUSED_FLOOR_ROWS}-row bucket would take the prefill-tile path)"
        )
    if lo != _GUARD_NEUTRAL_LO or hi not in _GUARD_NEUTRAL_HI:
        raise ValueError(
            f"{DECODE_FUSED_MAX_ROWS_ENV}={value}: only 128 and 128:256 are accepted (the "
            "gate may add no bound on the traced token count; T95u refused its 256-bucket "
            "trace under the old 'rows >= M' test)"
        )
    return lo, hi


def decode_fused_allows(tokens: int, limit) -> bool:
    """Whether the HCGATE limits let an HC of ``tokens`` rows take the fused decode
    kernel (the decode tile is checked separately). ``limit`` is
    ``decode_fused_max_rows()`` (an int N is read as ``(N, 0)``). Off returns True before
    the row count is compared, so an unset env adds no token-count guard while tracing."""
    lo, hi = (limit, 0) if isinstance(limit, int) else limit
    # hi (only 256 is accepted) is decided as "a multiple of the 128-row tile": no value bound
    # on the token symbol (see the guard-neutrality note above).
    return lo <= 0 or tokens <= lo or (hi > 0 and tokens % _SINGLE_TILE_MAX == 0)


def _kernel(
    *refs,
    branches: int,
    hidden: int,
    eps: float,
    prologue: bool,
    inject: bool,
    up_t: bool = False,
    chunk_tail: bool = False,
):
    refs = list(refs)
    if prologue:
        prev_ref, block_ref, prev_inj_ref = refs[:3]
        refs = refs[3:]
        x_ref = None
    else:
        x_ref = refs.pop(0)
    w_hbm, down_hbm, up_hbm = refs[:3]
    refs = refs[3:]
    injw_hbm = refs.pop(0) if inject else None
    mixed_ref = refs.pop(0)
    inj_ref = refs.pop(0) if inject else None
    h_ref = refs.pop(0) if prologue else None
    w_vm, down_vm, up_vm = refs[:3]
    refs = refs[3:]
    injw_vm = refs.pop(0) if inject else None
    norm_vm, sems = refs

    first = pl.program_id(0) == 0
    w_copy = pltpu.make_async_copy(w_hbm, w_vm, sems.at[0])
    down_copy = pltpu.make_async_copy(down_hbm, down_vm, sems.at[1])
    if up_t:
        # up.T: branch b is the [lowrank, hidden] column slab b.
        up_copies = [
            pltpu.make_async_copy(
                up_hbm.at[:, pl.ds(b * hidden, hidden)],
                up_vm.at[:, pl.ds(b * hidden, hidden)],
                sems.at[2 + b],
            )
            for b in range(branches)
        ]
    else:
        up_copies = [
            pltpu.make_async_copy(
                up_hbm.at[pl.ds(b * hidden, hidden)],
                up_vm.at[pl.ds(b * hidden, hidden)],
                sems.at[2 + b],
            )
            for b in range(branches)
        ]
    inj_copy = (
        pltpu.make_async_copy(injw_hbm, injw_vm, sems.at[2 + branches])
        if inject
        else None
    )

    # Weights stay resident in scratch across token tiles: fetch them once.
    @pl.when(first)
    def _start_weights():
        w_copy.start()
        if inject:
            inj_copy.start()
        down_copy.start()
        for copy in up_copies:
            copy.start()

    def wait_first(copy):
        @pl.when(first)
        def _wait():
            copy.wait()

    def lane_column(values, column):
        # values[:, column:column + 1] as a masked lane reduction (exact:
        # one non-zero term), avoiding an unaligned lane slice.
        lanes = jax.lax.broadcasted_iota(jnp.int32, values.shape, 1)
        return jnp.sum(jnp.where(lanes == column, values, 0.0), axis=1, keepdims=True)

    nt = (((1,), (1,)), ((), ()))
    nn = (((1,), (0,)), ((), ()))
    wait_first(w_copy)
    if inject:
        wait_first(inj_copy)

    def normalize_rows(offset, rows):
        """Norm (and combine prologue, injection gate) for ``rows`` rows."""
        weight = w_vm[...].astype(jnp.float32)
        if prologue:
            block = block_ref[pl.ds(offset, rows), :].astype(jnp.float32)
            prev_inj = prev_inj_ref[pl.ds(offset, rows), :]
        inj_terms = [None] * branches
        for b in range(branches):
            start = b * hidden
            if prologue:
                h = (
                    prev_ref[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                        jnp.float32
                    )
                    + block * lane_column(prev_inj, b)
                ).astype(jnp.bfloat16)
                h_ref[pl.ds(offset, rows), pl.ds(start, hidden)] = h
                values = h.astype(jnp.float32)
            else:
                values = x_ref[pl.ds(offset, rows), pl.ds(start, hidden)].astype(
                    jnp.float32
                )
            variance = jnp.sum(values * values, axis=-1, keepdims=True) / hidden
            normalized = values * jax.lax.rsqrt(variance + eps)
            norm = normalized * (1.0 + weight[None, start : start + hidden])
            norm = norm.astype(jnp.bfloat16)
            norm_vm[pl.ds(offset, rows), pl.ds(start, hidden)] = norm
            if inject:
                # The [T, hc] injection logits on the VPU: BF16 x BF16
                # products are exact in FP32; only the summation order
                # differs from the MXU.
                norm32 = norm.astype(jnp.float32)
                for j in range(branches):
                    row = injw_vm[pl.ds(j, 1), pl.ds(start, hidden)]
                    term = norm32 * row.astype(jnp.float32)
                    inj_terms[j] = term if inj_terms[j] is None else inj_terms[j] + term
        if inject:
            shape = (rows, branches)
            lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
            logits = jnp.zeros(shape, jnp.float32)
            for j in range(branches):
                column = jnp.sum(inj_terms[j], axis=1, keepdims=True)
                logits = jnp.where(lanes == j, column, logits)
            logits = logits.astype(jnp.bfloat16).astype(jnp.float32)
            inj_ref[pl.ds(offset, rows), :] = 2.0 * jax.nn.sigmoid(logits / branches)

    tile = norm_vm.shape[0]
    chunks, tail = divmod(tile, _ROW_CHUNK)
    # q38 PORT-BL HC (TILE_ROWS, Benliang Wang 28419c4a5): ``chunk_tail`` (only with
    # TPU_QWEN4_HC_DECODE_TILE_ROWS set) lets a tile of 16 x odd rows run the 32-row loop plus
    # one 16-row tail. Off, the condition below is exactly the old one (tile >= 64 and
    # tile % 32 == 0) and ``tail`` is 0, so the kernel is unchanged.
    if chunks >= 2 and (tail == 0 or (chunk_tail and tail % 16 == 0)):
        # Bound the live FP32 row working set like hc_full's 32-row loop.
        def chunk(index, carry):
            normalize_rows(pl.multiple_of(index * _ROW_CHUNK, _ROW_CHUNK), _ROW_CHUNK)
            return carry

        jax.lax.fori_loop(0, chunks, chunk, jnp.int32(0))
        if tail:
            normalize_rows(chunks * _ROW_CHUNK, tail)
    else:
        normalize_rows(0, tile)

    wait_first(down_copy)
    down_acc = jax.lax.dot_general(
        norm_vm[...],
        down_vm[...],
        dimension_numbers=nt,
        preferred_element_type=jnp.float32,
    )

    gate = (down_acc.astype(jnp.bfloat16) / branches).astype(jnp.bfloat16)
    gate = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
    mixed = None
    for b in range(branches):
        start = b * hidden
        wait_first(up_copies[b])
        if up_t:
            up_acc = jax.lax.dot_general(
                gate,
                up_vm[:, pl.ds(start, hidden)],
                dimension_numbers=nn,
                preferred_element_type=jnp.float32,
            )
        else:
            up_acc = jax.lax.dot_general(
                gate,
                up_vm[pl.ds(start, hidden), :],
                dimension_numbers=nt,
                preferred_element_type=jnp.float32,
            )
        logits = up_acc.astype(jnp.bfloat16).astype(jnp.float32)
        weighted = jax.nn.sigmoid(logits) * norm_vm[:, pl.ds(start, hidden)].astype(
            jnp.float32
        )
        mixed = weighted if mixed is None else mixed + weighted
    mixed_ref[...] = (mixed / branches).astype(jnp.bfloat16)


def hc_decode_mix(
    x,
    weight,
    down,
    up,
    inject_weight=None,
    *,
    branches: int,
    hidden: int,
    eps: float,
    block=None,
    prev_injection=None,
    up_t=False,
    interpret=False,
    max_tile=0,
):
    """Fused HC mix for at most ``MAX_TOKENS`` rows.

    ``x`` is the [T, hc*H] hyper input, or with ``block``/``prev_injection``
    the previous HC's residual that is combined first. Returns
    ``(mixed, injection)`` (``injection`` is None without ``inject_weight``),
    plus the combined hyper input when ``block`` is given. ``up`` is always
    the checkpoint ``[hc * hidden, lowrank]`` weight; ``up_t`` only changes
    how the kernel reads it (see the module docstring). ``max_tile``: see
    ``decode_kernel_tile`` (TPU_QWEN4_HC_DECODE_TILE_ROWS; 0 = the
    ``decode_token_tile`` tiling).
    """
    return _hc_decode_mix(
        x,
        weight,
        down,
        up,
        inject_weight,
        block,
        prev_injection,
        branches=branches,
        hidden=hidden,
        eps=eps,
        up_t=bool(up_t),
        interpret=interpret,
        max_tile=int(max_tile),
    )


@functools.partial(
    jax.jit,
    static_argnames=("branches", "hidden", "eps", "up_t", "interpret", "max_tile"),
)
def _hc_decode_mix(
    x,
    weight,
    down,
    up,
    inject_weight,
    block,
    prev_injection,
    *,
    branches,
    hidden,
    eps,
    up_t,
    interpret,
    max_tile=0,
):
    tokens, width = x.shape
    lowrank = down.shape[0]
    prologue = block is not None
    inject = inject_weight is not None
    tile = decode_kernel_tile(tokens, max_tile)
    if (
        tile is None
        or width != branches * hidden
        or hidden % 128
        or weight.shape != (width,)
        or down.shape != (lowrank, width)
        or up.shape != (width, lowrank)
        or (inject and inject_weight.shape != (branches, width))
        or (prologue and block.shape != (tokens, hidden))
        or (prologue and prev_injection.shape != (tokens, branches))
        or (prologue and prev_injection.dtype != jnp.float32)
        or any(
            a.dtype != jnp.bfloat16
            for a in (x, weight, down, up)
            + ((inject_weight,) if inject else ())
            + ((block,) if prologue else ())
        )
    ):
        raise ValueError("HC decode mix requires BF16 decode-sized token tiles")

    def rows(cols):
        return pl.BlockSpec((tile, cols), lambda i: (i, 0))

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    in_specs = (
        [rows(width), rows(hidden), rows(branches)] if prologue else [rows(width)]
    )
    in_specs += [hbm] * (4 if inject else 3)
    out_specs = [rows(hidden)]
    out_shape = [jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)]
    if inject:
        out_specs.append(rows(branches))
        out_shape.append(jax.ShapeDtypeStruct((tokens, branches), jnp.float32))
    if prologue:
        out_specs.append(rows(width))
        out_shape.append(jax.ShapeDtypeStruct((tokens, width), jnp.bfloat16))
    scratch = [
        pltpu.VMEM((width,), jnp.bfloat16),
        pltpu.VMEM((lowrank, width), jnp.bfloat16),
        pltpu.VMEM((lowrank, width) if up_t else (width, lowrank), jnp.bfloat16),
    ]
    if inject:
        scratch.append(pltpu.VMEM((branches, width), jnp.bfloat16))
    scratch += [
        pltpu.VMEM((tile, width), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((3 + branches,)),
    ]
    args = [x, block, prev_injection] if prologue else [x]
    args += [weight, down, up.T if up_t else up] + ([inject_weight] if inject else [])
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            branches=branches,
            hidden=hidden,
            eps=eps,
            prologue=prologue,
            inject=inject,
            up_t=up_t,
            chunk_tail=bool(max_tile),
        ),
        grid=(tokens // tile,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            # Weights are fetched on the first tile only: keep tiles in order.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=56 << 20,
        ),
        interpret=interpret,
        name=(
            f"qwen4exp_hc_decode{'_combine' if prologue else ''}"
            f"{'_upt' if up_t else ''}"
        ),
    )(*args)
    mixed = outs[0]
    injection = outs[1] if inject else None
    if prologue:
        return mixed, injection, outs[-1]
    return mixed, injection
