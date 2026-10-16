# SPDX-License-Identifier: Apache-2.0

"""V4.1 single-pass shifted mHC with BF16 RMS output and separate FP32 gates."""

import functools
import math
from enum import StrEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class PostPreTuning(NamedTuple):
    token_tile: int
    mix_group: int
    collapse_columns: int
    vmem_limit_bytes: int


class GateInputMode(StrEnum):
    """HBM DMA control or compiler-managed VMEM input windows."""

    HBM = "hbm"
    COMPILER = "compiler"


def pack_mix_weight(fn: jax.Array) -> jax.Array:
    """Pack FP32 [24,20480] as three BF16 components in [20480,128]."""
    assert fn.shape == (24, 20480) and fn.dtype == jnp.float32
    high = _truncate_bf16(fn)
    middle = _truncate_bf16(fn - high)
    low = fn - high - middle
    parts = jnp.concatenate([high, middle, low]).astype(jnp.bfloat16)
    return jnp.pad(parts.T, ((0, 0), (0, 128 - parts.shape[0])))


def _truncate_bf16(values: jax.Array) -> jax.Array:
    bits = jax.lax.bitcast_convert_type(values, jnp.uint32) & jnp.uint32(4294901760)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def shifted_post_pre_rms(
    x: jax.Array,
    flat: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    pre: jax.Array,
    *,
    rms_eps: float = 1e-20,
    hc_pre_eps: float = 1e-06,
    hc_sinkhorn_eps: float = 1e-06,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
    gate_input_mode: GateInputMode = GateInputMode.HBM,
    constrain_activations_to_hbm: bool = True,
    norm_weight: jax.Array | None = None,
    norm_eps: float = 1e-06,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update residuals and collapse with incoming pre; predict gates separately.

    TPU7x: M=4, D=5120, positive local T divisible by 32.
    Activations are BF16; gates are FP32. Weight is packed BF16 [20480,128].
    Three-component mixing and BF16 residual/collapse rounding are retained;
    streamed square sums change the reference reduction order.

    Returns:
        (residual [T,M*D], next_post [T,M,1], next_comb [T,M,M],
        value [T,D], next_pre [T,M]). Value includes RMSNorm when
        norm_weight [D] is supplied. Other keyword arguments are JIT-static.
    """
    t, d = x.shape
    token_tile = 128 if t % 128 == 0 else 96 if t % 96 == 0 else 32
    num_tiles = t // token_tile
    input_buffers = min(3, num_tiles)
    output_buffers = min(2, num_tiles)
    # Early stores shorten the drain for batches that fit in the input buffers.
    early_output = num_tiles <= input_buffers
    m = pre.shape[-1]
    width = m * d
    parts = 3
    info = pltpu.get_tpu_info()
    assert info.chip_version == pltpu.ChipVersion.TPU_7X, "Requires TPU7x"
    mixes = m * (m + 2)
    assert t > 0 and t % 32 == 0 and m == 4 and d == 5120
    gate_tokens = (t + 127) // 128 * 128
    tuning = PostPreTuning(token_tile, 32, 256, 50 * 1024 * 1024)
    tile, group, columns = (
        tuning.token_tile,
        tuning.mix_group,
        tuning.collapse_columns,
    )
    assert flat.shape == (t, width) and pre.shape == (t, m)
    assert post.shape in ((t, m), (t, m, 1)) and comb.shape == (t, m, m)
    assert x.dtype == flat.dtype == jnp.bfloat16
    assert post.dtype == comb.dtype == pre.dtype == jnp.float32
    assert weight.shape == (width, 128) and weight.dtype == jnp.bfloat16
    assert scale.shape == (3,) and base.shape == (mixes,)
    assert scale.dtype == base.dtype == jnp.float32
    gate_input_mode = GateInputMode(gate_input_mode)
    norm_args = () if norm_weight is None else (norm_weight,)
    if norm_weight is not None:
        assert norm_weight.shape == (d,)
        assert norm_weight.dtype in (jnp.bfloat16, jnp.float32)

    def kernel(
        xh,
        rh,
        poh,
        ch,
        ph,
        wh,
        norm_refs,
        yh,
        vh,
        mh,
        sh,
        output_y,
        output_v,
        output_sems,
        input_x,
        input_residual,
        input_sems,
        gate_scratch,
        norm_weight_scratch,
        wr,
        coeff,
        acts,
        sbuf,
        norm_square,
        sems,
        projection_buf,
    ):
        if gate_input_mode == GateInputMode.HBM:
            por, cr, pr = gate_scratch
            copy_pairs = [(poh, por), (ch, cr), (ph, pr), (wh, wr)]
        else:
            por, cr, pr = (poh, ch, ph)
            copy_pairs = [(wh, wr)]
        norm_work_refs = norm_refs
        if norm_weight is not None:
            copy_pairs.insert(-1, (norm_refs[0], norm_weight_scratch[0]))
            norm_work_refs = norm_weight_scratch
        copies = [
            pltpu.make_async_copy(src, dst, sems.at[i])
            for i, (src, dst) in enumerate(copy_pairs)
        ]
        for copy in copies[:-1]:
            copy.start()

        def load_gate(ref, tile_index):
            if tile % 128 == 0:
                return ref[..., pl.ds(tile_index * tile, tile)]
            # A 96-token tile may cross two 128-lane windows.
            start = tile_index * tile
            aligned = start // 128 * 128
            first = ref[..., pl.ds(aligned, 128)]
            second = ref[..., pl.ds(jnp.minimum(aligned + 128, gate_tokens - 128), 128)]
            pair = jnp.concatenate([first, second], axis=first.ndim - 1)
            return pltpu.roll(pair, (256 - start % 128) % 256, axis=pair.ndim - 1)[
                ..., :tile
            ]

        def prepare(tile_index, slot):
            # Map tokens to block diagonals; accumulate BF16 parts low to high.
            post_tile = load_gate(por, tile_index)
            comb_tile = load_gate(cr, tile_index)
            for ri in range(tile // group):
                diagonal = jnp.eye(group, dtype=jnp.float32)

                p = post_tile[:, ri * group : (ri + 1) * group]
                c = comb_tile[:, :, ri * group : (ri + 1) * group]
                cf = jnp.concatenate(
                    [
                        jnp.concatenate(
                            [p[j : j + 1, :] * diagonal]
                            + [c[i, j : j + 1, :] * diagonal for i in range(m)],
                            axis=1,
                        )
                        for j in range(m)
                    ],
                    axis=0,
                )
                pieces = []
                for part in range(parts):
                    bf = cf.astype(jnp.bfloat16)
                    pieces.append(bf)
                    cf = cf - bf.astype(jnp.float32)
                coeff[slot, ri, :, :] = jnp.concatenate(pieces[::-1], axis=1)

        def finish_projection(projected):
            pt = projected.T
            return pt[:mixes] + pt[mixes : 2 * mixes] + pt[2 * mixes : 3 * mixes]

        def store_vector(values, output, start):
            if tile % 128 == 0:
                output[:, pl.ds(start, tile)] = values
                return
            # Preserve adjacent tiles when this write crosses a lane window.
            block = start // 128 * 128
            offset = start % 128
            pair = jnp.pad(values, ((0, 0), (0, 256 - tile)))
            positioned = pltpu.roll(pair, offset, axis=1)
            lane = jnp.arange(128)
            mask = (lane >= offset) & (lane < offset + tile)
            output[:, pl.ds(block, 128)] = jnp.where(
                mask[None, :], positioned[:, :128], output[:, pl.ds(block, 128)]
            )

            @pl.when(offset + tile > 128)
            def second():
                mask = lane + 128 < offset + tile
                output[:, pl.ds(block + 128, 128)] = jnp.where(
                    mask[None, :],
                    positioned[:, 128:],
                    output[:, pl.ds(block + 128, 128)],
                )

        def store_projection(projected, tile_index):
            store_vector(finish_projection(projected), mh, tile_index * tile)

        if tile % 128 != 0:
            mh[...] = jnp.zeros((mixes, gate_tokens), jnp.float32)
            sh[...] = jnp.zeros((1, gate_tokens), jnp.float32)
        projection_buf[...] = jnp.zeros((tile, 128), dtype=jnp.float32)

        def input_copies(tile_index):
            slot = tile_index % input_buffers
            rows = pl.ds(tile_index * tile, tile)
            return (
                pltpu.make_async_copy(
                    xh.at[rows, :], input_x.at[slot], input_sems.at[slot, 0]
                ),
                pltpu.make_async_copy(
                    rh.at[rows, :], input_residual.at[slot], input_sems.at[slot, 1]
                ),
            )

        def output_copy(tile_index, component):
            slot = tile_index % output_buffers
            rows = pl.ds(tile_index * tile, tile)
            source = (output_y, output_v)[component]
            target = (yh, vh)[component]
            return pltpu.make_async_copy(
                source.at[slot], target.at[rows, :], output_sems.at[slot, component]
            )

        def output_copies(tile_index):
            return tuple(output_copy(tile_index, component) for component in range(2))

        def run_tile(tile_index, xr, rr, yr, vr):
            previous_projected = projection_buf[...]

            @pl.when(tile_index == 0)
            def wait_gates():
                for copy in copies[:-1]:
                    copy.wait()
                prepare(0, 0)

            pre_tile = load_gate(pr, tile_index)
            slot = tile_index % 2
            for ri in range(tile // group):
                r = pl.ds(ri * group, group)
                acts[:group, :] = xr[r, :]
                for i in range(m):
                    acts[pl.ds((i + 1) * group, group), :] = rr[r, pl.ds(i * d, d)]
                # Accumulate squares from the rounded residual in column order.
                residual_squares = jnp.zeros((group, 128), jnp.float32)
                c = jnp.pad(
                    coeff[slot, ri, :, :], ((0, 0), (0, -parts * (m + 1) * group % 256))
                )
                value_squares = jnp.zeros((group, 128), jnp.float32)
                for ec in range(d // columns):
                    cs = pl.ds(ec * columns, columns)
                    a = acts[:, cs]
                    rhs = jnp.pad(
                        jnp.concatenate([a] * parts, axis=0),
                        ((0, -parts * (m + 1) * group % 256), (0, 0)),
                    )
                    result = jax.lax.dot(
                        c, rhs, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)
                    for j in range(m):
                        yr[r, pl.ds(j * d + ec * columns, columns)] = result[
                            j * group : (j + 1) * group, :
                        ]
                    terms = []
                    for j in range(m):
                        bf = result[j * group : (j + 1) * group, :]
                        for k in range(columns // 128):
                            q = bf[:, k * 128 : (k + 1) * 128].astype(jnp.float32)
                            residual_squares = residual_squares + q * q
                        terms.append(
                            pre_tile[j : j + 1, ri * group : (ri + 1) * group].T
                            * bf.astype(jnp.float32)
                        )
                    value_bf16 = functools.reduce(jnp.add, terms).astype(jnp.bfloat16)
                    vr[r, pl.ds(ec * columns, columns)] = value_bf16
                    if norm_weight is not None:
                        value_f32 = value_bf16.astype(jnp.float32)
                        for lane_group in range(columns // 128):
                            chunk = value_f32[
                                :, lane_group * 128 : (lane_group + 1) * 128
                            ]
                            value_squares = value_squares + chunk * chunk
                if norm_weight is not None:
                    norm_square[0][r, :] = jnp.sum(value_squares, axis=1, keepdims=True)
                sbuf[r, :] = jnp.sum(residual_squares, axis=1, keepdims=True)
                if norm_weight is not None:
                    _rms_norm_inplace(
                        vr.at[pl.ds(ri * group, group), :],
                        norm_work_refs[0],
                        norm_square[0][pl.ds(ri * group, group), :],
                        columns=columns,
                        eps=norm_eps,
                    )
            if early_output:
                output_copy(tile_index, 0).start()
                output_copy(tile_index, 1).start()
            prepare(
                jnp.minimum(tile_index + 1, num_tiles - 1),
                (tile_index + 1) % 2,
            )

            @pl.when(tile_index == 0)
            def wait_weight():
                copies[-1].wait()

            projection_buf[...] = jax.lax.dot(
                yr[...], wr[...], preferred_element_type=jnp.float32
            )
            store_projection(previous_projected, jnp.maximum(tile_index - 1, 0))
            store_vector(sbuf[...].T, sh, tile_index * tile)

        for index in range(input_buffers):
            for copy in input_copies(index):
                copy.start()
            if index == 0:
                copies[-1].start()

        def body(tile_index, unused):
            for copy in input_copies(tile_index):
                copy.wait()
            if num_tiles > output_buffers:

                @pl.when(tile_index >= output_buffers)
                def wait_output_slot():
                    for copy in output_copies(tile_index - output_buffers):
                        copy.wait()

            slot = tile_index % input_buffers
            output_slot = tile_index % output_buffers
            run_tile(
                tile_index,
                input_x.at[slot],
                input_residual.at[slot],
                output_y.at[output_slot],
                output_v.at[output_slot],
            )
            if not early_output:
                for copy in output_copies(tile_index):
                    copy.start()
            if num_tiles > input_buffers:

                @pl.when(tile_index + input_buffers < num_tiles)
                def prefetch():
                    for copy in input_copies(tile_index + input_buffers):
                        copy.start()

            return unused

        jax.lax.fori_loop(0, num_tiles, body, None)
        for index in range(num_tiles - output_buffers, num_tiles):
            for copy in output_copies(index):
                copy.wait()
        store_projection(projection_buf[...], num_tiles - 1)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    gate_args = (post.reshape(t, m).T, comb.transpose(1, 2, 0), pre.T)
    if gate_input_mode == GateInputMode.HBM:
        gate_spec = hbm
        gate_args = tuple(
            pltpu.with_memory_space_constraint(gate, pltpu.HBM) for gate in gate_args
        )
        gate_scratch_shapes = tuple(
            pltpu.VMEM(gate.shape, gate.dtype) for gate in gate_args
        )
    else:
        gate_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
        gate_scratch_shapes = ()
    residual, value, mix_values, square = pl.pallas_call(
        kernel,
        in_specs=[hbm, hbm, gate_spec, gate_spec, gate_spec, hbm]
        + [tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in norm_args)],
        out_specs=[hbm, hbm] + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 2,
        out_shape=[jax.ShapeDtypeStruct((t, k), flat.dtype) for k in (width, d)]
        + [jax.ShapeDtypeStruct((k, gate_tokens), jnp.float32) for k in (mixes, 1)],
        scratch_shapes=[
            pltpu.VMEM((output_buffers, tile, width), jnp.bfloat16),
            pltpu.VMEM((output_buffers, tile, d), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((output_buffers, 2)),
            pltpu.VMEM((input_buffers, tile, d), jnp.bfloat16),
            pltpu.VMEM((input_buffers, tile, width), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((input_buffers, 2)),
            gate_scratch_shapes,
            tuple(pltpu.VMEM(arg.shape, arg.dtype) for arg in norm_args),
            pltpu.VMEM(weight.shape, weight.dtype),
            pltpu.VMEM(
                (2, tile // group, m * group, parts * (m + 1) * group), jnp.bfloat16
            ),
            pltpu.VMEM(((m + 1) * group, d), jnp.bfloat16),
            pltpu.VMEM((tile, 1), jnp.float32),
            (pltpu.VMEM((tile, 1), jnp.float32),) if norm_weight is not None else (),
            pltpu.SemaphoreType.DMA(
                (len(gate_scratch_shapes) + 1 + int(norm_weight is not None),)
            ),
            pltpu.VMEM((tile, 128), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tuning.vmem_limit_bytes,
            allow_input_fusion=(False, False, False, False, False, False)
            + (False,) * len(norm_args),
        ),
    )(
        pltpu.with_memory_space_constraint(x, pltpu.HBM)
        if constrain_activations_to_hbm
        else x,
        pltpu.with_memory_space_constraint(flat, pltpu.HBM)
        if constrain_activations_to_hbm
        else flat,
        *gate_args,
        pltpu.with_memory_space_constraint(weight, pltpu.HBM),
        tuple(pltpu.with_memory_space_constraint(arg, pltpu.HBM) for arg in norm_args),
    )
    new_pre, new_post, new_comb = _pre_gates(
        mix_values,
        square,
        scale,
        base,
        hidden_size=d,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
    return (residual, new_post[:t, :, None], new_comb[:t], value, new_pre[:t])


def _rms_norm_inplace(value_ref, weight_ref, square, *, columns: int, eps: float):
    """Normalize a BF16 token group in VMEM using column fragments."""
    _, width = value_ref.shape
    inverse_rms = jax.lax.rsqrt(square / width + eps)
    for column in range(width // columns):
        cs = pl.ds(column * columns, columns)
        part = value_ref[:, cs].astype(jnp.float32)
        weight = weight_ref[cs].astype(jnp.float32)[None, :]
        value_ref[:, cs] = (part * inverse_rms * weight).astype(value_ref.dtype)


def _pre_gates(
    mix: jax.Array,
    square: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    *,
    hidden_size: int,
    rms_eps: float = 1e-20,
    hc_pre_eps: float = 1e-06,
    hc_sinkhorn_eps: float = 1e-06,
    hc_post_mult_value: float = 2.0,
    sinkhorn_repeat: int = 20,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FP32 gates from logits [M*(M+2),T] and square sums [1,T]; hidden_size=D."""
    mixes, t = mix.shape
    m = math.isqrt(mixes + 1) - 1
    assert m > 0 and mixes == m * (m + 2), "Expected M*(M+2) mix logits"
    block = t if t <= 1024 else math.gcd(t, 1024)
    assert t > 0 and block % 128 == 0 and (t % block == 0), (
        "Gate token count must be a positive multiple of 128"
    )
    assert hidden_size > 0, "hidden_size must be positive"
    assert square.shape == (1, t) and scale.shape == (3,) and (base.shape == (mixes,))
    assert mix.dtype == square.dtype == scale.dtype == base.dtype == jnp.float32
    if not isinstance(sinkhorn_repeat, int) or sinkhorn_repeat < 1:
        raise ValueError("sinkhorn_repeat must be a positive static integer")

    def gate_kernel(mr, sr, scr, br, pre_out, post_out, comb_out):
        mixed = mr[...] * jax.lax.rsqrt(sr[...] / (m * hidden_size) + rms_eps)
        scales = scr[...][:, None]
        bias = br[...][:, None]
        pre_logits = mixed[:m, :] * scales[:1, :] + bias[:m, :]
        pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps
        post_logits = mixed[m : 2 * m, :] * scales[1:2, :] + bias[m : 2 * m, :]
        post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value
        comb_logits = (mixed[2 * m :, :] * scales[2:3, :] + bias[2 * m :, :]).reshape(
            m, m, block
        )
        comb_mix = jax.nn.softmax(comb_logits, axis=1) + hc_sinkhorn_eps
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=0, keepdims=True) + hc_sinkhorn_eps
        )
        for _ in range(sinkhorn_repeat - 1):
            comb_mix = comb_mix / (
                jnp.sum(comb_mix, axis=1, keepdims=True) + hc_sinkhorn_eps
            )
            comb_mix = comb_mix / (
                jnp.sum(comb_mix, axis=0, keepdims=True) + hc_sinkhorn_eps
            )
        pre_out[...] = pre_mix
        post_out[...] = post_mix
        comb_out[...] = comb_mix

    pre, post, comb = pl.pallas_call(
        gate_kernel,
        grid=(t // block,),
        in_specs=[
            pl.BlockSpec((mixes, block), lambda i: (0, i)),
            pl.BlockSpec((1, block), lambda i: (0, i)),
            pl.BlockSpec((3,), lambda i: (0,)),
            pl.BlockSpec((mixes,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((m, block), lambda i: (0, i)),
            pl.BlockSpec((m, block), lambda i: (0, i)),
            pl.BlockSpec((m, m, block), lambda i: (0, 0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape, jnp.float32)
            for shape in ((m, t), (m, t), (m, m, t))
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(mix, square, scale, base)
    return (pre.T, post.T, comb.transpose(2, 0, 1))
