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


def is_supported_shape(t: int, m: int, d: int) -> bool:
    return t > 0 and t % 16 == 0 and m == 4 and d == 5120


def _token_tile(t: int) -> int:
    """Tokens per pipelined tile: 32-token mix groups, else 16 (T % 32 != 0)."""
    for tile in (128, 96, 32, 80, 48, 16):
        if t % tile == 0:
            return tile
    raise ValueError(f"T={t} is not a multiple of 16")


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
    unpadded_gates: bool = False,
    early_prep: bool = False,
    fused_gates: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update residuals and collapse with incoming pre; predict gates separately.

    TPU7x: M=4, D=5120, positive local T divisible by 16 (by 32 uses 32-token
    mix groups; otherwise 16-token groups, e.g. T=240 in 80-token tiles).
    Activations are BF16; gates are FP32. Weight is packed BF16 [20480,128].
    Three-component mixing and BF16 residual/collapse rounding are retained;
    streamed square sums change the reference reduction order.

    Returns:
        (residual [T,M*D], next_post [T,M,1], next_comb [T,M,M],
        value [T,D], next_pre [T,M]). Value includes RMSNorm when
        norm_weight [D] is supplied. Other keyword arguments are JIT-static.
        ``unpadded_gates`` emits the gates T tokens wide instead of padded to a
        multiple of 128 (T <= 1024), so that between layers the [T,M] views
        are transposes of the next call's [M,T] gate inputs, not slices.
        ``early_prep`` builds tile 0's gate coefficients while the first
        activation tiles are still in flight instead of after they land.
        ``fused_gates`` computes the next gates inside the kernel instead of a
        second ``pallas_call``; with the next call in ``GateInputMode.HBM``
        they reach it through HBM, so they do not have to come back into VMEM
        ahead of it (bitwise identical to the separate gate kernel).
    """
    t, d = x.shape
    token_tile = _token_tile(t)
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
    assert is_supported_shape(t, m, d)
    gate_tokens = (t + 127) // 128 * 128
    gate_width = t if unpadded_gates and gate_tokens <= 1024 else gate_tokens
    tuning = PostPreTuning(
        token_tile, 32 if token_tile % 32 == 0 else 16, 256, 50 * 1024 * 1024
    )
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

    gate_constants = dict(
        hidden_size=d,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    def kernel(
        xh,
        rh,
        poh,
        ch,
        ph,
        wh,
        norm_refs,
        gate_param_refs,
        yh,
        vh,
        mh,
        sh,
        gate_outputs,
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
        gate_params,
        gate_sems,
        gate_wide,
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
        param_copies = []
        if fused_gates:
            param_copies = [
                pltpu.make_async_copy(src, dst, gate_sems.at[i])
                for i, (src, dst) in enumerate(zip(gate_param_refs, gate_params))
            ]
            for copy in param_copies:
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

            if not early_prep:

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
            if num_tiles > 1:
                # Unconditional on the last tile too: a branch here would stop
                # the scheduler overlapping it with the tile's MXU work.
                prepare(
                    jnp.minimum(tile_index + 1, num_tiles - 1),
                    (tile_index + 1) % 2,
                )

            store_vector(sbuf[...].T, sh, tile_index * tile)

            @pl.when(tile_index == 0)
            def wait_weight():
                copies[-1].wait()

            if early_output:
                # Behind the weight read: started earlier, these writes share
                # the DMA queue with it and delay the projection.
                output_copy(tile_index, 0).start()
                output_copy(tile_index, 1).start()
            projection_buf[...] = jax.lax.dot(
                yr[...], wr[...], preferred_element_type=jnp.float32
            )
            store_projection(previous_projected, jnp.maximum(tile_index - 1, 0))

        for index in range(input_buffers):
            for copy in input_copies(index):
                copy.start()
            if index == 0:
                copies[-1].start()
        if early_prep:
            for copy in copies[:-1]:
                copy.wait()
            prepare(0, 0)

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

        if fused_gates:
            # Lane-broadcast the gate parameters ahead of the tiles, not on the
            # epilogue's critical path.
            for copy in param_copies:
                copy.wait()
            for param, wide in zip(gate_params, gate_wide, strict=True):
                wide[...] = jnp.broadcast_to(param[...][:, None], wide.shape)
        jax.lax.fori_loop(0, num_tiles, body, None)
        store_projection(projection_buf[...], num_tiles - 1)
        if not fused_gates:
            for index in range(num_tiles - output_buffers, num_tiles):
                for copy in output_copies(index):
                    copy.wait()
            return
        block = _gate_block(gate_tokens)
        scale_value, base_value = (
            jnp.concatenate([wide[...]] * (block // 128), axis=1) for wide in gate_wide
        )
        for start in range(0, gate_tokens, block):
            values = _gate_values(
                mh[:, pl.ds(start, block)],
                sh[:, pl.ds(start, block)],
                scale_value,
                base_value,
                **gate_constants,
            )
            for output, value in zip(gate_outputs, values, strict=True):
                output[..., pl.ds(start, min(block, gate_width))] = value[
                    ..., : min(block, gate_width)
                ]
        for index in range(num_tiles - output_buffers, num_tiles):
            for copy in output_copies(index):
                copy.wait()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
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
        gate_spec = vmem
        gate_scratch_shapes = ()
    gate_shapes = ((m, gate_width), (m, gate_width), (m, m, gate_width))
    if fused_gates:
        # Logits and square sums stay in scratch. Gates leave in VMEM: XLA
        # evicts them to HBM behind the next ops, and the next seam DMAs them in.
        param_args = tuple(
            pltpu.with_memory_space_constraint(arg, pltpu.HBM) for arg in (scale, base)
        )
        logit_outputs = []
        logit_scratch = [pltpu.VMEM((k, gate_tokens), jnp.float32) for k in (mixes, 1)]
        gate_out_specs = [vmem] * 3
        gate_out_shapes = [
            jax.ShapeDtypeStruct(shape, jnp.float32) for shape in gate_shapes
        ]
        gate_extra_scratch = [
            tuple(pltpu.VMEM(arg.shape, arg.dtype) for arg in param_args),
            pltpu.SemaphoreType.DMA((len(param_args),)),
            tuple(pltpu.VMEM((arg.shape[0], 128), arg.dtype) for arg in param_args),
        ]
    else:
        param_args = ()
        logit_outputs = [
            jax.ShapeDtypeStruct((k, gate_tokens), jnp.float32) for k in (mixes, 1)
        ]
        logit_scratch = []
        gate_out_specs, gate_out_shapes = [], []
        gate_extra_scratch = [(), (), ()]

    def wrapped(*refs):
        # Logits leave as outputs for the separate gate kernel, else stay in scratch.
        inputs, rest = refs[:8], refs[8:]
        if fused_gates:
            yh, vh, *gate_outputs = rest[:5]
            *scratch, mh, sh = rest[5:]
        else:
            yh, vh, mh, sh, *scratch = rest
            gate_outputs = ()
        kernel(*inputs, yh, vh, mh, sh, tuple(gate_outputs), *scratch)

    outputs = pl.pallas_call(
        wrapped,
        in_specs=[hbm, hbm, gate_spec, gate_spec, gate_spec, hbm]
        + [tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in norm_args)]
        + [tuple(hbm for _ in param_args)],
        out_specs=[hbm, hbm] + [vmem] * len(logit_outputs) + gate_out_specs,
        out_shape=[jax.ShapeDtypeStruct((t, k), flat.dtype) for k in (width, d)]
        + logit_outputs
        + gate_out_shapes,
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
            *gate_extra_scratch,
            *logit_scratch,
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tuning.vmem_limit_bytes,
            allow_input_fusion=(False, False, False, False, False, False)
            + (False,) * len(norm_args)
            + (False,) * len(param_args),
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
        param_args,
    )
    residual, value = outputs[:2]
    if fused_gates:
        new_pre, new_post, new_comb = (
            gate.T if gate.ndim == 2 else gate.transpose(2, 0, 1)
            for gate in outputs[2:]
        )
    else:
        mix_values, square = outputs[2:]
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
            tokens=t if gate_width == t else None,
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
    tokens: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FP32 gates from logits [M*(M+2),T] and square sums [1,T]; hidden_size=D.

    ``tokens`` (single-block T only) keeps the first ``tokens`` gate columns.
    """
    mixes, t = mix.shape
    m = math.isqrt(mixes + 1) - 1
    assert m > 0 and mixes == m * (m + 2), "Expected M*(M+2) mix logits"
    block = _gate_block(t)
    assert hidden_size > 0, "hidden_size must be positive"
    assert square.shape == (1, t) and scale.shape == (3,) and (base.shape == (mixes,))
    assert mix.dtype == square.dtype == scale.dtype == base.dtype == jnp.float32
    if not isinstance(sinkhorn_repeat, int) or sinkhorn_repeat < 1:
        raise ValueError("sinkhorn_repeat must be a positive static integer")
    width = t if tokens is None else tokens
    assert width == t or (block == t and 0 < width < t)

    def gate_kernel(mr, sr, scr, br, pre_out, post_out, comb_out):
        pre_mix, post_mix, comb_mix = _gate_values(
            mr[...],
            sr[...],
            scr[...],
            br[...],
            hidden_size=hidden_size,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        pre_out[...] = pre_mix[..., :width]
        post_out[...] = post_mix[..., :width]
        comb_out[...] = comb_mix[..., :width]

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
            pl.BlockSpec((m, min(block, width)), lambda i: (0, i)),
            pl.BlockSpec((m, min(block, width)), lambda i: (0, i)),
            pl.BlockSpec((m, m, min(block, width)), lambda i: (0, 0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape, jnp.float32)
            for shape in ((m, width), (m, width), (m, m, width))
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(mix, square, scale, base)
    return (pre.T, post.T, comb.transpose(2, 0, 1))


def _gate_block(t: int) -> int:
    """Gate columns computed together: all of T up to 1024, else a divisor."""
    block = t if t <= 1024 else math.gcd(t, 1024)
    assert t > 0 and block % 128 == 0 and (t % block == 0), (
        "Gate token count must be a positive multiple of 128"
    )
    return block


def _gate_values(
    mix: jax.Array,
    square: jax.Array,
    scale: jax.Array,
    base: jax.Array,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """FP32 (pre [M,N], post [M,N], comb [M,M,N]) from logits [M*(M+2),N]."""
    mixes, block = mix.shape
    m = math.isqrt(mixes + 1) - 1
    mixed = mix * jax.lax.rsqrt(square / (m * hidden_size) + rms_eps)
    # Parameters as [k] or already lane-broadcast to [k, N].
    scales = scale if scale.ndim == 2 else scale[:, None]
    bias = base if base.ndim == 2 else base[:, None]
    if block == 128 and m == 4:
        return _gate_values_by_entry(
            mixed,
            scales,
            bias,
            m,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    pre_logits = mixed[:m, :] * scales[:1, :] + bias[:m, :]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps
    post_logits = mixed[m : 2 * m, :] * scales[1:2, :] + bias[m : 2 * m, :]
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value
    comb_logits = (mixed[2 * m :, :] * scales[2:3, :] + bias[2 * m :, :]).reshape(
        m, m, block
    )
    comb_mix = jax.nn.softmax(comb_logits, axis=1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=0, keepdims=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=0, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


def _gate_values_by_entry(mixed, scales, bias, m, pre_eps, eps, post_mult, repeat):
    """`_gate_values` for one 128-column block and M=4, each comb entry in its
    own vreg: the Sinkhorn sums become vector adds instead of sublane rotations.
    They keep the association Mosaic gives the [M,M,N] reductions (pairs (0, 2),
    (1, 3) over the minor axis, in order over the major one), so the gates are
    bitwise those of the [M,M,N] form; that is lowering behaviour, which
    test_one_gate_block_matches_the_wide_form_bitwise pins.
    """
    block = mixed.shape[1]
    pre_mix = jax.nn.sigmoid(mixed[:m, :] * scales[:1, :] + bias[:m, :]) + pre_eps
    post_logits = mixed[m : 2 * m, :] * scales[1:2, :] + bias[m : 2 * m, :]
    post_mix = jax.nn.sigmoid(post_logits) * post_mult
    logits = mixed[2 * m :, :] * scales[2:3, :] + bias[2 * m :, :]
    rows = 8  # one FP32 sublane tile, so every entry sits at the same offset
    c = [
        [
            jnp.broadcast_to(logits[i * m + j : i * m + j + 1, :], (rows, block))
            for j in range(m)
        ]
        for i in range(m)
    ]

    def minor_sum(v):
        return (v[0] + v[2]) + (v[1] + v[3])

    def major_sum(v):
        return functools.reduce(jnp.add, v)

    for i in range(m):
        top = functools.reduce(jnp.maximum, c[i])
        e = [jnp.exp(v - top) for v in c[i]]
        total = minor_sum(e)
        c[i] = [v / total + eps for v in e]

    def normalize_major(c):
        denom = [major_sum([c[i][j] for i in range(m)]) + eps for j in range(m)]
        return [[c[i][j] / denom[j] for j in range(m)] for i in range(m)]

    def normalize_minor(c):
        denom = [minor_sum(row) + eps for row in c]
        return [[v / denom[i] for v in row] for i, row in enumerate(c)]

    c = normalize_major(c)
    for _ in range(repeat - 1):
        c = normalize_major(normalize_minor(c))
    comb_mix = jnp.stack([jnp.concatenate([v[:1] for v in row]) for row in c])
    return pre_mix, post_mix, comb_mix
