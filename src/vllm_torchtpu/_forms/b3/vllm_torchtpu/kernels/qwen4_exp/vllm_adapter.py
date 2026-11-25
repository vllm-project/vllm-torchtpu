# SPDX-License-Identifier: Apache-2.0
"""Torch custom-op boundaries for Qwen TPU kernels."""

import functools
import hashlib
import logging

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_cmp import hc_decode_mix_auto
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import logger as _hck_logger
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import rank_fields as _hck_rank_fields
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import (
    MODES as WIDE_MODES,
    hc_decode_mix_wide_auto,
)
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue

logger = logging.getLogger(__name__)


@functools.cache
def build_gated_residual_op(
    branches: int, hidden: int, token_tile: int, partial: bool = False
):
    op = pallas.jax_op(
        f"qwen4_exp::gated_residual_b{branches}_h{hidden}_m{token_tile}"
        f"{'_p' if partial else ''}_v1",
        gated_residual_fn(branches, hidden, token_tile, partial),
    )

    @op.register_fake
    def fake(residual, block, gates):
        return residual.new_empty(residual.shape)

    return op


@functools.cache
def build_full_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
):
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _full_mix(
            x,
            weight,
            down,
            up,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            partial=partial,
            norm_impl=norm_impl,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_full_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"{'_p' if partial else ''}{'' if norm_impl == 'vpu' else '_n' + norm_impl}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty(x.shape, dtype=torch.bfloat16),
        )

    return op


@functools.cache
def build_norm_op(branches: int, hidden: int, eps: float, token_tile: int):
    """Register a local Pallas op for complete prefill token tiles."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(input_tensor: jax.Array, weight: jax.Array) -> jax.Array:
        return _normalize(
            input_tensor,
            weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_norm_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}_v2",
        core,
    )

    @op.register_fake
    def fake(input_tensor, weight):
        return input_tensor.new_empty(input_tensor.shape)

    return op


@functools.cache
def build_up_epilogue_op(branches: int, hidden: int, token_tile: int):
    """Register a local JAX op; callers gate it to whole prefill tiles."""

    def core(mid: jax.Array, norm: jax.Array, weight: jax.Array) -> jax.Array:
        return _up_epilogue(
            mid, norm, weight, branches=branches, hidden=hidden, token_tile=token_tile
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_up_epilogue_c{branches}_h{hidden}_m{token_tile}_v1", core
    )

    @op.register_fake
    def fake(mid, norm, weight):
        return mid.new_empty((mid.shape[0], hidden), dtype=torch.bfloat16)

    return op


@functools.cache
def _log_w1_sched(tag: str) -> None:
    """q38 HCK runtime engagement line (once per tag per process = per TP rank, at decode HC op build: model
    construction, so it prints on a warm compile cache too). q38 HCK-2: plain info through the vLLM logger with the
    rank identity appended (per-rank engagement rule)."""
    msg = (
        "q38 HCK: TPU_QWEN4_HC_DECODE_W1_SCHED=1 engaged: hc_decode_wide mode 1 "
        f"schedule v2 (all copies at entry, one-pass B2), op tag {tag},"
    )
    _hck_logger.info(msg + _hck_rank_fields())


@functools.cache
def _log_wide_floor(tag: str, rows: int) -> None:
    """q38 HCK floor-key runtime engagement line (once per tag, at decode HC op build)."""
    msg = (
        f"q38 HCK: TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS={rows} engaged: hc_decode_wide mode 1 "
        f"from {rows} rows (multiples of 16, >= 2 32-row pieces per block), op tag {tag}"
    )
    logger.warning(msg)
    print(msg, flush=True)


def _decode_mix_choice():
    """The fused decode HC mix and its op-name tag.

    TPU_QWEN4_HC_DECODE_WIDE (q38 HC-HIGHROWS, 0 = off) 1 / 2 runs
    hc_decode_wide.py's served-order / cmp-math mode at 160-511 rows (a
    multiple of 16); TPU_QWEN4_HC_DECODE_COMPARE runs hc_decode_cmp.py at
    16-256 rows. They are exclusive. TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID
    (q38 HC-HIGHROWS-2 CMP-S, only with COMPARE) takes the served sigmoid in
    the cmp mix term (_cmps). Unset, it is hc_decode_mix, untagged.
    TPU_QWEN4_HC_DECODE_INJECT_FOLD (q38 INJ-MXU, numerics-changing) needs
    TPU_QWEN4_HC_DECODE_WIDE=1: W1 takes the injection logits from its down
    projection (_w1f); with WIDE 0, 2 or 3 it is refused.
    TPU_QWEN4_HC_DECODE_W1_SCHED=1 (q38 HCK) needs TPU_QWEN4_HC_DECODE_WIDE=1:
    the W1 kernel runs its schedule v2 (_w1s / _w1fs); refused otherwise.
    TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS (q38 HCK floor key) needs
    TPU_QWEN4_HC_DECODE_WIDE=1: mode 1 runs from that row count (tags gain
    l<rows>); refused otherwise.
    """
    wide = envs.TPU_QWEN4_HC_DECODE_WIDE
    fold = envs.TPU_QWEN4_HC_DECODE_INJECT_FOLD
    sched = envs.TPU_QWEN4_HC_DECODE_W1_SCHED == "1"
    floor = envs.TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS
    min_rows = int(floor) if floor else None
    if fold and wide != 1:
        raise ValueError(
            "TPU_QWEN4_HC_DECODE_INJECT_FOLD needs TPU_QWEN4_HC_DECODE_WIDE=1 "
            f"(got {wide})"
        )
    if sched and wide != 1:
        raise ValueError(
            "TPU_QWEN4_HC_DECODE_W1_SCHED=1 needs TPU_QWEN4_HC_DECODE_WIDE=1 "
            f"(got {wide})"
        )
    if min_rows is not None and wide != 1:
        raise ValueError(
            f"TPU_QWEN4_HC_DECODE_WIDE_MIN_ROWS={floor} needs TPU_QWEN4_HC_DECODE_WIDE=1 "
            f"(got {wide})"
        )
    if wide:
        if wide not in WIDE_MODES:
            raise ValueError(
                f"TPU_QWEN4_HC_DECODE_WIDE={wide}: expected 0 (off), 1, 2 or 3"
            )
        cmp_set = envs.TPU_QWEN4_HC_DECODE_COMPARE
        if cmp_set or envs.TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID:
            raise ValueError(
                "TPU_QWEN4_HC_DECODE_WIDE and TPU_QWEN4_HC_DECODE_COMPARE are "
                "exclusive"
            )
        fl = {} if min_rows is None else {"min_rows": min_rows}
        sfx = "" if min_rows is None else f"l{min_rows}"
        if sched:
            tag = ("_w1fs" if fold else "_w1s") + sfx
            _log_w1_sched(tag)
            if fl:
                _log_wide_floor(tag, min_rows)
            return (
                functools.partial(hc_decode_mix_wide_auto, mode=1, fold=fold, sched=True, **fl),
                tag,
            )
        tag = ("_w1f" if fold else f"_w{wide}") + sfx
        if fl:
            _log_wide_floor(tag, min_rows)
        if fold:
            return functools.partial(hc_decode_mix_wide_auto, mode=1, fold=True, **fl), tag
        return functools.partial(hc_decode_mix_wide_auto, mode=wide, **fl), tag
    if envs.TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID:
        if not envs.TPU_QWEN4_HC_DECODE_COMPARE:
            raise ValueError(
                "TPU_QWEN4_HC_DECODE_COMPARE_SIGMOID needs "
                "TPU_QWEN4_HC_DECODE_COMPARE=1"
            )
        return functools.partial(hc_decode_mix_auto, sigmoid=True), "_cmps"
    if envs.TPU_QWEN4_HC_DECODE_COMPARE:
        return hc_decode_mix_auto, "_cmp"
    return hc_decode_mix, ""


@functools.cache
def build_decode_mix_op(branches: int, hidden: int, eps: float, inject: bool):
    """Register the fused decode HC mix; callers gate it on decode_token_tile.

    Returns ``(mixed, injection)`` with ``inject`` (the combine's f32 gate,
    from the same normalized input), else ``mixed`` only.
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    max_tile = envs.TPU_QWEN4_HC_DECODE_TILE_ROWS
    static = dict(branches=branches, hidden=hidden, eps=eps, max_tile=max_tile)
    mix, cmp_tag = _decode_mix_choice()
    if inject:

        def core(
            x: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return mix(x, weight, down, up, inject_weight, **static)

    else:

        def core(
            x: jax.Array, weight: jax.Array, down: jax.Array, up: jax.Array
        ) -> jax.Array:
            return mix(x, weight, down, up, **static)[0]

    op = pallas.jax_op(
        f"qwen4_exp::hc_decode_c{branches}_h{hidden}_e{eps_tag}"
        f"{'_inj' if inject else ''}{f'_t{max_tile}' if max_tile else ''}"
        f"{cmp_tag}_v1",
        core,
    )

    def mixed_like(x):
        return x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16)

    if inject:

        @op.register_fake
        def fake(x, weight, down, up, inject_weight):
            gate = x.new_empty((x.shape[0], branches), dtype=torch.float32)
            return mixed_like(x), gate

    else:

        @op.register_fake
        def fake(x, weight, down, up):
            return mixed_like(x)

    return op


@functools.cache
def build_decode_combine_mix_op(
    branches: int, hidden: int, eps: float, inject: bool = True
):
    """The previous HC's combine fused ahead of this HC's decode mix.

    Returns ``(mixed, injection, combined)``, or ``(mixed, combined)`` without
    ``inject`` (the final mixer); ``combined`` is the hyper input this HC
    mixed (the next combine's residual).
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]
    max_tile = envs.TPU_QWEN4_HC_DECODE_TILE_ROWS
    static = dict(branches=branches, hidden=hidden, eps=eps, max_tile=max_tile)
    tile_tag = f"_t{max_tile}" if max_tile else ""
    mix, impl_tag = _decode_mix_choice()
    tile_tag += impl_tag
    if inject:

        def core(
            residual: jax.Array,
            block: jax.Array,
            prev_injection: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
            inject_weight: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return mix(
                residual,
                weight,
                down,
                up,
                inject_weight,
                block=block,
                prev_injection=prev_injection,
                **static,
            )

        name = (
            f"qwen4_exp::hc_decode_combine_c{branches}_h{hidden}_e{eps_tag}"
            f"{tile_tag}_v1"
        )
    else:

        def core(
            residual: jax.Array,
            block: jax.Array,
            prev_injection: jax.Array,
            weight: jax.Array,
            down: jax.Array,
            up: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            mixed, _, combined = mix(
                residual,
                weight,
                down,
                up,
                block=block,
                prev_injection=prev_injection,
                **static,
            )
            return mixed, combined

        name = (
            f"qwen4_exp::hc_decode_combine_c{branches}_h{hidden}_e{eps_tag}"
            f"_noinj{tile_tag}_v1"
        )

    op = pallas.jax_op(name, core)

    def mixed_like(residual):
        return residual.new_empty((residual.shape[0], hidden), dtype=torch.bfloat16)

    def combined_like(residual):
        return residual.new_empty(residual.shape, dtype=torch.bfloat16)

    if inject:

        @op.register_fake
        def fake(residual, block, prev_injection, weight, down, up, inject_weight):
            gate = residual.new_empty(
                (residual.shape[0], branches), dtype=torch.float32
            )
            return mixed_like(residual), gate, combined_like(residual)

    else:

        @op.register_fake
        def fake(residual, block, prev_injection, weight, down, up):
            return mixed_like(residual), combined_like(residual)

    return op


@functools.cache
def build_prefill_inj_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
    down_asm: bool = False,
):
    """TPU_QWEN4_HC_PREFILL_FUSED_MIX: prefill HC mix returning ``(mixed,
    injection)``; ``norm`` stays on chip. Callers gate it to whole tiles.
    ``down_asm``: TPU_QWEN4_HC_PREFILL_DOWN_ASM."""
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        x: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return hc_prefill_mix(
            x,
            weight,
            down,
            up,
            inject_weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            partial=partial,
            norm_impl=norm_impl,
            down_asm=down_asm,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_inj_c{branches}_h{hidden}_m{token_tile}_e{eps_tag}"
        f"{'_p' if partial else ''}{'' if norm_impl == 'vpu' else '_n' + norm_impl}"
        f"{'_da' if down_asm else ''}_v1",
        core,
    )

    @op.register_fake
    def fake(x, weight, down, up, inject_weight):
        return (
            x.new_empty((x.shape[0], hidden), dtype=torch.bfloat16),
            x.new_empty((x.shape[0], branches), dtype=torch.float32),
        )

    return op


@functools.cache
def build_prefill_combine_mix_op(
    branches: int,
    hidden: int,
    eps: float,
    token_tile: int,
    partial: bool = False,
    norm_impl: str = "vpu",
    reread: bool = False,
    down_asm: bool = False,
):
    """The previous HC's combine fused ahead of this HC's prefill mix.

    Returns ``(mixed, injection, combined)``; ``combined`` is the hyper input
    this HC mixed (the next combine's residual). ``reread``: the norm reads
    the combined rows back from VMEM (TPU_QWEN4_HC_PREFILL_FOLD); ``down_asm``:
    TPU_QWEN4_HC_PREFILL_DOWN_ASM.
    """
    eps_tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:8]

    def core(
        residual: jax.Array,
        block: jax.Array,
        prev_injection: jax.Array,
        weight: jax.Array,
        down: jax.Array,
        up: jax.Array,
        inject_weight: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return hc_prefill_mix(
            residual,
            weight,
            down,
            up,
            inject_weight,
            branches=branches,
            hidden=hidden,
            eps=eps,
            token_tile=token_tile,
            block=block,
            prev_injection=prev_injection,
            partial=partial,
            norm_impl=norm_impl,
            reread=reread,
            down_asm=down_asm,
        )

    op = pallas.jax_op(
        f"qwen4_exp::hc_prefill_combine_c{branches}_h{hidden}_m{token_tile}"
        f"_e{eps_tag}{'_p' if partial else ''}"
        f"{'' if norm_impl == 'vpu' else '_n' + norm_impl}"
        f"{'_rr' if reread else ''}{'_da' if down_asm else ''}_v1",
        core,
    )

    @op.register_fake
    def fake(residual, block, prev_injection, weight, down, up, inject_weight):
        tokens = residual.shape[0]
        return (
            residual.new_empty((tokens, hidden), dtype=torch.bfloat16),
            residual.new_empty((tokens, branches), dtype=torch.float32),
            residual.new_empty(residual.shape, dtype=torch.bfloat16),
        )

    return op
