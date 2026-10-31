# SPDX-License-Identifier: Apache-2.0
"""Validated tile configurations and measured TPU7x tuning entries."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class QBRopeConfig:
    """Block sizes of ``qb_rope.fused_qb_rope``; see its validity rules.

    ``segment_tokens=None`` lets the wrapper pick the largest token segment
    whose resident VMEM fits ``vmem_limit_bytes``.
    """

    token_block: int
    head_block: int = 16
    output_slots: int = 8
    quant_fragment: int = 64
    post_fragment: int = 32
    segment_tokens: int | None = None
    vmem_limit_bytes: int = 60 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class QARopeConfig:
    """Block sizes of ``qa_rope.fused_qa_kv_norm_rope``.

    ``weight_chunk`` columns of the resident FP8 weight share one DMA
    semaphore; the first token tile waits for and multiplies each chunk in
    turn. Like ``CompressConfig.weight_chunk`` it is the first tile's matmul
    N, so re-validate bitwise parity before changing the default.
    """

    token_block: int
    weight_chunk: int = 256
    vmem_limit_bytes: int = 48 * 1024 * 1024


# (tpu_version, tokens, hidden, projected width) -> measured best config; width
# 512 is the kv-only (emit_qr=False) mode. Unlisted shapes use
# qa_rope.default_config. GAP sweep (TPU7x, P-state 7), each entry bitwise
# equal to the default config: T=2048 36.71 -> 34.82 us, T=288 9.16 -> 7.95,
# kv-only T=288 5.20 -> 4.86, T=240 8.41 -> 7.96. weight_chunk=896 is NOT
# bitwise with token blocks below 288; keep it paired with these tiles.
QA_ROPE_TUNING = {
    (7, 2048, 5120, 1792): QARopeConfig(token_block=512, weight_chunk=896),
    (7, 288, 5120, 1792): QARopeConfig(token_block=288, weight_chunk=896),
    (7, 288, 5120, 512): QARopeConfig(token_block=288),
    (7, 240, 5120, 1792): QARopeConfig(token_block=240, weight_chunk=1792),
}


@dataclasses.dataclass(frozen=True)
class CompressConfig:
    """Block sizes of ``compress_rope.fused_compress_rope``.

    ``weight_chunk`` columns of the resident BF16 weight share one DMA
    semaphore; the first token tile waits for each chunk before its matmul.
    The chunk is also the matmul's N, which changes FP32 accumulation on
    TPU7x (128 vs 256 measured not bitwise), so keep the default 256 unless
    re-validated against the unfused model chain.
    """

    token_block: int
    weight_chunk: int = 256
    vmem_limit_bytes: int = 48 * 1024 * 1024


# (tpu_version, tokens, hidden, ratio) -> measured best config. Unlisted shapes
# use compress_rope.default_config. GAP sweep, bitwise equal to the default:
# ratio 2 T=2048 36.85 -> 34.29 us, T=288 20.78 -> 19.18, T=240 18.50 -> 16.84.
# Ratio 1 is fastest with the default 256-token tiles (21.17 us at T=2048).
COMPRESS_ROPE_TUNING = {
    (7, 2048, 5120, 2): CompressConfig(token_block=512, weight_chunk=512),
    (7, 288, 5120, 2): CompressConfig(token_block=288, weight_chunk=512),
    (7, 240, 5120, 2): CompressConfig(token_block=240, weight_chunk=512),
    (7, 2048, 5120, 1): CompressConfig(token_block=256),
}

# (tpu_version, tokens, rank, width). T=2048: 512-token tiles measured 124.79
# us; 1024 spilled (143.9 us); output slots do not matter. GAP sweep, bitwise
# equal to the default: T=288 24.37 -> 23.77 us with 8-head windows (chain:
# 26.20). T=96 8-head windows did not reproduce a gain, so T=96 keeps the default.
QB_ROPE_TUNING = {
    (7, 2048, 1280, 32768): QBRopeConfig(token_block=512),
    # quant_fragment must divide token_block: 64 does not divide 288.
    (7, 288, 1280, 32768): QBRopeConfig(
        token_block=288, head_block=8, output_slots=4, quant_fragment=32
    ),
}
