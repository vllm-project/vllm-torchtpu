# SPDX-License-Identifier: Apache-2.0
"""Initial serial tiles requested for the prefill correctness baseline.

These are design inputs, not measured optimal tile sizes.
"""

import dataclasses

TOKEN_BLOCK = 256
HEAD_BLOCK = 8

# Preserve the baseline tile's output size with full 16-head BF16 rows.
QB_TOKEN_BLOCK = 128
QB_HEAD_BLOCK = 16
QB_QUANT_TOKEN_BLOCK = 64

QB_OUTPUT_BUFFERS = 2

# Strided-staging Q-B: 256 tokens per weight window restores the serial
# kernel's weight reuse; one 256-column slice of every head is staged at a time.
QB2_TOKEN_BLOCK = 256
QB2_HEAD_BLOCK = 16
QB2_COLUMN_BLOCK = 256
QB2_QUANT_TOKEN_BLOCK = 64
QB2_VMEM_LIMIT_BYTES = 56 * 1024 * 1024

# Double-buffered Q-A tiles: 2 x 2.5 MiB activations + 8.75 MiB FP8 weight.
COMPRESS_TOKEN_BLOCK = 256
# Rows of post-matmul work held in registers at once.
COMPRESS_FRAGMENT = 32
COMPRESS_VMEM_LIMIT_BYTES = 48 * 1024 * 1024

# Packed-pair Q-B: 256-token weight reuse, 16-head windows, 4 output word tiles
# (512 KiB each) in flight.
QB3_TOKEN_BLOCK = 256
QB3_HEAD_BLOCK = 16
QB3_QUANT_TOKEN_BLOCK = 64
QB3_OUTPUT_SLOTS = 4
QB3_VMEM_LIMIT_BYTES = 56 * 1024 * 1024

QA_TOKEN_BLOCK = 256
QA_VMEM_LIMIT_BYTES = 48 * 1024 * 1024

# Compressor: 256-token tiles; the BF16 [5120, 1024] weight stays resident.
CMP_TOKEN_BLOCK = 256
CMP_VMEM_LIMIT_BYTES = 56 * 1024 * 1024

# Overlapped Q-B: packed-pair layout; pair g-1 is post-processed while pair g's
# matmul issues. 16 output word slots (two steps of pairs, 8 MiB) in flight.
QB5_TOKEN_BLOCK = 256
QB5_HEAD_BLOCK = 16
QB5_QUANT_TOKEN_BLOCK = 64
QB5_VMEM_LIMIT_BYTES = 56 * 1024 * 1024

# Overlapped Q-B with larger token tiles: matmul-only ablation measured 124.7
# (256) / 105.2 (512) / 95.1 (1024) us, from fewer weight pushes per row.
# Output slots are recycled per head pair.
QB6_TOKEN_BLOCK = 512
QB6_HEAD_BLOCK = 16
QB6_QUANT_TOKEN_BLOCK = 64
QB6_OUTPUT_SLOTS = 8
QB6_VMEM_LIMIT_BYTES = 60 * 1024 * 1024


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
    re-validated against the epic chain.
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
