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
"""Standalone tests for forward Splash attention and segment tile refinement."""

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_kernel,
    splash_attention_mask,
    splash_attention_mask_info,
)

_refine_mask_info_with_segments = vars(splash_attention_kernel)[
    "_refine_mask_info_with_segments"
]

BLOCK_SIZE = 128
SEQ_LEN_SMALL = 256
SEQ_LEN_MEDIUM = 512
SEQ_LEN_LARGE = 2048
HEAD_DIM = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS_MHA = 2
NUM_Q_HEADS_LARGE = 2
NUM_KV_HEADS_LARGE = 1

BLOCK_PARTIAL = 1
BLOCK_FULL = 2
BLOCK_SEGMENT_ONLY = 3

INT8_MAX_SIGNED = int(np.iinfo(np.int8).max)
MIN_UNIQUE_PARTIAL_BLOCKS_FOR_INT16 = INT8_MAX_SIGNED + 2
UNIQUE_STATIC_PATTERN_MOD = BLOCK_SIZE - 1

SMALL_GRID_DIM = SEQ_LEN_SMALL // BLOCK_SIZE
MEDIUM_GRID_DIM = SEQ_LEN_MEDIUM // BLOCK_SIZE
LARGE_GRID_DIM = SEQ_LEN_LARGE // BLOCK_SIZE
NUM_UNIQUE_BLOCKS_2048 = LARGE_GRID_DIM * LARGE_GRID_DIM
CHECKERBOARD_MOD = 2

SEGMENT_SPLIT_HALF = 128
SEGMENT_SPLIT_STRADDLE = 64
SEGMENT_SPLIT_MEDIUM = 192
SEGMENT_SPLIT_LARGE_MAIN = 1024
SEGMENT_SPLIT_LARGE_Q = 1088
SEGMENT_SPLIT_LARGE_TAIL = 960
SEGMENT_SPLIT_CAUSAL_512_HEAD = 192
SEGMENT_SPLIT_CAUSAL_512_TAIL = 320

POSITIVE_CAUSAL_OFFSET = 128
INVALID_CAUSAL_OFFSET = 64
DIAG_SKIP_GRID = 2

BF16_EPS = float(jnp.finfo(jnp.bfloat16).eps)
MAX_ABS_EPS_FACTOR = 4.0
RMSE_EPS_FACTOR = 1.0
LSE_EPS_FACTOR = 3.0


@dataclasses.dataclass(frozen=True, slots=True)
class DiagSkipMode:
    """Boolean flag pair selecting diagonal skip paths."""

    qk_diag_skip: bool
    sv_diag_skip: bool


def _require_tpu() -> None:
    """Fail immediately unless JAX is initialized with a TPU backend."""
    try:
        backend = jax.default_backend()
    except (RuntimeError, ValueError) as exc:
        pytest.fail(f"JAX TPU backend failed to initialize: {exc}")
    if backend != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {backend}.")


def _make_segment_ids_array(lengths: tuple[int, ...]) -> np.ndarray:
    """Build contiguous integer segment labels matching `lengths`."""
    parts = [
        np.full((length,), segment_idx, dtype=np.int32)
        for segment_idx, length in enumerate(lengths)
    ]
    return np.concatenate(parts, axis=0)


def _make_segment_ids(lengths: tuple[int, ...]) -> splash_attention_kernel.SegmentIds:
    """Construct `SegmentIds` with identical Q and KV segment lengths."""
    seg = jnp.asarray(_make_segment_ids_array(lengths), dtype=jnp.int32)
    return splash_attention_kernel.SegmentIds(q=seg, kv=seg)


def _active_tile_block_mask(
    mask_info: splash_attention_mask_info.MaskInfo,
) -> dict[tuple[int, int], int]:
    """Return `{ (row, col): block_mask }` for valid active tiles."""
    assert mask_info.active_rows is not None
    assert mask_info.active_cols is not None
    assert mask_info.block_mask is not None
    assert mask_info.num_active_blocks is not None
    count = int(jax.device_get(mask_info.num_active_blocks[0]))
    rows = np.asarray(jax.device_get(mask_info.active_rows[:count]), dtype=np.int64)
    cols = np.asarray(jax.device_get(mask_info.active_cols[:count]), dtype=np.int64)
    codes = np.asarray(jax.device_get(mask_info.block_mask[:count]), dtype=np.int64)
    return {
        (int(row), int(col)): int(code)
        for row, col, code in zip(rows, cols, codes, strict=True)
    }


def _active_tile_mask_next(
    mask_info: splash_attention_mask_info.MaskInfo,
) -> dict[tuple[int, int], int]:
    """Return `{ (row, col): mask_next }` for valid active tiles."""
    assert mask_info.active_rows is not None
    assert mask_info.active_cols is not None
    assert mask_info.mask_next is not None
    assert mask_info.num_active_blocks is not None
    count = int(jax.device_get(mask_info.num_active_blocks[0]))
    rows = np.asarray(jax.device_get(mask_info.active_rows[:count]), dtype=np.int64)
    cols = np.asarray(jax.device_get(mask_info.active_cols[:count]), dtype=np.int64)
    next_ids = np.asarray(jax.device_get(mask_info.mask_next[:count]), dtype=np.int64)
    return {
        (int(row), int(col)): int(next_id)
        for row, col, next_id in zip(rows, cols, next_ids, strict=True)
    }


def _make_unique_partial_mask(grid_dim: int, block_size: int) -> np.ndarray:
    """Create a 2D mask where every tile is partial with a unique bit pattern."""
    total_blocks = grid_dim * grid_dim
    assert total_blocks >= MIN_UNIQUE_PARTIAL_BLOCKS_FOR_INT16
    blocks = np.zeros((grid_dim, grid_dim, block_size, block_size), dtype=np.bool_)
    blocks[:, :, :, 0] = True
    for block_idx in range(total_blocks):
        row_block = block_idx // grid_dim
        col_block = block_idx % grid_dim
        inner_row = block_idx // UNIQUE_STATIC_PATTERN_MOD
        inner_col = 1 + (block_idx % UNIQUE_STATIC_PATTERN_MOD)
        blocks[row_block, col_block, inner_row, inner_col] = True
    seq_len = grid_dim * block_size
    return blocks.swapaxes(1, 2).reshape(seq_len, seq_len)


def _make_non_causal_mask_array(seq_len: int) -> np.ndarray:
    """Build a 2D mask mixing full tiles with a partial checkerboard quadrant."""
    mask = np.ones((seq_len, seq_len), dtype=np.bool_)
    half = seq_len // 2
    rows = np.arange(half, dtype=np.int32)[:, None]
    cols = np.arange(half, dtype=np.int32)[None, :]
    mask[:half, half:] = ((rows + cols) % CHECKERBOARD_MOD) == 0
    return mask


def _compute_fp64_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    segment_ids_np: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference attention output and natural logsumexp in float64."""
    q64 = np.asarray(q, dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    v64 = np.asarray(v, dtype=np.float64)
    mqa_kv_rank = 2  # MQA K/V has shape [seq, head_dim].
    if k64.ndim == mqa_kv_rank:
        k64 = np.broadcast_to(k64[None, :, :], (q64.shape[0], *k64.shape))
        v64 = np.broadcast_to(v64[None, :, :], (q64.shape[0], *v64.shape))
    elif k64.shape[0] != q64.shape[0]:
        repeats = q64.shape[0] // k64.shape[0]
        k64 = np.repeat(k64, repeats, axis=0)
        v64 = np.repeat(v64, repeats, axis=0)

    effective_mask = mask.copy()
    if segment_ids_np is not None:
        effective_mask = effective_mask & (
            segment_ids_np[:, None] == segment_ids_np[None, :]
        )
    logits = np.einsum("hqd,hkd->hqk", q64, k64)
    logits = np.where(effective_mask[None, :, :], logits, -np.inf)
    row_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - row_max)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp
    out = np.einsum("hqk,hkd->hqd", probs, v64)
    logsumexp = row_max[..., 0] + np.log(sum_exp[..., 0])
    return out, logsumexp


def _assert_attention_close(
    out: jax.Array,
    ref: np.ndarray,
    *,
    max_abs_factor: float = MAX_ABS_EPS_FACTOR,
    rmse_factor: float = RMSE_EPS_FACTOR,
) -> tuple[float, float]:
    """Assert finiteness plus max-abs and RMSE bounds against an FP64 reference."""
    out64 = np.asarray(jax.device_get(out), dtype=np.float64)
    assert out64.shape == ref.shape
    assert np.all(np.isfinite(out64))
    diff = np.abs(out64 - ref)
    max_abs = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    ref_scale = max(1.0, float(np.max(np.abs(ref))))
    max_abs_tol = max_abs_factor * BF16_EPS * ref_scale
    rmse_tol = rmse_factor * BF16_EPS * ref_scale
    assert max_abs <= max_abs_tol
    assert rmse <= rmse_tol
    return max_abs, rmse


def _make_qkv(
    rng: np.random.Generator,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate bfloat16 Q, K, and V inputs for MHA or MQA attention."""
    scale = float(HEAD_DIM**-0.5)
    q = jnp.asarray(
        rng.standard_normal((num_q_heads, seq_len, HEAD_DIM)) * scale,
        dtype=jnp.bfloat16,
    )
    kv_shape = (
        (seq_len, HEAD_DIM)
        if num_kv_heads is None
        else (num_kv_heads, seq_len, HEAD_DIM)
    )
    k = jnp.asarray(rng.standard_normal(kv_shape), dtype=jnp.bfloat16)
    v = jnp.asarray(rng.standard_normal(kv_shape), dtype=jnp.bfloat16)
    return q, k, v


def _build_parity_kernel(
    mask_kind: Literal["causal", "non_causal"],
    attention_kind: Literal["mha", "mqa"],
    grid_kind: Literal["static", "dynamic"],
    mask_np: np.ndarray,
) -> splash_attention_kernel.SplashAttentionKernel:
    """Instantiate a static or dynamic MHA/MQA Splash kernel for parity checks."""
    config = splash_attention_kernel.SplashConfig(
        block_q=BLOCK_SIZE,
        block_kv=BLOCK_SIZE,
        block_kv_compute=BLOCK_SIZE,
    )
    if grid_kind == "static":
        mask_obj: splash_attention_mask.Mask = (
            splash_attention_mask.CausalMask((SEQ_LEN_SMALL, SEQ_LEN_SMALL))
            if mask_kind == "causal"
            else splash_attention_mask.NumpyMask(mask_np)
        )
        maker = (
            splash_attention_kernel.make_splash_mha_single_device
            if attention_kind == "mha"
            else splash_attention_kernel.make_splash_mqa_single_device
        )
        return maker(mask_obj, config=config)

    dynamic_maker = (
        splash_attention_kernel.make_dynamic_splash_mha
        if attention_kind == "mha"
        else splash_attention_kernel.make_dynamic_splash_mqa
    )
    kernel = dynamic_maker(jnp.asarray(mask_np), config=config)
    assert isinstance(kernel, splash_attention_kernel.SplashAttentionKernel)
    return kernel


def _run_forward_parity_case(
    mask_kind: Literal["causal", "non_causal"],
    attention_kind: Literal["mha", "mqa"],
    grid_kind: Literal["static", "dynamic"],
    segment_case: Literal["none", "split_128_128", "split_64_192"],
) -> tuple[float, float]:
    """Execute one 256x256 forward parity check and segmented negative control."""
    rng = np.random.default_rng(17)
    num_kv_heads = NUM_KV_HEADS_MHA if attention_kind == "mha" else None
    q, k, v = _make_qkv(rng, SEQ_LEN_SMALL, NUM_Q_HEADS, num_kv_heads)
    mask_np = (
        np.tril(np.ones((SEQ_LEN_SMALL, SEQ_LEN_SMALL), dtype=np.bool_))
        if mask_kind == "causal"
        else _make_non_causal_mask_array(SEQ_LEN_SMALL)
    )
    kernel = _build_parity_kernel(mask_kind, attention_kind, grid_kind, mask_np)

    seg_lengths_map: dict[str, tuple[int, ...] | None] = {
        "none": None,
        "split_128_128": (SEGMENT_SPLIT_HALF, SEGMENT_SPLIT_HALF),
        "split_64_192": (SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM),
    }
    seg_lengths = seg_lengths_map[segment_case]
    seg_np = None if seg_lengths is None else _make_segment_ids_array(seg_lengths)
    seg_ids = (
        None
        if seg_np is None
        else splash_attention_kernel.SegmentIds(
            q=jnp.asarray(seg_np), kv=jnp.asarray(seg_np)
        )
    )

    out = kernel(q, k, v, seg_ids)
    assert isinstance(out, jax.Array)
    ref, _ = _compute_fp64_attention(
        np.asarray(jax.device_get(q)),
        np.asarray(jax.device_get(k)),
        np.asarray(jax.device_get(v)),
        mask_np,
        seg_np,
    )
    metrics = _assert_attention_close(out, ref)

    if seg_np is not None:
        perm_seg_np = seg_np[rng.permutation(SEQ_LEN_SMALL)]
        perm_seg = splash_attention_kernel.SegmentIds(
            q=jnp.asarray(perm_seg_np), kv=jnp.asarray(perm_seg_np)
        )
        perm_out = kernel(q, k, v, perm_seg)
        assert isinstance(perm_out, jax.Array)
        with pytest.raises(AssertionError):
            _assert_attention_close(perm_out, ref)
    return metrics


def _run_diag_skip_case(
    diag_skip: DiagSkipMode,
) -> tuple[float, float]:
    """Execute one causal diagonal-skip check with straddling `(64, 192)` segments."""
    rng = np.random.default_rng(29)
    q, k, v = _make_qkv(rng, SEQ_LEN_SMALL, NUM_Q_HEADS, NUM_KV_HEADS_MHA)
    mask_np = np.tril(np.ones((SEQ_LEN_SMALL, SEQ_LEN_SMALL), dtype=np.bool_))
    seg_np = _make_segment_ids_array((SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM))
    seg_ids = splash_attention_kernel.SegmentIds(
        q=jnp.asarray(seg_np), kv=jnp.asarray(seg_np)
    )
    config = splash_attention_kernel.SplashConfig(
        block_q=BLOCK_SIZE,
        block_kv=BLOCK_SIZE,
        block_kv_compute=BLOCK_SIZE,
        qk_diag_skip=diag_skip.qk_diag_skip,
        sv_diag_skip=diag_skip.sv_diag_skip,
        qk_diag_grid=DIAG_SKIP_GRID,
    )
    kernel = splash_attention_kernel.make_splash_mha_single_device(
        splash_attention_mask.CausalMask((SEQ_LEN_SMALL, SEQ_LEN_SMALL)),
        config=config,
    )
    out = kernel(q, k, v, seg_ids)
    assert isinstance(out, jax.Array)
    ref, _ = _compute_fp64_attention(
        np.asarray(jax.device_get(q)),
        np.asarray(jax.device_get(k)),
        np.asarray(jax.device_get(v)),
        mask_np,
        seg_np,
    )
    return _assert_attention_close(out, ref)


def _run_dynamic_over_127_case() -> tuple[float, float]:
    """Execute end-to-end 2048x2048 dynamic attention past block index 127."""
    rng = np.random.default_rng(53)
    q, k, v = _make_qkv(rng, SEQ_LEN_LARGE, NUM_Q_HEADS_LARGE, NUM_KV_HEADS_LARGE)
    mask_np = np.tril(np.ones((SEQ_LEN_LARGE, SEQ_LEN_LARGE), dtype=np.bool_))
    seg_np = _make_segment_ids_array((SEGMENT_SPLIT_LARGE_Q, SEGMENT_SPLIT_LARGE_TAIL))
    seg_ids = splash_attention_kernel.SegmentIds(
        q=jnp.asarray(seg_np), kv=jnp.asarray(seg_np)
    )
    config = splash_attention_kernel.SplashConfig(
        block_q=BLOCK_SIZE,
        block_kv=BLOCK_SIZE,
        block_kv_compute=BLOCK_SIZE,
    )
    kernel = splash_attention_kernel.make_dynamic_splash_mha(
        jnp.asarray(mask_np), config=config
    )
    assert isinstance(kernel, splash_attention_kernel.SplashAttentionKernel)
    out = kernel(q, k, v, seg_ids)
    assert isinstance(out, jax.Array)
    ref, _ = _compute_fp64_attention(
        np.asarray(jax.device_get(q)),
        np.asarray(jax.device_get(k)),
        np.asarray(jax.device_get(v)),
        mask_np,
        seg_np,
    )
    return _assert_attention_close(out, ref)


def _run_base2_natural_logsumexp_case(
    reciprocal_mode: Literal["fused", "unfused"],
) -> tuple[float, float]:
    """Verify `use_base2_exp` + `q_prescaled_log2e` returns natural-log LSE."""
    rng = np.random.default_rng(71)
    q, k, v = _make_qkv(rng, SEQ_LEN_SMALL, NUM_Q_HEADS, NUM_KV_HEADS_MHA)
    q_prescaled = jnp.asarray(
        np.asarray(jax.device_get(q), dtype=np.float32) * splash_attention_kernel.LOG2E,
        dtype=jnp.bfloat16,
    )
    mask_np = np.tril(np.ones((SEQ_LEN_SMALL, SEQ_LEN_SMALL), dtype=np.bool_))
    seg_np = _make_segment_ids_array((SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM))
    seg_ids = splash_attention_kernel.SegmentIds(
        q=jnp.asarray(seg_np), kv=jnp.asarray(seg_np)
    )
    config = splash_attention_kernel.SplashConfig(
        block_q=BLOCK_SIZE,
        block_kv=BLOCK_SIZE,
        block_kv_compute=BLOCK_SIZE,
        use_base2_exp=True,
        q_prescaled_log2e=True,
        fuse_reciprocal=reciprocal_mode == "fused",
    )
    kernel = splash_attention_kernel.make_splash_mha_single_device(
        splash_attention_mask.CausalMask((SEQ_LEN_SMALL, SEQ_LEN_SMALL)),
        config=config,
        save_residuals=True,
    )
    out, stats = kernel(q_prescaled, k, v, seg_ids)
    ref_out, ref_lse = _compute_fp64_attention(
        np.asarray(jax.device_get(q)),
        np.asarray(jax.device_get(k)),
        np.asarray(jax.device_get(v)),
        mask_np,
        seg_np,
    )
    out_max, _ = _assert_attention_close(out, ref_out)

    lse64 = np.asarray(jax.device_get(stats["logsumexp"]), dtype=np.float64)
    assert lse64.shape == (NUM_Q_HEADS, SEQ_LEN_SMALL)
    assert np.all(np.isfinite(lse64))
    lse_max = float(np.max(np.abs(lse64 - ref_lse)))
    lse_scale = max(1.0, float(np.max(np.abs(ref_lse))))
    lse_tol = LSE_EPS_FACTOR * BF16_EPS * lse_scale
    assert lse_max <= lse_tol
    return out_max, lse_max


def test_refine_mask_info_prunes_and_promotes_segments() -> None:
    """Prune disjoint tiles and promote uniform tiles across mask types."""
    causal_info, _ = splash_attention_mask_info.process_mask(
        splash_attention_mask.CausalMask((SEQ_LEN_MEDIUM, SEQ_LEN_MEDIUM)),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    assert causal_info.num_active_blocks is not None
    causal_base_active = MEDIUM_GRID_DIM * (MEDIUM_GRID_DIM + 1) // 2
    assert int(causal_info.num_active_blocks[0]) == causal_base_active
    causal_refined, _, _ = _refine_mask_info_with_segments(
        causal_info,
        _make_segment_ids(
            (SEGMENT_SPLIT_CAUSAL_512_HEAD, SEGMENT_SPLIT_CAUSAL_512_TAIL)
        ),
        q_blocks=MEDIUM_GRID_DIM,
        kv_blocks=MEDIUM_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert causal_refined.num_active_blocks is not None
    # Segment split prunes the 2 lower-left tiles (2, 0) and (3, 0).
    assert (
        int(jax.device_get(causal_refined.num_active_blocks[0]))
        == causal_base_active - 2
    )
    causal_tiles = _active_tile_block_mask(causal_refined)
    assert (2, 0) not in causal_tiles
    assert (3, 0) not in causal_tiles
    assert causal_tiles[(1, 0)] == BLOCK_PARTIAL
    assert causal_tiles[(3, 1)] == BLOCK_PARTIAL
    assert causal_tiles[(3, 2)] == BLOCK_FULL

    full_info, _ = splash_attention_mask_info.process_mask(
        splash_attention_mask.FullMask((SEQ_LEN_SMALL, SEQ_LEN_SMALL)),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    full_split, _, _ = _refine_mask_info_with_segments(
        full_info,
        _make_segment_ids((SEGMENT_SPLIT_HALF, SEGMENT_SPLIT_HALF)),
        q_blocks=SMALL_GRID_DIM,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert full_split.num_active_blocks is not None
    assert int(jax.device_get(full_split.num_active_blocks[0])) == SMALL_GRID_DIM
    assert _active_tile_block_mask(full_split) == {
        (0, 0): BLOCK_FULL,
        (1, 1): BLOCK_FULL,
    }

    full_straddle, _, _ = _refine_mask_info_with_segments(
        full_info,
        _make_segment_ids((SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM)),
        q_blocks=SMALL_GRID_DIM,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert full_straddle.num_active_blocks is not None
    assert (
        int(jax.device_get(full_straddle.num_active_blocks[0]))
        == SMALL_GRID_DIM * SMALL_GRID_DIM
    )
    assert _active_tile_block_mask(full_straddle) == {
        (0, 0): BLOCK_PARTIAL,
        (0, 1): BLOCK_PARTIAL,
        (1, 0): BLOCK_PARTIAL,
        (1, 1): BLOCK_FULL,
    }

    full_qseq_info = splash_attention_mask_info.MaskInfo(
        mask_next=full_info.mask_next,
        active_rows=full_info.active_rows,
        active_cols=full_info.active_cols,
        block_mask=full_info.block_mask,
        num_active_blocks=full_info.num_active_blocks,
        partial_mask_blocks=full_info.partial_mask_blocks,
        q_sequence=jnp.arange(SEQ_LEN_SMALL, dtype=jnp.int32),
    )
    full_qseq_refined, _, _ = _refine_mask_info_with_segments(
        full_qseq_info,
        _make_segment_ids((SEQ_LEN_SMALL,)),
        q_blocks=SMALL_GRID_DIM,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert full_qseq_refined.num_active_blocks is not None
    # Causal q_sequence on a 2x2 FullMask prunes only upper-right tile (0, 1).
    assert (
        int(jax.device_get(full_qseq_refined.num_active_blocks[0]))
        == SMALL_GRID_DIM * (SMALL_GRID_DIM + 1) // 2
    )
    assert _active_tile_block_mask(full_qseq_refined) == {
        (0, 0): BLOCK_PARTIAL,
        (1, 0): BLOCK_FULL,
        (1, 1): BLOCK_PARTIAL,
    }


@pytest.mark.parametrize("grid_kind", ["dynamic", "static"])
def test_refine_mask_info_demotes_base_full_with_partial_blocks_to_segment_only_code(
    grid_kind: Literal["dynamic", "static"],
) -> None:
    """Demote base-full tiles with partial blocks to segment-only code 3."""
    dense_causal = np.tril(np.ones((SEQ_LEN_SMALL, SEQ_LEN_SMALL), dtype=np.bool_))
    seg_ids = _make_segment_ids((SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM))

    if grid_kind == "dynamic":
        mask_info = splash_attention_mask_info.process_dynamic_mask(
            jnp.asarray(dense_causal),
            (BLOCK_SIZE, BLOCK_SIZE),
        )
    else:
        mask_info, _ = splash_attention_mask_info.process_mask(
            splash_attention_mask.NumpyMask(dense_causal),
            (BLOCK_SIZE, BLOCK_SIZE),
        )
    refined, _, _ = _refine_mask_info_with_segments(
        mask_info,
        seg_ids,
        q_blocks=SMALL_GRID_DIM,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert _active_tile_block_mask(refined) == {
        (0, 0): BLOCK_PARTIAL,
        (1, 0): BLOCK_SEGMENT_ONLY,
        (1, 1): BLOCK_PARTIAL,
    }


def test_refine_mask_info_preserves_mask_next_over_127_dynamic() -> None:
    """Keep int16/int32 precision on refined dynamic mask_next IDs above 127."""
    dense_causal = jnp.tril(jnp.ones((SEQ_LEN_LARGE, SEQ_LEN_LARGE), dtype=jnp.bool_))
    mask_info = splash_attention_mask_info.process_dynamic_mask(
        dense_causal, (BLOCK_SIZE, BLOCK_SIZE)
    )
    refined, _, _ = _refine_mask_info_with_segments(
        mask_info,
        _make_segment_ids((SEGMENT_SPLIT_LARGE_MAIN, SEGMENT_SPLIT_LARGE_MAIN)),
        q_blocks=LARGE_GRID_DIM,
        kv_blocks=LARGE_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert refined.mask_next is not None
    assert refined.num_active_blocks is not None
    assert refined.mask_next.dtype in (jnp.int16, jnp.int32)
    count = int(jax.device_get(refined.num_active_blocks[0]))
    active_mask_next = np.asarray(
        jax.device_get(refined.mask_next[:count]), dtype=np.int64
    )
    assert bool(np.all(active_mask_next >= 0))

    tile_bm = _active_tile_block_mask(refined)
    tile_next = _active_tile_mask_next(refined)
    high_partial_ids = [
        tile_next[coord]
        for coord, code in tile_bm.items()
        if code == BLOCK_PARTIAL and coord[0] == coord[1]
    ]
    assert max(high_partial_ids) > INT8_MAX_SIGNED
    for coord, code in tile_bm.items():
        if code == BLOCK_PARTIAL:
            assert tile_next[coord] == coord[0] * LARGE_GRID_DIM + coord[1]


def test_refine_mask_info_preserves_mask_next_over_127_static() -> None:
    """Keep int16/int32 precision on refined static NumpyMask IDs above 127."""
    unique_mask = _make_unique_partial_mask(LARGE_GRID_DIM, BLOCK_SIZE)
    mask_info, _ = splash_attention_mask_info.process_mask(
        splash_attention_mask.NumpyMask(unique_mask),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    assert mask_info.partial_mask_blocks is not None
    assert mask_info.partial_mask_blocks.shape[0] == NUM_UNIQUE_BLOCKS_2048

    refined, _, _ = _refine_mask_info_with_segments(
        mask_info,
        _make_segment_ids((SEGMENT_SPLIT_LARGE_MAIN, SEGMENT_SPLIT_LARGE_MAIN)),
        q_blocks=LARGE_GRID_DIM,
        kv_blocks=LARGE_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert refined.mask_next is not None
    assert refined.num_active_blocks is not None
    assert refined.mask_next.dtype in (jnp.int16, jnp.int32)
    count = int(jax.device_get(refined.num_active_blocks[0]))
    active_mask_next = np.asarray(
        jax.device_get(refined.mask_next[:count]), dtype=np.int64
    )
    assert bool(np.all(active_mask_next >= 0))
    assert int(np.max(active_mask_next)) > INT8_MAX_SIGNED

    tile_next = _active_tile_mask_next(refined)
    for coord, next_id in tile_next.items():
        assert next_id == coord[0] * LARGE_GRID_DIM + coord[1]


def test_refine_mask_info_demotes_diag_skip_causal_full_tile_to_segment_only_code() -> (
    None
):
    """Demote lower-triangular full tile to code 3 when `segment_only_full=True`."""
    mask_info, _ = splash_attention_mask_info.process_mask(
        splash_attention_mask.CausalMask((SEQ_LEN_SMALL, SEQ_LEN_SMALL)),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    refined, _, _ = _refine_mask_info_with_segments(
        mask_info,
        _make_segment_ids((SEGMENT_SPLIT_STRADDLE, SEGMENT_SPLIT_MEDIUM)),
        q_blocks=SMALL_GRID_DIM,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
        segment_only_full=True,
    )
    assert _active_tile_block_mask(refined) == {
        (0, 0): BLOCK_PARTIAL,
        (1, 0): BLOCK_SEGMENT_ONLY,
        (1, 1): BLOCK_PARTIAL,
    }


def test_refine_mask_info_keeps_positive_offset_causal_tiles() -> None:
    """Retain active positive-offset causal tiles across segment refinement."""
    mask_info, _ = splash_attention_mask_info.process_mask(
        splash_attention_mask.CausalMask(
            (BLOCK_SIZE, SEQ_LEN_SMALL), offset=POSITIVE_CAUSAL_OFFSET
        ),
        (BLOCK_SIZE, BLOCK_SIZE),
    )
    refined, _, _ = _refine_mask_info_with_segments(
        mask_info,
        splash_attention_kernel.SegmentIds(
            q=jnp.zeros((BLOCK_SIZE,), dtype=jnp.int32),
            kv=jnp.zeros((SEQ_LEN_SMALL,), dtype=jnp.int32),
        ),
        q_blocks=1,
        kv_blocks=SMALL_GRID_DIM,
        bq=BLOCK_SIZE,
        bkv=BLOCK_SIZE,
    )
    assert refined.num_active_blocks is not None
    assert int(jax.device_get(refined.num_active_blocks[0])) == SMALL_GRID_DIM
    assert _active_tile_block_mask(refined) == {
        (0, 0): BLOCK_FULL,
        (0, 1): BLOCK_PARTIAL,
    }


def test_process_mask_all_full_computable_mask_clears_q_sequence() -> None:
    """All-full computable masks return `q_sequence=None` with `mask_function=None`."""
    mask = splash_attention_mask.CausalMask(
        (SEQ_LEN_SMALL, SEQ_LEN_SMALL), offset=SEQ_LEN_SMALL - 1
    )
    mask_info, mask_function = splash_attention_mask_info.process_mask(
        mask, (BLOCK_SIZE, BLOCK_SIZE)
    )
    assert mask_info.block_mask is None
    assert mask_function is None
    assert mask_info.q_sequence is None


def test_diag_skip_rejects_nonzero_offset_and_dynamic_masks() -> None:
    """Reject diagonal skip flags for non-zero offset and dynamic masks."""
    configs = (
        splash_attention_kernel.SplashConfig(
            block_q=BLOCK_SIZE,
            block_kv=BLOCK_SIZE,
            block_kv_compute=BLOCK_SIZE,
            qk_diag_skip=True,
        ),
        splash_attention_kernel.SplashConfig(
            block_q=BLOCK_SIZE,
            block_kv=BLOCK_SIZE,
            block_kv_compute=BLOCK_SIZE,
            sv_diag_skip=True,
        ),
    )
    offset_mask = splash_attention_mask.CausalMask(
        (SEQ_LEN_SMALL, SEQ_LEN_SMALL), offset=INVALID_CAUSAL_OFFSET
    )
    dynamic_mask = jnp.tril(jnp.ones((SEQ_LEN_SMALL, SEQ_LEN_SMALL), dtype=jnp.bool_))

    for config in configs:
        with pytest.raises(ValueError, match="CausalMask\\(offset=0\\)"):
            splash_attention_kernel.make_splash_mha_single_device(
                offset_mask, config=config
            )
        with pytest.raises(ValueError, match="unsupported for dynamic masks"):
            splash_attention_kernel.make_dynamic_splash_mha(dynamic_mask, config=config)


@pytest.mark.tpu_test
@pytest.mark.parametrize("mask_kind", ["causal", "non_causal"])
@pytest.mark.parametrize("attention_kind", ["mha", "mqa"])
@pytest.mark.parametrize("grid_kind", ["static", "dynamic"])
@pytest.mark.parametrize("segment_case", ["none", "split_128_128", "split_64_192"])
def test_splash_attention_forward_parity(
    mask_kind: Literal["causal", "non_causal"],
    attention_kind: Literal["mha", "mqa"],
    grid_kind: Literal["static", "dynamic"],
    segment_case: Literal["none", "split_128_128", "split_64_192"],
) -> None:
    """Verify forward Splash attention parity against an FP64 NumPy reference."""
    _require_tpu()
    _run_forward_parity_case(mask_kind, attention_kind, grid_kind, segment_case)


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "diag_skip",
    [
        DiagSkipMode(qk_diag_skip=True, sv_diag_skip=False),
        DiagSkipMode(qk_diag_skip=False, sv_diag_skip=True),
        DiagSkipMode(qk_diag_skip=True, sv_diag_skip=True),
    ],
    ids=["qk_only", "sv_only", "qk_and_sv"],
)
def test_splash_attention_causal_diag_skip_with_segments_tpu(
    diag_skip: DiagSkipMode,
) -> None:
    """Verify diagonal skip modes match FP64 reference under straddling segments."""
    _require_tpu()
    _run_diag_skip_case(diag_skip)


@pytest.mark.tpu_test
def test_splash_attention_dynamic_over_127_blocks_tpu() -> None:
    """Verify 2048x2048 dynamic mask parity past block ID 127 on TPU."""
    _require_tpu()
    _run_dynamic_over_127_case()


@pytest.mark.tpu_test
@pytest.mark.parametrize("reciprocal_mode", ["fused", "unfused"])
def test_splash_attention_base2_exp_returns_natural_logsumexp_tpu(
    reciprocal_mode: Literal["fused", "unfused"],
) -> None:
    """Verify `use_base2_exp` + `q_prescaled_log2e` outputs natural-log LSE."""
    _require_tpu()
    _run_base2_natural_logsumexp_case(reciprocal_mode)
