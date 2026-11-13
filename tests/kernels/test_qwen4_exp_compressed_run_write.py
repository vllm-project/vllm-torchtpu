# SPDX-License-Identifier: Apache-2.0
"""Run-based compressed-key writer (interpret mode): the same pool as
write_compressed's tiled / per-pair kernels."""

import jax
import numpy as np
import pytest
from test_qwen4_exp_fused_kv_write import (  # noqa: F401 (fixture)
    LAYOUTS,
    interpret,
    make_step,
    raw,
    run,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import compressed_write

CASES = {
    # MTP verify and plain decode steps (the per-pair path without the flag).
    "verify": ([4, 4, 4, 0, 4, 4, 0, 4], [37, 0, 509, 0, 1022, 6, 0, 2]),
    "decode": ([1] * 8, [0, 1, 2, 3, 4, 7, 511, 1023]),
    "ragged": ([2, 4, 1, 3], [510, 3, 8, 126]),
    # Prefill-sized mixed chunks (the tiled path without the flag): requests
    # misaligned against their contexts, crossing pages, plus decode rows.
    "prefill_mixed": ([250, 4, 4, 126], [3, 509, 0, 1001]),
    "prefill_aligned": ([256, 128], [0, 512]),
}


def written(monkeypatch, run_write, args, options):
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", run_write)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS", 0)
    jax.clear_caches()
    pool, _, _ = run(monkeypatch, False, args, **options)
    return raw(pool)


@pytest.mark.parametrize(
    "layout", ["fp8_paired_slot1_packed_k", "fp8_whole_rows", "bf16_whole_rows"]
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_run_writer_matches_write_compressed(
    interpret,  # noqa: F811
    monkeypatch,
    case,
    layout,
):
    dtype, options = LAYOUTS[layout]
    query_lens, contexts = CASES[case]
    args = make_step(np.random.default_rng(5), query_lens, contexts, dtype)
    options = {**options, "max_query_len": max(query_lens)}
    expected = written(monkeypatch, False, args, options)
    got = written(monkeypatch, True, args, options)
    np.testing.assert_array_equal(got, expected)


def test_capacity_bounds_the_writers():
    # Every request of n tokens touches at most n // (2 * ratio) + 2 pairs.
    for tokens, requests in ((8192, 33), (32, 8), (8, 8)):
        cap = compressed_write.writer_capacity(tokens, requests, 4)
        assert cap >= tokens // 8 + 2 * requests and cap % 8 == 0


@pytest.mark.parametrize("layout", ["fp8_paired_slot1_packed_k", "bf16_whole_rows"])
@pytest.mark.parametrize("case", ["prefill_mixed", "ragged", "verify"])
def test_packed_mask_matches_write_compressed(
    interpret,  # noqa: F811
    monkeypatch,
    case,
    layout,
):
    # Force the one-int32-per-writer half mask (the >8192-token step layout).
    dtype, options = LAYOUTS[layout]
    query_lens, contexts = CASES[case]
    args = make_step(np.random.default_rng(7), query_lens, contexts, dtype)
    options = {**options, "max_query_len": max(query_lens)}
    expected = written(monkeypatch, False, args, options)
    monkeypatch.setattr(compressed_write, "MASK_2D_MAX_CAP", 0)
    got = written(monkeypatch, True, args, options)
    np.testing.assert_array_equal(got, expected)


def test_mask_layout_fits_smem():
    # 2-D int32 SMEM rows pad to 128 words; the 2-D mask must stay < 1 MiB and
    # 8192-token steps (32 requests, ratio 4) must keep the 2-D layout.
    cap_8k = compressed_write.writer_capacity(8192, 32, 4)
    cap_16k = compressed_write.writer_capacity(16384, 32, 4)
    assert cap_8k <= compressed_write.MASK_2D_MAX_CAP < cap_16k
    assert compressed_write.MASK_2D_MAX_CAP * 128 * 4 < (1 << 20) - (64 << 10)
