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
"""`TPU_MLA_DECODE_ROW_WIDTHS`: decode grid steps attend only the rows they need.

`sparse_ragged_paged_attention(decode_row_widths=...)` must return the same
partials and lse, bit for bit, as without it, on the served decode path
(prologue rope, kernel-owned lse, capped SparseCore gather) -- with per-token
live counts chosen so every width bucket, the full-width bucket and the
overflow fix-up (kv_len > cap) each serve some grid step. Positive control:
the `row_width_undersize` fault (each bucket attends one width too narrow)
must be caught.
"""

import importlib.util
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla

pytestmark = pytest.mark.tpu_test

WIDTHS = (128, 256, 384)


def _load(rel):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", rel)
    spec = importlib.util.spec_from_file_location(
        "_rw_" + os.path.basename(rel)[:-3], path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("kernels/test_mla_decode_prologue.py")

# Per 4-token grid step (batch max in brackets): [128] [256] [384] [512]
# [>512, fix-up] [<=128 with an empty row] [129: just over the first width].
COUNTS = (
    [0, 50, 100, 128]
    + [129, 200, 1, 256]
    + [257, 300, 10, 384]
    + [385, 450, 511, 512]
    + [513, 3, 700, 64]
    + [0, 0, 7, 127]
    + [129, 1, 2, 3]
    + [64, 64, 64, 64]
)


def _with_counts(p, counts):
    """`p` with token t's top-k row cut to its first counts[t] entries (all of
    `_problem`'s rows are re-drawn full first)."""
    topk = np.asarray(p["topk_full"])
    n = topk.shape[0]
    c = np.resize(np.asarray(counts), n)
    live = np.arange(topk.shape[1])[None, :] < c[:, None]
    p = dict(p)
    p["topk"] = jnp.asarray(np.where(live, topk, -1).astype(np.int32))
    p["counts"] = c
    return p


def _problem(seed, query_tokens, seq_q):
    rng = np.random.default_rng(seed)
    p = P._problem(rng, query_tokens, seq_q)
    # Full rows: rebuild the logical rows without the -1 tail.
    full = np.asarray(p["topk"]).copy()
    for t in range(full.shape[0]):
        row = full[t]
        if (row == -1).any():
            used = set(row[row != -1].tolist())
            pool = [x for x in rng.permutation(8192) if x not in used]
            row[row == -1] = np.asarray(pool[: int((row == -1).sum())], np.int32)
    p["topk_full"] = full
    return p


def _call(p, widths, fault=None):
    cache = P._cos_sin_cache()
    out, lse = sparse_mla.sparse_ragged_paged_attention(
        p["ql_nope"],
        p["nope"],
        p["rope"],
        p["topk"],
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=P.SM_SCALE,
        k_scale=P.K_SCALE,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=256,
        topk_cap=512,
        small_chunk_batch_size=4,
        small_chunk_tokens=32,
        lse_lane_dense=True,
        lse_mask_empty=True,
        fused_window_prep=True,
        mixed_live_gather=True,
        mixed_live_widths=(256, 384),
        mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
        q_pe=p["q_raw"],
        q_rope_table=P.rope_table(p["positions"], cache),
        lse_in_kernel=True,
        prologue_fault=fault,
        decode_row_widths=widths,
    )
    return jax.block_until_ready((out[..., : P.NOPE_DIM], lse))


CELLS = (
    (16, (4,)),
    (16, (4, 4)),
    (16, (4, 4, 4, 4)),
    (16, (4, 4, 4, 1)),
    (32, (4,) * 8),
    # Mixed bucket: the live-group prep owns it; widths must not apply.
    (512, (4, 4, 492)),
)


@pytest.mark.parametrize("query_tokens,seq_q", CELLS)
@pytest.mark.parametrize("shift", [0, 4, 12])
def test_bit_identical(query_tokens, seq_q, shift):
    P._require_tpu()
    p = _problem(2026 + query_tokens, query_tokens, seq_q)
    counts = COUNTS[shift:] + COUNTS[:shift]
    p = _with_counts(p, counts)
    ref, ref_lse = _call(p, ())
    out, lse = _call(p, WIDTHS)
    np.testing.assert_array_equal(P._bits16(out), P._bits16(ref))
    np.testing.assert_array_equal(P._bits32(lse), P._bits32(ref_lse))
    real = int(sum(seq_q))
    assert np.isfinite(np.asarray(out, np.float32)[:real]).all()


@pytest.mark.parametrize("widths", [(128,), (256,), (128, 384), (128, 256, 384)])
def test_bit_identical_width_sets(widths):
    P._require_tpu()
    p = _with_counts(_problem(77, 32, (4,) * 8), COUNTS)
    ref, ref_lse = _call(p, ())
    out, lse = _call(p, widths)
    np.testing.assert_array_equal(P._bits16(out), P._bits16(ref))
    np.testing.assert_array_equal(P._bits32(lse), P._bits32(ref_lse))


def test_positive_control_fails():
    P._require_tpu()
    p = _with_counts(_problem(2042, 32, (4,) * 8), COUNTS)
    ref, ref_lse = _call(p, ())
    out, lse = _call(p, WIDTHS, fault="row_width_undersize")
    same = np.array_equal(P._bits16(out), P._bits16(ref)) and np.array_equal(
        P._bits32(lse), P._bits32(ref_lse)
    )
    assert not same, "row_width_undersize not caught"


def test_rejects_bad_widths():
    p = _with_counts(_problem(5, 16, (4,)), COUNTS)
    for bad in ((100,), (512,), (256, 128)):
        with pytest.raises(ValueError):
            _call(p, bad)
