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
"""`TPU_DCP_MLA_OUT_VMEM_MAX_TOKENS`: the 64-token DCP MLA call's partial and
lse are VMEM outputs.

`sparse_ragged_paged_attention(out_vmem=True)`, on the served 64-token decode
arguments (live-group prep, capped windows, split q with the in-kernel rope,
lse_mask_empty), must return the same partial and lse, bit for bit, on every
row the attention grid writes; rows past the grid keep lse -inf, so the DCP
merge's rescale (`lse_rescale_head_major`) on the merged lse of every rank is
bit-identical and 0 on them. Positive control: a padding row whose lse is not
-inf lets the undefined partial into the rescale.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import lse_rescale
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

TOKENS, HEADS, NOPE, RAW_HEAD, TOPK, CAP, RANKS = 64, 32, 512, 256, 2048, 512, 8
PAGE_SIZE, CACHE_TOKENS = 256, 8192
K_SCALE, SM_SCALE = 0.75, 0.125
OUT_VMEM_BYTES = 32 * 1024 * 1024


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore-layout sparse MLA needs a TPU")


def _bits16(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint16))


def _problem(real, seed):
    """One DCP rank's 64-token decode call with `real` real tokens (4 per
    request): an empty row, an overflowing row, padding rows past `real`."""
    rng = np.random.default_rng(seed)
    pages = CACHE_TOKENS // PAGE_SIZE

    def quantized(dim):
        values = rng.random((pages, PAGE_SIZE, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values / K_SCALE).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope = quantized(NOPE).reshape(pages, PAGE_SIZE, 4, 128)
    rope = np.pad(quantized(64), ((0, 0), (0, 0), (0, 64)))
    rope = rope.reshape(pages, PAGE_SIZE // 4, 4, 128)
    counts = np.clip(rng.normal(256, 70, size=TOKENS), 0, CAP).astype(int)
    counts[3], counts[5], counts[real:] = 0, 700, 0
    physical = np.stack(
        [rng.choice(CACHE_TOKENS, size=TOPK, replace=False) for _ in range(TOKENS)]
    )
    page_table = rng.permutation(pages).astype(np.int32)
    inverse = np.argsort(page_table)
    logical = inverse[physical // PAGE_SIZE] * PAGE_SIZE + physical % PAGE_SIZE
    live = np.arange(TOPK)[None, :] < counts[:, None]
    seqs = real // 4
    angle = rng.uniform(0, 2 * np.pi, size=(TOKENS, 32)).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    table = np.stack([np.tile(cos, 4), np.concatenate([-sin, sin, -sin, sin], 1)], 1)
    return dict(
        nope=jnp.asarray(native_sc_cache_layout.pack_nope(nope)),
        rope=jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope)),
        topk=jnp.asarray(np.where(live, logical, -1).astype(np.int32)),
        block_tables=jnp.asarray(np.tile(page_table, seqs)),
        cu_q_lens=jnp.asarray(np.arange(seqs + 1) * 4, jnp.int32),
        distribution=jnp.array([seqs, seqs, seqs], jnp.int32),
        ql_nope=jnp.asarray(rng.normal(size=(TOKENS, HEADS, NOPE)), jnp.bfloat16),
        q_raw=jnp.asarray(rng.normal(size=(TOKENS, HEADS, RAW_HEAD)), jnp.bfloat16),
        rope_table=jnp.asarray(table, jnp.float32),
        other_lses=_other_ranks_lse(rng, real),
    )


def _other_ranks_lse(rng, real):
    """The lse of the other DCP ranks: the same -inf rows (empty, padding)."""
    lses = rng.normal(0, 2, size=(RANKS - 1, TOKENS, HEADS)).astype(np.float32)
    lses[:, 3] = lses[:, real:] = -np.inf
    return lses


ARRAYS = (
    "ql_nope",
    "nope",
    "rope",
    "topk",
    "block_tables",
    "cu_q_lens",
    "distribution",
    "q_raw",
    "rope_table",
)


def _attention(*arrays, out_vmem):
    ql_nope, nope, rope, topk, block_tables, cu_q_lens, distribution, q_raw, table = (
        arrays
    )
    return sparse_mla.sparse_ragged_paged_attention(
        ql_nope,
        nope,
        rope,
        topk,
        block_tables,
        cu_q_lens,
        distribution,
        sm_scale=SM_SCALE,
        k_scale=K_SCALE,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=512,
        topk_cap=CAP,
        small_chunk_batch_size=4,
        small_chunk_tokens=32,
        q_pe=q_raw,
        q_rope_table=table,
        lse_lane_dense=True,
        lse_mask_empty=True,
        is_dcp_mode=True,
        fused_window_prep=True,
        mixed_live_gather=True,
        mixed_live_widths=(256, 384),
        mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT
        + sparse_mla.MIXED_KERNEL_V2_LEVERS,
        mixed_fused_prep=True,
        decode_row_widths=(128, 256, 384),
        decode_body_levers=sparse_mla.DECODE_BODY_DEFAULT,
        out_vmem=out_vmem,
    )


def _arrays(p):
    return tuple(p[k] for k in ARRAYS)


def _call(p, out_vmem):
    out_vmem = OUT_VMEM_BYTES if out_vmem else 0
    out, lse = jax.block_until_ready(_attention(*_arrays(p), out_vmem=out_vmem))
    return np.asarray(out), np.asarray(lse)


def _rescaled(out, lse, other_lses):
    """The merge's weighted partial, head-major, as cp_lse_ag_out_rs runs it."""
    merged = jax.scipy.special.logsumexp(
        jnp.concatenate([jnp.asarray(lse)[None], jnp.asarray(other_lses)]), axis=0
    )
    weighted = lse_rescale.lse_rescale_head_major(
        jnp.asarray(out), jnp.asarray(lse), merged, "bfloat16"
    )
    return _bits16(weighted)


@pytest.mark.parametrize("real", [36, 64])
def test_written_rows_and_merge_bit_identical(real):
    _require_tpu()
    p = _problem(real, seed=real)
    off_out, off_lse = _call(p, out_vmem=False)
    on_out, on_lse = _call(p, out_vmem=True)
    assert np.array_equal(_bits16(on_out[:real]), _bits16(off_out[:real]))
    assert np.array_equal(on_lse[:real].view(np.uint32), off_lse[:real].view(np.uint32))
    assert np.all(on_lse[real:] == -np.inf) and np.all(off_lse[real:] == -np.inf)
    off_x = _rescaled(off_out, off_lse, p["other_lses"])
    on_x = _rescaled(on_out, on_lse, p["other_lses"])
    assert np.array_equal(on_x, off_x)
    assert not on_x[:, real:].any()


def test_positive_control_padding_lse():
    _require_tpu()
    real = 36
    p = _problem(real, seed=real)
    out, lse = _call(p, out_vmem=True)
    out = out.copy()
    out[real:] = np.nan
    finite_pad = lse.copy()
    finite_pad[real:] = 0.0
    assert not np.array_equal(
        _rescaled(out, finite_pad, p["other_lses"]),
        _rescaled(out, lse, p["other_lses"]),
    )


def test_switch_places_outputs_in_vmem():
    _require_tpu()
    p = _problem(64, seed=64)

    def lowered(out_vmem):
        return (
            jax.jit(_attention, static_argnames="out_vmem")
            .lower(*_arrays(p), out_vmem=OUT_VMEM_BYTES if out_vmem else 0)
            .as_text()
        )

    on, off = lowered(True), lowered(False)
    assert "output_memory_space_colors" in on and "capw_4-ov" in on
    assert "output_memory_space_colors" not in off and "capw_4-ov" not in off
