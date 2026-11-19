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
"""`TPU_DCP_MLA_Q_HEAD_MAJOR`: the 64-token DCP MLA call reads the absorbed
query head-major.

`sparse_ragged_paged_attention(q_head_major=swapaxes(q, 0, 1))` on the served
64-token decode arguments (live-group prep, capped windows, split q with the
in-kernel rope, VMEM outputs) must return the same partial and lse, bit for
bit, as the token-major call, on every row the attention grid writes.
Positive control: the test-only fault `q_head_major_reshape` reads the
head-major block as if it were token-major and must change the partial.
A lowering check asserts that, behind XLA's batched absorption matmul, the
head-major call has no transposing copy of the query and the token-major
call has one.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

TOKENS, NOPE, RAW_HEAD, TOPK, CAP = 64, 512, 256, 2048, 512
PAGE_SIZE, CACHE_TOKENS = 256, 8192
K_SCALE, SM_SCALE = 0.75, 0.125
OUT_VMEM_BYTES = 32 * 1024 * 1024
# TPU_DCP_MLA_MIXED_BODY_SMALL of the served GLM-5.2 decode configuration.
SMALL_LEVERS = ("tr", "zu", "pv512", "noinf", "fxskip", "quad")


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore-layout sparse MLA needs a TPU")


def _bits16(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint16))


def _problem(real, heads, seed):
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
    counts = np.clip(rng.normal(150, 40, size=TOKENS), 0, CAP).astype(int)
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
        ql_nope=jnp.asarray(rng.normal(size=(TOKENS, heads, NOPE)), jnp.bfloat16),
        nope=jnp.asarray(native_sc_cache_layout.pack_nope(nope)),
        rope=jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope)),
        topk=jnp.asarray(np.where(live, logical, -1).astype(np.int32)),
        block_tables=jnp.asarray(np.tile(page_table, seqs)),
        cu_q_lens=jnp.asarray(np.arange(seqs + 1) * 4, jnp.int32),
        distribution=jnp.array([seqs, seqs, seqs], jnp.int32),
        q_raw=jnp.asarray(rng.normal(size=(TOKENS, heads, RAW_HEAD)), jnp.bfloat16),
        rope_table=jnp.asarray(table, jnp.float32),
    )


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


def _attention(*arrays, head_major, fault=None):
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
        mixed_body_small_levers=SMALL_LEVERS,
        mixed_fused_prep=True,
        decode_row_widths=(128, 256, 384),
        decode_body_levers=sparse_mla.DECODE_BODY_DEFAULT,
        out_vmem=OUT_VMEM_BYTES,
        prologue_fault=fault,
        q_head_major=jnp.swapaxes(ql_nope, 0, 1) if head_major else None,
    )


def _call(p, head_major, fault=None):
    fn = jax.jit(functools.partial(_attention, head_major=head_major, fault=fault))
    out, lse = jax.block_until_ready(fn(*(p[k] for k in ARRAYS)))
    return np.asarray(out), np.asarray(lse)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("real", [36, 64])
def test_written_rows_bit_identical(real, heads):
    _require_tpu()
    p = _problem(real, heads, seed=real + heads)
    off_out, off_lse = _call(p, head_major=False)
    on_out, on_lse = _call(p, head_major=True)
    assert np.array_equal(_bits16(on_out[:real]), _bits16(off_out[:real]))
    assert np.array_equal(on_lse[:real].view(np.uint32), off_lse[:real].view(np.uint32))
    assert np.all(on_lse[real:] == -np.inf)


@pytest.mark.parametrize("heads", [32, 64])
def test_positive_control_reshape_fault(heads):
    _require_tpu()
    real = 64
    p = _problem(real, heads, seed=7 + heads)
    off_out, _ = _call(p, head_major=False)
    bad_out, _ = _call(p, head_major=True, fault="q_head_major_reshape")
    live = np.asarray(p["topk"])[:real, 0] >= 0
    diff = np.any(_bits16(bad_out[:real]) != _bits16(off_out[:real]), axis=(1, 2))
    # Every token with live keys sees another token's query rows.
    assert diff[live].all()


def test_head_major_removes_query_copy():
    _require_tpu()
    heads, rank_in = 64, 192
    p = _problem(64, heads, seed=11)
    w_uk = jnp.asarray(
        np.random.default_rng(3).normal(size=(heads, rank_in, NOPE)) * 0.05,
        jnp.bfloat16,
    )
    q_in = p["ql_nope"][..., :rank_in]
    rest = tuple(p[k] for k in ARRAYS[1:])

    def chain(q_nope, w, *rest, head_major):
        ql = jnp.einsum("thd,hdr->thr", q_nope, w, preferred_element_type=jnp.float32)
        return _attention(ql.astype(jnp.bfloat16), *rest, head_major=head_major)

    def copies(head_major):
        hlo = (
            jax.jit(functools.partial(chain, head_major=head_major))
            .lower(q_in, w_uk, *rest)
            .compile()
            .as_text()
        )
        # The query copy reads the absorption's output; the copy of the
        # kernel's VMEM output to the jit's HBM result reads a tuple element.
        shape = f"bf16[{heads},{TOKENS},{NOPE}]"
        return [
            ln
            for ln in hlo.splitlines()
            if " copy(" in ln
            and ln.split("=", 1)[1].strip().startswith(shape)
            and not ln.split(" copy(", 1)[1].startswith("%get-tuple-element")
        ]

    assert copies(head_major=False), "token-major call lost its query copy"
    assert not copies(head_major=True), copies(head_major=True)
