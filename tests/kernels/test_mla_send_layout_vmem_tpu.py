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
"""`TPU_MLA_SEND_LAYOUT_VMEM_CAP` on a TPU: the send-layout DCP decode call.

`sparse_ragged_paged_attention(send_vmem_cap=True)`, on a 32-token DCP decode
call that takes the send layout without the fused exchange (split q with the
in-kernel rope, capped windows, fused window prep), compiles and returns the
same outputs, bit for bit, as with the cap off: the attention output and the
send buffer. Two shapes: the 32 MiB cap (batch 8, top-k cap 512) and a
scratch larger than the cap (batch 16, top-k cap 1024), where the limit is the
call's own VMEM scratch plus the headroom. Positive control: the switch pins
the call's operands to HBM and sets its scoped VMEM limit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

TOKENS, HEADS, NOPE, TOPK, RANKS = 32, 64, 512, 2048, 8
PAGE_SIZE, CACHE_TOKENS = 256, 8192
K_SCALE, SM_SCALE = 0.75, 0.125
# (batch, top-k cap): the 32 MiB cap, and a scratch above it.
SHAPES = [(8, 512), (16, 1024)]


def _require_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore-layout sparse MLA needs a TPU")


def _problem(cap, seed):
    """One DCP rank's 32-token decode call (4 tokens per request): an empty
    row and an overflowing row among the top-k counts."""
    rng = np.random.default_rng(seed)
    pages = CACHE_TOKENS // PAGE_SIZE

    def quantized(dim):
        values = rng.random((pages, PAGE_SIZE, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values / K_SCALE).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope = quantized(NOPE).reshape(pages, PAGE_SIZE, 4, 128)
    rope = np.pad(quantized(64), ((0, 0), (0, 0), (0, 64)))
    rope = rope.reshape(pages, PAGE_SIZE // 4, 4, 128)
    counts = np.clip(rng.normal(cap // 2, cap // 6, size=TOKENS), 0, cap)
    counts = counts.astype(int)
    counts[3], counts[5] = 0, cap + 188
    physical = np.stack(
        [rng.choice(CACHE_TOKENS, size=TOPK, replace=False) for _ in range(TOKENS)]
    )
    page_table = rng.permutation(pages).astype(np.int32)
    inverse = np.argsort(page_table)
    logical = inverse[physical // PAGE_SIZE] * PAGE_SIZE + physical % PAGE_SIZE
    live = np.arange(TOPK)[None, :] < counts[:, None]
    seqs = TOKENS // 4
    angle = rng.uniform(0, 2 * np.pi, size=(TOKENS, 32)).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    table = np.stack([np.tile(cos, 4), np.concatenate([-sin, sin, -sin, sin], 1)], 1)
    return (
        jnp.asarray(rng.normal(size=(TOKENS, HEADS, NOPE)), jnp.bfloat16),
        jnp.asarray(native_sc_cache_layout.pack_nope(nope)),
        jnp.asarray(native_sc_cache_layout.pack_rope_banded(rope)),
        jnp.asarray(np.where(live, logical, -1).astype(np.int32)),
        jnp.asarray(np.tile(page_table, seqs)),
        jnp.asarray(np.arange(seqs + 1) * 4, jnp.int32),
        jnp.array([seqs, seqs, seqs], jnp.int32),
        jnp.asarray(
            rng.normal(size=(TOKENS, HEADS, sparse_mla.Q_ROPE_HEAD_DIM)), jnp.bfloat16
        ),
        jnp.asarray(table, jnp.float32),
    )


def _attention(*arrays, batch, cap, send_vmem_cap):
    ql_nope, nope, rope, topk, block_tables, cu_q_lens, distribution, q_pe, table = (
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
        attention_kernel_batch_size=batch,
        topk_cap=cap,
        q_pe=q_pe,
        q_rope_table=table,
        lse_lane_dense=True,
        lse_mask_empty=True,
        lse_in_kernel=True,
        is_dcp_mode=True,
        fused_window_prep=True,
        decode_row_widths=(128, 256, 384),
        decode_body_levers=sparse_mla.DECODE_BODY_DEFAULT,
        send_world=RANKS,
        send_vmem_cap=send_vmem_cap,
    )


def _bits(result):
    """Every output array as raw bytes (bf16 / f32 / int16 alike)."""
    return [np.asarray(x).view(np.uint8) for x in jax.tree.leaves(result)]


@pytest.mark.parametrize(("batch", "cap"), SHAPES)
def test_capped_send_layout_call_bit_identical(batch, cap):
    _require_tpu()
    arrays = _problem(cap, seed=batch)
    off = _bits(
        jax.block_until_ready(
            _attention(*arrays, batch=batch, cap=cap, send_vmem_cap=False)
        )
    )
    on = _bits(
        jax.block_until_ready(
            _attention(*arrays, batch=batch, cap=cap, send_vmem_cap=True)
        )
    )
    assert len(on) == len(off) == 2
    for a, b in zip(on, off):
        assert np.array_equal(a, b)


def test_switch_pins_the_call_and_sets_its_limit():
    _require_tpu()
    batch, cap = SHAPES[0]
    arrays = _problem(cap, seed=batch)

    def lowered(send_vmem_cap):
        return (
            jax.jit(_attention, static_argnames=("batch", "cap", "send_vmem_cap"))
            .lower(*arrays, batch=batch, cap=cap, send_vmem_cap=send_vmem_cap)
            .as_text()
        )

    on, off = lowered(True), lowered(False)
    limit = f"\\22size\\22: {sparse_mla.SEND_LAYOUT_VMEM_LIMIT_BYTES}"
    assert "input_memory_space_colors" in on and limit in on
    assert "input_memory_space_colors" not in off and limit not in off
