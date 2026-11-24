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
"""TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q under tracing only: the 4-bit
write + candidates half traces (``jax.eval_shape``) at served prefill shapes
(an 8K chunk over ~150K tokens, DCP 2 / DCP 4) with the same output shapes
as the default tiles, and the torch DCP op's fake keeps its schema with a
symbolic token count under ``FakeTensorMode``. Nothing lowers or runs."""

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4

ENV = "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q"
PAGE, D, TOPK = 128, 128, 16


def _shapes(num_tokens, dcp, g, ctx=150_000, max_seqs=16):
    pps = -(-(ctx + num_tokens) // (dcp * PAGE))
    pages = pps + 1
    sds = jax.ShapeDtypeStruct
    return (
        sds((num_tokens, g, D), jnp.bfloat16),  # k
        sds((num_tokens, g, D), jnp.bfloat16),  # v
        sds((num_tokens, g, D), jnp.bfloat16),  # index q
        sds((num_tokens, D), jnp.bfloat16),  # index k
        sds((pages, PAGE // 8, 4, 2 * g * D), jnp.uint8),  # kv cache
        sds((pages, PAGE // 8 + 1, 4, D), jnp.uint8),  # index cache (folded)
        sds((max_seqs,), jnp.int32),
        sds((max_seqs * pps,), jnp.int32),
        sds((max_seqs + 1,), jnp.int32),
        sds((3,), jnp.int32),
        sds((pages, msa_fp4.scale_rows(g), PAGE), jnp.float32),
    )


def _trace(num_tokens, dcp, g):
    def fn(k, v, iq, ik, kv, index, seq_lens, tables, cu, dist, sc):
        return msa_dcp.msa_write_and_candidates_dcp(
            k, v, iq, ik, kv, index, seq_lens, tables, cu, dist,
            topk=TOPK, init_blocks=0, local_blocks=1, rank=0, dcp=dcp,
            scale_cache=sc,
        )  # fmt: skip

    return jax.eval_shape(fn, *_shapes(num_tokens, dcp, g))


@pytest.mark.parametrize("dcp,g", [(2, 1), (4, 2)], ids=["dcp2", "dcp4"])
@pytest.mark.parametrize("num_tokens", [4096, 8192])
def test_served_prefill_traces(dcp, g, num_tokens, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    ref = _trace(num_tokens, dcp, g)
    for wide in ("1024", "2048"):
        monkeypatch.setenv(ENV, wide)
        out = _trace(num_tokens, dcp, g)
        assert jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), out) == (
            jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), ref)
        )
    assert ref[-1].shape == (num_tokens, g * 2 * TOPK)


def test_dcp_op_fake_symbolic(monkeypatch):
    """The knob is read when JAX traces; the torch op's fake is unchanged."""
    pytest.importorskip("torch_tpu")
    from tests.models.minimax_m3 import test_kv4_shared_scales_fake_tracing as f

    monkeypatch.setenv(ENV, "1024")
    for name in sorted(f.SHAPES):
        f.test_dcp_op_shared_page_symbolic(name)
