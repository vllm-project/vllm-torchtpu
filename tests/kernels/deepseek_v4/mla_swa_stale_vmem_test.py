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
"""Stale VMEM must not reach the batched SWA decode kernel's output.

A decode step fills at most `sliding_window` of each sequence's `bkv_sz`
buffer slots; the rest hold whatever the previous kernel left in VMEM. Here a
preceding kernel fills VMEM with NaN or -inf bytes, then one decode step runs
with nothing in between. Every output must be finite and match a zero fill.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla_swa

# DeepSeek-V4-Flash geometry: window 128, page 128, 2 pages per block, so
# bkv_sz == 256 >= sliding_window selects the batched decode kernel.
HEADS, HEAD_DIM, WINDOW, PAGE, PAGES_PER_SEQ = 128, 512, 128, 128, 8
POISON_ROWS = (40 * 1024 * 1024) // (8 * 128)  # covers the 16 MiB bkv buffer
BLK = 1024
VMEM_LIMIT = 60 * 1024 * 1024


def _poison_vmem(fill):
    """Fill 40 MiB of scoped VMEM with `fill` bytes.

    0xFF everywhere is bf16 NaN; (0x80, 0xFF) on alternating sublanes is
    bf16 -inf in the byte layout `prepare_kv_inputs` uses.
    """

    def kernel(o_ref, scratch):
        if fill == "nan":
            blk = jnp.full((BLK, 8, 128), 0xFF, jnp.uint8)
        elif fill == "ninf":
            sub = jax.lax.broadcasted_iota(jnp.int32, (BLK, 8, 128), 1)
            blk = jnp.where(sub % 2 == 0, 0x80, 0xFF).astype(jnp.uint8)
        else:
            blk = jnp.zeros((BLK, 8, 128), jnp.uint8)

        @pl.loop(0, POISON_ROWS // BLK)
        def _(i):
            scratch[pl.ds(pl.multiple_of(i * BLK, BLK), BLK)] = blk

        o_ref[...] = scratch[POISON_ROWS - 1].astype(jnp.int32)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
        scratch_shapes=[pltpu.VMEM((POISON_ROWS, 8, 128), jnp.uint8)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT),
    )()


# The aliased operands are donated so XLA inserts no copies between the
# poison and the kernel; a copy would overwrite the poisoned VMEM.
@functools.partial(jax.jit, donate_argnums=(2, 6, 7, 8))
def _decode(
    q, new_kv, cache, kv_lens, page_indices, cu_q_lens, in_out, l_sum, m_max, sinks
):
    batch = kv_lens.shape[0]
    return mla_swa.run_mla_batched_decode_kernel(
        q,
        new_kv,
        cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=2,
        num_queries_per_block=1,
        start_seq_idx=jnp.array(0),
        end_seq_idx=jnp.array(batch),
        in_output=in_out,
        in_l=l_sum,
        in_m=m_max,
        attention_sinks=sinks,
        sm_scale=HEAD_DIM**-0.5,
        sliding_window=WINDOW,
        logical_page_size=PAGE,
        vmem_limit_bytes=mla_swa.DEFAULT_VMEM_LIMIT_BYTES,
        unnormalized_output=True,
        seq_batch_sz=32,
    )


class MlaSwaStaleVmemTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.kv_lens = np.array([3, 9, 17, 40, 64, 100, 127, 128])
        batch = len(self.kv_lens)
        rng = np.random.default_rng(0)
        total_pages = batch * PAGES_PER_SEQ
        tok = jnp.asarray(
            rng.standard_normal((total_pages * PAGE, HEAD_DIM)), jnp.bfloat16
        )
        self.cache = mla_swa.prepare_kv_inputs(tok).reshape(
            total_pages, PAGE * 2, 4, 128
        )
        self.q = jnp.asarray(
            rng.standard_normal((batch, HEADS, HEAD_DIM)), jnp.bfloat16
        )
        new_kv = jnp.asarray(rng.standard_normal((batch, HEAD_DIM)), jnp.bfloat16)
        self.new_kv = mla_swa.prepare_kv_inputs(new_kv).reshape(batch, 8, 128)
        self.page_indices = jnp.arange(total_pages, dtype=jnp.int32)
        self.cu_q_lens = jnp.arange(batch + 1, dtype=jnp.int32)
        self.sinks = jnp.zeros((HEADS,), jnp.float32)

    def _step(self, fill):
        batch = len(self.kv_lens)
        cache = jnp.copy(self.cache)
        in_out = jnp.zeros_like(self.q)
        l_sum = jnp.zeros((batch, 128), jnp.float32)
        m_max = jnp.zeros((batch, 128), jnp.float32)
        for a in (cache, in_out, l_sum, m_max):
            a.block_until_ready()
        _poison_vmem(fill).block_until_ready()
        kv_lens = jnp.asarray(self.kv_lens, jnp.int32)
        out, _, l_sum, m_max = _decode(
            self.q,
            self.new_kv,
            cache,
            kv_lens,
            self.page_indices,
            self.cu_q_lens,
            in_out,
            l_sum,
            m_max,
            self.sinks,
        )
        out = np.asarray(out.astype(jnp.float32))
        return out, np.asarray(l_sum), np.asarray(m_max)

    @parameterized.parameters("nan", "ninf")
    def test_stale_tail_does_not_reach_output(self, fill):
        ref, _, _ = self._step("zero")
        out, l_sum, m_max = self._step(fill)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(l_sum).all())
        self.assertTrue(np.isfinite(m_max).all())
        np.testing.assert_array_equal(out, ref)


if __name__ == "__main__":
    absltest.main()
