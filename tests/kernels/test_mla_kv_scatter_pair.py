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
"""`TPU_KV_SCATTER_PAIR` on CPU: dispatch and argument plumbing.

With the flag on, `kv_cache_utils._scatter_caches` writes the nope and rope
caches with one SparseCore program (`scatter_utils.scatter_rows_pair`) instead
of two `scatter_utils.scatter_rows` calls. The SparseCore kernels cannot run on
CPU, so these tests replace both kernels with a pure-jnp model of their common
contract: row `t` of `src` goes to row `dst_rows[t]` of the cache, and a
negative or out-of-range destination is dropped. The tests then check that:

- the pair kernel receives the same packed rows and destinations as the two
  single-cache kernels, with `TPU_KV_PACK_FUSED_PAD` on and off;
- the written caches equal the pure-XLA writer, for the unsharded and the DCP
  writers;
- the flag changes nothing unless both caches use the SparseCore layout.

The kernel itself is compared with the two-program path on TPU in
`test_mla_kv_cache_scatter.py` (`test_scatter_pair_matches_two_scatters`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import kv_cache_utils, scatter_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)

LKV_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
NUM_PAGES = 64
# Plan size of a 16- or 32-token step on tpu7x. CPU has no SparseCore info, so
# `padded_num_tokens` is replaced by this rounding.
PLAN_TOKENS = 32


def _plan_tokens(num_tokens):
    return -(-num_tokens // PLAN_TOKENS) * PLAN_TOKENS


def _write_rows(cache, src, dst_rows):
    """Pure-jnp model of one SparseCore cache write (the kernels' contract)."""
    src, dst_rows = scatter_utils.pad_to_plan(
        src, dst_rows, _plan_tokens(dst_rows.shape[0])
    )
    rows = cache.reshape(-1, src.shape[-1])
    keep = (dst_rows >= 0) & (dst_rows < rows.shape[0])
    target = jnp.where(keep, dst_rows, rows.shape[0])  # out of range: dropped
    return rows.at[target].set(src, mode="drop").reshape(cache.shape)


class _FakeKernels:
    """Records the arguments each fake kernel receives."""

    def __init__(self):
        self.single = []
        self.pair = []

    def scatter_rows(self, cache, src, dst_rows, name=""):
        self.single.append((src, dst_rows))
        return _write_rows(cache, src, dst_rows)

    def scatter_rows_pair(self, cache_a, src_a, cache_b, src_b, dst_rows, name=""):
        self.pair.append((src_a, src_b, dst_rows))
        return _write_rows(cache_a, src_a, dst_rows), _write_rows(
            cache_b, src_b, dst_rows
        )


@pytest.fixture
def fakes(monkeypatch):
    k = _FakeKernels()
    monkeypatch.setattr(scatter_utils, "scatter_rows", k.scatter_rows)
    monkeypatch.setattr(scatter_utils, "scatter_rows_pair", k.scatter_rows_pair)
    monkeypatch.setattr(scatter_utils, "padded_num_tokens", _plan_tokens)
    return k


def _specs(nope_layout=KVCacheLayout.SPARSECORE, rope_layout=KVCacheLayout.SPARSECORE):
    return {
        "nope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.NOPE, nope_layout, NUM_PAGES, PAGE_SIZE, LKV_DIM, 4
        ),
        "rope_spec": SparseMLAKVCacheSpec.create(
            KVCacheType.ROPE, rope_layout, NUM_PAGES, PAGE_SIZE, ROPE_DIM, 4
        ),
    }


def _args(specs, num_seqs, tokens_per_seq, num_pad=0):
    """Random latents (raw bytes as fp8) and addressing for one step.

    `num_pad` extra tokens past `query_start_loc[-1]` model a bucketed runner;
    their destination is SKIP_ROW and they must not reach either cache.
    """
    rng = np.random.default_rng(num_seqs * 100 + tokens_per_seq)
    num_tokens = num_seqs * tokens_per_seq
    pages_per_seq = 4
    seq_lens = rng.integers(
        tokens_per_seq, pages_per_seq * PAGE_SIZE + 1, size=num_seqs
    ).astype(np.int32)
    block_tables = rng.permutation(num_seqs * pages_per_seq).astype(np.int32)
    query_start_loc = np.arange(0, num_tokens + 1, tokens_per_seq, dtype=np.int32)

    def fp8(dim, seed):
        u8 = jax.random.randint(
            jax.random.PRNGKey(seed), (num_tokens + num_pad, dim), 0, 256, jnp.uint8
        )
        return jax.lax.bitcast_convert_type(u8, jnp.float8_e4m3fn)

    return (
        jnp.zeros(specs["nope_spec"].shape, specs["nope_spec"].jax_dtype),
        jnp.zeros(specs["rope_spec"].shape, specs["rope_spec"].jax_dtype),
        fp8(LKV_DIM, 1),
        fp8(ROPE_DIM, 2),
        jnp.asarray(seq_lens),
        jnp.asarray(block_tables),
        jnp.asarray(query_start_loc),
    )


def _assert_caches_equal(got, want, what):
    for name, g, w in zip(("nope", "rope"), got, want):
        np.testing.assert_array_equal(
            np.asarray(g), np.asarray(w), err_msg=f"{what}: {name} cache differs"
        )


@pytest.mark.parametrize("fused_pad", ["0", "1"])
@pytest.mark.parametrize("num_seqs,tokens_per_seq,num_pad", [(16, 1, 0), (4, 8, 8)])
def test_pair_gets_the_same_kernel_inputs_as_two_scatters(
    monkeypatch, fakes, fused_pad, num_seqs, tokens_per_seq, num_pad
):
    monkeypatch.setenv("TPU_KV_PACK_FUSED_PAD", fused_pad)
    specs = _specs()
    args = _args(specs, num_seqs, tokens_per_seq, num_pad)

    monkeypatch.setenv("TPU_KV_SCATTER_PAIR", "0")
    off = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    monkeypatch.setenv("TPU_KV_SCATTER_PAIR", "1")
    on = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)

    assert len(fakes.single) == 2 and len(fakes.pair) == 1
    (nope_src, nope_dst), (rope_src, rope_dst) = fakes.single
    pair_nope_src, pair_rope_src, pair_dst = fakes.pair[0]
    for got, want in (
        (pair_nope_src, nope_src),
        (pair_rope_src, rope_src),
        (pair_dst, nope_dst),
        (pair_dst, rope_dst),
    ):
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    _assert_caches_equal(on, off, "pair vs two scatters")
    want = kv_cache_utils.update_sparse_mla_kv_cache_jax(*args, **specs)
    _assert_caches_equal(on, want, "pair vs XLA writer")


@pytest.mark.parametrize("dcp_size,interleave_size", [(2, 4), (8, 16), (16, 4)])
def test_pair_dcp_matches_the_jnp_writer(monkeypatch, fakes, dcp_size, interleave_size):
    """DCP rows that another rank owns become SKIP_ROW; the pair must drop
    them in both caches."""
    monkeypatch.setenv("TPU_KV_PACK_FUSED_PAD", "1")
    monkeypatch.setenv("TPU_KV_SCATTER_PAIR", "1")
    specs = _specs()
    args = _args(specs, 8, 4, num_pad=8)
    kwargs = dict(specs, dcp_size=dcp_size, interleave_size=interleave_size)
    for rank in range(dcp_size):
        got = kv_cache_utils.update_sparse_mla_kv_cache_dcp(
            *args, jnp.int32(rank), **kwargs
        )
        want = kv_cache_utils.update_sparse_mla_kv_cache_dcp_jax(
            *args, jnp.int32(rank), **kwargs
        )
        _assert_caches_equal(got, want, f"rank {rank}")
    assert len(fakes.pair) == dcp_size and not fakes.single


def test_positive_control_detects_a_wrong_rope_row(monkeypatch, fakes):
    """The comparison above must be able to fail: a pair writer that puts the
    rope rows one row off is caught."""
    monkeypatch.setenv("TPU_KV_SCATTER_PAIR", "1")

    def shifted_pair(cache_a, src_a, cache_b, src_b, dst_rows, name=""):
        return _write_rows(cache_a, src_a, dst_rows), _write_rows(
            cache_b, src_b, jnp.where(dst_rows >= 0, dst_rows + 1, dst_rows)
        )

    monkeypatch.setattr(scatter_utils, "scatter_rows_pair", shifted_pair)
    specs = _specs()
    args = _args(specs, 16, 1)
    got = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    want = kv_cache_utils.update_sparse_mla_kv_cache_jax(*args, **specs)
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
    assert not np.array_equal(np.asarray(got[1]), np.asarray(want[1]))


@pytest.mark.parametrize(
    "nope_layout,rope_layout,num_single",
    [
        (KVCacheLayout.TENSORCORE, KVCacheLayout.SPARSECORE, 2),
        (KVCacheLayout.SPARSECORE, KVCacheLayout.TENSORCORE, 1),
        (KVCacheLayout.TENSORCORE, KVCacheLayout.TENSORCORE, 1),
    ],
)
def test_flag_is_ignored_unless_both_layouts_are_sparsecore(
    monkeypatch, fakes, nope_layout, rope_layout, num_single
):
    """Only the dispatch is checked: the jnp model of the kernel does not
    implement the TensorCore-tiled fold, so cache contents are not compared.
    A rope/TENSORCORE cache is written by XLA, not by `scatter_rows`."""
    monkeypatch.setenv("TPU_KV_SCATTER_PAIR", "1")
    specs = _specs(nope_layout, rope_layout)
    kv_cache_utils.update_sparse_mla_kv_cache(*_args(specs, 16, 1), **specs)
    assert not fakes.pair
    assert len(fakes.single) == num_single


def test_scatter_rows_pair_rejects_tensorcore_caches():
    cache = jnp.zeros((16, 512), jnp.uint8)
    rope = jnp.zeros((16, 16), jnp.uint32)
    with pytest.raises(AssertionError, match="uint32 \\(SparseCore\\) caches only"):
        scatter_utils.scatter_rows_pair(
            cache,
            jnp.zeros((8, 512), jnp.uint8),
            rope,
            jnp.zeros((8, 16), jnp.uint32),
            jnp.zeros((8,), jnp.int32),
        )


def test_scatter_pair_env_defaults_to_false(monkeypatch):
    import vllm_torchtpu.envs as envs
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    monkeypatch.delenv("TPU_KV_SCATTER_PAIR", raising=False)
    assert envs.environment_variables["TPU_KV_SCATTER_PAIR"]() is False
    assert "TPU_KV_SCATTER_PAIR" in TpuPlatform.additional_env_vars
