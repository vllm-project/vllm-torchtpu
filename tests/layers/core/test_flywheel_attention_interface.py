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
"""flywheel_attention_interface: the host's per-step kernel choice and KV
write table, and, on a TPU, the sharded kernels against numpy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flywheel_tpu.pallas.copy_utils import merged_cache_heads

from vllm_torchtpu.layers.core import flywheel_attention_interface as fa


def naive_kv_write_slices(cu_q_lens, kv_lens, block_table, page_size):
    """Token by token: (cache_row, new_row, length), merging runs that stay in
    one page and are contiguous on both sides."""
    pieces = []
    for req, kv_len in enumerate(kv_lens):
        q_len = cu_q_lens[req + 1] - cu_q_lens[req]
        for t in range(q_len):
            pos = kv_len - q_len + t
            cache_row = block_table[req, pos // page_size] * page_size + pos % page_size
            new_row = cu_q_lens[req] + t
            if (
                pieces
                and t > 0
                and pos % page_size != 0
                and pieces[-1][0] + pieces[-1][2] == cache_row
                and pieces[-1][1] + pieces[-1][2] == new_row
            ):
                pieces[-1][2] += 1
            else:
                pieces.append([cache_row, new_row, 1])
    return pieces


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("seed", range(5))
def test_kv_write_slices_match_a_token_by_token_reference(page_size, seed):
    rng = np.random.default_rng(seed)
    num_reqs, pages_per_seq = 6, 8
    q_lens = rng.integers(0, 3 * page_size, num_reqs)
    q_lens[:2] = 1
    kv_lens = q_lens + rng.integers(0, 4 * page_size, num_reqs)
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    block_table = (
        rng.permutation(num_reqs * pages_per_seq)
        .reshape(num_reqs, pages_per_seq)
        .astype(np.int32)
    )
    bound = fa.kv_write_slice_bound(int(cu_q_lens[-1]), num_reqs, page_size)
    out = np.full((3, bound), -1, np.int32)

    count = fa.build_kv_write_slices(
        cu_q_lens, kv_lens.astype(np.int32), block_table, page_size=page_size, out=out
    )

    expected = naive_kv_write_slices(
        cu_q_lens.tolist(), kv_lens.tolist(), block_table, page_size
    )
    assert count == len(expected) <= bound
    np.testing.assert_array_equal(out[:, :count].T, np.array(expected))
    np.testing.assert_array_equal(out[:, count:], 0)


@pytest.mark.parametrize(
    ("num_decode", "seq_lens", "phase"),
    [
        (2, [(1, 300), (1, 129)], "decode"),
        (0, [(300, 300), (129, 129)], "prefill"),
        (1, [(1, 300), (129, 129)], "extend"),
        (0, [(200, 457)], "extend"),
    ],
)
def test_attention_phase(num_decode, seq_lens, phase):
    q_lens = [q_len for q_len, _ in seq_lens]
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)])
    kv_lens = np.array([kv_len for _, kv_len in seq_lens])

    assert fa.select_attention_phase(num_decode, cu_q_lens, kv_lens) == phase


def test_attention_phase_needs_decode_rows_first():
    with pytest.raises(ValueError, match="decode requests first"):
        fa.select_attention_phase(1, np.array([0, 5, 6]), np.array([5, 9]))


@pytest.mark.parametrize(
    ("scheduled_tokens", "bucket", "dense"),
    [
        ([1000], 1024, True),
        ([100], 1024, True),
        ([500, 500], 1024, False),
        # The dense kernel cannot tile these buckets.
        ([64], 64, False),
        ([3000], 3072, False),
    ],
)
def test_dense_prefill_for_one_request(scheduled_tokens, bucket, dense):
    span = fa.varlen_block_span([scheduled_tokens], bucket)

    assert fa.use_dense_prefill([scheduled_tokens], bucket, span) is dense


@pytest.mark.parametrize(
    ("scheduled_tokens", "bucket", "span"),
    [([128] * 16, 2048, 256), ([2048], 2048, 1024), ([100], 128, 128)],
)
def test_varlen_block_span(scheduled_tokens, bucket, span):
    assert fa.varlen_block_span([scheduled_tokens], bucket) == span
    assert span in fa.varlen_block_spans(bucket)


def test_varlen_block_span_tie_goes_to_the_wider_span(monkeypatch):
    monkeypatch.setattr(fa, "ragged_step_cost", lambda *args: 1)

    assert fa.varlen_block_span([[1000]], 2048) == 2048


@pytest.mark.parametrize(
    ("num_decode", "q_lens", "kv_lens", "bucket", "kernel"),
    [
        (2, [1, 1], [10, 300], 128, "decode"),
        (1, [1, 40], [10, 300], 128, "extend"),
        (0, [200], [200], 256, "dense"),
        (0, [100, 100], [100, 100], 256, 256),
    ],
)
def test_step_kernel_index(num_decode, q_lens, kv_lens, bucket, kernel):
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

    index = fa.step_kernel_index(
        num_decode, cu_q_lens, np.array(kv_lens, np.int32), bucket
    )

    assert fa.step_kernels(bucket)[index] == kernel


def test_flywheel_gdn_reads_each_state_from_the_slot_it_writes(monkeypatch):
    calls = []

    def fake_gdn(*args, **kwargs):
        calls.append((args, kwargs))
        return (args[3], args[4]), "out"

    monkeypatch.setattr(fa.gdn_wrapper, "fused_conv1d_gdn", fake_gdn)
    args = [object() for _ in range(12)]
    conv = np.arange(2 * 3 * 1 * 8).reshape(2, 3, 1, 8)  # [slots, k - 1, 1, dim]
    recurrent = np.zeros((2, 1, 4, 4))
    args[3] = (conv, recurrent)
    read_offsets = object()

    (new_conv, new_recurrent), out = fa.flywheel_conv1d_gdn(*args, read_offsets, n_kq=2)

    ((passed, kwargs),) = calls
    state_indices = args[9]
    assert passed[3].shape == (2, 3, 8) and passed[4] is recurrent
    assert passed[13:] == (state_indices, read_offsets)
    assert kwargs == {"n_kq": 2}
    assert new_conv.shape == conv.shape and new_recurrent is recurrent
    assert out == "out"


# --------------------------------------------------------------------------
# The sharded kernels on a TPU, against numpy.
# --------------------------------------------------------------------------

PAGE_SIZE = 128
HEAD_DIM = 128
# The largest token bucket, which sizes every step's KV write table.
MAX_NUM_TOKENS = 256
# Untouched cache rows and padding K/V rows hold these, so a stray write by
# the writer or the decode kernel's append shows up in the cache.
CACHE_SENTINEL, PADDING_SENTINEL = 0.25, 0.75


@pytest.fixture
def tpu():
    # In a fixture, not at import: collection must not claim the TPU.
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


def _bf16(x):
    return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))


def _reference(queries, keys, values, q_len):
    """Causal attention of a request's q_len last positions over all its keys,
    with KV head n // (N / H) per query head n."""
    group = queries.shape[1] // keys.shape[1]
    k = np.repeat(keys, group, axis=1)
    v = np.repeat(values, group, axis=1)
    kv_len = keys.shape[0]
    scores = np.einsum("tnd,snd->nts", queries, k) * HEAD_DIM**-0.5
    q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
    scores = np.where(np.arange(kv_len)[None, :] <= q_pos, scores, -np.inf)
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    return np.einsum("nts,snd->tnd", probs, v)


class _Engine:
    """Requests on their own pages of one FlyWheel cache, [pages, page, 2 * H,
    D] with K and V interleaved per KV head, served step by step the way the
    runner plans and calls flywheel_kernel_entry. With kv_scale, the cache is
    fp8 and K/V reach the kernel quantized by it, as RPA's forward hands them
    over; the expected cache is then not tracked."""

    def __init__(
        self, pages, *, num_heads, num_kv_heads, max_num_seqs, seed=0, kv_scale=None
    ):
        rng = np.random.default_rng(seed)
        self.pages = pages  # request -> its page ids
        self.num_kv_heads, self.max_num_seqs = num_kv_heads, max_num_seqs
        self.kv_scale = kv_scale
        self.pages_per_seq = max(len(p) for p in pages.values())
        num_pages = 1 + max(max(p) for p in pages.values())
        self.streams = {
            r: tuple(
                _bf16(rng.standard_normal((len(p) * PAGE_SIZE, heads, HEAD_DIM)))
                for heads in (num_kv_heads, num_kv_heads, num_heads)
            )
            for r, p in pages.items()
        }
        if kv_scale is None:
            self.cache = np.full(
                (num_pages, PAGE_SIZE, 2 * num_kv_heads, HEAD_DIM),
                CACHE_SENTINEL,
                np.float32,
            )
            self.expected_cache = self.cache.copy()
            return
        self.cache = jnp.zeros(
            (
                num_pages,
                PAGE_SIZE,
                merged_cache_heads(num_kv_heads, jnp.float8_e4m3fn),
                HEAD_DIM,
            ),
            jnp.float8_e4m3fn,
        )
        self.expected_cache = None
        # K/V as the fp8 cache holds them, so the reference sees what it stores.
        self.streams = {
            r: (self._fp8(keys) * kv_scale, self._fp8(values) * kv_scale, queries)
            for r, (keys, values, queries) in self.streams.items()
        }

    def _fp8(self, x):
        """x / kv_scale in fp8, back as float32."""
        return np.asarray(
            jnp.asarray(x / self.kv_scale).astype(jnp.float8_e4m3fn).astype(jnp.float32)
        )

    def preload(self, request, num_tokens):
        """Cache a request's first num_tokens tokens, without a kernel."""
        for cache in (self.cache, self.expected_cache):
            self._write(cache, request, range(num_tokens))

    def _write(self, cache, request, positions):
        keys, values, _ = self.streams[request]
        for pos in positions:
            page = self.pages[request][pos // PAGE_SIZE]
            cache[page, pos % PAGE_SIZE, 0::2] = keys[pos]
            cache[page, pos % PAGE_SIZE, 1::2] = values[pos]

    def step(self, bucket, num_decode, rows):
        """rows = [(request, q_len, kv_len)], decode rows first. Returns the
        output and reference rows of the scheduled tokens."""
        num_heads = self.streams[rows[0][0]][2].shape[1]
        q = np.zeros((bucket, num_heads, HEAD_DIM), np.float32)
        k = np.full((bucket, self.num_kv_heads, HEAD_DIM), PADDING_SENTINEL, np.float32)
        v = k.copy()
        reference = np.zeros_like(q)
        cu_q_lens = np.zeros(self.max_num_seqs + 1, np.int32)
        # Stale values past the scheduled requests, as the runner leaves them.
        kv_lens = np.full(self.max_num_seqs, 33, np.int32)
        block_table = np.tile(
            np.array(self.pages[rows[0][0]][:1] * self.pages_per_seq, np.int32),
            (self.max_num_seqs, 1),
        )
        for row, (r, q_len, kv_len) in enumerate(rows):
            start = cu_q_lens[row]
            cu_q_lens[row + 1 :] = start + q_len
            kv_lens[row] = kv_len
            block_table[row] = 0
            block_table[row, : len(self.pages[r])] = self.pages[r]
            keys, values, queries = self.streams[r]
            new = slice(kv_len - q_len, kv_len)
            q[start : start + q_len] = queries[new]
            k[start : start + q_len] = keys[new]
            v[start : start + q_len] = values[new]
            reference[start : start + q_len] = _reference(
                queries[new], keys[:kv_len], values[:kv_len], q_len
            )
            if self.expected_cache is not None:
                self._write(self.expected_cache, r, range(kv_len - q_len, kv_len))
        num_reqs = len(rows)
        host_cu, host_kv = cu_q_lens[: num_reqs + 1], kv_lens[:num_reqs]
        slices = np.zeros(
            (3, fa.kv_write_slice_bound(MAX_NUM_TOKENS, self.max_num_seqs, PAGE_SIZE)),
            np.int32,
        )
        num_slices = fa.build_kv_write_slices(
            host_cu[num_decode:],
            host_kv[num_decode:],
            block_table[num_decode:num_reqs],
            page_size=PAGE_SIZE,
            out=slices,
        )
        kernel_index = fa.step_kernel_index(num_decode, host_cu, host_kv, bucket)
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:1]), (fa.ShardingAxisName.ATTN_HEAD,)
        )
        if self.kv_scale is None:
            cache = jnp.asarray(self.cache, jnp.bfloat16)
            k, v = jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.bfloat16)
        else:
            cache = self.cache
            k, v = (
                jnp.asarray(x / self.kv_scale).astype(jnp.float8_e4m3fn) for x in (k, v)
            )
        new_cache, output = fa.flywheel_kernel_entry(
            cache,
            jnp.asarray(q, jnp.bfloat16),
            k,
            v,
            jnp.asarray(kv_lens),
            jnp.asarray(block_table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray([num_decode, num_decode, num_reqs], jnp.int32),
            None,
            jnp.asarray(slices.reshape(-1)),
            jnp.asarray([num_slices], jnp.int32),
            jnp.asarray([kernel_index], jnp.int32),
            mesh=mesh,
            sliding_window=None,
            sm_scale=HEAD_DIM**-0.5,
            soft_cap=None,
            q_scale=None,
            k_scale=self.kv_scale,
            v_scale=self.kv_scale,
            skip_kv_update=False,
            use_causal_mask=True,
        )
        self.cache = (
            np.asarray(new_cache.astype(jnp.float32))
            if self.kv_scale is None
            else new_cache
        )
        num_tokens = int(cu_q_lens[num_reqs])
        return np.asarray(output.astype(jnp.float32))[:num_tokens], reference[
            :num_tokens
        ]


def _assert_rows_close(output, reference, what, tol=3e-2):
    bad = np.nonzero(
        ~np.isclose(output, reference, atol=tol, rtol=tol).all(axis=(1, 2))
    )[0]
    assert not bad.size, f"{what}: wrong rows {bad.tolist()}"


def _small_engine(kv_lens, *, num_kv_heads=2, kv_scale=None):
    pages = {r: list(range(1 + 4 * r, 5 + 4 * r)) for r in range(len(kv_lens))}
    return _Engine(
        pages, num_heads=4, num_kv_heads=num_kv_heads, max_num_seqs=4, kv_scale=kv_scale
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    ("steps", "num_decode", "bucket"),
    [
        pytest.param([(1, 1), (1, 130), (1, 257)], 3, 128, id="decode"),
        pytest.param([(200, 200)], 0, 256, id="dense_prefill"),
        pytest.param([(100, 100), (60, 60)], 0, 256, id="ragged_prefill"),
        pytest.param([(1, 140), (70, 300), (5, 5)], 1, 128, id="extend"),
    ],
)
def test_one_step_on_tpu(tpu, steps, num_decode, bucket):
    engine = _small_engine([kv_len for _, kv_len in steps])
    for r, (q_len, kv_len) in enumerate(steps):
        engine.preload(r, kv_len - q_len)

    output, reference = engine.step(
        bucket,
        num_decode,
        [(r, q_len, kv_len) for r, (q_len, kv_len) in enumerate(steps)],
    )

    _assert_rows_close(output, reference, "output")
    np.testing.assert_array_equal(engine.cache, engine.expected_cache)


@pytest.mark.tpu_test
def test_decode_in_a_small_bucket_reads_what_a_large_bucket_prefill_wrote(tpu):
    # Write tables are sized by MAX_NUM_TOKENS in every bucket, as the
    # runner's are: a 256-token prefill, then decodes in the 16-token bucket.
    engine = _small_engine([153, 93])
    _assert_rows_close(*engine.step(256, 0, [(0, 150, 150), (1, 90, 90)]), "prefill")
    for i in range(1, 4):
        output, reference = engine.step(16, 2, [(0, 1, 150 + i), (1, 1, 90 + i)])
        _assert_rows_close(output, reference, f"decode {i}")
    np.testing.assert_array_equal(engine.cache, engine.expected_cache)


@pytest.mark.tpu_test
@pytest.mark.parametrize("num_kv_heads", [2, 1])  # 1: the pool pads a zero head
def test_fp8_decode_and_extend_read_what_the_fp8_prefill_wrote(tpu, num_kv_heads):
    engine = _small_engine([153, 93], num_kv_heads=num_kv_heads, kv_scale=0.5)
    output, reference = engine.step(256, 0, [(0, 150, 150), (1, 90, 90)])
    _assert_rows_close(output, reference, "prefill", tol=6e-2)
    for i in range(1, 4):
        output, reference = engine.step(16, 2, [(0, 1, 150 + i), (1, 1, 90 + i)])
        _assert_rows_close(output, reference, f"decode {i}", tol=6e-2)
    # The extend kernel reads request 1's fp8 prefix through the descales.
    output, reference = engine.step(128, 1, [(0, 1, 154), (1, 70, 163)])
    _assert_rows_close(output, reference, "extend", tol=6e-2)
