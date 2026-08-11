"""Token-packed KV cache layout: converters and reference oracle.

Three properties, in increasing order of what they buy:

1. **Shape agreement** -- the torch-side allocator shape and the JAX-side kernel
   shape are identical for every configuration. An allocator and a kernel that
   disagree produce wrong attention rather than an error, so nothing else here
   means much without this.
2. **Round-trip** -- the converters are inverses, so a cache built by today's
   trusted code can be handed to the new layout and back.
3. **Differential** -- `ref_ragged_paged_attention` returns identical outputs
   under both layouts on logically identical inputs. This is the executable
   spec PRs 3 and 4 debug their kernels against.

Pure JAX, no Pallas, so this runs on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import kv_cache_layout as torch_layout
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel as K
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel_hd64 as K64
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import layout as L
from vllm_torchtpu.kv_cache_layout import KVCacheLayout

HEAD = KVCacheLayout.HEAD_PACKED
TOKEN = KVCacheLayout.TOKEN_PACKED

# fp32 gives kv_packing == 1, where token-packing is a no-op. Cheap insurance
# that nothing assumes kv_packing > 1.
DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float32]
NUM_KV_HEADS = [1, 2, 3, 4, 8]
HEAD_DIMS = [64, 128, 256]
PAGE_SIZES = [16, 256]

_TORCH_EQUIVALENT = {
    jnp.bfloat16: torch.bfloat16,
    jnp.float8_e4m3fn: torch.float8_e4m3fn,
    jnp.float8_e5m2: torch.float8_e5m2,
    jnp.float32: torch.float32,
}


@pytest.fixture(autouse=True)
def _run_on_cpu():
    """These are correctness checks, not kernels; keep them off the TPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


class TestTorchAndJaxAgreeOnShape:
    """The allocator (torch) and the kernels (JAX) must build the same array."""

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    @pytest.mark.parametrize("page_size", PAGE_SIZES)
    @pytest.mark.parametrize("head_dim", [128, 256])
    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_v3_matches_the_torch_allocator(self, layout, page_size, head_dim,
                                            num_kv_heads, dtype):
        jax_shape = K.get_kv_cache_shape(7,
                                         page_size,
                                         num_kv_heads,
                                         head_dim,
                                         dtype,
                                         layout=layout)
        torch_shape = torch_layout.kv_cache_shape(layout, 7, page_size,
                                                  num_kv_heads, head_dim,
                                                  _TORCH_EQUIVALENT[dtype])
        assert jax_shape == torch_shape

    @pytest.mark.parametrize("page_size", PAGE_SIZES)
    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_hd64_matches_the_torch_allocator(self, page_size, num_kv_heads,
                                              dtype):
        """head_dim 64 folds K and V along head_dim, so Kx2 == num_kv_heads.

        The torch side encodes that rule in `kv_vectors_per_token`; hd64 is the
        kernel it has to agree with, not v3.
        """
        jax_shape = K64.get_kv_cache_shape(7, page_size, num_kv_heads, 64,
                                           dtype)
        torch_shape = torch_layout.kv_cache_shape(HEAD, 7, page_size,
                                                  num_kv_heads, 64,
                                                  _TORCH_EQUIVALENT[dtype])
        assert jax_shape == torch_shape


class TestTokenPackedShape:

    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_page_bytes_halve_versus_bf16(self, num_kv_heads, dtype):
        """The point of the exercise: no rounding, so narrowing always pays."""

        def page_elems(dt):
            shape = L.kv_cache_shape(TOKEN, 1, 256, num_kv_heads * 2, 128, dt)
            return int(np.prod(shape)) * jnp.dtype(dt).itemsize

        expected = jnp.dtype(jnp.bfloat16).itemsize / jnp.dtype(dtype).itemsize
        assert page_elems(jnp.bfloat16) / page_elems(dtype) == expected

    @pytest.mark.parametrize("page_size", [1, 2, 6])
    def test_indivisible_page_size_is_refused(self, page_size):
        with pytest.raises(ValueError, match="divisible"):
            L.kv_cache_shape(TOKEN, 1, page_size, 4, 128, jnp.float8_e4m3fn)

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_describe_cache_recovers_the_logical_quantities(
            self, layout, dtype):
        shape = L.kv_cache_shape(layout, 7, 256, 6, 128, dtype)
        cache = jnp.zeros(shape, dtype=dtype)
        pages, page_size, kx2, packing, head_dim = L.describe_cache(
            cache, layout)
        assert (pages, page_size, head_dim) == (7, 256, 128)
        assert packing == L.get_dtype_packing(dtype)
        # Head-packing rounds Kx2 up to a whole word; token-packing does not.
        assert kx2 == (L.align_to(6, packing) if layout is HEAD else 6)


class TestConvertersRoundTrip:

    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
    def test_head_to_token_and_back_is_lossless(self, num_kv_heads, dtype):
        shape = K.get_kv_cache_shape(5, 64, num_kv_heads, 128, dtype)
        cache = jax.random.normal(jax.random.key(0), shape,
                                  dtype=jnp.float32).astype(dtype)
        again = L.token_packed_to_head_packed(
            L.head_packed_to_token_packed(cache))
        assert again.shape == cache.shape
        np.testing.assert_array_equal(np.asarray(again), np.asarray(cache))

    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    def test_trimming_padding_keeps_the_real_heads(self, num_kv_heads):
        """Dropping head-packing's padding must not disturb real data."""
        dtype = jnp.float8_e4m3fn
        shape = K.get_kv_cache_shape(5, 64, num_kv_heads, 128, dtype)
        cache = jax.random.normal(jax.random.key(1), shape,
                                  dtype=jnp.float32).astype(dtype)
        trimmed = L.head_packed_to_token_packed(
            cache, actual_num_kv_heads_x2=num_kv_heads * 2)
        assert trimmed.shape[1] == num_kv_heads * 2

        before = L.to_token_major(cache, HEAD)[:, :num_kv_heads * 2, :]
        after = L.to_token_major(trimmed, TOKEN)
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

    def test_trimming_beyond_what_is_stored_is_refused(self):
        dtype = jnp.bfloat16
        cache = jnp.zeros(K.get_kv_cache_shape(2, 32, 2, 128, dtype), dtype)
        with pytest.raises(ValueError, match="exceeds"):
            L.head_packed_to_token_packed(cache, actual_num_kv_heads_x2=99)

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_token_major_is_invertible(self, layout, dtype):
        shape = L.kv_cache_shape(layout, 3, 32, 4, 128, dtype)
        cache = jax.random.normal(jax.random.key(2), shape,
                                  dtype=jnp.float32).astype(dtype)
        tokens = L.to_token_major(cache, layout)
        assert tokens.shape[0] == 3 * 32
        back = L.from_token_major(tokens, layout, shape)
        np.testing.assert_array_equal(np.asarray(back), np.asarray(cache))

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    def test_token_row_order_is_page_major(self, layout):
        """Token t of page i must land at row i * page_size + t."""
        page_size, kx2, head_dim, pages = 8, 4, 128, 3
        dtype = jnp.float32  # packing 1, so values map one-to-one
        marks = jnp.arange(pages * page_size,
                           dtype=jnp.float32).reshape(pages * page_size, 1, 1)
        tokens = jnp.broadcast_to(
            marks, (pages * page_size, kx2, head_dim)).astype(dtype)
        shape = L.kv_cache_shape(layout, pages, page_size, kx2, head_dim,
                                 dtype)
        cache = L.from_token_major(tokens, layout, shape)
        recovered = L.to_token_major(cache, layout)
        np.testing.assert_array_equal(np.asarray(recovered),
                                      np.asarray(tokens))


def _make_case(rng, *, num_kv_heads, q_heads_per_kv, head_dim, page_size,
               dtype, seq_specs):
    """Build one ragged batch. `seq_specs` is a list of (kv_len, q_len)."""
    max_num_seqs = len(seq_specs)
    q_lens = [q for _, q in seq_specs]
    kv_lens = [kv for kv, _ in seq_specs]
    cu_q_lens = np.zeros(max_num_seqs + 1, dtype=np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)
    total_q = int(cu_q_lens[-1])

    pages_per_seq = max((kv + page_size - 1) // page_size for kv in kv_lens)
    total_pages = max_num_seqs * pages_per_seq + 1
    # Deliberately not identity: a layout bug that ignores the block table
    # would still pass against sequential pages.
    page_indices = np.arange(1,
                             max_num_seqs * pages_per_seq + 1,
                             dtype=np.int32)

    def rand(shape):
        return jax.random.normal(rng(), shape, dtype=jnp.float32).astype(dtype)

    return dict(
        queries=rand((total_q, num_kv_heads * q_heads_per_kv, head_dim)),
        keys=rand((total_q, num_kv_heads, head_dim)),
        values=rand((total_q, num_kv_heads, head_dim)),
        kv_lens=jnp.asarray(kv_lens, dtype=jnp.int32),
        page_indices=jnp.asarray(page_indices, dtype=jnp.int32),
        cu_q_lens=jnp.asarray(cu_q_lens, dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, max_num_seqs], dtype=jnp.int32),
        _total_pages=total_pages,
    )


# (name, seq_specs) — ragged shapes the kernel actually sees.
BATCH_PATTERNS = [
    ("decode_only", [(37, 1), (129, 1), (8, 1)]),
    ("prefill_only", [(48, 48), (33, 33)]),
    ("mixed", [(64, 1), (100, 40), (17, 17), (255, 1)]),
    ("boundary_aligned", [(64, 16), (128, 64)]),
    ("single_token_seq", [(1, 1)]),
]


class TestOracleAgreesAcrossLayouts:
    """The differential test PRs 3 and 4 will debug against."""

    @pytest.mark.parametrize("pattern_name,seq_specs", BATCH_PATTERNS)
    @pytest.mark.parametrize("page_size", PAGE_SIZES)
    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_same_outputs_under_both_layouts(self, pattern_name, seq_specs,
                                             page_size, num_kv_heads, dtype):
        counter = [0]

        def rng():
            counter[0] += 1
            return jax.random.key(counter[0])

        case = _make_case(rng,
                          num_kv_heads=num_kv_heads,
                          q_heads_per_kv=2,
                          head_dim=128,
                          page_size=page_size,
                          dtype=dtype,
                          seq_specs=seq_specs)
        total_pages = case.pop("_total_pages")

        head_shape = K.get_kv_cache_shape(total_pages, page_size, num_kv_heads,
                                          128, dtype)
        head_cache = jax.random.normal(jax.random.key(99),
                                       head_shape,
                                       dtype=jnp.float32).astype(dtype)
        # Same contents, other layout. Padding rows are dropped because
        # token-packing does not store them.
        token_cache = L.head_packed_to_token_packed(
            head_cache, actual_num_kv_heads_x2=num_kv_heads * 2)

        out_head, cache_head = K.ref_ragged_paged_attention(
            kv_cache=head_cache, layout=HEAD, **case)
        out_token, cache_token = K.ref_ragged_paged_attention(
            kv_cache=token_cache, layout=TOKEN, **case)

        np.testing.assert_array_equal(np.asarray(out_token),
                                      np.asarray(out_head))
        # The written-back caches must agree too, or PR 4 would be debugging
        # against an oracle that only checks outputs.
        np.testing.assert_array_equal(
            np.asarray(L.to_token_major(cache_token, TOKEN)),
            np.asarray(
                L.to_token_major(cache_head, HEAD)[:, :num_kv_heads * 2, :]))

    @pytest.mark.parametrize("head_dim", HEAD_DIMS)
    def test_head_dims_including_unaligned(self, head_dim):
        """head_dim 64 pads to 128 lanes; 256 spans two."""
        if head_dim == 64:
            pytest.skip("head_dim 64 is the hd64 kernel's layout, not v3's")
        counter = [0]

        def rng():
            counter[0] += 1
            return jax.random.key(100 + counter[0])

        dtype = jnp.bfloat16
        case = _make_case(rng,
                          num_kv_heads=2,
                          q_heads_per_kv=4,
                          head_dim=head_dim,
                          page_size=32,
                          dtype=dtype,
                          seq_specs=[(70, 5), (33, 33)])
        total_pages = case.pop("_total_pages")

        head_cache = jnp.zeros(
            K.get_kv_cache_shape(total_pages, 32, 2, head_dim, dtype), dtype)
        token_cache = L.head_packed_to_token_packed(head_cache,
                                                    actual_num_kv_heads_x2=4)

        out_head, _ = K.ref_ragged_paged_attention(kv_cache=head_cache,
                                                   layout=HEAD,
                                                   **case)
        out_token, _ = K.ref_ragged_paged_attention(kv_cache=token_cache,
                                                    layout=TOKEN,
                                                    **case)
        np.testing.assert_array_equal(np.asarray(out_token),
                                      np.asarray(out_head))

    def test_skip_kv_update_agrees(self):
        """KV-sharing layers read the cache without writing it back."""
        counter = [0]

        def rng():
            counter[0] += 1
            return jax.random.key(200 + counter[0])

        dtype = jnp.bfloat16
        case = _make_case(rng,
                          num_kv_heads=2,
                          q_heads_per_kv=2,
                          head_dim=128,
                          page_size=32,
                          dtype=dtype,
                          seq_specs=[(64, 8), (20, 20)])
        total_pages = case.pop("_total_pages")

        head_cache = jax.random.normal(jax.random.key(7),
                                       K.get_kv_cache_shape(
                                           total_pages, 32, 2, 128, dtype),
                                       dtype=jnp.float32).astype(dtype)
        token_cache = L.head_packed_to_token_packed(head_cache,
                                                    actual_num_kv_heads_x2=4)

        out_head, _ = K.ref_ragged_paged_attention(kv_cache=head_cache,
                                                   layout=HEAD,
                                                   skip_kv_update=True,
                                                   **case)
        out_token, _ = K.ref_ragged_paged_attention(kv_cache=token_cache,
                                                    layout=TOKEN,
                                                    skip_kv_update=True,
                                                    **case)
        np.testing.assert_array_equal(np.asarray(out_token),
                                      np.asarray(out_head))


class TestDefaultsAreUnchanged:
    """PR 2 must not move today's behaviour."""

    @pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_shape_default_is_head_packed(self, num_kv_heads, dtype):
        assert K.get_kv_cache_shape(4, 64, num_kv_heads, 128,
                                    dtype) == K.get_kv_cache_shape(
                                        4,
                                        64,
                                        num_kv_heads,
                                        128,
                                        dtype,
                                        layout=HEAD)
