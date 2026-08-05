"""Shape and page-size math for both KV cache layouts.

`HEAD_PACKED` page sizes are pinned by test_kv_cache_dtype_page_size.py. This
file covers `TOKEN_PACKED`: that a narrowed dtype halves the page at every head
count, that bf16 page sizes are unchanged between the layouts except where
HEAD_PACKED was padding a head the rank does not own, and that both layouts
keep `num_blocks` at dim 0.

Pure integer arithmetic; no hardware required.
"""
import pytest
import torch

from vllm_torchtpu.kv_cache_layout import (KVCacheLayout, get_dtype_packing,
                                           kv_cache_page_size_bytes,
                                           kv_cache_shape,
                                           kv_vectors_per_token,
                                           resolve_kv_cache_layout)
from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
INT8 = torch.int8
BLOCK_SIZE = 256

HEAD = KVCacheLayout.HEAD_PACKED
TOKEN = KVCacheLayout.TOKEN_PACKED

# Every (num_kv_heads, head_size) the backend can be asked for. The first four
# are the ones fp8 silently fails on today.
BROKEN_TODAY = [(1, 128), (1, 256), (1, 64), (2, 64)]
WORKS_TODAY = [(2, 128), (4, 128), (8, 128), (2, 256), (8, 256), (4, 64),
               (8, 64)]
ALL_CONFIGS = BROKEN_TODAY + WORKS_TODAY


def page_bytes(layout, num_kv_heads, head_size, dtype, block_size=BLOCK_SIZE):
    return kv_cache_page_size_bytes(layout, block_size, num_kv_heads,
                                    head_size, dtype)


class TestResolverIsPinned:
    """The resolver is constant, so nothing allocates TOKEN_PACKED yet."""

    def test_resolver_returns_head_packed(self):
        assert resolve_kv_cache_layout() is KVCacheLayout.HEAD_PACKED

    @pytest.mark.parametrize("num_kv_heads,head_size", ALL_CONFIGS)
    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_backend_default_matches_head_packed(self, num_kv_heads, head_size,
                                                 dtype):
        """The backend's default path is the head-packed path, exactly."""
        assert PallasAttentionBackend.get_kv_cache_shape(
            7, BLOCK_SIZE, num_kv_heads, head_size,
            dtype) == kv_cache_shape(HEAD, 7, BLOCK_SIZE, num_kv_heads,
                                     head_size, dtype)
        assert PallasAttentionBackend.get_kv_cache_page_size_bytes(
            BLOCK_SIZE, num_kv_heads, head_size,
            dtype) == page_bytes(HEAD, num_kv_heads, head_size, dtype)

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    def test_explicit_layout_kwarg_overrides_the_resolver(self, layout):
        """The kwarg is new public API; only HEAD_PACKED is otherwise reached."""
        assert PallasAttentionBackend.get_kv_cache_shape(
            7, BLOCK_SIZE, 1, 128, FP8,
            layout=layout) == kv_cache_shape(layout, 7, BLOCK_SIZE, 1, 128,
                                             FP8)
        assert PallasAttentionBackend.get_kv_cache_page_size_bytes(
            BLOCK_SIZE, 1, 128, FP8,
            layout=layout) == page_bytes(layout, 1, 128, FP8)


class TestLayoutArgumentIsCoerced:
    """`KVCacheLayout` is a str Enum, so `"head_packed" == HEAD_PACKED` while
    `"head_packed" is HEAD_PACKED` is False. Branching on identity alone would
    send a caller passing the string form to the *other* layout, silently.
    """

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    def test_string_form_matches_the_member(self, layout):
        assert kv_cache_shape(layout.value, 7, BLOCK_SIZE, 1, 128,
                              FP8) == kv_cache_shape(layout, 7, BLOCK_SIZE, 1,
                                                     128, FP8)

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError):
            kv_cache_shape("interleaved", 7, BLOCK_SIZE, 1, 128, FP8)


class TestTokenPackedAlwaysHalves:
    """The property this whole project exists to obtain."""

    @pytest.mark.parametrize("num_kv_heads,head_size", ALL_CONFIGS)
    @pytest.mark.parametrize("narrow", [FP8, FP8_E5M2, INT8])
    def test_narrow_dtype_halves_the_page(self, num_kv_heads, head_size,
                                          narrow):
        wide = page_bytes(TOKEN, num_kv_heads, head_size, BF16)
        assert wide == 2 * page_bytes(TOKEN, num_kv_heads, head_size, narrow)

    @pytest.mark.parametrize("num_kv_heads,head_size", BROKEN_TODAY)
    def test_fixes_exactly_the_configurations_that_are_broken(
            self, num_kv_heads, head_size):
        """These four are `strict=True` xfails in the head-packed suite."""
        assert page_bytes(HEAD, num_kv_heads, head_size,
                          BF16) == page_bytes(HEAD, num_kv_heads, head_size,
                                              FP8)
        assert page_bytes(
            TOKEN, num_kv_heads, head_size,
            BF16) == 2 * (page_bytes(TOKEN, num_kv_heads, head_size, FP8))


class TestBf16IsUnmoved:
    """A layout change that shrinks bf16 would be a different change."""

    @pytest.mark.parametrize("num_kv_heads,head_size", WORKS_TODAY)
    def test_bf16_page_size_is_identical_across_layouts(
            self, num_kv_heads, head_size):
        assert page_bytes(HEAD, num_kv_heads, head_size,
                          BF16) == page_bytes(TOKEN, num_kv_heads, head_size,
                                              BF16)

    @pytest.mark.parametrize("num_kv_heads,head_size", BROKEN_TODAY)
    def test_bf16_shrinks_only_where_it_was_padding_a_phantom_head(
            self, num_kv_heads, head_size):
        """hd64 at 1-2 heads pads bf16 too; token-packing recovers that.

        head_size 128/256 gives Kx2 == 2, which already fills a bf16 word, so
        bf16 is unchanged there. head_size 64 folds K and V into head_dim, so
        Kx2 == num_kv_heads and a single head leaves half a bf16 word empty.
        """
        head_packed = page_bytes(HEAD, num_kv_heads, head_size, BF16)
        token_packed = page_bytes(TOKEN, num_kv_heads, head_size, BF16)
        if head_size == 64 and num_kv_heads == 1:
            assert token_packed * 2 == head_packed
        else:
            assert token_packed == head_packed


class TestShapesAreWhatTheDocSays:
    """Golden values, so a refactor cannot quietly move either layout."""

    @pytest.mark.parametrize(
        "num_kv_heads,head_size,dtype,expected",
        [
            # DeepSeek-R1-Distill-Qwen-7B at TP=4: 1 KV head per rank.
            (1, 128, BF16, (7, 256, 1, 2, 128)),
            (1, 128, FP8, (7, 256, 1, 4, 128)),  # same bytes as bf16
            # Llama-3.1-8B at TP=4: 2 KV heads, already fine.
            (2, 128, BF16, (7, 256, 2, 2, 128)),
            (2, 128, FP8, (7, 256, 1, 4, 128)),
            # hd64 folds K and V along head_dim, so Kx2 == num_kv_heads.
            (2, 64, FP8, (7, 256, 1, 4, 128)),
            (8, 64, FP8, (7, 256, 2, 4, 128)),
        ],
    )
    def test_head_packed_golden(self, num_kv_heads, head_size, dtype,
                                expected):
        assert kv_cache_shape(HEAD, 7, 256, num_kv_heads, head_size,
                              dtype) == expected

    @pytest.mark.parametrize(
        "num_kv_heads,head_size,dtype,expected",
        [
            # [num_blocks, Kx2, block_size / kv_packing, kv_packing, head_dim]
            (1, 128, BF16, (7, 2, 128, 2, 128)),
            (1, 128, FP8, (7, 2, 64, 4, 128)),  # half of bf16
            (2, 128, BF16, (7, 4, 128, 2, 128)),
            (2, 128, FP8, (7, 4, 64, 4, 128)),
            (2, 64, FP8, (7, 2, 64, 4, 128)),
            (8, 64, FP8, (7, 8, 64, 4, 128)),
        ],
    )
    def test_token_packed_golden(self, num_kv_heads, head_size, dtype,
                                 expected):
        assert kv_cache_shape(TOKEN, 7, 256, num_kv_heads, head_size,
                              dtype) == expected

    @pytest.mark.parametrize("layout", [HEAD, TOKEN])
    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_num_blocks_stays_at_dim_zero(self, layout, dtype):
        """vLLM's offloading worker finds the block dim with shape.index()."""
        shape = kv_cache_shape(layout, 1234, BLOCK_SIZE, 8, 128, dtype)
        assert shape.index(1234) == 0


class TestTokenPackedByteFormula:

    @pytest.mark.parametrize("num_kv_heads,head_size", ALL_CONFIGS)
    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_bytes_are_exact_with_no_padding(self, num_kv_heads, head_size,
                                             dtype):
        """Kx2 * block_size * head_dim * itemsize, nothing rounded up."""
        expected = (kv_vectors_per_token(num_kv_heads, head_size) *
                    BLOCK_SIZE * (128 if head_size <= 128 else 256) *
                    dtype.itemsize)
        assert page_bytes(TOKEN, num_kv_heads, head_size, dtype) == expected

    @pytest.mark.parametrize("block_size", [16, 32, 128, 256])
    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_bytes_per_token_independent_of_block_size(self, block_size,
                                                       dtype):
        per_token = page_bytes(TOKEN, 8, 128, dtype, block_size) / block_size
        reference = page_bytes(TOKEN, 8, 128, dtype, BLOCK_SIZE) / BLOCK_SIZE
        assert per_token == reference

    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_dims_three_and_four_multiply_back_to_block_size(self, dtype):
        """They are the token axis regrouped, not extra space."""
        _, _, groups, per_word, _ = kv_cache_shape(TOKEN, 7, BLOCK_SIZE, 8,
                                                   128, dtype)
        assert groups * per_word == BLOCK_SIZE
        assert per_word == get_dtype_packing(dtype)


class TestIndivisibleBlockSizeIsRejected:
    """block_size=1 is a live call path, and it cannot mean a page here."""

    @pytest.mark.parametrize("block_size", [1, 2, 6])
    def test_token_packed_raises(self, block_size):
        with pytest.raises(ValueError, match="divisible"):
            kv_cache_shape(TOKEN, 7, block_size, 8, 128, FP8)

    @pytest.mark.parametrize("block_size", [1, 2, 6])
    def test_head_packed_still_accepts_it(self, block_size):
        """platforms/tpu_block_size_utils.py asks for block_size=1 today."""
        assert kv_cache_shape(HEAD, 7, block_size, 8, 128,
                              FP8)[1] == block_size
