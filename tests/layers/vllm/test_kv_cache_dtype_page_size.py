"""Page-size accounting for the Pallas KV cache layout.

`get_kv_cache_shape` rounds up twice, and either can waste HBM:

    page = block_size x ceil(Kx2 / packing) * packing  x  cdiv(head_size, 128) * 128
                        \\_____ packing dimension ____/     \\______ lanes ______/

`packing` is `32 / bitwidth` (2 for bf16, 4 for fp8), so a narrower dtype makes
the first round-up bite sooner. `Kx2` is `num_kv_heads * 2`, except under the
hd64 kernel (`head_size == 64`), which folds K and V along head_dim — which is
why bf16 can be padded too. The second round-up is dtype-independent: any
head_size neither 64 nor a multiple of 128 leaves lanes empty.
"""
import contextlib
import logging

import pytest
import torch

from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
from vllm_torchtpu.runner.tpu_runner import _warn_if_kv_cache_is_padded

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
BLOCK_SIZE = 256

# (num_kv_heads, head_size, dtype) -> is the page padded today?
PADDING_CASES = [
    # Packing-dimension padding. hd64 folds K and V, so Kx2 == num_kv_heads
    # and an odd count pads even bf16.
    (1, 64, BF16, True),
    (3, 64, BF16, True),
    (2, 64, BF16, False),
    (4, 64, BF16, False),
    # head_size 128/256 gives an even Kx2, which always fills a bf16 word.
    (1, 128, BF16, False),
    (8, 128, BF16, False),
    # fp8 needs Kx2 divisible by 4, so it pads wherever the head count is odd
    # or too small.
    (1, 128, FP8, True),
    (3, 128, FP8, True),
    (1, 256, FP8, True),
    (1, 64, FP8, True),
    (2, 64, FP8, True),
    (2, 128, FP8, False),
    (8, 128, FP8, False),
    (4, 64, FP8, False),
    # Lane padding, independent of the packing dimension. Every row here fills
    # its 32-bit word exactly, so packing is not what pads them.
    (1, 80, BF16, True),
    (2, 80, BF16, True),
    (1, 96, BF16, True),
    (2, 96, FP8, True),
    (2, 192, BF16, True),
]

# fp8 must halve the page against bf16 wherever the head count fills a word.
FP8_HALVES = [(2, 128), (4, 128), (8, 128), (2, 256), (8, 256), (4, 64),
              (8, 64)]
FP8_SHOULD_HALVE = [(1, 128), (1, 256), (1, 64), (2, 64)]


def page_bytes(num_kv_heads, head_size, dtype):
    return PallasAttentionBackend.get_kv_cache_page_size_bytes(
        BLOCK_SIZE, num_kv_heads, head_size, dtype)


def unpadded_bytes(num_kv_heads, head_size, dtype):
    """What the model needs: a key and a value vector per head, per token.

    No hd64 special case — folding changes where values sit, not how many.
    """
    return BLOCK_SIZE * 2 * num_kv_heads * head_size * dtype.itemsize


@pytest.mark.parametrize("num_kv_heads,head_size,dtype,padded", PADDING_CASES)
def test_padding_matches_expectation(num_kv_heads, head_size, dtype, padded):
    """Pins which configurations waste HBM, at every dtype."""
    actual = page_bytes(num_kv_heads, head_size, dtype)
    assert (actual > unpadded_bytes(num_kv_heads, head_size, dtype)) is padded


@pytest.mark.parametrize("num_kv_heads,head_size", FP8_HALVES)
def test_fp8_halves_the_page(num_kv_heads, head_size):
    """A layout change must keep every one of these at 2x."""
    assert page_bytes(num_kv_heads, head_size,
                      BF16) == 2 * page_bytes(num_kv_heads, head_size, FP8)


@pytest.mark.xfail(
    strict=True,
    reason="KV heads sit on the packing dimension, so fp8 is padded back up "
    "to the bf16 page size. Fixed by moving the sequence dimension onto the "
    "packing dimension; the MLA kernel already does this.",
)
@pytest.mark.parametrize("num_kv_heads,head_size", FP8_SHOULD_HALVE)
def test_fp8_should_halve_the_page(num_kv_heads, head_size):
    """These xfail today; the xfail turning green is the signal it is fixed."""
    assert page_bytes(num_kv_heads, head_size,
                      BF16) == 2 * page_bytes(num_kv_heads, head_size, FP8)


@contextlib.contextmanager
def captured_warnings(logger_name: str = "vllm"):
    """Collect warnings emitted under `logger_name`.

    Deliberately not pytest's `caplog`: vLLM's dictConfig sets
    `propagate=False` on the `vllm` logger, so these records never reach the
    root logger where `caplog` installs its handler.
    """
    messages: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: messages.append(record.getMessage())
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    # Another test may have raised the level or called logging.disable();
    # either drops the record before any handler runs.
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


WARNING_PHRASE = "larger than"


@pytest.mark.parametrize("num_kv_heads,head_size,dtype,padded", PADDING_CASES)
def test_warns_exactly_when_padded(num_kv_heads, head_size, dtype, padded):
    """Including the bf16 cases: what matters is whether HBM is wasted."""
    with captured_warnings() as messages:
        _warn_if_kv_cache_is_padded(PallasAttentionBackend, BLOCK_SIZE,
                                    num_kv_heads, head_size, dtype)
    assert any(WARNING_PHRASE in m for m in messages) is padded


def test_capture_helper_actually_captures():
    """Guards the unpadded rows above from passing vacuously — an always-empty
    list satisfies them however the code behaves, which is exactly how the
    `caplog` version stayed green in CI while capturing nothing."""
    with captured_warnings() as messages:
        logging.getLogger("vllm.test").warning(f"x {WARNING_PHRASE} y")
    assert any(WARNING_PHRASE in m for m in messages)
