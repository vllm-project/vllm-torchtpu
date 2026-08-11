"""Page-size accounting for narrowed KV cache dtypes.

`get_kv_cache_shape` places KV heads on the Pallas 32-bit packing dimension and
rounds their count up to a whole word. bf16 fits 2 values per word and fp8 fits
4, so a page costs `ceil(num_kv_heads_x2 / packing)` words either way whenever
`num_kv_heads_x2 <= 2` — fp8 then frees no memory at all.
"""
import contextlib
import logging

import pytest
import torch

from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
BLOCK_SIZE = 256

# `num_kv_heads` too small to fill an fp8 word, and the rest. hd64
# (`head_size == 64`) folds K and V along head_dim, so it starts wasting one
# tensor-parallel step earlier, at 2 heads.
UNDERFILLED = [(1, 128), (1, 256), (1, 64), (2, 64)]
FILLED = [(2, 128), (4, 128), (8, 128), (2, 256), (8, 256), (4, 64), (8, 64)]


def fp8_saving(num_kv_heads, head_size):
    """How many times smaller an fp8 page is than a bf16 one."""
    page = PallasAttentionBackend.get_kv_cache_page_size_bytes
    return (page(BLOCK_SIZE, num_kv_heads, head_size, BF16) /
            page(BLOCK_SIZE, num_kv_heads, head_size, FP8))


@pytest.mark.parametrize("num_kv_heads,head_size", FILLED)
def test_fp8_halves_the_page(num_kv_heads, head_size):
    """A layout change must keep every one of these at 2x."""
    assert fp8_saving(num_kv_heads, head_size) == pytest.approx(2.0)


@pytest.mark.xfail(
    strict=True,
    reason="KV heads sit on the packing dimension, so fp8 is padded back up "
    "to the bf16 page size. Fixed by moving the sequence dimension onto the "
    "packing dimension; the MLA kernel already does this.",
)
@pytest.mark.parametrize("num_kv_heads,head_size", UNDERFILLED)
def test_fp8_should_halve_the_page(num_kv_heads, head_size):
    """These xfail today; the xfail turning green is the signal it is fixed."""
    assert fp8_saving(num_kv_heads, head_size) == pytest.approx(2.0)


@pytest.fixture(autouse=True)
def _reset_warning_once():
    """`warning_once` dedups by message for the life of the process, so without
    this a test passes or fails depending on what ran before it."""
    import vllm.logger

    vllm.logger._print_warning_once.cache_clear()
    yield
    vllm.logger._print_warning_once.cache_clear()


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


WARNING_PHRASE = "frees less memory"


@pytest.mark.parametrize("num_kv_heads,head_size,dtype,expected", [
    (1, 128, FP8, True),
    (8, 128, FP8, False),
    (1, 128, BF16, False),
])
def test_warns_exactly_when_saving_is_lost(num_kv_heads, head_size, dtype,
                                           expected):
    with captured_warnings() as messages:
        PallasAttentionBackend.warn_if_kv_cache_dtype_is_ineffective(
            BLOCK_SIZE, num_kv_heads, head_size, dtype)
    assert any(WARNING_PHRASE in m for m in messages) is expected


def test_capture_helper_actually_captures():
    """Guards the two `expected=False` rows above from passing vacuously — an
    always-empty list satisfies them however the code behaves, which is exactly
    how the `caplog` version stayed green in CI while capturing nothing."""
    with captured_warnings() as messages:
        logging.getLogger("vllm.test").warning(f"x {WARNING_PHRASE}")
    assert any(WARNING_PHRASE in m for m in messages)
