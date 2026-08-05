"""Page-size accounting for narrowed KV cache dtypes (b/510425663).

`get_kv_cache_shape` places KV heads on the Pallas 32-bit packing dimension and
rounds their count up to a whole word. bf16 fits 2 values per word and fp8 fits
4, so a page costs `ceil(num_kv_heads_x2 / packing)` words either way whenever
`num_kv_heads_x2 <= 2` — fp8 then frees no memory at all.

These tests pin both halves of that behaviour: the configurations where fp8
already works (which any layout fix must not regress) and the configurations
where it silently does not (which the fix must make pass).
"""
import contextlib
import logging

import pytest
import torch

from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend

BF16 = torch.bfloat16
# `--kv-cache-dtype fp8` allocates e4m3fn: both the runner and the worker
# resolve it through TPU_STR_DTYPE_TO_TORCH_DTYPE. Note this disagrees with the
# "Automatically using fp8_e5m2" startup line, which comes from the unrelated
# TpuPlatform.fp8_dtype(). Page size is identical either way (both are 1 byte),
# so these tests hold for both; e4m3fn is used because it is what is allocated.
FP8 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
BLOCK_SIZE = 256


def page_bytes(num_kv_heads, head_size, dtype, block_size=BLOCK_SIZE):
    return PallasAttentionBackend.get_kv_cache_page_size_bytes(
        block_size, num_kv_heads, head_size, dtype)


def saving(num_kv_heads, head_size, block_size=BLOCK_SIZE):
    return PallasAttentionBackend.kv_cache_dtype_saving_ratio(
        block_size, num_kv_heads, head_size, FP8)


class TestPackingFactor:

    @pytest.mark.parametrize("dtype,expected", [(BF16, 2), (FP8, 4),
                                                (FP8_E5M2, 4)])
    def test_values_per_32_bit_word(self, dtype, expected):
        from vllm_torchtpu.layers.vllm.attention import get_dtype_packing
        assert get_dtype_packing(dtype) == expected

    def test_both_fp8_variants_cost_the_same(self):
        """Guards the claim that the e4m3fn/e5m2 mismatch is memory-neutral."""
        for num_kv_heads in (1, 2, 8):
            assert page_bytes(num_kv_heads, 128,
                              FP8) == page_bytes(num_kv_heads, 128, FP8_E5M2)


class TestFp8SavesMemoryWhereItCan:
    """Configurations with enough KV heads to fill an fp8 word.

    A layout change must keep every one of these at 2x.
    """

    # (num_kv_heads, head_size) -> enough values to fill a 4-wide fp8 word.
    @pytest.mark.parametrize("num_kv_heads,head_size", [
        (2, 128),
        (4, 128),
        (8, 128),
        (2, 256),
        (8, 256),
        (4, 64),
        (8, 64),
    ])
    def test_fp8_halves_the_page(self, num_kv_heads, head_size):
        assert page_bytes(num_kv_heads, head_size,
                          BF16) == 2 * page_bytes(num_kv_heads, head_size, FP8)
        assert saving(num_kv_heads, head_size) == pytest.approx(2.0)


class TestPageSizeIsIndependentOfBlockSize:
    """Capacity in tokens must not depend on how tokens are grouped."""

    @pytest.mark.parametrize("block_size", [16, 32, 128, 256])
    @pytest.mark.parametrize("dtype", [BF16, FP8])
    def test_bytes_per_token_constant(self, block_size, dtype):
        per_token = page_bytes(8, 128, dtype, block_size) / block_size
        reference = page_bytes(8, 128, dtype, BLOCK_SIZE) / BLOCK_SIZE
        assert per_token == reference


class TestFp8SavesNothingWhenHeadsUnderfillAWord:
    """The bug. These xfail today and must pass once the layout is fixed.

    `num_kv_heads == 1` at head_size 128/256 is the ordinary result of running
    a GQA model at a tensor-parallel size at or above its KV head count — e.g.
    Llama-3.1-8B or Llama-3.3-70B at TP=8. The hd64 kernel packs K and V along
    head_dim instead, so it starts wasting one TP step earlier, at 2 heads.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="b/510425663: KV heads sit on the packing dimension, so fp8 is "
        "padded back up to the bf16 page size. Fixed by moving the sequence "
        "dimension onto the packing dimension (cf. tpu-inference#3170); the "
        "MLA kernel already does this.",
    )
    @pytest.mark.parametrize("num_kv_heads,head_size", [
        (1, 128),
        (1, 256),
        (1, 64),
        (2, 64),
    ])
    def test_fp8_should_halve_the_page(self, num_kv_heads, head_size):
        assert page_bytes(num_kv_heads, head_size,
                          BF16) == 2 * page_bytes(num_kv_heads, head_size, FP8)

    @pytest.mark.parametrize("num_kv_heads,head_size", [
        (1, 128),
        (1, 256),
        (1, 64),
        (2, 64),
    ])
    def test_currently_saves_nothing(self, num_kv_heads, head_size):
        """Documents today's behaviour so the xfail above cannot rot silently.

        Delete this class's `test_currently_saves_nothing` when the fix lands;
        the xfail turning green is the signal.
        """
        assert saving(num_kv_heads, head_size) == pytest.approx(1.0)


@pytest.fixture(autouse=True)
def _reset_warning_once():
    """`warning_once` dedups by message across the whole process.

    Without clearing it these tests would pass or fail depending on what ran
    before them — including the same test a second time.
    """
    import vllm.logger

    vllm.logger._print_warning_once.cache_clear()
    yield
    vllm.logger._print_warning_once.cache_clear()


class _RecordingLogHandler(logging.Handler):
    """Same shape as the one in tests/distributed/tpu_connector_v2_test_utils.py."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def captured_warnings(logger_name: str = "vllm"):
    """Collect warnings emitted under `logger_name`.

    Deliberately not pytest's `caplog`. vLLM's dictConfig sets
    `propagate=False` on the `vllm` logger, and `vllm_torchtpu.logger` prefixes
    every name with `vllm.`, so these records stop at `vllm` and never reach
    the root logger where `caplog` installs its handler. Whether `caplog` sees
    anything then depends on pytest's handler placement racing vLLM's
    dictConfig, which differs between a single-file local run and CI.
    Attaching directly is deterministic in both.
    """
    logger = logging.getLogger(logger_name)
    handler = _RecordingLogHandler()
    old_level = logger.level
    # Another test in this process may have raised the level or called
    # logging.disable(); either drops the record before any handler runs.
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


def warned_about_bug(messages):
    return any("b/510425663" in message for message in messages)


class TestWarningFiresExactlyWhenSavingIsLost:

    def test_warns_on_underfilled_word(self):
        with captured_warnings() as messages:
            PallasAttentionBackend.warn_if_kv_cache_dtype_is_ineffective(
                BLOCK_SIZE, 1, 128, FP8)
        assert warned_about_bug(messages)

    def test_silent_when_fp8_actually_helps(self):
        with captured_warnings() as messages:
            PallasAttentionBackend.warn_if_kv_cache_dtype_is_ineffective(
                BLOCK_SIZE, 8, 128, FP8)
        assert not warned_about_bug(messages)

    def test_silent_for_bf16(self):
        with captured_warnings() as messages:
            PallasAttentionBackend.warn_if_kv_cache_dtype_is_ineffective(
                BLOCK_SIZE, 1, 128, BF16)
        assert not warned_about_bug(messages)

    def test_capture_helper_actually_captures(self):
        """Guards the three tests above from passing vacuously.

        Two of them assert an *absence*, which an always-empty record list
        satisfies no matter what the code does. That is exactly how the
        `caplog` version stayed green in CI while capturing nothing.
        """
        with captured_warnings() as messages:
            logging.getLogger("vllm.test").warning("sentinel b/510425663")
        assert warned_about_bug(messages)


class TestMeasuredHardwareValues:
    """Per-token costs measured on a v6e-4 at TP=4 (see b/510425663).

    Guards the arithmetic against a refactor that changes the numbers without
    anyone re-running hardware.
    """

    @pytest.mark.parametrize(
        "label,num_kv_heads,head_size,layers,bf16_per_tok,fp8_per_tok", [
            ("DeepSeek-R1-Distill-Qwen-7B", 1, 128, 28, 14336, 14336),
            ("Meta-Llama-3.1-8B-Instruct", 2, 128, 32, 32768, 16384),
        ])
    def test_bytes_per_token(self, label, num_kv_heads, head_size, layers,
                             bf16_per_tok, fp8_per_tok):
        bf16 = page_bytes(num_kv_heads, head_size, BF16) / BLOCK_SIZE * layers
        fp8 = page_bytes(num_kv_heads, head_size, FP8) / BLOCK_SIZE * layers
        assert bf16 == bf16_per_tok, label
        assert fp8 == fp8_per_tok, label
