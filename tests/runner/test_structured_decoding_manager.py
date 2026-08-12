# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.structured_decoding_manager import \
    StructuredDecodingManager

VOCAB_SIZE = 64  # 2 packed int32 words per bitmask row.
ALLOW_ALL = -1  # All 32 bits set.
BLOCK_ALL = 0


def make_manager(vocab_size: int = VOCAB_SIZE,
                 max_num_reqs: int = 8,
                 req_id_to_index: dict[str, int] | None = None):
    runner = SimpleNamespace(
        vocab_size=vocab_size,
        max_num_reqs=max_num_reqs,
        input_batch=SimpleNamespace(req_id_to_index=req_id_to_index or {}),
    )
    return StructuredDecodingManager(runner)


def make_grammar_output(req_ids: list[str], rows: list[list[int]]):
    return SimpleNamespace(
        structured_output_request_ids=req_ids,
        grammar_bitmask=np.array(rows, dtype=np.int32),
    )


def reference_apply_bitmask(logits: torch.Tensor, bitmask: torch.Tensor,
                            vocab_size: int) -> torch.Tensor:
    """Per-bit reference for the vectorized unpack."""
    out = logits.clone()
    for row in range(logits.shape[0]):
        for token in range(vocab_size):
            word = int(bitmask[row, token // 32])
            if not (word >> (token % 32)) & 1:
                out[row, token] = float("-inf")
    return out


class TestPrepareStructuredDecodingInput:

    def test_scatter_mixed_batch(self):
        # 3 requests; req0 and req2 are structured, req1 is not.
        manager = make_manager(req_id_to_index={
            "req0": 0,
            "req1": 1,
            "req2": 2
        })
        grammar_output = make_grammar_output(
            ["req0", "req2"],
            [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]],
        )
        logits = torch.zeros(4, VOCAB_SIZE)  # Padded to 4 rows.

        require, bitmask, arange = manager.prepare_structured_decoding_input(
            logits, grammar_output, cur_start_idx=0, cur_end_idx=3)

        assert require.shape == (4, 1)
        assert bitmask.shape == (4, 2)
        assert require[:, 0].tolist() == [True, False, True, False]
        assert bitmask[0].tolist() == [ALLOW_ALL, BLOCK_ALL]
        assert bitmask[1].tolist() == [0, 0]
        assert bitmask[2].tolist() == [BLOCK_ALL, ALLOW_ALL]
        assert arange.tolist() == list(range(32))

    def test_second_chunk_uses_local_rows(self):
        # Chunk 2 covers batch rows [2, 4): its structured request (global
        # row 3) must land on local row 1, and chunk 1's rows must not
        # leak in.
        manager = make_manager(req_id_to_index={
            "req0": 0,
            "req1": 1,
            "req2": 2,
            "req3": 3
        })
        grammar_output = make_grammar_output(
            ["req0", "req3"],
            [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]],
        )
        logits = torch.zeros(2, VOCAB_SIZE)

        require, bitmask, _ = manager.prepare_structured_decoding_input(
            logits, grammar_output, cur_start_idx=2, cur_end_idx=4)

        assert require[:, 0].tolist() == [False, True]
        assert bitmask[0].tolist() == [0, 0]
        assert bitmask[1].tolist() == [BLOCK_ALL, ALLOW_ALL]

    def test_finished_request_keeps_row_alignment(self):
        # A request that was already removed from the batch still owns a
        # bitmask row; later rows must not shift up.
        manager = make_manager(req_id_to_index={"req1": 0})
        grammar_output = make_grammar_output(
            ["req0", "req1"],
            [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]],
        )
        logits = torch.zeros(1, VOCAB_SIZE)

        require, bitmask, _ = manager.prepare_structured_decoding_input(
            logits, grammar_output, cur_start_idx=0, cur_end_idx=1)

        assert require[0, 0].item() is True
        assert bitmask[0].tolist() == [BLOCK_ALL, ALLOW_ALL]

    def test_stale_rows_are_reset(self):
        # A later call must not see rows written by an earlier one.
        manager = make_manager(req_id_to_index={"req0": 0, "req1": 1})
        first = make_grammar_output(["req0", "req1"],
                                    [[ALLOW_ALL, ALLOW_ALL]] * 2)
        logits = torch.zeros(2, VOCAB_SIZE)
        manager.prepare_structured_decoding_input(logits, first, 0, 2)

        second = make_grammar_output(["req1"], [[BLOCK_ALL, BLOCK_ALL]])
        require, bitmask, _ = manager.prepare_structured_decoding_input(
            logits, second, 0, 2)

        assert require[:, 0].tolist() == [False, True]
        assert bitmask[0].tolist() == [0, 0]
        assert bitmask[1].tolist() == [BLOCK_ALL, BLOCK_ALL]


class TestStructuredDecode:

    @pytest.mark.parametrize("vocab_size", [64, 50])
    def test_matches_per_bit_reference(self, vocab_size):
        # vocab_size=50 also checks tail truncation when the vocab is not
        # a multiple of 32.
        torch.manual_seed(0)
        num_reqs = 4
        num_words = -(-vocab_size // 32)
        manager = make_manager(vocab_size=vocab_size)
        logits = torch.randn(num_reqs, vocab_size)
        bitmask = torch.randint(torch.iinfo(torch.int32).min,
                                torch.iinfo(torch.int32).max,
                                (num_reqs, num_words),
                                dtype=torch.int32)
        arange = torch.arange(32)

        out = manager.apply_grammar_bitmask(logits, bitmask, arange)

        expected = reference_apply_bitmask(logits, bitmask, vocab_size)
        assert torch.equal(out, expected)

    def test_unstructured_rows_pass_through(self):
        torch.manual_seed(0)
        manager = make_manager()
        logits = torch.randn(3, VOCAB_SIZE)
        # Row 0 structured (blocks the upper half), rows 1-2 not.
        bitmask = torch.tensor([[ALLOW_ALL, BLOCK_ALL], [0, 0], [0, 0]],
                               dtype=torch.int32)
        require = torch.tensor([[True], [False], [False]])
        arange = torch.arange(32)

        out = manager._structured_decode(require, bitmask, logits, arange)

        assert torch.equal(out[0, :32], logits[0, :32])
        assert torch.isinf(out[0, 32:]).all()
        assert torch.equal(out[1:], logits[1:])
