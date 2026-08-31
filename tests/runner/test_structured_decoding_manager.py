# SPDX-License-Identifier: Apache-2.0
import contextlib
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
                 req_id_to_index: dict[str, int] | None = None,
                 num_spec_tokens: int | None = None,
                 num_tokens_paddings: list[int] | None = None,
                 pcp_mtp_k1: bool = False,
                 device: torch.device | str = "cpu"):
    speculative_config = (SimpleNamespace(
        num_speculative_tokens=num_spec_tokens)
                          if num_spec_tokens is not None else None)
    runner = SimpleNamespace(
        vocab_size=vocab_size,
        max_num_reqs=max_num_reqs,
        device=torch.device(device),
        input_batch=SimpleNamespace(req_id_to_index=req_id_to_index or {}),
        speculative_config=speculative_config,
        num_tokens_paddings=num_tokens_paddings or [16, 32, 64],
        _pcp_mtp_k1_enabled=pcp_mtp_k1,
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


def row(value: int) -> list[int]:
    """A distinguishable bitmask row: both packed words carry `value`."""
    return [value, value]


class TestPrepareSpecStructuredDecodingInput:
    """Each structured request owns 1 + s_r consecutive
    bitmask rows (draft + bonus)"""

    def test_spec_buffers_only_allocated_with_spec_config(self):
        manager = make_manager()
        assert not hasattr(manager, "target_grammar_bitmask_cpu")

        spec_manager = make_manager(num_spec_tokens=3,
                                    max_num_reqs=8,
                                    num_tokens_paddings=[16, 32, 64])
        # 8 reqs * (1 + 3) = 32 rows -> the 32 bucket covers it exactly.
        assert spec_manager.target_grammar_bitmask_cpu.shape == (32, 2)
        assert spec_manager.require_structured_out_target_cpu.shape == (32, 1)

    def test_prefill_only_pcp_mtp_k1_skips_target_buffers(self):
        # Prefill-only PCP MTP K1 never verifies drafts.
        manager = make_manager(num_spec_tokens=1, pcp_mtp_k1=True)
        assert not hasattr(manager, "target_grammar_bitmask_cpu")
        assert not hasattr(manager, "require_structured_out_target_cpu")

    def test_variable_draft_counts_single_chunk(self):
        # req0 structured, 2 drafts; req1 unstructured, 1 draft;
        # req2 structured, 0 drafts.
        manager = make_manager(num_spec_tokens=2,
                               req_id_to_index={
                                   "req0": 0,
                                   "req1": 1,
                                   "req2": 2
                               })
        grammar_output = make_grammar_output(
            ["req0", "req2"],
            # req0: draft rows 10, 11, bonus 12; req2: bonus row 20.
            [row(10), row(11), row(12), row(20)],
        )
        scheduled = {"req0": [7, 8], "req1": [9]}
        draft_lengths = np.array([2, 1, 0], dtype=np.int32)
        target_logits = torch.zeros(8, VOCAB_SIZE)
        bonus_logits = torch.zeros(4, VOCAB_SIZE)

        (require_target, target_bitmask, require_bonus, bonus_bitmask,
         arange) = manager.prepare_spec_structured_decoding_input(
             target_logits, bonus_logits, grammar_output, scheduled,
             draft_lengths, 0, 3)

        # Target row j aligns with target_logits row j (the target
        # model's logits at draft position j). Rows 0-1 belong to req0's
        # draft positions, row 2 to req1's (unstructured -> untouched),
        # rest padding.
        assert require_target.shape == (8, 1)
        assert require_target[:, 0].tolist() == [
            True, True, False, False, False, False, False, False
        ]
        assert target_bitmask[0].tolist() == row(10)
        assert target_bitmask[1].tolist() == row(11)
        assert target_bitmask[2].tolist() == [0, 0]
        # Bonus: req0 row 0, req2 row 2; req1 untouched.
        assert require_bonus[:, 0].tolist() == [True, False, True, False]
        assert bonus_bitmask[0].tolist() == row(12)
        assert bonus_bitmask[1].tolist() == [0, 0]
        assert bonus_bitmask[2].tolist() == row(20)
        assert arange.tolist() == list(range(32))

    def test_second_chunk_uses_local_rows(self):
        # Chunk 2 covers batch rows [2, 4). req0 (chunk 1) has 1 draft;
        # req2/req3 (chunk 2) have 1 and 2 drafts, req3 structured.
        manager = make_manager(num_spec_tokens=2,
                               req_id_to_index={
                                   "req0": 0,
                                   "req1": 1,
                                   "req2": 2,
                                   "req3": 3
                               })
        grammar_output = make_grammar_output(
            ["req0", "req3"],
            # req0: draft row 10, bonus 11; req3: draft rows 30, 31, bonus 32.
            [row(10), row(11), row(30),
             row(31), row(32)],
        )
        scheduled = {"req0": [1], "req2": [2], "req3": [3, 4]}
        # Chunk-local draft lengths for [req2, req3].
        draft_lengths = np.array([1, 2], dtype=np.int32)
        target_logits = torch.zeros(4, VOCAB_SIZE)
        bonus_logits = torch.zeros(2, VOCAB_SIZE)

        (require_target, target_bitmask, require_bonus, bonus_bitmask,
         _) = manager.prepare_spec_structured_decoding_input(
             target_logits, bonus_logits, grammar_output, scheduled,
             draft_lengths, 2, 4)

        # Target rows by draft position:
        # [req2 draft0, req3 draft0, req3 draft1, pad].
        assert require_target[:, 0].tolist() == [False, True, True, False]
        assert target_bitmask[1].tolist() == row(30)
        assert target_bitmask[2].tolist() == row(31)
        # req0's rows (chunk 1) must not leak in.
        assert target_bitmask[0].tolist() == [0, 0]
        # Bonus: [req2 (unstructured), req3].
        assert require_bonus[:, 0].tolist() == [False, True]
        assert bonus_bitmask[1].tolist() == row(32)

    def test_draft_free_chunk_walks_row_strides(self):
        # md is None for this chunk (its requests carry no drafts), but a
        # structured request in another chunk owns 1 + 2 rows; the cursor
        # must stride over them to find this chunk's bonus row.
        manager = make_manager(num_spec_tokens=2,
                               req_id_to_index={
                                   "req0": 0,
                                   "req1": 1
                               })
        grammar_output = make_grammar_output(
            ["req0", "req1"],
            # req0 (other chunk): drafts 10, 11, bonus 12; req1: bonus 20.
            [row(10), row(11), row(12), row(20)],
        )
        scheduled = {"req0": [5, 6]}
        logits = torch.zeros(2, VOCAB_SIZE)

        (require_target, target_bitmask, require_bonus, bonus_bitmask,
         _) = manager.prepare_spec_structured_decoding_input(
             None, logits, grammar_output, scheduled, None, 1, 2)

        assert require_target is None
        assert target_bitmask is None
        # req1 is local row 0 of this chunk.
        assert require_bonus[:, 0].tolist() == [True, False]
        assert bonus_bitmask[0].tolist() == row(20)

    def test_finished_request_keeps_row_alignment(self):
        # A request already removed from the batch still owns its 1 + s_r
        # rows; later requests must not shift up.
        manager = make_manager(num_spec_tokens=2, req_id_to_index={"req1": 0})
        grammar_output = make_grammar_output(
            ["req0", "req1"],
            # req0 (gone, had 2 drafts): rows 10, 11, 12; req1: draft 20,
            # bonus 21.
            [row(10), row(11), row(12),
             row(20), row(21)],
        )
        scheduled = {"req0": [1, 2], "req1": [3]}
        draft_lengths = np.array([1], dtype=np.int32)
        target_logits = torch.zeros(2, VOCAB_SIZE)
        bonus_logits = torch.zeros(1, VOCAB_SIZE)

        (require_target, target_bitmask, require_bonus, bonus_bitmask,
         _) = manager.prepare_spec_structured_decoding_input(
             target_logits, bonus_logits, grammar_output, scheduled,
             draft_lengths, 0, 1)

        assert require_target[:, 0].tolist() == [True, False]
        assert target_bitmask[0].tolist() == row(20)
        assert require_bonus[0, 0].item() is True
        assert bonus_bitmask[0].tolist() == row(21)

    def test_stale_rows_are_reset(self):
        manager = make_manager(num_spec_tokens=1,
                               req_id_to_index={
                                   "req0": 0,
                                   "req1": 1
                               })
        first = make_grammar_output(
            ["req0", "req1"],
            [row(10), row(11), row(20), row(21)],
        )
        scheduled = {"req0": [1], "req1": [2]}
        draft_lengths = np.array([1, 1], dtype=np.int32)
        target_logits = torch.zeros(4, VOCAB_SIZE)
        bonus_logits = torch.zeros(2, VOCAB_SIZE)
        manager.prepare_spec_structured_decoding_input(target_logits,
                                                       bonus_logits, first,
                                                       scheduled,
                                                       draft_lengths, 0, 2)

        # Second step: only req1 is structured.
        second = make_grammar_output(["req1"], [row(30), row(31)])
        scheduled = {"req1": [3]}
        draft_lengths = np.array([0, 1], dtype=np.int32)
        (require_target, target_bitmask, require_bonus, bonus_bitmask,
         _) = manager.prepare_spec_structured_decoding_input(
             target_logits, bonus_logits, second, scheduled, draft_lengths, 0,
             2)

        # req0's rows from the first call must be gone.
        assert require_target[:, 0].tolist() == [True, False, False, False]
        assert target_bitmask[0].tolist() == row(30)
        assert require_bonus[:, 0].tolist() == [False, True]
        assert bonus_bitmask[0].tolist() == [0, 0]
        assert bonus_bitmask[1].tolist() == row(31)


@contextlib.contextmanager
def capture_trace():
    """Captures PyTorch CPU profiler events during test execution."""
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        yield prof


class TestMaskLogitsWithTrace:

    def test_mask_logits(self, monkeypatch):
        manager = make_manager(req_id_to_index={"req0": 0, "req1": 1})
        grammar_output = make_grammar_output(
            ["req0"],
            [[ALLOW_ALL, BLOCK_ALL]],
        )
        logits = torch.zeros(4, VOCAB_SIZE)  # Padded to 4 rows for 2 requests
        monkeypatch.setattr(manager, "structured_decode",
                            manager._structured_decode)

        with capture_trace() as prof:
            out = manager.mask_logits(logits,
                                      grammar_output,
                                      cur_start_idx=0,
                                      cur_end_idx=2)

        # 1. Output correctness: row 0 masked on second half, row 1 untouched.
        assert torch.equal(out[0, :32], logits[0, :32])
        assert torch.isinf(out[0, 32:]).all()
        assert torch.equal(out[1:], logits[1:])

        # 2. TraceAnnotation verification.
        events = [evt.name for evt in prof.events()]
        assert any(
            "SD:PrepareInput#num_reqs=2,padded_num_reqs=4,cur_start_idx=0,cur_end_idx=2#"
            in name
            for name in events), f"SD:PrepareInput not found in {events}"
        assert any(
            "SD:MaskLogits#num_reqs=2,padded_num_reqs=4,cur_start_idx=0,cur_end_idx=2#"
            in name for name in events), f"SD:MaskLogits not found in {events}"

    def test_mask_spec_logits(self, monkeypatch):
        manager = make_manager(num_spec_tokens=1, req_id_to_index={"req0": 0})
        grammar_output = make_grammar_output(
            ["req0"],
            [[ALLOW_ALL, BLOCK_ALL], [BLOCK_ALL, ALLOW_ALL]],
        )
        target_logits = torch.zeros(1, VOCAB_SIZE)
        bonus_logits = torch.zeros(1, VOCAB_SIZE)
        scheduled = {"req0": [1]}
        draft_lengths = np.array([1], dtype=np.int32)
        monkeypatch.setattr(manager, "structured_decode",
                            manager._structured_decode)

        with capture_trace() as prof:
            out_target, out_bonus = manager.mask_spec_logits(
                target_logits, bonus_logits, grammar_output, scheduled,
                draft_lengths, 0, 1)

        # 1. Output correctness: target masked top half, bonus masked bottom half.
        assert torch.isinf(out_target[0, 32:]).all()
        assert torch.isinf(out_bonus[0, :32]).all()

        # 2. TraceAnnotation verification with target_logits.
        events = [evt.name for evt in prof.events()]
        assert any(
            "SD:PrepareSpecInput#num_reqs=1,num_target_rows=1,num_bonus_rows=1,cur_start_idx=0,cur_end_idx=1#"
            in name
            for name in events), f"SD:PrepareSpecInput not found in {events}"
        assert any(
            "SD:MaskSpecLogits#num_reqs=1,num_target_rows=1,num_bonus_rows=1,cur_start_idx=0,cur_end_idx=1#"
            in name
            for name in events), f"SD:MaskSpecLogits not found in {events}"

        # 3. TraceAnnotation verification when target_logits is None (non-draft chunk).
        grammar_output_none = make_grammar_output(["req0"],
                                                  [[ALLOW_ALL, BLOCK_ALL]])
        with capture_trace() as prof_none:
            manager.mask_spec_logits(None, bonus_logits, grammar_output_none,
                                     {}, None, 0, 1)
        events_none = [evt.name for evt in prof_none.events()]
        assert any(
            "SD:PrepareSpecInput#num_reqs=1,num_target_rows=0,num_bonus_rows=1,cur_start_idx=0,cur_end_idx=1#"
            in name for name in events_none
        ), f"SD:PrepareSpecInput (target=None) not found in {events_none}"
