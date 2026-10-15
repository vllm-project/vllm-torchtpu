# SPDX-License-Identifier: Apache-2.0
"""Structured outputs on the megakernel decoder: the runner's grammar-bitmask
row mapping and the op's mask unpacking (CPU)."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_test

VOCAB_WORDS = 163840 // 32  # 5120


def _grammar_output(pairs):
    """``(req_id, row)`` pairs as vLLM's GrammarOutput."""
    return SimpleNamespace(
        structured_output_request_ids=[req_id for req_id, _ in pairs],
        grammar_bitmask=np.asarray([row for _, row in pairs], np.int32),
    )


def test_grammar_bitmask_rows_map_to_batch_order():
    from vllm_torchtpu.megakernel.runner import grammar_bitmask_rows

    row_b = np.full(VOCAB_WORDS, 0x55555555, np.int32)
    row_d = np.full(VOCAB_WORDS, 0x33333333, np.int32)
    grammar = _grammar_output([("b", row_b), ("d", row_d)])
    index = {"a": 0, "b": 1, "c": 2, "d": 3}
    mask = grammar_bitmask_rows(grammar, index, 4, VOCAB_WORDS)
    assert mask is not None and mask.shape == (4, VOCAB_WORDS)
    np.testing.assert_array_equal(mask[1], row_b)
    np.testing.assert_array_equal(mask[3], row_d)
    # Unconstrained rows are all ones.
    assert mask[0].all() and mask[2].all()
    assert (mask[0] == -1).all() and (mask[2] == -1).all()


def test_grammar_bitmask_rows_skips_requests_outside_the_batch():
    from vllm_torchtpu.megakernel.runner import grammar_bitmask_rows

    row = np.zeros(VOCAB_WORDS, np.int32)
    grammar = _grammar_output([("gone", row), ("b", row)])
    mask = grammar_bitmask_rows(grammar, {"a": 0, "b": 1}, 2, VOCAB_WORDS)
    assert mask is not None
    np.testing.assert_array_equal(mask[1], row)
    assert (mask[0] == -1).all()
    # Nothing mapped: the plain unmasked step (None), not an all-ones mask.
    grammar = _grammar_output([("gone", row)])
    assert grammar_bitmask_rows(grammar, {"a": 0}, 1, VOCAB_WORDS) is None


def test_grammar_bitmask_rows_take_each_requests_first_row_with_drafts():
    """With speculative decoding vLLM sends one row per scheduled draft and
    then the bonus row; the plain step uses each request's first row."""
    from vllm_torchtpu.megakernel.runner import grammar_bitmask_rows

    def row(value):
        return np.full(VOCAB_WORDS, value, np.int32)

    pairs = (
        [("a", row(1)), ("a", row(2)), ("a", row(3))]  # two drafts + bonus
        + [("gone", row(4)), ("gone", row(5))]  # one draft, not in the batch
        + [("b", row(6))]  # no drafts
        + [("c", row(7))]
        + [("c", row(-1))] * 3  # three -1 drafts + bonus
    )
    grammar = SimpleNamespace(
        structured_output_request_ids=["a", "gone", "b", "c"],
        grammar_bitmask=np.asarray([r for _, r in pairs], np.int32),
    )
    spec = {"a": [11, 12], "gone": [13], "c": [-1, -1, -1]}
    mask = grammar_bitmask_rows(grammar, {"a": 0, "b": 1, "c": 2}, 3, VOCAB_WORDS, spec)
    assert mask is not None
    for index, first in enumerate((1, 6, 7)):
        assert (mask[index] == first).all()


def test_mask_logits_unpacking_and_sharding():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE
    from vllm_torchtpu.megakernel.kimi_k3.sampling import mask_logits

    width = 163840 // TP_SIZE  # this rank's tokens
    rows = 3
    rng = np.random.default_rng(0)
    mask = rng.integers(-(2**31), 2**31, (rows, VOCAB_WORDS), dtype=np.int32)
    logits = rng.normal(size=(rows, width)).astype(np.float32)

    masked = jax.jit(mask_logits)
    for rank in (0, 1, 3, TP_SIZE - 1):
        out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(mask), rank))
        words = mask[:, rank * width // 32 : (rank + 1) * width // 32]
        allowed = ((words[:, :, None] >> np.arange(32, dtype=np.int32)) & 1) != 0
        allowed = allowed.reshape(rows, width)
        assert ((out == -1e30) == ~allowed).all()
        np.testing.assert_array_equal(out[allowed], logits[allowed])


def test_mask_logits_all_ones_is_identity_and_finite_when_rank_empty():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE
    from vllm_torchtpu.megakernel.kimi_k3.sampling import mask_logits

    width = 163840 // TP_SIZE
    logits = np.random.default_rng(1).normal(size=(2, width)).astype(np.float32)
    masked = jax.jit(mask_logits)
    ones = np.full((2, VOCAB_WORDS), -1, np.int32)
    for rank in (0, TP_SIZE - 1):
        np.testing.assert_array_equal(
            np.asarray(masked(jnp.asarray(logits), jnp.asarray(ones), rank)), logits
        )
    # A schema allowing only rank 3's tokens fully masks every other rank:
    # finite sentinel, never NaN or -inf (the per-rank normalizers stay sane).
    narrow = np.zeros((2, VOCAB_WORDS), np.int32)
    narrow[:, 3 * width // 32 : 4 * width // 32] = -1
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 0))
    assert (out == -1e30).all() and np.isfinite(out).all()
    out = np.asarray(masked(jnp.asarray(logits), jnp.asarray(narrow), 3))
    np.testing.assert_array_equal(out, logits)


def test_deferred_structured_step_runs_in_sample_tokens():
    """execute_model stashes the step; sample_tokens runs it with the mask."""
    import torch

    from vllm_torchtpu.megakernel import runner
    from vllm_torchtpu.megakernel.runner import (
        MegakernelModelRunner,
        MegakernelStepState,
    )
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

    vocab_words = VOCAB_WORDS
    tokens = torch.tensor([[101], [202]], dtype=torch.int32)
    copy_state = AsyncTPUCopyState(
        chunks_cpu=[tokens], chunk_real_lens=[2], copy_ready_event=None, completed=True
    )

    class Program:
        step_input = None

        def step(self, step_input):
            self.step_input = step_input
            return SimpleNamespace(tokens=tokens)

    program = Program()
    step_input = SimpleNamespace(mask=None)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 1}, scheduled_spec_decode_tokens={}
    )
    row = np.full(vocab_words, 0x55555555, np.int32)
    grammar = _grammar_output([("b", row)])
    instance = SimpleNamespace(
        execute_model_state=MegakernelStepState(scheduled, None, {}, step_input),
        input_batch=SimpleNamespace(
            num_reqs=2, req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}
        ),
        requests={
            req: SimpleNamespace(num_computed_tokens=10, num_tokens=11)
            for req in ("a", "b")
        },
        scheduler_config=SimpleNamespace(async_scheduling=True),
        vocab_size=vocab_words * 32,
        program=program,
        _is_output_rank=True,
        _modify_prev_results=lambda: None,
        maybe_wait_for_kv_save=lambda: None,
        _build_kv_connector_output=lambda _: None,
        _update_placeholder=lambda *args: {"a": 0, "b": 1},
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(AsyncTPUCopyState, "from_device_chunks", lambda *args: copy_state)
    monkey.setattr(runner, "has_kv_transfer_group", lambda: False)
    result = MegakernelModelRunner.sample_tokens(instance, grammar).get_output()
    monkey.undo()
    # The program ran in sample_tokens, masked: row b constrained, row a all ones.
    mask = program.step_input.mask
    assert mask is not None and mask.shape == (2, vocab_words)
    np.testing.assert_array_equal(mask[1], row)
    assert (mask[0] == -1).all()
    assert result.sampled_token_ids == [[101], [202]]
