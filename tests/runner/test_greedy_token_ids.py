# SPDX-License-Identifier: Apache-2.0
"""All-greedy steps pick tokens from vocab shards instead of gathered logits.

execute_model hands sample_tokens [n, 1] int64 token ids in place of logits;
these tests pin when that happens and that every consumer yields the same
tokens as the logits path.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

from vllm_torchtpu.layers.adapter.sample.vocab_parallel import local_argmax_pair
from vllm_torchtpu.runner.tpu_runner import (
    TPUModelRunner,
    _vocab_parallel_greedy_head,
)


def _head(vocab=257, num_embeddings=None, bounds=True):
    head = vpe.VocabParallelEmbedding.__new__(vpe.VocabParallelEmbedding)
    torch.nn.Module.__init__(head)
    head.org_vocab_size = vocab
    head.num_embeddings = vocab if num_embeddings is None else num_embeddings
    head.tp_size = 1
    if bounds:
        head.register_buffer("org_vocab_start_index", torch.tensor(0))
        head.register_buffer("org_vocab_end_index", torch.tensor(vocab))
    return head


def _model(head, logits_as_input=False, local=True, wrap=False):
    processor = SimpleNamespace(
        logits_as_input=logits_as_input, org_vocab_size=head.org_vocab_size
    )
    lm = SimpleNamespace(lm_head=head, logits_processor=processor)
    model = SimpleNamespace(language_model=lm) if wrap else lm
    if local:
        weight = torch.randn(head.org_vocab_size, 16)
        model.compute_logits_local = lambda hidden: hidden @ weight.T
        model.compute_logits = model.compute_logits_local
    return model


@pytest.mark.parametrize("wrap", [False, True])
def test_head_found_for_plain_and_wrapped_language_models(wrap):
    head = _head()
    assert _vocab_parallel_greedy_head(_model(head, wrap=wrap), False) is head


@pytest.mark.parametrize(
    "case",
    ["lora", "no_local_logits", "logits_as_input", "added_vocab", "no_bounds", "huge"],
)
def test_head_rejected_when_shards_cannot_pick_exactly(case):
    head = _head(
        vocab=2**24 + 1 if case == "huge" else 257,
        num_embeddings=300 if case == "added_vocab" else None,
        bounds=case != "no_bounds",
    )
    model = _model(
        head,
        logits_as_input=case == "logits_as_input",
        local=case != "no_local_logits",
    )
    assert _vocab_parallel_greedy_head(model, case == "lora") is None


@pytest.mark.parametrize(
    "change",
    [None, "no_head", "sampled", "logprobs", "structured", "nan_checker"],
)
def test_token_ids_only_when_nothing_reads_logits(change):
    runner = SimpleNamespace(
        _greedy_head=None if change == "no_head" else object(),
        input_batch=SimpleNamespace(
            all_greedy=change != "sampled",
            max_num_logprobs=5 if change == "logprobs" else None,
        ),
        _nan_checker=object() if change == "nan_checker" else None,
    )
    scheduler_output = SimpleNamespace(
        has_structured_output_requests=change == "structured"
    )
    assert TPUModelRunner._samples_greedy_token_ids(runner, scheduler_output) is (
        change is None
    )


def test_greedy_token_ids_match_logits_argmax():
    torch.manual_seed(5)
    head = _head()
    model = _model(head)
    runner = SimpleNamespace(_greedy_head=head, model=model)
    hidden = torch.randn(12, 16)
    ids = TPUModelRunner._greedy_token_ids(runner, hidden)
    assert ids.shape == (12, 1) and ids.dtype == torch.int64
    expected = torch.argmax(model.compute_logits(hidden), dim=-1, keepdim=True)
    assert torch.equal(ids, expected)


def test_spec_tokens_from_token_ids_match_logits_path():
    torch.manual_seed(3)
    logits = torch.randn(12, 257)
    logits[4, [9, 200]] = 50  # tie: the first index wins on both paths
    bonus = torch.tensor([3, 7, 11])
    target = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 10])
    fn = TPUModelRunner.spec_bonus_and_target_tokens._torchdynamo_orig_callable
    ids = logits.argmax(dim=-1, keepdim=True)
    for got, want in zip(
        fn(None, ids, bonus, target), fn(None, logits, bonus, target), strict=True
    ):
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_greedy_sample_passes_token_ids_through():
    ids = torch.tensor([[4], [9]])
    assert TPUModelRunner._greedy_sample(None, ids) is ids


@pytest.mark.parametrize("ties", [False, True])
def test_int32_head_bounds_pick_like_int64(ties):
    """LM head bounds are int32; the argmax pair matches int64 bounds."""
    torch.manual_seed(11)
    logits = torch.randn(6, 64).to(torch.bfloat16)
    if ties:
        logits[:, [3, 40]] = 9
    logits[:, 60:] = float("inf")  # padding columns past the shard end
    pairs = [
        local_argmax_pair(
            logits, torch.tensor(128, dtype=d), torch.tensor(188, dtype=d)
        )
        for d in (torch.int32, torch.int64)
    ]
    assert torch.equal(pairs[0], pairs[1])
    assert torch.equal(pairs[0][:, 1].long(), logits[:, :60].argmax(dim=-1) + 128)
