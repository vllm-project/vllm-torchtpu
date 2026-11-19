# SPDX-License-Identifier: Apache-2.0
"""The samplers leave out the top-k and top-p searches no row of a step uses.

The runner reads from the host copies of a chunk's sampling parameters whether
any row applies top-k and whether any applies top-p, and passes both to the
samplers as static arguments. A search no row uses is left out with the same
tokens; when any row applies a mask, the search runs.
"""

import contextlib
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import MASK_FLAG_COMBINATIONS
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import PaddedSamplingParams, TPUModelRunner

VOCAB = 64
ROWS = 8


def _params_runner(top_k, top_p):
    n = len(top_k)
    runner = SimpleNamespace(
        vocab_size=VOCAB,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=torch.ones(n),
            top_k_cpu_tensor=torch.tensor(top_k, dtype=torch.int32),
            top_p_cpu_tensor=torch.tensor(top_p, dtype=torch.float32),
        ),
    )
    runner.build = MethodType(TPUModelRunner._build_padded_sampling_params, runner)
    return runner


@pytest.mark.parametrize(
    "top_k, top_p, expected",
    [
        ([0, VOCAB, VOCAB + 7], [1.0, 1.0, 1.0], (False, False)),
        ([0, 3, VOCAB], [1.0, 1.0, 1.0], (True, False)),
        ([0, VOCAB, 0], [1.0, 0.9, 1.0], (False, True)),
        ([5, VOCAB, 0], [1.0, 1.0, 0.5], (True, True)),
    ],
)
def test_padded_params_say_which_masks_a_row_applies(top_k, top_p, expected):
    """top_k 0 or at least the vocab and top_p 1 apply no mask."""
    params = _params_runner(top_k, top_p).build(0, 3, torch.zeros((4, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == expected
    assert params.top_k[3].item() == 0 and params.top_p[3].item() == 1.0


def test_padded_params_read_only_the_chunk_rows():
    """A mask in another chunk's rows does not turn a search on."""
    runner = _params_runner([3, 0, VOCAB, 0], [0.5, 1.0, 1.0, 1.0])
    params = runner.build(1, 3, torch.zeros((2, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == (False, False)
    params = runner.build(0, 2, torch.zeros((2, VOCAB)))
    assert (params.use_top_k, params.use_top_p) == (True, True)


def _sampler_inputs(seed):
    """Logits with grammar-masked entries, a greedy row, and drafts."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((ROWS, VOCAB), generator=g) * 3
    logits[1, ::5] = float("-inf")
    temperatures = torch.tensor([[1.0], [0.7], [0.0], [1.3]] * (ROWS // 4))
    u = torch.rand((ROWS, VOCAB), generator=g)
    noise = torch.randn((ROWS, VOCAB), generator=g)
    draft = torch.argmax(logits + noise, dim=-1).to(torch.int32)
    return logits, temperatures, u, draft


def _sample(logits, temperatures, u, top_k, top_p, use_top_k, use_top_p):
    runner = SimpleNamespace()
    runner._apply_temperature = MethodType(TPUModelRunner._apply_temperature, runner)
    return TPUModelRunner.sample_from_logits.__wrapped__(
        runner,
        logits,
        temperatures,
        u,
        top_k,
        top_p,
        all_greedy=False,
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )


@pytest.fixture
def eager_rejection(monkeypatch):
    """Runs the rejection sampler's program body eagerly on CPU."""
    monkeypatch.setattr(
        rs,
        "_random_rejection_sample_with_segment",
        rs._random_rejection_sample_with_segment.__wrapped__,
    )
    return rs.RejectionSampler()


def _reject(sampler, inputs, top_k, top_p, use_top_k, use_top_p):
    logits, temperatures, u, draft = inputs
    return sampler(
        draft_token_ids=draft,
        num_draft_tokens=torch.tensor([3, 3, 2], dtype=torch.int32),
        target_logits=logits,
        bonus_token_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        segment_ids=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64),
        group_indices=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.int64),
        max_draft_tokens=3,
        temperatures=temperatures,
        top_k=top_k,
        top_p=top_p,
        accept_u=u[:, 0].contiguous(),
        recover_u=u,
        do_sampling=True,
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )


@pytest.mark.parametrize("seed", range(4))
def test_sampling_without_masks_skips_searches_with_same_tokens(seed, eager_rejection):
    inputs = _sampler_inputs(seed)
    logits, temperatures, u, _ = inputs
    top_k = torch.zeros((ROWS, 1), dtype=torch.int32)
    top_p = torch.ones((ROWS, 1))
    full = _sample(logits, temperatures, u, top_k, top_p, True, True)
    skipped = _sample(logits, temperatures, u, top_k, top_p, False, False)
    assert torch.equal(full, skipped)

    full = _reject(eager_rejection, inputs, top_k, top_p, True, True)
    skipped = _reject(eager_rejection, inputs, top_k, top_p, False, False)
    assert torch.equal(full, skipped)


@pytest.mark.parametrize(
    "mask_row_top_k, mask_row_top_p", [(1, 1.0), (0, 0.05), (2, 0.05)]
)
def test_a_row_with_a_mask_is_still_searched(mask_row_top_k, mask_row_top_p):
    """Row 0 applies a mask that removes the token its noise would pick."""
    logits = torch.linspace(4.0, 0.0, VOCAB).repeat(ROWS, 1)
    temperatures = torch.ones((ROWS, 1))
    u = torch.full((ROWS, VOCAB), 0.5)
    u[:, -1] = 1.0 - 1e-6
    top_k = torch.zeros((ROWS, 1), dtype=torch.int32)
    top_p = torch.ones((ROWS, 1))
    top_k[0], top_p[0] = mask_row_top_k, mask_row_top_p
    runner = _params_runner(top_k.view(-1).tolist(), top_p.view(-1).tolist())
    params = runner.build(0, ROWS, logits)
    assert (params.use_top_k, params.use_top_p) == (
        mask_row_top_k > 0,
        mask_row_top_p < 1.0,
    )

    flagged = _sample(
        logits, temperatures, u, top_k, top_p, params.use_top_k, params.use_top_p
    )
    full = _sample(logits, temperatures, u, top_k, top_p, True, True)
    unmasked = _sample(logits, temperatures, u, top_k, top_p, False, False)
    assert torch.equal(flagged, full)
    assert flagged[0].item() == 0
    assert unmasked[0].item() == VOCAB - 1
    assert torch.equal(flagged[1:], unmasked[1:])


class _RejectionSampler:
    """Stands in for RejectionSampler; non-greedy calls go to `fn`."""

    synthetic_mode = False

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, **kwargs):
        return self._fn(**kwargs)

    def greedy_from_token_ids(self, **kwargs):
        return torch.zeros((2, 2), dtype=torch.int32)


def _record(calls):
    def sample_from_logits(logits, temperatures, u, top_k, top_p, **kwargs):
        calls.append(("sample", kwargs))
        return torch.zeros((logits.shape[0], 1), dtype=torch.int32)

    return sample_from_logits


@pytest.mark.parametrize("use_top_k, use_top_p", MASK_FLAG_COMBINATIONS)
def test_verify_chunk_passes_the_flags_to_both_samplers(use_top_k, use_top_p):
    calls = []
    logits = torch.zeros((2, VOCAB))
    params = PaddedSamplingParams(
        temperatures=torch.ones((2, 1)),
        top_k=torch.zeros((2, 1), dtype=torch.int32),
        top_p=torch.ones((2, 1)),
        use_top_k=use_top_k,
        use_top_p=use_top_p,
    )

    def rejection_sampler(**kwargs):
        calls.append(("reject", kwargs))
        return torch.zeros((2, 2), dtype=torch.int32)

    runner = SimpleNamespace(
        spec_gather_bonus_and_target_logits=lambda logits, b, t: (logits, logits),
        _build_padded_sampling_params=lambda start, end, logits: params,
        sample_from_logits_func=_record(calls),
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        rejection_sampler=_RejectionSampler(rejection_sampler),
    )
    md = SimpleNamespace(
        bonus_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        draft_token_ids=torch.zeros((2,), dtype=torch.int32),
        draft_lengths=torch.ones((2,), dtype=torch.int32),
        segment_ids=torch.tensor([0, 1], dtype=torch.int64),
        group_indices=torch.tensor([0, 0], dtype=torch.int64),
    )
    TPUModelRunner._sample_spec_verify_chunk(
        runner,
        logits,
        md,
        grammar_output=None,
        scheduler_output=SimpleNamespace(),
        cur_start_idx=0,
        cur_end_idx=2,
        all_greedy=False,
        sampling_generator=None,
    )
    assert [kind for kind, _ in calls] == ["sample", "reject"]
    for _, kwargs in calls:
        assert (kwargs["use_top_k"], kwargs["use_top_p"]) == (use_top_k, use_top_p)


def test_warmup_compiles_every_flag_combination(monkeypatch):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda *a, **kw: None)
    calls = []
    runner = SimpleNamespace(
        num_reqs_paddings=[2, 4],
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        _dummy_logits=lambda n: torch.zeros((n, VOCAB)),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        sample_from_logits_func=_record(calls),
    )
    TPUModelRunner._precompile_sample_from_logits(runner)
    warmed = [(k["all_greedy"], k["use_top_k"], k["use_top_p"]) for _, k in calls]
    expected = [(True, True, True)] + [(False, *f) for f in MASK_FLAG_COMBINATIONS]
    assert warmed == expected * 2

    rejected = []

    def rejection_sampler(**kwargs):
        if kwargs.get("do_sampling"):
            rejected.append((kwargs["use_top_k"], kwargs["use_top_p"]))
        return torch.zeros((2, 2), dtype=torch.int32)

    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        rejection_sampler=_RejectionSampler(rejection_sampler),
        num_tokens_paddings=[4],
        num_reqs_paddings=[2],
        max_num_reqs=2,
        _fused_greedy_head=None,
        vocab_size=VOCAB,
        _hidden_states_dtype=torch.float32,
        device=torch.device("cpu"),
        _spec_verify_bucket_reachable=lambda tokens, reqs: True,
        spec_bonus_and_target_token_ids=lambda logits, b, t: (
            torch.zeros(b.shape[0], dtype=torch.int32),
            torch.zeros(t.shape[0], dtype=torch.int32),
        ),
        spec_gather_bonus_and_target_logits=lambda logits, b, t: (
            logits[: b.shape[0]],
            logits,
        ),
        _greedy_sample=lambda logits: torch.zeros(
            (logits.shape[0], 1), dtype=torch.int32
        ),
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    TPUModelRunner._precompile_rejection_sampler(runner)
    assert rejected == list(MASK_FLAG_COMBINATIONS)
