# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parity of TPUModelRunner logprobs modes against vLLM's v1 Sampler on CPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    CANDIDATE_BLOCK,
    MASKED_LOGIT_VALUE,
    apply_top_k_top_p,
)
from vllm_torchtpu.runner import min_tokens, repetition_penalty, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

VOCAB = 2 * CANDIDATE_BLOCK
PADDED_ROWS = 16
MODES = ("processed_logprobs", "processed_logits")
ALL_MODES = ("raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits")


def _requests(params: list[dict], vocab: int) -> SimpleNamespace:
    """Per-row temperature/top_k/top_p as vLLM stores them in InputBatch."""
    sampling_params = [SamplingParams(**p) for p in params]
    greedy = [sp.sampling_type == SamplingType.GREEDY for sp in sampling_params]
    return SimpleNamespace(
        temperature=torch.tensor(
            [0.0 if g else sp.temperature for g, sp in zip(greedy, sampling_params)],
            dtype=torch.float32,
        ),
        top_k=torch.tensor(
            [sp.top_k if 0 < sp.top_k < vocab else vocab for sp in sampling_params],
            dtype=torch.int32,
        ),
        top_p=torch.tensor([sp.top_p for sp in sampling_params], dtype=torch.float32),
        greedy=torch.tensor(greedy),
    )


def _vllm_sampling_metadata(
    reqs: SimpleNamespace,
    vocab: int,
    *,
    max_num_logprobs: int | None = 0,
    no_penalties: bool = True,
    prompt_token_ids: torch.Tensor | None = None,
    repetition_penalties: torch.Tensor | None = None,
    output_token_ids: list[list[int]] | None = None,
    logitsprocs: LogitsProcessors | None = None,
) -> SamplingMetadata:
    n = reqs.temperature.shape[0]
    return SamplingMetadata(
        temperature=reqs.temperature.clone(),
        all_greedy=bool(reqs.greedy.all()),
        all_random=not bool(reqs.greedy.any()),
        top_p=None if bool((reqs.top_p >= 1.0).all()) else reqs.top_p.clone(),
        top_k=None if bool((reqs.top_k >= vocab).all()) else reqs.top_k.clone(),
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=torch.zeros(n),
        presence_penalties=torch.zeros(n),
        repetition_penalties=(
            torch.ones(n) if repetition_penalties is None else repetition_penalties
        ),
        output_token_ids=(
            [[] for _ in range(n)] if output_token_ids is None else output_token_ids
        ),
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors() if logitsprocs is None else logitsprocs,
    )


def _vllm_processed(
    logits: torch.Tensor, reqs: SimpleNamespace, mode: str
) -> torch.Tensor:
    """vLLM's `Sampler.sample` processed logits/logprobs."""
    metadata = _vllm_sampling_metadata(reqs, logits.shape[1])
    _, processed = Sampler(logprobs_mode=mode).sample(logits.clone().float(), metadata)
    assert processed is not None
    return processed.float()


def _tpu_runner(reqs: SimpleNamespace, mode: str, vocab: int) -> SimpleNamespace:
    runner = SimpleNamespace(
        vocab_size=vocab,
        speculative_config=None,
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=reqs.temperature,
            top_k_cpu_tensor=reqs.top_k,
            top_p_cpu_tensor=reqs.top_p,
            no_top_k=bool((reqs.top_k >= vocab).all()),
            no_top_p=bool((reqs.top_p >= 1.0).all()),
        ),
        model_config=SimpleNamespace(logprobs_mode=mode, max_logprobs=4),
    )
    for name in (
        "_apply_temperature",
        "_build_padded_sampling_params",
        "_sampler_applies_topk_topp",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _tpu_sample(
    logits: torch.Tensor,
    reqs: SimpleNamespace,
    mode: str,
    topk_candidates: int = 0,
    seed: int = 0,
    apply_topk_topp: bool | None = None,
):
    """Sampled tokens, processed logits, uniform noise, and gathered logprobs
    of the TPU sampler on a batch padded to PADDED_ROWS."""
    n, vocab = logits.shape
    runner = _tpu_runner(reqs, mode, vocab)
    padded = torch.zeros(PADDED_ROWS, vocab, dtype=logits.dtype)
    padded[:n] = logits
    temperatures, top_k, top_p = runner._build_padded_sampling_params(0, n, padded)
    if apply_topk_topp is None:
        apply_topk_topp = runner._sampler_applies_topk_topp()
    u = torch.rand(
        padded.shape,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    )
    with torch.compiler.set_stance("force_eager"):
        tokens, processed = TPUModelRunner.sample_with_processed_logits(
            runner,
            padded,
            temperatures,
            u,
            top_k,
            top_p,
            topk_candidates=topk_candidates,
            apply_topk_topp=apply_topk_topp,
        )
        gathered = TPUModelRunner.gather_logprobs(runner, processed, tokens)
    return tokens[:n], processed[:n], u[:n], gathered


def _as_mode(processed: torch.Tensor, mode: str) -> torch.Tensor:
    if mode.endswith("_logprobs"):
        return processed.log_softmax(dim=-1, dtype=torch.float32)
    return processed.to(torch.float32)


def _assert_matches_vllm(
    tpu: torch.Tensor,
    vllm: torch.Tensor,
    mode: str,
    logits: torch.Tensor,
    reqs: SimpleNamespace,
) -> None:
    """`tpu` (processed logits) matches vLLM's processed logits/logprobs."""
    temperature = torch.where(reqs.greedy, 1.0, reqs.temperature).unsqueeze(1)
    scaled = logits.float() / temperature
    tpu_masked = torch.isneginf(tpu)
    vllm_masked = torch.isneginf(vllm)
    assert not bool((tpu_masked & ~vllm_masked).any()), "TPU masks a kept entry"
    extra = vllm_masked & ~tpu_masked
    if bool(extra.any()):
        assert bool((reqs.top_p.unsqueeze(1).expand_as(extra)[extra] >= 1.0).all())
        assert torch.equal(tpu[extra], scaled[extra]), "masked to a finite value"
        tpu_probs = tpu.softmax(dim=-1, dtype=torch.float32)
        assert bool((tpu_probs[extra] == 0).all()), "vLLM masks an entry with mass"
    kept = ~vllm_masked
    if mode == "processed_logits":
        torch.testing.assert_close(tpu[kept], vllm[kept], rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            _as_mode(tpu, mode)[kept], vllm[kept], rtol=1e-6, atol=1e-5
        )


def _pick(options: list, generator: torch.Generator):
    idx = int(torch.randint(len(options), (1,), generator=generator))
    return options[idx]


def _random_params(generator: torch.Generator, rows: int, vocab: int) -> list[dict]:
    params: list[dict] = []
    any_random = False
    for _ in range(rows):
        if _pick([True, False, False], generator):
            params.append({"temperature": 0.0})
            continue
        any_random = True
        params.append(
            {
                "temperature": _pick([0.3, 0.7, 1.0, 1.6], generator),
                "top_k": _pick(
                    [-1, 0, 1, 2, 7, 50, vocab - 1, vocab, vocab + 13], generator
                ),
                "top_p": _pick([1.0, 1.0, 0.99, 0.9, 0.6, 0.2], generator),
            }
        )
    if not any_random:
        params[0] = {"temperature": 0.8, "top_k": 5, "top_p": 0.9}
    return params


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(40))
def test_mixed_batch_matches_vllm_sampler(seed: int, mode: str) -> None:
    """Random batches of greedy and random rows with mixed settings match vLLM."""
    generator = torch.Generator().manual_seed(seed)
    rows = int(torch.randint(1, 13, (1,), generator=generator))
    logits = 3.0 * torch.randn(rows, VOCAB, generator=generator)
    reqs = _requests(_random_params(generator, rows, VOCAB), VOCAB)

    tokens, processed, u, gathered = _tpu_sample(logits, reqs, mode, seed=seed)

    _assert_matches_vllm(
        processed, _vllm_processed(logits, reqs, mode), mode, logits, reqs
    )
    greedy = reqs.greedy
    assert torch.equal(tokens[greedy], logits[greedy].argmax(-1, keepdim=True))
    assert bool(torch.isfinite(processed.gather(-1, tokens)).all())
    finfo = torch.finfo(u.dtype)
    gumbel = -torch.log(-torch.log(u.clamp(finfo.tiny, 1.0 - finfo.eps)))
    winners = (processed + gumbel).argmax(-1, keepdim=True)
    assert torch.equal(tokens[~greedy], winners[~greedy])
    expected = _as_mode(processed, mode).gather(-1, tokens)
    torch.testing.assert_close(gathered.logprobs[:rows, :1], expected)


@pytest.mark.parametrize("mode", MODES)
def test_greedy_rows_in_mixed_batch_match_vllm(mode: str) -> None:
    """Greedy rows in a mixed batch keep their unscaled, unmasked logits."""
    logits = torch.randn(3, VOCAB, generator=torch.Generator().manual_seed(1))
    reqs = _requests(
        [
            {"temperature": 0.0, "top_k": 3, "top_p": 0.5},
            {"temperature": 0.9, "top_k": 4, "top_p": 0.8},
            {"temperature": 0.0},
        ],
        VOCAB,
    )
    assert reqs.top_k.tolist() == [VOCAB, 4, VOCAB]
    _, processed, _, _ = _tpu_sample(logits, reqs, mode)
    vllm = _vllm_processed(logits, reqs, mode)
    _assert_matches_vllm(processed, vllm, mode, logits, reqs)
    for row in (0, 2):
        torch.testing.assert_close(processed[row], logits[row], rtol=0, atol=0)
        torch.testing.assert_close(
            _as_mode(processed, mode)[row], vllm[row], rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize("mode", MODES)
def test_all_greedy_batch_matches_vllm(mode: str) -> None:
    """An all-greedy batch returns the logits unscaled and unmasked via TPU runner."""
    logits = torch.randn(4, VOCAB, generator=torch.Generator().manual_seed(2))
    reqs = _requests([{"temperature": 0.0}] * 4, VOCAB)
    runner = _tpu_runner(reqs, mode, VOCAB)
    runner._greedy_sample = lambda x: torch.argmax(x, dim=-1, keepdim=True)
    _, _, processed = TPUModelRunner._sample_chunk_logits(
        runner,
        logits,
        cur_start_idx=0,
        cur_end_idx=4,
        all_greedy=True,
        grammar_output=None,
        sampling_generator=None,
        need_processed_logits=True,
    )
    assert processed is not None
    _assert_matches_vllm(
        processed, _vllm_processed(logits, reqs, mode), mode, logits, reqs
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("top_k", "expected_kept"),
    [(-1, VOCAB), (0, VOCAB), (VOCAB, VOCAB), (VOCAB + 100, VOCAB), (2, 4)],
)
def test_top_k_edges_and_ties_match_vllm(
    mode: str, top_k: int, expected_kept: int
) -> None:
    """top_k -1 / 0 / >= vocab keeps every token; ties at the k-th logit stay."""
    logits = torch.linspace(-4.0, 0.0, VOCAB).unsqueeze(0).repeat(2, 1)
    logits[:, 7] = 5.0
    logits[:, [11, 40, 99]] = 3.0
    reqs = _requests(
        [{"temperature": 1.0, "top_k": top_k}, {"temperature": 0.0}], VOCAB
    )
    _, processed, _, _ = _tpu_sample(logits, reqs, mode)
    _assert_matches_vllm(
        processed, _vllm_processed(logits, reqs, mode), mode, logits, reqs
    )
    assert int(torch.isfinite(processed[0]).sum()) == expected_kept


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("top_p", [1.0, 0.999, 0.5, 1e-6])
def test_top_p_edges_match_vllm(mode: str, top_p: float) -> None:
    """top_p 1 keeps everything; a tiny top_p keeps only the argmax."""
    logits = torch.randn(2, VOCAB, generator=torch.Generator().manual_seed(3))
    reqs = _requests(
        [{"temperature": 0.7, "top_p": top_p}, {"temperature": 1.2, "top_p": 0.9}],
        VOCAB,
    )
    _, processed, _, _ = _tpu_sample(logits, reqs, mode)
    _assert_matches_vllm(
        processed, _vllm_processed(logits, reqs, mode), mode, logits, reqs
    )
    if top_p == 1e-6:
        assert int(torch.isfinite(processed[0]).sum()) == 1


def test_top_p_one_zero_mass_entries_are_the_only_difference() -> None:
    """Entries 200 nats below the max have float32 probability 0: vLLM's top-p
    op masks them at top_p=1 when another row sets top_p, while TPU skips top-p."""
    logits = torch.zeros(2, VOCAB)
    logits[:, 0] = 200.0
    reqs = _requests(
        [{"temperature": 1.0, "top_p": 1.0}, {"temperature": 1.0, "top_p": 0.5}],
        VOCAB,
    )
    _, processed, _, _ = _tpu_sample(logits, reqs, "processed_logits")
    vllm = _vllm_processed(logits, reqs, "processed_logits")
    assert int(torch.isfinite(vllm[0]).sum()) == 1
    assert int(torch.isfinite(processed[0]).sum()) == VOCAB
    _assert_matches_vllm(processed, vllm, "processed_logits", logits, reqs)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(8))
def test_candidate_path_matches_vllm(mode: str, seed: int) -> None:
    """The candidate-block top-k path returns the same processed logits as vLLM."""
    vocab = 8 * CANDIDATE_BLOCK
    generator = torch.Generator().manual_seed(seed)
    rows = 6
    logits = 2.0 * torch.randn(rows, vocab, generator=generator)
    params = [
        {"temperature": t, "top_k": k, "top_p": p}
        for t, k, p in zip(
            [0.5, 1.0, 1.4, 0.8, 1.0, 0.6],
            [1, 2, 2, 1, 2, 2],
            [1.0, 0.9, 0.5, 1.0, 0.99, 0.7],
        )
    ]
    reqs = _requests(params, vocab)
    tokens, processed, _, _ = _tpu_sample(logits, reqs, mode, topk_candidates=4)
    _assert_matches_vllm(
        processed, _vllm_processed(logits, reqs, mode), mode, logits, reqs
    )
    assert bool(torch.isfinite(processed.gather(-1, tokens)).all())


def test_sample_from_logits_draws_the_same_tokens() -> None:
    """`sample_with_processed_logits` and `sample_from_logits` draw identical tokens."""
    generator = torch.Generator().manual_seed(4)
    logits = torch.randn(PADDED_ROWS, VOCAB, generator=generator)
    reqs = _requests(_random_params(generator, PADDED_ROWS, VOCAB), VOCAB)
    tokens, _, _, _ = _tpu_sample(logits, reqs, "processed_logprobs", seed=4)
    runner = _tpu_runner(reqs, "processed_logprobs", VOCAB)
    temperatures, top_k, top_p = runner._build_padded_sampling_params(
        0, PADDED_ROWS, logits
    )
    u = torch.rand(logits.shape, generator=torch.Generator().manual_seed(4))
    with torch.compiler.set_stance("force_eager"):
        plain = TPUModelRunner.sample_from_logits(
            runner,
            logits,
            temperatures,
            u,
            top_k,
            top_p,
            apply_topk_topp=runner._sampler_applies_topk_topp(),
        )
    assert torch.equal(tokens, plain)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("apply_topk_topp", [False, True])
def test_bfloat16_large_offset_t07_no_topk_topp_matches_vllm(
    mode: str, apply_topk_topp: bool
) -> None:
    """Regression test for bfloat16 T=0.7 vocab-wiping and quantization drift
    at large logit offsets (20-35) with top_k=-1, top_p=1.0."""
    gen = torch.Generator().manual_seed(7)
    rows = 8
    logits = (25.0 + 3.0 * torch.randn(rows, VOCAB, generator=gen)).to(torch.bfloat16)
    reqs = _requests([{"temperature": 0.7, "top_k": -1, "top_p": 1.0}] * rows, VOCAB)

    runner = _tpu_runner(reqs, mode, VOCAB)
    assert not runner._sampler_applies_topk_topp()
    padded = torch.zeros(PADDED_ROWS, VOCAB, dtype=torch.bfloat16)
    padded[:rows] = logits
    temps, _, _ = runner._build_padded_sampling_params(0, rows, padded)
    assert temps.dtype == torch.float32
    torch.testing.assert_close(
        temps[:rows], torch.full((rows, 1), 0.7, dtype=torch.float32), rtol=0, atol=0
    )

    tokens, processed, _, gathered = _tpu_sample(
        logits, reqs, mode, seed=7, apply_topk_topp=apply_topk_topp
    )
    assert processed.dtype == torch.float32
    assert bool(torch.isfinite(processed).all())
    assert int(torch.isneginf(processed).sum()) == 0
    assert bool(torch.isfinite(gathered.logprobs[:rows]).all())

    vllm = _vllm_processed(logits, reqs, mode)
    _assert_matches_vllm(processed, vllm, mode, logits, reqs)
    expected_sampled = vllm.gather(-1, tokens)
    torch.testing.assert_close(
        gathered.logprobs[:rows, :1], expected_sampled, rtol=1e-6, atol=1e-5
    )


def _assert_bf16_matches_vllm_allowing_topp_ties(
    tpu: torch.Tensor,
    vllm: torch.Tensor,
    mode: str,
    logits: torch.Tensor,
    reqs: SimpleNamespace,
) -> None:
    """Checks `tpu` (processed logits) against vLLM's `Sampler` for bfloat16
    logits where ties at a `top_p < 1` cutoff are common."""
    temperature = torch.where(reqs.greedy, 1.0, reqs.temperature).unsqueeze(1)
    scaled = logits.float() / temperature
    tpu_kept = ~torch.isneginf(tpu)
    vllm_kept = ~torch.isneginf(vllm)
    assert int((~tpu_kept & vllm_kept).sum()) == 0, "TPU masked a vLLM-kept entry"
    torch.testing.assert_close(tpu[tpu_kept], scaled[tpu_kept], rtol=0, atol=0)
    tpu_mode = _as_mode(tpu, mode)
    rtol = 0.0 if mode == "processed_logits" else 1e-6
    atol = 0.0 if mode == "processed_logits" else 1e-5
    non_tie_rows = 0
    for r in range(tpu.shape[0]):
        extra = tpu_kept[r] & ~vllm_kept[r]
        if bool(extra.any()):
            min_vllm = scaled[r, vllm_kept[r]].min()
            assert bool((scaled[r, extra] == min_vllm).all()), (
                "extra kept token is not a top_p cutoff tie"
            )
            continue
        non_tie_rows += 1
        torch.testing.assert_close(
            tpu_mode[r, vllm_kept[r]], vllm[r, vllm_kept[r]], rtol=rtol, atol=atol
        )
    assert non_tie_rows > 0


@pytest.mark.parametrize("mode", MODES)
def test_bfloat16_large_offset_t07_active_topk_topp_matches_vllm(mode: str) -> None:
    """bfloat16 logits at Qwen3.5 scale (20-35) with T=0.7 and active
    top_k=50, top_p=0.9 (and top_k=50, top_p=1.0): zero -inf on surviving
    tokens and < 1e-5 parity with vLLM's v1 Sampler."""
    gen = torch.Generator().manual_seed(11)
    rows = 8
    logits = (25.0 + 3.0 * torch.randn(rows, VOCAB, generator=gen)).to(torch.bfloat16)

    reqs_k = _requests([{"temperature": 0.7, "top_k": 50, "top_p": 1.0}] * rows, VOCAB)
    tokens_k, processed_k, _, gathered_k = _tpu_sample(logits, reqs_k, mode, seed=11)
    vllm_k = _vllm_processed(logits, reqs_k, mode)
    _assert_matches_vllm(processed_k, vllm_k, mode, logits, reqs_k)
    torch.testing.assert_close(
        gathered_k.logprobs[:rows, :1],
        vllm_k.gather(-1, tokens_k),
        rtol=1e-6,
        atol=1e-5,
    )

    reqs = _requests([{"temperature": 0.7, "top_k": 50, "top_p": 0.9}] * rows, VOCAB)
    runner = _tpu_runner(reqs, mode, VOCAB)
    assert runner._sampler_applies_topk_topp()

    tokens, processed, _, gathered = _tpu_sample(logits, reqs, mode, seed=11)
    assert processed.dtype == torch.float32
    assert bool(torch.isfinite(processed.gather(-1, tokens)).all())
    assert bool(torch.isfinite(gathered.logprobs[:rows, :1]).all())

    vllm = _vllm_processed(logits, reqs, mode)
    _assert_bf16_matches_vllm_allowing_topp_ties(processed, vllm, mode, logits, reqs)
    expected_self = _as_mode(processed, mode).gather(-1, tokens)
    torch.testing.assert_close(
        gathered.logprobs[:rows, :1], expected_self, rtol=1e-6, atol=1e-5
    )


@pytest.mark.parametrize(
    "params",
    [
        {"temperature": 0.7, "top_k": -1, "top_p": 1.0},
        {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
    ],
)
def test_div_vs_rcp_ulp_stress_no_false_neginf(monkeypatch, params: dict) -> None:
    """Simulates TPU XLA rematerialization where `logits / T` vs `logits * rcp(T)`
    differs by 1 ULP across `apply_top_k_top_p`."""
    gen = torch.Generator().manual_seed(19)
    rows = 8
    logits = (25.0 + 3.0 * torch.randn(rows, VOCAB, generator=gen)).to(torch.bfloat16)
    reqs = _requests([params] * rows, VOCAB)
    _, clean_processed, _, _ = _tpu_sample(
        logits, reqs, "processed_logprobs", seed=19, apply_topk_topp=True
    )
    clean_kept = ~torch.isneginf(clean_processed)
    clean_logprobs = _as_mode(clean_processed, "processed_logprobs")

    def ulp_drifted_top_k_top_p(scaled, top_k, top_p, topk_candidates=0):
        masked = apply_top_k_top_p(scaled, top_k, top_p, topk_candidates)
        kept = masked > MASKED_LOGIT_VALUE
        drifted = torch.nextafter(masked, torch.full_like(masked, float("inf")))
        return torch.where(kept, drifted, masked)

    monkeypatch.setattr(tpu_runner, "apply_top_k_top_p", ulp_drifted_top_k_top_p)
    tokens, processed, _, gathered = _tpu_sample(
        logits, reqs, "processed_logprobs", seed=19, apply_topk_topp=True
    )
    assert torch.equal(~torch.isneginf(processed), clean_kept)
    assert int(torch.isneginf(processed[clean_kept]).sum()) == 0
    assert bool(torch.isfinite(processed.gather(-1, tokens)).all())
    assert bool(torch.isfinite(gathered.logprobs[:rows, :1]).all())
    torch.testing.assert_close(
        _as_mode(processed, "processed_logprobs")[clean_kept],
        clean_logprobs[clean_kept],
        rtol=1e-5,
        atol=1e-5,
    )

    if params["top_k"] == -1 and params["top_p"] == 1.0:
        ref_vllm = _vllm_processed(logits, reqs, "processed_logprobs")

        def _must_not_call(*_args, **_kwargs):
            raise AssertionError("apply_top_k_top_p called on no-op fast path")

        monkeypatch.setattr(tpu_runner, "apply_top_k_top_p", _must_not_call)
        tokens_fast, processed_fast, _, gathered_fast = _tpu_sample(
            logits, reqs, "processed_logprobs", seed=19
        )
        assert bool(torch.isfinite(processed_fast).all())
        assert bool(torch.isfinite(gathered_fast.logprobs[:rows]).all())
        _assert_matches_vllm(
            processed_fast, ref_vllm, "processed_logprobs", logits, reqs
        )
        assert torch.equal(tokens_fast, tokens)


@pytest.mark.parametrize("seed", range(8))
def test_bfloat16_mixed_batch_sampling_and_logprobs_parity(seed: int) -> None:
    """bfloat16 Qwen-scale mixed batches keep 100% sampling parity with
    `sample_from_logits` and finite processed_logprobs matching vLLM."""
    gen = torch.Generator().manual_seed(100 + seed)
    rows = 8
    logits = (25.0 + 3.0 * torch.randn(rows, VOCAB, generator=gen)).to(torch.bfloat16)
    reqs = _requests(
        [
            {"temperature": 0.0},
            {"temperature": 0.7, "top_k": -1, "top_p": 1.0},
            {"temperature": 0.7, "top_k": 50, "top_p": 0.9},
            {"temperature": 1.0, "top_k": -1, "top_p": 1.0},
            {"temperature": 0.7, "top_k": 20, "top_p": 1.0},
            {"temperature": 0.7, "top_k": -1, "top_p": 0.95},
            {"temperature": 0.0},
            {"temperature": 0.5, "top_k": 50, "top_p": 0.9},
        ],
        VOCAB,
    )
    tokens, processed, _, gathered = _tpu_sample(
        logits, reqs, "processed_logprobs", seed=seed
    )
    runner = _tpu_runner(reqs, "processed_logprobs", VOCAB)
    padded = torch.zeros(PADDED_ROWS, VOCAB, dtype=torch.bfloat16)
    padded[:rows] = logits
    temperatures, top_k, top_p = runner._build_padded_sampling_params(0, rows, padded)
    u = torch.rand(
        padded.shape,
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
    )
    with torch.compiler.set_stance("force_eager"):
        plain = TPUModelRunner.sample_from_logits(
            runner,
            padded,
            temperatures,
            u,
            top_k,
            top_p,
            apply_topk_topp=runner._sampler_applies_topk_topp(),
        )
    assert torch.equal(tokens, plain[:rows])
    assert bool(torch.isfinite(processed.gather(-1, tokens)).all())
    assert bool(torch.isfinite(gathered.logprobs[:rows, :1]).all())
    vllm = _vllm_processed(logits, reqs, "processed_logprobs")
    for row in (0, 1, 3, 6):
        assert int(torch.isneginf(processed[row]).sum()) == 0
        torch.testing.assert_close(
            _as_mode(processed[row], "processed_logprobs"),
            vllm[row],
            rtol=1e-6,
            atol=1e-5,
        )


def _plain_chunk_runner(
    reqs: SimpleNamespace,
    mode: str,
    vocab: int,
    *,
    stop_token: int,
    rep_penalty: float,
    blocked_token: int,
) -> tuple[SimpleNamespace, min_tokens.MinTokensState]:
    runner = _tpu_runner(reqs, mode, vocab)
    runner._stage_timing = MagicMock()
    runner._greedy_sample = lambda x: torch.argmax(x, dim=-1, keepdim=True)
    runner._sampler_topk_candidates = lambda *_a: 0
    runner.structured_decoding_manager = SimpleNamespace(
        mask_logits=lambda x, *_a: x.clone().index_fill_(
            1, torch.tensor([blocked_token]), float("-inf")
        )
    )
    min_tracker = min_tokens.MinTokensTracker()
    min_tracker.add("r0", SamplingParams(min_tokens=4, stop_token_ids=[stop_token]))
    min_state = min_tracker.batch_state({"r0": 0}, lambda _rid: 0)
    rep_tracker = repetition_penalty.RepetitionPenaltyTracker(
        vocab, PADDED_ROWS, async_scheduling=False, speculative_decoding=False
    )
    runner._repetition_penalty = rep_tracker
    runner.requests = {"r0": SimpleNamespace(prompt_token_ids=[5], output_token_ids=[])}
    runner.input_batch.req_ids = ["r0"]
    runner.input_batch.req_id_to_index = {"r0": 0}
    runner.input_batch.req_output_token_ids = [[]]
    runner.input_batch.repetition_penalties_reqs = (
        {"r0"} if rep_penalty != 1.0 else set()
    )
    runner.input_batch.repetition_penalties_cpu = np.array(
        [rep_penalty], dtype=np.float32
    )
    for name in (
        "_mask_min_tokens",
        "_sample_chunk_logits",
        "_sample_plain_chunk",
        "sample_from_logits",
        "sample_with_processed_logits",
        "gather_logprobs",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    runner.sample_from_logits_func = runner.sample_from_logits
    return runner, min_state


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("all_greedy", [True, False], ids=["greedy", "random"])
def test_four_modes_plain_chunk_with_grammar_min_tokens_and_rep_pen_match_vllm(
    monkeypatch, mode: str, all_greedy: bool
) -> None:
    """All 4 logprobs_mode values across greedy and random (T=0.7, top_k=4)
    with grammar, min_tokens, and repetition_penalty match vLLM's Sampler."""
    monkeypatch.setattr("vllm.utils.torch_utils.is_pin_memory_available", lambda: False)
    monkeypatch.setattr(
        "vllm.v1.sample.ops.penalties.is_pin_memory_available",
        lambda: False,
        raising=False,
    )
    vocab = 32
    raw = torch.linspace(-2.0, 4.0, vocab, dtype=torch.float32).unsqueeze(0)
    # 31 is grammar-blocked, 30 is stop (below min_tokens), 5 is penalized.
    raw[0, 31], raw[0, 30], raw[0, 5], raw[0, 12] = 12.0, 10.0, 8.0, 6.0
    params = (
        [{"temperature": 0.0}]
        if all_greedy
        else [{"temperature": 0.7, "top_k": 4, "top_p": 1.0}]
    )
    reqs = _requests(params, vocab)
    runner, min_state = _plain_chunk_runner(
        reqs, mode, vocab, stop_token=30, rep_penalty=2.0, blocked_token=31
    )
    padded = torch.zeros(8, vocab, dtype=torch.float32)
    padded[0] = raw[0]
    rep_active = runner._repetition_penalty.begin_step(runner.input_batch)

    with torch.compiler.set_stance("force_eager"):
        selected, lp_logits = runner._sample_plain_chunk(
            padded,
            cur_start_idx=0,
            cur_end_idx=1,
            top_tokens=False,
            all_greedy=all_greedy,
            sampling_generator=torch.Generator().manual_seed(0),
            grammar_output=object(),
            scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={}),
            is_spec_step=False,
            min_toks=min_state,
            rep_pen=rep_active,
            need_processed_logits=mode in MODES,
        )
        tpu_lp = runner.gather_logprobs(lp_logits[:1], selected[:1])

    # Post-grammar logits fed to vLLM's Sampler (grammar runs before Sampler in vLLM).
    post_grammar = raw.clone()
    post_grammar[0, 31] = float("-inf")

    class _MinTokensProc:
        def apply(self, x: torch.Tensor) -> torch.Tensor:
            x[:, 30] = float("-inf")
            return x

    procs = LogitsProcessors()
    procs.non_argmax_invariant.append(_MinTokensProc())
    vllm_meta = _vllm_sampling_metadata(
        reqs,
        vocab,
        max_num_logprobs=4,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[5]], dtype=torch.int64),
        repetition_penalties=torch.tensor([2.0]),
        logitsprocs=procs,
    )
    vllm_out = Sampler(logprobs_mode=mode).forward(post_grammar.clone(), vllm_meta)
    assert vllm_out.logprobs_tensors is not None

    # Compare top-4 logprobs/indices and sampled token score against vLLM.
    torch.testing.assert_close(
        tpu_lp.logprobs[0, 1:],
        vllm_out.logprobs_tensors.logprobs[0, 1:],
        rtol=1e-6,
        atol=1e-5,
    )
    assert torch.equal(
        tpu_lp.logprob_token_ids[0, 1:],
        vllm_out.logprobs_tensors.logprob_token_ids[0, 1:],
    )
    tok = int(selected[0, 0].item())
    assert tok not in (30, 31)
    vllm_full_meta = _vllm_sampling_metadata(
        reqs,
        vocab,
        max_num_logprobs=-1,
        no_penalties=False,
        prompt_token_ids=torch.tensor([[5]], dtype=torch.int64),
        repetition_penalties=torch.tensor([2.0]),
        logitsprocs=procs,
    )
    vllm_full = Sampler(logprobs_mode=mode).forward(
        post_grammar.clone(), vllm_full_meta
    )
    assert vllm_full.logprobs_tensors is not None
    torch.testing.assert_close(
        tpu_lp.logprobs[0, 0],
        vllm_full.logprobs_tensors.logprobs[0, tok],
        rtol=1e-6,
        atol=1e-5,
    )


def test_raw_logprobs_keeps_logits_dtype_temperature_staging() -> None:
    """Default raw_logprobs keeps BF16 temperature staging so default path is unchanged."""
    logits = torch.zeros(PADDED_ROWS, VOCAB, dtype=torch.bfloat16)
    reqs = _requests([{"temperature": 0.7}] * 4, VOCAB)
    raw_runner = _tpu_runner(reqs, "raw_logprobs", VOCAB)
    raw_temps, _, _ = raw_runner._build_padded_sampling_params(0, 4, logits)
    assert raw_temps.dtype == torch.bfloat16
    proc_runner = _tpu_runner(reqs, "processed_logprobs", VOCAB)
    proc_temps, _, _ = proc_runner._build_padded_sampling_params(0, 4, logits)
    assert proc_temps.dtype == torch.float32
