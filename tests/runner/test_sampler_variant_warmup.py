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
"""TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS: the synthetic one-row verify warms
only the sampler variants serving can pick, and serving's verify steps then
compile nothing new."""

import functools
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import (
    SAMPLER_VARIANTS,
    TPUModelRunner,
    default_sampler_variant,
    sampler_variant_name,
)

ENV = "TPU_SPEC_PRECOMPILE_SAMPLER_VARIANTS"
K = 3
VOCAB = 2 * CANDIDATE_BLOCK
BLOCKS = 2

# Default sampling params (generation configs) and the variant a batch on
# them picks. MiniMax-M3.1-Flash ships temperature 1.0, top_p 0.95.
DEFAULTS = [
    ({"temperature": 1.0, "top_p": 0.95}, "top_p"),
    ({}, "no_topk_topp"),
    ({"temperature": 0.0, "top_p": 0.95, "top_k": 20}, "no_topk_topp"),
    ({"temperature": 0.7, "top_k": 1}, "topk_candidates"),
    ({"temperature": 0.7, "top_k": 50, "top_p": 0.9}, "full"),
    ({"top_k": VOCAB}, "no_topk_topp"),
]


def _bind(runner, *names):
    for name in names:
        setattr(runner, name, functools.partial(getattr(TPUModelRunner, name), runner))


def _input_batch(defaults: dict, num_reqs: int):
    """The sampling fields InputBatch.add_request keeps for ``num_reqs``
    requests that all take ``defaults`` (vLLM's own SamplingParams applies
    the greedy rules)."""
    params = SamplingParams(**defaults)
    top_k = params.top_k if 0 < params.top_k < VOCAB else VOCAB
    return SimpleNamespace(
        no_top_k=not (0 < params.top_k < VOCAB),
        no_top_p=not (params.top_p < 1),
        top_k_cpu_tensor=torch.full((num_reqs,), top_k, dtype=torch.int32),
    )


def _runner(defaults: dict, num_reqs: int = 4, blocks: int = BLOCKS):
    runner = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=rs.RejectionSampler(
            SimpleNamespace(
                rejection_sample_method="synthetic",
                synthetic_acceptance_rates=[0.5] * K,
            ),
            torch.device("cpu"),
        ),
        vocab_size=VOCAB,
        _hidden_states_dtype=torch.float32,
        device="cpu",
        model_config=SimpleNamespace(
            get_diff_sampling_param=lambda: dict(defaults), seed=0
        ),
        input_batch=_input_batch(defaults, num_reqs),
        _topk_candidate_blocks=lambda: blocks,
        _synthetic_rng_counter=None,
        _spec_glue_logged=set(),
    )
    _bind(
        runner,
        "_warm_sampler_variant_names",
        "_warm_sampler_variants",
        "_note_sampler_variant",
        "_sampler_applies_topk_topp",
        "_sampler_applies_top_k",
        "_sampler_topk_candidates",
        "_get_synthetic_rng_counter",
        "_log_spec_glue_once",
        "_plan_synthetic_verify",
        "_sample_synthetic_verify",
    )
    return runner


def _served_variant(runner, num_reqs: int) -> str:
    return sampler_variant_name(
        runner._sampler_topk_candidates(0, num_reqs, None),
        runner._sampler_applies_topk_topp(),
        runner._sampler_applies_top_k(),
    )


@pytest.mark.parametrize("defaults,expected", DEFAULTS)
def test_auto_warms_the_variant_default_requests_pick(monkeypatch, defaults, expected):
    monkeypatch.setenv(ENV, "auto")
    assert default_sampler_variant(defaults, VOCAB, BLOCKS) == expected
    runner = _runner(defaults)
    # The runner's own per-step choice on a batch of default requests.
    assert _served_variant(runner, 4) == expected
    assert runner._warm_sampler_variant_names() == (expected,)


def test_unset_warms_every_variant_in_the_old_order(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    runner = _runner({"temperature": 1.0, "top_p": 0.95})
    assert runner._warm_sampler_variant_names() == SAMPLER_VARIANTS
    assert runner._warm_sampler_variants() == [
        {},
        {"apply_topk_topp": False},
        {"apply_top_k": False},
        {"topk_candidates": BLOCKS},
    ]
    no_blocks = _runner({}, blocks=0)
    assert no_blocks._warm_sampler_variants() == [
        {},
        {"apply_topk_topp": False},
        {"apply_top_k": False},
    ]


def test_explicit_list_and_unknown_name(monkeypatch):
    monkeypatch.setenv(ENV, "no_topk_topp, top_p")
    runner = _runner({})
    assert runner._warm_sampler_variants() == [
        {"apply_topk_topp": False},
        {"apply_top_k": False},
    ]
    monkeypatch.setenv(ENV, "top_p,bogus")
    with pytest.raises(ValueError, match="bogus"):
        _runner({})._warm_sampler_variant_names()


def _recording(fn, graphs):
    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    return torch.compile(
        fn._torchdynamo_orig_callable, backend=record, fullgraph=True, dynamic=False
    )


def _warm_then_serve(monkeypatch, env, defaults, num_tokens=16, num_reqs=4):
    """Warm one (padded_logits_length, padded_num_reqs) pair the way
    _precompile_rejection_sampler does, then run serving's verify step on
    inputs built the way SpeculativeDecodingManager builds them. Returns the
    graphs each compiled, by function."""
    if env is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, env)
    monkeypatch.setenv("TPU_SPEC_SYNTHETIC_INGRAPH_RNG", "1")
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    names = (
        "synthetic_verify_rows",
        "synthetic_verify_rows_rng",
        "synthetic_sample_and_assemble",
        "synthetic_sample_and_assemble_rng",
    )
    graphs = {name: [] for name in names}
    for name in names:
        monkeypatch.setattr(
            tpu_runner, name, _recording(getattr(rs, name), graphs[name])
        )
    runner = _runner(defaults, num_reqs)
    runner._precompile_synthetic_verify = functools.partial(
        TPUModelRunner._precompile_synthetic_verify, runner
    )
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(
            cache_size_limit=1024, accumulated_recompile_limit=1024
        ):
            runner._precompile_synthetic_verify(
                torch.zeros(num_tokens, dtype=torch.int32),
                torch.zeros(num_tokens, dtype=torch.int32),
                torch.zeros(num_tokens, dtype=torch.int64),
                torch.zeros(num_tokens, dtype=torch.int32),
                torch.zeros(num_reqs, dtype=torch.int32),
                torch.zeros(num_reqs, dtype=torch.int32),
                torch.zeros(num_tokens, dtype=torch.float32),
            )
            warmed = {name: len(g) for name, g in graphs.items()}
            # Serving: every metadata tensor is its own int32/int64 upload
            # (speculative_decoding_manager.get_spec_decode_metadata).
            md = SimpleNamespace(
                draft_token_ids=torch.randint(
                    0, VOCAB, (num_tokens,), dtype=torch.int32
                ),
                draft_lengths=torch.full((num_reqs,), K, dtype=torch.int32),
                segment_ids=torch.clamp(
                    torch.arange(num_tokens, dtype=torch.int64) // K, max=num_reqs
                ),
                group_indices=torch.arange(num_tokens, dtype=torch.int32) % K,
                target_logits_indices=torch.arange(num_tokens, dtype=torch.int32),
                bonus_logits_indices=torch.arange(num_reqs, dtype=torch.int32),
                final_logits_indices=torch.arange(num_tokens, dtype=torch.int32),
                synthetic_verify=None,
            )
            runner._build_padded_sampling_params = lambda s, e, logits: (
                torch.ones((num_reqs, 1), dtype=torch.float32),
                torch.zeros((num_reqs, 1), dtype=torch.int32),
                torch.full((num_reqs, 1), 0.95, dtype=torch.float32),
            )
            for _ in range(2):
                runner._plan_synthetic_verify(md)
                logits = torch.randn(num_reqs, VOCAB)
                out = runner._sample_synthetic_verify(logits, md, 0, num_reqs, None)
                assert out.shape == (num_reqs, K + 1)
    finally:
        torch._dynamo.reset()
    served = {name: len(g) - warmed[name] for name, g in graphs.items()}
    return warmed, served


MINIMAX = {"temperature": 1.0, "top_p": 0.95}


def test_auto_serving_compiles_nothing_after_warmup(monkeypatch):
    warmed, served = _warm_then_serve(monkeypatch, "auto", MINIMAX)
    # One row selection (on serving's inputs) and the one sampler variant.
    assert warmed == {
        "synthetic_verify_rows": 0,
        "synthetic_verify_rows_rng": 1,
        "synthetic_sample_and_assemble": 0,
        "synthetic_sample_and_assemble_rng": 1,
    }
    assert served == dict.fromkeys(served, 0)


def test_unset_warms_five_programs_and_serving_recompiles_the_rows(monkeypatch):
    """The default sweep (control): both row selections and every sampler
    variant, and the warmed row selection does not serve (its aliased final
    indices fail Dynamo's guard), so serving compiles it again."""
    warmed, served = _warm_then_serve(monkeypatch, None, MINIMAX)
    assert warmed == {
        "synthetic_verify_rows": 1,
        "synthetic_verify_rows_rng": 1,
        "synthetic_sample_and_assemble": 0,
        "synthetic_sample_and_assemble_rng": 4,
    }
    assert served == {
        "synthetic_verify_rows": 0,
        "synthetic_verify_rows_rng": 1,
        "synthetic_sample_and_assemble": 0,
        "synthetic_sample_and_assemble_rng": 0,
    }


def test_wrong_guess_compiles_on_first_use_and_logs(monkeypatch):
    """Requests that override the defaults pick an unwarmed variant: serving
    compiles it once and logs it once."""
    warnings: list[str] = []
    monkeypatch.setattr(
        tpu_runner.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    warmed, served = _warm_then_serve(monkeypatch, "top_p", {})
    assert warmed["synthetic_sample_and_assemble_rng"] == 1
    assert served["synthetic_sample_and_assemble_rng"] == 1
    assert served["synthetic_verify_rows_rng"] == 0
    assert len(warnings) == 1 and "no_topk_topp, not warmed" in warnings[0]
