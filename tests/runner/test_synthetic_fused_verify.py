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

"""`TPU_SPEC_SYNTHETIC_FUSED_VERIFY`: the synthetic verify's plan, LM head and
sampler as one program return the three programs' tokens on the same draw."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm.distributed

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K, VOCAB, HIDDEN = 3, 160, 16


def _runner(gen, top_p):
    weight = torch.randn(VOCAB, HIDDEN, generator=gen)
    spec_config = SimpleNamespace(
        num_speculative_tokens=K,
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=[0.8, 0.5, 0.3],
    )
    runner = object.__new__(TPUModelRunner)
    runner.is_pooling_model = False
    runner.model = SimpleNamespace(compute_logits=lambda h: 4 * h @ weight.T)
    runner.speculative_config = spec_config
    runner.rejection_sampler = RejectionSampler(spec_config, torch.device("cpu"))
    runner.vocab_size = VOCAB
    runner._synthetic_row_rank = None
    runner._synthetic_shard = None
    runner._synthetic_fused_warm = set()
    runner._synthetic_fused_shapes = set()
    runner._sampling_params_reuse = None
    runner.input_batch = SimpleNamespace(
        temperature_cpu_tensor=torch.tensor([1.0, 0.7, 1.3]),
        top_k_cpu_tensor=torch.zeros(3, dtype=torch.int32),
        top_p_cpu_tensor=torch.tensor(top_p),
        no_top_k=True,
        no_top_p=all(p == 1.0 for p in top_p),
    )
    return runner


def _metadata(gen):
    """Three requests at k=3 (one with two drafts), padded to 4 requests and
    16 verify rows."""
    lengths, padded_reqs, padded_rows = [3, 3, 2], 4, 16
    target, segment, group, bonus = [], [], [], []
    row = 0
    for r, n in enumerate(lengths):
        target += [row + j for j in range(n)]
        segment += [r] * n
        group += list(range(n))
        bonus.append(row + n)
        row += n + 1
    pad = padded_rows - len(target)
    i32 = torch.int32
    return SpecDecodeMetadata(
        draft_token_ids=torch.randint(0, VOCAB, (padded_rows,), generator=gen).to(i32),
        draft_lengths=torch.tensor(lengths + [0], dtype=i32),
        draft_lengths_cpu=np.array(lengths, dtype=np.int32),
        target_logits_indices=torch.tensor(target + [0] * pad, dtype=i32),
        bonus_logits_indices=torch.tensor(
            bonus + [0] * (padded_reqs - len(lengths)), dtype=i32
        ),
        final_logits_indices=torch.randperm(padded_rows, generator=gen).to(i32),
        segment_ids=torch.tensor(segment + [len(lengths)] * pad, dtype=torch.int64),
        group_indices=torch.tensor(group + [0] * pad, dtype=i32),
    )


def _step(runner, hidden, md, seed):
    gen = torch.Generator().manual_seed(seed)
    runner._get_sampling_generator = lambda: gen
    with torch.compiler.set_stance("force_eager"):
        logits = runner._synthetic_verify_logits(hidden, md)
        tokens = runner._sample_spec_verify_chunk(
            logits, md, None, None, 0, 3, False, gen
        )
    return tokens, gen.get_state()


@pytest.mark.parametrize("warm", [True, False])
@pytest.mark.parametrize("top_p", [[1.0, 1.0, 1.0], [0.9, 1.0, 0.6]])
@pytest.mark.parametrize("seed", range(3))
def test_fused_verify_matches_three_programs(monkeypatch, warm, top_p, seed):
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    gen = torch.Generator().manual_seed(seed)
    runner = _runner(gen, top_p)
    hidden = torch.randn(16, HIDDEN, generator=gen)
    md_gen_state = gen.get_state()

    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_FUSED_VERIFY", False)
    expected, expected_state = _step(runner, hidden, _metadata(gen), seed)

    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_FUSED_VERIFY", True)
    gen.set_state(md_gen_state)
    md = _metadata(gen)
    if warm:
        runner._synthetic_fused_shapes.add((16, 16, 4))
        runner._synthetic_fused_warm.add("warmed")
        monkeypatch.setattr(tpu_runner, "program_key", lambda _args: "warmed")
    fused_calls = []
    fused = TPUModelRunner.synthetic_verify_and_sample
    monkeypatch.setattr(
        TPUModelRunner,
        "synthetic_verify_and_sample",
        lambda *a, **kw: fused_calls.append(1) or fused(*a, **kw),
    )
    tokens, state = _step(runner, hidden, md, seed)

    assert fused_calls == ([1] if warm else [])
    assert md.synthetic_hidden_states is None
    assert torch.equal(tokens, expected)
    # The same draws from the sampling stream.
    assert torch.equal(state, expected_state)


def _served_metadata(num_tokens_paddings, scheduled):
    """`get_spec_decode_metadata` of a step scheduling ``scheduled`` tokens
    per request, K drafts for each verifying request (K + 1 tokens) and none
    for a prefill chunk, padded to 4 requests."""
    num_draft = np.array([K if n == K + 1 else 0 for n in scheduled], np.int32)
    manager = SimpleNamespace(
        _meta_cache_key=None,
        runner=SimpleNamespace(
            input_ids_cpu=torch.zeros(64, dtype=torch.int32),
            arange_np=np.arange(64),
            num_tokens_paddings=num_tokens_paddings,
            device=torch.device("cpu"),
        ),
    )
    return SpeculativeDecodingManager.get_spec_decode_metadata(
        manager,  # type: ignore[arg-type]
        num_draft,
        np.cumsum(scheduled).astype(np.int32),
        padded_num_reqs=4,
    )


@pytest.mark.parametrize(
    "scheduled, fused",
    [
        # Three verifying requests: 12 tokens, verify and token bucket 16.
        ([K + 1] * 3, True),
        # A 2-token prefill chunk beside them still lands in bucket 16.
        ([K + 1] * 3 + [2], True),
        # A 9-token one moves the token bucket to 32: three programs, with
        # the LM head in execute_model as before.
        ([K + 1] * 3 + [9], False),
    ],
)
def test_served_step_meets_the_warmup_keys(monkeypatch, scheduled, fused):
    """The warmup's keys (`_precompile_synthetic_fused_verify`) against a
    step built as serving builds it: a step in a warmed (token bucket, verify
    rows, requests) runs the one program; another keeps its LM head in
    execute_model (no deferral, no warm-miss warning); same tokens and
    stream either way."""
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    gen = torch.Generator().manual_seed(0)
    runner = _runner(gen, [0.9, 1.0, 0.6])
    runner.model_config = SimpleNamespace(get_hidden_size=lambda: HIDDEN)
    runner._hidden_states_dtype = torch.float32
    runner.device = torch.device("cpu")
    with torch.compiler.set_stance("force_eager"):
        runner._precompile_synthetic_fused_verify(16, 4)
    assert runner._synthetic_fused_shapes == {(16, 16, 4)}

    paddings = [8, 16, 32]
    bucket = tpu_runner._get_padded_token_len(paddings, sum(scheduled))
    hidden = torch.randn(bucket, HIDDEN, generator=gen)
    md_state = gen.get_state()
    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_FUSED_VERIFY", False)
    expected, expected_state = _step(
        runner, hidden, _served_metadata(paddings, scheduled), 0
    )

    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_FUSED_VERIFY", True)
    gen.set_state(md_state)
    md = _served_metadata(paddings, scheduled)
    calls = []
    fused_program = TPUModelRunner.synthetic_verify_and_sample
    monkeypatch.setattr(
        TPUModelRunner,
        "synthetic_verify_and_sample",
        lambda *a, **kw: calls.append(1) or fused_program(*a, **kw),
    )
    warnings = []
    monkeypatch.setattr(
        tpu_runner.logger, "warning_once", lambda *a, **kw: warnings.append(a)
    )
    deferred = []
    verify_logits = runner._synthetic_verify_logits

    def record(hidden_states, md):
        out = verify_logits(hidden_states, md)
        deferred.append(md.synthetic_hidden_states is not None)
        return out

    runner._synthetic_verify_logits = record
    tokens, state = _step(runner, hidden, md, 0)

    assert calls == ([1] if fused else [])
    assert deferred == [fused]
    assert warnings == []
    assert torch.equal(tokens, expected)
    assert torch.equal(state, expected_state)
