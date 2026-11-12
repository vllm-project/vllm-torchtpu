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
"""TPU_SPEC_SYNTHETIC_HEAD_FUSION: the fused synthetic verify head gives the
three-program sequence's tokens and counter bit for bit (synthetic_verify_rows_rng,
compute_selected_logits on its rows, synthetic_sample_and_assemble_rng with the
host-staged temperatures), on CPU through the uncompiled body."""

import types

import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _synthetic_sample_and_assemble_rng_impl,
    _synthetic_verify_rows_rng_impl,
    initial_rng_counter,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _head_body():
    fn = TPUModelRunner.synthetic_verify_head
    return getattr(fn, "_torchdynamo_orig_callable", fn)


def _inputs(num_reqs, padded_reqs, k, padded_len, gen):
    """Decode-only verify chunk metadata as get_spec_decode_metadata lays it
    out: every request carries k drafts, rows [r*(k+1) .. r*(k+1)+k]; logits
    length and request count padded."""
    total = num_reqs * k
    draft = torch.zeros(padded_len, dtype=torch.int32)
    draft[:total] = torch.randint(0, 500, (total,), generator=gen, dtype=torch.int32)
    seg = torch.full((padded_len,), num_reqs, dtype=torch.int64)
    seg[:total] = torch.arange(num_reqs).repeat_interleave(k)
    grp = torch.zeros(padded_len, dtype=torch.int32)
    grp[:total] = torch.arange(k, dtype=torch.int32).repeat(num_reqs)
    target = torch.zeros(padded_len, dtype=torch.int32)
    target[:total] = torch.arange(total, dtype=torch.int32) + seg[:total].to(torch.int32)
    bonus = torch.zeros(padded_reqs, dtype=torch.int32)
    bonus[:num_reqs] = (torch.arange(num_reqs, dtype=torch.int32) + 1) * (k + 1) - 1
    final = torch.arange(padded_len, dtype=torch.int32)
    lens = torch.zeros(padded_reqs, dtype=torch.int32)
    lens[:num_reqs] = k
    return draft, lens, seg, grp, target, bonus, final


@pytest.mark.parametrize("num_reqs,padded_reqs,padded_len", [(1, 1, 8), (3, 4, 24), (5, 8, 48)])
@pytest.mark.parametrize("variant", [
    {},
    {"apply_topk_topp": False},
    {"apply_top_k": False},
])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_head_matches_three_programs(num_reqs, padded_reqs, padded_len, variant, dtype):
    k, hidden, vocab = 7, 64, 512
    gen = torch.Generator().manual_seed(num_reqs * 100 + padded_len)
    w = torch.randn(hidden, vocab, generator=gen).to(dtype)
    model = types.SimpleNamespace(compute_logits=lambda h: h @ w)
    fake_self = types.SimpleNamespace(model=model)
    hs = torch.randn(padded_len, hidden, generator=gen).to(dtype)
    draft, lens, seg, grp, target, bonus, final = _inputs(
        num_reqs, padded_reqs, k, padded_len, gen
    )
    rates = torch.tensor(
        unconditional_to_conditional_rates([0.82, 0.66, 0.5, 0.38, 0.29, 0.22, 0.17]),
        dtype=torch.float32,
    )
    temps_req = torch.ones(padded_reqs)
    temps_req[:num_reqs] = torch.tensor([0.7, 1.0, 0.33, 1.3, 0.9])[:num_reqs]
    top_k = torch.zeros((padded_reqs, 1), dtype=torch.int32)
    top_k[:num_reqs] = 20
    top_p = torch.ones((padded_reqs, 1))
    top_p[:num_reqs] = 0.95
    counter = torch.tensor(initial_rng_counter(3), dtype=torch.int32)
    for _ in range(4):
        # Reference: the three programs, temperatures staged in the logits dtype.
        acc, rows, rec, c1 = _synthetic_verify_rows_rng_impl(
            draft, lens, seg, grp, target, bonus, final, counter, rates
        )
        logits = model.compute_logits(torch.index_select(hs, 0, rows))
        ref, c2 = _synthetic_sample_and_assemble_rng_impl(
            logits, rec, acc, temps_req.to(logits.dtype).unsqueeze(1), top_k, top_p, c1,
            draft, lens, seg, grp, k, **variant,
        )
        out, c3 = _head_body()(
            fake_self, hs, draft, lens, seg, grp, target, bonus, final, counter, rates,
            temps_req.to(torch.float32).unsqueeze(1), top_k, top_p, k,
            variant.get("topk_candidates", 0),
            variant.get("apply_topk_topp", True),
            variant.get("apply_top_k", True),
        )
        assert torch.equal(out, ref)
        assert int(c3) == int(c2) == int(counter) + 2
        counter = c3


def test_fusion_applies_only_to_decode_only_chunks(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_HEAD_FUSION", True, raising=False)
    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_INGRAPH_RNG", True, raising=False)
    md = types.SimpleNamespace(draft_token_ids=torch.zeros(24, dtype=torch.int32))
    applies = TPUModelRunner._synthetic_head_fusion_applies
    assert applies(None, torch.zeros(24, 4), md)
    assert not applies(None, torch.zeros(8192, 4), md)
    md.draft_token_ids = None
    assert not applies(None, torch.zeros(24, 4), md)
    md.draft_token_ids = torch.zeros(24, dtype=torch.int32)
    monkeypatch.setattr(envs, "TPU_SPEC_SYNTHETIC_INGRAPH_RNG", False, raising=False)
    assert not applies(None, torch.zeros(24, 4), md)
