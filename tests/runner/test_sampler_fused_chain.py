# Copyright 2025 Google LLC
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
"""(Ported from patemotter 9646b9af7 onto rc7.) TPU_SAMPLER_FUSED_CHAIN / TPU_SAMPLER_GRAPH_RNG: the non-greedy verify branch
of ``_sample_spec_verify_chunk`` as one program.

Exactness: over a sequence of verify steps (real and synthetic rejection, mixed
greedy / sampled rows, with and without top-k/top-p), the fused chain selects
the same tokens, bit for bit, as the unfused branch driven by a generator with
the same seed -- with eager draws, and (TPU only) with the draws taken in-graph
from the default generator. Positive control: another seed changes some output.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

K = 6
PADDED_NUM_REQS = 8
TOKEN_PADDINGS = [16, 32, 64]
VOCAB = 129280
# Draft counts per request of each step.
CASES = [[6], [6, 6], [2, 6], [6, 0], [1, 6, 3], [6, 6, 2, 1]]
SYNTHETIC_RATES = [0.9, 0.7, 0.5, 0.4, 0.3, 0.2]


@pytest.fixture
def dev():
    try:
        import torch_tpu  # noqa: F401

        d = torch.device("tpu", 0)
        torch.zeros(1, device=d)
        return d
    except Exception:
        return torch.device("cpu")


@pytest.fixture(autouse=True)
def _raise_dynamo_recompile_limit():
    cfg = torch._dynamo.config
    names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
    saved = {n: getattr(cfg, n) for n in names}
    for n in names:
        setattr(cfg, n, 1024)
    yield
    for n, v in saved.items():
        setattr(cfg, n, v)


def _sampler(synthetic: bool, device) -> RejectionSampler:
    sampler = RejectionSampler()
    if synthetic:
        sampler.synthetic_conditional_rates = torch.tensor(
            SYNTHETIC_RATES, dtype=torch.float32, device=device
        )
        sampler.synthetic_mode = True
    return sampler


def _runner(device, synthetic, fused, graph_rng, seed, top_kp):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    temps = torch.tensor([0.6, 1.0, 0.0, 0.6] * 2, dtype=torch.float32)
    top_k = torch.tensor([0, 20, 0, 0] * 2 if top_kp else [0] * 8, dtype=torch.int32)
    top_p = torch.tensor([0.95, 1.0, 1.0, 1.0] * 2 if top_kp else [1.0] * 8)
    runner.__dict__.update(
        device=device,
        vocab_size=VOCAB,
        rejection_sampler=_sampler(synthetic, device),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        input_batch=SimpleNamespace(
            temperature_cpu_tensor=temps,
            top_k_cpu_tensor=top_k,
            top_p_cpu_tensor=top_p,
            # rc7 _sampler_applies_topk_topp: the no-top-k/top-p graph when
            # no request sets either.
            no_top_k=not top_kp,
            no_top_p=not top_kp,
        ),
        model_config=SimpleNamespace(seed=seed),
        _sampling_generator=None,
        _sampler_fused_chain=fused,
        _sampler_graph_rng=graph_rng,
    )
    runner.sample_from_logits_func = runner.sample_from_logits
    return runner


def _step(case, seed, device):
    gen = torch.Generator().manual_seed(seed)
    nd = np.asarray(case, dtype=np.int32)
    rows = int(np.sum(nd + 1))
    padded = next(p for p in TOKEN_PADDINGS if p >= rows)
    # Peaked and flat rows, so drafts are often accepted and recovery happens.
    scale = torch.rand(padded, 1, generator=gen) * 8.0
    logits = (torch.randn(padded, VOCAB, generator=gen) * scale).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (4096,), dtype=torch.int32, generator=gen)
    am = logits.float().argmax(dim=-1)
    pos = 0
    for n in case:
        for j in range(n):
            if torch.rand(1, generator=gen).item() < 0.7:
                input_ids[pos + j + 1] = int(am[pos + j])
        pos += n + 1
    mgr = SpeculativeDecodingManager(
        SimpleNamespace(
            input_ids_cpu=input_ids,
            arange_np=np.arange(4096, dtype=np.int64),
            num_tokens_paddings=TOKEN_PADDINGS,
            device=device,
        )
    )
    md = mgr.get_spec_decode_metadata(
        nd, np.cumsum(nd + 1).astype(np.int32), PADDED_NUM_REQS
    )
    return logits.to(device), md


def _steps(device, synthetic, fused, graph_rng, seed, top_kp):
    runner = _runner(device, synthetic, fused, graph_rng, seed, top_kp)
    outs = []
    for i, case in enumerate(CASES):
        logits, md = _step(case, 100 + i, device)
        out = runner._sample_spec_verify_chunk(
            logits,
            md,
            None,
            None,
            0,
            len(case),
            False,
            runner._get_sampling_generator(),
        )
        outs.append(out.cpu())
    return outs


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("top_kp", [False, True], ids=["plain", "topkp"])
@pytest.mark.parametrize("synthetic", [False, True], ids=["real", "synthetic"])
def test_fused_chain_is_exact(dev, synthetic, top_kp):
    unfused = _steps(dev, synthetic, False, False, 7, top_kp)
    fused = _steps(dev, synthetic, True, False, 7, top_kp)
    assert _equal(unfused, fused), (unfused, fused)
    assert not _equal(unfused, _steps(dev, synthetic, True, False, 8, top_kp))


@pytest.mark.parametrize("synthetic", [False, True], ids=["real", "synthetic"])
def test_graph_rng_is_exact(dev, synthetic):
    if dev.type != "tpu":
        pytest.skip("default-generator equivalence is a TPU property")
    unfused = _steps(dev, synthetic, False, False, 7, False)
    graph = _steps(dev, synthetic, True, True, 7, False)
    assert _equal(unfused, graph), (unfused, graph)
    assert not _equal(unfused, _steps(dev, synthetic, True, True, 8, False))
