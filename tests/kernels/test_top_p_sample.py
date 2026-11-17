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
"""spec_sampler_top_p_kernel: the dense-row top-p recovery kernel (interpret
mode) against `rejection_sampler._recover_tokens`, and the key-off
dispatch."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels import top_p_sample as kernel
from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample import top_p_sample


def _case(rows: int, vocab: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn((rows, vocab), generator=g) * 4
    temps = torch.rand((rows, 1), generator=g) + 0.2
    temps[0] = 0.0  # a greedy row
    top_p = torch.rand((rows, 1), generator=g) * 0.9 + 0.05
    top_p[-1] = 1.0  # no top-p on the last row
    draft = torch.randint(0, vocab, (rows,), generator=g)
    draft[1 % rows] = -1  # a bonus row: nothing removed
    u = torch.rand((rows, vocab), generator=g)
    return logits, temps, top_p, draft, u


def _reference(logits, temps, top_p, draft, u, apply_topk_topp=True):
    return rs._recover_tokens(
        logits,
        draft,
        temps,
        torch.zeros_like(draft, dtype=torch.int32).unsqueeze(1),
        top_p,
        u,
        0,
        apply_topk_topp,
        None,
        apply_top_k=False,
    )


def _kernel(logits, temps, top_p, draft, u):
    rows, vocab = logits.shape
    shape = (rows, vocab // 128, 128)
    params = jnp.asarray(np.stack([temps.view(rows).numpy(), top_p.view(rows).numpy()]))
    out = kernel.top_p_rows(
        jnp.asarray(logits.numpy().reshape(shape)),
        jnp.asarray(u.numpy().reshape(shape)),
        params,
        jnp.asarray(draft.numpy().astype(np.int32)),
        interpret=True,
    )
    return torch.from_numpy(np.asarray(out)[:rows, 0].astype(np.int64))


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kernel_recovers_the_xla_tokens(rows, seed):
    case = _case(rows, 1024, seed)
    want = _reference(*case)
    got = _kernel(*case)
    # Same steps in another summation order: the tokens agree away from
    # float32 near-ties, which random logits do not hit.
    assert torch.equal(got, want.to(torch.int64))


def test_kernel_without_top_p():
    logits, temps, _, draft, u = _case(8, 512, 5)
    want = _reference(logits, temps, torch.ones_like(temps), draft, u, False)
    got = _kernel(logits, temps, torch.ones_like(temps), draft, u)
    assert torch.equal(got, want.to(torch.int64))


def test_supports():
    assert kernel.supports(8, 1024)
    assert not kernel.supports(9, 1024)
    assert not kernel.supports(4, 1000)


def test_key_off_never_dispatches(monkeypatch):
    """Unset (never enabled), `_recover_tokens` keeps the XLA path."""
    monkeypatch.setattr(top_p_sample, "_op", None)
    assert not top_p_sample.enabled()
    logits = torch.zeros((2, 256))
    assert not top_p_sample.applies(logits)
    called = []
    monkeypatch.setattr(top_p_sample, "recover", lambda *a: called.append(a))
    case = _case(2, 256, 0)
    rs._recover_tokens(
        case[0],
        case[3],
        case[1],
        torch.zeros((2, 1), dtype=torch.int32),
        case[2],
        case[4],
        0,
        True,
        None,
        apply_top_k=False,
    )
    assert not called


def test_dispatch_conditions(monkeypatch):
    monkeypatch.setattr(top_p_sample, "_op", object())
    monkeypatch.setattr(top_p_sample, "applies", lambda logits: True)
    calls = []
    monkeypatch.setattr(
        top_p_sample, "recover", lambda *a: calls.append(a) or torch.zeros(2)
    )
    logits, temps, top_p, draft, u = _case(2, 256, 0)
    top_k = torch.zeros((2, 1), dtype=torch.int32)
    # Top-k applied, candidates, or draft probabilities: the XLA path.
    rs._recover_tokens(logits, draft, temps, top_k, top_p, u, 0, True, None)
    rs._recover_tokens(
        logits, draft, temps, top_k + 1, top_p, u, 2, True, None, apply_top_k=False
    )
    rs._recover_tokens(
        logits,
        draft,
        temps,
        top_k,
        top_p,
        u,
        0,
        True,
        torch.zeros_like(logits),
        apply_top_k=False,
    )
    assert not calls
    rs._recover_tokens(
        logits, draft, temps, top_k, top_p, u, 0, True, None, apply_top_k=False
    )
    rs._recover_tokens(logits, draft, temps, top_k, top_p, u, 0, False, None)
    assert len(calls) == 2
    # Without top-p/top-k the kernel gets no top_p.
    assert calls[0][3] is top_p and calls[1][3] is None
