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
"""TPU_SPEC_SYNTHETIC_INGRAPH_RNG: the counter uniforms are U[0, 1) at
torch.rand's resolution, the fused verify given them decides exactly as with
the same uniforms passed in, and its accepted counts and bonus / recovered
tokens follow the synthetic schedule and the target distribution, as they do
with torch.rand."""

import numpy as np
import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_torchtpu.layers.adapter.sample import fused_spec_verify as fsv
from vllm_torchtpu.layers.adapter.sample.counter_uniform import (
    counter_uniform,
    initial_counter,
)

torch._dynamo.config.recompile_limit = 64

K = 7
UNCOND = [0.82, 0.66, 0.5, 0.38, 0.29, 0.22, 0.17]  # mean acceptance length 4.04


@pytest.fixture
def device():
    """TPU when torch_tpu has one, else CPU (the uncompiled bodies)."""
    try:
        import torch_tpu  # noqa: F401

        return torch.device("tpu")
    except Exception:
        return torch.device("cpu")


def _on_tpu(device) -> bool:
    return device.type == "tpu"


def _uniforms(device, shape, counters):
    return torch.stack(
        [
            counter_uniform(shape, torch.tensor(c, dtype=torch.int32, device=device))
            for c in counters
        ]
    ).cpu()


def test_counter_uniform_is_uniform(device):
    c0 = initial_counter(0)
    u = _uniforms(device, (4096,), range(c0, c0 + 64)).double()
    assert u.min() >= 0.0 and u.max() < 1.0
    assert torch.equal(u * 2**24, torch.round(u * 2**24))
    n = u.numel()
    assert abs(u.mean().item() - 0.5) < 5 * (1 / 12 / n) ** 0.5
    assert abs(u.var().item() - 1 / 12) < 0.002
    hist = torch.histc(u.float(), bins=256, min=0.0, max=1.0)
    exp = n / 256
    chi2 = (((hist - exp) ** 2) / exp).sum().item()
    assert chi2 < 255 + 6 * 22.6
    flat = u.view(64, 4096)
    lag1 = np.corrcoef(flat[:, :-1].flatten(), flat[:, 1:].flatten())[0, 1]
    step = np.corrcoef(flat[:-1].flatten(), flat[1:].flatten())[0, 1]
    assert abs(lag1) < 0.01 and abs(step) < 0.01
    words = (u[0] * 2**24).long()
    assert words.unique().numel() > 0.999 * words.numel()


def test_counter_uniform_deterministic(device):
    c = torch.tensor(12345, dtype=torch.int32, device=device)
    a = counter_uniform((3, 5), c).cpu()
    assert torch.equal(a, counter_uniform((3, 5), c).cpu())
    assert not torch.equal(a, counter_uniform((3, 5), c + 1).cpu())
    assert torch.equal(counter_uniform((15,), c).cpu(), a.flatten())


def _chunk(num_reqs, vocab, device, seed=0):
    """A verify chunk where every request drafts K tokens, laid out as
    get_spec_decode_metadata does (no padding rows)."""
    g = torch.Generator().manual_seed(seed)
    rows = num_reqs * (K + 1)
    logits = torch.randn((rows, vocab), generator=g) * 1.5
    starts = torch.arange(num_reqs) * (K + 1)
    target = (starts.unsqueeze(1) + torch.arange(K)).flatten().to(torch.int32)
    bonus = (starts + K).to(torch.int32)
    draft = torch.randint(0, vocab, (num_reqs * K,), generator=g, dtype=torch.int32)
    seg = torch.arange(num_reqs, dtype=torch.int64).repeat_interleave(K)
    grp = torch.arange(K, dtype=torch.int32).repeat(num_reqs)
    lens = torch.full((num_reqs,), K, dtype=torch.int32)
    temps = torch.ones((num_reqs, 1))
    top_k = torch.full((num_reqs, 1), vocab, dtype=torch.int32)
    top_p = torch.ones((num_reqs, 1))
    rates = torch.tensor(unconditional_to_conditional_rates(UNCOND), dtype=torch.float32)
    t = [x.to(device) for x in (logits, bonus, target, draft, lens, seg, grp, temps, top_k, top_p)]
    return t, rates.to(device)


def _rng_fn(device):
    return fsv.spec_verify_sample_fused_rng if _on_tpu(device) else fsv._spec_verify_sample_fused_rng_impl


def test_ingraph_draws_decide_like_passed_in_draws(device):
    num_reqs, vocab = 8, 512
    t, rates = _chunk(num_reqs, vocab, device)
    logits, bonus, target, draft = t[:4]
    counter = torch.tensor(initial_counter(3), dtype=torch.int32, device=device)
    for _ in range(4):
        out, nxt = _rng_fn(device)(*t, counter, rates, True, K, apply_topk_topp=False)
        bu = counter_uniform((num_reqs, vocab), counter)
        au = counter_uniform(tuple(draft.shape), counter + 1)
        ru = counter_uniform((target.shape[0], vocab), counter + 2)
        want = fsv._spec_verify_sample_fused_impl(
            *t, bu, au, ru, rates, True, K, apply_topk_topp=False
        )
        assert torch.equal(out.cpu(), want.cpu())
        assert int(nxt.cpu()) == int(counter.cpu()) + 3
        counter = nxt


def test_ingraph_draws_follow_schedule_and_target(device):
    """Accepted counts follow the 4.04 schedule; bonus tokens (all accepted)
    and recovered tokens follow softmax(logits row) (recovered: with the
    rejected draft removed)."""
    num_reqs, vocab, steps = 16, 32, 300
    t, rates = _chunk(num_reqs, vocab, device)
    logits = t[0].cpu()
    draft = t[3].cpu()
    counter = torch.tensor(initial_counter(0), dtype=torch.int32, device=device)
    fn = _rng_fn(device)
    counts = []
    bonus_hist = torch.zeros(num_reqs, vocab)
    rec_obs, rec_exp = torch.zeros(vocab), torch.zeros(vocab)
    p_rows = torch.softmax(logits.float(), dim=-1)
    for _ in range(steps):
        out, counter = fn(*t, counter, rates, True, K, apply_topk_topp=False)
        out = out.cpu()
        acc = (out != -1).sum(-1) - 1
        counts.append(acc)
        for r in range(num_reqs):
            a = int(acc[r])
            tok = int(out[r, a])
            if a == K:
                bonus_hist[r, tok] += 1
            else:
                p = p_rows[r * (K + 1) + a].clone()
                p[int(draft[r * K + a])] = 0
                rec_obs[tok] += 1
                rec_exp += p / p.sum()
    counts = torch.cat(counts).double()
    n = counts.numel()
    for j in range(1, K + 1):
        p = (counts >= j).double().mean().item()
        sd = (UNCOND[j - 1] * (1 - UNCOND[j - 1]) / n) ** 0.5
        assert abs(p - UNCOND[j - 1]) < 5 * sd, (j, p)
    # Recovered tokens, pooled over rows: observed vs summed expected.
    mask = rec_exp > 5
    chi2 = (((rec_obs[mask] - rec_exp[mask]) ** 2) / rec_exp[mask]).sum().item()
    dof = int(mask.sum()) - 1
    assert chi2 < dof + 6 * (2 * dof) ** 0.5, (chi2, dof)
    # Bonus tokens, pooled: each row's count times its bonus-row softmax.
    exp = torch.zeros(vocab)
    for r in range(num_reqs):
        exp += bonus_hist[r].sum() * p_rows[r * (K + 1) + K]
    obs = bonus_hist.sum(0)
    mask = exp > 5
    chi2 = (((obs[mask] - exp[mask]) ** 2) / exp[mask]).sum().item()
    dof = int(mask.sum()) - 1
    assert chi2 < dof + 6 * (2 * dof) ** 0.5, (chi2, dof)
