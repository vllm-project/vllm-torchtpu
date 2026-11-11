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
"""TPU_SPEC_GLUE_FUSION and TPU_SPEC_SYNTHETIC_INGRAPH_RNG.

The glue programs must give the eager sequence's values bit for bit. The
in-graph uniforms must be U[0, 1) at torch.rand's resolution, and the
synthetic accepts and recovered tokens drawn from them must follow the same
distributions as with torch.rand.
"""

import numpy as np
import pytest
import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _synthetic_sample_and_assemble_impl,
    _synthetic_sample_and_assemble_rng_impl,
    _synthetic_verify_rows_impl,
    _synthetic_verify_rows_rng_impl,
    counter_uniform,
    initial_rng_counter,
)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    _spec_substitution_glue_impl,
    assemble_spec_next_tokens,
    compute_num_rejected,
    subtract_num_rejected_tokens,
)
from vllm_torchtpu.spec_decode.utils import fit_rows


@pytest.fixture
def device():
    """TPU when torch_tpu has one, else CPU (these tests also mean something
    on CPU: the uncompiled bodies against the eager reference)."""
    try:
        import torch_tpu  # noqa: F401

        return torch.device("tpu")
    except Exception:
        return torch.device("cpu")


def _on_tpu(device) -> bool:
    return device.type == "tpu"


def _random_rejection_output(gen, rows, k, live):
    """[rows, 1 + k] int32 rejection output: per live row an accepted prefix
    plus the bonus, then INVALID_TOKEN_ID; padded rows all invalid."""
    out = torch.full((rows, 1 + k), INVALID_TOKEN_ID, dtype=torch.int32)
    for r in range(live):
        n = int(torch.randint(1, k + 2, (1,), generator=gen))
        out[r, :n] = torch.randint(0, 150000, (n,), generator=gen, dtype=torch.int32)
    return out


def _eager_glue(next_tokens, drafts, num_draft, rows, num_reqs, src_len, rej_len):
    src = assemble_spec_next_tokens(next_tokens, fit_rows(drafts, rows), num_reqs)
    if src.shape[0] < src_len:
        src = torch.nn.functional.pad(src, (0, src_len - src.shape[0]))
    rej = torch.cat([compute_num_rejected(next_tokens, fit_rows(num_draft, rows))])
    return src, rej


@pytest.mark.parametrize("rows,live,drafts_rows,bucketed", [
    (8, 3, 8, True),
    (16, 16, 16, True),
    (64, 37, 64, True),
    (32, 5, 16, True),
    (16, 9, 9, False),
])
@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
def test_substitution_glue_bitwise(device, rows, live, drafts_rows, bucketed, draft_dtype):
    gen = torch.Generator().manual_seed(rows * 1000 + live)
    k, max_num_reqs = 7, 64
    nt = _random_rejection_output(gen, rows, k, live)
    drafts = torch.randint(0, 150000, (drafts_rows, k), generator=gen, dtype=draft_dtype)
    num_draft = torch.randint(0, k + 1, (rows,), generator=gen, dtype=torch.int32)
    num_reqs = None if bucketed else live
    rows_eff = rows if bucketed else live
    src_len, rej_len = max_num_reqs * (1 + k), max_num_reqs
    nt, drafts, num_draft = nt.to(device), drafts.to(device), num_draft.to(device)

    ref_src, ref_rej = _eager_glue(nt, drafts, num_draft, rows_eff, num_reqs, src_len, 0)
    fns = [_spec_substitution_glue_impl]
    if _on_tpu(device):
        from vllm_torchtpu.runner.tpu_runner_async_output import spec_substitution_glue

        fns.append(spec_substitution_glue)
    for fn in fns:
        src, rej = fn(nt, drafts, num_draft, rows_eff, num_reqs, src_len, rej_len)
        assert src.dtype == ref_src.dtype and rej.dtype == ref_rej.dtype
        assert torch.equal(src.cpu(), ref_src.cpu())
        n = ref_rej.shape[0]
        assert rej.shape[0] == max(n, rej_len)
        assert torch.equal(rej[:n].cpu(), ref_rej.cpu())
        assert not rej[n:].cpu().any()


@pytest.mark.parametrize("num_reqs,num_tokens,src_rows", [
    (8, 64, 8), (16, 128, 16), (64, 512, 64), (64, 8192, 64), (8, 2048, 4)])
def test_subtract_with_padded_rejected_bitwise(device, num_reqs, num_tokens, src_rows):
    """The fixed-length (zero-padded) rejected count, and the compiled
    subtract on TPU, give the eager subtract's values."""
    gen = torch.Generator().manual_seed(num_tokens + num_reqs)
    rej_len = 64
    seq_lens = torch.randint(100, 9000, (num_reqs,), generator=gen, dtype=torch.int32)
    positions = torch.randint(0, 9000, (num_tokens,), generator=gen, dtype=torch.int32)
    num_rejected = torch.randint(0, 8, (src_rows,), generator=gen, dtype=torch.int32)
    seq_idx = torch.randint(-1, src_rows, (num_reqs,), generator=gen, dtype=torch.int32)
    pos_idx = torch.randint(-1, src_rows, (num_tokens,), generator=gen, dtype=torch.int32)
    padded = torch.nn.functional.pad(num_rejected, (0, rej_len - src_rows))
    args = [t.to(device) for t in (seq_lens, positions, num_rejected, seq_idx, pos_idx)]
    ref = subtract_num_rejected_tokens(*args)
    args_padded = list(args)
    args_padded[2] = padded.to(device)
    outs = [subtract_num_rejected_tokens(*args_padded)]
    if _on_tpu(device):
        from vllm_torchtpu.runner.tpu_runner_async_output import (
            subtract_num_rejected_tokens_compiled,
        )

        outs.append(subtract_num_rejected_tokens_compiled(*args_padded))
        outs.append(subtract_num_rejected_tokens_compiled(*args))
    for out in outs:
        for a, b in zip(out, ref):
            assert a.dtype == b.dtype
            assert torch.equal(a.cpu(), b.cpu())


def _uniforms(device, shape, counters):
    return torch.stack(
        [
            counter_uniform(shape, torch.tensor(c, dtype=torch.int32, device=device))
            for c in counters
        ]
    ).cpu()


def test_counter_uniform_is_uniform(device):
    c0 = initial_rng_counter(0)
    u = _uniforms(device, (4096,), range(c0, c0 + 64)).double()
    assert u.dtype == torch.float64 and u.min() >= 0.0 and u.max() < 1.0
    # torch.rand's resolution: multiples of 2**-24.
    assert torch.equal(u * 2**24, torch.round(u * 2**24))
    n = u.numel()
    assert abs(u.mean().item() - 0.5) < 5 * (1 / 12 / n) ** 0.5
    assert abs(u.var().item() - 1 / 12) < 0.002
    # 256-bin chi-square, 255 dof: mean 255, sd ~22.6.
    hist = torch.histc(u.float(), bins=256, min=0.0, max=1.0)
    exp = n / 256
    chi2 = (((hist - exp) ** 2) / exp).sum().item()
    assert chi2 < 255 + 6 * 22.6
    # No correlation between neighbouring elements nor successive counters.
    flat = u.view(64, 4096)
    lag1 = np.corrcoef(flat[:, :-1].flatten(), flat[:, 1:].flatten())[0, 1]
    step = np.corrcoef(flat[:-1].flatten(), flat[1:].flatten())[0, 1]
    assert abs(lag1) < 0.01 and abs(step) < 0.01
    # Words of one draw are distinct (bijective hash).
    words = (u[0] * 2**24).long()
    assert words.unique().numel() > 0.999 * words.numel()


def test_counter_uniform_deterministic(device):
    c = torch.tensor(12345, dtype=torch.int32, device=device)
    a = counter_uniform((3, 5), c).cpu()
    b = counter_uniform((3, 5), c).cpu()
    assert torch.equal(a, b)
    d = counter_uniform((3, 5), c + 1).cpu()
    assert not torch.equal(a, d)
    # The first elements do not depend on the requested shape.
    assert torch.equal(counter_uniform((15,), c).cpu(), a.flatten())


def _verify_inputs(num_reqs, k, device):
    total = num_reqs * k
    draft = torch.randint(0, 1000, (total,), dtype=torch.int32)
    seg = torch.arange(num_reqs, dtype=torch.int64).repeat_interleave(k)
    grp = torch.arange(k, dtype=torch.int32).repeat(num_reqs)
    target_idx = torch.arange(total, dtype=torch.int32) + seg.to(torch.int32)
    bonus_idx = (torch.arange(num_reqs, dtype=torch.int32) + 1) * (k + 1) - 1
    final_idx = torch.arange(num_reqs * (k + 1), dtype=torch.int32)
    lens = torch.full((num_reqs,), k, dtype=torch.int32)
    return [t.to(device) for t in (draft, lens, seg, grp, target_idx, bonus_idx, final_idx)]


def test_synthetic_accepts_follow_schedule(device):
    """In-graph accept_u gives the acceptance-length distribution of the
    synthetic schedule (same as torch.rand would)."""
    k, num_reqs, steps = 7, 64, 200
    # Rates of a mean acceptance length about 4.04 (1 + sum = 4.04).
    uncond = [0.82, 0.66, 0.5, 0.38, 0.29, 0.22, 0.17]
    rates = torch.tensor(
        unconditional_to_conditional_rates(uncond), dtype=torch.float32, device=device
    )
    inputs = _verify_inputs(num_reqs, k, device)
    counter = torch.tensor(initial_rng_counter(0), dtype=torch.int32, device=device)
    rng_fn = _synthetic_verify_rows_rng_impl
    if _on_tpu(device):
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            synthetic_verify_rows_rng,
        )

        rng_fn = synthetic_verify_rows_rng
    counts = []
    for _ in range(steps):
        acc, rows, rec, nxt = rng_fn(*inputs, counter, rates)
        # Same decisions as the eager function on the same uniforms.
        u = counter_uniform(tuple(inputs[0].shape), counter)
        ref = _synthetic_verify_rows_impl(*inputs, u, rates)
        for a, b in zip((acc, rows, rec), ref):
            assert torch.equal(a.cpu(), b.cpu())
        assert int(nxt.cpu()) == int(counter.cpu()) + 1
        counter = nxt
        counts.append(acc.cpu())
    counts = torch.cat(counts).double()
    n = counts.numel()
    for j in range(1, k + 1):
        p = (counts >= j).double().mean().item()
        sd = (uncond[j - 1] * (1 - uncond[j - 1]) / n) ** 0.5
        assert abs(p - uncond[j - 1]) < 5 * sd, (j, p, uncond[j - 1])
    assert abs(1 + counts.mean().item() - (1 + sum(uncond))) < 0.05


def test_synthetic_recovery_distribution(device):
    """In-graph recover_u samples the target with the rejected draft
    removed, and the full target on bonus rows."""
    k, num_reqs, vocab, steps = 7, 16, 32, 400
    torch.manual_seed(0)
    base = torch.randn(vocab) * 1.5
    logits = base.repeat(num_reqs, 1)
    rejected = torch.full((num_reqs,), 3, dtype=torch.int64)
    rejected[num_reqs // 2 :] = -1  # bonus rows
    acc = torch.full((num_reqs,), 2, dtype=torch.int64)
    temps = torch.ones((num_reqs, 1))
    top_k = torch.zeros((num_reqs, 1), dtype=torch.int32)
    top_p = torch.ones((num_reqs, 1))
    draft = torch.randint(0, vocab, (num_reqs * k,), dtype=torch.int32)
    seg = torch.arange(num_reqs, dtype=torch.int64).repeat_interleave(k)
    grp = torch.arange(k, dtype=torch.int32).repeat(num_reqs)
    lens = torch.full((num_reqs,), k, dtype=torch.int32)
    args = [t.to(device) for t in (logits, rejected, acc, temps, top_k, top_p)]
    tail = [t.to(device) for t in (draft, lens, seg, grp)]
    counter = torch.tensor(7, dtype=torch.int32, device=device)
    fn = _synthetic_sample_and_assemble_rng_impl
    if _on_tpu(device):
        from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
            synthetic_sample_and_assemble_rng,
        )

        fn = synthetic_sample_and_assemble_rng
    rej_hist = torch.zeros(vocab)
    bonus_hist = torch.zeros(vocab)
    for _ in range(steps):
        out, nxt = fn(*args, counter, *tail, k, apply_topk_topp=False)
        u = counter_uniform((num_reqs, vocab), counter)
        ref = _synthetic_sample_and_assemble_impl(
            *args, u, *tail, k, apply_topk_topp=False
        )
        assert torch.equal(out.cpu(), ref.cpu())
        counter = nxt
        sampled = out[:, 2].cpu().long()
        half = num_reqs // 2
        rej_hist += torch.bincount(sampled[:half], minlength=vocab).float()
        bonus_hist += torch.bincount(sampled[half:], minlength=vocab).float()
    p = torch.softmax(base, dim=-1)
    p_rej = p.clone()
    p_rej[3] = 0
    p_rej /= p_rej.sum()
    assert rej_hist[3] == 0
    for hist, probs in ((rej_hist, p_rej), (bonus_hist, p)):
        n = hist.sum()
        exp = probs * n
        mask = exp > 5
        chi2 = (((hist[mask] - exp[mask]) ** 2) / exp[mask]).sum().item()
        dof = int(mask.sum()) - 1
        assert chi2 < dof + 6 * (2 * dof) ** 0.5, (chi2, dof)
