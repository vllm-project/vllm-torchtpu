# SPDX-License-Identifier: Apache-2.0
"""TPU exactness check for VLLM_TPU_VERIFY_FUSED (standalone; run on a TPU pod).

Compares `_verify_fused_compiled` with the separate-program verify path it
replaces, on the same inputs and uniforms, and checks that the fused path's
uniform draws (`torch.rand` of the same shape) equal the separate path's
`torch.rand_like` draws from an identically seeded generator.

The separate path is rebuilt from the same bodies with today's program
boundaries: a compiled gather, a compiled bonus sample, eager per-position
parameter gathers and the compiled rejection sampler.

usage: python tests/runner/verify_fused_tpu_check.py [--seeds 1000] [--vocab 248320]
Exit 0 only if every compared tensor is bit-equal.
"""

import argparse
import itertools
import sys

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _random_rejection_sample_with_segment,
)
from vllm_torchtpu.spec_decode.eagle3 import (
    _async_draft_seed_compiled,
    _async_seed_inputs_compiled,
    _maybe_pad_dim0,
)
from vllm_torchtpu.runner.tpu_runner import (
    _sample_from_logits_impl,
    _verify_fused_compiled,
)

# The server raises dynamo's limits for its per-bucket graphs; do the same so
# the sweep below (buckets x params x synthetic x candidates) can compile.
torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config.accumulated_recompile_limit = 8192

_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)


@_tpu_compile
def _gather(logits, bonus_idx, target_idx):
    return torch.index_select(logits, 0, bonus_idx), torch.index_select(
        logits, 0, target_idx
    )


_sample_compiled = _tpu_compile(_sample_from_logits_impl)


def separate(args):
    (logits, bonus_idx, target_idx, temps, top_k, top_p, bonus_u, accept_u,
     recover_u, draft, draft_lengths, seg, grp, rates, k, synthetic, cand) = args
    bonus_logits, target_logits = _gather(logits, bonus_idx, target_idx)
    bonus = _sample_compiled(bonus_logits, temps, bonus_u, top_k, top_p, cand).view(-1)
    return _random_rejection_sample_with_segment(
        draft, target_logits, draft_lengths, bonus, seg, grp, k,
        temps[seg], top_k[seg], top_p[seg], accept_u, recover_u, rates, synthetic, cand,
    )


def make_case(gen_cpu, num_reqs_real, num_reqs_pad, k, vocab, params, device):
    """One verify chunk laid out like get_spec_decode_metadata: per request
    d in [0, K] drafts + 1 bonus row; positions padded to a token bucket."""
    d = torch.randint(0, k + 1, (num_reqs_real,), generator=gen_cpu)
    total = int(d.sum())
    num_tokens = max(16, 1 << (max(total, num_reqs_pad) - 1).bit_length())
    rows = num_tokens + num_reqs_pad
    logits = (torch.randn(rows, vocab, generator=gen_cpu) * 4).to(torch.bfloat16)
    target_idx = torch.zeros(num_tokens, dtype=torch.int32)
    seg = torch.full((num_tokens,), num_reqs_real, dtype=torch.int64)
    grp = torch.zeros(num_tokens, dtype=torch.int32)
    draft = torch.full((num_tokens,), -1, dtype=torch.int32)
    pos = 0
    for r in range(num_reqs_real):
        for j in range(int(d[r])):
            target_idx[pos] = pos
            seg[pos] = r
            grp[pos] = j
            # Drafts: the row argmax half the time (so accepts happen), else random.
            if torch.rand(1, generator=gen_cpu).item() < 0.5:
                draft[pos] = int(logits[pos].float().argmax())
            else:
                draft[pos] = int(torch.randint(0, vocab, (1,), generator=gen_cpu))
            pos += 1
    bonus_idx = torch.arange(num_reqs_pad, dtype=torch.int32) + num_tokens
    draft_lengths = torch.zeros(num_reqs_pad, dtype=torch.int32)
    draft_lengths[:num_reqs_real] = d.to(torch.int32)
    temp, tk, tp = params
    temps = torch.ones(num_reqs_pad, 1, dtype=torch.bfloat16)
    top_k = torch.zeros(num_reqs_pad, 1, dtype=torch.int32)
    top_p = torch.ones(num_reqs_pad, 1, dtype=torch.float32)
    temps[:num_reqs_real] = temp
    top_k[:num_reqs_real] = tk
    top_p[:num_reqs_real] = tp
    t = lambda x: x.to(device)  # noqa: E731
    return dict(
        logits=t(logits), bonus_idx=t(bonus_idx), target_idx=t(target_idx),
        temps=t(temps), top_k=t(top_k), top_p=t(top_p), draft=t(draft),
        draft_lengths=t(draft_lengths), seg=t(seg), grp=t(grp),
    )


@_tpu_compile
def _seed_scatter(input_ids, last_token_indices, next_token_ids):
    shifted = input_ids
    if input_ids.shape[0] > 1:
        shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
    return shifted.index_put((last_token_indices,), next_token_ids.to(input_ids.dtype))


def check_seed(device, seeds: int) -> int:
    """VLLM_TPU_DRAFT_SEED_FUSED vs the separate seed / scatter / pad programs."""
    k = 3
    fail = 0
    for width, T, qsl_len in ((4, 16, 33), (8, 32, 33), (16, 64, 33), (24, 128, 33)):
        for seed in range(seeds):
            g = torch.Generator().manual_seed(seed)
            n = int(torch.randint(1, width + 1, (1,), generator=g))
            d = torch.randint(0, k + 1, (width,), generator=g)
            d[n:] = 0
            nt = torch.full((width, k + 1), -1, dtype=torch.int32)
            for r in range(n):
                acc = int(torch.randint(0, int(d[r]) + 1, (1,), generator=g))
                nt[r, : acc + 1] = torch.randint(0, 1000, (acc + 1,), generator=g, dtype=torch.int32)
            q = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), (d + 1).to(torch.int64)]), 0)
            qsl = torch.full((qsl_len,), int(q[n]), dtype=torch.int32)
            qsl[: n + 1] = q[: n + 1].to(torch.int32)
            ids = torch.randint(0, 1000, (T,), generator=g, dtype=torch.int32)
            num_reqs = torch.tensor(n, dtype=torch.int64)
            p = max(16, 1 << (n - 1).bit_length())
            dv = lambda x: x.to(device)  # noqa: E731
            args = (dv(nt), dv(d.to(torch.int32)), dv(qsl), dv(num_reqs))
            lti, nti, nrej = _async_draft_seed_compiled(*args)
            ref_ids = _seed_scatter(dv(ids), lti, nti)
            ref_g = _maybe_pad_dim0(lti[:p], p)
            out_ids, out_g, out_nrej = _async_seed_inputs_compiled(*args, dv(ids), p)
            ok = (torch.equal(ref_ids.cpu(), out_ids.cpu()) and torch.equal(ref_g.cpu(), out_g.cpu())
                  and torch.equal(nrej.cpu(), out_nrej.cpu()))
            if not ok:
                fail += 1
                if fail <= 10:
                    print(f"SEED MISMATCH width={width} T={T} n={n} seed={seed}")
    print(f"DRAFT_SEED_FUSED_TPU_CHECK failures={fail}", flush=True)
    return fail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--vocab", type=int, default=248320)
    a = ap.parse_args()
    device = torch.device("tpu")
    k = 3
    rates = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float32).to(device)
    # Served AgentX params (generation_config: T=1.0, top_k=20, top_p=0.95) first.
    param_sets = [(1.0, 20, 0.95), (0.6, 0, 0.8), (0.0, 0, 1.0)]
    buckets = [(1, 4), (3, 4), (6, 8), (16, 16)]
    fail = 0
    checked = 0
    for (nr, npad), params, synthetic, cand in itertools.product(
        buckets, param_sets, (True, False), (0, 64)
    ):
        if cand and params[1] == 0:
            continue  # candidates only apply when 1 <= top_k <= blocks // 2
        for seed in range(a.seeds if (nr, npad) == (1, 4) else max(a.seeds // 10, 20)):
            g = torch.Generator().manual_seed(seed)
            c = make_case(g, nr, npad, k, a.vocab, params, device)
            n_tok = c["draft"].shape[0]
            # Separate path's draws (rand_like on the gathered tensors) vs the
            # fused path's draws (rand of the same shapes), same seed.
            g1 = torch.Generator(device=device)
            g1.manual_seed(1000003 * seed + 7)
            bl, tl = _gather(c["logits"], c["bonus_idx"], c["target_idx"])
            bu1 = torch.rand_like(bl, dtype=torch.float32, generator=g1)
            au1 = torch.rand(c["draft"].shape, dtype=torch.float32, device=device, generator=g1)
            ru1 = torch.rand_like(tl, dtype=torch.float32, generator=g1)
            g2 = torch.Generator(device=device)
            g2.manual_seed(1000003 * seed + 7)
            bu2 = torch.rand((npad, a.vocab), dtype=torch.float32, device=device, generator=g2)
            au2 = torch.rand((n_tok,), dtype=torch.float32, device=device, generator=g2)
            ru2 = torch.rand((n_tok, a.vocab), dtype=torch.float32, device=device, generator=g2)
            rng_eq = bool(torch.equal(bu1, bu2) and torch.equal(au1, au2) and torch.equal(ru1, ru2))
            args = (c["logits"], c["bonus_idx"], c["target_idx"], c["temps"], c["top_k"],
                    c["top_p"], bu2, au2, ru2, c["draft"], c["draft_lengths"], c["seg"],
                    c["grp"], rates if synthetic else None, k, synthetic, cand)
            ref = separate(args)
            out = _verify_fused_compiled(*args)
            eq = bool(torch.equal(ref.cpu(), out.cpu()))
            checked += 1
            if not (eq and rng_eq):
                fail += 1
                if fail <= 20:
                    print(f"MISMATCH bucket={(nr, npad)} params={params} synthetic={synthetic} "
                          f"cand={cand} seed={seed} tokens_equal={eq} rng_equal={rng_eq}")
                    if not eq:
                        print("  ref", ref.cpu().tolist())
                        print("  out", out.cpu().tolist())
        print(f"done bucket={(nr, npad)} params={params} synthetic={synthetic} cand={cand} "
              f"checked={checked} failures={fail}", flush=True)
    print(f"VERIFY_FUSED_TPU_CHECK checked={checked} failures={fail}")
    seed_fail = check_seed(device, max(a.seeds // 3, 20))
    return 1 if (fail or seed_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
