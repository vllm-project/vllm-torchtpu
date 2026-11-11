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
"""Microbenchmark of the non-greedy spec-decode verify, unfused vs fused.

The unfused arm is the served sequence of `_sample_spec_verify_chunk`:
`spec_gather_bonus_and_target_logits`, the bonus draw, the bonus sampler, the
accept and recover draws, three `[segment_ids]` gathers and the rejection
sampler. The fused arms run `fused_spec_verify.spec_verify_sample_fused` on
the same draws.

Each verify reads its own logits (`--sets` distinct tensors, cycled), the
loop is not synchronized between iterations (the served step is not either),
and the time per verify is the slope (t(2N) - t(N)) / N. Before timing, every
fused arm is checked token-exact against the unfused arm on each set.

Usage (TPU, one chip):
  python scripts/benchmark_fused_spec_verify.py
  python scripts/benchmark_fused_spec_verify.py --shapes=512:64,128:16 --real
CPU (correctness only): add --cpu --vocab=4096 --iters=2
"""

import argparse
import functools
import time

import numpy as np
import torch

from vllm_torchtpu.layers.adapter.sample import fused_spec_verify
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.utils import synchronize_tensors

_compiled_sample_rows = torch.compile(
    fused_spec_verify.sample_rows, backend="tpu", fullgraph=True, dynamic=False
)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _spec_gather(logits, bonus_idx, target_idx):
    return (
        torch.index_select(logits, 0, bonus_idx),
        torch.index_select(logits, 0, target_idx),
    )


class _Case:
    def __init__(self, num_tokens, num_reqs, vocab, k, seed, device, set_id):
        rng = np.random.default_rng(seed * 1000 + set_id)
        lens = np.full(num_reqs, k)
        # Fill the bucket as the served full-decode step does: every request
        # verifies k drafts, the rows past num_reqs * (k + 1) are padding.
        real_reqs = min(num_reqs, num_tokens // (k + 1))
        lens[real_reqs:] = 0
        num_sampled = lens + 1
        num_sampled[real_reqs:] = 0
        cu = np.cumsum(num_sampled)
        starts = cu - num_sampled
        bonus = np.zeros(num_reqs, np.int32)
        bonus[:real_reqs] = cu[:real_reqs] - 1
        t = np.concatenate([s + np.arange(n) for s, n in zip(starts, lens)])
        target = np.zeros(num_tokens, np.int32)
        target[: t.size] = t
        seg = np.full(num_tokens, real_reqs, np.int64)
        seg[: t.size] = np.repeat(np.arange(num_reqs), lens)
        grp = np.zeros(num_tokens, np.int32)
        grp[: t.size] = np.concatenate([np.arange(n) for n in lens if n] or [[]])
        g = torch.Generator().manual_seed(seed * 1000 + set_id)
        logits = torch.randn((num_tokens, vocab), generator=g) * 3
        draft = np.zeros(num_tokens, np.int32)
        am = logits.argmax(-1).numpy()
        for i, row in enumerate(t):
            draft[i] = am[row] if rng.random() < 0.7 else rng.integers(vocab)
        self.real_reqs = real_reqs
        self.logits = logits.to(torch.bfloat16).to(device)
        self.bonus = torch.from_numpy(bonus).to(device)
        self.target = torch.from_numpy(target).to(device)
        self.draft = torch.from_numpy(draft).to(device)
        nd = np.zeros(num_reqs, np.int32)
        nd[:] = lens
        self.lens = torch.from_numpy(nd).to(device)
        self.seg = torch.from_numpy(seg).to(device)
        # Served: `req_temperatures[segment_ids]` with padding ids == num_reqs
        # (the TPU gather clamps). Eager CPU raises instead, so clamp there.
        self.param_rows = (
            self.seg.clamp(max=num_reqs - 1) if device.type == "cpu" else self.seg
        )
        self.grp = torch.from_numpy(grp).to(device)
        temps = torch.ones((num_reqs, 1), dtype=torch.bfloat16)
        self.temps = temps.to(device)
        self.top_k = torch.zeros((num_reqs, 1), dtype=torch.int32).to(device)
        self.top_p = torch.full((num_reqs, 1), 0.95, dtype=torch.float32).to(device)


def _draws(case, gen):
    vocab = case.logits.shape[-1]
    dev = case.logits.device
    bonus_u = torch.rand(
        (case.bonus.shape[0], vocab), dtype=torch.float32, device=dev, generator=gen
    )
    accept_u = torch.rand(
        case.draft.shape, dtype=torch.float32, device=dev, generator=gen
    )
    recover_u = torch.rand(
        (case.target.shape[0], vocab), dtype=torch.float32, device=dev, generator=gen
    )
    return bonus_u, accept_u, recover_u


def unfused(case, sampler, k, gen):
    bonus_logits, target_logits = _spec_gather(case.logits, case.bonus, case.target)
    bonus_u = torch.rand_like(bonus_logits, dtype=torch.float32, generator=gen)
    bonus_ids = _compiled_sample_rows(
        bonus_logits,
        case.temps,
        bonus_u,
        case.top_k,
        case.top_p,
        0,
        True,
        False,
    ).view(-1)
    accept_u = torch.rand(
        case.draft.shape, dtype=torch.float32, device=case.logits.device, generator=gen
    )
    recover_u = torch.rand_like(target_logits, dtype=torch.float32, generator=gen)
    return sampler(
        draft_token_ids=case.draft,
        num_draft_tokens=case.lens,
        target_logits=target_logits,
        bonus_token_ids=bonus_ids,
        segment_ids=case.seg,
        group_indices=case.grp,
        max_draft_tokens=k,
        temperatures=case.temps[case.param_rows],
        top_k=case.top_k[case.param_rows],
        top_p=case.top_p[case.param_rows],
        accept_u=accept_u,
        recover_u=recover_u,
        do_sampling=True,
        apply_top_k=False,
    )


def fused(case, sampler, k, gen):
    bonus_u, accept_u, recover_u = _draws(case, gen)
    return fused_spec_verify.spec_verify_sample_fused(
        case.logits,
        case.bonus,
        case.target,
        case.draft,
        case.lens,
        case.seg,
        case.grp,
        case.temps,
        case.top_k,
        case.top_p,
        bonus_u,
        accept_u,
        recover_u,
        sampler.synthetic_conditional_rates,
        sampler.synthetic_mode,
        k,
        topk_candidates=0,
        apply_topk_topp=True,
        apply_top_k=False,
    )


def _time(fn, cases, n):
    def run(iters):
        outs = []
        start = time.perf_counter()
        for i in range(iters):
            outs.append(fn(cases[i % len(cases)]))
        synchronize_tensors(outs)
        return time.perf_counter() - start

    run(4)
    t1 = min(run(n) for _ in range(3))
    t2 = min(run(2 * n) for _ in range(3))
    return (t2 - t1) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="512:64,256:32,128:16")
    p.add_argument("--vocab", type=int, default=200_064)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--sets", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--real", action="store_true", help="also non-synthetic")
    p.add_argument("--cpu", action="store_true", help="eager CPU dry run")
    args = p.parse_args()
    if args.cpu:
        torch.compiler.set_stance("force_eager")
        device = torch.device("cpu")
    else:
        import torch_tpu  # noqa: F401

        device = torch.device("tpu")
    # One graph per shape and arm, as in the runner (which raises it too).
    torch._dynamo.config.recompile_limit = 1024
    modes = [True] + ([False] if args.real else [])
    for synthetic in modes:
        cfg = None
        if synthetic:
            from types import SimpleNamespace

            cfg = SimpleNamespace(
                rejection_sample_method="synthetic",
                synthetic_acceptance_rates=None,
            )
            # AL 4.04 of k=7, the served setting.
            # vLLM's SpeculativeConfig._acceptance_length_to_rates.
            full = 3
            cfg.synthetic_acceptance_rates = (
                [1.0] * full + [4.04 - 1 - full] + [0.0] * (args.k - full - 1)
            )
        sampler = RejectionSampler(cfg, device)
        for shape in args.shapes.split(","):
            num_tokens, num_reqs = map(int, shape.split(":"))
            cases = [
                _Case(num_tokens, num_reqs, args.vocab, args.k, 0, device, s)
                for s in range(args.sets)
            ]
            # Exactness: same generator seed per arm, three draws per set.
            ref = []
            for c in cases:
                gen = torch.Generator(device=device).manual_seed(123)
                ref.append(unfused(c, sampler, args.k, gen).cpu())
            for c, want in zip(cases, ref):
                gen = torch.Generator(device=device).manual_seed(123)
                got = fused(c, sampler, args.k, gen).cpu()
                if not torch.equal(got, want):
                    diff = (got != want).any(-1).nonzero().view(-1).tolist()
                    print(f"MISMATCH synthetic={synthetic} shape={shape} rows={diff}")
            gen = torch.Generator(device=device).manual_seed(7)
            t_ref = _time(
                functools.partial(unfused, sampler=sampler, k=args.k, gen=gen),
                cases,
                args.iters,
            )
            t = _time(
                functools.partial(fused, sampler=sampler, k=args.k, gen=gen),
                cases,
                args.iters,
            )
            line = (
                f"synthetic={synthetic} tokens={num_tokens} reqs={num_reqs} "
                f"unfused {t_ref * 1e3:.3f} ms | fused {t * 1e3:.3f} ms"
            )
            print(line, flush=True)
            accepted = (ref[0][: cases[0].real_reqs] != -1).sum(-1) - 1
            print(f"  accepted per request (set 0): {accepted.tolist()[:16]} ...")


if __name__ == "__main__":
    main()
