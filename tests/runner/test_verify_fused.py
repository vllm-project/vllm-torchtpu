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
"""Exactness tests for VLLM_TPU_VERIFY_FUSED and VLLM_TPU_DRAFT_SEED_FUSED.

Compares `_verify_fused_impl` / `_verify_fused_compiled` with the separate
verify path across `apply_topk_topp` in `(True, False)`, `synthetic` in
`(True, False)`, and `topk_candidates`, and checks that the fused path's
uniform draws (`torch.rand` of the same shape) equal the separate path's
`torch.rand_like` draws from an identically seeded generator.
"""

import argparse
import itertools
import sys

import pytest
import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _random_rejection_sample_impl,
    _random_rejection_sample_with_segment,
)
from vllm_torchtpu.runner.tpu_runner import (
    _sample_from_logits_impl,
    _verify_fused_compiled,
    _verify_fused_impl,
)
from vllm_torchtpu.spec_decode.eagle3 import (
    _async_draft_seed_compiled,
    _async_draft_seed_impl,
    _async_seed_inputs_compiled,
    _async_seed_inputs_impl,
    _maybe_pad_dim0,
)

_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)


@_tpu_compile
def _gather(logits, bonus_idx, target_idx):
    return torch.index_select(logits, 0, bonus_idx), torch.index_select(
        logits, 0, target_idx
    )


_sample_compiled = _tpu_compile(_sample_from_logits_impl)


def separate_eager(args):
    (
        logits,
        bonus_idx,
        target_idx,
        temps,
        top_k,
        top_p,
        bonus_u,
        accept_u,
        recover_u,
        draft,
        draft_lengths,
        seg,
        grp,
        rates,
        k,
        synthetic,
        cand,
        apply_topk_topp,
    ) = args
    bonus_logits = torch.index_select(logits, 0, bonus_idx)
    target_logits = torch.index_select(logits, 0, target_idx)
    bonus = _sample_from_logits_impl(
        bonus_logits, temps, bonus_u, top_k, top_p, cand, apply_topk_topp
    ).view(-1)
    return _random_rejection_sample_impl(
        draft,
        target_logits,
        draft_lengths,
        bonus,
        seg,
        grp,
        k,
        temps[seg],
        top_k[seg],
        top_p[seg],
        accept_u,
        recover_u,
        rates,
        synthetic,
        cand,
        apply_topk_topp,
    )


def separate(args):
    (
        logits,
        bonus_idx,
        target_idx,
        temps,
        top_k,
        top_p,
        bonus_u,
        accept_u,
        recover_u,
        draft,
        draft_lengths,
        seg,
        grp,
        rates,
        k,
        synthetic,
        cand,
        apply_topk_topp,
    ) = args
    bonus_logits, target_logits = _gather(logits, bonus_idx, target_idx)
    bonus = _sample_compiled(
        bonus_logits, temps, bonus_u, top_k, top_p, cand, apply_topk_topp
    ).view(-1)
    return _random_rejection_sample_with_segment(
        draft,
        target_logits,
        draft_lengths,
        bonus,
        seg,
        grp,
        k,
        temps[seg],
        top_k[seg],
        top_p[seg],
        accept_u,
        recover_u,
        rates,
        synthetic,
        cand,
        apply_topk_topp,
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

    def to_dev(x):
        return x.to(device)

    return dict(
        logits=to_dev(logits),
        bonus_idx=to_dev(bonus_idx),
        target_idx=to_dev(target_idx),
        temps=to_dev(temps),
        top_k=to_dev(top_k),
        top_p=to_dev(top_p),
        draft=to_dev(draft),
        draft_lengths=to_dev(draft_lengths),
        seg=to_dev(seg),
        grp=to_dev(grp),
    )


def _seed_scatter_impl(input_ids, last_token_indices, next_token_ids):
    shifted = input_ids
    if input_ids.shape[0] > 1:
        shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
    return shifted.index_put((last_token_indices,), next_token_ids.to(input_ids.dtype))


_seed_scatter = _tpu_compile(_seed_scatter_impl)


def check_seed(device, seeds: int, *, compiled: bool = True) -> int:
    """VLLM_TPU_DRAFT_SEED_FUSED vs the separate seed / scatter / pad programs."""
    k = 3
    fail = 0
    seed_fn = _async_draft_seed_compiled if compiled else _async_draft_seed_impl
    scatter_fn = _seed_scatter if compiled else _seed_scatter_impl
    fused_fn = _async_seed_inputs_compiled if compiled else _async_seed_inputs_impl
    for width, total_tokens, qsl_len in (
        (4, 16, 33),
        (8, 32, 33),
        (16, 64, 33),
        (24, 128, 33),
    ):
        for seed in range(seeds):
            g = torch.Generator().manual_seed(seed)
            n = int(torch.randint(1, width + 1, (1,), generator=g))
            d = torch.randint(0, k + 1, (width,), generator=g)
            d[n:] = 0
            nt = torch.full((width, k + 1), -1, dtype=torch.int32)
            for r in range(n):
                acc = int(torch.randint(0, int(d[r]) + 1, (1,), generator=g))
                nt[r, : acc + 1] = torch.randint(
                    0, 1000, (acc + 1,), generator=g, dtype=torch.int32
                )
            q = torch.cumsum(
                torch.cat([torch.zeros(1, dtype=torch.int64), (d + 1).to(torch.int64)]),
                0,
            )
            qsl = torch.full((qsl_len,), int(q[n]), dtype=torch.int32)
            qsl[: n + 1] = q[: n + 1].to(torch.int32)
            ids = torch.randint(
                0, 1000, (total_tokens,), generator=g, dtype=torch.int32
            )
            num_reqs = torch.tensor(n, dtype=torch.int64)
            p = max(16, 1 << (n - 1).bit_length())
            args = (
                nt.to(device),
                d.to(torch.int32).to(device),
                qsl.to(device),
                num_reqs.to(device),
            )
            lti, nti, nrej = seed_fn(*args)
            ref_ids = scatter_fn(ids.to(device), lti, nti)
            ref_g = _maybe_pad_dim0(lti[:p], p)
            out_ids, out_g, out_nrej = fused_fn(*args, ids.to(device), p)
            ok = (
                torch.equal(ref_ids.cpu(), out_ids.cpu())
                and torch.equal(ref_g.cpu(), out_g.cpu())
                and torch.equal(nrej.cpu(), out_nrej.cpu())
            )
            if not ok:
                fail += 1
                if fail <= 10:
                    print(
                        f"SEED MISMATCH width={width} T={total_tokens} "
                        f"n={n} seed={seed}"
                    )
    return fail


@pytest.fixture
def tpu_device():
    dev = torch.device("tpu")
    probe = torch.tensor([17], dtype=torch.int32, device=dev)
    assert probe.cpu().tolist() == [17]
    return dev


@pytest.mark.parametrize("apply_topk_topp", [True, False])
@pytest.mark.parametrize("synthetic", [True, False])
@pytest.mark.parametrize("cand", [0, 2])
def test_verify_fused_impl_matches_separate_cpu(apply_topk_topp, synthetic, cand):
    if cand and not apply_topk_topp:
        pytest.skip("topk_candidates only applies when apply_topk_topp=True")
    device = torch.device("cpu")
    vocab = 512  # 4 blocks of CANDIDATE_BLOCK=128 -> top_k=1 <= cand // 2 for cand=2
    k = 3
    rates = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float32)
    params = (1.0, 1, 0.85)
    for nr, npad in ((1, 4), (3, 4)):
        for seed in range(4):
            g = torch.Generator().manual_seed(seed)
            c = make_case(g, nr, npad, k, vocab, params, device)
            n_tok = c["draft"].shape[0]
            ug = torch.Generator().manual_seed(1000003 * seed + 7)
            bu = torch.rand((npad, vocab), dtype=torch.float32, generator=ug)
            au = torch.rand((n_tok,), dtype=torch.float32, generator=ug)
            ru = torch.rand((n_tok, vocab), dtype=torch.float32, generator=ug)
            args = (
                c["logits"],
                c["bonus_idx"],
                c["target_idx"],
                c["temps"],
                c["top_k"],
                c["top_p"],
                bu,
                au,
                ru,
                c["draft"],
                c["draft_lengths"],
                c["seg"],
                c["grp"],
                rates if synthetic else None,
                k,
                synthetic,
                cand,
                apply_topk_topp,
            )
            assert torch.equal(separate_eager(args), _verify_fused_impl(*args))


def test_async_seed_inputs_impl_matches_separate_cpu():
    assert check_seed(torch.device("cpu"), seeds=5, compiled=False) == 0


@pytest.mark.tpu_test
@pytest.mark.parametrize("apply_topk_topp", [True, False])
@pytest.mark.parametrize("synthetic", [True, False])
def test_verify_fused_compiled_matches_separate_tpu(
    tpu_device, apply_topk_topp, synthetic
):
    vocab = 512
    k = 3
    rates = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float32, device=tpu_device)
    params = (1.0, 1, 0.85)
    cands = (0, 2) if apply_topk_topp else (0,)
    nr, npad = 3, 4
    for cand in cands:
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            c = make_case(g, nr, npad, k, vocab, params, tpu_device)
            n_tok = c["draft"].shape[0]
            g1 = torch.Generator(device=tpu_device).manual_seed(1000003 * seed + 7)
            bl, tl = _gather(c["logits"], c["bonus_idx"], c["target_idx"])
            bu1 = torch.rand_like(bl, dtype=torch.float32, generator=g1)
            au1 = torch.rand(
                c["draft"].shape, dtype=torch.float32, device=tpu_device, generator=g1
            )
            ru1 = torch.rand_like(tl, dtype=torch.float32, generator=g1)
            g2 = torch.Generator(device=tpu_device).manual_seed(1000003 * seed + 7)
            bu2 = torch.rand(
                (npad, vocab), dtype=torch.float32, device=tpu_device, generator=g2
            )
            au2 = torch.rand(
                (n_tok,), dtype=torch.float32, device=tpu_device, generator=g2
            )
            ru2 = torch.rand(
                (n_tok, vocab), dtype=torch.float32, device=tpu_device, generator=g2
            )
            assert torch.equal(bu1, bu2)
            assert torch.equal(au1, au2)
            assert torch.equal(ru1, ru2)
            args = (
                c["logits"],
                c["bonus_idx"],
                c["target_idx"],
                c["temps"],
                c["top_k"],
                c["top_p"],
                bu2,
                au2,
                ru2,
                c["draft"],
                c["draft_lengths"],
                c["seg"],
                c["grp"],
                rates if synthetic else None,
                k,
                synthetic,
                cand,
                apply_topk_topp,
            )
            ref = separate(args)
            out = _verify_fused_compiled(*args)
            assert torch.equal(ref.cpu(), out.cpu())


@pytest.mark.tpu_test
def test_draft_seed_fused_compiled_matches_separate_tpu(tpu_device):
    assert check_seed(tpu_device, seeds=4, compiled=True) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--vocab", type=int, default=248320)
    a = ap.parse_args()
    device = torch.device("tpu")
    k = 3
    rates = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float32).to(device)
    param_sets = [(1.0, 20, 0.95), (0.6, 0, 0.8), (0.0, 0, 1.0)]
    buckets = [(1, 4), (3, 4), (6, 8), (16, 16)]
    fail = 0
    checked = 0
    for (nr, npad), params, synthetic, cand, apply_topk_topp in itertools.product(
        buckets, param_sets, (True, False), (0, 64), (True, False)
    ):
        if cand and (params[1] == 0 or not apply_topk_topp):
            continue
        seed_limit = a.seeds if (nr, npad) == (1, 4) else max(a.seeds // 10, 20)
        for seed in range(seed_limit):
            g = torch.Generator().manual_seed(seed)
            c = make_case(g, nr, npad, k, a.vocab, params, device)
            n_tok = c["draft"].shape[0]
            g1 = torch.Generator(device=device)
            g1.manual_seed(1000003 * seed + 7)
            bl, tl = _gather(c["logits"], c["bonus_idx"], c["target_idx"])
            bu1 = torch.rand_like(bl, dtype=torch.float32, generator=g1)
            au1 = torch.rand(
                c["draft"].shape, dtype=torch.float32, device=device, generator=g1
            )
            ru1 = torch.rand_like(tl, dtype=torch.float32, generator=g1)
            g2 = torch.Generator(device=device)
            g2.manual_seed(1000003 * seed + 7)
            bu2 = torch.rand(
                (npad, a.vocab), dtype=torch.float32, device=device, generator=g2
            )
            au2 = torch.rand((n_tok,), dtype=torch.float32, device=device, generator=g2)
            ru2 = torch.rand(
                (n_tok, a.vocab), dtype=torch.float32, device=device, generator=g2
            )
            rng_eq = bool(
                torch.equal(bu1, bu2)
                and torch.equal(au1, au2)
                and torch.equal(ru1, ru2)
            )
            args = (
                c["logits"],
                c["bonus_idx"],
                c["target_idx"],
                c["temps"],
                c["top_k"],
                c["top_p"],
                bu2,
                au2,
                ru2,
                c["draft"],
                c["draft_lengths"],
                c["seg"],
                c["grp"],
                rates if synthetic else None,
                k,
                synthetic,
                cand,
                apply_topk_topp,
            )
            ref = separate(args)
            out = _verify_fused_compiled(*args)
            eq = bool(torch.equal(ref.cpu(), out.cpu()))
            checked += 1
            if not (eq and rng_eq):
                fail += 1
                if fail <= 20:
                    print(
                        f"MISMATCH bucket={(nr, npad)} params={params} "
                        f"synthetic={synthetic} cand={cand} "
                        f"apply_topk_topp={apply_topk_topp} seed={seed} "
                        f"tokens_equal={eq} rng_equal={rng_eq}"
                    )
    print(f"VERIFY_FUSED_TPU_CHECK checked={checked} failures={fail}")
    seed_fail = check_seed(device, max(a.seeds // 3, 20), compiled=True)
    return 1 if (fail or seed_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
