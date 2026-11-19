# SPDX-License-Identifier: Apache-2.0
"""q38 L4 LMH-STREAM stage 0 numerics: one TP8 rank's Qwen3.8 LM head (hidden 2560, vocab shard 31056, bf16)
stored [vocab, hidden] (today) against [hidden, vocab] (VLLM_TPU_LMH_HIDDEN_MAJOR_*). CDK CPU pod only.

  python3 tests/q38_lmhs0/q38_lmhs0_numerics.py <expected src root>   (from the head tree, PYTHONPATH=src)

Per row count M in 16 / 32 / 64 / 208 (batches of M rows, >= LMHS0_NUM_ROWS rows per cell, default 10,400, over
4 weight draws; stock / hm on the first LMHS0_NUM_REAL_ROWS rows, default 416), on synthetic data (x ~ N(0, 1), W ~ N(0, 1 / 2560); logits std about 1, a near-tie-heavy case):
  stock  = vLLM's UnquantizedEmbeddingMethod.apply on W [V, H]        (bf16 out; CPU kernel)
  hm     = HiddenMajorLMHeadMethod.apply on W.t().contiguous() [H, V] (bf16 out; CPU kernel)
  ref    = the f32 reference x.float() @ W.float().t()
  A / B  = an order model of the device: f32 sums over hidden tiles of 256 ascending (A) or 512 descending (B),
           each rounded to bf16, i.e. the same values summed in two different orders (the TPU's actual orders for
           the two layouts are not known on CPU; this is the class of change the layout switch can make).
Reported per cell: top-1 agreement stock==hm, A==B, and each against ref; bitwise logits equality stock==hm and
A==B; the largest difference in bf16 ulps; rows whose f32 top-2 gap is under one bf16 ulp of the top value.
PASS rule (fixed before the run): stock==hm (over its rows) and A==B top-1 agreement >= 99.99 % in every cell, and the
hidden-major method imported from <expected src root>. Last line: LMHS0_NUM PASS|FAIL [cells failing].
"""

import math
import os
import sys
import time

import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

import vllm_torchtpu.runner.utils as ru

H, V = 2560, 31056
ROWS = (16, 32, 64, 208)
TARGET = int(os.environ.get("LMHS0_NUM_ROWS", "10400"))
# The two real methods run bf16 matmuls, which a CPU without native bf16 runs slowly (num v1 j-e087dd72 timed out
# at 7,200 s): they take the first LMHS0_NUM_REAL_ROWS rows of each cell (default 416, where 99.99 % means every
# row); the f32 order model takes all of them.
REAL = int(os.environ.get("LMHS0_NUM_REAL_ROWS", "416"))
DRAWS = 4
GATE = 0.9999
BF16 = torch.bfloat16


def _layer(w):
    m = torch.nn.Module()
    m.weight = torch.nn.Parameter(w, requires_grad=False)
    return m


def _ulp(v):
    """One bf16 ulp at |v| (8 significant bits)."""
    a = v.abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def main():
    root = os.path.realpath(sys.argv[1]) if len(sys.argv) > 1 else None
    origin = os.path.realpath(ru.__file__)
    inside = root is not None and origin.startswith(root + os.sep)
    print(f"LMHS0_NUM_ORIGIN {origin} {'INSIDE' if inside else 'OUTSIDE'} (expected root {root})")
    torch.set_num_threads(int(os.environ.get("LMHS0_NUM_THREADS", str(min(8, os.cpu_count() or 8)))))
    print(f"LMHS0_NUM_ENV torch {torch.__version__} threads {torch.get_num_threads()} "
          f"mkldnn_bf16 {getattr(torch.backends.mkldnn, 'is_available', lambda: None)()} rows_target {TARGET} "
          f"real_rows {REAL}", flush=True)
    stock_m, hm_m = vpe.UnquantizedEmbeddingMethod(), ru.HiddenMajorLMHeadMethod()
    fails = []
    for M in ROWS:
        t0 = time.time()
        nb = math.ceil(TARGET / M)
        per = [nb // DRAWS + (1 if d < nb % DRAWS else 0) for d in range(DRAWS)]
        c = dict(n=0, nr=0, s_h=0, a_b=0, s_r=0, h_r=0, a_r=0, b_r=0, near=0, bit_sh=0, bit_ab=0, el=0, elr=0)
        ulp_sh = ulp_ab = 0.0
        for d in range(DRAWS):
            g = torch.Generator().manual_seed(1000 * M + d)
            w = (torch.randn(V, H, generator=g) * H ** -0.5).to(BF16)
            w_hm = w.t().contiguous()
            lw, lh = _layer(w), _layer(w_hm)
            w32 = w.float()
            ta = [(s, w32[:, s:s + 256].t().contiguous()) for s in range(0, H, 256)]
            tb = [(s, w32[:, s:s + 512].t().contiguous()) for s in range(0, H, 512)][::-1]
            for _ in range(per[d]):
                x = torch.randn(M, H, generator=g).to(BF16)
                real = c["nr"] < REAL
                if real:
                    stock = stock_m.apply(lw, x)
                    hm = hm_m.apply(lh, x)
                    assert stock.dtype == hm.dtype == BF16 and stock.shape == hm.shape == (M, V)
                x32 = x.float()
                ref = x32 @ w32.t()
                acc = torch.zeros(M, V)
                for s, t in ta:
                    acc += x32[:, s:s + 256] @ t
                a = acc.to(BF16)
                acc = torch.zeros(M, V)
                for s, t in tb:
                    acc += x32[:, s:s + 512] @ t
                b = acc.to(BF16)
                ta1, tb1 = (z.argmax(dim=-1) for z in (a, b))
                top2 = ref.topk(2, dim=-1).values
                tr = ref.argmax(dim=-1)
                c["n"] += M
                c["a_b"] += int((ta1 == tb1).sum())
                c["a_r"] += int((ta1 == tr).sum())
                c["b_r"] += int((tb1 == tr).sum())
                c["near"] += int(((top2[:, 0] - top2[:, 1]) < _ulp(top2[:, 0])).sum())
                c["bit_ab"] += int((a.view(torch.int16) == b.view(torch.int16)).sum())
                c["el"] += M * V
                af, bf = a.float(), b.float()
                ulp_ab = max(ulp_ab, float(((af - bf).abs() / _ulp(af)).max()))
                if real:
                    ts, th = stock.argmax(dim=-1), hm.argmax(dim=-1)
                    c["nr"] += M
                    c["s_h"] += int((ts == th).sum())
                    c["s_r"] += int((ts == tr).sum())
                    c["h_r"] += int((th == tr).sum())
                    c["bit_sh"] += int((stock.view(torch.int16) == hm.view(torch.int16)).sum())
                    c["elr"] += M * V
                    sf, hf = stock.float(), hm.float()
                    ulp_sh = max(ulp_sh, float(((sf - hf).abs() / _ulp(sf)).max()))
        n, nr = c["n"], c["nr"]

        def pct(k, d):
            return f"{c[k]}/{d} ({100.0 * c[k] / d:.4f}%)"

        ok = nr > 0 and c["s_h"] >= GATE * nr and c["a_b"] >= GATE * n
        print(f"LMHS0_NUM rows={M} n={n} top1 stock==hm {pct('s_h', nr)} A==B {pct('a_b', n)} | vs f32 ref: stock "
              f"{pct('s_r', nr)} hm {pct('h_r', nr)} A {pct('a_r', n)} B {pct('b_r', n)} | bitwise logits stock==hm "
              f"{100.0 * c['bit_sh'] / max(c['elr'], 1):.4f}% A==B {100.0 * c['bit_ab'] / c['el']:.4f}% | max ulp "
              f"stock/hm {ulp_sh:g} A/B {ulp_ab:g} | ref top-2 gap < 1 bf16 ulp {c['near']}/{n} | "
              f"{time.time() - t0:.1f}s {'ok' if ok else 'BELOW 99.99%'}", flush=True)
        if not ok:
            fails.append(str(M))
    if not inside:
        fails.append("origin")
    print("LMHS0_NUM PASS" if not fails else f"LMHS0_NUM FAIL {' '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
