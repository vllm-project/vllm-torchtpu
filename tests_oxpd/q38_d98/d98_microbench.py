# SPDX-License-Identifier: Apache-2.0
"""q38 L4 D98 CPU dispatch-overhead microbench (label M, CPU caveat).

Per-call host microseconds of one compiled helper dispatch on a CPU pod with
the aot_eager stand-in backend (Dynamo cache entries, guards, transformed code
and the AOTAutograd runtime wrapper as on TPU; the kernels are tiny CPU ops):
eager Python, the plain compiled call, the D98 fast call (guard-checked direct
dispatch), the fast call with the guard memo, and with the memo in check mode.
The saving (plain - fast) is the Dynamo frame this lever removes; on the TPU
host the absolute numbers differ (slower CPU, torch_tpu launch below the
graph). Every fast variant must engage on every measured call, or it aborts.
"""

import statistics
import sys
import time
import types

import torch

from vllm_torchtpu.compilation import fast_compiled_call as fcc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7


def clone(fn):
    g = types.FunctionType(
        fn.__code__.replace(), fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    g.__module__ = fn.__module__
    g.__qualname__ = fn.__qualname__
    return g


def h_tiny(x):
    return x * 2


def h_vps_like(logits, shard_offset, bidx, tidx, temps, top_k, top_p, bu, au, ru,
               drafts, dlen, seg, grp, rates, max_draft, synthetic, topk_c, num_reqs,
               fb_rows, fb_u):
    """21 positional arguments shaped like _vps_verify_fb_compiled's call."""
    b = torch.index_select(logits, 0, bidx) + temps.unsqueeze(1)
    t = torch.index_select(logits, 0, tidx) * top_p.mean()
    n = b.shape[0]
    out = b.sum(1) + t.sum(1)[:n] + bu + au[:n] + ru[:n] + (dlen + seg[:n] + grp[:n]).float()
    if rates is not None:
        out = out + rates.sum()
    out = out + float(shard_offset + max_draft + topk_c + fb_rows) + drafts.float().sum()
    if synthetic:
        out = out * 2
    return out + fb_u.sum() + top_k.float().sum() + num_reqs


def h_update_like(read_offsets, groups, next_tokens, num_reqs):
    """Shaped like _update_read_offsets_impl (list arg, 0-d tensor, in place)."""
    width = groups[0].shape[0]
    nt = next_tokens[:width]
    num_valid = (nt != -1).sum(dim=1)
    valid = torch.arange(width) < num_reqs
    offsets = torch.where(valid, (num_valid - 1).clamp(min=0), 0).to(read_offsets.dtype)
    for g in groups:
        read_offsets.index_put_((g.long(),), offsets)


class Owner:
    def __init__(self):
        self.w = torch.randn(8)
        self.k = 2

    def h_method(self, x):
        return x * self.w * self.k


def cases():
    B = 16
    vps = (torch.randn(64, 8), 0, torch.arange(B), torch.arange(48) % 64, torch.rand(B),
           torch.full((B,), 20), torch.rand(B), torch.rand(B), torch.rand(48), torch.rand(48),
           torch.randint(0, 9, (48,)), torch.full((B,), 3), torch.arange(48) % B,
           torch.arange(48), None, 3, False, 0, torch.tensor(12), 8, torch.rand(8))
    upd = (torch.zeros(64, dtype=torch.int32),
           [torch.randperm(64)[:16].to(torch.int32) for _ in range(2)],
           torch.randint(0, 100, (16, 4)), torch.tensor(12))
    o = Owner()
    return [
        ("tiny_1tensor", h_tiny, (torch.randn(8),)),
        ("update_like_list_inplace", h_update_like, upd),
        ("vps_like_21args", h_vps_like, vps),
        ("method_self_attr_guards", Owner.h_method, (o, torch.randn(8))),
    ]


def per_call_us(fn, args):
    for _ in range(50):
        fn(*args)
    reps = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        for _ in range(N):
            fn(*args)
        reps.append((time.perf_counter_ns() - t0) / N / 1e3)
    return statistics.median(reps)


def main():
    fcc.ELIGIBLE_BACKENDS = ("tpu", "aot_eager")
    fcc.MODULE_PREFIXES = ("vllm_torchtpu", __name__)
    print(f"D98MB_ENV python {sys.version.split()[0]} torch {torch.__version__} "
          f"threads {torch.get_num_threads()} N {N} REPS {REPS} (M, CPU pod, aot_eager stand-in)")
    bad = 0
    with torch.no_grad(), torch._dynamo.config.patch(recompile_limit=64):
        for name, fn, args in cases():
            res = {"eager": per_call_us(fn, args)}
            plain = torch.compile(clone(fn), backend="aot_eager", fullgraph=True, dynamic=False)
            res["plain"] = per_call_us(plain, args)
            for label, memo in (("fast", ""), ("memo", "1"), ("check", "check")):
                fcc._MEMO_MODE = memo
                fcc._reset_stats_for_tests()
                w = fcc.make_hook(torch.compile, fast=True, drop_global_state=False)(
                    clone(fn), backend="aot_eager", fullgraph=True, dynamic=False)
                assert hasattr(w, "_d98_fast_state"), name
                res[label] = per_call_us(w, args)
                s = w._d98_fast_state
                calls = 50 + REPS * N
                ok = s["slow"] == 1 and s["fast"] == calls - 1
                if memo:
                    ok = ok and (fcc.STATS["memo_hit"] >= calls - 3
                                 or fcc.STATS["memo_unsafe"] > 0)
                if memo == "check":
                    ok = ok and fcc.STATS["memo_mismatch"] == 0
                res[label + "_engaged"] = ok
                res[label + "_memo_hit"] = fcc.STATS["memo_hit"]
                res[label + "_memo_unsafe"] = fcc.STATS["memo_unsafe"]
                bad += not ok
            print(
                f"D98MB case={name} eager_us={res['eager']:.2f} plain_us={res['plain']:.2f} "
                f"fast_us={res['fast']:.2f} memo_us={res['memo']:.2f} check_us={res['check']:.2f} "
                f"save_fast_us={res['plain'] - res['fast']:.2f} "
                f"save_memo_us={res['plain'] - res['memo']:.2f} "
                f"engaged fast={res['fast_engaged']} memo={res['memo_engaged']} "
                f"check={res['check_engaged']} memo_hits={res['memo_memo_hit']} "
                f"memo_unsafe={res['memo_memo_unsafe']}"
            )
    print(f"D98MB_SUMMARY {'PASS' if not bad else 'FAIL'} not_engaged={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
