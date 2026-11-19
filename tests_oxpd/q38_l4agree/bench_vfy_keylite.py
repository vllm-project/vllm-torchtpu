# SPDX-License-Identifier: Apache-2.0
"""q38 L4 KEYLITE microbench (AGREE-BUILD; the deciding test of H2 card T1-VFY-KEYLITE). CDK CPU pod only:
    PYTHONPATH=src python3 -u tests/q38_l4agree/bench_vfy_keylite.py
Times, per VFY step, the real runner/vfy_one.key + warm-set membership (today's warm-key test, "full") against
KEYLITE's per-step path (lite_key + VfyOne.warm_check on a memo hit, "lite"), KEYLITE's amortized cost with the
full-key audit every KEYLITE_AUDIT_EVERY calls, and versions() of the DH-C stash (unchanged by KEYLITE). Calls are
built as tpu_runner._vfy_verify builds them, with real torch CPU tensors, at served Flash shapes: K = 3, 37
read-offset groups of width 16 (served log 'VPS_FUSED_READ_OFFSETS: folded N chunks (groups=37 width=16'), live
requests n = 4 / 9 / 11 (padded rows 8 / 16 / 16, token buckets 16 / 48 / 48), fallback rows 0 and 8, VFY value 1
(no seed) and 2 (DraftAsm's superset seed). Each case also checks that both paths give the same verdict.
Lines: 'Q38_KEYLITE_BENCH ...' per case, then 'Q38_KEYLITE_BENCH_DONE ok=<bool>'."""
import importlib.metadata as md_
import os
import platform
import statistics
import sys
import time
from types import SimpleNamespace

import torch

from vllm_torchtpu.runner import vfy_one as V

K, G, W, VOCAB, H, CAP, MAXR = 3, 37, 16, 4096, 4096, 64, 64
PADS = [8, 16, 32, 64]
TOKS = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]


def seed_super(*a):
    pass


def bucket(xs, n):
    return next(x for x in xs if x >= n)


def build(n, F, value):
    P, T = bucket(PADS, n), bucket(TOKS, n * (K + 1))
    D = P * K
    t = lambda *s, d=torch.int32: torch.zeros(s, dtype=d)  # noqa: E731
    hidden, indices = t(T, H, d=torch.bfloat16), t(P * (K + 1))
    md = SimpleNamespace(bonus_logits_indices=t(P), target_logits_indices=t(D), draft_token_ids=t(D),
                         draft_lengths=t(P), segment_ids=t(D), group_indices=t(D))
    args = (md.bonus_logits_indices, md.target_logits_indices, t(P, 1, d=torch.float32), t(P, 1),
            t(P, 1, d=torch.float32), t(P, VOCAB, d=torch.float32), t(D, d=torch.float32),
            t(D, VOCAB, d=torch.float32), md.draft_token_ids, md.draft_lengths, md.segment_ids, md.group_indices,
            t(K + 1, d=torch.float32), K, "conditional")
    groups = [t(W) for _ in range(G)]
    fb_u = t(F, VOCAB, d=torch.float32) if F else None
    call = (object(), hidden, indices, *args, t(1), F, True, fb_u, list(groups), t(), torch.int32,
            t(3, d=torch.int64))
    spec = None
    if value == 2:
        base = (t(P), t(CAP + 1), t(), t(T), bucket(PADS, n), CAP)
        spec = V.VfySeedSpec(seed_super, (*base[:4], t(CAP), base[4], base[5], MAXR), base)
        call = (*call, spec.fn, *spec.args)
    return call, (hidden, indices, P, md, F, call[22], spec), spec, T, P


def timed(fn, steps, reps=7):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(steps):
            fn()
        out.append((time.perf_counter_ns() - t0) / steps / 1e3)
    return min(out), statistics.median(out)


def main():
    cpu = "?"
    try:
        cpu = next(ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name"))
    except Exception:  # noqa: BLE001
        pass
    vers = {}
    for name in ("torch", "vllm", "torch_tpu"):
        try:
            vers[name] = md_.version(name)
        except Exception:  # noqa: BLE001
            vers[name] = "absent"
    print(f"Q38_KEYLITE_BENCH_ENV python {platform.python_version()} {vers} cpu '{cpu}' nproc {os.cpu_count()} "
          f"vfy_one {os.path.realpath(V.__file__)}", flush=True)
    ok = True
    steps = int(os.environ.get("Q38_KEYLITE_STEPS", "20000"))
    for value in (1, 2):
        for F in (0, 8):
            for n in (4, 9, 11):
                call, lite_in, spec, T, P = build(n, F, value)
                warm = {V.key(*build(m, F, value)[0][1:]) for m in range(1, 17)}  # the steady keys around it
                full_verdict = V.key(*call[1:]) in warm
                v = V.VfyOne(value, 0, "", keylite=True)
                v.warm = warm
                lite = V.lite_key(*lite_in)
                v.warm_check(call, lite, False)  # fill the memo
                v.lite_calls = 1  # keep the timed calls off the audit step (timed separately below)

                def full():
                    return V.key(*call[1:]) in warm

                def hit():
                    v.lite_calls = 1
                    return v.warm_check(call, V.lite_key(*lite_in), False)
                same = full() is hit() is full_verdict is True
                ok = ok and same
                f_min, f_med = timed(full, steps)
                h_min, h_med = timed(hit, steps)
                amort = h_med + f_med / V.KEYLITE_AUDIT_EVERY
                vtxt = "-"
                if spec is not None:
                    out = torch.zeros(P, K + 1, dtype=torch.int32)
                    vargs = (out, *spec.args, *spec.base_args)
                    v_min, v_med = timed(lambda: V.versions(vargs), steps)
                    vtxt = f"{v_min:.2f}/{v_med:.2f} ({len(vargs)} entries)"
                print(f"Q38_KEYLITE_BENCH value={value} n={n} P={P} T={T} G={G} F={F} entries={len(call) - 1} "
                      f"full_us={f_min:.2f}/{f_med:.2f} lite_hit_us={h_min:.2f}/{h_med:.2f} "
                      f"lite_amortized_us={amort:.2f} saving_us={f_med - amort:.2f} versions_us={vtxt} "
                      f"same_verdict={same}", flush=True)
    print(f"Q38_KEYLITE_BENCH_DONE ok={ok}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
