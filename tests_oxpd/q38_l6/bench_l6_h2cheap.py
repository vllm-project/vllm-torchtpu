#!/usr/bin/env python3
"""q38 L6-H2 (overflow unit l6_h2_cheap) CPU microbench, CDK CPU pod only.
Usage (from the head tree, PYTHONPATH=src:tests/q38_l6): python3 bench_l6_h2cheap.py

B1 toklists: the per-step token-row host work at N = 4 / 9 / 11 / 25 / 52 running requests (row
   buckets 16 / 16 / 16 / 32 / 64, 1 + K = 4 columns, acceptance like AL 2.35): stock = get_output's
   tensor path + the main thread's tensor path (two passes over the same rows); lists = one tolist
   + both list filters (the real get_output / l6_main_rows). Median us per step; outputs compared.
B2 fskip: the stream harness (tests/q38_l6/test_l6_fskip.py) at N = 4 / 9 / 11 / 25 / 52 with CARRY +
   STEADY + fskip on: skipped / recorded fences on a steady and a churn stream, and the host cost of
   the record-site rule (skip()) per call. The fence record itself (torch.tpu.Event().record(), 51 us
   in IX's lap timers) does not exist on CPU, so its saving is not measured here.
B3 aotskip: us per call of a real AOT artifact through vLLM's aot_compile with the guard check on and
   off (1 and 30 tensor arguments), and the per-step on_step() cost (no change / a flip, 3 artifacts).
Last line: BENCH_H2C_DONE identity=PASS|FAIL
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time
from types import SimpleNamespace as NS

import torch

ROWS = {4: 16, 9: 16, 11: 16, 25: 32, 52: 64}
FAILS = []


def med_us(fn, reps=2000, warm=200):
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        fn()
        xs.append(time.perf_counter_ns() - t)
    return statistics.median(xs) / 1000.0


def bench_toklists():
    from vllm_torchtpu.runner import tpu_runner_async_output as tao
    from vllm_torchtpu.runner.tpu_runner_async_output import (
        INVALID_TOKEN_ID,
        AsyncTPUCopyState,
        AsyncTPUModelRunnerOutput,
    )

    class Ev:
        def synchronize(self):
            pass

    for n, rows in ROWS.items():
        g = torch.Generator().manual_seed(n)
        t = torch.randint(0, 150000, (rows, 4), generator=g, dtype=torch.int32)
        acc = torch.clamp(torch.poisson(torch.full((rows,), 1.35), generator=g).long() + 1, 1, 4)
        for i in range(rows):
            t[i, acc[i]:] = INVALID_TOKEN_ID

        def state(lists):
            return AsyncTPUCopyState(chunks_cpu=[t], chunk_real_lens=[n], copy_ready_event=Ev(), l6_lists=lists)

        def stock():
            st = state(False)
            out = AsyncTPUModelRunnerOutput(NS(sampled_token_ids=None, logprobs=None), st, []).get_output()
            main, _ = tao.tensor_valid_rows(st.sampled_token_ids_cpu[:n])
            return out.sampled_token_ids, main

        def lists():
            st = state(True)
            out = AsyncTPUModelRunnerOutput(NS(sampled_token_ids=None, logprobs=None), st, []).get_output()
            main = tao.l6_valid_rows(st.sampled_token_ids_list[:n], 4)
            return out.sampled_token_ids, main

        same = stock() == lists()
        a, b = med_us(stock), med_us(lists)
        print(f"BENCH toklists N={n} rows={rows} us_per_step stock={a:.2f} lists={b:.2f} saved={a - b:.2f} equal={same}")
        if not same:
            FAILS.append(f"toklists N={n}")


def bench_fskip():
    import test_l6_fskip as T
    from vllm_torchtpu.core import l6_ab
    from vllm_torchtpu.runner import l6_fskip
    from vllm_torchtpu.runner import tpu_runner
    import l6_carry_harness as H

    torch.tpu = NS(Event=T._Event)
    sub = tpu_runner._substitute_placeholder_token
    eager = getattr(sub, "_torchdynamo_orig_callable", None) or getattr(sub, "__wrapped__", None)
    if eager is not None:
        tpu_runner._substitute_placeholder_token = eager
    H.install_eager_programs()
    os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"
    for k, v in T.ON.items():
        os.environ[k] = v
    l6_ab._resolve()
    for n in ROWS:
        shape = dict(nreq=n, max_num_reqs=64, paddings=(32, 64, 128, 256), mamba=True, comb=True)
        T.SHAPES["bench"] = shape
        out = []
        for sched in ("steady", "churn_a"):
            l6_fskip.S.n, l6_fskip.S.carried = 0, False
            _, sim = T.run_stream("bench", sched, steps=40)
            bad = [f for f in sim.fences if f["skipped"] and (f["total"] or f["staging"])]
            if bad:
                FAILS.append(f"fskip N={n} {sched}")
            sk = sum(f["skipped"] for f in sim.fences)
            out.append(f"{sched} skipped={sk}/{len(sim.fences)} bad={len(bad)}")
        print(f"BENCH fskip N={n} " + " ".join(out))
    l6_ab.begin_step()

    def rule_on():
        l6_fskip.S.carried = True
        l6_fskip.skip()

    def mark():
        l6_fskip.S.n += 1

    print(f"BENCH fskip rule us_per_call skip()={med_us(rule_on, reps=20000):.3f} mark={med_us(mark, reps=20000):.3f}")


def bench_aotskip():
    import test_l6_aotskip as A
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W
    from vllm_torchtpu.compilation import l6_aotskip as la
    from vllm_torchtpu.core import l6_ab

    def f30(*xs):
        acc = xs[0] * 0
        for i, t in enumerate(xs):
            acc = acc + t * (i + 1)
        return acc

    os.environ["VLLM_TPU_L6_AOT_SKIP"] = "1"
    l6_ab._resolve()
    la.reset()
    la.install()
    for nargs, fn_py in ((1, A.f1), (30, f30)):
        xs = tuple(torch.randn(8) for _ in range(nargs))
        fn = A._compile(W, fn_py, xs[0]) if nargs == 1 else None
        if fn is None:
            torch._dynamo.reset()
            with torch._dynamo.config.patch(enable_aot_compile=True):
                c = torch.compile(fn_py, fullgraph=True, dynamic=False, backend=A.serializable_eager)
            owner = NS(_compiled_callable=c, compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=False)))
            fn = W.aot_compile(owner, *xs)
        res = {}
        for st in (True, False):
            fn._guard_check_enabled = st
            res[st] = med_us(lambda: fn(*xs), reps=4000)
        print(f"BENCH aotskip args={nargs} us_per_call guard_on={res[True]:.2f} guard_off={res[False]:.2f} "
              f"saved={res[True] - res[False]:.2f}")
    l6_ab.begin_step()
    la.on_step()
    print(f"BENCH aotskip on_step us handles={len(la._handles)} steady={med_us(la.on_step, reps=20000):.3f}")

    def flip():
        la._state["applied"] = not la._state["applied"]
        la.on_step()

    print(f"BENCH aotskip on_step us handles={len(la._handles)} flip={med_us(flip, reps=20000):.3f}")


def main():
    for name, fn in (("toklists", bench_toklists), ("fskip", bench_fskip), ("aotskip", bench_aotskip)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            FAILS.append(f"{name} exception {e!r}"[:200])
    print(f"BENCH_H2C_DONE identity={'PASS' if not FAILS else 'FAIL'} {' '.join(FAILS)}".rstrip())
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
