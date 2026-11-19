# SPDX-License-Identifier: Apache-2.0
"""q38 L6-MK CPU microbench (CDK CPU pod only): host µs of the stock vs lever paths at N = 4 / 9 / 11 /
25 / 52 running requests, with an identity check of every arm's outputs.

* mkctx: Qwen38SegmentDriver.try_forward per call (fake programs; the driver's own host work only).
* mkthin: _SplitCompiledExecutable call per call at 8 / 64 / 300 / 500 arguments with a constant fast
  runner (torch_tpu's Python layers 11-14 only); per call, so the same at every N; the per-step row
  uses 9 helper calls at 16 args + 3 program calls at 400 args (D98 census: 12 per decode step).
* fuserpc: engine step wall time over the real RPC rig (2 forked workers, KV aggregator), split vs fused,
  payload sized by N.
Last line: BENCH_DONE identity=PASS|FAIL.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_mk_harness as H  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.runner import l6_mk  # noqa: E402

NS_ = (4, 9, 11, 25, 52)
OK = [True]


def _env(**kv):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            del os.environ[k]
    os.environ.update(kv)
    l6_ab._resolve()
    l6_mk._reset_for_tests()


def _med(f, reps=5):
    return statistics.median(f() for _ in range(reps))


def bench_mkctx():
    import numpy as np
    from types import SimpleNamespace as NS

    from vllm_torchtpu.megakernel_runtime import driver as base_drv
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv

    buckets = (8, 16, 24, 32, 48, 64, 96, 128, 256)
    for n in NS_:
        rows = next(b for b in buckets if b >= n * drv.T_VERIFY)
        res = {}
        for arm, env in (("off", {}), ("on", {"VLLM_TPU_L6_MK_CTX": "1"})):
            _env(**env)
            rec = []
            d = H.make_driver(drv, rec)
            d.buckets, d.whole_max_rows = buckets, 96
            base_drv.MegakernelDriver._inspect_step_detail = staticmethod(lambda *a, n=n: (True, n, drv.T_VERIFY, ""))
            runner = NS(input_batch=NS(num_computed_tokens_cpu=np.full(64, 900, np.int32)), warmup_done=True,
                        vllm_config=NS(label="runner_cfg"))
            ids = torch.zeros(rows, dtype=torch.int32)
            pos = torch.zeros(rows, dtype=torch.int32)
            for _ in range(200):
                l6_ab.begin_step()
                d.try_forward(runner, ids, pos, None, None, {})

            def run(k=2000):
                t0 = time.perf_counter_ns()
                for _ in range(k):
                    l6_ab.begin_step()
                    d.try_forward(runner, ids, pos, None, None, {})
                return (time.perf_counter_ns() - t0) / k / 1000

            res[arm] = (_med(run), sorted(d.stats.items()), rec[-1] if rec else None)
        same = res["off"][1] == res["on"][1] and res["off"][2] == res["on"][2]
        OK[0] &= same
        route = "whole" if rows <= 96 else "stock (above the whole-step bucket)"
        print(f"BENCH mkctx N={n} rows={rows} route={route} off_us={res['off'][0]:.2f} on_us={res['on'][0]:.2f} "
              f"delta_us={res['on'][0] - res['off'][0]:+.2f} identity={'ok' if same else 'MISMATCH'}")


def _thin_exe(k):
    from torch_tpu._internal.compile import split_compiler as sc

    log = H.RunnerLog()
    src = ", ".join(f"a{i}" for i in range(k))
    ns: dict = {"torch": torch}
    exec(f"def f({src}):\n    return (torch.stack([{src}]).sum(0),)\n", ns)
    exe = H.build_split_exe(ns["f"], log, f"k{k}", 1, False)
    wrapper = [m for m in exe._split_gm.children() if isinstance(m, sc._WrapperModule)][0]
    const = (torch.zeros(1),)
    wrapper.submod.__dict__["_fast_runner"] = lambda args: const
    return exe


def bench_mkthin():
    per = {}
    for k in (8, 16, 64, 300, 400, 500):
        args = [torch.zeros(1) for _ in range(k)]
        exe = _thin_exe(k)  # one executable, both arms
        res = {}
        for arm, env in (("stock", {}), ("thin", {"VLLM_TPU_L6_MK_THIN_TT": "1"})):
            _env(**env)
            if arm == "thin":
                assert l6_mk.install_mkthin()
            l6_ab.begin_step()
            out0 = exe(args)

            def run(m=4000):
                t0 = time.perf_counter_ns()
                for _ in range(m):
                    exe(args)
                return (time.perf_counter_ns() - t0) / m / 1000

            res[arm] = (_med(run), out0)
            if arm == "thin":
                assert l6_ab._state["counts"].get("mkthin.thin", 0) > 0, "thin path not engaged"
                l6_mk.uninstall_mkthin_for_tests()
        same = type(res["stock"][1]) is type(res["thin"][1]) and len(res["stock"][1]) == len(res["thin"][1]) and all(
            a is b for a, b in zip(res["stock"][1], res["thin"][1]))
        OK[0] &= same
        per[k] = (res["stock"][0], res["thin"][0])
        print(f"BENCH mkthin args={k} stock_us={res['stock'][0]:.2f} thin_us={res['thin'][0]:.2f} "
              f"delta_us={res['thin'][0] - res['stock'][0]:+.2f} identity={'ok' if same else 'MISMATCH'}")
    step = 9 * (per[16][1] - per[16][0]) + 3 * (per[400][1] - per[400][0])
    for n in NS_:
        print(f"BENCH mkthin N={n} per_step_delta_us={step:+.2f} (9 calls x 16 args + 3 x 400 args; per call, N-independent)")


def _rpc_arms(name, world, arms, tpu_async=False, delay_ms=0.0, steps=300):
    """One spawned rig per arm, all N on it (the fake worker sizes its output by the scheduled requests)."""
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    from vllm_torchtpu.core import l6_fuse_rpc as F

    MultiprocExecutor.execute_and_sample = F.executor_execute_and_sample
    res: dict = {}
    for arm, fn, env in arms:
        _env(**env)
        F._KV.update(check=None, turn=0)
        with tempfile.TemporaryDirectory() as td:
            rig = H.RpcRig(td, world=world, kv_agg=True, n_req=4, tpu_async=tpu_async, delay_ms=delay_ms,
                           mode="spawn")
            try:
                for n in NS_:
                    H.run_rpc_stream(rig, fn, [{}] * 30, n_req=n)  # warm
                    times, tr = [], None
                    for _ in range(3):
                        t0 = time.perf_counter_ns()
                        tr = H.run_rpc_stream(rig, fn, [{}] * steps, n_req=n)
                        times.append((time.perf_counter_ns() - t0) / steps / 1000)
                    res[(arm, n)] = (statistics.median(times), tr["updates"])
            finally:
                rig.close()
    (a0, _, _), (a1, _, _) = arms
    for n in NS_:
        same = res[(a0, n)][1] == res[(a1, n)][1]
        OK[0] &= same
        print(f"BENCH {name} N={n} world={world} {a0}_us_per_step={res[(a0, n)][0]:.1f} "
              f"{a1}_us_per_step={res[(a1, n)][0]:.1f} delta_us={res[(a1, n)][0] - res[(a0, n)][0]:+.1f} "
              f"identity={'ok' if same else 'MISMATCH'}")


def bench_fuserpc():
    from vllm.v1.engine.core import EngineCore

    from vllm_torchtpu.core import l6_fuse_rpc as F

    vstep = EngineCore.__dict__["step_with_batch_queue"]
    _rpc_arms("fuserpc", 2, (("split", vstep, {}), ("fused", F.step_with_batch_queue, {"VLLM_TPU_L6_FUSE_RPC": "1"})))


def bench_kvonly():
    """8 ranks, TPU async outputs with rank-staggered copy-ready times (0.2 ms x (1 + rank))."""
    from vllm_torchtpu.core import l6_fuse_rpc as F

    _rpc_arms("e2kvonly", 8, (("fused", F.step_with_batch_queue, {"VLLM_TPU_L6_FUSE_RPC": "1"}),
                              ("kvonly", F.step_with_batch_queue,
                               {"VLLM_TPU_L6_FUSE_RPC": "1", "VLLM_TPU_L6_OUT_KVONLY": "1"})),
              tpu_async=True, delay_ms=0.2, steps=200)


if __name__ == "__main__":
    print(f"BENCH host nproc={os.cpu_count()} torch={torch.__version__}")
    bench_mkctx()
    bench_mkthin()
    bench_fuserpc()
    bench_kvonly()
    print(f"BENCH_DONE identity={'PASS' if OK[0] else 'FAIL'}")
