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
"""TPU microbench for TPU_HOST_KEEP_INPUT_MUTATIONS.

A decode-step stand-in with the MiniMax cache-write pattern: `layers` layers,
each writing a kv cache and an index cache through a functional op followed by
`cache.copy_(new)` (as models/minimax_m3/ops/msa.py does), compiled with
torch.compile(backend="tpu"). For one mode (0 = torch_tpu default, 1 = keep
mutations in graph, 2 = keep + donate the written placeholders) it reports:
  - correct: caches and outputs after `steps` calls bitwise equal to eager TPU;
  - epilogue_copies: host `Tensor.copy_` calls per compiled call (AOT epilogue);
  - host_us: host time per call (no sync), wall_us: per call with the device.
usage: benchmark_keep_input_mutations.py --mode {0,1,2} --variant {index,jaxop}
"""

import argparse
import json
import time
import traceback

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--variant", default="jaxop", choices=["index", "jaxop"])
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--slots", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    res = {"mode": a.mode, "variant": a.variant, "layers": a.layers}
    try:
        import torch_tpu  # noqa: F401

        from vllm_torchtpu.runner import host_path

        host_path.install_keep_input_mutations(a.mode)
        dev = torch.device("tpu")

        if a.variant == "jaxop":
            import jax.numpy as jnp
            from torch_tpu._internal import pallas

            def _write(q, cache, slots):
                new = cache.at[slots].set(q.astype(cache.dtype))
                return new, jnp.tanh(q + new[slots].astype(q.dtype))

            op = pallas.jax_op(
                "host_path_bench::cache_write", _write, donate_argnums=(1,)
            )
            op.register_fake(
                lambda q, cache, slots: (torch.empty_like(cache), torch.empty_like(q))
            )

            def write(q, cache, slots):
                new, out = op(q, cache, slots)
                cache.copy_(new)
                return out
        else:

            def write(q, cache, slots):
                new = cache.index_copy(0, slots, q.to(cache.dtype))
                out = torch.tanh(q + new[slots].to(q.dtype))
                cache.copy_(new)
                return out

        def step(x, slots, kv, idx):
            for i in range(len(kv)):
                x = write(x, kv[i], slots)
                x = x + write(x * 0.5, idx[i], slots)
            return x

        compiled = torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)
        g = torch.Generator().manual_seed(0)
        mk = lambda: [  # noqa: E731
            torch.randn(a.slots, a.dim, generator=g).to(torch.bfloat16)
            for _ in range(a.layers)
        ]
        kv0, idx0 = mk(), mk()
        xs = [torch.randn(a.rows, a.dim, generator=g) for _ in range(a.steps)]
        sl = [torch.randperm(a.slots, generator=g)[: a.rows] for _ in range(a.steps)]

        def run(fn):
            kv = [t.to(dev) for t in kv0]
            idx = [t.to(dev) for t in idx0]
            outs = [fn(x.to(dev), s.to(dev), kv, idx).cpu() for x, s in zip(xs, sl)]
            return outs, [t.cpu() for t in kv], [t.cpu() for t in idx]

        ref = run(step)
        got = run(compiled)
        res["correct"] = bool(
            all(torch.equal(r, o) for r, o in zip(ref[0], got[0]))
            and all(torch.equal(r, o) for r, o in zip(ref[1], got[1]))
            and all(torch.equal(r, o) for r, o in zip(ref[2], got[2]))
        )
        res["max_abs_out"] = max(float((r - o).abs().max()) for r, o in zip(ref[0], got[0]))
        res["caches_equal"] = sum(
            torch.equal(r, o) for r, o in zip(ref[1] + ref[2], got[1] + got[2])
        )

        kv = [t.to(dev) for t in kv0]
        idx = [t.to(dev) for t in idx0]
        x, s = xs[0].to(dev), sl[0].to(dev)
        for _ in range(10):
            compiled(x, s, kv, idx)
        torch.tpu.synchronize()
        n = [0]
        orig = torch.Tensor.copy_

        def counted(self, *args, **kw):
            n[0] += 1
            return orig(self, *args, **kw)

        torch.Tensor.copy_ = counted
        try:
            compiled(x, s, kv, idx)
        finally:
            torch.Tensor.copy_ = orig
        res["epilogue_copies"] = n[0]
        torch.tpu.synchronize()
        host = 0.0
        t0 = time.perf_counter()
        for _ in range(a.iters):
            h = time.perf_counter()
            compiled(x, s, kv, idx)
            host += time.perf_counter() - h
        torch.tpu.synchronize()
        res["host_us"] = round(host / a.iters * 1e6, 1)
        res["wall_us"] = round((time.perf_counter() - t0) / a.iters * 1e6, 1)
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"[:600]
        traceback.print_exc()
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
