# SPDX-License-Identifier: Apache-2.0
"""Device time of the stock decode forward's PLE programs per MK_IO_LEAN / PLE_TRIM arm.

    python tests/models/vllm/tpu_bench_ple_stock.py --part conv --tokens 4 8 16 40 56 96 128
    python tests/models/vllm/tpu_bench_ple_stock.py --part ids --tokens 4 8 16 40 56 96 128

One device. Decode/verify steps of 4 tokens per request (3 speculative tokens) with
``max(1, T // 4)`` active requests out of R request slots (R = 16 up to 64 tokens, else
32: MAX_NUM_SEQS of the c16 / c32 recipes; ``--reqs`` overrides).

* ``conv`` (jax): what ``pooled_short_conv_speculative`` computes around its Pallas pool
  reads/writes for a stock verify step: ``short_conv`` (request-owned history, the
  history-fast path) and ``token_checkpoints``. Arms by VLLM_TPU_MK_IO_LEAN: ``off``,
  ``pleconv``, ``ple``, ``all``.
* ``ids`` (torch_tpu): ``_lookup_ids`` of the stock path (``token_requests``, the real
  ``Qwen4ExpNGramEmbedding.compute_ids`` with ``lean=False``, int32 cast, valid mask).
  Arms by VLLM_TPU_MK_PLE_TRIM: ``off`` (int64 hash buffers as operands), ``trim``
  (trace-time constants).

Per arm: the sum of XLA op self time and the XLA module time per call, from a profile
(``RESULT`` lines), and whether every output is bit-identical to the ``off`` arm.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import shutil

import numpy as np

C, DIL, K, WINDOW = 4 * 2560, 3, 4, 4
VOCAB, EOS, HEADS_PER_NGRAM, NGRAM = 248320, 248044, 8, 3
BUFS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


def geometry(T, reqs, gen):
    R = reqs or (16 if T <= 64 else 32)
    active = max(1, min(R, T // 4))
    qsl = np.full(R + 1, T, np.int32)
    qsl[: active + 1] = np.minimum(np.arange(active + 1) * 4, T)
    qsl[active] = T  # the last active request takes any remainder
    return R, active, qsl


def device_times(xplane_dir, n):
    from jax.profiler import ProfileData

    files = sorted(glob.glob(xplane_dir + "/**/*.xplane.pb", recursive=True))
    if not files:
        return None
    ops, mods = collections.Counter(), collections.Counter()
    for plane in ProfileData.from_file(files[-1]).planes:
        if plane.name != "/device:TPU:0":
            continue
        for line in plane.lines:
            for ev in line.events:
                if line.name == "XLA Ops":
                    ops[ev.name.split(" ")[0]] += ev.duration_ns / n / 1e3
                elif line.name == "XLA Modules":
                    mods[ev.name.split("(")[0]] += ev.duration_ns / n / 1e3
    return ops, mods


def write_ops(path, ops, mods):
    with open(path, "w") as f:
        for k, v in ops.most_common():
            f.write(f"OPS XLA_Ops {v:.3f} {k}\n")
        for k, v in mods.most_common():
            f.write(f"OPS XLA_Modules {v:.3f} {k}\n")


def run_conv(a):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv
    from vllm_torchtpu.kernels.qwen4_exp.pool import token_checkpoints

    envs.VLLM_TPU_PLE_CONV_HISTORY_FAST = True
    state_len = (K - 1) * DIL
    gen = np.random.default_rng(0)
    summary = {}
    for T in a.tokens:
        R, active, qsl = geometry(T, a.reqs, gen)
        x = jnp.asarray(gen.standard_normal((T, C)), jnp.bfloat16)
        w = jnp.asarray(gen.standard_normal((C, K)), jnp.bfloat16)
        hist = jnp.asarray(gen.standard_normal((R, state_len, C)), jnp.bfloat16)
        ids = np.where(np.arange(R) < active, np.arange(R) + 1, 0).astype(np.int32)
        init = jnp.asarray(gen.integers(0, 2, R).astype(bool))
        ckpt = jnp.asarray((np.arange(R * WINDOW) + 7).reshape(R, WINDOW).astype(np.int32))
        windowed = jnp.ones((R,), bool)
        starts = jnp.asarray(qsl)
        args = (x, w, hist, starts, jnp.asarray(ids), init, ckpt, windowed)

        def make_step():  # a fresh function per arm: jax.jit caches traces per function
            def step(x, w, history, starts, ids, initial, ckpt_ids, windowed):
                lengths = starts[1:] - starts[:-1]
                act = (ids > 0) & (lengths > 0)
                slots = jnp.where(act, jnp.arange(R, dtype=jnp.int32), -1)
                output, _ = short_conv(
                    x, w, history, starts, slots, initial,
                    dilation=DIL, request_owned_history=True,
                )
                values, targets = token_checkpoints(
                    x, history, starts, initial, act, ckpt_ids, windowed
                )
                return output, values, targets

            return step

        res, outs = {"R": R, "active": active}, {}
        for name, lean in (("off", ""), ("pleconv", "pleconv"), ("ple", "ple"), ("all", "all")):
            envs.VLLM_TPU_MK_IO_LEAN = lean
            f = jax.jit(make_step())
            # Proves the arm took: the traced program differs where the lever acts.
            res[f"{name}_hlo"] = hashlib.sha1(f.lower(*args).as_text().encode()).hexdigest()[:10]
            outs[name] = jax.block_until_ready(f(*args))
            for _ in range(3):
                jax.block_until_ready(f(*args))
            d = f"/tmp/prof_pconv_{name}_{T}"
            shutil.rmtree(d, ignore_errors=True)
            with jax.profiler.trace(d):
                for _ in range(a.iters):
                    o = f(*args)
                jax.block_until_ready(o)
            ops, mods = device_times(d, a.iters)
            res[f"{name}_ops_us"] = round(sum(ops.values()), 2)
            res[f"{name}_module_us"] = round(sum(mods.values()), 2)
            res[f"{name}_n_ops"] = len(ops)
            write_ops(f"{a.out}/pconvops_{name}_T{T}.txt", ops, mods)
            res[f"{name}_bitwise"] = all(
                np.array_equal(np.asarray(p).view(np.uint8), np.asarray(q).view(np.uint8))
                for p, q in zip(outs["off"], outs[name], strict=True)
            )
        summary[T] = res
        print("RESULT conv", T, json.dumps(res), flush=True)
    print("SUMMARY conv", json.dumps(summary), flush=True)
    assert all(r[f"{n}_bitwise"] for r in summary.values() for n in ("pleconv", "ple", "all"))


def run_ids(a):
    import torch

    from vllm_torchtpu import envs
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import (
        Qwen4ExpNGramEmbedding,
        head_vocab_sizes,
        layer_multipliers,
        token_requests,
    )

    dev = "tpu"
    mult = layer_multipliers(VOCAB, NGRAM, 0, 1234)
    sizes = head_vocab_sizes(20_000_000, (NGRAM - 1) * HEADS_PER_NGRAM, 0)
    offs = [sum(sizes[:h]) for h in range(len(sizes))]
    consts = (mult, sizes, offs)
    bufs = [torch.tensor(x, dtype=torch.int64).to(dev) for x in consts]
    gen = np.random.default_rng(0)
    summary = {}
    for T in a.tokens:
        R, active, qsl = geometry(T, a.reqs, gen)

        class Emb(torch.nn.Module):
            """The n-gram embedding's hash state as module buffers (the served layout)."""

            compute_ids = Qwen4ExpNGramEmbedding.compute_ids

            def __init__(self):
                super().__init__()
                self.ngram_size, self.heads_per_ngram, self.eos_token_id = NGRAM, HEADS_PER_NGRAM, EOS
                self._hash_constants = consts
                for n, b in zip(BUFS, bufs, strict=True):
                    self.register_buffer(n, b)

            def forward(self, ids, qsl_t, ctx):
                req, pos = token_requests(qsl_t, T)
                out = self.compute_ids(ids, ctx, req, pos).to(torch.int32)
                valid = torch.arange(T, device=ids.device) < qsl_t[-1:]
                return torch.where(valid[:, None], out, -1)

        ids = torch.from_numpy(gen.integers(0, VOCAB, T).astype(np.int64))
        ids[0] = EOS
        ctx = torch.from_numpy(gen.integers(0, VOCAB, (R, NGRAM - 1)))
        ctx[min(1, R - 1), 0] = EOS
        args = [ids.to(dev), torch.from_numpy(qsl).to(dev), ctx.to(dev)]
        res, outs = {"R": R, "active": active}, {}
        for name, trim in (("off", False), ("trim", True)):
            envs.VLLM_TPU_MK_PLE_TRIM = trim
            torch._dynamo.reset()
            run = torch.compile(Emb(), backend="tpu", fullgraph=True, dynamic=False)
            outs[name] = run(*args).cpu()
            for _ in range(3):
                run(*args).cpu()
            from torch_tpu._internal.profiler import TpuProfilerConfig

            d = f"/tmp/prof_pids_{name}_{T}"
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                experimental_config=TpuProfilerConfig(run_dir=d),
            ):
                for _ in range(a.iters):
                    o = run(*args)
                o.cpu()
            prof = device_times(d, a.iters)
            if prof is None:
                res[f"{name}_ops_us"] = None
            else:
                ops, mods = prof
                res[f"{name}_ops_us"] = round(sum(ops.values()), 2)
                res[f"{name}_module_us"] = round(sum(mods.values()), 2)
                res[f"{name}_n_ops"] = len(ops)
                write_ops(f"{a.out}/pidsops_{name}_T{T}.txt", ops, mods)
        res["trim_bitwise"] = torch.equal(outs["off"], outs["trim"])
        summary[T] = res
        print("RESULT ids", T, json.dumps(res), flush=True)
    print("SUMMARY ids", json.dumps(summary), flush=True)
    assert all(r["trim_bitwise"] for r in summary.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("conv", "ids"), required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 8, 16, 40, 56, 96, 128])
    ap.add_argument("--reqs", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=os.environ.get("CDK_OUTPUT_DIR", "/tmp"))
    a = ap.parse_args()
    (run_conv if a.part == "conv" else run_ids)(a)


if __name__ == "__main__":
    main()
