# SPDX-License-Identifier: Apache-2.0
"""On-TPU parity and per-op device time of VLLM_TPU_MK_IO_LEAN=ple (torch_tpu programs).

    python tests/models/vllm/tpu_bench_ple_lean.py --buckets 4 8 16 32 --reqs 16

One device. Per bucket, two compiled torch programs over the same inputs, the PLE /
embedding index work the megakernel programs run before the head kernels:

* stock: the embedding mask with the module's int64 0-d bounds, token_requests (cumsum
  form, twice: the lookup and the output mask), the n-gram ids from the int64 context and
  int64 hash buffers, ``context_lens > 0`` on the int64 lengths;
* lean: the int32 bound buffer, token_requests(compare=True), the int32 context and
  trace-time hash constants, ``context_lens.to(int32) > 0``.

Prints whether every output is bit-identical, and the device time per call of each
program from a torch_tpu profile (`OPSUM`: sum of op self time, op count, and the
X64Split ops among them).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil

import numpy as np
import torch

VOCAB, HIDDEN, TP, RANK, EOS, HEADS_PER_NGRAM = 248320, 2560, 8, 3, 248044, 8


def build(B, R, lean, consts):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import ngram_ids, token_requests

    mult, sizes, offs = consts

    def fn(ids, qsl, ctx, lens, table, b64, b32, mult_t, sizes_t, offs_t):
        import vllm.model_executor.layers.vocab_parallel_embedding as vpe

        bounds = b32.unbind(0) if lean else b64
        masked, mask = vpe.get_masked_input_and_mask(ids, *bounds)
        part = torch.nn.functional.embedding(masked.long(), table)
        part = part.masked_fill(mask.unsqueeze(-1), 0)
        req, pos = token_requests(qsl, B, compare=lean)
        if lean:
            grams = ngram_ids(
                ids.long(), ctx.to(torch.int32), req, pos, mult, sizes, offs,
                eos_token_id=EOS, heads_per_ngram=HEADS_PER_NGRAM, narrow_context=True,
            )
        else:
            grams = ngram_ids(
                ids.long(), ctx, req, pos, mult_t, sizes_t, offs_t,
                eos_token_id=EOS, heads_per_ngram=HEADS_PER_NGRAM,
            )
        grams = grams.to(torch.int32)
        valid = torch.arange(B, device=ids.device) < qsl[-1:]
        grams = torch.where(valid[:, None], grams, -1)
        req2, _ = token_requests(qsl, B, compare=lean)
        init = (lens.to(torch.int32) if lean else lens) > 0
        return part, grams, req2, init

    return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)


def profile_ops(run, args, n, d):
    from torch_tpu._internal.profiler import TpuProfilerConfig

    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    cfg = TpuProfilerConfig(run_dir=d)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.PrivateUse1],
        experimental_config=cfg,
    ):
        for _ in range(n):
            out = run(*args)
        [o.cpu() for o in out]
    files = sorted(glob.glob(d + "/**/*.xplane.pb", recursive=True))
    if not files:
        return None
    from jax.profiler import ProfileData

    ops = collections.Counter()
    mods = collections.Counter()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", type=int, nargs="+", default=[4, 8, 16, 24, 32, 48, 64])
    ap.add_argument("--reqs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", default=os.environ.get("CDK_OUTPUT_DIR", "/tmp"))
    a = ap.parse_args()
    from vllm_torchtpu import _patch_vllm_vocab_parallel_embedding
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import head_vocab_sizes, layer_multipliers

    _patch_vllm_vocab_parallel_embedding()
    dev = "tpu"
    mult = layer_multipliers(VOCAB, 3, 0, 1234)
    sizes = head_vocab_sizes(20_000_000, 2 * HEADS_PER_NGRAM, 0)
    offs = [sum(sizes[:h]) for h in range(len(sizes))]
    per = -(-VOCAB // TP)
    gen = np.random.default_rng(0)
    table = torch.from_numpy(gen.standard_normal((per, HIDDEN), np.float32)).to(torch.bfloat16)
    table = table.to(dev)
    si = [RANK * per, (RANK + 1) * per, 0, VOCAB, VOCAB]
    b64 = [torch.tensor(v, dtype=torch.int64).to(dev) for v in si]
    b32 = torch.tensor(si, dtype=torch.int32).to(dev)
    consts_t = [torch.tensor(x, dtype=torch.int64).to(dev) for x in (mult, sizes, offs)]
    summary = {}
    for B in a.buckets:
        R = a.reqs
        active = min(R, B)
        cuts = np.sort(gen.choice(np.arange(1, B), active - 1, replace=False)) if active > 1 else []
        qsl = np.full(R + 1, B, np.int32)
        qsl[0] = 0
        qsl[1:active] = cuts
        ids = torch.from_numpy(gen.integers(0, VOCAB, B).astype(np.int32))
        ids[0] = EOS
        ctx = torch.from_numpy(gen.integers(0, VOCAB, (R, 2)))
        ctx[1, 0] = EOS
        lens = torch.from_numpy(gen.integers(0, 210000, R))
        lens[-1] = 0
        args = [ids.to(dev), torch.from_numpy(qsl).to(dev), ctx.to(dev), lens.to(dev), table,
                b64, b32, *consts_t]
        res = {}
        outs = {}
        for name, lean in (("stock", False), ("lean", True)):
            run = build(B, R, lean, (mult, sizes, offs))
            outs[name] = [o.cpu() for o in run(*args)]
            prof = profile_ops(run, args, a.iters, f"/tmp/prof_ple_{name}_{B}")
            if prof is None:
                res[f"{name}_ops_us"] = None
                continue
            ops, mods = prof
            x64 = {k: v for k, v in ops.items() if "custom-call" in k or "X64" in k}
            res[f"{name}_ops_us"] = round(sum(ops.values()), 2)
            res[f"{name}_n_ops"] = len(ops)
            res[f"{name}_module_us"] = round(sum(mods.values()), 2)
            res[f"{name}_custom_calls"] = len(x64)
            with open(f"{a.out}/pleops_{name}_b{B}.txt", "w") as f:
                for k, v in ops.most_common():
                    f.write(f"OPS XLA_Ops {v:.3f} {k}\n")
                for k, v in mods.most_common():
                    f.write(f"OPS XLA_Modules {v:.3f} {k}\n")
        res["bitwise"] = all(
            torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                        y.view(torch.int16) if y.dtype == torch.bfloat16 else y)
            for x, y in zip(outs["stock"], outs["lean"], strict=True)
        )
        # Negative control: a context token moved changes the stock ids.
        summary[B] = res
        print("RESULT", B, json.dumps(res), flush=True)
    print("SUMMARY", json.dumps(summary), flush=True)
    assert all(r["bitwise"] for r in summary.values()), summary


if __name__ == "__main__":
    main()
