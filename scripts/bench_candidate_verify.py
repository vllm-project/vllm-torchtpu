# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Spec-verify sampler microbench on one TPU host (TP 8, one rank per core).

Times the logits, the uniform draw(s) and the verify (+ the mamba read-offset
scatter) of one non-greedy verify chunk at the served Qwen3.8 geometry
(vocabulary 248320 over 8 shards, hidden 2560, 3 drafts per request) for
three arms, each through the runner's own methods on a stub runner:

  B         gathered logits (local LM-head GEMM + all-gather), one flat draw
            and the fused verify with the read-offset scatter in-graph
            (VERIFY_FUSED + ONE_DRAW + RO_INGRAPH + LOGITS_BLOCKS 2 +
            top-k candidates 64, candidates only)
  B+CS      local logits shard, candidate verify as separate programs
            (VLLM_TPU_CANDIDATE_SAMPLING), separate read-offset scatter
  B+CS+P    local logits shard, candidate verify as one program with the
            read-offset scatter (VLLM_TPU_CANDIDATE_VERIFY_PROGRAM + _RO)

Requests are served-like sampled: temperature in [0.6, 1.0], top_k in
[1, 48], top_p in [0.8, 1.0]. The padded sampling params are prebuilt, so
their host cost is in no arm. Per part: `lat` is the median wall time of one
synchronized call, `thr` the wall time per call of back-to-back calls.

With --dist-iters N it also samples one chunk N times per arm and compares
the arms' histograms (chi-square homogeneity per request): accepted length,
the bonus token when every draft was accepted, and the recovered token when
the first draft was rejected.

Usage (on the TPU host): python scripts/bench_candidate_verify.py
[--rows 4,8,16,32] [--iters 200] [--dist-iters 4000] [--out results.json]
--dry runs the stub-runner plumbing on CPU in one process (no TPU, op
stubbed, compiled regions eager).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

WORLD = 8
VOCAB = 248320
HIDDEN = 2560
K = 3
RO_GROUPS = 37  # mamba KV groups, one [capacity] slot tensor each
RO_CAPACITY = 32
ARMS = ("B", "B+CS", "B+CS+P")
BASE_ENV = {
    "VLLM_TPU_VERIFY_FUSED": "1",
    "VLLM_TPU_VERIFY_ONE_DRAW": "1",
    "VLLM_TPU_VERIFY_RO_INGRAPH": "1",
    "VLLM_TPU_VERIFY_FUSED_READ_OFFSETS": "1",
    "VLLM_TPU_SAMPLER_LOGITS_BLOCKS": "2",
    "VLLM_TPU_SAMPLER_TOPP_NARROW": "1",
    "VLLM_TPU_SAMPLER_CANDIDATES_ONLY": "1",
    "TPU_SAMPLER_TOPK_CANDIDATES": "64",
    "VLLM_TPU_FUSED_READ_OFFSET_UPDATE": "1",
}


def _set_arm(arm):
    program = "1" if arm == "B+CS+P" else "0"
    os.environ["VLLM_TPU_CANDIDATE_VERIFY_PROGRAM"] = program
    os.environ["VLLM_TPU_CANDIDATE_VERIFY_RO"] = program


class Bench:
    def __init__(self, rank, tp, device, dry):
        import torch

        from vllm_torchtpu.runner import tpu_runner as tr

        self.torch, self.tr = torch, tr
        self.rank, self.tp, self.device, self.dry = rank, tp, device, dry
        self.shard = VOCAB // tp
        # This rank's LM-head shard (the full head is never materialized).
        g = torch.Generator().manual_seed(1000 + rank)
        w = torch.randn(self.shard, HIDDEN, generator=g) * (3.0 / HIDDEN**0.5)
        self.w_local = w.to(torch.bfloat16).to(device)
        self.read_offsets = torch.zeros(4096, dtype=torch.int32).to(device)
        self.groups = [
            torch.arange(i * RO_CAPACITY, (i + 1) * RO_CAPACITY, dtype=torch.int32).to(
                device
            )
            for i in range(RO_GROUPS)
        ]
        self.runner = self._runner()
        self._build_logits_fns()

    def _build_logits_fns(self):
        torch, tr = self.torch, self.tr
        w = self.w_local
        tp, dry = self.tp, self.dry

        def local(hidden, indices):
            selected = torch.index_select(hidden, 0, indices)
            return torch.nn.functional.linear(selected, w)

        def gathered(hidden, indices):
            part = local(hidden, indices)
            if dry:
                full = part.repeat(1, tp)
            else:
                from vllm.distributed.parallel_state import get_world_group

                full = get_world_group().all_gather(part, dim=-1)
            return full[:, :VOCAB]

        self.local_logits = tr._tpu_compile(local)
        self.gathered_logits = tr._tpu_compile(gathered)

    def _runner(self):
        torch, tr = self.torch, self.tr
        R = tr.TPUModelRunner
        r = types.SimpleNamespace()
        r.device = self.device
        r.rejection_sampler = types.SimpleNamespace(
            synthetic_mode=False, synthetic_conditional_rates=None
        )
        r.speculative_config = types.SimpleNamespace(num_speculative_tokens=K)
        r._candidate_bounds = torch.tensor(
            [self.rank * self.shard, (self.rank + 1) * self.shard], dtype=torch.int32
        ).to(self.device)
        r._candidate_head = types.SimpleNamespace(
            tp_size=self.tp,
            org_vocab_size=VOCAB,
            num_embeddings_per_partition=self.shard,
        )
        r.mamba_slot_read_offsets = self.read_offsets
        r._fused_read_offsets = (None, 0)
        r._fused_read_offsets_applied = False
        r._verify_fused_checks_left = 0
        r._cvp_compiling = False
        r._params = None
        r._padded_sampling_params = lambda *a, **kw: r._params
        r._sampler_topk_candidates = lambda *a: 64
        r._sampler_candidates_only = lambda tc: True
        r._sampler_logits_blocks = lambda: 2
        r._ro_ingraph_warmed_for = lambda *a: True
        for name in (
            "_num_reqs_tensor",
            "_sample_spec_verify_fused",
            "_sample_spec_verify_candidates",
            "_candidate_verify_one_program",
            "_candidate_ro_fold",
            "spec_gather_bonus_and_target_logits",
        ):
            setattr(r, name, types.MethodType(getattr(R, name), r))
        return r

    def chunk(self, num_reqs, seed):
        """One verify chunk: hidden states, metadata, sampling params, drafts
        (argmax for even requests, a top-5 token for odd ones)."""
        import numpy as np

        torch, tr = self.torch, self.tr
        g = torch.Generator().manual_seed(seed)
        rows = num_reqs * (K + 1)
        hidden = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
        reqs = torch.arange(num_reqs, dtype=torch.int32)
        bonus_idx = reqs * (K + 1) + K
        target_idx = (reqs[:, None] * (K + 1) + torch.arange(K)).view(-1).int()
        temps = 0.6 + 0.4 * torch.rand(num_reqs, 1, generator=g)
        top_k = torch.randint(1, 49, (num_reqs, 1), generator=g, dtype=torch.int32)
        top_p = 0.8 + 0.2 * torch.rand(num_reqs, 1, generator=g)
        # Drafts from the gathered logits (the same on every rank).
        dev = self.device
        hidden_d = hidden.to(dev)
        all_idx = torch.arange(rows, dtype=torch.int32).to(dev)
        full = self.gathered_logits(hidden_d, all_idx).float().cpu()
        order = full.topk(5, dim=-1).indices
        pick = torch.where(
            (torch.arange(rows) // (K + 1)) % 2 == 0,
            torch.zeros(rows, dtype=torch.long),
            torch.randint(1, 5, (rows,), generator=g),
        )
        draft = order[torch.arange(rows), pick].int()[target_idx.long()]
        md = tr.SpecDecodeMetadata(
            draft_token_ids=draft.to(dev),
            draft_lengths=torch.full((num_reqs,), K, dtype=torch.int32).to(dev),
            draft_lengths_cpu=np.full(num_reqs, K, dtype=np.int32),
            target_logits_indices=target_idx.to(dev),
            bonus_logits_indices=bonus_idx.to(dev),
            final_logits_indices=target_idx.to(dev),
            segment_ids=torch.repeat_interleave(torch.arange(num_reqs), K).to(dev),
            group_indices=torch.arange(K, dtype=torch.int32).repeat(num_reqs).to(dev),
        )
        params = (
            temps.to(torch.bfloat16).to(dev),
            top_k.to(dev),
            top_p.float().to(dev),
        )
        return types.SimpleNamespace(
            hidden=hidden_d,
            indices=all_idx,
            md=md,
            params=params,
            draft=draft,
            num_reqs=num_reqs,
            temps=temps,
            top_k=top_k,
            top_p=top_p,
        )

    # --- the three arms' parts -------------------------------------------

    def logits(self, arm, c):
        if arm == "B":
            return self.gathered_logits(c.hidden, c.indices)
        return self.local_logits(c.hidden, c.indices)

    def verify(self, arm, c, logits):
        """Draw(s) + verify + read-offset scatter of one chunk."""
        tr, r = self.tr, self.runner
        r._params = c.params
        groups = self.groups
        r._fused_read_offsets = (groups, c.num_reqs)
        r._fused_read_offsets_applied = False
        if arm == "B":
            out = r._sample_spec_verify_fused(logits, c.md, 0, c.num_reqs, None)
        else:
            out = r._sample_spec_verify_candidates(logits, c.md, 0, c.num_reqs, None)
        if not r._fused_read_offsets_applied:
            tr._update_read_offsets_compiled(
                self.read_offsets, list(groups), out, r._num_reqs_tensor(c.num_reqs)
            )
        return out

    def uniforms(self, arm, c):
        """The arm's uniform draw(s) alone, at its shapes."""
        torch, tr = self.torch, self.tr
        n_req, dev = c.num_reqs, self.device
        n_t = n_req * K
        f32 = torch.float32
        if arm == "B":
            n = tr._verify_uniforms_count(n_req, VOCAB, (n_t,), n_t)
            return torch.rand((n,), dtype=f32, device=dev)
        width = 64 * self.tp
        if arm == "B+CS+P":
            n = tr._candidate_uniforms_count(n_req, n_t, n_t, width)
            return torch.rand((n,), dtype=f32, device=dev)
        return (
            torch.rand((n_req, width), dtype=torch.bfloat16, device=dev),
            torch.rand((n_t,), dtype=f32, device=dev),
            torch.rand((n_t, width), dtype=f32, device=dev),
            torch.rand(2, device=dev),
        )

    def warm(self, arm, c):
        """Compile the arm's programs at this chunk's key (the runner's
        warmup does the same for the one-program arm)."""
        r = self.runner
        r._cvp_compiling = arm == "B+CS+P"
        try:
            self.sync(self.verify(arm, c, self.logits(arm, c)))
        finally:
            r._cvp_compiling = False
        for _ in range(3):
            self.sync(self.verify(arm, c, self.logits(arm, c)))
            self.sync(self.uniforms(arm, c))

    def sync(self, out):
        self.tr.synchronize_tensors(out)

    def time_part(self, fn, iters):
        lat = []
        for _ in range(iters):
            t = time.perf_counter()
            self.sync(fn())
            lat.append(time.perf_counter() - t)
        t = time.perf_counter()
        out = None
        for _ in range(iters):
            out = fn()
        self.sync(out)
        thr = (time.perf_counter() - t) / iters
        return {"lat": statistics.median(lat) * 1e6, "thr": thr * 1e6}

    def time_arm(self, arm, c, iters):
        _set_arm(arm)
        self.warm(arm, c)
        logits = self.logits(arm, c)
        return {
            "logits": self.time_part(lambda: self.logits(arm, c), iters),
            "uniform": self.time_part(lambda: self.uniforms(arm, c), iters),
            "verify": self.time_part(lambda: self.verify(arm, c, logits), iters),
            "step": self.time_part(
                lambda: self.verify(arm, c, self.logits(arm, c)), iters
            ),
        }

    def time_rows(self, num_reqs, iters):
        c = self.chunk(num_reqs, seed=num_reqs)
        return {arm: self.time_arm(arm, c, iters) for arm in ARMS}

    def sample_dist(self, num_reqs, iters):
        """Per arm, `iters` [num_reqs, 1 + K] outputs of one fixed chunk."""
        torch = self.torch
        c = self.chunk(num_reqs, seed=777)
        outs = {}
        for arm in ARMS:
            _set_arm(arm)
            self.warm(arm, c)
            logits = self.logits(arm, c)
            got = [self.verify(arm, c, logits) for _ in range(iters)]
            outs[arm] = torch.stack(got).cpu()
        return c, outs


def _decompose(draft, out):
    """[iters, reqs, 1 + K] outputs → accepted length, the token after the
    accepted drafts (recovered or bonus)."""
    iters, reqs, _ = out.shape
    d = draft.view(1, reqs, K)
    accepted = (out[:, :, :K] == d).int().cumprod(dim=2).sum(dim=2)
    nxt = out.gather(2, accepted.unsqueeze(2).long()).squeeze(2)
    return accepted, nxt


def _chi2_p(a, b, min_expected=5.0):
    import numpy as np
    from scipy.stats import chi2_contingency

    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 20 or len(b) < 20:
        return None
    cats = np.union1d(a, b)
    table = np.stack([[np.sum(a == v) for v in cats], [np.sum(b == v) for v in cats]])
    expected = table.sum(0) * min(table.sum(1)) / table.sum()
    keep = expected >= min_expected
    table = np.concatenate(
        [table[:, keep], table[:, ~keep].sum(1, keepdims=True)], axis=1
    )
    table = table[:, table.sum(0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(table)[1])


def dist_report(c, outs):
    """Per request and path, chi-square p of each arm against B+CS+P."""
    rep = {}
    acc_p, nxt = {}, {}
    for arm, out in outs.items():
        acc_p[arm], nxt[arm] = _decompose(c.draft, out.int())
    for ref in ("B", "B+CS"):
        rows = []
        for q in range(c.num_reqs):
            a1, a2 = acc_p[ref][:, q].numpy(), acc_p["B+CS+P"][:, q].numpy()
            n1, n2 = nxt[ref][:, q].numpy(), nxt["B+CS+P"][:, q].numpy()
            rows.append(
                {
                    "req": q,
                    "temp": round(float(c.temps[q]), 3),
                    "top_k": int(c.top_k[q]),
                    "top_p": round(float(c.top_p[q]), 3),
                    "accept_rate_ref": float((a1 == K).mean()),
                    "accept_rate_port": float((a2 == K).mean()),
                    "p_accepted": _chi2_p(a1, a2),
                    "p_bonus": _chi2_p(n1[a1 == K], n2[a2 == K]),
                    "p_recovered": _chi2_p(n1[a1 == 0], n2[a2 == 0]),
                }
            )
        ps = {
            k: [r[k] for r in rows if r[k] is not None]
            for k in ("p_accepted", "p_bonus", "p_recovered")
        }
        rep[f"{ref}_vs_B+CS+P"] = {
            "per_request": rows,
            "summary": {
                k: {
                    "n": len(v),
                    "min": min(v) if v else None,
                    "below_0.01": sum(p < 0.01 for p in v),
                }
                for k, v in ps.items()
            },
        }
    return rep


def run(rank, tp, device, dry, args):
    bench = Bench(rank, tp, device, dry)
    rows = [int(x) for x in args.rows.split(",")]
    result = {"timing": {}, "env": BASE_ENV}
    for n in rows:
        result["timing"][n] = bench.time_rows(n, args.iters)
        if rank == 0:
            for arm, parts in result["timing"][n].items():
                print(
                    f"rows {n:2d} {arm:7s} "
                    + " ".join(
                        f"{p} lat {v['lat']:.0f} thr {v['thr']:.0f}"
                        for p, v in parts.items()
                    ),
                    flush=True,
                )
    if args.dist_iters:
        c, outs = bench.sample_dist(args.dist_rows, args.dist_iters)
        if rank == 0 and args.out:
            bench.torch.save(
                {"outs": outs, "draft": c.draft, "temps": c.temps,
                 "top_k": c.top_k, "top_p": c.top_p},
                args.out + ".dist.pt",
            )  # fmt: skip
        try:
            import scipy  # noqa: F401
        except ImportError:
            print("DIST: no scipy here; histograms saved to --out .dist.pt")
            return result
        if rank == 0:
            result["distribution"] = dist_report(c, outs)
            for k, v in result["distribution"].items():
                print("DIST", k, json.dumps(v["summary"]), flush=True)
    return result


def _worker(args):
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    dist.init_process_group(backend="tpu_dist")
    torch.empty((1,), device="tpu").cpu()
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_distributed_environment,
    )

    init_distributed_environment(
        world_size=WORLD,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    from torch_tpu._internal.pallas import get_pallas_mesh

    from vllm_torchtpu.layers.adapter.sample import candidate_sampling as csa

    mesh = get_pallas_mesh(
        axis_names=(csa.TP_AXIS,),
        mesh_shape=(WORLD,),
        process_group=get_world_group().device_group,
    )
    csa.get_tp_mesh = lambda: mesh  # the TP group here is the world group
    try:
        result = run(rank, WORLD, torch.device("tpu"), False, args)
        if rank == 0 and args.out:
            Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def _dry(args):
    """CPU plumbing check: compiled regions eager, candidate op stubbed."""
    import torch

    from vllm_torchtpu.runner import tpu_runner as tr

    def eager(obj):
        inner = getattr(obj, "__wrapped__", obj)
        return getattr(inner, "_torchdynamo_orig_callable", None)

    tr._tpu_compile = lambda fn: fn
    tr._update_read_offsets_compiled = tr._update_read_offsets_impl
    tr.candidate_rejection_output = lambda acc, *a: tr.assemble_rejection_output(
        acc != 0, *a
    )
    tr.TPUModelRunner.spec_gather_bonus_and_target_logits = lambda self, x, b, t: (
        x.index_select(0, b),
        x.index_select(0, t),
    )
    for space in (tr, tr.TPUModelRunner):
        for name, obj in list(vars(space).items()):
            if callable(obj) and eager(obj) is not None:
                setattr(space, name, eager(obj))

    def fake_op(vocab, synthetic):
        def op(*operands):
            b, t = operands[0].shape[1], operands[1].shape[1]
            z = torch.zeros
            return (
                z(1, b, dtype=torch.int32),
                z(1, t, dtype=torch.int32),
                z(1, t, dtype=torch.int32),
                z(1, dtype=torch.int32),
            )

        return op

    tr.candidate_verify_op = fake_op
    tr._candidate_verify_program.cache_clear()

    def fake_decisions(bonus, target, *a, **kw):
        z = torch.zeros
        return (
            z(bonus.shape[0], dtype=torch.int32),
            z(target.shape[0], dtype=torch.int32),
            z(target.shape[0], dtype=torch.int32),
            z((), dtype=torch.int32),
        )

    tr.candidate_verify_decisions = fake_decisions
    tr._on_tpu = lambda t: True
    global HIDDEN
    HIDDEN = 64
    res = run(0, 8, torch.device("cpu"), True, args)
    print(json.dumps(res["timing"], default=str)[:400])
    print("DRY OK")


def _prepare_tpu_environment(world_size):
    """Single-host launch env for `world_size` ranks (as
    tests/distributed/tpu_test_utils.py, without its pytest import)."""
    import torch
    import torch_tpu  # noqa: F401

    tpu_dist = getattr(getattr(torch, "tpu", None), "distributed", None)
    if tpu_dist is not None and hasattr(tpu_dist, "set_tpu_launch_env"):
        tpu_dist.set_tpu_launch_env(nproc_per_node=world_size)
        return
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )

    prepare_tpu_environment(world_size=world_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="4,8,16,32")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--dist-iters", type=int, default=0)
    p.add_argument("--dist-rows", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    os.environ.update(BASE_ENV)
    if args.dry:
        return _dry(args)
    if args.worker:
        return _worker(args)
    _prepare_tpu_environment(WORLD)
    env = os.environ.copy()
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD}",
        str(Path(__file__).resolve()),
        "--worker",
        *sys.argv[1:],
    ]
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
