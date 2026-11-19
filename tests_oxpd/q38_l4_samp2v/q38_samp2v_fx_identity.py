#!/usr/bin/env python3
"""q38 L4 SAMP2 compile identity (CDK CPU pod). Usage: cd <tree> && PYTHONPATH=src python3 q38_samp2_fx_identity.py <tree>

Captures today's VPS / sampling programs the way the runner compiles them (torch.compile fullgraph, dynamic=False,
with a capturing backend instead of the TPU one; flags off) and prints one digest per program of its normalized graph:
one row per node = (op, target qualname, args / kwargs with nodes as indices, dtypes / devices / constants as text);
node names and source locations are excluded. A tree whose programs run exactly the same ops with the same arguments
prints the same lines (make_fx cannot trace the fallback programs: one_hot's data-dependent checks):

  ORIGIN <module> <path> INSIDE|FALLBACK
  FXGRAPH <name> <sha16> nodes=<N>
  FXGRAPH_ALL <sha16> n=<count>

The proof requires the head lines (VLLM_TPU_VPS_INGRAPH_RNG / _TOPK_CUTOFF unset) to equal the base lines
(composed v5 tree 2e166829a): the levers' default-off path leaves every base program's graph unchanged, including
D83's read-offset variants and the VFY-ONE program. Exit 1 on a FALLBACK origin.
"""

import hashlib
import os
import sys

TREE = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else ".")
for k in [k for k in os.environ if k.startswith(("VLLM_TPU_VPS_", "VLLM_TPU_L4_", "VLLM_TPU_L6_"))]:
    del os.environ[k]

import torch  # noqa: E402

import vllm.distributed  # noqa: E402

vllm.distributed.tensor_model_parallel_all_gather = lambda x, dim=-1: x

import vllm_torchtpu.runner.tpu_runner as tr  # noqa: E402
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as tk  # noqa: E402
from vllm_torchtpu.layers.adapter.sample import vp_sampler as vp  # noqa: E402

K, TV, P = 3, 10240, 8


def origin_ok():
    ok = True
    for mod in (tr, tk, vp):
        where = os.path.realpath(mod.__file__)
        inside = where.startswith(os.path.join(TREE, "src") + os.sep)
        ok = ok and inside
        print(f"ORIGIN {mod.__name__} {where} {'INSIDE' if inside else 'FALLBACK'}")
    return ok


def problem():
    g = torch.Generator().manual_seed(1)
    N, T = P * (K + 1), P * K
    logits = (torch.randn(N, TV, generator=g) * 3).bfloat16()
    rows = torch.arange(N).view(P, K + 1)
    bidx = rows[:, K].contiguous().to(torch.int32)
    tidx = rows[:, :K].reshape(-1).contiguous().to(torch.int32)
    draft = torch.randint(0, TV, (T,), generator=g).to(torch.int32)
    dl = torch.full((P,), K, dtype=torch.int32)
    seg = torch.arange(P).repeat_interleave(K).to(torch.int32)
    grp = torch.arange(K).repeat(P).to(torch.int32)
    temps = torch.full((P, 1), 0.7, dtype=torch.bfloat16)
    top_k = torch.tensor([20, 20, 0, 5, 40, 1, 32, 20], dtype=torch.int32).unsqueeze(-1)
    top_p = torch.tensor([1.0, 0.95, 0.8, 1.0, 0.9, 1.0, 0.95, 0.8]).unsqueeze(-1)
    bu, au, ru = torch.rand(P, TV, generator=g), torch.rand(T, generator=g), torch.rand(T, TV, generator=g)
    return logits, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp


def _arg(x, idx):
    if isinstance(x, torch.fx.Node):
        return f"%{idx[x]}"
    if isinstance(x, (list, tuple)):
        return "(" + ",".join(_arg(y, idx) for y in x) + ")"
    if isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_arg(v, idx)}" for k, v in sorted(x.items())) + "}"
    if isinstance(x, slice):
        return f"slice({_arg(x.start, idx)},{_arg(x.stop, idx)},{_arg(x.step, idx)})"
    return repr(x)


def _target(t):
    if isinstance(t, str):
        return t
    return f"{getattr(t, '__module__', '')}.{getattr(t, '__qualname__', None) or getattr(t, '__name__', None) or t}"


def digest(name, fn, *tensors):
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch.compile(fn, backend=capture, fullgraph=True, dynamic=False)(*tensors)
    assert len(graphs) == 1, (name, len(graphs))
    gm = graphs[0]
    idx, rows = {}, []
    for i, nd in enumerate(gm.graph.nodes):
        idx[nd] = i
        tgt = "placeholder" if nd.op == "placeholder" else _target(nd.target)
        rows.append(f"{nd.op} {tgt} {_arg(nd.args, idx)} {_arg(dict(nd.kwargs), idx)}")
    n = sum(1 for nd in gm.graph.nodes if nd.op in ("call_function", "call_method"))
    h = hashlib.sha1("\n".join(rows).encode()).hexdigest()[:16]
    print(f"FXGRAPH {name} {h} nodes={n}")
    return name, h


def main():
    if not origin_ok():
        return 1
    lg, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp = problem()
    na = torch.tensor([P], dtype=torch.int32)
    cand_bu, cand_ru = bu[:, :32].contiguous(), ru[:, :32].contiguous()
    fb_u = torch.rand(4, TV, generator=torch.Generator().manual_seed(2))
    rows = []
    rows.append(digest("vp.local_candidates", lambda x: vp.local_candidates(x, 0), lg))
    rows.append(digest("vp._candidate_mask", lambda c, k, p: vp._candidate_mask(c, k, p),
                       (torch.randn(16, 256) * 3).bfloat16(), top_k.repeat(2, 1), top_p.repeat(2, 1)))
    rows.append(digest("tk.apply_top_k_top_p", lambda x, k, p: tk.apply_top_k_top_p(x, k, p),
                       (torch.randn(16, 256) * 3).bfloat16(), top_k.repeat(2, 1), top_p.repeat(2, 1)))
    rows.append(digest("tr._vps_verify_impl",
                       lambda *t: tr._vps_verify_impl(t[0], 0, *t[1:9], t[9], t[10], t[11], t[12], None, K, False),
                       lg, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp))
    for F, cand in ((0, False), (4, False), (4, True)):
        b_u, r_u = (cand_bu, cand_ru) if cand else (bu, ru)
        fu = fb_u if cand else None
        rows.append(digest(f"tr._vps_verify_fb_impl[F={F},cand={cand}]",
                           lambda *t, F=F, cand=cand, fu=fu: tr._vps_verify_fb_impl(
                               t[0], 0, *t[1:9], t[9], t[10], t[11], t[12], None, K, False, t[13], F, cand, fu),
                           lg, bidx, tidx, temps, top_k, top_p, b_u, au, r_u, draft, dl, seg, grp, na))
    # composed v5 tree: D83's read-offset variants and the VFY-ONE program
    groups = [torch.randint(1, 4096, (16,), generator=torch.Generator().manual_seed(3), dtype=torch.int32)
              for _ in range(3)]
    nr = torch.tensor(P)
    for F in (0, 4):
        rows.append(digest(f"tr._vps_verify_fb_impl[F={F},ro]",
                           lambda *t, F=F: tr._vps_verify_fb_impl(
                               t[0], 0, *t[1:9], t[9], t[10], t[11], t[12], None, K, False, t[13], F, False, None,
                               list(t[14:17]), t[17], torch.int32),
                           lg, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp, na, *groups, nr))
    rows.append(digest("tr._vps_verify_impl[ro]",
                       lambda *t: tr._vps_verify_impl(t[0], 0, *t[1:9], t[9], t[10], t[11], t[12], None, K, False,
                                                      list(t[13:16]), t[16], torch.int32),
                       lg, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp, *groups, nr))
    from vllm_torchtpu.runner import vps_wiring

    class _Proc:
        def _get_logits(self, hidden, head, bias, skip_gather=False):
            return torch.nn.functional.linear(hidden, head.weight)

    head = type("H", (), {})()
    head.weight = (torch.randn(TV, 64, generator=torch.Generator().manual_seed(4)) * 0.5).bfloat16()
    vh = vps_wiring.VpsHead(_Proc(), head, 0, TV, TV, 1)
    hidden = (torch.randn(lg.shape[0] + 8, 64, generator=torch.Generator().manual_seed(5))).bfloat16()
    hidx = torch.arange(lg.shape[0], dtype=torch.int32)
    acc = torch.zeros(3, dtype=torch.int64)
    for F in (0, 4):
        rows.append(digest(f"tr._vfy_one_impl[F={F}]",
                           lambda *t, F=F: tr._vfy_one_impl(
                               vh, t[0], t[1], *t[2:10], t[10], t[11], t[12], t[13], None, K, False, t[14], F, False,
                               None, list(t[15:18]), t[18], torch.int32, t[19]),
                           hidden, hidx, bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp, na,
                           *groups, nr, acc))
    rows.append(digest("tr._verify_fused_impl",
                       lambda *t: tr._verify_fused_impl(t[0], *t[1:9], t[9], t[10], t[11], t[12], None, K, False, 0),
                       lg.float()[:, :TV], bidx, tidx, temps, top_k, top_p, bu, au, ru, draft, dl, seg, grp))
    rows.sort()
    allh = hashlib.sha1(repr(rows).encode()).hexdigest()[:16]
    print(f"FXGRAPH_ALL {allh} n={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
