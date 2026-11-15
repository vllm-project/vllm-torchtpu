# SPDX-License-Identifier: Apache-2.0
"""TPU timing of shard-local top-k variants for VLLM_TPU_VERIFY_SHARD_TOPK
(standalone; run on a TPU pod, one core).

Each variant computes one LM-head shard's logits (hidden @ W.T, bf16) and their
k largest values and ids per row, compiled as one program that repeats the work
REPEAT times on perturbed inputs; the reported time is wall time per repeat
after warm-up, so dispatch cost is amortised. Results are checked against
torch.topk on the host.

usage: python tests/runner/vocab_shard_topk_tpu_bench.py [--rows 128 256] [--k 32]
"""

import argparse
import contextlib
import time

import torch

with contextlib.suppress(ImportError):  # registers the "tpu" device and backend
    import torch_tpu  # noqa: F401

REPEAT = 10
HIDDEN = 2560
SHARD = 31040


def _two_stage_rowmajor(h, w, k, block):
    """Today's shard_candidates stage: logits [R, n], blocks along the vocab."""
    logits = h @ w.T
    rows, n = logits.shape
    pad = (-n) % block
    if pad:
        logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
    blocks = logits.reshape(rows, -1, block)
    bid = torch.topk(blocks.amax(-1), k, dim=-1, sorted=False).indices
    g = torch.gather(blocks, 1, bid.unsqueeze(-1).expand(-1, -1, block)).reshape(
        rows, k * block
    )
    v, pos = torch.topk(g, k, dim=-1, sorted=False)
    ids = torch.gather(bid, 1, pos // block) * block + pos % block
    return v.float(), ids


def _two_stage_vocab_major(h, w, k, block):
    """Logits as [n, R] (vocab major, rows minor, the dot's natural layout):
    block maxima over the vocab axis, then a per-column gather."""
    lt = w @ h.T
    n, rows = lt.shape
    bmax = lt.reshape(n // block, block, rows).amax(1)
    bid = torch.topk(bmax.T, k, dim=-1, sorted=False).indices  # [R, k]
    cols = (bid.unsqueeze(-1) * block + torch.arange(block, device=h.device)).reshape(
        rows, k * block
    )
    g = torch.gather(lt, 0, cols.T).T  # [R, k*block]
    v, pos = torch.topk(g, k, dim=-1, sorted=False)
    return v.float(), torch.gather(cols, 1, pos)


def _two_stage_flat(h, w, k, block):
    """As _two_stage_vocab_major, gathering from the flattened [n, R] buffer."""
    lt = w @ h.T
    n, rows = lt.shape
    bmax = lt.reshape(n // block, block, rows).amax(1)
    bid = torch.topk(bmax.T, k, dim=-1, sorted=False).indices
    cols = (bid.unsqueeze(-1) * block + torch.arange(block, device=h.device)).reshape(
        rows, k * block
    )
    flat = cols * rows + torch.arange(rows, device=h.device).unsqueeze(-1)
    g = lt.reshape(-1)[flat]
    v, pos = torch.topk(g, k, dim=-1, sorted=False)
    return v.float(), torch.gather(cols, 1, pos)


def _direct(h, w, k, block):
    v, ids = torch.topk(h @ w.T, k, dim=-1, sorted=False)
    return v.float(), ids


def _direct_f32(h, w, k, block):
    v, ids = torch.topk((h @ w.T).float(), k, dim=-1, sorted=False)
    return v, ids


def _vocab_major_direct(h, w, k, block):
    """torch.topk over the transposed [R, n] view of the [n, R] logits."""
    lt = w @ h.T
    v, ids = torch.topk(lt.T, k, dim=-1, sorted=False)
    return v.float(), ids


VARIANTS = {
    "rowmajor_b64": (_two_stage_rowmajor, 64),
    "vocab_major_b64": (_two_stage_vocab_major, 64),
    "vocab_major_b16": (_two_stage_vocab_major, 16),
    "flat_b64": (_two_stage_flat, 64),
    "direct_topk": (_direct, 0),
    "direct_topk_f32": (_direct_f32, 0),
    "vocab_major_direct": (_vocab_major_direct, 0),
    "vocab_major_b32": (_two_stage_vocab_major, 32),
    "dot_only": (None, 0),
}


def _repeated(fn, block, k):
    def body(h, w):
        acc = torch.zeros((), dtype=torch.float32, device=h.device)
        out = None
        for i in range(REPEAT):
            hi = h + torch.tensor(i * 1e-3, dtype=h.dtype, device=h.device)
            if fn is None:
                out = (hi @ w.T).float()
                acc = acc + out.amax()
            else:
                v, ids = fn(hi, w, k, block)
                out = (v, ids)
                acc = acc + v.sum() + ids.sum().float()
        return acc, out

    return torch.compile(body, backend="tpu", fullgraph=True, dynamic=False)


def main():
    # One body per variant and row count: lift dynamo's per-code-object limit.
    for name, value in (
        ("cache_size_limit", 1024),
        ("recompile_limit", 1024),
        ("accumulated_cache_size_limit", 4096),
        ("accumulated_recompile_limit", 4096),
    ):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, value)
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[128, 256])
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()
    dev = torch.device("tpu")
    gen = torch.Generator().manual_seed(0)
    w_cpu = (torch.randn(SHARD, HIDDEN, generator=gen) * 0.02).to(torch.bfloat16)
    w = w_cpu.to(dev)
    for rows in a.rows:
        h_cpu = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16)
        h = h_cpu.to(dev)
        hl = h + torch.tensor((REPEAT - 1) * 1e-3, dtype=h.dtype, device=dev)
        ref = torch.compile(
            lambda x, y: torch.topk(x @ y.T, a.k, dim=-1).values.float(),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )(hl, w).cpu()
        for name, (fn, block) in VARIANTS.items():
            try:
                prog = _repeated(fn, block, a.k)
                acc, out = prog(h, w)
                acc.cpu()
                start = time.perf_counter()
                for _ in range(a.iters):
                    acc, out = prog(h, w)
                acc.cpu()
                per = (time.perf_counter() - start) / a.iters / REPEAT * 1e6
                ok = ""
                if fn is not None:
                    # `out` is the last repeat's, whose input is `hl`.
                    want = torch.sort(ref, dim=-1, descending=True).values
                    got = torch.sort(out[0].cpu(), dim=-1, descending=True).values
                    ok = (
                        "match"
                        if torch.equal(got, want)
                        else (f"MISMATCH max|d|={float((got - want).abs().max()):.3g}")
                    )
                print(
                    f"VSBENCH rows={rows} variant={name} us_per_repeat={per:.1f} {ok}",
                    flush=True,
                )
            except Exception as e:  # report and continue with the next variant
                print(
                    f"VSBENCH rows={rows} variant={name} ERROR {type(e).__name__}: "
                    f"{str(e)[:300]}",
                    flush=True,
                )
        del ref


if __name__ == "__main__":
    main()
