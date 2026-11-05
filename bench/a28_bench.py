"""A28 microbench, served shape (vocab 248320 / TP4, 16 requests, 64 verify rows).

    python3 bench/a28_bench.py jax    # shard candidates (1 core); candidate program
    python3 bench/a28_bench.py torch  # served torch non-greedy verify (1 core)

Wall time per call after warm-up (median of 20 calls, synchronized), so dispatch
is included.
"""

import functools
import os
import statistics
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "tests", "spec_decode"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

MODE = sys.argv[1] if len(sys.argv) > 1 else "jax"


def timeit(fn, n=20):
    fn()
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return 1e6 * statistics.median(t)


def bench_jax():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import test_candidate_sampling_tpu as t
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.layers.core import candidate_sampling as cs

    x = t._inputs(0, "peaked", 0.7, 20, 0.8)
    rows = jnp.concatenate([x["logits"][x["bonus_idx"]], x["logits"]])[:, : t.SHARD]
    bounds = jnp.asarray([0, t.SHARD], jnp.int32)
    sel = jax.jit(functools.partial(cs.shard_candidates, kc=t.KC))
    us = timeit(lambda: jax.block_until_ready(sel(rows, bounds)))
    print(f"BENCH shard_candidates [{rows.shape[0]},{t.SHARD}] kc={t.KC}: {us:.1f} us")

    mesh = Mesh(np.asarray(t._devices()), ("tp",))
    seg = jnp.asarray(x["segment"])
    top_k = t._top_k(x, pad=1)
    width = t.KC * t.TP

    def shards(r):
        return r.reshape(r.shape[0], t.TP, t.SHARD).transpose(1, 0, 2)

    replicated = [
        x["draft"],
        x["temps_b"],
        top_k,
        x["top_p_b"],
        x["temps_b"][seg],
        top_k[seg],
        x["top_p_b"][seg],
        jnp.full((t.NUM_REQS, width), 0.5, jnp.bfloat16),
        x["accept_u"],
        jnp.full((t.ROWS, width), 0.5, jnp.float32),
        x["accept_u"],
        jnp.zeros((2,), jnp.int32),
    ]
    bounds = jnp.asarray(
        [[r * t.SHARD, (r + 1) * t.SHARD] for r in range(t.TP)], jnp.int32
    )
    operands = [
        shards(x["logits"][x["bonus_idx"]]),
        shards(x["logits"][x["target_idx"]]),
        bounds,
    ] + [jnp.broadcast_to(a[None], (t.TP, *a.shape)) for a in replicated]
    for force_full in (False, True):
        local = functools.partial(
            cs.verify_local,
            axis="tp",
            kc=t.KC,
            vocab_size=t.VOCAB,
            synthetic=False,
            force_full=force_full,
        )
        fn = jax.jit(
            jax.shard_map(
                lambda *a, f=local: tuple(o[None] for o in f(*(v[0] for v in a))),
                mesh=mesh,
                in_specs=(P("tp"),) * 15,
                out_specs=(P("tp"),) * 6,
                check_vma=False,
            )
        )
        us = timeit(lambda f=fn: jax.block_until_ready(f(*operands)))
        print(f"BENCH candidate_verify force_full={force_full}: {us:.1f} us")


def bench_torch():
    import types

    import torch
    import torch_tpu  # noqa: F401

    from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    V, B, T, K = 248320, 16, 64, 3
    dev = "tpu"
    logits = (torch.randn(T, V) * 3).to(torch.bfloat16).to(dev)
    temps = torch.full((B, 1), 0.7, dtype=torch.bfloat16).to(dev)
    top_k = torch.full((B, 1), 20, dtype=torch.int32).to(dev)
    top_p = torch.full((B, 1), 0.8, dtype=torch.float32).to(dev)
    bidx = torch.arange(B).to(dev) * 4 % T
    tidx = torch.arange(T).to(dev)
    seg = (torch.arange(T) // K).clamp(max=B - 1).to(dev)
    group = (torch.arange(T) % K).to(torch.int32).to(dev)
    draft = torch.randint(0, V, (T,), dtype=torch.int32).to(dev)
    nd = torch.full((B,), K, dtype=torch.int32).to(dev)
    runner = types.SimpleNamespace(
        _apply_temperature=functools.partial(TPUModelRunner._apply_temperature, None)
    )
    g = torch.Generator(device=dev)
    g.manual_seed(0)

    def step():
        bonus_logits = torch.index_select(logits, 0, bidx)
        target_logits = torch.index_select(logits, 0, tidx)
        u = torch.rand_like(bonus_logits, generator=g)
        bonus = TPUModelRunner.sample_from_logits(
            runner, bonus_logits, temps, u, top_k, top_p, all_greedy=False
        ).view(-1)
        acc_u = torch.rand((T,), dtype=torch.float32, device=dev, generator=g)
        rec_u = torch.rand_like(target_logits, dtype=torch.float32, generator=g)
        out = rs._random_rejection_sample_with_segment(
            draft, target_logits, nd, bonus, seg, group, K, temps[seg],
            top_k[seg], top_p[seg], acc_u, rec_u, None, False,
        )  # fmt: skip
        out.cpu()

    us = timeit(step)
    print(f"BENCH served torch verify [{T},{V}] + bonus [{B},{V}]: {us:.1f} us")


{"jax": bench_jax, "torch": bench_torch}[MODE]()
