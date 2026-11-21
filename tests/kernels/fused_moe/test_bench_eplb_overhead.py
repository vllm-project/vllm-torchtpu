"""Microbenchmark: what dynamic EPLB's slot layout costs the fused EP MoE kernel.

Not a correctness test. It times `fused_ep_moe_v2` alone at the DeepSeek-V4.1
Flash shape (fp4 w4a8, hidden 5120, inter 2560 padded from 2304, 384 experts,
top-6, sqrtsoftplus + score bias, sharded plan, plan kernel on) on the
eight-way EP mesh, for layouts that differ in one thing at a time:

  base48        48 slots per shard, no slot table, no load counter (EPLB off)
  id56          56 slots per shard: the same 48 experts + 8 empty slots, mapped
                through a [384, 8] slot table (replica_cap = EP, as EPLB builds)
  id56_load     id56 + the expert_load accumulator
  blind56_load  EPLB's blind layout padded to 56 (run B8): 48 own + a replica of
                the next shard's first expert + 7 empty staging slots, + load
  blind51_load  EPLB's blind layout unpadded (run B1): 48 + 1 replica + 2 staging
                (51 slots: the routing-plan kernel is ineligible), + load

Kernel time is device time from profiler traces (the fused_ep_moe_v2 op on the
TensorCore planes). Variants are interleaved over ROUNDS rounds in one process
so drift on the node cancels in the comparison. Results print with -s and the
traces stay under $CDK_OUTPUT_DIR/eplb-bench (or a temp dir).

Run: pytest -s tests/kernels/fused_moe/test_bench_eplb_overhead.py
     (EPLB_BENCH_ROUNDS, EPLB_BENCH_ITERS, EPLB_BENCH_TOKENS, EPLB_BENCH_ROUTING
      override the defaults.)
"""

import functools
import glob
import os
import statistics
import tempfile
import time

# Serving's switches, read lazily by the kernel layer at build time.
os.environ.setdefault("MOE_FUSED_EP_V2_PLAN_KERNEL", "1")
os.environ.setdefault("MOE_FUSED_EP_V2_SHARDED_PLAN", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

EP = 8
E = 384  # logical experts
G = E // EP  # 48
HIDDEN, INTER, LOGICAL_INTER = 5120, 2560, 2304
TOPK, QB, CAPACITY = 6, 512, 128
REPLICA_CAP = EP


# ---------------------------------------------------------------------------
# Layouts (pure numpy; checked on CPU by test_layouts_are_well_formed).


def layout(kind):
    """(expert_of_slot [EP*width] with -1 for empty slots, width, slot table
    [E, REPLICA_CAP] or None)."""
    if kind == "base48":
        return np.arange(E), G, None
    extra = {"id56": (0, 8), "blind56": (1, 7), "blind51": (1, 2)}[kind]
    replicas, empty = extra
    width = G + replicas + empty
    slots = []
    for r in range(EP):
        held = list(range(r * G, (r + 1) * G))
        if replicas:
            held.append(((r + 1) % EP) * G)  # first expert of the next shard
        held += [-1] * empty
        slots += held
    expert_of_slot = np.array(slots)
    table = np.full((E, REPLICA_CAP), -1, np.int32)
    fill = np.zeros(E, np.int64)
    for s, e in enumerate(expert_of_slot):
        if e >= 0:
            table[e, fill[e]] = s
            fill[e] += 1
    return expert_of_slot, width, table


VARIANTS = {
    # name: (layout kind, with expert_load)
    "base48": ("base48", False),
    "id56": ("id56", False),
    "id56_load": ("id56", True),
    "blind56_load": ("blind56", True),
    "blind51_load": ("blind51", True),
    # base48 with packed fp4 widening (MOE_FUSED_EP_V2_FP4_PACKED_WIDEN=1 while
    # it is traced): test_bench_fp4_packed_widening, on a tree that has it.
    "base48_packed": ("base48", False),
}
DEFAULT_VARIANTS = "base48,id56,id56_load,blind56_load,blind51_load"


def test_layouts_are_well_formed():
    """Every logical expert is held at least once, empty slots are never in
    the table, and the slot counts are the ones the runs used."""
    for kind, width in (("base48", 48), ("id56", 56), ("blind56", 56), ("blind51", 51)):
        eos, w, table = layout(kind)
        assert w == width and eos.size == EP * width
        assert set(eos[eos >= 0]) == set(range(E))
        if table is None:
            continue
        assert table.shape == (E, REPLICA_CAP)
        listed = table[table >= 0]
        assert np.all(eos[listed] >= 0)  # no empty slot is mapped
        for e in range(E):
            for s in table[e][table[e] >= 0]:
                assert eos[s] == e
        reps = (table >= 0).sum(1)
        assert reps.min() == 1
        assert reps.max() == (2 if kind.startswith("blind") else 1)


# ---------------------------------------------------------------------------
# Device benchmark.


def _gating(rng, tokens, routing):
    g = rng.normal(size=(tokens, E)).astype(np.float32)
    if routing == "skewed":
        # Zipf-like popularity over a random expert order: a few hot experts,
        # as served traffic has, so per-shard load is uneven.
        pop = 1.0 / np.arange(1, E + 1) ** 1.0
        bias = np.log(pop / pop.mean())[rng.permutation(E)]
        g = g + bias[None, :].astype(np.float32)
    return g


@pytest.mark.multichip
def test_bench_eplb_layout_overhead():
    _bench(os.environ.get("EPLB_BENCH_VARIANTS", DEFAULT_VARIANTS).split(","))


def test_bench_fp4_packed_widening():
    """base48 with today's widening vs packed widening, same weights and inputs."""
    from vllm_torchtpu.kernels.fused_moe.v2 import kernel as moe_kernel

    if not hasattr(moe_kernel, "fp4_packed_words"):
        pytest.skip("this tree has no packed widening")
    _bench(["base48", "base48_packed"])


def _bench(variants):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.profiler import ProfileData
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

    if jax.device_count() < EP:
        pytest.skip("needs the eight-way EP slice")
    mesh = Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))
    shard = NamedSharding(mesh, P(AXIS))
    rounds = int(os.environ.get("EPLB_BENCH_ROUNDS", "5"))
    iters = int(os.environ.get("EPLB_BENCH_ITERS", "10"))
    token_list = [
        int(t) for t in os.environ.get("EPLB_BENCH_TOKENS", "32,1024").split(",")
    ]
    routings = os.environ.get("EPLB_BENCH_ROUTING", "uniform,skewed").split(",")
    assert "base48" in variants and set(variants) <= set(VARIANTS), variants
    out_root = os.path.join(
        os.environ.get("CDK_OUTPUT_DIR") or tempfile.mkdtemp(), "eplb-bench"
    )
    os.makedirs(out_root, exist_ok=True)
    print(
        f"\n[eplb-bench] jax {jax.__version__}, devices {jax.devices()[:EP]}, "
        f"traces -> {out_root}",
        flush=True,
    )

    def fp4_weights(n_slots, seed):
        """Random fp4 weights and block scales for n_slots, built a shard and an
        expert at a time on device (values do not matter, shapes do)."""
        g = n_slots // EP

        def local(_):
            me = lax.axis_index(AXIS)
            key = jax.random.fold_in(jax.random.key(seed), me)

            def one(k, shape):
                w = jnp.clip(jax.random.normal(k, shape, jnp.float32) * 2, -6, 6)
                return w.astype(jnp.float4_e2m1fn)

            k1, k2, k3, k4 = jax.random.split(key, 4)
            w1 = lax.map(lambda k: one(k, (HIDDEN, 2 * INTER)), jax.random.split(k1, g))
            w2 = lax.map(lambda k: one(k, (INTER, HIDDEN)), jax.random.split(k2, g))
            s1 = jax.random.uniform(
                k3, (g, HIDDEN // QB, 2 * INTER), jnp.float32, 0.01, 0.02
            )
            s2 = jax.random.uniform(
                k4, (g, INTER // QB, HIDDEN), jnp.float32, 0.01, 0.02
            )
            return w1, w2, s1, s2

        return jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(P(AXIS),),
                out_specs=(P(AXIS),) * 4,
                check_vma=False,
            )
        )(jnp.zeros((EP,), jnp.float32))

    weights = {}
    for kind in sorted({VARIANTS[v][0] for v in variants}):
        _, width, _ = layout(kind)
        n = EP * width
        if n not in weights:
            weights[n] = fp4_weights(n, seed=n)
            jax.block_until_ready(weights[n])
    print(f"[eplb-bench] weights built for slot counts {sorted(weights)}", flush=True)

    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=QB,
        act_fn="silu_and_mul_with_clamp",
        sharded_plan=True,
        scoring_fn="sqrtsoftplus",
        routed_scaling_factor=1.0,
        logical_intermediate_size=LOGICAL_INTER,
    )

    def kernel_us(trace_dir):
        """Per-core median and per-core count of fused_ep_moe_v2 op time (us)."""
        per_core = {}
        for f in glob.glob(
            os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True
        ):
            pd = ProfileData.from_file(f)
            for pl in pd.planes:
                if not pl.name.startswith("/device:TPU:") or "SparseCore" in pl.name:
                    continue
                durs = []
                for ln in pl.lines:
                    if ln.name != "XLA Ops":
                        continue
                    for e in ln.events:
                        if e.name.lstrip("%").startswith("fused_ep_moe_v2"):
                            d = next(
                                (
                                    float(v)
                                    for k, v in e.stats
                                    if k == "device_duration_ps"
                                ),
                                e.duration_ns * 1e3,
                            )
                            durs.append(d / 1e6)
                if durs:
                    per_core[pl.name] = durs
        return per_core

    results = []
    for routing in routings:
        for tokens_per_shard in token_list:
            tokens = tokens_per_shard * EP
            rng = np.random.default_rng(7)
            x = jax.device_put(
                (rng.normal(size=(tokens, HIDDEN)) / 10).astype(np.float32), shard
            ).astype(jnp.bfloat16)
            gating = jax.device_put(_gating(rng, tokens, routing), shard)
            score_bias = jnp.zeros((E,), jnp.float32)
            fns = {}
            for name in variants:
                kind, with_load = VARIANTS[name]
                eos, width, table = layout(kind)
                w1, w2, s1, s2 = weights[EP * width]
                kw = dict(common, score_bias=score_bias)
                if table is not None:
                    kw["expert_slot"] = jnp.asarray(table)
                # One jit per variant, arrays as arguments: an eager call
                # re-traces and re-compiles the layer's pieces on every call.
                if with_load:
                    load = jnp.zeros((EP * width,), jnp.int32)
                    jf = jax.jit(
                        lambda x, w1, w2, s1, s2, g, ld, kw=kw: fused_ep_moe_v2(
                            x, w1, w2, s1, s2, g, expert_load=ld, **kw
                        )
                    )
                    fns[name] = functools.partial(jf, x, w1, w2, s1, s2, gating, load)
                else:
                    jf = jax.jit(
                        lambda x, w1, w2, s1, s2, g, kw=kw: fused_ep_moe_v2(
                            x, w1, w2, s1, s2, g, **kw
                        )
                    )
                    fns[name] = functools.partial(jf, x, w1, w2, s1, s2, gating)
            for name, run in fns.items():  # compile + warm up
                # The flag is read when the layer is traced, i.e. here.
                os.environ["MOE_FUSED_EP_V2_FP4_PACKED_WIDEN"] = str(
                    int(name.endswith("_packed"))
                )
                for _ in range(3):
                    jax.block_until_ready(run())
            os.environ.pop("MOE_FUSED_EP_V2_FP4_PACKED_WIDEN")
            timing = {n: [] for n in fns}
            wall = {n: [] for n in fns}
            for r in range(rounds):
                order = list(fns) if r % 2 == 0 else list(fns)[::-1]
                for name in order:
                    d = os.path.join(
                        out_root, f"{routing}_t{tokens_per_shard}", name, f"round{r}"
                    )
                    jax.block_until_ready(fns[name]())
                    jax.profiler.start_trace(d)
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        o = fns[name]()
                    jax.block_until_ready(o)
                    wall[name].append((time.perf_counter() - t0) / iters * 1e6)
                    jax.profiler.stop_trace()
                    per_core = kernel_us(d)
                    if not per_core:
                        print(
                            f"[eplb-bench] WARNING no fused_ep_moe_v2 ops in {d}",
                            flush=True,
                        )
                        continue
                    counts = sorted({len(v) for v in per_core.values()})
                    # The slowest core's median: the layer ends when it does.
                    timing[name].append(
                        max(statistics.median(v) for v in per_core.values())
                    )
                    if r == 0:
                        print(
                            f"[eplb-bench] {routing} t{tokens_per_shard} {name}: "
                            f"{len(per_core)} cores, ops/core {counts}",
                            flush=True,
                        )
            base = (
                statistics.median(timing["base48"])
                if timing["base48"]
                else float("nan")
            )
            for name in fns:
                ts = timing[name]
                if not ts:
                    continue
                med = statistics.median(ts)
                results.append(
                    (
                        routing,
                        tokens_per_shard,
                        name,
                        med,
                        min(ts),
                        max(ts),
                        100 * (med - base) / base,
                        statistics.median(wall[name]),
                    )
                )
    print(
        "\n[eplb-bench] fused_ep_moe_v2 device time per call "
        "(slowest core's median, us); spread = min..max over rounds",
        flush=True,
    )
    print(
        f"{'routing':8s} {'tok/shard':>9s} {'variant':14s} {'median':>9s} "
        f"{'min':>9s} {'max':>9s} {'vs base48':>10s} {'wall us':>9s}",
        flush=True,
    )
    for row in results:
        print(
            f"{row[0]:8s} {row[1]:9d} {row[2]:14s} {row[3]:9.1f} "
            f"{row[4]:9.1f} {row[5]:9.1f} {row[6]:+9.1f}% {row[7]:9.1f}",
            flush=True,
        )
    assert results, "no timings collected"


@pytest.mark.multichip
def test_eplb_layouts_are_bit_exact_at_dsv41_shape():
    """EPLB layouts change only where experts run: every token's output must
    be bit-identical to the unplaced layer at the DeepSeek-V4.1 shape, for the
    buckets whose build choices the occupied-slots fix touches (32 and 1024
    tokens per shard), under even and skewed routing. Placed weights are the
    unplaced experts gathered into slot order (empty slots hold expert 0's
    copy, never read)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

    if jax.device_count() < EP:
        pytest.skip("needs the eight-way EP slice")
    mesh = Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))
    shard = NamedSharding(mesh, P(AXIS))

    def weights_for(kind, seed=384):
        """Every slot's weights generated from its GLOBAL expert id's key, on
        the shard that holds it: no gathers or collectives, and a replica is
        bit-identical to its expert by construction. Empty slots get expert 0's
        weights (never read)."""
        eos, width, table = layout(kind)
        ids_all = jnp.asarray(np.where(eos >= 0, eos, 0).reshape(EP, width), jnp.int32)

        def gen(e):
            k1, k2, k3, k4 = jax.random.split(
                jax.random.fold_in(jax.random.key(seed), e), 4
            )

            def fp4(k, shape):
                return jnp.clip(
                    jax.random.normal(k, shape, jnp.float32) * 2, -6, 6
                ).astype(jnp.float4_e2m1fn)

            return (
                fp4(k1, (HIDDEN, 2 * INTER)),
                fp4(k2, (INTER, HIDDEN)),
                jax.random.uniform(
                    k3, (HIDDEN // QB, 2 * INTER), jnp.float32, 0.01, 0.02
                ),
                jax.random.uniform(k4, (INTER // QB, HIDDEN), jnp.float32, 0.01, 0.02),
            )

        def local(ids):
            return lax.map(gen, ids[0])

        built = jax.jit(
            jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(P(AXIS),),
                out_specs=(P(AXIS),) * 4,
                check_vma=False,
            )
        )(ids_all)
        return built, (None if table is None else jnp.asarray(table))

    base, _ = weights_for("base48")
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=QB,
        act_fn="silu_and_mul_with_clamp",
        sharded_plan=True,
        scoring_fn="sqrtsoftplus",
        routed_scaling_factor=1.0,
        logical_intermediate_size=LOGICAL_INTER,
        score_bias=jnp.zeros((E,), jnp.float32),
    )
    placed = {kind: weights_for(kind) for kind in ("id56", "blind56", "blind51")}
    print(f"[eplb-exact] weights built for base48 + {sorted(placed)}", flush=True)

    unplaced = jax.jit(lambda x, g, *w: fused_ep_moe_v2(x, *w, g, **common))
    slotted = jax.jit(
        lambda x, g, t, *w: fused_ep_moe_v2(x, *w, g, expert_slot=t, **common)
    )

    failures = []
    for routing in ("uniform", "skewed"):
        for tps in (32, 1024):
            rng = np.random.default_rng(11)
            x = jax.device_put(
                (rng.normal(size=(tps * EP, HIDDEN)) / 10).astype(np.float32), shard
            ).astype(jnp.bfloat16)
            g = jax.device_put(_gating(rng, tps * EP, routing), shard)
            # Weights and tables are jit ARGUMENTS: closed over, they would
            # be embedded in the program as constants (GBs on the host).
            ref = np.asarray(unplaced(x, g, *base))
            for kind, (w, table) in placed.items():
                out = np.asarray(slotted(x, g, table, *w))
                same = np.array_equal(out.view(np.uint16), ref.view(np.uint16))
                n_diff = int((out.view(np.uint16) != ref.view(np.uint16)).sum())
                print(
                    f"[eplb-exact] {routing:7s} t{tps:<5d} {kind:8s}: "
                    + (
                        "BIT-IDENTICAL"
                        if same
                        else f"{n_diff} of {out.size} elements differ"
                    ),
                    flush=True,
                )
                if not same:
                    failures.append((routing, tps, kind, n_diff))
    assert not failures, failures
