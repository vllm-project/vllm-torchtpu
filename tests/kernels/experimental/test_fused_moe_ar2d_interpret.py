# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Gate G1 of the AR2D exchange (FUSED_MOE_DECODE_KERNEL_AR2D_MIN_TOKENS) in the
Pallas interpreter on 8 host devices: the exchange's output equals the NumPy
model of its summation order (ar2d_reference.py) bit for bit on all eight ranks,
fed the per-rank f32 accumulators that enter it.

- The exchange alone (`_ar2d_exchange` in a minimal kernel) on chosen
  accumulators, including ones that cancel across chips (256, 1, -256), where
  the model, the kernel and today's all-to-all result are recorded side by side.
- The whole fused MoE decode kernel, folded as the server builds it, at every
  row bucket the flag turns on (dense, and routed at two buckets), its f32
  accumulators read back through the test-only `ar2d_partials` output.

Needs 8 host devices: JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_decode import (
    fused_moe_decode_kernel as kmod,
)
from vllm_torchtpu.kernels.experimental.fused_moe_decode.ar2d_reference import (
    BF16,
    ar2d_mismatches,
    ar2d_reference,
    round_bf16,
)

N, H = 8, 2560
E, INTER, INTER_S, TOPK = 2, 128, 128, 2
F8 = jnp.float8_e4m3fn
# Served row buckets the flag turns on at its proposed value 128 (multiples of 64).
AR2D_MIN = 128
BUCKETS = (128, 192, 256, 320, 384, 448, 512)


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _exchange_only(mesh, partials):
    """Outputs [8, m, H] of _ar2d_exchange alone on per-rank f32 partials."""
    m = partials.shape[1]
    scratch = [
        pltpu.VMEM((m, H), jnp.bfloat16),
        pltpu.VMEM((m, H // 2), jnp.bfloat16),
        pltpu.VMEM((m, H // 2), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((len(kmod.AR2D_STREAMS),)),
        pltpu.SemaphoreType.DMA((len(kmod.AR2D_STREAMS),)),
        *kmod._ar2d_scratch(m, H, jnp.bfloat16),
    ]

    def body(acc_ref, out_ref, a, s1_recv, af, ssem, rsem, *bufs):
        me = lax.axis_index("d").astype(jnp.int32)
        barrier = pltpu.get_barrier_semaphore()
        for d in range(N):
            pl.semaphore_signal(
                barrier, 1, device_id=(jnp.int32(d),), device_id_type=pl.DeviceIdType.MESH
            )
        kmod._ar2d_exchange(
            me, barrier, N, acc_ref, a, s1_recv, af, out_ref, ssem, rsem, bufs, m, H,
            None, None,
        )

    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((m, H), jnp.bfloat16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(),
    )
    f = jax.jit(
        jax.shard_map(
            lambda x: call(x[0])[None],
            mesh=mesh,
            in_specs=(P("d"),),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    x = jax.device_put(jnp.asarray(partials, jnp.float32), NamedSharding(mesh, P("d")))
    return np.asarray(f(x))


def _all_to_all_result(partials):
    """Today's exchange: bf16 partials summed in f32 in rank order, one rounding."""
    xs = round_bf16(partials).astype(np.float32)
    acc = xs[0]
    for r in range(1, N):
        acc = acc + xs[r]
    return acc.astype(BF16)


def test_exchange_random_partials(mesh):
    rng = np.random.default_rng(1)
    partials = rng.normal(size=(N, 64, H)).astype(np.float32)
    out = _exchange_only(mesh, partials)
    res = ar2d_mismatches(out, partials)
    assert res["mismatched"] == 0 and res["ranks_identical"], res


def test_exchange_cancelling_partials(mesh):
    """256, 1, -256 on ranks 0, 2, 4 (three chips), zeros elsewhere, in one
    column of each part of each half. Today's sum is 1 everywhere. The model
    gives 1 in part 0 (256 - 256 cancels first); in part 1 it gives 1 on the
    rows whose rank keeps 257 in f32 and 0 on the rows where 257 is sent as
    bf16 (256). The kernel must give the model's bits."""
    m = 64
    partials = np.zeros((N, m, H), np.float32)
    cols = [0, 640, 1280, 1920]  # part 0 / part 1 of half 0, then of half 1
    for c in cols:
        partials[0, :, c] = 256.0
        partials[2, :, c] = 1.0
        partials[4, :, c] = -256.0
    out = _exchange_only(mesh, partials)
    model = ar2d_reference(partials).astype(np.float32)
    today = _all_to_all_result(partials)
    record = {
        c: (
            float(today[0, c]),
            sorted({float(v) for v in model[:, c]}),
            [int((out[:, :, c].astype(np.float32) == v).sum()) for v in (0.0, 1.0)],
        )
        for c in cols
    }
    print("AR2D_CANCEL column: (all-to-all, model values, kernel count of 0 / 1)", record)
    u = (np.arange(m) % (m // 2)) // (m // 4)
    assert all(float(today[0, c]) == 1.0 for c in cols)
    for c in (0, 1280):
        assert (model[:, c] == 1.0).all()
        assert (model[:, c + 640] == np.where(u == 0, 1.0, 0.0)).all()
    res = ar2d_mismatches(out, partials)
    assert res["mismatched"] == 0 and res["ranks_identical"], res


def _q(w, axis):
    s = np.max(np.abs(w), axis=axis, keepdims=True) / 448.0
    return jnp.asarray(w / s).astype(F8), s.astype(np.float32)


def _moe_inputs(tokens, seed):
    rng = np.random.default_rng(seed)
    w1, w1_s = _q(rng.normal(size=(N, E, H, 2 * INTER)) * 0.02, 2)
    w2, w2_s = _q(rng.normal(size=(N, E, INTER, H)) * 0.02, 2)
    sw1 = jnp.asarray(rng.normal(size=(N, H, 2 * INTER_S)) * 0.02, jnp.bfloat16)
    sw2 = jnp.asarray(rng.normal(size=(N, INTER_S, H)) * 0.02, jnp.bfloat16)
    gate = jnp.asarray(rng.normal(size=(1, H)) * 0.02, jnp.bfloat16)
    x = jnp.asarray(rng.normal(size=(tokens, H)), jnp.bfloat16)
    ids = np.stack([rng.choice(N * E, TOPK, replace=False) for _ in range(tokens)])
    tw = jnp.asarray(rng.dirichlet(np.ones(TOPK), size=tokens), jnp.bfloat16)
    return x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, jnp.asarray(ids, jnp.int32)


def _moe(mesh, tokens, ar2d_min, routed, partials, seed=0, exact_width=False):
    x, w1, w1_s, w2, w2_s, sw1, sw2, gate, tw, ids = _moe_inputs(tokens, seed)
    sh, rep = NamedSharding(mesh, P("d")), NamedSharding(mesh, P())
    rank = np.stack([np.arange(N), np.arange(N) * E], 1).astype(np.int32)

    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate, tw, ids):
        out = kmod.fused_moe_decode(
            rank,
            x,
            w1[0],
            w2[0],
            w1_s[0][:, :, None, :],
            w2_s[0][:, :, None, :],
            sw1[0],
            jnp.ones((2 * INTER_S,), jnp.float32),
            sw2[0],
            jnp.ones((H,), jnp.float32),
            gate,
            tw,
            ids,
            n_ranks=N,
            expert_start="rank",
            fold=True,
            routed=routed,
            interpret=True,
            ar2d_min_tokens=ar2d_min,
            ar2d_partials=partials,
            exact_width=exact_width,
        )
        return tuple(o[None] for o in out) if partials else out[None]

    args = [
        jax.device_put(rank, sh),
        *(jax.device_put(a, sh) for a in (w1, w2, w1_s, w2_s, sw1, sw2)),
        *(jax.device_put(a, rep) for a in (x, gate, tw, ids)),
    ]
    out = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P("d"),) * 7 + (P(),) * 4,
            out_specs=(P("d"), P("d")) if partials else P("d"),
            check_vma=False,
        )
    )(*args)
    return jax.tree.map(np.asarray, out)


@pytest.mark.parametrize(
    "tokens,routed,exact_width",
    [(b, None, False) for b in BUCKETS]
    + [(256, (8, 64), False), (512, (8, 64), False), (120, None, False)]
    + [(128, None, True), (512, (8, 64), True)],
)
def test_moe_kernel_matches_model(mesh, monkeypatch, tokens, routed, exact_width):
    """The folded kernel with the flag: output equal to the model of its own
    partials on all 8 ranks. 120 tokens pad to the 128-row bucket. With
    FUSED_MOE_DECODE_EXACT_WIDTH too, the output is [m, H] and the same."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    out, partials = _moe(
        mesh, tokens, AR2D_MIN, routed, True, exact_width=exact_width
    )
    m = tokens + (-tokens) % 16
    assert out.shape == (N, tokens, H) and partials.shape == (N, m, H)
    res = ar2d_mismatches(out, partials)
    print(f"AR2D_G1 tokens {tokens} routed {routed} exact {exact_width}: {res}")
    assert res["mismatched"] == 0 and res["ranks_identical"], res
    # The flag-off call on the same inputs: today's exchange of the same partials.
    off = _moe(mesh, tokens, 0, routed, False, exact_width=exact_width)
    today = _all_to_all_result(partials)[:tokens]
    assert (off.view(np.uint16) == today.view(np.uint16)[None]).all()


def test_flag_off_and_refused_shapes_keep_the_all_to_all(monkeypatch):
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    assert kmod.ar2d_applies(128, H, N, 128)
    assert not kmod.ar2d_applies(128, H, N, 0)  # flag off
    assert not kmod.ar2d_applies(64, H, N, 128)  # below the flag
    assert not kmod.ar2d_applies(160, H, N, 128)  # 160 / 4 is not whole tiles
    assert not kmod.ar2d_applies(128, H, 16, 128)  # another rank count
    assert not kmod.ar2d_applies(128, 1536, N, 128)  # another hidden width
    assert not kmod.ar2d_applies(128, H, N, 128, scatter_rows=True)
    assert not kmod.ar2d_applies(128, H, N, 128, whole_experts=False)
    for m in BUCKETS:
        on = kmod.vmem_bytes(m, H, INTER, INTER_S, E, N, F8, ar2d_min_tokens=AR2D_MIN)
        off = kmod.vmem_bytes(m, H, INTER, INTER_S, E, N, F8)
        assert on - off == m * (14720 - 13056)  # design: exchange bytes a row


@pytest.mark.parametrize("tokens", [64, 160])
@pytest.mark.parametrize("exact_width", [False, True])
def test_unselected_buckets_are_the_all_to_all_call(
    mesh, monkeypatch, tokens, exact_width
):
    """Below the flag, or a row count that is not a multiple of 64, the flag
    leaves the call as it is: same bits as flag off."""
    monkeypatch.setattr(kmod, "_vmem_capacity", lambda: 64 << 20)
    on = _moe(mesh, tokens, AR2D_MIN, None, False, exact_width=exact_width)
    off = _moe(mesh, tokens, 0, None, False, exact_width=exact_width)
    assert (on.view(np.uint16) == off.view(np.uint16)).all()

