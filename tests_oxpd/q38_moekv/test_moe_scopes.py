# SPDX-License-Identifier: Apache-2.0
"""q38 IQD #10 measurement: fused_moe_decode's in-kernel named scopes (``scopes``).

Traces the kernel at O1's served per-rank shapes (EP8, 64 local experts, hidden 2560, inter 640, fold on, TP all-reduce)
with abstract shapes only (nothing allocated or run) and checks:
  * scopes=0 is the program of a call without the argument: identical jaxpr text and identical kernel name stacks;
  * scopes=0 puts none of the region names in the kernel's name stacks;
  * scopes=1 puts all six region names in them (moe_prologue, moe_shared, moe_experts, ar_barrier_wait, ar_a2a,
    ar_gather) and leaves the printed jaxpr unchanged (the same ops; only the names differ).

Needs 8 host devices: JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 (skipped otherwise).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.experimental.fused_moe_decode import fused_moe_decode_kernel as kmod  # noqa: E402

N, E, H, INTER, ISP = 8, 64, 2560, 640, 128
REGIONS = ("moe_prologue", "moe_shared", "moe_experts", "ar_barrier_wait", "ar_a2a", "ar_gather")
S = jax.ShapeDtypeStruct


@pytest.fixture(scope="module")
def mesh():
    if jax.default_backend() != "cpu" or jax.device_count() < N:
        pytest.skip(f"needs {N} host devices (see module docstring)")
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    pltpu.get_tpu_info = lambda: info
    kmod._vmem_capacity = lambda: 64 << 20
    return Mesh(np.asarray(jax.devices()[:N]), ("d",))


def _jaxpr(mesh, tokens, **kw):
    def body(rank, w1, w2, w1_s, w2_s, sw1, sw2, x, gate_row, tw, ids):
        return kmod.fused_moe_decode(
            rank, x, w1[0], w2[0], w1_s[0][:, :, None, :], w2_s[0][:, :, None, :], sw1[0],
            jnp.ones((2 * ISP,), jnp.float32), sw2[0], jnp.ones((H,), jnp.float32), gate_row, tw, ids,
            n_ranks=N, expert_start="rank", fold=True, **kw)[None]

    f8, bf, f32 = jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32
    args = [S((N, 2), jnp.int32), S((N, E, H, 2 * INTER), f8), S((N, E, INTER, H), f8), S((N, E, 1, 2 * INTER), f32),
            S((N, E, 1, H), f32), S((N, H, 2 * ISP), bf), S((N, ISP, H), bf), S((tokens, H), bf), S((1, H), bf),
            S((tokens, 10), bf), S((tokens, 10), jnp.int32)]
    f = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),) * 7 + (P(),) * 4, out_specs=P("d"), check_vma=False))
    return jax.make_jaxpr(f)(*args)


def _name_stacks(closed):
    """Every equation's name stack, recursively through sub-jaxprs (pallas_call, cond, while, scan, shard_map)."""
    out = []

    def walk(jp):
        for eqn in jp.eqns:
            out.append(str(eqn.source_info.name_stack))
            for v in eqn.params.values():
                for sub in (v if isinstance(v, (list, tuple)) else [v]):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    walk(closed.jaxpr)
    return out


@pytest.mark.parametrize("tokens", [256, 128])
def test_scopes_off_is_the_same_program(mesh, tokens):
    base = _jaxpr(mesh, tokens)
    off = _jaxpr(mesh, tokens, scopes=0)
    assert str(off) == str(base)
    assert _name_stacks(off) == _name_stacks(base)
    stacks = " ".join(_name_stacks(off))
    assert not any(r in stacks for r in REGIONS), "scopes=0 must not name any region"


def test_scopes_on_names_every_region_and_keeps_the_ops(mesh):
    off = _jaxpr(mesh, 256, scopes=0)
    on = _jaxpr(mesh, 256, scopes=1)
    stacks = " ".join(_name_stacks(on))
    missing = [r for r in REGIONS if r not in stacks]
    assert not missing, f"regions missing from the scoped kernel: {missing}"
    assert str(on) == str(off), "scopes must only name regions, not change the ops"


def test_scopes_on_with_compact_rows(mesh):
    """The compact path (lever 5b, compact_rows=32) carries the experts region too."""
    stacks = " ".join(_name_stacks(_jaxpr(mesh, 256, scopes=1, compact_rows=32)))
    assert "moe_experts" in stacks and "ar_barrier_wait" in stacks
