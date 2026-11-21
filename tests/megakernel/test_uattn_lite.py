"""VLLM_TPU_MK_UATTN_LITE: union attention's per-tile op cuts (`attend_request(lite=True)`).
CPU interpret, small shapes: every output byte matches `lite=False` over served-like, edge
(full tiles, long runs), empty and stale-slot unions, with and without main-KV DCP and
physical ids; under DCP, groups with code 0 (unowned) hold NaN bytes. Mutations (no NaN
sanitize, the wrong expand half) must be caught."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.blocks.qsa_attn.block import MICRO, TILE
from vllm_torchtpu.megakernel.wholestep import union_attn as ua

KU, T, NH, PER_PAGE, GROUPS = 2048, 4, 2, 8, 1200  # 8 groups per page: many page crossings
PPS = -(-GROUPS // PER_PAGE)


def _served(rng, nu: int, groups: int) -> np.ndarray:
    lens = []
    while sum(lens) < nu:
        lens.append(int(rng.choice([1, 1, 1, 2, 2, 3, 5, 9, 20])))
    lens[-1] -= sum(lens) - nu
    gaps = 1 + rng.multinomial(groups - nu - len(lens), np.ones(len(lens)) / len(lens))
    out, v = [], -1
    for g, n in zip(gaps, lens):
        v += int(g)
        out += range(v, v + n)
        v += n - 1
    return np.array(out, np.int64)


def _unions(kind: str, dcp: int) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    if kind == "served":
        own = [_served(rng, 150, GROUPS // dcp), _served(rng, 97, GROUPS // dcp)]
    elif kind == "edge":  # exactly two full tiles; long runs and singles ending mid-tile
        own = [np.arange(0, 128), np.r_[np.arange(5, 90), [93, 95, 96, 101], np.arange(120, 141)]]
    elif kind == "empty":
        own = [np.zeros((0,), np.int64), np.array([7])]
    else:
        raise ValueError(kind)
    return [(x // PER_PAGE) * (dcp * PER_PAGE) + x % PER_PAGE for x in own]


def _inputs(unions, dcp: int, phys: bool, poison: bool, stale_nan: bool):
    rng = np.random.default_rng(5)
    nreq, pages = len(unions), PPS * 2
    ids = np.zeros((nreq, KU + 128), np.int32)
    n = np.zeros((nreq,), np.int32)
    codes = np.zeros((nreq, 1, KU), np.int32)
    tab = np.stack([rng.permutation(pages)[:PPS] for _ in range(nreq)]).astype(np.int32)
    for q, u in enumerate(unions):
        ids[q, : len(u)] = u
        ids[q, len(u): KU] = rng.integers(0, PPS * PER_PAGE * dcp, KU - len(u))
        n[q] = len(u)
        codes[q, 0, : len(u)] = rng.integers(1, 1 << T, len(u))
    pool = np.array((jax.random.normal(jax.random.PRNGKey(0),
                                       (pages, PER_PAGE * MICRO, 1, 4, 256)) * 2.0
                     ).astype(jnp.float8_e4m3fn))
    sh = PER_PAGE.bit_length() - 1 + dcp.bit_length() - 1

    def phys_of(q, v):
        return int(tab[q, min(v >> sh, PPS - 1)]) * PER_PAGE + (v & (PER_PAGE - 1))

    raw = pool.view(np.uint8)
    nan_mp = []
    if poison:  # unowned groups: code 0, NaN bytes where no request scores the micro-page
        for q, u in enumerate(unions):
            codes[q, 0, : len(u)][rng.random(len(u)) < 0.2] = 0
        live = {phys_of(q, int(v)) for q, u in enumerate(unions)
                for j, v in enumerate(u) if codes[q, 0, j]}
        nan_mp = [phys_of(q, int(v)) for q, u in enumerate(unions)
                  for j, v in enumerate(u) if not codes[q, 0, j]]
        if stale_nan:  # the slots past n_union in each request's last tile
            for q, u in enumerate(unions):
                end = -(-len(u) // 64) * 64
                nan_mp += [phys_of(q, int(v)) for v in ids[q, len(u):end]]
        nan_mp = [p for p in nan_mp if p not in live]
        for p in nan_mp:
            pg, mp = divmod(p, PER_PAGE)
            raw[pg, mp * MICRO:(mp + 1) * MICRO] = rng.choice(
                np.array([0x7F, 0xFF, 0x80], np.uint8), raw[pg, mp * MICRO:(mp + 1) * MICRO].shape)
    if phys:
        for q in range(nreq):
            ids[q] = [phys_of(q, int(v)) for v in ids[q]]
    q = jax.random.normal(jax.random.PRNGKey(1), (T, NH, 256), jnp.float32)
    return [jnp.asarray(x) for x in (ids, n, tab, codes)] + [q, jnp.asarray(pool)], len(nan_mp)


def _run(unions, dcp: int, lite: bool, phys: bool = False, poison: bool = False,
         stale_nan: bool = False) -> np.ndarray:
    args, _ = _inputs(unions, dcp, phys, poison, stale_nan)
    nreq = len(unions)

    def kernel(ids, n_ref, tab, codes, q_ref, pool, o_ref, ring, sems):
        z8 = jnp.zeros((8, 256), jnp.float32)
        ok = [jnp.zeros((1, 8), jnp.bool_)] * T
        qs = [q_ref[t] for t in range(T)]
        for q in range(nreq):
            res = ua.attend_request(
                T=T, nh=NH, pool=pool, ring=ring, sems=sems, ids=ids, codes=codes.at[q],
                n_union=n_ref[q], table=tab, table_row=q, qs=qs, gates=qs,
                has=[jnp.bool_(True)] * T, off=1, scale_k=0.5, scale_v=0.25,
                sm_scale=256 ** -0.5, interpret=True, recent=(z8, z8, ok, jnp.zeros((8, 1), bool)),
                phys=phys, dcp=dcp, ids_row=q, **(dict(lite=True) if lite else {}))
            for t in range(T):
                if dcp > 1:
                    o_ref[q * T + t, :NH] = res[t][0]
                    o_ref[q * T + t, NH:] = jnp.broadcast_to(res[t][1], (NH, 256))
                else:
                    o_ref[q * T + t, :NH] = res[t]
                    o_ref[q * T + t, NH:] = jnp.zeros((NH, 256), jnp.float32)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((nreq * T, 2 * NH, 256), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM)] * 3
        + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 2 + [pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((3, TILE, 1, 4, 256), jnp.float8_e4m3fn),
                        pltpu.SemaphoreType.DMA((3,))],
        interpret=pltpu.InterpretParams(),
    )
    return np.asarray(jax.jit(call)(*args)).view(np.uint32)


@pytest.mark.parametrize("kind,dcp,phys,poison,stale_nan", [
    ("served", 1, False, False, False),
    ("served", 2, False, True, True),
    ("edge", 1, True, False, False),
    ("edge", 2, True, True, True),
    ("empty", 2, False, True, True),
])
def test_lite_bit_identical(kind, dcp, phys, poison, stale_nan):
    unions = _unions(kind, dcp)
    if poison:
        assert _inputs(unions, dcp, phys, poison, stale_nan)[1] > 0
    base = _run(unions, dcp, False, phys, poison, stale_nan)
    out = base.view(np.float32)
    assert np.isfinite(out[:, :NH]).all()
    np.testing.assert_array_equal(_run(unions, dcp, True, phys, poison, stale_nan), base)


def _mutated(monkeypatch, old: str, new: str) -> None:
    src = inspect.getsource(ua.attend_request)
    bad = src.replace(old, new)
    assert bad != src
    ns = dict(ua.__dict__)
    exec(bad, ns)  # noqa: S102
    monkeypatch.setattr(ua, "attend_request", ns["attend_request"])


@pytest.mark.parametrize("old,new,dcp", [
    # lite drops the NaN sanitize: unowned NaN V reach the PV matmul
    ("jnp.where(value == value, value * scale_v, 0.0)", "value * scale_v", 2),
    # lite picks the wrong expand half: tokens get the other half's codes
    ("expand[0], expand[1])\n", "expand[1], expand[0])\n", 1),
])
def test_lite_mutation_is_caught(monkeypatch, old, new, dcp):
    unions = _unions("served", dcp)
    base = _run(unions, dcp, False, poison=dcp > 1, stale_nan=dcp > 1)
    _mutated(monkeypatch, old, new)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(
            _run(unions, dcp, True, poison=dcp > 1, stale_nan=dcp > 1), base)
