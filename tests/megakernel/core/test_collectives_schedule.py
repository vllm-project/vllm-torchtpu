# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in all-reduce schedule options of `AllReduceRing` (`Tiles.hier1_peers`,
`Tiles.hx`, `allreduce(fill=)`, the ARPIPE deferred all-reduce, AR0S).

Each option is checked two ways: the 8-rank interpret-mode all-reduce is
bit-identical to the default schedule (the option moves data, it does not
change any sum), and a structural check shows the option reaches the copy
descriptors, so an option that is silently ignored fails.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.megakernel.core.test_stream_schedule import prim_counts
from vllm_torchtpu.megakernel.core import AllReduceRing, collectives
from vllm_torchtpu.megakernel.core.collectives import ring as ring_mod
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32

_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 900
_TINY_TILES = Tiles(
    bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32, mb8=2, war=1
)
_ALT_PEERS = (3, 5, 7)


def _spawn_with_cpu_devices(R: int, tag: str) -> None:
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, tag],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def _ring_outputs(
    tiles: Tiles,
    *,
    R: int = 8,
    H: int = 1024,
    B: int = 4,
    seed: int = 0,
    counts: bool = False,
    fills: list[int] | None = None,
    arp: bool = False,
    ar0: bool = False,
) -> tuple[np.ndarray, np.ndarray, collectives.CommParams]:
    """Run 2 layers x 2 alternating slots of `allreduce` + `add_residual`.

    Returns the per-rank residual rows `[R, B, H]` (f32), the float64 host sum
    of the four reductions `[B, H]`, and the `CommParams` used. With `counts`,
    returns the step's primitive histogram (`prim_counts`) instead of the rows.
    With `fills`, every all-reduce gets a `fill` that appends its slot (at
    trace time) and zeroes a spare buffer. With `arp`, slot 1 is the ARPIPE
    deferred all-reduce: primed, its other half added before every slot-0
    all-reduce (as a layer's input projection would) and after the last step. With `ar0`, slot 0 is the AR0S
    all-reduce: its payload is packed and sent from `ar0_spec`'s column-group
    hooks (fed the partial's columns, as an output projection would) and
    finished by `ar0_finish`.
    """
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Expected >= {R} CPU devices, found {len(devs)}"
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    pc = H // 2 if tiles.war == 0 else None
    cp = collectives.comm_params(ex, tiles, H=H, PC=pc)
    if arp or ar0:
        cp = collectives.params.with_schedule(
            cp, tiles, comm=True, dense=True, in_kind=(128, 1), unroll=8, out_nt=1
        )
        assert arp == cp.ARP and ar0 == cp.AR0S, cp
    scratch = {
        "res": pltpu.VMEM((ex.MB, H), F32),
        "spare": pltpu.VMEM((8, 128), F32),
        **collectives.ring_scratch(cp),
        **collectives.stage_scratch(cp),
    }
    s_names = list(scratch.keys())

    def body(x0_ref, x1_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = AllReduceRing(cp, scr, me)
        if ar.comm:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, R - 1)
        scr["res"][...] = jnp.zeros((ex.MB, H), F32)
        if arp:
            ar.arp_prime()
        for step_idx in range(4):
            slot = step_idx % 2
            src = x0_ref[...] if slot == 0 else x1_ref[...]
            fill = None
            if fills is not None:

                def fill(slot=slot):
                    fills.append(slot)
                    scr["spare"][...] = jnp.zeros((8, 128), F32)

            first = jnp.bool_(step_idx < 2)
            if arp and slot == 1:
                assert ar.allreduce(slot, src, first, fill=fill, defer=True) is None
                continue
            if ar0 and slot == 1:
                ar.add_residual(ar.allreduce(slot, src, first, fill=fill))
                continue
            if arp:  # slot 0: the deferred half (layer 0: the primed zeros)
                ar.arp_round2_add()
            if ar0:  # slot 0
                spec = ar.ar0_spec(0, first)
                spec.pre()
                for cols, hook in zip(spec.cols, spec.hooks):
                    hook([x0_ref[:, pl.ds(c0, cw)] for c0, cw in cols])
                ar.ar0_finish(0, first)
                continue
            ar.add_residual(ar.allreduce(slot, src, first, fill=fill))
        if arp:
            ar.arp_round2_add()
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, H), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(),
    )
    step = jax.jit(
        jax.shard_map(
            lambda x0, x1: call(x0[0], x1[0])[None],
            mesh=Mesh(devs[:R], ("tp",)),
            in_specs=(P("tp"), P("tp")),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    k0, k1, kt = jax.random.split(jax.random.PRNGKey(seed), 3)
    x0 = jax.random.normal(k0, (R, ex.MB, H), dtype=BF16).astype(F32)
    x1 = jax.random.normal(k1, (R, ex.MB, H), dtype=BF16).astype(F32)
    spikes = jax.random.bernoulli(kt, 0.05, (R, ex.MB, H)).astype(F32) * 8.0
    x0 = (x0 + spikes).astype(BF16).astype(F32)
    x1 = (x1 - spikes).astype(BF16).astype(F32)
    if counts:
        return prim_counts(step, x0, x1), None, cp
    out = np.asarray(step(x0, x1), dtype=np.float32)[:, :B]
    x0_64 = np.asarray(x0[:, :B], dtype=np.float64)
    x1_64 = np.asarray(x1[:, :B], dtype=np.float64)
    host = 2.0 * (np.sum(x0_64, axis=0) + np.sum(x1_64, axis=0))
    return out, host, cp


def _assert_close_to_host_sum(out: np.ndarray, host: np.ndarray) -> None:
    """Loose sanity bound: the reduced rows track the float64 sum (non-vacuous)."""
    scale = float(np.max(np.abs(host))) + 1.0
    for r in range(out.shape[0]):
        assert float(np.max(np.abs(out[r] - host))) <= 0.1 * scale, r
    assert float(np.max(np.abs(out))) > 1.0


# ---------------------------------------------------------------------------
# hier1_peers
# ---------------------------------------------------------------------------
def _check_hier1_peers_bit_identical() -> None:
    for war, H, B, seed in ((1, 1024, 4, 0), (1, 1024, 3, 17), (0, 2048, 4, 42)):
        base_t = dataclasses.replace(_TINY_TILES, hier=1, war=war)
        alt_t = dataclasses.replace(base_t, hier1_peers=_ALT_PEERS)
        base, host, cp0 = _ring_outputs(base_t, H=H, B=B, seed=seed)
        alt, _, cp1 = _ring_outputs(alt_t, H=H, B=B, seed=seed)
        assert (cp0.HIER, cp1.HIER) == (1, 1)
        assert cp1.HIER1_PEERS == _ALT_PEERS != cp0.HIER1_PEERS
        np.testing.assert_array_equal(alt, base)
        _assert_close_to_host_sum(alt, host)


class _FakeRef:
    """Stands in for a scratch ref: `.at[idx]` returns `(name, idx)`."""

    def __init__(self, name: str):
        self.name = name

    @property
    def at(self) -> Any:
        name = self.name

        class _At:
            def __getitem__(self, idx: Any) -> tuple[str, Any]:
                return (name, idx)

        return _At()


def _round_peers(tiles: Tiles, rnd_i: int, me: int) -> list[int]:
    """Device ids of `ar_copies(0, rnd_i)` on rank `me` (host-side, no kernel)."""
    ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
    cp = collectives.comm_params(ex, tiles, H=1024)
    names = ("send", "recv", "ag_send", "ag_recv", "s2_send", "s2_recv")
    scr = {k: _FakeRef(k) for k in (*names, "ar_ssem", "ar_rsem")}
    fake = types.SimpleNamespace(make_async_remote_copy=lambda **kw: kw)
    with mock.patch.object(ring_mod, "pltpu", fake):
        descs = AllReduceRing(cp, scr, me).ar_copies(0, rnd_i)
    return [int(d["device_id"][0]) for d in descs]


def test_hier1_peers_reach_the_cross_chip_round():
    """`hier1_peers` sets round 1's peers; round 0 (on-chip twin) is unchanged."""
    base = dataclasses.replace(_TINY_TILES, hier=1)
    alt = dataclasses.replace(base, hier1_peers=_ALT_PEERS)
    for me in range(8):
        assert _round_peers(base, 1, me) == [me ^ 2, me ^ 4, me ^ 6]
        got = _round_peers(alt, 1, me)
        assert got == [me ^ 3, me ^ 5, me ^ 7]
        # one core on each other chip, on the other core parity
        assert sorted(p // 2 for p in got) == sorted({0, 1, 2, 3} - {me // 2})
        assert all(p % 2 != me % 2 for p in got)
        assert _round_peers(alt, 0, me) == _round_peers(base, 0, me) == [me ^ 1]


def test_hier1_peers_only_change_hier1():
    """`hier=2` keeps its (2, 4, 6) pushes whatever `hier1_peers` says."""
    base = dataclasses.replace(_TINY_TILES, hier=2)
    alt = dataclasses.replace(base, hier1_peers=_ALT_PEERS)
    for me in range(8):
        assert _round_peers(alt, 1, me) == _round_peers(base, 1, me)


@pytest.mark.parametrize("peers", [(2, 4, 4), (1, 4, 6), (2, 4), (2, 4, 6, 3)])
def test_hier1_peers_invalid_raises(peers: tuple[int, ...]):
    tiles = dataclasses.replace(_TINY_TILES, hier=1, hier1_peers=peers)
    ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
    with pytest.raises(MegakernelConfigError, match="hier1_peers"):
        collectives.comm_params(ex, tiles, H=1024)


def test_hier1_peers_bit_identical_to_default():
    """(3, 5, 7) peers: same chip sums in the same order, so identical outputs."""
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "hier1_peers")
        return
    _check_hier1_peers_bit_identical()


# ---------------------------------------------------------------------------
# hx
# ---------------------------------------------------------------------------
# (war, H, B): unfolded payload with HC2 = 512 (quarter 128 lanes), and folded
# with PC = 2048, HC2 = 1024 (quarter 256 lanes).
_HX_CASES = ((1, 1024, 4, 0), (1, 1024, 3, 17), (1, 1024, 16, 5), (0, 4096, 4, 42))


def _check_hx_bit_identical() -> None:
    for war, H, B, seed in _HX_CASES:
        base_t = dataclasses.replace(_TINY_TILES, hier=2, war=war)
        hx_t = dataclasses.replace(base_t, hx=True)
        base, host, cp0 = _ring_outputs(base_t, H=H, B=B, seed=seed)
        alt, _, cp1 = _ring_outputs(hx_t, H=H, B=B, seed=seed)
        assert (cp0.HX, cp1.HX) == (False, True), (war, H)
        np.testing.assert_array_equal(alt, base)
        _assert_close_to_host_sum(alt, host)


def _check_hx_structure() -> None:
    """hx adds round 3: per all-reduce 3 more remote copies (5 -> 8), and the
    same number of semaphore waits per copy."""
    base_t = dataclasses.replace(_TINY_TILES, hier=2)
    c0, _, cp0 = _ring_outputs(base_t, counts=True)
    c1, _, cp1 = _ring_outputs(dataclasses.replace(base_t, hx=True), counts=True)
    assert cp0.AR_ROUNDS == (0, 1, 2) and cp1.AR_ROUNDS == (0, 1, 2, 3)
    n_ar = 4  # 2 layers x 2 slots
    assert c1["dma_start"] - c0["dma_start"] == 3 * n_ar, (c0, c1)
    assert c1["dma_wait"] > c0["dma_wait"]
    assert c1["dot_general"] == c0["dot_general"]


def test_hx_gate_and_rounds():
    """hx engages only on hier=2 at R=8 with quarters of 128 lanes."""

    def cp_for(**kw: Any) -> collectives.CommParams:
        tiles = dataclasses.replace(_TINY_TILES, **kw)
        ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
        return collectives.comm_params(ex, tiles, H=1024)

    on = cp_for(hier=2, hx=True)
    assert on.HX and on.AR_ROUNDS == (0, 1, 2, 3)
    assert collectives.ring_scratch(on)["ar_ssem"].shape == (4, 2, 3)
    assert not cp_for(hier=2).HX
    assert not cp_for(hier=1, hx=True).HX
    tiles = dataclasses.replace(_TINY_TILES, hier=2, hx=True)
    ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
    narrow = collectives.comm_params(ex, tiles, H=512)  # HC2 = 256: quarter 64
    assert not narrow.HX and narrow.AR_ROUNDS == (0, 1, 2)


def test_hx_bit_identical_to_hier2():
    """hx: the same quarter sums in the same chip order, so identical outputs."""
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "hx")
        return
    _check_hx_bit_identical()


def test_hx_adds_the_all_gather_round():
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "hx_structure")
        return
    _check_hx_structure()


# ---------------------------------------------------------------------------
# allreduce(fill=)
# ---------------------------------------------------------------------------
_FILL_MODES = (
    dict(hier=0),
    dict(hier=0, rsag=True),
    dict(hier=1),
    dict(hier=2),
    dict(hier=2, hx=True),
    dict(hier=3),
)


def _check_fill_once_and_bit_identical() -> None:
    for kw in _FILL_MODES:
        tiles = dataclasses.replace(_TINY_TILES, **kw)
        base, _, _ = _ring_outputs(tiles)
        fills: list[int] = []
        out, _, _ = _ring_outputs(tiles, fills=fills)
        assert fills == [0, 1, 0, 1], (kw, fills)
        np.testing.assert_array_equal(out, base)


def test_allreduce_fill_runs_once_per_all_reduce():
    """`fill` is issued exactly once inside every all-reduce mode, and the
    reduced rows do not change."""
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "fill")
        return
    _check_fill_once_and_bit_identical()


def test_allreduce_fill_without_comm():
    """`R=1`: no rounds, `fill` still runs once."""
    fills: list[int] = []
    _ring_outputs(dataclasses.replace(_TINY_TILES, hier=0), R=1, fills=fills)
    assert fills == [0, 1, 0, 1]


# ---------------------------------------------------------------------------
# ARPIPE: deferred all-reduce, other half added later
# ---------------------------------------------------------------------------
_ARP_MODES = (
    dict(hier=2, arpipe=True),
    dict(hier=2, arpipe=True, hx=True),
    dict(hier=1, arpipe=True, arpipe_h1=True),
)


def _check_arp_bit_identical() -> None:
    """Each residual lane gets the same sums in the same order whether slot 1
    is added at once or as two halves around the next layer's start."""
    for kw in _ARP_MODES:
        for war, H, B in ((1, 1024, 4), (0, 4096, 3)):
            tiles = dataclasses.replace(_TINY_TILES, war=war, **kw)
            plain = dataclasses.replace(tiles, arpipe=False, arpipe_h1=False)
            base, host, _ = _ring_outputs(plain, H=H, B=B, seed=9)
            fills: list[int] = []
            out, _, cp = _ring_outputs(tiles, H=H, B=B, seed=9, arp=True, fills=fills)
            assert cp.ARP and (kw["hier"] == 1) == cp.ARP_H1, (kw, cp)
            assert fills == [0, 1, 0, 1], (kw, fills)
            np.testing.assert_array_equal(out, base)
            _assert_close_to_host_sum(out, host)


# ---------------------------------------------------------------------------
# AR0S: round 0 packed and sent from the producer's column groups
# ---------------------------------------------------------------------------
_AR0_MODES = (
    (dict(hier=2, ar0s=True), False),
    (dict(hier=2, ar0s=True, hx=True), False),
    (dict(hier=2, ar0s=True, arpipe=True), True),
    (dict(hier=1, ar0s=True, arpipe=True, arpipe_h1=True), True),
)


def _check_ar0_bit_identical() -> None:
    """Same payload values and the same sums: bit-identical to the plain
    all-reduce of the same partial (alone and with the deferred slot 1)."""
    for kw, arp in _AR0_MODES:
        for war, H, B in ((1, 1024, 4), (0, 4096, 3)):
            tiles = dataclasses.replace(_TINY_TILES, war=war, **kw)
            plain = dataclasses.replace(tiles, ar0s=False, arpipe=False)
            base, host, _ = _ring_outputs(plain, H=H, B=B, seed=21)
            out, _, cp = _ring_outputs(tiles, H=H, B=B, seed=21, arp=arp, ar0=True)
            assert cp.AR0S and cp.A0_CW == 512, (kw, cp)
            np.testing.assert_array_equal(out, base)
            _assert_close_to_host_sum(out, host)


def test_ar0s_bit_identical_to_plain_all_reduce():
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "ar0")
        return
    _check_ar0_bit_identical()


def test_with_schedule_gates():
    def cp_for(
        in_kind: Any = (128, 1), unroll: int = 8, out_nt: Any = None, **kw: Any
    ) -> Any:
        tiles = dataclasses.replace(_TINY_TILES, **kw)
        ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
        cp = collectives.comm_params(ex, tiles, H=1024)
        return collectives.params.with_schedule(
            cp,
            tiles,
            comm=True,
            dense=True,
            in_kind=in_kind,
            unroll=unroll,
            out_nt=out_nt,
        )

    on2 = cp_for(hier=2, arpipe=True)
    assert on2.ARP and not on2.ARP_H1 and on2.AR_ROUNDS == (0, 1, 2)
    on1 = cp_for(hier=1, arpipe=True, arpipe_h1=True)
    assert on1.ARP and on1.ARP_H1 and on1.AR_ROUNDS == (0, 1, 2)
    assert cp_for(hier=2, arpipe=True, hx=True).AR_ROUNDS == (0, 1, 2, 3)
    assert not cp_for(hier=1, arpipe=True).ARP  # hier=1 needs arpipe_h1
    assert not cp_for(hier=0, arpipe=True).ARP
    assert not cp_for(hier=2, arpipe=True, in_kind=None).ARP
    assert not cp_for(hier=2, arpipe=True, in_kind=(128, 9)).ARP  # not unrolled
    assert not cp_for(hier=2, arpipe=True, in_kind=(384, 1)).ARP  # HC2 % tk
    assert cp_for(hier=2) == cp_for(hier=2, in_kind=None)
    a2 = cp_for(hier=2, ar0s=True, out_nt=2)
    assert a2.AR0S and not a2.ARP and a2.A0_CW == 512
    assert not cp_for(hier=1, ar0s=True, out_nt=2).AR0S  # hier=1 needs ARPIPE
    a1 = cp_for(hier=1, ar0s=True, arpipe=True, arpipe_h1=True, out_nt=2)
    assert a1.AR0S and a1.ARP_H1
    assert not cp_for(hier=2, ar0s=True, out_nt=None).AR0S
    assert not cp_for(hier=2, ar0s=True, out_nt=9).AR0S  # not unrolled


def test_arp_deferred_all_reduce_bit_identical():
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "arp")
        return
    _check_arp_bit_identical()


_CHECKS = {
    "ar0": _check_ar0_bit_identical,
    "arp": _check_arp_bit_identical,
    "fill": _check_fill_once_and_bit_identical,
    "hier1_peers": _check_hier1_peers_bit_identical,
    "hx": _check_hx_bit_identical,
    "hx_structure": _check_hx_structure,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _CHECKS[sys.argv[1]]()
