# SPDX-License-Identifier: Apache-2.0
"""q38 L1-D08S1 diagnostic (not part of the identity suite): B3's three parts one at a time against MM's block, every
output compared (no early stop), NaN rows listed, optionally with eager DMA execution in the interpreter."""
import numpy as np
import pytest

from tests.megakernel import test_wholestep as tw


def _report(run, base, D, KU, what):
    bad = []
    for i, name in ((0, "residual"), (3, "ck_new"), (4, "ring_new"), (6, "index queries")):
        a, b = np.asarray(run[i]), np.asarray(base[i])
        if not np.array_equal(a, b, equal_nan=False):
            nan = np.isnan(a)
            rows = sorted(set(map(tuple, np.argwhere(nan.any(axis=-1)).tolist())))[:40] if nan.any() else []
            bad.append(f"{name}: {int((a != b).sum())} differ, NaN {int(nan.sum())} in rows {rows}")
    for k in ("_kr_new", "_ir_new"):
        if not np.array_equal(run[8][k], base[8][k]):
            bad.append(f"{k} differs")
    n_r, ids_r, cd_r, kvl_r, sga_r, z_r = tw._s1_union(run, D, KU)
    n_b, ids_b, cd_b, kvl_b, sga_b, z_b = tw._s1_union(base, D, KU)
    for name, x, y in (("qu_n", n_r, n_b), ("kvl", kvl_r, kvl_b), ("sga", sga_r, sga_b)):
        if not np.array_equal(x, y):
            bad.append(f"{name} differs")
    if z_r.any():
        bad.append("qi_z nonzero")
    print(f"L1-D08S1 DIAG {what}: {'IDENTICAL' if not bad else ' | '.join(bad)}", flush=True)
    return bad


@pytest.mark.parametrize("scen", ["d1_r2_8", "d1_r2_32"])
@pytest.mark.parametrize("mode", ["on_wait", "eager"])
def test_l1d08s1_diag_b3(monkeypatch, scen, mode):
    from jax.experimental.pallas import tpu as pltpu

    if mode == "eager":
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Eager(pltpu.InterpretParams):
            dma_execution_mode: str = "eager"

        monkeypatch.setattr(pltpu, "InterpretParams", Eager)
        print(f"L1-D08S1 DIAG interpret params {pltpu.InterpretParams()}", flush=True)
    kw = tw.S1_SCEN[scen]
    D, KU = kw["dcp"], tw._s1_ku(kw["budget"])
    base = tw._s1_run(monkeypatch, scen, None)
    bad = {}
    for parts in ("newgrp", "recent", "uids"):
        monkeypatch.setitem(tw.S1_CONFIGS, "P", dict(slot_batch=True, slot_batch_parts=parts))
        bad[parts] = _report(tw._s1_run(monkeypatch, scen, "P"), base, D, KU, f"{scen} {mode} B3[{parts}]")
    assert not any(bad.values()), bad
