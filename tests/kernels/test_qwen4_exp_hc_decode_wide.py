# SPDX-License-Identifier: Apache-2.0
"""TPU_QWEN4_HC_DECODE_WIDE (q38 HC-HIGHROWS): gate, row blocks, dispatch, op names,
compile factors and the real torch_tpu wrapper. Numerics against hc_decode.py and a
float64 reference run in q38_hchigh/hcw_numerics.py (interpret mode, excess precision
off) under the locked prereg; this file holds the fast checks."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode_wide
from vllm_torchtpu.kernels.qwen4_exp.hc_decode_wide import (
    MAX_TOKENS,
    MIN_TOKENS,
    hc_decode_mix_wide,
    hc_decode_mix_wide_auto,
    row_blocks,
    sem_count,
    wide_applies,
)

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN


def test_wide_applies():
    assert (MIN_TOKENS, MAX_TOKENS) == (160, 511)
    assert [t for t in range(1, 1100) if wide_applies(t)] == list(range(160, 512, 16))


@pytest.mark.parametrize("tokens", range(160, 512, 16))
def test_row_blocks_are_the_served_pieces(tokens):
    blocks = row_blocks(tokens)
    assert len(blocks) == -(-tokens // 128)
    offset = 0
    for i, (r0, rows) in enumerate(blocks):
        assert r0 == offset and 0 < rows <= 128
        pieces, tail = divmod(rows, 32)
        assert pieces >= 2
        # hc_decode.py's 32-row pieces start at multiples of 32 from row 0;
        # only the last block carries its 16-row tail.
        assert r0 % 32 == 0
        assert tail == (tokens % 32 if i == len(blocks) - 1 else 0)
        offset += rows
    assert offset == tokens


@pytest.mark.parametrize("prologue", [True, False])
@pytest.mark.parametrize("inject", [True, False])
def test_sem_count_matches_the_copies(prologue, inject):
    n = len(row_blocks(448))
    copies = 3 + n + n  # w, down, up; rows in and mixed out per block
    copies += 2 if inject else 0
    copies += (1 + 2 * n) if prologue else 0
    assert sem_count(448, prologue, inject) == copies


def _specs(tokens, inject=True, prologue=True):
    s = jax.ShapeDtypeStruct
    bf = jnp.bfloat16
    args = [s((tokens, WIDTH), bf), s((WIDTH,), bf), s((LOWRANK, WIDTH), bf), s((WIDTH, LOWRANK), bf)]
    args.append(s((BRANCHES, WIDTH), bf) if inject else None)
    extra = {}
    if prologue:
        extra = dict(block=s((tokens, HIDDEN), bf), prev_injection=s((tokens, BRANCHES), jnp.float32))
    return args, extra


@pytest.mark.parametrize("tokens,mode", [(144, 1), (512, 1), (168, 2), (448, 3)])
def test_wide_rejects_unsupported_rows_and_modes(tokens, mode):
    args, extra = _specs(tokens)

    def f(*a):
        return hc_decode_mix_wide(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, **extra)

    with pytest.raises(ValueError, match="wide"):
        jax.eval_shape(f, *args)


@pytest.mark.parametrize("tokens", [16, 112, 128, 144, 512])
def test_auto_keeps_served_kernel_outside_wide_rows(tokens, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix", lambda *a, **k: calls.append(("served", k["max_tile"])))
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(("wide", k["mode"])))
    args, extra = _specs(tokens)
    hc_decode_mix_wide_auto(*args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=1, max_tile=512, **extra)
    assert calls == [("served", 512)]


@pytest.mark.parametrize("tokens", [160, 240, 448, 496])
@pytest.mark.parametrize("mode", [1, 2])
def test_auto_takes_wide_kernel_at_wide_rows(tokens, mode, monkeypatch):
    calls = []
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix", lambda *a, **k: calls.append(("served", k["max_tile"])))
    monkeypatch.setattr(hc_decode_wide, "hc_decode_mix_wide", lambda *a, **k: calls.append(("wide", k["mode"])))
    args, extra = _specs(tokens)
    hc_decode_mix_wide_auto(*args, branches=BRANCHES, hidden=HIDDEN, eps=EPS, mode=mode, max_tile=512, **extra)
    assert calls == [("wide", mode)]


class _FakeOp:
    def register_fake(self, fn):
        return fn


def _names(monkeypatch, wide=None, compare=False):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    if wide is None:
        monkeypatch.delenv("TPU_QWEN4_HC_DECODE_WIDE", raising=False)
    else:
        monkeypatch.setenv("TPU_QWEN4_HC_DECODE_WIDE", str(wide))
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_COMPARE", "1" if compare else "0")
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_TILE_ROWS", "512")
    names = []
    monkeypatch.setattr(vllm_adapter.pallas, "jax_op", lambda name, core: names.append(name) or _FakeOp())
    for build in (vllm_adapter.build_decode_mix_op, vllm_adapter.build_decode_combine_mix_op):
        build.__wrapped__(BRANCHES, HIDDEN, EPS, True)
        build.__wrapped__(BRANCHES, HIDDEN, EPS, False)
    return names


def test_op_names_follow_the_flag(monkeypatch):
    base = _names(monkeypatch)
    assert len(base) == 4 and all(n.endswith("_t512_v1") for n in base), base
    assert _names(monkeypatch, 0) == base
    for mode in (1, 2):
        names = _names(monkeypatch, mode)
        assert all(n.endswith(f"_t512_w{mode}_v1") for n in names), names
        assert [n.replace(f"_w{mode}_v1", "_v1") for n in names] == base


@pytest.mark.parametrize("wide,compare,match", [(1, True, "exclusive"), (2, True, "exclusive"), (3, False, "expected 0")])
def test_op_build_refuses_bad_settings(monkeypatch, wide, compare, match):
    with pytest.raises(ValueError, match=match):
        _names(monkeypatch, wide, compare)


def test_compile_factors_key_the_flag(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert "TPU_QWEN4_HC_DECODE_WIDE" in envs.environment_variables
    monkeypatch.delenv("TPU_QWEN4_HC_DECODE_WIDE", raising=False)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("TPU_QWEN4_HC_DECODE_WIDE", "0")
    zero = tpu_compiler._tpu_compile_env_factors()
    assert unset == zero and unset["TPU_QWEN4_HC_DECODE_WIDE"] == 0
    for mode in (1, 2):
        monkeypatch.setenv("TPU_QWEN4_HC_DECODE_WIDE", str(mode))
        on = tpu_compiler._tpu_compile_env_factors()
        assert on["TPU_QWEN4_HC_DECODE_WIDE"] == mode
        assert {k for k in on if on[k] != unset.get(k)} == {"TPU_QWEN4_HC_DECODE_WIDE"}


def test_runtime_cache_key_covers_the_wide_kernel():
    from vllm_torchtpu.compilation import tpu_compiler

    root = Path(tpu_compiler.__file__).resolve().parents[2]
    files = {p.resolve() for p in tpu_compiler._iter_runtime_cache_key_files(root)}
    assert Path(hc_decode_wide.__file__).resolve() in files


_REAL_WRAPPER = r"""
import os, sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
from torch_tpu._internal import pallas
from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va
assert va.pallas is pallas and pallas.jax_op.__module__.startswith("torch_tpu"), pallas.jax_op.__module__
built = 0
for mode in ("1", "2"):
    os.environ["TPU_QWEN4_HC_DECODE_WIDE"] = mode
    os.environ["TPU_QWEN4_HC_DECODE_TILE_ROWS"] = "512"
    for build in (va.build_decode_mix_op, va.build_decode_combine_mix_op):
        for inject in (True, False):
            build.__wrapped__(4, 2560, 1e-6, inject)
            built += 1
try:
    pallas.jax_op("q38hcw_rule12::unannotated", lambda x: x)
    print("NEGATIVE CONTROL NOT REFUSED")
    sys.exit(1)
except ValueError as e:
    assert "annotation" in str(e), e
print(f"REAL WRAPPER OK built {built}")
"""


def test_real_torch_tpu_wrapper_accepts_the_wide_ops():
    """Methodology rule 12: build every wide op through the real torch_tpu
    pallas.jax_op (signature / annotation checks), in a fresh process so op
    names are registered once; an unannotated function must be refused."""
    pytest.importorskip("torch_tpu")
    src = str(Path(hc_decode_wide.__file__).resolve().parents[3])
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", _REAL_WRAPPER], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and "REAL WRAPPER OK built 8" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-3000:]
