# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-HC: the fused prefill HC mix options behind TPU_QWEN4_HC_PREFILL_FOLD
(``reread``) and TPU_QWEN4_HC_PREFILL_DOWN_ASM (``down_asm``), interpret mode at
Qwen3.8-Flash dims (4 x 2560, low rank 320, 256-row tiles) and the served
prefill step sizes (6,400 = 25 tiles; 3,200 = 12 tiles + a 128-row partial).

* reread: the previous HC's combine as the prologue, with the norm reading the
  combined rows back from VMEM, must equal gated_residual followed by the plain
  fused mix bit for bit (mixed, injection gate, combined streams).
* down_asm: [down; inject] assembled in VMEM must give the plain kernel's
  outputs bit for bit, with no per-call [384, W] concatenate left in the jaxpr.

Rule 23 (CPU-interpret bitwise claims): excess precision is pinned off below,
before jax is imported, and a one-ulp bf16 positive control must read NOT
bitwise under the same flags.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import functools  # noqa: E402
import re  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_fused import hc_prefill_mix  # noqa: E402

BRANCHES, HIDDEN, LOWRANK, EPS, TILE = 4, 2560, 320, 1e-6, 256
WIDTH = BRANCHES * HIDDEN
INTERPRET = pltpu.InterpretParams()
SERVED_ROWS = [6400, 3200]

fused = functools.partial(
    hc_prefill_mix,
    branches=BRANCHES,
    hidden=HIDDEN,
    eps=EPS,
    token_tile=TILE,
    interpret=INTERPRET,
    partial=True,
    norm_impl="bf16ep64",
)


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint16) if a.dtype == jnp.bfloat16 else a.view(np.uint32)


def _weights(seed=7, lowrank=LOWRANK):
    rng = np.random.default_rng(seed)
    shapes = ((WIDTH,), (lowrank, WIDTH), (WIDTH, lowrank), (BRANCHES, WIDTH))
    scales = (0.3, 0.02, 0.05, 0.02)
    return tuple(
        jnp.asarray(rng.normal(size=s) * c, jnp.bfloat16) for s, c in zip(shapes, scales)
    )


def _stream(rows, seed):
    rng = np.random.default_rng(seed)
    prev = jnp.asarray(4.0 * rng.normal(size=(rows, WIDTH)), jnp.bfloat16)
    block = jnp.asarray(rng.normal(size=(rows, HIDDEN)), jnp.bfloat16)
    gates = jnp.asarray(2.0 * rng.random((rows, BRANCHES)), jnp.float32)
    return prev, block, gates


def _gated_residual(prev, block, gates):
    return gated_residual_fn(BRANCHES, HIDDEN, TILE, partial=True, interpret=INTERPRET)(
        prev, block, gates
    )


def _assert_bitwise(name, got, want):
    g, w = _bits(got), _bits(want)
    assert g.shape == w.shape, (name, g.shape, w.shape)
    diff = int((g != w).sum())
    assert diff == 0, f"{name}: {diff} of {g.size} values differ"


def test_excess_precision_pinned_in_this_file():
    assert _STRICT in os.environ["XLA_FLAGS"]


@pytest.mark.parametrize("rows", SERVED_ROWS)
def test_reread_prologue_matches_gated_residual_then_mix(rows):
    """The folded mlp combine (reread prologue) == gated_residual + fused mix."""
    weight, down, up, inject = _weights()
    prev, block, gates = _stream(rows, seed=rows)
    h_ref = _gated_residual(prev, block, gates)
    mixed_ref, inj_ref = fused(h_ref, weight, down, up, inject)
    mixed, inj, h = fused(
        prev, weight, down, up, inject, block=block, prev_injection=gates, reread=True
    )
    _assert_bitwise("h", h, h_ref)
    _assert_bitwise("mixed", mixed, mixed_ref)
    _assert_bitwise("injection", inj, inj_ref)


def test_one_ulp_positive_control_reads_not_bitwise():
    """Rule 23 (b): a one-ulp bf16 change of the combined streams must show."""
    rows = 3 * TILE + 128
    weight, down, up, inject = _weights()
    prev, block, gates = _stream(rows, seed=11)
    h = _gated_residual(prev, block, gates)
    mixed_ref, inj_ref = fused(h, weight, down, up, inject)
    bits = _bits(h).copy()
    bits[5, 7] += 1  # one bf16 ulp on one element
    h_ulp = jnp.asarray(bits.view(jnp.bfloat16))
    mixed, inj = fused(h_ulp, weight, down, up, inject)
    assert int((_bits(mixed) != _bits(mixed_ref)).sum()) + int(
        (_bits(inj) != _bits(inj_ref)).sum()
    ) > 0


def _jaxpr(rows, **kw):
    weight, down, up, inject = _weights()
    prev, block, gates = _stream(rows, seed=1)
    if kw.pop("prologue", False):
        kw.update(block=block, prev_injection=gates)
    return jax.make_jaxpr(lambda *a: fused(*a, **kw))(prev, weight, down, up, inject).jaxpr


def _kernel_names(jaxpr):
    """pallas_call kernel names (interpret-mode lowerings carry none, so read the jaxpr)."""
    names = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            info = eqn.params.get("name_and_src_info", eqn.params.get("name", ""))
            name = getattr(info, "name", None)
            if not isinstance(name, str):
                found = re.search(r"qwen4exp_hc_prefill\w*", str(info) or str(eqn.params))
                name = found.group(0) if found else "?"
            names.append(name)
        for sub in _subjaxprs(eqn):
            names += _kernel_names(sub)
    return names


def test_reread_engagement_and_prologue_only():
    rows = 3 * TILE
    plain = _jaxpr(rows, prologue=True)
    rr = _jaxpr(rows, prologue=True, reread=True)
    assert _kernel_names(rr) == ["qwen4exp_hc_prefill_combine_inj_m256_rr"], _kernel_names(rr)
    assert _kernel_names(plain) == ["qwen4exp_hc_prefill_combine_inj_m256"], _kernel_names(plain)
    assert str(plain) != str(rr)
    # Without a prologue there is nothing to read back: same program, same name.
    assert str(_jaxpr(rows, reread=True)) == str(_jaxpr(rows))
    assert _kernel_names(_jaxpr(rows, reread=True)) == ["qwen4exp_hc_prefill_inj_m256"]


@pytest.mark.parametrize("prologue", [False, True])
@pytest.mark.parametrize("rows", SERVED_ROWS)
def test_down_asm_matches_plain_kernel(rows, prologue):
    weight, down, up, inject = _weights()
    prev, block, gates = _stream(rows, seed=rows + 1)
    kw = dict(block=block, prev_injection=gates) if prologue else {}
    want = fused(prev, weight, down, up, inject, **kw)
    got = fused(prev, weight, down, up, inject, down_asm=True, **kw)
    for name, g, w in zip(("mixed", "injection", "h"), got, want):
        _assert_bitwise(name, g, w)


def _subjaxprs(eqn):
    for value in eqn.params.values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if hasattr(item, "eqns"):
                yield item
            elif hasattr(item, "jaxpr") and hasattr(item.jaxpr, "eqns"):
                yield item.jaxpr


def _concat_shapes(jaxpr):
    shapes = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "concatenate":
            shapes.append(tuple(eqn.outvars[0].aval.shape))
        for sub in _subjaxprs(eqn):
            shapes += _concat_shapes(sub)
    return shapes


@pytest.mark.parametrize("prologue", [False, True])
def test_down_asm_engagement(prologue):
    rows = 3 * TILE
    weight, down, up, inject = _weights()
    prev, block, gates = _stream(rows, seed=2)
    kw = dict(block=block, prev_injection=gates) if prologue else {}

    def run(asm):
        return jax.make_jaxpr(lambda *a: fused(*a, down_asm=asm, **kw))(
            prev, weight, down, up, inject
        ).jaxpr

    ext = (LOWRANK + BRANCHES + (-(LOWRANK + BRANCHES) % 128), WIDTH)
    assert ext in _concat_shapes(run(False))
    shapes = _concat_shapes(run(True))
    assert ext not in shapes and (16, WIDTH) in shapes, shapes
    name = f"qwen4exp_hc_prefill{'_combine' if prologue else ''}_inj_m256"
    assert _kernel_names(run(True)) == [name + "_da"], _kernel_names(run(True))
    assert _kernel_names(run(False)) == [name], _kernel_names(run(False))


def test_down_asm_rejects_unaligned_low_rank():
    weight, down, up, inject = _weights(lowrank=72)
    prev, _, _ = _stream(2 * TILE, seed=3)
    with pytest.raises(ValueError, match="DOWN_ASM"):
        fused(prev, weight, down, up, inject, down_asm=True)
