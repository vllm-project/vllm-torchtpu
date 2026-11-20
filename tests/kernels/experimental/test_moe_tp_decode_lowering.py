# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The TPU lowering of the moe_tp_decode kernels, pinned by hash.

Each case lowers one decode MoE layer for TPU7x on the host (no TPU needed)
and compares the sha256 of the lowered HLO text, Mosaic kernel body included,
with the hash recorded in _HASHES. Lowering runs with
jax_traceback_in_locations_limit=0, so the text carries no source locations:
moving code between lines does not change a hash, while any change to an op,
operand, constant or op order does. Both weight formats are pinned, so an edit
made for one cannot silently change the other's program. The pinned program
includes everything the kernels trace: kernel.py, nvfp4.py, all_reduce.py and
quantized_matmul.util.quantize_block.

The hashes are recorded for one jax version, _JAX_VERSION, which must equal
the jax pin in pyproject.toml; with any other installed jax the lowering cases
skip. When a change to the program is intended, or when the jax pin moves, set
_JAX_VERSION if needed, run this file, and copy the new hash from each failure
message into _HASHES. To see what changed, diff str(f.trace(*shapes).jaxpr) of
the case at the old and the new commit (the kernel jaxpr, without locations;
the Mosaic body in as_text() is one opaque blob).
"""

import hashlib
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as fp8_kernel
from vllm_torchtpu.kernels.experimental.moe_tp_decode.nvfp4 import moe_tp_decode_nvfp4

FP8 = jnp.float8_e4m3fn
FP4 = jnp.float4_e2m1fn
N_RANKS = 8
ROUTED_SCALE = 2.5
_JAX_VERSION = "0.11.2"

# Per-rank shapes: e experts routed, e // ep local, intermediate i per rank.
TP8 = dict(h=6144, e=256, i=256, fs=256, topk=8, b13=512, b2=256, ep=1)
EP2TP4 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=512, ep=2)
EP2TP4_B256 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=256, ep=2)
# I = 128: one FP8 ring item per expert, NVFP4 group 4.
TP16 = dict(h=6144, e=256, i=128, fs=128, topk=8, b13=512, b2=128, ep=1)
# 192 routed experts: the routing table is padded to 256 lanes.
E192 = dict(h=6144, e=192, i=256, fs=256, topk=8, b13=512, b2=256, ep=1)
CFGS = {
    "tp8": TP8,
    "ep2tp4": EP2TP4,
    "ep2tp4_b256": EP2TP4_B256,
    "tp16": TP16,
    "e192": E192,
}

# (weight format, layout, rows, kernel kwargs) -> sha256 of the lowered HLO.
_HASHES = {
    ("fp8", "tp8", 16, (("fast", False),)): (
        "3889ddf81d0a0504f21246825499a9483df9ac1f2449e892e7b74504b417d4f4"
    ),
    ("fp8", "tp8", 16, (("fast", True), ("hbm_operands", True))): (
        "14859a9ddace044faa741155002361a2671b4898e1f0bf49456cdd74d27a5a33"
    ),
    ("fp8", "ep2tp4", 64, (("fast", True), ("split_meta", True))): (
        "baf87219b02d6279499a29ef12a1ae4cf13bce3a9880789b6c7aca653ddc0681"
    ),
    ("fp8", "tp16", 16, (("fast", True),)): (
        "0ec610b3e52036fe06b0b90e0d9ceaff769e363b637067755bc4dcbf88690d3e"
    ),
    ("fp8", "tp16", 32, (("fast", False),)): (
        "b4e516223e824cb462e14f7259a6f8b4bb43533a7d9f262cb54da37143c31a6f"
    ),
    ("fp8", "e192", 16, (("fast", True),)): (
        "a26497afe01d90681c43cbdda56c6ee93c245b232c9499c07109792335ff6744"
    ),
    ("nvfp4", "tp8", 16, (("fast", False),)): (
        "427912ec6daddd22f7298033adb3a62845fb7660ec9ec66f882866a12b9ed112"
    ),
    ("nvfp4", "tp8", 16, (("fast", True),)): (
        "73cf209b3be726cc7277f9e10dab733f7de4405fd752597955001ff16be60356"
    ),
    ("nvfp4", "ep2tp4", 64, (("fast", True),)): (
        "dfc287b95def0bb0760398782cc8e1a14349e441ef5d3a90208e3bc6550e7184"
    ),
    ("nvfp4", "ep2tp4", 128, (("fast", True),)): (
        "6eac72a7695e7261e54bc2e4a552ff7ccde0feb044de8e39602be2bc8fc5c53c"
    ),
    ("nvfp4", "ep2tp4", 16, (("fast", True), ("split_meta", False))): (
        "d8a11b24ead52db1a8ae8e0f72f6a4ef1ab83eb2e6640dc9062ad1e8532514c6"
    ),
    ("nvfp4", "ep2tp4_b256", 32, (("fast", True), ("w2_n_chunk", 6144))): (
        "969cdf9f23862d68a9ecbe19c2da516e0ee41d46ced9b390a67782f44e41b56c"
    ),
    ("nvfp4", "tp16", 16, (("fast", True),)): (
        "0f59c2be60e17494b2cef3c00aa407a8b957b7fe0ba942709361cbc15f1b0bfa"
    ),
    ("nvfp4", "tp16", 64, (("fast", False),)): (
        "8d24a47b33e52a7cabdc0ba16e3d9e0c23a25f8f1d6ce864c777b2bff0c2254c"
    ),
    ("nvfp4", "e192", 16, (("fast", True),)): (
        "5dc8ef662708c946fe17c7b8b4ccea74a6e22b929c358c90356bdda44f89e101"
    ),
}


def _pinned_jax_version() -> str:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    (pin,) = [d.split("==", 1)[1] for d in deps if d.startswith("jax==")]
    return pin


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _shapes(rows, cfg, fmt):
    sds = jax.ShapeDtypeStruct
    h, e, i, fs = cfg["h"], cfg["e"], cfg["i"], cfg["fs"]
    el = e // cfg["ep"]
    if fmt == "fp8":
        w13, w2 = sds((el, h, 2 * i), FP8), sds((el, i, h), FP8)
        s13 = sds((el, 1, 1, 2 * i), jnp.float32)
        s2 = sds((el, 1, 1, h), jnp.float32)
    else:
        w13, w2 = sds((el, h, 2 * i), FP4), sds((el, i, h), FP4)
        s13 = sds((el, h // cfg["b13"], 1, 2 * i), jnp.float32)
        s2 = sds((el, i // cfg["b2"], 1, h), jnp.float32)
    return (
        sds((rows, h), jnp.bfloat16),
        sds((rows, e), jnp.float32),
        sds((e,), jnp.float32),
        sds((1,), jnp.int32),
        sds((1, 1), jnp.int32),
        sds((1, 1), jnp.int32),
        w13,
        w2,
        s13,
        s2,
        sds((h, 2 * fs), FP8),
        sds((2 * fs,), jnp.float32),
        sds((fs, h), FP8),
        sds((h,), jnp.float32),
    )


def _lower(fn, rows, cfg, fmt, kwargs):
    """HLO text of fn under shard_map, lowered for TPU without locations."""
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("d",))
    specs = (P("d"), P("d"), P(), P(), P("d"), P("d")) + (P("d"),) * 8

    def body(x, scores, bias, n_valid, rank, e0, *weights):
        return fn(
            x,
            scores,
            bias,
            n_valid[0],
            rank.reshape(()),
            *weights,
            e0.reshape(()),
            routed_scale=ROUTED_SCALE,
            topk=cfg["topk"],
            n_ranks=N_RANKS,
            **kwargs,
        )

    f = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )
    # No file:line locations: no traceback frames, and none from the user frame
    # that jax_include_full_tracebacks_in_locations=False would add back.
    limit = jax.config.jax_traceback_in_locations_limit
    full = jax.config.jax_include_full_tracebacks_in_locations
    jax.config.update("jax_traceback_in_locations_limit", 0)
    jax.config.update("jax_include_full_tracebacks_in_locations", True)
    try:
        lowered = f.trace(*_shapes(rows, cfg, fmt)).lower(lowering_platforms=("tpu",))
        return lowered.as_text()
    finally:
        jax.config.update("jax_traceback_in_locations_limit", limit)
        jax.config.update("jax_include_full_tracebacks_in_locations", full)


def _case_id(case):
    fmt, layout, rows, kw = case
    return "-".join([fmt, layout, f"m{rows}", *(f"{k}-{v}" for k, v in kw)])


def test_recorded_jax_version_is_the_pin():
    assert _pinned_jax_version() == _JAX_VERSION, (
        "pyproject.toml pins a different jax: re-record _HASHES under it and "
        "update _JAX_VERSION"
    )


@pytest.mark.parametrize("case", list(_HASHES), ids=_case_id)
def test_lowering_matches_recorded_hash(monkeypatch, request, case):
    if jax.__version__ != _pinned_jax_version():
        pytest.skip(f"installed jax {jax.__version__} is not the pinned one")
    fmt, layout, rows, kw = case
    kwargs = dict(kw)
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    if fmt == "fp8":
        # moe_tp_decode reads its scalar-operand layout from the environment.
        split = kwargs.pop("split_meta", False)
        monkeypatch.setattr(fp8_kernel, "_split_meta", lambda: split)
        fn = fp8_kernel.moe_tp_decode
    else:
        fn = moe_tp_decode_nvfp4
    # The jit trace cache ignores the patched _split_meta and TPU info: start
    # and leave it empty.
    fn.clear_cache()
    request.addfinalizer(fn.clear_cache)
    got = hashlib.sha256(_lower(fn, rows, CFGS[layout], fmt, kwargs).encode())
    assert got.hexdigest() == _HASHES[case], (
        f"the TPU lowering of {_case_id(case)} changed: {got.hexdigest()} "
        f"(recorded {_HASHES[case]}). If the change is intended, record the new "
        "hash in _HASHES."
    )
