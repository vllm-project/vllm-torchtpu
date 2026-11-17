# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The TPU lowering of the moe_tp_decode kernels, pinned by hash.

Each case lowers one GLM-5.2 decode MoE layer for TPU7x on the host (no TPU
needed) and compares a hash of the result with the one recorded in _HASHES:
the HLO text with each Mosaic kernel body replaced by the hash of its
assembly without locations, so moving code between lines does not change it
while any change to an op, operand, constant or op order does. Both weight
formats are pinned, so an edit made for one cannot silently change the
other's program.

The hashes hold for the jax version they were recorded with (_JAX_VERSION);
other versions skip. After an intended kernel change, or for a new jax, run
this file and copy the printed hashes into _HASHES.
"""

import base64
import hashlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as fp8_kernel
from vllm_torchtpu.kernels.experimental.moe_tp_decode_nvfp4 import (
    moe_tp_decode_nvfp4,
)

FP8 = jnp.float8_e4m3fn
FP4 = jnp.float4_e2m1fn
N_RANKS = 8
ROUTED_SCALE = 2.5
_JAX_VERSION = "0.11.2"

TP8 = dict(h=6144, e=256, i=256, fs=256, topk=8, b13=512, b2=256, ep=1)
EP2TP4 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=512, ep=2)
EP2TP4_B256 = dict(h=6144, e=256, i=512, fs=256, topk=8, b13=512, b2=256, ep=2)
CFGS = {"tp8": TP8, "ep2tp4": EP2TP4, "ep2tp4_b256": EP2TP4_B256}

# (weight format, layout, rows, kernel kwargs) -> sha256 of the normalized HLO.
_HASHES = {
    ("fp8", "tp8", 16, (("fast", False),)): (
        "b1e05590fe2d9a036dc51d5e0c4f616b19f5fb295459779372bf15389d6620e4"
    ),
    ("fp8", "tp8", 16, (("fast", True), ("hbm_operands", True))): (
        "030f0de08325ec25c0200176427433940d5a3154b7ced443d2e401fb821e7f18"
    ),
    ("fp8", "ep2tp4", 64, (("fast", True), ("split_meta", True))): (
        "d62dbf4d950a20615c5fa959870433dd21f28c889a2c7a3f0ec45a9932446631"
    ),
    ("nvfp4", "tp8", 16, (("fast", False),)): (
        "bdab0e323889b09ddb07b9f93580daff34c5b4213d2511a447555ff3d299df2b"
    ),
    ("nvfp4", "tp8", 16, (("fast", True),)): (
        "27bf2c26d9693a989c0e86c5f9ba4c4c3d042138c334b6145184489cb77ef278"
    ),
    ("nvfp4", "ep2tp4", 64, (("fast", True),)): (
        "7d5e0b127218f8ce1a14a1a0f55e2f7465590c132b0ea819fb8835e90900685b"
    ),
    ("nvfp4", "ep2tp4", 16, (("fast", True), ("split_meta", False))): (
        "deadd8e460cbceb8f2fec3fa33017ee774fb8678455d0b427c89cc96b63102a4"
    ),
    ("nvfp4", "ep2tp4_b256", 32, (("fast", True), ("w2_n_chunk", 6144))): (
        "47ffc0fc5f2c83672a8e41c1e22ad693459f154b9e6821e601550e57bf6990d3"
    ),
}

_BODY = re.compile(r"(\\22body\\22: \\22)([A-Za-z0-9+/=]+)(\\22)")


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
    """HLO text of fn under shard_map, lowered for TPU (abstract arguments)."""
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
    lowered = f.trace(*_shapes(rows, cfg, fmt)).lower(lowering_platforms=("tpu",))
    return lowered.as_text()


def _normalized_hash(hlo: str) -> str:
    from jax._src.lib import tpu as tpu_lib
    from jax._src.lib.mlir import ir

    def mosaic(m):
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        tpu_lib.register_dialect(ctx)
        with ctx:
            module = ir.Module.parse(base64.b64decode(m.group(2)))
            asm = module.operation.get_asm(enable_debug_info=False)
        digest = hashlib.sha256(asm.encode()).hexdigest()
        return f"{m.group(1)}MOSAIC:{digest}{m.group(3)}"

    normalized, n = _BODY.subn(mosaic, hlo)
    assert n == 1, f"expected one Mosaic kernel, found {n}"
    return hashlib.sha256(normalized.encode()).hexdigest()


@pytest.mark.parametrize("case", list(_HASHES), ids=lambda c: "-".join(map(str, c[:3])))
def test_lowering_matches_recorded_hash(monkeypatch, case):
    if jax.__version__ != _JAX_VERSION:
        pytest.skip(f"hashes recorded with jax {_JAX_VERSION}, not {jax.__version__}")
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
    # A trace cached by another test would ignore the patched _split_meta.
    fn.clear_cache()
    got = _normalized_hash(_lower(fn, rows, CFGS[layout], fmt, kwargs))
    assert got == _HASHES[case], (
        f"the TPU lowering of {case} changed: {got} (recorded {_HASHES[case]}). "
        "If the change is intended, record the new hash in _HASHES."
    )
