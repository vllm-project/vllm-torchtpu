# SPDX-License-Identifier: Apache-2.0
"""q38 DEC-O4-4 [plan-v4 DEC-O6 CP1]: closure-carrying MoE ops are named by their closure, not their build count.

C1 j-6a86d9cc (O6, two engines per host): the engine on TPU cores 0-3 builds EP order (0, 1, 2, 3), the one on 4-7
builds (2, 3, 0, 1). Both named their op ``pallas::adaptive_fused_moe``, so they shared one torch.compile key and loaded
each other's programs; CPU proof j-f1f003f3: the wrong order reads the wrong expert for every selected id.

Each check builds ops in FRESH interpreters (different PYTHONHASHSEED), with torch_tpu's ``jax_op`` stubbed, so the name
under test is the one ``vllm_adapter._build_op`` / ``fused_moe_ep._build_op`` really compute:
  * different EP orders -> different names; the same order -> the same name in another process, whatever the build order;
  * the token replica groups enter the name;
  * an opaque stand-in object (no content) gives the same name in every process (never its repr / address);
  * the identity-order adaptive name for a fixed reference closure is pinned (it changes once, from the bare
    ``pallas::adaptive_fused_moe``, at the first build on this tree).
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

_ADAPTIVE = r'''
import json, sys
import vllm_torchtpu.kernels.experimental.adaptive_fused_moe.vllm_adapter as A
class _Stub:
    def __init__(self, name): self.name = name
    def register_fake(self, f): return f
A.jax_op = lambda name, fn, **kw: _Stub(name)
mesh = object() if sys.argv[1] == "object" else "MESH"
out = []
for spec in json.loads(sys.argv[2]):
    order = None if spec["order"] is None else tuple(spec["order"])
    groups = None if spec["groups"] is None else tuple(tuple(g) for g in spec["groups"])
    out.append(A._build_op(mesh, 8, True, "silu", order, "fp8", None, token_replica_groups=groups).name)
print("NAMES " + json.dumps(out))
'''

_FUSED_EP = r'''
import json, sys
import vllm_torchtpu.layers.adapter.fused_moe_ep as B
class _Stub:
    def __init__(self, name): self.name = name
    def register_fake(self, f): return f
B.jax_op = lambda name, fn, **kw: _Stub(name)
out = []
for spec in json.loads(sys.argv[2]):
    order = None if spec["order"] is None else tuple(spec["order"])
    groups = None if spec["groups"] is None else tuple(tuple(g) for g in spec["groups"])
    out.append(B._build_op(None, 2, True, "silu", order, False, token_replica_groups=groups).name)
print("NAMES " + json.dumps(out))
'''


# The three ops that used build-count names too (no EP order in their closures, but build-order-dependent names):
# fused_moe_decode (FR10b's decode / prefill path; its key also held id(mesh)), moe_tp_decode and its selftest op.
_FUSED_DECODE = r"""
import json, sys
import vllm_torchtpu.layers.adapter.fused_moe_decode as D
class _Stub:
    def __init__(self, name): self.name = name
    def register_fake(self, f): return f
D.jax_op = lambda name, fn, **kw: _Stub(name)
mesh = object() if sys.argv[1] == "object" else "MESH"
out = [D._build_op(mesh, spec["n_ranks"], 10, True, 256).name for spec in json.loads(sys.argv[2])]
print("NAMES " + json.dumps(out))
"""

_TP_DECODE = r"""
import json, sys
import numpy as np, jax
from jax.sharding import Mesh
import vllm_torchtpu.layers.adapter.moe_tp_decode as M
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from jax.experimental.pallas import tpu as pltpu
_info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
pltpu.get_tpu_info = lambda: _info   # the selftest op sizes its VMEM from the chip; CPU has none
class _Stub:
    def __init__(self, name): self.name = name
    def register_fake(self, f): return f
M.pallas.jax_op = lambda name, fn, **kw: _Stub(name)
mesh = Mesh(np.asarray(jax.devices()[:1]), ("ep",))
out = []
for spec in json.loads(sys.argv[2]):
    if spec.get("selftest"):
        out.append(M._selftest_op(mesh, all_reduce.AllReduceSpec(n=1, m=8, h=spec["h"], tile_rows=16, lanes=128)).name)
    else:
        st = M._OpStatics(topk=spec["topk"], routed_scale=1.0, activation="silu", use_sparse_core=False, onehot_threshold=0)
        out.append(M._get_op(mesh, st).name)
print("NAMES " + json.dumps(out))
"""

def _names(code, specs, seed, mesh="str"):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONHASHSEED=str(seed), JAX_PLATFORMS="cpu")
    p = subprocess.run([sys.executable, "-B", "-c", code, mesh, json.dumps(specs)], capture_output=True, text=True,
                       env=env, cwd=str(ROOT), timeout=600)
    line = [l for l in p.stdout.splitlines() if l.startswith("NAMES ")]
    assert p.returncode == 0 and line, (p.returncode, (p.stdout + p.stderr)[-2000:])
    return json.loads(line[-1][6:])


ID, HALF2 = None, [2, 3, 0, 1]
G1, G2 = [[0, 1]], [[0], [1]]


@pytest.mark.parametrize("code", [_ADAPTIVE, _FUSED_EP], ids=["adaptive", "fused_moe_ep"])
def test_orders_get_different_names_and_the_same_name_across_processes(code):
    forward = [{"order": ID, "groups": None}, {"order": HALF2, "groups": None}]
    a = _names(code, forward, seed=1)
    b = _names(code, list(reversed(forward)), seed=2)[::-1]   # the other build order, another hash seed
    print(f"names {a} / {b}")
    assert a[0] != a[1], "EP orders (0,1,2,3) and (2,3,0,1) must not share an op name (compile key)"
    assert a == b, "the same closure must get the same name in every process, whatever the build order"


@pytest.mark.parametrize("code", [_ADAPTIVE, _FUSED_EP], ids=["adaptive", "fused_moe_ep"])
def test_replica_groups_enter_the_name(code):
    a = _names(code, [{"order": HALF2, "groups": G1}, {"order": HALF2, "groups": G2}], seed=3)
    assert a[0] != a[1]
    assert _names(code, [{"order": HALF2, "groups": G2}], seed=4) == [a[1]]


def test_an_opaque_mesh_object_gives_the_same_name_in_every_process():
    specs = [{"order": HALF2, "groups": None}]
    a = _names(_ADAPTIVE, specs, seed=5, mesh="object")
    b = _names(_ADAPTIVE, specs, seed=6, mesh="object")
    assert a == b, "an object's repr (address) must never reach the op name"


def test_identity_order_reference_name_is_pinned():
    # Fixed reference closure: mesh "MESH", topk 8, renormalize, silu, identity order, fp8, rhs_qb None, no groups.
    assert _names(_ADAPTIVE, [{"order": ID, "groups": None}], seed=7) == ["pallas::adaptive_fused_moe_0f91e60dabdd"]


def test_fused_moe_decode_names_by_closure():
    fwd = [{"n_ranks": 8}, {"n_ranks": 4}]
    a = _names(_FUSED_DECODE, fwd, seed=11)
    b = _names(_FUSED_DECODE, list(reversed(fwd)), seed=12)[::-1]
    assert a[0] != a[1] and a == b
    # its key held id(mesh): an opaque mesh object must now give the same name in every process
    assert _names(_FUSED_DECODE, fwd[:1], seed=13, mesh="object") == _names(_FUSED_DECODE, fwd[:1], seed=14, mesh="object")


def test_moe_tp_decode_names_by_closure():
    fwd = [{"topk": 8}, {"topk": 4}]
    a = _names(_TP_DECODE, fwd, seed=15)
    b = _names(_TP_DECODE, list(reversed(fwd)), seed=16)[::-1]
    assert a[0] != a[1] and a == b, "the statics dataclass must enter the name by content"


def test_moe_tp_decode_selftest_names_by_closure():
    fwd = [{"selftest": True, "h": 2560}, {"selftest": True, "h": 1024}]
    a = _names(_TP_DECODE, fwd, seed=17)
    b = _names(_TP_DECODE, list(reversed(fwd)), seed=18)[::-1]
    assert a[0] != a[1] and a == b
