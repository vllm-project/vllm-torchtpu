# SPDX-License-Identifier: Apache-2.0
"""q38 H8-FIX: the fix moves a name binding only, so the op's traced program must be identical to 5d368f166's.
Builds global_fn from the fixed module and from 5d368f166's file bytes (fixtures/, loaded under another module name) on
8 host devices in the engine's mesh order, and compares the printed jaxprs (Pallas TPU interpreter, as the AR-FIX math
test) for both wires at rows the kernel takes (16, 64, 256) and rows that fall back to lax.psum in the op (24, 512).
Needs JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8. Last line: Q38_H8FIX_JAXPR PASS|FAIL."""
import importlib.machinery
import importlib.util
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as new

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NAME = "q38_h8fix_tp8_ar_decode_fn_5d368f166"
_PATH = os.path.join(ROOT, "tests/q38_h8fix/fixtures/tp8_ar_decode_fn.5d368f166.py.txt")
spec = importlib.util.spec_from_file_location(_NAME, _PATH, loader=importlib.machinery.SourceFileLoader(_NAME, _PATH))
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
assert jax.default_backend() == "cpu" and jax.device_count() >= 8, "needs 8 host devices"
devs = {d.id: d for d in jax.devices()[:8]}
mesh = Mesh(np.array([devs[i] for i in ORDER]), ("ep",))
pairs = new.pairs_from_device_ids(ORDER)
assert pairs == old.pairs_from_device_ids(ORDER)
ok = True
n = 0
for wire in ("bf16", "f32"):
    for rows in (16, 64, 256, 24, 512):
        x = jax.ShapeDtypeStruct((8 * rows, 2560), jnp.bfloat16)
        kw = dict(axis="ep", pairs=pairs, wire=wire, max_rows_=256, interpret=True)
        a = str(jax.make_jaxpr(new.global_fn(mesh, **kw))(x))
        b = str(jax.make_jaxpr(old.global_fn(mesh, **kw))(x))
        same = a == b
        kernel = "pallas_call" in a
        n += 1
        print(f"wire {wire} rows {rows}: jaxpr {'IDENTICAL' if same else 'DIFFERENT'} ({len(a)} chars, pallas_call={kernel})")
        ok &= same
# Sensitivity control: the comparison must see a real program change (wire bf16 vs f32 at 64 rows must differ).
x = jax.ShapeDtypeStruct((8 * 64, 2560), jnp.bfloat16)
ctl_a = str(jax.make_jaxpr(new.global_fn(mesh, axis="ep", pairs=pairs, wire="bf16", max_rows_=256, interpret=True))(x))
ctl_b = str(jax.make_jaxpr(old.global_fn(mesh, axis="ep", pairs=pairs, wire="f32", max_rows_=256, interpret=True))(x))
print(f"control bf16 (fixed) vs f32 (5d368f166) at 64 rows: {'DIFFERENT (comparison is sensitive)' if ctl_a != ctl_b else 'IDENTICAL (control FAILED)'}")
ok &= ctl_a != ctl_b
print(f"Q38_H8FIX_JAXPR {'PASS' if ok else 'FAIL'} ({n} cases + 1 control)")
sys.exit(0 if ok else 1)
