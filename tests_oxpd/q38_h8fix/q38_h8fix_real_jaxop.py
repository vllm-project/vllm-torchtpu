# SPDX-License-Identifier: Apache-2.0
"""q38 H8-FIX P8: torch_tpu's REAL jax_op on the H8 function, on CPU (when torch_tpu is importable, as in the CDK
parasim-cpu image). Builds global_fn on 8 host devices in the engine's mesh order from the fixed module and from
5d368f166's bytes (fixtures/), and hands each to the real ``torch_tpu._internal.pallas.jax_op`` with the arguments
``tp_ar_decode.arm`` passes. Also calls jax_op's own helpers on the signature (``_map_jax_return_to_schema_str``).

Verdict rules (pre-stated):
- FAIL if the fixed function raises NameError anywhere in jax_op, or its return annotation maps to anything but "Tensor".
- For 5d368f166's function, "NameError: name 'jax' is not defined" = the TPU failure reproduced through the real entry
  point (REPRODUCED). Anything else is reported, not judged (the real op build may need more than CPU offers).
- torch_tpu absent / helpers missing: INFO only (P1's stand-in covers the evaluation), never a silent PASS of a check.
Last line: Q38_H8FIX_REALJAXOP PASS|FAIL|INFO."""
import importlib.machinery
import importlib.util
import inspect
import os
import sys
import traceback

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as new

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_NAME = "q38_h8fix_tp8_ar_decode_fn_5d368f166_p8"
_PATH = os.path.join(ROOT, "tests/q38_h8fix/fixtures/tp8_ar_decode_fn.5d368f166.py.txt")
_spec = importlib.util.spec_from_file_location(_NAME, _PATH, loader=importlib.machinery.SourceFileLoader(_NAME, _PATH))
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)

if importlib.util.find_spec("torch_tpu") is None:
    print("torch_tpu not importable here")
    print("Q38_H8FIX_REALJAXOP INFO")
    sys.exit(0)
try:
    from torch_tpu._internal.pallas import jax_op
except Exception as e:  # noqa: BLE001
    print(f"torch_tpu._internal.pallas.jax_op not importable: {type(e).__name__}: {e}")
    print("Q38_H8FIX_REALJAXOP INFO")
    sys.exit(0)
try:
    from torch_tpu._internal.pallas import pallas as impl
except Exception:  # noqa: BLE001
    impl = None
import torch_tpu

print(f"torch_tpu {getattr(torch_tpu, '__version__', '?')} at {os.path.dirname(torch_tpu.__file__)}; "
      f"jax_op from {inspect.getsourcefile(inspect.unwrap(jax_op))}")
src = inspect.getsource(inspect.unwrap(jax_op))
print(f"jax_op source evaluates eval_str=True: {'eval_str=True' in src}")

ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
assert jax.default_backend() == "cpu" and jax.device_count() >= 8, "needs 8 host devices"
devs = {d.id: d for d in jax.devices()[:8]}
mesh = Mesh(np.array([devs[i] for i in ORDER]), ("ep",))
pairs = new.pairs_from_device_ids(ORDER)
spec = PartitionSpec("ep")
verdict = "PASS"


def build(mod, tag):
    fn = mod.global_fn(mesh, axis="ep", pairs=pairs, wire="bf16", max_rows_=256)
    try:
        op = jax_op(f"q38h8fix::tp_ar_{tag}", fn, mesh=mesh, input_partition_specs=(spec,), output_partition_specs=spec)
        return fn, "BUILT", f"{type(op).__name__}"
    except NameError as e:
        return fn, "NameError", str(e)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        return fn, type(e).__name__, f"{e} | {' / '.join(tb[-3:])[:400]}"


fn_new, r_new, info_new = build(new, "fixed")
print(f"fixed (H8-FIX head): real jax_op -> {r_new}: {info_new}")
if r_new == "NameError" or "is not defined" in info_new:
    verdict = "FAIL"
fn_old, r_old, info_old = build(old, "orig5d368f166")
print(f"5d368f166 bytes:     real jax_op -> {r_old}: {info_old}")
if r_old == "NameError" and "name 'jax' is not defined" in info_old:
    print("5d368f166: the TPU NameError REPRODUCED through torch_tpu's real jax_op on CPU")
else:
    print("5d368f166: real jax_op did not raise the NameError (reported, not judged)")

if impl is not None and hasattr(impl, "_map_jax_return_to_schema_str"):
    sig = inspect.signature(inspect.unwrap(fn_new), eval_str=True)
    ret = impl._map_jax_return_to_schema_str(sig.return_annotation)
    print(f"fixed: real _map_jax_return_to_schema_str(return annotation) = {ret!r}")
    if ret != "Tensor":
        verdict = "FAIL"
    if hasattr(impl, "_verify_signature"):
        try:
            bound = impl._get_bound_names(fn_new) if hasattr(impl, "_get_bound_names") else None
            impl._verify_signature(sig, bound_names=bound) if bound is not None else impl._verify_signature(sig)
            print("fixed: real _verify_signature OK")
        except Exception as e:  # noqa: BLE001
            print(f"fixed: real _verify_signature raised {type(e).__name__}: {e}")
            verdict = "FAIL"
else:
    print("torch_tpu pallas helpers not found (INFO)")
print(f"Q38_H8FIX_REALJAXOP {verdict}")
sys.exit(0 if verdict == "PASS" else 1)
