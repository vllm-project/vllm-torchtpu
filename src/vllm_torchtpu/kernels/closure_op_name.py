# SPDX-License-Identifier: Apache-2.0
"""Deterministic torch op qualnames for closure-carrying Pallas/JAX ops.

An op built by ``torch_tpu``'s ``jax_op`` closes over static configuration (for the fused EP MoE: the EP
mesh order that relabels expert ids after top-k, the token replica groups, ...). Dynamo's graph -- and
so vLLM's torch.compile cache key -- sees only the op's qualname, never the closure. Naming ops by build
count (``base`` for the first, ``base_<n>`` after) therefore gives two processes whose closures differ
the SAME name, and a compile-cache artifact saved by one is loaded by the other with the wrong closure
baked in.

q38 DEC-O4-4 (2026-11-11), measured on O6's two engines per host (C1 j-6a86d9cc): the engine on TPU
cores 0-3 builds EP order (0, 1, 2, 3), the one on cores 4-7 builds (2, 3, 0, 1); both named their op
``pallas::adaptive_fused_moe``, saved into one torch.compile key, and loaded each other's programs. On
CPU (j-f1f003f3) a program closed over one order reads the wrong expert for every selected id on the
other half's weight layout.

``closure_op_name(base, key)`` is a pure function of the closure key: same closure -> same name in every
process (and so the same compile key across hosts whose orders match); different closure -> different
name. Plain data (None, bool, int, float, str, Enum, tuples/lists of those), dataclass instances (by field) and a
jax Mesh (axis names and device ids) enter by content; any other object enters by its type only, never by its repr, so an object
whose repr carries an address can never make names (and compile keys) differ from process to process.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from typing import Any

DIGEST_HEX = 12


def _canon(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, _canon(value.value))
    if isinstance(value, (tuple, list)):
        return tuple(_canon(v) for v in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # A frozen statics dataclass (moe_tp_decode's _OpStatics, all_reduce.AllReduceSpec): its fields by content.
        return ("dc", type(value).__qualname__, tuple((f.name, _canon(getattr(value, f.name))) for f in dataclasses.fields(value)))
    devices = getattr(value, "devices", None)
    axis_names = getattr(value, "axis_names", None)
    if devices is not None and axis_names is not None:
        # A jax.sharding.Mesh: axis names and device ids in mesh order. The ids are this process's
        # logical ids; the EP order that tells two TPU-core halves apart is a separate key element.
        flat = getattr(devices, "flat", devices)
        return ("mesh", tuple(str(a) for a in axis_names), tuple(int(d.id) for d in flat))
    # Anything else contributes its type only -- never its repr, which may carry an address and would
    # make the name (and the compile key) differ from process to process.
    return ("obj", type(value).__module__, type(value).__qualname__)


def closure_op_name(base: str, key: tuple) -> str:
    """``base`` + ``_`` + a digest of ``key``'s canonical content (see the module docstring)."""
    canon = repr(tuple(_canon(v) for v in key))
    return f"{base}_{hashlib.sha1(canon.encode()).hexdigest()[:DIGEST_HEX]}"
