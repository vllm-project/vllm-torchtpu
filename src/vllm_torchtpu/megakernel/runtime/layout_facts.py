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
"""Physical facts of one recurrent pool slot, and their bind-time comparison.

The serving framework owns the unified KV + recurrent pool; the megakernel
reads and writes it through `PoolLayout`. Both sides describe one rank's slot
as `PoolLayoutFacts` (the framework from its own layout objects, the
megakernel from `PoolLayout`), and `compare_layout_facts` fails closed on any
difference, before a single kernel touches the pool.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import jax.numpy as jnp

from ..errors import EligibilityResult
from ..pool.layout import PoolLayout

LAYOUT_FACTS_SCHEMA = "megakernel-pool-slot-facts-v1"

# Conv window element order `(taps, [Q | K | V])`, tap-major and segment-major,
# Q/K/V each head-major. The only order the megakernel pool I/O implements.
CONV_ORDER_TAPS_QKV = "taps_qkv"


def dtype_name(dt: Any) -> str:
    """Canonical dtype name (`'bfloat16'`, ...) for torch / jax / numpy / str."""
    s = str(dt).strip()
    for pfx in ("torch.", "jax.numpy.", "numpy."):
        s = s.removeprefix(pfx)
    try:
        return jnp.dtype(s).name
    except TypeError:
        return s


@dataclasses.dataclass(frozen=True)
class PoolLayoutFacts:
    """One rank's view of a unified pool page and one recurrent state slot.

    Byte offsets are relative to the first page of the slot. A slot is one
    framework cache block (`slot_tokens` tokens = whole pool pages).
    """

    page_dtype: str
    page_shape: tuple[int, ...]
    page_size: int  # tokens per pool page
    token_bytes: int
    slot_tokens: int
    ssm_dtype: str
    ssm_bytes: int
    ssm_offset_bytes: int
    conv_dtype: str
    conv_bytes: int
    conv_offset_bytes: int
    conv_order: str
    local_kh: int
    local_vh: int
    d_k: int
    d_v: int
    conv_taps: int
    conv_dim: int

    def payload(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["page_shape"] = list(self.page_shape)
        return {"schema": LAYOUT_FACTS_SCHEMA, **d}

    def fingerprint(self) -> str:
        """sha256 of the canonical JSON payload (stable across processes)."""
        enc = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(enc.encode("ascii")).hexdigest()


def facts_from_pool_layout(layout: PoolLayout, R: int) -> PoolLayoutFacts:
    """The slot facts the megakernel's pool I/O assumes on each of `R` ranks."""
    rec_isz = jnp.dtype(layout.rec_dtype).itemsize
    conv_isz = jnp.dtype(layout.conv_dtype).itemsize
    qkv = layout.qkv_per_rank(R)
    return PoolLayoutFacts(
        page_dtype=dtype_name(jnp.dtype(layout.page_dtype)),
        page_shape=tuple(int(d) for d in layout.kv_page_shape),
        page_size=int(layout.page_size),
        token_bytes=layout.page_bytes // layout.page_size,
        slot_tokens=layout.pages_per_slot * layout.page_size,
        ssm_dtype=dtype_name(jnp.dtype(layout.rec_dtype)),
        ssm_bytes=layout.vh_per_rank(R) * layout.KD * layout.VD * rec_isz,
        ssm_offset_bytes=0,
        conv_dtype=dtype_name(jnp.dtype(layout.conv_dtype)),
        conv_bytes=layout.conv_taps * qkv * conv_isz,
        conv_offset_bytes=layout.conv_page * layout.page_bytes,
        conv_order=CONV_ORDER_TAPS_QKV,
        local_kh=layout.kh_per_rank(R),
        local_vh=layout.vh_per_rank(R),
        d_k=int(layout.KD),
        d_v=int(layout.VD),
        conv_taps=int(layout.conv_taps),
        conv_dim=qkv,
    )


def compare_layout_facts(
    megakernel: PoolLayoutFacts, framework: PoolLayoutFacts
) -> EligibilityResult:
    """Compare the megakernel's and the framework's slot facts, field by field.

    Any difference is ineligible (`pool_layout_mismatch`); a conv state the
    framework stores in another dtype gets its own code
    (`conv_state_dtype_unsupported`), since that is a megakernel capability
    gap rather than a misconfiguration.
    """
    mk_d, fw_d = dataclasses.asdict(megakernel), dataclasses.asdict(framework)
    diffs = [k for k in mk_d if mk_d[k] != fw_d[k]]
    if not diffs:
        return EligibilityResult(True, "ok", "")
    detail = ", ".join(
        f"{k}: megakernel={mk_d[k]!r} framework={fw_d[k]!r}" for k in diffs
    )
    code = (
        "conv_state_dtype_unsupported"
        if "conv_dtype" in diffs
        else "pool_layout_mismatch"
    )
    msg = (
        "Megakernel pool slot layout differs from the framework's "
        f"({detail}); skipping."
    )
    return EligibilityResult(False, code, msg)


__all__ = [
    "CONV_ORDER_TAPS_QKV",
    "LAYOUT_FACTS_SCHEMA",
    "PoolLayoutFacts",
    "compare_layout_facts",
    "dtype_name",
    "facts_from_pool_layout",
]
