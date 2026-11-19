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
# ==============================================================================
"""Named layer modes of the De-Dup MoE layer and their resolution.

A mode names one TensorCore configuration of `monolithic.dedup_ep_moe`:

  `direct_w<W>[_scdma[_g]_s4]`  direct wave dispatch: each (token, device) pair
                 is sent once, in W ramp waves (wave_dispatch.wave_plan).
  `dedup_w<W>[_scdma[_g]_s4[_scfwd]]`  De-Dup wave dispatch: tokens needed by
                 several devices of a remote board are sent once to that board's
                 gateway, which forwards them (needs a `BoardTopology`).
  `_scdma[_g]_s4`  SparseCore sender on 4 vector subcores (`_g`: per-row gather
                 sends, else contiguous sends of TC-packed rows); without it
                 the TensorCore start kernel sends.
  `_scfwd`       (De-Dup, SparseCore sender) the gateway K4 forwards are issued
                 by the gateway's SparseCore as each stage run lands
                 (`sc_sender.sc_forward_start`), not by the fused kernel.

W is one of SUPPORTED_WAVES. Every mode computes the same arithmetic, so all
of them give bit-identical outputs; they differ only in how the tokens move.
"""

import dataclasses
import itertools
import re
from typing import Any

from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib

DIRECT = "direct"
DEDUP = "dedup"
SUPPORTED_WAVES = (1, 2, 4, 8)
DIRECT_W1 = "direct_w1"
DIRECT_W2 = "direct_w2"
DIRECT_W4 = "direct_w4"
DIRECT_W8 = "direct_w8"
DECODE_MODES = (DIRECT_W1, DIRECT_W2)
# The mode every configuration supports (no topology needed).
FALLBACK_MODE = DIRECT_W4

# SMEM a per-target-wave forward table may take: the fused kernel's whole
# XLA callee share is fused_kernel.config.SMEM_CALLEE_BUDGET_BYTES, and its
# other tables need much less than the rest.
FWD_TAB_SMEM_BUDGET_BYTES = 384 * 1024

def _sc_sender():
  """The SparseCore sender module, imported only when an SC mode is used."""
  from tokamax._src.ops.experimental.dedup_moe.dispatch import sc_sender  # pylint: disable=g-import-not-at-top

  return sc_sender


_MODE_RE = re.compile(
    r"^(direct|dedup)_w(\d+)(?:_(scdma(?:_g)?_s4)(?:_(scfwd))?)?$"
)


class SkipCaseError(Exception):
  """A mode that does not apply to this shape or topology."""


@dataclasses.dataclass(frozen=True)
class ModeSpec:
  """One layer mode.

  Attributes:
    dedup: Whether the mode De-Dups the dispatch.
    waves: Expert waves of the dispatch.
    sc_sender: Whether the mode uses the SparseCore sender.
    sc_gather: Whether the SparseCore sender gathers rows directly (`_g`).
    sc_forward: Whether the gateway K4 forwards run on the SparseCore.
  """

  dedup: bool
  waves: int
  sc_sender: bool = False
  sc_gather: bool = False
  sc_forward: bool = False

  @property
  def name(self) -> str:
    """The mode name."""
    return mode_name(
        dedup=self.dedup,
        waves=self.waves,
        sc_sender=self.sc_sender,
        sc_gather=self.sc_gather,
        sc_forward=self.sc_forward,
    )

  @property
  def impl(self) -> str:
    """Layer implementation tag (`product` or `product_scdma_cfg`)."""
    return "product_scdma_cfg" if self.sc_sender else "product"

  @property
  def sc_kwargs(self) -> dict[str, Any]:
    """Keyword arguments for `sc_sender.wrap_layer` when `sc_sender` is True."""
    if not self.sc_sender:
      return {}
    kw: dict[str, Any] = dict(num_subcores=_sc_sender().NUM_SUBCORES)
    if self.sc_gather:
      kw["gather"] = True
    if self.sc_forward:
      kw["sc_forward"] = True
    return kw

  @property
  def label(self) -> str:
    """Human-readable description."""
    kind = "De-Dup" if self.dedup else "Direct"
    if not self.sc_sender:
      return f"{kind} wave dispatch, W={self.waves}, TensorCore sender"
    sender = "gather" if self.sc_gather else "contiguous"
    fwd = ", SparseCore gateway forwards" if self.sc_forward else ""
    return (
        f"{kind} wave dispatch, W={self.waves}, SparseCore {sender} sender"
        f" (4 subcores){fwd}"
    )


def mode_name(
    *,
    dedup: bool,
    waves: int,
    sc_sender: bool = False,
    sc_gather: bool = False,
    sc_forward: bool = False,
) -> str:
  """Returns the name of the mode with these settings."""
  base = f"{DEDUP if dedup else DIRECT}_w{waves}"
  if not sc_sender:
    return base
  sc_tag = "_scdma_g_s4" if sc_gather else "_scdma_s4"
  fwd_tag = "_scfwd" if sc_forward else ""
  return f"{base}{sc_tag}{fwd_tag}"


def is_decode_mode(name: str) -> bool:
  """Returns whether `name` is a low-wave direct decode mode."""
  spec = parse_mode(name)
  return not spec.dedup and spec.waves <= 2


def parse_mode(name: str) -> ModeSpec:
  """Parses a mode name.

  Args:
    name: A mode name such as `direct_w4`, `dedup_w4`, or
      `dedup_w4_scdma_g_s4_scfwd`.

  Returns:
    The ModeSpec.

  Raises:
    ValueError: If the name is not a mode.
  """
  m = _MODE_RE.match(name)
  if m is None or int(m.group(2)) not in SUPPORTED_WAVES:
    raise ValueError(
        f"unknown mode {name!r}: want direct_w<W> or dedup_w<W> (optionally"
        f" with _scdma[_g]_s4[_scfwd]) with W in {SUPPORTED_WAVES}"
    )
  dedup = m.group(1) == DEDUP
  waves = int(m.group(2))
  sc_group = m.group(3)
  sc_fwd = m.group(4) == "scfwd"
  if sc_fwd and not dedup:
    raise ValueError(
        f"unknown mode {name!r}: _scfwd requires a dedup_w<W> mode"
    )
  return ModeSpec(
      dedup=dedup,
      waves=waves,
      sc_sender=sc_group is not None,
      sc_gather=sc_group == "scdma_g_s4",
      sc_forward=sc_fwd,
  )


# Default TensorCore modes in MODES (kept identical for existing callers/tests);
# SC_MODES adds the SparseCore sender and SparseCore gateway-forward modes.
MODES: dict[str, ModeSpec] = {
    spec.name: spec
    for spec in (
        ModeSpec(dedup=d, waves=w)
        for d, w in itertools.product((False, True), SUPPORTED_WAVES)
    )
}

SC_MODES: dict[str, ModeSpec] = {
    spec.name: spec
    for spec in [
        ModeSpec(dedup=False, waves=w, sc_sender=True, sc_gather=g)
        for w, g in itertools.product(SUPPORTED_WAVES, (False, True))
    ] + [
        ModeSpec(
            dedup=True,
            waves=w,
            sc_sender=True,
            sc_gather=g,
            sc_forward=f,
        )
        for w, g, f in itertools.product(
            SUPPORTED_WAVES, (False, True), (False, True)
        )
    ]
}

ALL_MODES: dict[str, ModeSpec] = {**MODES, **SC_MODES}


def impl_compiler_options(impl: str) -> dict[str, str]:
  """XLA compile options an impl needs on TPU ({} for the TensorCore impl).

  The SparseCore impl (`product_scdma_cfg`) does not compile or run correctly
  without these; they cannot be set from inside a jitted function, so the
  caller passes them to `jax.jit(..., compiler_options=...)` (or
  `.lower(...).compile(compiler_options)`). Not applied in interpret mode.

  Args:
    impl: A ModeSpec impl (`product` or `product_scdma_cfg`).

  Returns:
    A fresh dict of option name -> value.
  """
  if impl.startswith("product_scdma"):
    return dict(_sc_sender().COMPILER_OPTIONS)
  return {}


def resolve_mode(
    name: str,
    *,
    ep: int,
    g_local: int,
    topology: topology_lib.BoardTopology | None,
) -> dict[str, Any]:
  """Resolves a mode name to `dedup_ep_moe` keyword arguments.

  A De-Dup mode's forward-table depth (dedup_fwd_max_runs) is left at its
  default; layer/plan.py sizes it statically. When `name` is a SparseCore
  mode, the returned dict also carries `sc_kwargs` for `sc_sender.wrap_layer`.

  Args:
    name: A mode name.
    ep: Expert-parallel width.
    g_local: Experts per device.
    topology: The board topology, or None.

  Returns:
    The keyword arguments (`waves`, `dedup_topology`, and optional `sc_kwargs`).

  Raises:
    SkipCaseError: If the mode cannot run on this shape or topology.
    ValueError: If the name is not a mode.
  """
  spec = parse_mode(name)
  if g_local % spec.waves:
    raise SkipCaseError(f"waves={spec.waves} does not divide g_local={g_local}")
  if spec.sc_sender and ep % _sc_sender().NUM_SUBCORES != 0:
    raise SkipCaseError(
        f"ep={ep} must be divisible by num_subcores={_sc_sender().NUM_SUBCORES}"
    )
  if spec.dedup:
    if topology is None:
      raise SkipCaseError(f"{name} needs a BoardTopology")
    if topology.num_devices != ep:
      raise SkipCaseError(
          f"the topology covers {topology.num_devices} devices, not ep={ep}"
      )
    if spec.waves > dedup_plan_mod.MAX_PTW_WAVES:
      raise SkipCaseError(
          f"De-Dup supports at most {dedup_plan_mod.MAX_PTW_WAVES} waves"
      )
  kw: dict[str, Any] = dict(
      waves=spec.waves, dedup_topology=topology if spec.dedup else None
  )
  if spec.sc_sender:
    kw["sc_kwargs"] = spec.sc_kwargs
  return kw
