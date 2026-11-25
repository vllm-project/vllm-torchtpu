# SPDX-License-Identifier: Apache-2.0
"""q38 HIX700 KV-SCALE-GUARD (lead ruling after U1, 2026-11-24; always on, no env key: host-side only, it changes no traced program).

The whole step's in-kernel QSA attention (megakernel/wholestep/union_attn.py) dequantizes the FP8 K / V BEFORE the dots: key = k8 * scale_k,
value = v8 * scale_v in f32, and Mosaic pushes that f32 right-hand operand to the MXU as bf16 in ONE pass (U1B j-9b68df9c, deviceless tpu7x
LLO: 32 vmatpush1.bf16 for an f32 x f32 dot at default precision, the same as bf16 x bf16). Stock RPA instead pushes the FP8 K / V exactly
and scales after the dot. The two agree bit for bit on the operand only when bf16(scale x e4m3) == scale x e4m3 for every e4m3 code, i.e.
when the f32 scale's significand has <= 4 bits (e4m3 carries 4 significant bits, bf16 8). The served checkpoint has no KV scale tensors, so
both scales are vLLM's default 1.0 (1 bit) and the rounding is exactly zero (U1 (i)); a checkpoint that ships other scales would silently
reintroduce a relative error of up to 2^-8 per K / V element. This guard makes that fail loudly at bind time.

check_kv_scales(k, v, rank, where, site): both scales must be finite, positive and have a float32 significand of <= MAX_SIG_BITS (4) bits, else
Qwen38KVScaleGuardError (v2, a ValueError subclass; the whole-step attach re-raises it, so engine start fails). The first check in a process logs ONE line per rank (logger.info, never info_once: the per-rank engagement rule), which a car's
reader can require from every rank:
  q38 KV-SCALE-GUARD rank <r>: k_scale <k> (<n> significand bits) v_scale <v> (<m> significand bits) <= 4: in-kernel bf16 K / V operand exact
"""

from __future__ import annotations

import math
import struct

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

MAX_SIG_BITS = 4
TAG = "q38 KV-SCALE-GUARD"
_LOGGED: set[str] = set()  # v2: one logged line per SITE per process (wholestep, draft)


class Qwen38KVScaleGuardError(ValueError):
    """v2 (lead ruling (b), hard fail): a KV scale pair the in-kernel QSA attention cannot use exactly. qwen38_driver._attach_whole
    re-raises it (no segment-program fallback) and the MTP draft-attention attach does not catch it, so engine start fails."""


def significand_bits(x: float) -> int:
    """Significant bits of `x` rounded to float32 (the kernel's qsa_scales dtype): 1.0 / 0.5 / 2.0 -> 1, 0.375 -> 2, 2.5 -> 3, 1.1 -> 24."""
    f = struct.unpack("<f", struct.pack("<f", float(x)))[0]
    if not math.isfinite(f) or f <= 0:
        raise Qwen38KVScaleGuardError("%s: scale %r is not a finite positive float32" % (TAG, x))
    m, _ = math.frexp(f)  # m in [0.5, 1)
    for b in range(1, 25):
        if float(m * (1 << b)).is_integer():
            return b
    return 24


def check_kv_scales(k_scale: float, v_scale: float, *, rank: int = -1, where: str = "", site: str = "wholestep") -> tuple[int, int]:
    """Refuse a KV scale pair whose in-kernel dequantized operand would not be exact in bf16 (Qwen38KVScaleGuardError); log the pair
    once per process per site (the whole step's line keeps its v1 text; other sites carry ' [<site>]' after the tag)."""
    at = (" " + where) if where else ""
    for nm, x in (("k_scale", k_scale), ("v_scale", v_scale)):
        f = struct.unpack("<f", struct.pack("<f", float(x)))[0]
        if not math.isfinite(f) or f <= 0:
            raise Qwen38KVScaleGuardError("%s%s: %s %r is not a finite positive float32" % (TAG, at, nm, x))
    kb, vb = significand_bits(k_scale), significand_bits(v_scale)
    if kb > MAX_SIG_BITS or vb > MAX_SIG_BITS:
        raise Qwen38KVScaleGuardError(
            "%s%s: k_scale %r (%d significand bits) / v_scale %r (%d bits): more than %d bits, so bf16(scale x e4m3) is inexact in the "
            "in-kernel QSA attention (union_attn.py scales K / V before the dot; one bf16 MXU pass). Use stock-order scaling or exact scales."
            % (TAG, at, k_scale, kb, v_scale, vb, MAX_SIG_BITS))
    if site not in _LOGGED:
        _LOGGED.add(site)
        logger.info("%s%s rank %d: k_scale %r (%d significand bits) v_scale %r (%d significand bits) <= %d: in-kernel bf16 K / V operand exact",
                    TAG, "" if site == "wholestep" else " [%s]" % site, rank, float(k_scale), kb, float(v_scale), vb, MAX_SIG_BITS)
    return kb, vb
