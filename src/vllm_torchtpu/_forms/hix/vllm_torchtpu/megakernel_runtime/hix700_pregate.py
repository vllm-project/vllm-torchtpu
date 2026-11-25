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
"""q38 HIX700 PREGATE, served whole step: `VLLM_TPU_HIX_PREGATE=P` turns on the MoE block's predicted-expert prefetch
(`MoETunables.pregate = P`, in-kernel predictor, issued before the mixer all-reduce) for whole-step buckets of at most
`VLLM_TPU_HIX_PREGATE_MAX_ROWS` rows (default 16: the HIX 3-4 live shape). Both keys are registered in `envs` (so they
move the compile key); 0 / unset = off and the tiles are returned unchanged (the flag-off program is the base's).

Engagement on a STEP / served car (warm-cache safe, read in this order):
  1. the compiled whole-step artifact differs from the flag-off one (the kernel changes);
  2. the construction line below, logged once per process and bucket when `build_wholestep_call` builds the tiles
     (construction time, so it prints on a warm compile cache too);
  3. the kernel's own per-layer engagement output `moe_pg_eng` (issued / hit / n_hit per rank and layer), read by the
     MB-PGPF microbench and any harness that keeps the kernel's outputs.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGED: set = set()


def pregate_rows_cap() -> int:
    from vllm_torchtpu import envs

    return int(envs.VLLM_TPU_HIX_PREGATE_MAX_ROWS)


def pregate_count() -> int:
    from vllm_torchtpu import envs

    return int(envs.VLLM_TPU_HIX_PREGATE)


def pregate_tiles(t: Any, B: int) -> Any:
    """`t` with the MoE block's `pregate` set for this bucket, or `t` itself (off, or the bucket is above the cap)."""
    P = pregate_count()
    if P <= 0 or B > pregate_rows_cap():
        return t
    from vllm_torchtpu.megakernel.blocks.moe.params import pregate_refuse_ar0st

    pregate_refuse_ar0st(P)  # D-2: no composed path with HXSMALL AR0S-T yet -> refuse at construction (fail closed)
    out = t.with_blocks({"moe": {"pregate": P, "pregate_at": "ar"}})  # merges with the bucket's other MoE tunables
    if B not in _LOGGED:
        _LOGGED.add(B)
        logging.getLogger("vllm." + __name__).info(
            "q38 HIX700 PREGATE construct: whole-step bucket B=%d pregate=%d at=ar src=pred (predicted experts "
            "prefetched into the expert ring before routing; bitwise side-buffer replay)", B, P)
    return out


# q38 S3A-ENG T1 begin (PREGATE bind twin; HIX-LEAD-6 rulings 2026-11-24 items 3-5 and 9). `pregate_tiles` logs its construction line
# only when `build_wholestep_call` runs, i.e. when the whole step is TRACED; a warm compile cache never runs it. This twin is called
# from `Qwen38WholeStep.__init__` (every engine start, every rank) and logs through logger.info (never info_once). It never calls
# `pregate_tiles` (whose once-per-bucket set it would fill) and never calls `pregate_refuse_ar0st` (the AR0S-T refusal stays at build).
# q38 HIX-HMERGE T1 fix (F3 widened; HIX-LEAD-7 GO 2026-11-24): `_WholeKind.callable_for` hands build_wholestep_call
# `qwen38_wholestep.served_tiles(B, max_groups)`, never None, so build_wholestep_call's `pregate_tiles` branch never runs on the served
# path, for ANY VLLM_TPU_HIX_PREGATE. The twin used to log 'pregate=P' from the key alone (an engagement line reporting success while
# the served program had no PREGATE). It now reads the served tiles themselves: the bind line names only the buckets whose served tiles
# carry pregate > 0 (no line if none do), and the twin REFUSES (Qwen38PregateUnservedError; the driver's `_attach_whole` re-raises it, so
# engine start stops) when the key asks for pregate on a bucket whose served tiles do not carry it, when a served bucket carries pregate
# the key did not ask for, or when a key does not parse.
BIND_PREFIX = "q38 HIX700 PREGATE bind rank="
REFUSE_PREFIX = "q38 HIX700 PREGATE refused rank="
F3_CAUSE = "PREGATE requested but the served whole-step path never applies pregate tiles (F3)"


class Qwen38PregateUnservedError(RuntimeError):
    """q38 HIX-HMERGE T1 fix: VLLM_TPU_HIX_PREGATE asks for what the served whole-step tiles do not carry (or does not parse)."""


def pregate_served(buckets: Any, max_groups: int) -> dict:
    """{bucket: (pregate, pregate_at)} read from the tiles `_WholeKind.callable_for` hands build_wholestep_call (`served_tiles`)."""
    from .qwen38_wholestep import served_tiles

    out = {}
    for b in sorted({int(x) for x in buckets}):
        moe = served_tiles(b, max_groups).block("moe")
        out[b] = (int(moe.pregate), str(moe.pregate_at))
    return out


def pregate_bind_line(rank: int, buckets: Any, max_groups: int) -> str:
    """'q38 HIX700 PREGATE bind rank=<r>: pregate=<P> buckets=<b,...> at=<at>' for the buckets whose SERVED tiles carry pregate P > 0,
    logged once per call; "" (no line) when no served bucket carries pregate. Raises Qwen38PregateUnservedError (after a per-rank
    'q38 HIX700 PREGATE refused rank=<r>: ...' error line) when any bucket's served pregate differs from the requested one
    (VLLM_TPU_HIX_PREGATE on buckets <= VLLM_TPU_HIX_PREGATE_MAX_ROWS, 0 elsewhere), or when a key does not parse."""
    log = logging.getLogger("vllm." + __name__)
    try:
        P, cap = pregate_count(), pregate_rows_cap()
    except Exception as e:  # noqa: BLE001  a key that does not parse refuses: the served whole-step path never reads it (F3)
        msg = ("%s%d: VLLM_TPU_HIX_PREGATE / VLLM_TPU_HIX_PREGATE_MAX_ROWS does not parse (%s: %s); the served whole-step path never "
               "reads it (F3)" % (REFUSE_PREFIX, int(rank), type(e).__name__, e))
        log.error(msg)
        raise Qwen38PregateUnservedError(msg) from e
    served = pregate_served(buckets, max_groups)
    want = {b: (P if P > 0 and b <= cap else 0) for b in served}
    bad = [b for b in served if served[b][0] != want[b]]
    if bad:
        f3 = all(served[b][0] == 0 and want[b] > 0 for b in bad)
        cause = F3_CAUSE if f3 else "the served whole-step tiles do not carry the requested pregate"
        msg = "%s%d: %s: VLLM_TPU_HIX_PREGATE=%d VLLM_TPU_HIX_PREGATE_MAX_ROWS=%d; requested / served pregate per bucket %s" % (
            REFUSE_PREFIX, int(rank), cause, P, cap, " ".join("b%d=%d/%d" % (b, want[b], served[b][0]) for b in bad))
        log.error(msg)
        raise Qwen38PregateUnservedError(msg)
    bound = [b for b in served if served[b][0] > 0]
    if not bound:
        return ""
    msg = "%s%d: pregate=%s buckets=%s at=%s" % (BIND_PREFIX, int(rank), ",".join(sorted({str(served[b][0]) for b in bound})),
                                               ",".join(str(b) for b in bound), ",".join(sorted({served[b][1] for b in bound})))
    log.info(msg)
    return msg
# q38 S3A-ENG T1 end
