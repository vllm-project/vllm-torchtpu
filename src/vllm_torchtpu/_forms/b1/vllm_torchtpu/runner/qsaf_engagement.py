# SPDX-License-Identifier: Apache-2.0
"""q38 QSAF-INT: per-rank, warm-cache-safe engagement lines of the QSAF fused-union route.

``VLLM_TPU_QSA_DECODE_FUSED_UNION`` routes ``qsa_rpa.union_decode_attention`` to the fused
kernel at TRACE time, so its 'traced' line exists only on a cold compile.
``compilation/qsaf_record.py`` keeps the route's trace keys as a record next to every
compiled artifact (with the cache hash and a checksum) and registers the record of every
artifact this process used: compiled and saved, loaded from the compile cache, or replayed
from a peer rank (pattern: IDXXLINE, runner/idxx_engagement.py, credited).

After the warmup ladder (``TPUModelRunner._precompile_backbone``, every start, every rank)
``report`` prints, per rank, with plain logger.info (never a once-per-node form):

  MISMATCH     a used artifact's record failed its checks
  ENGAGED      one line per distinct trace key in the UNION of the used verified records:
               'q38 QSAF FUSED_UNION engaged: rank R tp_rank T key rows=..,slots=..,covered=..,
               issue=.. program <cache hash>:<artifact> source <compiled|loaded|peer|recompiled>
               ...'. Fail-closed: when any used artifact's record is missing or failed its
               checks, or no artifact was used, this rank prints NO engaged line and one
               NOT_ENGAGED line with the reason.
  NOT_ENGAGED  the key is on and nothing on this rank vouches for an engaged trace
  SUMMARY      one per rank

Label: the record says what was TRACED while this cache hash was built (cumulative per
process, as IDXXLINE); 'program' names the first used artifact (by path) whose verified
record carries the key, and 'programs' how many do. That the device runs the fused kernel
is inferred from the compile key; the STEP car's per-rank xplane census is the device check.

With the key off this module reads nothing else and prints nothing.
Log only: nothing here enters a program.
"""

from typing import Any

from vllm_torchtpu.compilation import qsaf_record
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENGAGED = (
    "q38 QSAF FUSED_UNION engaged: rank %d tp_rank %d key %s program %s source %s "
    "dp_rank %d pid %d programs %d records %d"
)
NOT_ENGAGED = (
    "q38 QSAF FUSED_UNION not engaged: rank %d tp_rank %d dp_rank %d pid %d cache %s "
    "programs %d records_ok %d reason %s"
)
MISMATCH = (
    "q38 QSAF FUSED_UNION record MISMATCH: rank %d tp_rank %d dp_rank %d pid %d cache %s "
    "program %s reason %s"
)
SUMMARY = (
    "q38 QSAF FUSED_UNION summary: rank %d tp_rank %d dp_rank %d pid %d cache %s "
    "programs %d records_ok %d missing %d mismatched %d keys %d engaged %d"
)


def key_text(e: dict) -> str:
    return f"rows={e['rows']},slots={e['slots']},covered={e['covered']},issue={e['issue']}"


def _rank_fields(runner: Any) -> tuple[int, int, int, int]:
    from vllm_torchtpu.runner.g1_engagement import rank_fields

    return rank_fields(runner)


def report(runner: Any, *, ids: tuple | None = None) -> dict | None:
    """Print this rank's QSAF lines after the warmup ladder. Silent (returns None, reads
    nothing) with the key off. ``ids`` (rank, tp_rank, dp_rank, pid) override the
    runner's (tests)."""
    if not qsaf_record.active():
        return None
    rank, tp_rank, dp_rank, pid = ids if ids is not None else _rank_fields(runner)
    used = qsaf_record.used()
    ok = [u for u in used if u["status"] == "ok"]
    missing = [u for u in used if u["status"] == "missing"]
    bad = [u for u in used if u["status"] == "mismatch"]
    cache = ",".join(sorted({str(u.get("cache_hash")) for u in used})) or "none"
    for u in bad:
        logger.info(MISMATCH, rank, tp_rank, dp_rank, pid, u.get("cache_hash"),
                    u["artifact"], u["reason"])
    union: dict[str, dict] = {}
    for u in sorted(ok, key=lambda x: x["artifact"]):
        for e in u["entries"]:
            k = key_text(e)
            slot = union.setdefault(k, {"entry": e, "first": u, "programs": 0})
            slot["programs"] += 1
    if not used:
        reason = "no compiled, loaded or peer artifact registered on this rank"
    elif missing or bad:
        reason = (f"no verified trace record for {len(missing)} missing / {len(bad)} "
                  f"mismatched program(s) of cache {cache}")
    elif not union:
        reason = f"no fused-union trace in the records of the {len(ok)} program(s) used"
    else:
        reason = None
    engaged = []
    if reason is None:
        for k in sorted(union):
            u = union[k]["first"]
            program = f"{u.get('cache_hash')}:{u['artifact'].rsplit('/', 1)[-1]}"
            logger.info(ENGAGED, rank, tp_rank, k, program, u["source"], dp_rank, pid,
                        union[k]["programs"], len(ok))
            engaged.append(k)
    else:
        logger.info(NOT_ENGAGED, rank, tp_rank, dp_rank, pid, cache, len(used), len(ok),
                    reason)
    logger.info(SUMMARY, rank, tp_rank, dp_rank, pid, cache, len(used), len(ok),
                len(missing), len(bad), len(union), len(engaged))
    return {"engaged": engaged, "reason": reason, "used": len(used), "ok": len(ok),
            "missing": len(missing), "mismatched": len(bad), "keys": len(union)}
