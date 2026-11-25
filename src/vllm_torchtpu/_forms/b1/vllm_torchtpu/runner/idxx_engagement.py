# SPDX-License-Identifier: Apache-2.0
"""q38 IDXXLINE: per-rank, warm-cache-safe engagement lines of the IDXX members.

The IDXX members (VLLM_TPU_QSA_INDEXER_KEY_LANES, VLLM_TPU_QSA_DCP_SC_BOUNDED_COPY,
VLLM_TPU_QSA_DCP_FUSED_MERGE) decide per call shape while the QSA ops are traced, so
their trace-time lines are absent on a warm start. `compilation/idxx_record.py` keeps
those decisions as a record next to every compiled artifact (with the cache hash and a
checksum) and registers the record of every artifact this process used: compiled and
saved, loaded from the compile cache, or replayed from a peer rank.

After the warmup ladder (`TPUModelRunner._precompile_backbone`, every start, every rank)
`report` prints, per rank, with plain logger.info (never a once-per-node form):

  FORM         the form the runtime facts select from the pre-stated table, and its
               decisive buckets
  MISMATCH     a used artifact's record failed its checks (checksum, cache hash, artifact,
               env, or an entry its member's rule contradicts)
  ENGAGED / NOT_ENGAGED
               per decisive bucket and member: the bucket's call shape (pre-stated table)
               looked up in the UNION of the used records. Fail-closed: a shape absent
               from the union is 'reason=no record for shape'; a used artifact whose record
               is missing or failed its checks makes every line of this rank
               'reason=no trace record for cache <hash>'.
  SUMMARY      one per rank

Label: "engaged at bucket B" is inferred by determinism -- shape S engaged while this
cache hash was built (the record), and bucket B makes call shape S (the table). The
record says what was TRACED for this cache hash, not what the device ran; the SB1-1
STEP car's per-rank xplane census is the one-time device check.

With every IDXX key unset this module reads nothing else and prints nothing.
Log only: nothing here enters a program.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from vllm_torchtpu.compilation import idxx_record
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

FORM = (
    "q38 IDXX form: rank=%d tp_rank=%d dp_rank=%d pid=%d form=%s table=%s k=%s "
    "row_shard=%s row_shard_min_rows=%s decisive=%s"
)
ENGAGED = (
    "q38 IDXX %s engaged: rank=%d tp_rank=%d dp_rank=%d pid=%d form=%s bucket=%d "
    "shape=%s path=%s cache=%s records=%d"
)
NOT_ENGAGED = (
    "q38 IDXX %s not engaged: rank=%d tp_rank=%d dp_rank=%d pid=%d form=%s bucket=%d "
    "shape=%s cache=%s records=%d reason=%s"
)
MISMATCH = (
    "q38 IDXX record MISMATCH: rank=%d tp_rank=%d dp_rank=%d pid=%d cache=%s "
    "artifact=%s reason=%s"
)
SUMMARY = (
    "q38 IDXX summary: rank=%d tp_rank=%d dp_rank=%d pid=%d form=%s cache=%s "
    "artifacts=%d records_ok=%d missing=%d mismatched=%d entries=%d engaged=%d "
    "not_engaged=%d"
)

# The pre-stated bucket -> call-shape table. Per form: the runtime facts that select it,
# whether its rows are DECISIVE or DESCRIPTIVE, and per bucket and member the call shape
# (a subset of the record entry's shape) the bucket's trace makes. Derivation:
#   B1 (Car K c212: k2, VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD=1, min rows 128; TP8, DCP 4):
#     288 -> select rows 144: M, cold Car K log e_k46gj 'q38 BLSTACK-IDX compressed-DCP
#            row-shard select traced: tokens 288, slots 128, shard rows 144 (2 shards)'.
#     SC b = select rows: C, pooled_qsa._select_blocks chunk = min(rows,
#            decode_query_chunk_size(...)) = rows at TPU7x SMEM (512, I) and _dcp_topk
#            hands the chunk's rows to the SparseCore top-k.
#     FUSED_MERGE rows = bucket: C, qsa_op _gathered_lists brings every pair's lists
#            (all 288 rows) to compressed_dcp_own_blocks / compressed_dcp_blocks.
#   B3 rows: C only, DESCRIPTIVE until a B3 stack carries IDXX and a cold B3 log confirms
#     them (DSTEP-LEAD-6 ruling 2026-11-24). B3-RS96: row shard min rows 96 (b3pkgq2 RSB3).
# The first cold device log with the IDXX trace lines confirms the table on device; a
# disagreement voids before any served read.
FORMS: tuple[tuple[str, dict, str, dict], ...] = (
    (
        "B1",
        {"k": 2, "row_shard": True, "row_shard_min_rows": 128},
        "decisive",
        {
            288: {
                "KEY_LANES": {"rows": 144, "decode": True},
                "SC_BOUNDED_COPY": {"tag": "dcp", "b": 144},
                "FUSED_MERGE": {"rows": 288},
            },
        },
    ),
    (
        "B3",
        {"k": 3, "row_shard": False},
        "descriptive",
        {
            r: {
                "KEY_LANES": {"rows": r, "decode": True},
                "SC_BOUNDED_COPY": {"tag": "dcp", "b": r},
                "FUSED_MERGE": {"rows": r},
            }
            for r in (64, 96, 112)
        },
    ),
    (
        "B3-RS96",
        {"k": 3, "row_shard": True, "row_shard_min_rows": 96},
        "descriptive",
        {
            r: {
                "KEY_LANES": {"rows": s, "decode": True},
                "SC_BOUNDED_COPY": {"tag": "dcp", "b": s},
                "FUSED_MERGE": {"rows": r},
            }
            for r, s in ((64, 64), (96, 48), (112, 56))
        },
    ),
)


def runtime_facts(runner: Any) -> dict:
    """The facts the table's forms are selected by: speculative k and the compressed-DCP
    row shard (registered env keys; read, never set)."""
    from vllm_torchtpu import envs

    k = None
    try:
        spec = runner.vllm_config.speculative_config
        k = int(spec.num_speculative_tokens) if spec is not None else 0
    except Exception:  # noqa: BLE001 - an unknown fact selects no form
        pass
    return {
        "k": k,
        "row_shard": bool(envs.VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD),
        "row_shard_min_rows": int(envs.VLLM_TPU_QSA_COMPRESSED_DCP_ROW_SHARD_MIN_ROWS),
    }


def select_form(facts: dict) -> tuple[str, dict, str, dict] | None:
    for form in FORMS:
        if all(facts.get(name) == value for name, value in form[1].items()):
            return form
    return None


def _fmt(shape: dict) -> str:
    def v(x):
        return str(int(x)) if isinstance(x, bool) else str(x)

    return ",".join(f"{name}={v(shape[name])}" for name in sorted(shape))


def _rank_fields(runner: Any) -> tuple[int, int, int, int]:
    from vllm_torchtpu.runner.g1_engagement import rank_fields

    return rank_fields(runner)


def report(
    runner: Any,
    buckets: Iterable[int],
    *,
    facts: dict | None = None,
    ids: Sequence[int] | None = None,
) -> dict | None:
    """Print this rank's IDXX lines after the warmup ladder. ``buckets``: the token
    buckets the ladder ran. Silent (returns None, reads nothing) with no IDXX key set.
    ``facts`` / ``ids`` (rank, tp_rank, dp_rank, pid) override the runner's (tests)."""
    if not idxx_record.active():
        return None
    rank, tp_rank, dp_rank, pid = ids if ids is not None else _rank_fields(runner)
    facts = facts if facts is not None else runtime_facts(runner)
    form = select_form(facts)
    used = idxx_record.used()
    ok = [u for u in used if u["status"] == "ok"]
    missing = [u for u in used if u["status"] == "missing"]
    bad = [u for u in used if u["status"] == "mismatch"]
    hashes = sorted({str(u.get("cache_hash")) for u in used}) or ["none"]
    cache = ",".join(hashes)
    for u in bad:
        logger.info(MISMATCH, rank, tp_rank, dp_rank, pid, u.get("cache_hash"),
                    u["artifact"], u["reason"])
    union: dict[str, dict] = {}
    for u in ok:
        for e in u["entries"]:
            union[idxx_record._canon(e)] = e
    entries = list(union.values())
    # Strict: one used artifact without a good record and nothing on this rank vouches.
    no_record = not ok or bool(missing) or bool(bad)
    name = form[0] if form else "none"
    table = form[2] if form else "none"
    rows = form[3] if form else {}
    logger.info(FORM, rank, tp_rank, dp_rank, pid, name, table, facts.get("k"),
                facts.get("row_shard"), facts.get("row_shard_min_rows"),
                ",".join(str(b) for b in sorted(rows)) or "none")
    compiled = set(int(b) for b in buckets)
    n_eng = n_not = 0
    lines = []
    for bucket in sorted(rows):
        for member in idxx_record.MEMBERS:
            want = rows[bucket][member]
            matched = [
                e for e in entries
                if e["member"] == member
                and all(e["shape"].get(f) == v for f, v in want.items())
            ]
            if no_record:
                reason = f"no trace record for cache {cache}"
            elif bucket not in compiled:
                reason = "bucket not in this start's warmup ladder"
            elif not matched:
                reason = "no record for shape"
            elif not all(e["engaged"] for e in matched):
                first = next(e for e in matched if not e["engaged"])
                reason = f"recorded {first['path']}: {first['reason']}"
            else:
                reason = None
            shape = "|".join(sorted(_fmt(e["shape"]) for e in matched)) or _fmt(want)
            if reason is None:
                path = "|".join(sorted({e["path"] for e in matched}))
                logger.info(ENGAGED, member, rank, tp_rank, dp_rank, pid, name, bucket,
                            shape, path, cache, len(ok))
                n_eng += 1
                lines.append((member, bucket, True, None))
            else:
                logger.info(NOT_ENGAGED, member, rank, tp_rank, dp_rank, pid, name,
                            bucket, shape, cache, len(ok), reason)
                n_not += 1
                lines.append((member, bucket, False, reason))
    logger.info(SUMMARY, rank, tp_rank, dp_rank, pid, name, cache, len(used), len(ok),
                len(missing), len(bad), len(entries), n_eng, n_not)
    return {"form": name, "lines": lines, "used": len(used), "ok": len(ok),
            "missing": len(missing), "mismatched": len(bad), "entries": len(entries)}
