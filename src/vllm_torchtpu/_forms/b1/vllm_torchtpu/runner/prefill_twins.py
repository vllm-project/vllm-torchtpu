# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-WARMTWIN: per-rank, warm-cache-safe twins of the prefill QPF / QPREP engagement lines.

The QPF (VLLM_TPU_QSA_PREFILL_QPROJ_FILL) and QPREP (VLLM_TPU_QSA_PREFILL_QPREP_FUSED)
engagement lines print while the QSA ops are traced, so they are silent when the compile
cache serves a bucket. The warmup ladder (`TPUModelRunner._precompile_backbone`) runs
every bucket on every start, warm or cold, on every worker. After each bucket's dummy run
the runner calls `bucket`, which predicts, from the same functions the trace calls
(`qsa_qproj_fill_op.fill_slices`, `qsa_qprep_fused.decide`) and the static per-graph
inputs, and prints with plain logger.info on this rank (never a once-per-node form):

  q38 QPF-TWIN engaged|not engaged: one per (bucket, request-slot variant, QSA layer)
  q38 QPREP-TWIN engaged|not engaged: one per bucket
  q38 QPF-TWIN / QPREP-TWIN summary: one each per rank, from `finish`

The cold-start guard (MISMATCH) does not diff a per-bucket snapshot: a bucket is traced
whenever its graph compiles, which can be in the profile run or in an earlier bucket's
dummy run (shape-variant retrace / compile_all_ranges; M, Car K PP7 cold log). It reads
cumulative per-process trace records keyed by rows instead, which are complete for a
bucket once its own dummy run returns:

  qsa_qproj_fill_op._TRACED   every fill-op trace (0 slices included) with its inputs
  qsa_qproj_fill_op._LOGGED   the keys whose trace-time ENGAGED line printed
  qsa_qprep_fused._DECISIONS  every trace-time `applies` decision
  qsa_qprep_fused._ENGAGED_ROWS  traced fused kernels per rows
  tpu_compiler.range_traces   backend runs vs artifact replays of the bucket's range

A trace record that disagrees with the prediction, a predicted call with no record on a
fully traced bucket (backend runs, no replays), or an unexplained trace-time line prints a
MISMATCH line, which every reader fails on. A warm bucket (no record, no backend run)
checks nothing; a partly warm one (backend runs AND replays) checks the records it has,
skips the absence check and says so on a 'partly warm' line. Not covered: QSA layers
outside the runner's backbone model (the MTP drafter's, which compiles after
`_precompile_backbone`). With both keys unset nothing is read and nothing is printed.
Log only: nothing here enters a program.
"""

import os
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

QPF_LINE = (
    "q38 QPF-TWIN %s: rank=%d pp_rank=%d pid=%d bucket=%d slots=%d mql=%d chunk=%d "
    "layer=%s slices=%d groups=%d..%d precision=%s"
)
QPF_SUMMARY = (
    "q38 QPF-TWIN summary: rank=%d engaged_buckets=%s lines=%d traced=%d mismatch=%d"
)
QPF_MISMATCH = (
    "q38 QPF-TWIN MISMATCH: rank=%d bucket=%d layer=%s predicted=%s traced=%s"
)
QPREP_LINE = (
    "q38 QPREP-TWIN %s: rank=%d pp_rank=%d pid=%d bucket=%d qsa_layers=%d heads=%d "
    "head_dim=%d rotary=%d reason=%s"
)
QPREP_SUMMARY = (
    "q38 QPREP-TWIN summary: rank=%d engaged_buckets=%s lines=%d traced=%d mismatch=%d"
)
QPREP_MISMATCH = "q38 QPREP-TWIN MISMATCH: rank=%d bucket=%d predicted=%s traced=%s"
PARTLY_WARM = (
    "q38 %s-TWIN partly warm: rank=%d bucket=%d backend_runs=%d replays=%d "
    "absence_check=skipped"
)


def rank_fields() -> tuple[int, int, int]:
    """(rank, pp_rank, pid) of this worker process; -1 where unknown."""
    rank = pp_rank = -1
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = int(dist.get_rank())
    except Exception:  # noqa: BLE001 - a log field
        pass
    try:
        from vllm.distributed.parallel_state import get_pp_group

        pp_rank = int(get_pp_group().rank_in_group)
    except Exception:  # noqa: BLE001 - a log field
        pass
    return rank, pp_rank, os.getpid()


def backbone_qsa_layers(runner: Any) -> list:
    """The QSA layers of the runner's backbone model on this stage, in module order
    (the MTP drafter's model is not the runner's `model`)."""
    model = getattr(runner, "model", None)
    if model is None:
        return []
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    return [m for _, m in model.named_modules() if isinstance(m, Qwen4ExpQSA)]


def sequence_parallel_bucket(runner: Any, rows: int) -> bool:
    """Whether the bucket runs Qwen4Exp's sequence-parallel backbone (a separate
    compiled graph over the same layers, routed by bucket_filter)."""
    sp_module = getattr(getattr(runner, "model", None), "model_sp", None)
    if sp_module is None:
        return False
    from vllm_torchtpu.compilation import bucket_filter

    serves = bucket_filter.assigned(sp_module)
    return bool(serves(rows)) if serves is not None else False


def forward_calls_fill(layer: Any, sequence_parallel: bool) -> bool:
    """Whether `Qwen4ExpQSA.forward_with_indices` (rows=None, shared=None: the
    backbone step) calls the layer's fill op: `op is self._op`, no group q/gate, no
    shared selection, not a single-graph (MTP) module."""
    if getattr(layer, "_qproj_fill", None) is None or getattr(layer, "single_graph", False):
        return False
    replicated_indexer = sequence_parallel and bool(envs.VLLM_TPU_QSA_INDEXER_SHARD_SKIP_SP)
    token_shard = sequence_parallel and bool(envs.VLLM_TPU_QSA_INDEXER_TOKEN_SHARD)
    if (
        getattr(layer, "pooled", False)
        and getattr(layer, "_shard_op", None) is not None
        and not replicated_indexer
    ):
        return False
    if token_shard and getattr(layer, "_token_shard_op", None) is not None:
        return False
    if getattr(layer, "_dcp_op", None) is not None:
        return False
    if (
        getattr(layer, "_selection_source", None) is not None
        and getattr(layer, "_shared_ops", None) is not None
    ):
        return False
    return True


@dataclass(frozen=True)
class QpfPrediction:
    layer: str
    rows: int
    calls_fill: bool  # the forward calls the fill op here (it traces even at 0 slices)
    slots: int
    mql: int
    chunk: int
    group_base: int
    precision: str
    slices: int

    @property
    def key(self) -> tuple:
        """The fill op's trace record key (qsa_qproj_fill_op._TRACED)."""
        return (self.rows, self.slots, self.mql, self.chunk, self.group_base, self.precision)


def predict_qpf(layer: Any, rows: int, slots: int, sequence_parallel: bool = False) -> QpfPrediction:
    """The fill op's decision for this layer at a bucket of `rows` tokens over `slots`
    request slots (`context_lens`' static length = the dummy run's num_reqs)."""
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_qproj_fill_op as qpf_op

    prefix = str(getattr(layer, "prefix", "?"))
    mql = int(getattr(layer, "max_query_len", 1))
    fill = getattr(layer, "_qproj_fill", None)
    if fill is None:
        return QpfPrediction(prefix, int(rows), False, int(slots), mql, -1, -1, "none", 0)
    kw = fill.keywords
    chunk, group_base, precision = int(kw["chunk_override"]), int(kw["group_base"]), str(kw["precision"])
    calls = forward_calls_fill(layer, sequence_parallel)
    slices = qpf_op.fill_slices(rows, slots, mql, chunk) if calls else 0
    return QpfPrediction(prefix, int(rows), calls, int(slots), mql, chunk, group_base, precision, int(slices))


def qsa_geometry(layer: Any) -> tuple[int, int, int]:
    """(heads, head_dim, rotary_dim) that `qsa.produce` sees for this layer's step:
    the TP-local query heads of `q_proj`, the query norm width and the op's rotary dim."""
    op = getattr(layer, "_op", None)
    rotary = int(op.keywords["rotary_dim"]) if op is not None and "rotary_dim" in getattr(op, "keywords", {}) else -1
    return int(layer.num_heads), int(layer.head_dim), rotary


def predict_qprep(layers: list, rows: int) -> tuple[bool, str, tuple[int, int, int]]:
    """(engaged, reason, (heads, head_dim, rotary)) of QPREP-FUSED at a bucket."""
    from vllm_torchtpu.kernels.qwen4_exp import qsa_qprep_fused as qprep

    if not layers:
        return False, "no_qsa_layer", (0, 0, 0)
    geoms = sorted({qsa_geometry(layer) for layer in layers})
    if len(geoms) != 1:
        return False, "qsa_geometries_differ", geoms[0]
    reason = qprep.decide(rows, *geoms[0])
    return reason is None, "applies" if reason is None else reason, geoms[0]


def _fmt_qpf(p: QpfPrediction | None, slices: int | None = None, key: tuple | None = None) -> str:
    if key is not None:
        rows, slots, mql, chunk, gb, prec = key
        return f"slices:{slices},slots:{slots},mql:{mql},chunk:{chunk},group_base:{gb},precision:{prec}"
    if p is None:
        return "none"
    if not p.calls_fill:
        return "no_fill_call"
    return (
        f"slices:{p.slices},slots:{p.slots},mql:{p.mql},chunk:{p.chunk},"
        f"group_base:{p.group_base},precision:{p.precision}"
    )


class Twins:
    """One rank's twins over one warmup ladder."""

    def __init__(self, runner: Any, layers: list | None, qpf: bool, qprep: bool):
        self.runner = runner
        self.qpf = qpf
        self.qprep = qprep
        self.layers = backbone_qsa_layers(runner) if layers is None else list(layers)
        self.rank, self.pp_rank, self.pid = rank_fields()
        variants = [getattr(runner, "num_reqs_max_model_len", None)]
        if getattr(runner, "most_model_len", None) is not None:
            variants.append(getattr(runner, "num_reqs_most_model_len", None))
        self.variant_slots = [int(v) for v in variants if v is not None]
        self.buckets: list[int] = []
        self.qpf_predicted: set = set()  # (rows, slices, group_base, precision) predicted engaged
        self.counts = {
            "QPF": {"engaged": set(), "lines": 0, "traced": 0, "mismatch": 0},
            "QPREP": {"engaged": set(), "lines": 0, "traced": 0, "mismatch": 0},
        }
        self.qprep_done: set = set()

    # -- trace status of a bucket's range
    @staticmethod
    def _range(rows: int) -> tuple[int, int]:
        from vllm_torchtpu.compilation import tpu_compiler

        return tpu_compiler.range_traces(rows)

    def bucket(self, rows: int, slots: int, variant: str) -> None:
        if rows not in self.buckets:
            self.buckets.append(rows)
        backend, replays = self._range(rows)
        traced_full = backend > 0 and replays == 0
        partly = backend > 0 and replays > 0
        if self.qpf:
            self._qpf_bucket(rows, slots, traced_full, partly, backend, replays)
        if self.qprep and rows not in self.qprep_done:
            self.qprep_done.add(rows)
            self._qprep_bucket(rows, traced_full, partly, backend, replays)

    def _qpf_bucket(self, rows, slots, traced_full, partly, backend, replays) -> None:
        from vllm_torchtpu.layers.adapter.custom_ops import qsa_qproj_fill_op as qpf_op

        c = self.counts["QPF"]
        sp = sequence_parallel_bucket(self.runner, rows)
        records = dict(qpf_op._TRACED)
        logged = set(qpf_op._LOGGED)
        if partly:
            logger.info(PARTLY_WARM, "QPF", self.rank, rows, backend, replays)
        for layer in self.layers:
            p = predict_qpf(layer, rows, slots, sp)
            engaged = p.calls_fill and p.slices >= 1
            logger.info(
                QPF_LINE, "engaged" if engaged else "not engaged", self.rank, self.pp_rank,
                self.pid, rows, p.slots, p.mql, p.chunk, p.layer, p.slices, p.group_base,
                p.group_base + p.slices - 1, p.precision,
            )
            c["lines"] += 1
            if engaged:
                c["engaged"].add(rows)
                self.qpf_predicted.add((rows, p.slices, p.group_base, p.precision))
            if p.group_base < 0:
                continue  # no fill op on this layer: nothing is ever traced for it
            want = p.key
            mine = {k: v for k, v in records.items() if k[0] == rows and k[4] == p.group_base}
            expected = {
                (rows, s, p.mql, p.chunk, p.group_base, p.precision) for s in self.variant_slots
            } | {want}
            bad = None
            if want in mine:
                c["traced"] += 1
                if not p.calls_fill or mine[want] != p.slices or (mine[want] >= 1) != engaged:
                    bad = _fmt_qpf(None, mine[want], want)
                elif mine[want] >= 1 and (rows, mine[want], p.group_base, p.precision) not in logged:
                    bad = _fmt_qpf(None, mine[want], want) + ",trace_line:absent"
            else:
                stray = sorted(k for k in mine if k not in expected)
                if stray:
                    c["traced"] += 1
                    bad = _fmt_qpf(None, mine[stray[0]], stray[0])
                elif traced_full and p.calls_fill:
                    # Predicted a fill call on a fully traced bucket, and this variant's
                    # graph left no record (another variant's, if any, is named).
                    c["traced"] += 1
                    other = sorted(mine)
                    bad = "none" if not other else "none;other:" + _fmt_qpf(None, mine[other[0]], other[0])
                elif traced_full:
                    c["traced"] += 1
            if bad is not None:
                logger.info(QPF_MISMATCH, self.rank, rows, p.layer, _fmt_qpf(p), bad)
                c["mismatch"] += 1

    def _qprep_bucket(self, rows, traced_full, partly, backend, replays) -> None:
        from vllm_torchtpu.kernels.qwen4_exp import qsa_qprep_fused as qprep

        c = self.counts["QPREP"]
        engaged, reason, (heads, head_dim, rotary) = predict_qprep(self.layers, rows)
        if partly:
            logger.info(PARTLY_WARM, "QPREP", self.rank, rows, backend, replays)
        logger.info(
            QPREP_LINE, "engaged" if engaged else "not engaged", self.rank, self.pp_rank,
            self.pid, rows, len(self.layers), heads, head_dim, rotary, reason,
        )
        c["lines"] += 1
        if engaged:
            c["engaged"].add(rows)
        if not self.layers:
            return
        want = (rows, heads, head_dim, rotary)
        decisions = {k: v for k, v in dict(qprep._DECISIONS).items() if k[0] == rows}
        kernels = int(qprep._ENGAGED_ROWS.get(rows, 0))
        predicted = "fused" if engaged else f"reference({reason})"
        bad = None
        if want in decisions:
            c["traced"] += 1
            got = decisions[want]
            if (got is None) != engaged or (got is not None and got != reason):
                bad = "fused" if got is None else f"reference({got})"
            elif engaged and kernels < 1:
                bad = "fused,kernel_traced:0"
            elif not engaged and kernels > 0:
                bad = f"reference({got}),kernel_traced:{kernels}"
            stray = sorted(k for k in decisions if k != want)
            if bad is None and stray:
                bad = f"geometry:{stray[0][1]}x{stray[0][2]}r{stray[0][3]}"
        elif decisions:
            c["traced"] += 1
            k = sorted(decisions)[0]
            bad = f"geometry:{k[1]}x{k[2]}r{k[3]}"
        elif kernels > 0:
            c["traced"] += 1
            bad = f"no_decision,kernel_traced:{kernels}"
        elif traced_full:
            c["traced"] += 1
            bad = "none"
        if bad is not None:
            logger.info(QPREP_MISMATCH, self.rank, rows, predicted, bad)
            c["mismatch"] += 1

    def finish(self) -> dict:
        out = {}
        if self.qpf:
            from vllm_torchtpu.layers.adapter.custom_ops import qsa_qproj_fill_op as qpf_op

            c = self.counts["QPF"]
            ladder = set(self.buckets)
            by_base = {
                int(layer._qproj_fill.keywords["group_base"]): str(getattr(layer, "prefix", "?"))
                for layer in self.layers
                if getattr(layer, "_qproj_fill", None) is not None
            }
            for key in sorted(k for k in set(qpf_op._LOGGED) if k[0] in ladder):
                if key not in self.qpf_predicted:
                    rows, slices, gb, prec = key
                    logger.info(
                        QPF_MISMATCH, self.rank, rows, by_base.get(gb, "?"), "none",
                        f"trace_line:slices:{slices},group_base:{gb},precision:{prec}",
                    )
                    c["mismatch"] += 1
            logger.info(
                QPF_SUMMARY, self.rank, _buckets(c["engaged"]), c["lines"], c["traced"],
                c["mismatch"],
            )
            out["QPF"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in c.items()}
        if self.qprep:
            c = self.counts["QPREP"]
            logger.info(
                QPREP_SUMMARY, self.rank, _buckets(c["engaged"]), c["lines"], c["traced"],
                c["mismatch"],
            )
            out["QPREP"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in c.items()}
        return out


def _buckets(rows: set) -> str:
    return ",".join(str(r) for r in sorted(rows)) or "-"


def start(runner: Any, layers: list | None = None) -> Twins | None:
    """The twins of one warmup ladder, or None (reading nothing else) when both
    VLLM_TPU_QSA_PREFILL_QPROJ_FILL and VLLM_TPU_QSA_PREFILL_QPREP_FUSED are unset."""
    qpf, qprep = bool(envs.VLLM_TPU_QSA_PREFILL_QPROJ_FILL), bool(envs.VLLM_TPU_QSA_PREFILL_QPREP_FUSED)
    if not (qpf or qprep):
        return None
    return Twins(runner, layers, qpf, qprep)


def bucket(twins: Twins | None, rows: int, slots: int, variant: str) -> None:
    """After one dummy run of the ladder: this bucket's lines (and MISMATCH checks)."""
    if twins is not None:
        twins.bucket(int(rows), int(slots), variant)


def finish(twins: Twins | None) -> dict | None:
    """After the ladder: the unexplained trace-line check and the per-rank summaries."""
    return None if twins is None else twins.finish()
