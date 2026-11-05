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
"""Kimi-K3.1 prefill DSA selection microbenchmarks over the stage mesh
(``stage dsa-micro``; no weights).

    stage dsa-micro --prompts 8192x1 [--cases score-loop,select-tc,...]
        [--rows 1024,2048,8192] [--candidates 8192,16384,32768,147456]
        [--repeats 20]

Each case is a JAX body run per core under ``shard_map`` over a stage's
cores (every stage at once), each core on its own ``R`` query rows against
``C`` candidates, so the cores share their chips' SparseCores as the serving
stages do. Inputs are random with a fixed seed per core: FP8-valued queries
``[R, 32, 128]`` (BF16), head weights ``[R, 32]`` (FP32), FP8-valued keys
``[C, 128]`` (BF16), causal-like validity (row ``r`` sees the candidates
``c <= C - R + r``) and the scores ``dsa_prefill.candidate_scores`` gives
them, with a tie at the top-k boundary injected into 1% of the rows (the
row's ``k``-th best score copied onto its next best).

* ``score-loop``: ``dsa_prefill.candidate_scores`` (a head loop over ``[R, C]``).
* ``score-einsum``: one einsum over all heads, then the weighted ReLU sum
  (``rel``: its largest deviation from ``score-loop`` per row, relative to the
  row's largest score, when both ran).
* ``select-tc``: ``dsa_prefill.select_mask``, the TensorCore exact threshold.
* ``threshold-bisect`` / ``threshold-r8`` / ``threshold-r4``: the threshold
  alone (``(tau, quota)`` from the scores' order keys): ``dsa_prefill``'s
  32-pass bisection over the key bits, and a radix select with 8-bit and
  4-bit digits (:func:`radix_threshold`). Checked on the same rows against
  the reference's ``k``-th score (``mismatch``).
* ``select-topk``: ``jax.lax.top_k``, then a threshold at its ``k``-th key.
* ``select-sc``: ``sparsecore_topk``'s indices, then their bitmask either by
  ``generate_mask_sc`` or by a scatter; both are recorded (``variant``, with
  the indices alone as ``indices``) and the cheaper is the case's.
* ``exchange-bits`` / ``exchange-int8``: an all-gather over the stage's cores
  of an ``[R, C / 32]`` packed bitmask or of an ``[R, C]`` int8 mask.

Every select case ends in ``dsa_prefill.pack_bits``. A call is timed from a
barrier to a one-element read of its output, after three warm-ups: ``ms`` is
the median over ``--repeats`` of the slowest core. Select cases are checked on
64 rows per core (the tied rows first) against ``dsa_reference.select_topk``:
``mismatch`` counts the rows whose selected set differs, ``ties_differ`` those
whose set differs only among the ``k``-th score's ties. A shape whose
estimated working set exceeds the device's memory is recorded as ``oom`` and
skipped; a case that fails to build or run on any core is recorded with its
``error``.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time

import numpy as np

from tools.megakernel import harness

CASES = (
    "score-loop",
    "score-einsum",
    "select-sc",
    "select-tc",
    "select-topk",
    "threshold-r8",
    "threshold-r4",
    "threshold-bisect",
    "exchange-bits",
    "exchange-int8",
)
SC_VARIANTS = ("mask-sc", "scatter")
TOPK = 2048  # dsa.INDEX_TOPK
HEADS, DIM = 32, 128
TIE_EVERY = 100  # one row in a hundred gets a tie at the top-k boundary
CHECK_ROWS = 64
WARMUPS = 3
MEMORY_FRACTION = 0.85  # of the device's limit a case's working set may take
FALLBACK_LIMIT = 64 << 30  # when the device reports no limit


def _ints(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected comma separated integers"
        ) from None
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected comma separated positive integers"
        )
    return values


def _candidates(text: str) -> list[int]:
    values = _ints(text)
    if any(value % 32 for value in values):
        raise argparse.ArgumentTypeError(f"{text!r}: candidates are whole words of 32")
    return values


def _case_names(text: str) -> list[str]:
    names = [part for part in text.split(",") if part]
    unknown = [name for name in names if name not in CASES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown case {unknown}; choose from {list(CASES)}"
        )
    return names


def add_arguments(parser) -> None:
    parser.add_argument("--cases", type=_case_names, default=list(CASES))
    parser.add_argument(
        "--rows", type=_ints, default=[1024, 2048, 8192], help="query rows per core"
    )
    parser.add_argument(
        "--candidates",
        type=_candidates,
        default=[8192, 16384, 32768, 147456],
        help="candidate keys per row (valid: a causal-like prefix)",
    )
    parser.add_argument("--repeats", type=int, default=20)


def tie_rows(rows: int) -> np.ndarray:
    return np.arange(0, rows, TIE_EVERY)


def check_rows(rows: int) -> np.ndarray:
    """The rows each core checks: the tied rows (at most half), then rows
    spread over the rest."""
    tied = tie_rows(rows)[: CHECK_ROWS // 2]
    rest = np.setdiff1d(np.arange(rows), tied)
    spread = rest[np.linspace(0, rest.size - 1, CHECK_ROWS - tied.size).astype(int)]
    return np.concatenate((tied, spread))


def lengths_of(rows: int, candidates: int):
    import jax.numpy as jnp

    return jnp.maximum(candidates - rows + 1 + jnp.arange(rows, dtype=jnp.int32), 0)


def valid_of(lengths, candidates: int):
    import jax.numpy as jnp

    return jnp.arange(candidates, dtype=jnp.int32)[None] < lengths[:, None]


def inject_ties(scores, lengths, rows, k: int):
    """``scores`` with each of ``rows``' ``k``-th best valid score copied onto
    its ``k + 1``-th (a tie straddling the top-k boundary)."""
    import jax
    import jax.numpy as jnp

    rows = jnp.asarray(rows, jnp.int32)
    masked = jnp.where(valid_of(lengths[rows], scores.shape[1]), scores[rows], -jnp.inf)
    values, index = jax.lax.top_k(masked, k + 1)
    return scores.at[rows, index[:, k]].set(values[:, k - 1])


def inputs(seed, rows: int, candidates: int, k: int = TOPK):
    """``(q, w, keys, lengths, scores)`` of one core (see the module doc)."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_prefill

    kq, kw, kk = jax.random.split(jax.random.key(seed), 3)
    q, _ = dsa_cache.quantize_index_queries(
        jax.random.normal(kq, (rows, HEADS, DIM), jnp.float32)
    )
    w = jax.random.normal(kw, (rows, HEADS), jnp.float32)
    raw = jax.random.normal(kk, (candidates, DIM), jnp.float32).astype(jnp.bfloat16)
    keys = dsa_cache.dequantize_index_keys(*dsa_cache.quantize_index_keys(raw))
    keys = keys.astype(jnp.bfloat16)
    lengths = lengths_of(rows, candidates)
    scores = dsa_prefill.candidate_scores(q, w, keys)
    scores = inject_ties(scores, lengths, tie_rows(rows), k)
    return q.astype(jnp.bfloat16), w, keys, lengths, scores


def einsum_scores(q, w, keys):
    """``dsa_prefill.candidate_scores`` as one ``[R, heads, C]`` einsum, then
    the weighted ReLU sum over heads (FP32)."""
    import jax
    import jax.numpy as jnp

    logits = jnp.einsum(
        "rhd,cd->rhc",
        q.astype(jnp.bfloat16),
        keys.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    return jnp.sum(w[:, :, None] * jax.nn.relu(logits), axis=1)


def topk_mask(scores, lengths, k: int):
    """Each row's top ``k`` valid candidates by ``jax.lax.top_k``, then a
    threshold at its ``k``-th key: the keys above it and, of the keys equal to
    it, those up to the last column ``top_k`` took (ties to the lower column,
    as ``top_k`` takes them)."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    columns = jnp.arange(scores.shape[1], dtype=jnp.int32)[None]
    valid = columns < lengths[:, None]
    values, index = jax.lax.top_k(jnp.where(valid, scores, -jnp.inf), k)
    keys = dsa_prefill.order_keys(scores)
    top = dsa_prefill.order_keys(values)
    tau = top[:, k - 1 :]
    last = jnp.max(jnp.where(top == tau, index, -1), axis=1, keepdims=True)
    chosen = valid & ((keys > tau) | ((keys == tau) & (columns <= last)))
    return jnp.where((lengths <= k)[:, None], valid, chosen)


def radix_threshold(keys, valid, k: int, bits: int):
    """``dsa_prefill.topk_threshold``'s ``(tau, quota)`` by a radix select,
    one ``bits``-bit digit a pass from the most significant (``32 / bits``
    passes): each pass histograms the next digit of the valid keys that match
    ``tau``'s digits so far and takes the highest digit whose count from the
    top reaches the keys still to take."""
    import jax.numpy as jnp

    bins = jnp.arange(1 << bits, dtype=jnp.uint32)
    tau = jnp.zeros(keys.shape[:1], jnp.uint32)
    left = jnp.full(keys.shape[:1], k, jnp.int32)
    for shift in range(32 - bits, -1, -bits):
        match = valid
        if shift + bits < 32:
            high = shift + bits
            match = valid & (keys >> high == (tau >> high)[:, None])
        digit = keys >> shift & bins[-1]
        counts = jnp.sum(
            match[:, None] & (digit[:, None] == bins[None, :, None]),
            axis=2,
            dtype=jnp.int32,
        )
        from_top = jnp.cumsum(counts[:, ::-1], axis=1)[:, ::-1]
        chosen = jnp.maximum(jnp.sum(from_top >= left[:, None], axis=1) - 1, 0)
        above = jnp.take_along_axis(from_top - counts, chosen[:, None], axis=1)[:, 0]
        left = left - above
        tau = tau | chosen.astype(jnp.uint32) << shift
    return tau, left


def threshold_form(case: str):
    """The ``(keys, valid, k) -> (tau, quota)`` of a ``threshold-*`` case."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    if case == "threshold-bisect":
        return dsa_prefill.topk_threshold
    bits = 8 if case == "threshold-r8" else 4
    return lambda keys, valid, k: radix_threshold(keys, valid, k, bits)


def host_order_keys(scores: np.ndarray) -> np.ndarray:
    """Host ``dsa_prefill.order_keys``."""
    bits = np.asarray(scores, np.float32).view(np.uint32).copy()
    bits[bits == 0x80000000] = 0
    return np.where(bits >> 31 == 1, ~bits, bits | np.uint32(0x80000000))


def threshold_mismatches(tau, quota, scores, reference, lengths, k: int) -> int:
    """Rows of more than ``k`` valid candidates whose ``(tau, quota)`` are not
    the order key of the reference's ``k``-th score (``reference``:
    ``select_topk``'s indices) and the count of keys equal to it the top
    ``k`` takes."""
    bad = 0
    for r in range(scores.shape[0]):
        if lengths[r] <= k:
            continue
        keys = host_order_keys(scores[r, : lengths[r]])
        want = host_order_keys(scores[r, reference[r, k - 1]])
        if tau[r] != want or quota[r] != k - int((keys > want).sum()):
            bad += 1
    return bad


def sc_indices(scores, lengths, k: int):
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

    return sparsecore_topk(scores, k, lengths)


def indices_mask(indices, candidates: int, variant: str):
    """The bool ``[R, candidates]`` of ``indices`` (``-1`` padded):
    ``mask-sc`` by ``generate_mask_sc``, ``scatter`` by an XLA scatter."""
    import jax.numpy as jnp

    if variant == "mask-sc":
        from vllm_torchtpu.kernels.mla.masked_dense.csa_mask import generate_mask_sc

        return generate_mask_sc(indices, candidates) != 0
    rows = jnp.arange(indices.shape[0])[:, None]
    target = jnp.where(indices >= 0, indices, candidates)
    return (
        jnp.zeros((indices.shape[0], candidates), bool)
        .at[rows, target]
        .set(True, mode="drop")
    )


def unpack_rows(words: np.ndarray, candidates: int) -> np.ndarray:
    """Host :func:`dsa_prefill.unpack_bits` of int32-viewed words."""
    bits = (words.view(np.uint32)[..., None] >> np.arange(32, dtype=np.uint32)) & 1
    return bits.reshape(*words.shape[:-1], -1)[..., :candidates].astype(bool)


def verdicts(selected, reference, scores, lengths, k: int) -> tuple[int, int]:
    """``(mismatch, ties_differ)`` over the rows of ``selected`` (bool ``[n,
    C]``) against ``reference`` (``select_topk``'s indices, ``-1`` padded) on
    ``scores`` ``[n, C]`` with ``lengths`` ``[n]``: a row whose set differs
    counts as ``ties_differ`` when it still holds ``k`` valid candidates all
    scoring at least the reference's ``k``-th score."""
    mismatch = ties = 0
    for r in range(selected.shape[0]):
        want = reference[r][reference[r] >= 0]
        got = np.flatnonzero(selected[r])
        if np.array_equal(np.sort(want), got):
            continue
        if (
            want.size == k
            and got.size == k
            and got.max() < lengths[r]
            and (scores[r, got] >= scores[r, want].min()).all()
        ):
            ties += 1
        else:
            mismatch += 1
    return mismatch, ties


_OPS: dict[str, object] = {}


def _op(mesh, name: str, body, inputs: int, outputs: int):
    """The torch op of ``body`` over ``mesh`` (every operand and result
    sharded over the stage's cores), registered once per ``name``."""
    op_name = "pallas::k3_stage_dsa_micro_" + re.sub(r"[^0-9A-Za-z_]", "_", name)
    if op_name not in _OPS:
        from jax.sharding import PartitionSpec as P

        from vllm_torchtpu.megakernel import bridge

        spec = P("k3_sp")
        _OPS[op_name] = bridge.register_op(
            op_name, mesh, body, (spec,) * inputs, (spec,) * outputs
        )
    return _OPS[op_name]


def _outputs(result) -> tuple:
    return tuple(result) if isinstance(result, (tuple, list)) else (result,)


def _local(fn):
    """A body over each core's ``[1, ...]`` blocks from ``fn`` over the
    blocks' contents."""

    def body(*blocks):
        return tuple(x[None] for x in _outputs(fn(*(b[0] for b in blocks))))

    return body


def _words(mask):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

    return jax.lax.bitcast_convert_type(dsa_prefill.pack_bits(mask), jnp.int32)


def _gather_all(value) -> list:
    import torch

    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered


def _errors(fn) -> list[str]:
    """Run ``fn`` on every core: why it raised on the cores where it did (the
    same list on every core, empty when it ran everywhere)."""
    try:
        fn()
        why = None
    except Exception as exc:  # noqa: BLE001 -- reported through the gather
        why = f"{type(exc).__name__}: {str(exc)[:400]}"
    return [w for w in _gather_all(why) if w is not None]


def _time(op, operands, repeats: int):
    """``(ms, rank median ms, last outputs)``: ``ms`` the median over repeats
    of the slowest core's call."""
    from vllm_torchtpu.megakernel import bridge

    times = []
    for _ in range(repeats):
        harness.sync()
        started = time.perf_counter()
        outputs = _outputs(op(*operands))
        bridge.materialize(outputs[0])
        times.append(1e3 * (time.perf_counter() - started))
    gathered = _gather_all(times)
    slowest = [max(rank[i] for rank in gathered) for i in range(repeats)]
    rank_median = statistics.median(statistics.median(rank) for rank in gathered)
    return statistics.median(slowest), rank_median, outputs


def _warm(op, operands):
    from vllm_torchtpu.megakernel import bridge

    for _ in range(WARMUPS):
        bridge.materialize(_outputs(op(*operands))[0])


def working_bytes(case: str, rows: int, candidates: int, cores: int) -> int:
    """A case's estimated per-core working set, its inputs included."""
    scores = 4 * rows * candidates
    base = scores + 2 * rows * HEADS * DIM + 2 * candidates * DIM
    extra = {
        "score-loop": 2 * scores,
        "score-einsum": HEADS * scores + scores,
        "exchange-bits": (cores + 1) * scores // 32,
        "exchange-int8": (cores + 1) * scores // 4,
    }.get(case, 3 * scores)
    return base + extra


def memory_limit() -> int:
    import jax

    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 -- not every backend reports it
        stats = {}
    return int(stats.get("bytes_limit", 0)) or FALLBACK_LIMIT


def run(args, records: harness.Records) -> None:
    import torch

    from tools.megakernel import k3_stage

    stage, core, mesh, _ = k3_stage._stage_mesh()
    cores = k3_stage.CORES
    limit = int(MEMORY_FRACTION * memory_limit())
    harness.log(
        f"dsa-micro: cases {args.cases}, rows {args.rows}, candidates "
        f"{args.candidates}, top-{TOPK}, {args.repeats} repeats; working-set "
        f"limit {limit / 2**30:.1f} GiB per core"
    )
    seed = torch.tensor(
        [[args.seed * 1000 + stage * cores + core]], dtype=torch.int32
    ).to("tpu")
    for rows in args.rows:
        for candidates in args.candidates:
            _Shape(args, records, mesh, cores, rows, candidates, limit).run(seed)


class _Shape:
    """One ``(R, C)`` point: its inputs, reference selection and cases."""

    def __init__(self, args, records, mesh, cores, rows, candidates, limit):
        self.args, self.records, self.mesh, self.cores = args, records, mesh, cores
        self.rows, self.candidates, self.limit = rows, candidates, limit
        self.tag = f"r{rows}_c{candidates}"
        self.point = dict(rows=rows, candidates=candidates)
        self.picks = check_rows(rows)
        self.loop_rows = None
        self.order_keys = None

    def op(self, name, fn, inputs, outputs):
        return _op(self.mesh, f"{name}_{self.tag}", _local(fn), inputs, outputs)

    def add(self, case, **fields) -> None:
        self.records.add(case=case, **self.point, **fields)

    def log(self, label, result, extra=None) -> None:
        ms, rank_median, _ = result
        harness.log(
            f"{label} {self.tag}: {ms:.3f} ms (slowest core, median; median core "
            f"{rank_median:.3f})" + (f" {extra}" if extra else "")
        )

    def fits(self, case: str) -> bool:
        need = working_bytes(case, self.rows, self.candidates, self.cores)
        if need <= self.limit:
            return True
        harness.log(f"{case} {self.tag}: {need / 2**30:.1f} GiB working set, skipped")
        self.add(case, oom=True)
        return False

    def measure(self, case, op, operands, **fields):
        """``_time``'s result, or None after recording why the case failed."""
        errors = _errors(lambda: _warm(op, operands))
        if errors:
            harness.log(
                f"{case} {self.tag}: failed on {len(errors)} cores: {errors[0]}"
            )
            self.add(case, **fields, error=errors[0], failed_cores=len(errors))
            return None
        return _time(op, operands, self.args.repeats)

    def run(self, seed) -> None:
        from tools.megakernel import dsa_reference

        rows, candidates = self.rows, self.candidates
        cases = [case for case in self.args.cases if self.fits(case)]
        if not cases:
            return
        q, w, keys, lengths, scores = _outputs(
            self.op(
                "inputs",
                lambda s: inputs(s[0], rows, candidates),
                1,
                5,
            )(seed)
        )
        picks = self.picks  # host indices: constants of the traced bodies

        def reference_rows(s, n):
            return s[picks], dsa_reference.select_topk(s[picks], n[picks], TOPK)

        reference = self.op("reference", reference_rows, 2, 2)
        self.ref_scores, self.ref_index = (
            _outputs(t)[0][0].cpu().numpy() for t in reference(scores, lengths)
        )
        host_lengths = np.maximum(candidates - rows + 1 + np.arange(rows), 0)
        self.ref_lengths = host_lengths[self.picks]
        self.pick_words = self.op("pick_words", lambda x: x[picks], 1, 1)
        self.pick_scores = self.op("pick_scores", lambda x: x[picks], 1, 1)
        self.pick_rows = self.op("pick_rows", lambda x: x[picks], 1, 1)
        for case in cases:
            if case.startswith("score-"):
                self.score(case, (q, w, keys))
            elif case == "select-sc":
                self.select_sc((scores, lengths))
            elif case.startswith("threshold-"):
                self.threshold(case, (scores, lengths))
            elif case.startswith("select-"):
                self.select(case, (scores, lengths))
            else:
                self.exchange(case, scores)

    def score(self, case, operands) -> None:
        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

        fn = dsa_prefill.candidate_scores if case == "score-loop" else einsum_scores
        result = self.measure(case, self.op(case, fn, 3, 1), operands)
        if result is None:
            return
        got = _outputs(self.pick_scores(result[2][0]))[0][0].cpu().numpy()
        extra = {}
        if case == "score-loop":
            self.loop_rows = got
        elif self.loop_rows is not None:
            base = self.loop_rows
            rel = np.abs(got - base).max(axis=1) / np.maximum(
                np.abs(base).max(axis=1), 1e-30
            )
            extra["rel"] = max(_gather_all(float(rel.max())))
        self.log(case, result, extra)
        self.add(case, **_timing(result), **extra)

    def select(self, case, operands) -> None:
        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

        candidates = self.candidates
        if case == "select-tc":

            def fn(s, n):
                return _words(dsa_prefill.select_mask(s, valid_of(n, candidates), TOPK))
        else:

            def fn(s, n):
                return _words(topk_mask(s, n, TOPK))

        result = self.measure(case, self.op(case, fn, 2, 1), operands)
        if result is not None:
            check = self.check(result[2][0])
            self.log(case, result, check)
            self.add(case, **_timing(result), **check)

    def select_sc(self, operands) -> None:
        candidates = self.candidates
        indices = self.op(
            "select_sc_indices", lambda s, n: sc_indices(s, n, TOPK), 2, 1
        )
        result = self.measure("select-sc", indices, operands, variant="indices")
        if result is None:
            return
        self.log("select-sc indices", result)
        self.add("select-sc", variant="indices", **_timing(result))
        best = None
        for variant in SC_VARIANTS:

            def fn(s, n, variant=variant):
                chosen = sc_indices(s, n, TOPK)
                return _words(indices_mask(chosen, candidates, variant))

            op = self.op(f"select_sc_{variant}", fn, 2, 1)
            timed = self.measure("select-sc", op, operands, variant=variant)
            if timed is None:
                continue
            check = self.check(timed[2][0])
            self.log(f"select-sc {variant}", timed, check)
            self.add("select-sc", variant=variant, **_timing(timed), **check)
            if best is None or timed[0] < best[1][0]:
                best = (variant, timed, check)
        if best is not None:
            variant, timed, check = best
            self.add("select-sc", bitmask=variant, **_timing(timed), **check)

    def check(self, words) -> dict:
        got = _outputs(self.pick_words(words))[0][0].cpu().numpy()
        mismatch, ties = verdicts(
            unpack_rows(got, self.candidates),
            self.ref_index,
            self.ref_scores,
            self.ref_lengths,
            TOPK,
        )
        totals = np.sum(_gather_all((mismatch, ties)), axis=0)
        return dict(
            exact=bool(totals.sum() == 0),
            mismatch=int(totals[0]),
            ties_differ=int(totals[1]),
        )

    def threshold(self, case, operands) -> None:
        import jax
        import jax.numpy as jnp

        from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_prefill

        scores, lengths = operands
        if self.order_keys is None:

            def keys_of(s):
                return jax.lax.bitcast_convert_type(
                    dsa_prefill.order_keys(s), jnp.int32
                )

            self.order_keys = _outputs(self.op("order_keys", keys_of, 1, 1)(scores))[0]
        candidates, form = self.candidates, threshold_form(case)

        def fn(keys, n):
            keys = jax.lax.bitcast_convert_type(keys, jnp.uint32)
            tau, quota = form(keys, valid_of(n, candidates), TOPK)
            return jax.lax.bitcast_convert_type(tau, jnp.int32), quota

        result = self.measure(case, self.op(case, fn, 2, 2), (self.order_keys, lengths))
        if result is None:
            return
        tau, quota = (
            _outputs(self.pick_rows(x))[0][0].cpu().numpy() for x in result[2][:2]
        )
        bad = threshold_mismatches(
            tau.view(np.uint32),
            quota,
            self.ref_scores,
            self.ref_index,
            self.ref_lengths,
            TOPK,
        )
        bad = int(np.sum(_gather_all(bad)))
        check = dict(exact=bad == 0, mismatch=bad)
        self.log(case, result, check)
        self.add(case, **_timing(result), **check)

    def exchange(self, case, scores) -> None:
        import jax
        import jax.numpy as jnp

        def prepare(s):
            if case == "exchange-bits":
                return _words(s > 0)
            return (s > 0).astype(jnp.int8)

        payload = _outputs(self.op(f"{case}_payload", prepare, 1, 1)(scores))

        def gather(x):
            return jax.lax.all_gather(x, "k3_sp", axis=0, tiled=True)

        result = self.measure(case, self.op(case, gather, 1, 1), payload)
        if result is not None:
            self.log(case, result)
            self.add(case, **_timing(result))


def _timing(result) -> dict:
    ms, rank_median, _ = result
    return dict(ms=round(ms, 3), rank_median_ms=round(rank_median, 3))
