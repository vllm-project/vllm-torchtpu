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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# tools/megakernel/bench.py
"""Kernel-only timing of the serving ops (``bench decode`` / ``bench prefill``)."""

from __future__ import annotations

import importlib
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from .. import config, settings
from .. import prefill as k3_prefill
from .. import program as k3_program
from ..decode_ops import Control, DecodeOps
from ..draft import draft_class
from ..runtime_shim import bridge
from ..runtime_shim import program as mk
from . import harness, k3

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterator

_MAX_WIDE_DRAFTS_BATCH = 4
_MAX_SPEC_BATCH = 8
_WIDE_DRAFTS = 7
_SHORT_DRAFTS = 3
_RANDOM_TOKEN_LOW = 1000
_RANDOM_TOKEN_HIGH = 100000
_STEP_SLACK_TOKENS = 64
_DRAFT_POS_SLACK = 256
_DEFAULT_TOP_P = 0.95
_DEFAULT_SEED = 7
_VERIFY_WIDTH = 8
_KDA_GROUPS = 3


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``decode`` and ``prefill`` subparsers on ``parser``."""
    sub = parser.add_subparsers(dest="role", required=True)
    decode = sub.add_parser("decode", help="the decode ops")
    decode.add_argument("--batches", default="1,8")
    decode.add_argument(
        "--contexts",
        default="1152",
        help="per-request context (all rows of a batch)",
    )
    decode.add_argument(
        "--mix",
        default="",
        help="also one mixed batch, rows by context: e.g. 1152x4,131072x4",
    )
    decode.add_argument("--steps", type=int, default=40)
    decode.add_argument("--repeats", type=int, default=3)
    decode.add_argument("--temperature", type=float, default=0.0)
    decode.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="slot layout as serving (default: max batch)",
    )
    decode.add_argument(
        "--dspark",
        default="",
        metavar="CHECKPOINT",
        help="time DSpark verify + draft steps with this speculator",
    )
    decode.add_argument(
        "--drafts",
        type=int,
        choices=(_SHORT_DRAFTS, _WIDE_DRAFTS),
        default=0,
        help="with --dspark: this many drafts at every batch up to eight",
    )
    decode.add_argument(
        "--acceptance",
        default="",
        help="with --dspark: synthetic per-position acceptance rates of seven drafts",
    )
    decode.add_argument(
        "--spec-drafts",
        type=int,
        choices=(0, _SHORT_DRAFTS, _WIDE_DRAFTS),
        default=0,
        help="with --dspark: this many drafts at every batch, any size",
    )
    decode.add_argument(
        "--save-tokens",
        default="",
        help="file: rank 0 saves each case's timed greedy tokens [steps, batch] (npz)",
    )
    prefill = sub.add_parser("prefill", help="the prefill stack")
    prefill.add_argument(
        "--prompts",
        default="8192x1,2048x4",
        help="cases LENxN[@PREFIX]: N prompts of LEN new tokens per step",
    )
    prefill.add_argument("--repeats", type=int, default=3)
    prefill.add_argument(
        "--token-file",
        default="",
        help="a .npy of int32 token ids for real text slices",
    )
    for role in (decode, prefill):
        role.add_argument(
            "--trace",
            default="",
            help="directory: rank 0 saves an xprof trace of each case",
        )


def _prompt_tokens(
    rng: np.random.Generator,
    corpus: np.ndarray | None,
    size: int,
) -> np.ndarray:
    """Return ``size`` token ids from ``corpus`` or uniform random integers."""
    if corpus is None:
        return rng.integers(_RANDOM_TOKEN_LOW, _RANDOM_TOKEN_HIGH, size, dtype=np.int32)
    if size > corpus.size:
        harness.fail(f"--token-file holds {corpus.size} tokens, a prompt needs {size}")
    start = int(rng.integers(0, corpus.size - size + 1))
    return corpus[start : start + size]


@contextmanager
def _trace(directory: str, tag: str, *, last: bool) -> Iterator[None]:
    """Save an xprof trace of the last repeat under ``directory`` on trace ranks."""
    ranks = [int(r) for r in settings.TRACE_RANKS.split(",")]
    if not (directory and last and harness.rank() in ranks):
        yield
        return
    profiler = importlib.import_module("torch_tpu._internal.profiler._impl")

    base = Path(directory)
    path = str(base / tag if ranks == [0] else base / f"rank{harness.rank()}" / tag)
    profiler.start_trace(path)
    try:
        yield
    finally:
        profiler.stop_trace()
        harness.log(f"trace of {tag}: {path}")


def run(args: argparse.Namespace, records: harness.Records) -> None:
    """Dispatch ``args.role`` to ``run_decode`` or ``run_prefill``."""
    (run_decode if args.role == "decode" else run_prefill)(args, records)


def _drafts_at(args: argparse.Namespace, batch: int) -> int:
    """Return speculative draft count for ``batch`` under ``args``."""
    if args.spec_drafts:
        return int(args.spec_drafts)
    if args.drafts:
        return int(args.drafts) if batch <= _MAX_SPEC_BATCH else 0
    if batch <= _MAX_WIDE_DRAFTS_BATCH:
        return _WIDE_DRAFTS
    return _SHORT_DRAFTS if batch <= _MAX_SPEC_BATCH else 0


def _build_draft_and_schedule(
    args: argparse.Namespace,
    cases: list[tuple[int, int | list[int]]],
    max_context: int,
    max_num_seqs: int,
) -> tuple[object | None, tuple[tuple[int, int, int], ...] | None]:
    """Build optional DSpark draft instance and per-batch schedule tuple."""
    if not (args.dspark and any(_drafts_at(args, b) for b, _ in cases)):
        return None, None
    rates = None
    if args.acceptance:
        seven = tuple(float(r) for r in args.acceptance.split(","))
        if len(seven) != _WIDE_DRAFTS:
            msg = "--acceptance takes seven rates"
            raise SystemExit(msg)
        rates = {_WIDE_DRAFTS: seven, _SHORT_DRAFTS: seven[:_SHORT_DRAFTS]}
    max_pos = max_context + _VERIFY_WIDTH * args.steps * args.repeats + _DRAFT_POS_SLACK
    draft = draft_class(args.dspark)(
        max_position=max_pos,
        checkpoint=args.dspark,
        presharded=args.presharded,
        synthetic_rates_by_width=rates,
    )
    if args.spec_drafts:
        return draft, ((1, max_num_seqs, int(args.spec_drafts)),)
    if args.drafts:
        spans = [(1, min(_MAX_SPEC_BATCH, max_num_seqs), int(args.drafts))]
        if max_num_seqs > _MAX_SPEC_BATCH:
            spans.append((_MAX_SPEC_BATCH + 1, max_num_seqs, 0))
        return draft, tuple(spans)
    spans = [(1, min(_MAX_WIDE_DRAFTS_BATCH, max_num_seqs), _WIDE_DRAFTS)]
    if max_num_seqs > _MAX_WIDE_DRAFTS_BATCH:
        spans.append(
            (
                _MAX_WIDE_DRAFTS_BATCH + 1,
                min(_MAX_SPEC_BATCH, max_num_seqs),
                _SHORT_DRAFTS,
            )
        )
    if max_num_seqs > _MAX_SPEC_BATCH:
        spans.append((_MAX_SPEC_BATCH + 1, max_num_seqs, 0))
    return draft, tuple(spans)


def _decode_cases(
    args: argparse.Namespace,
) -> tuple[list[tuple[int, int | list[int]]], int]:
    """Parse ``--batches``, ``--contexts``, and ``--mix`` into benchmark cases."""
    batches = [int(b) for b in args.batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    mix = [
        int(part.split("x")[0])
        for part in filter(None, args.mix.split(","))
        for _ in range(int(part.split("x")[1]))
    ]
    cases: list[tuple[int, int | list[int]]] = [
        (batch, context) for context in contexts for batch in batches
    ]
    if mix:
        cases.append((len(mix), mix))
    return cases, max(contexts + mix)


def _rows_for_decode(
    args: argparse.Namespace,
    pools: k3.Pools,
    pages: int,
    draft: object | None,
    case_spec: tuple[int, int | list[int], bool, str],
) -> mk.StepInput:
    """Build ``StepInput`` for one benchmark decode step."""
    batch, context, first, tag = case_spec
    dp = config.active().attention_dp
    sampling = (
        args.temperature,
        _DEFAULT_TOP_P if args.temperature > 0 else 1.0,
        0,
        _DEFAULT_SEED,
    )
    per_row = context if isinstance(context, list) else [context] * batch
    out: list[mk.RequestInput] = []
    for r in range(batch):
        table = pools.scheduled([1 + r // dp * pages + p for p in range(pages)], r)
        speculating = draft is not None and _drafts_at(args, batch) > 0 and not first
        query = _drafts_at(args, batch) + 1 if speculating else 1
        tokens: list[int] | np.ndarray = (
            [_RANDOM_TOKEN_LOW + 37 * r] if first else np.zeros(query, np.int32)
        )
        computed = per_row[r] - 1 if first else 0
        out.append(k3.request(f"{tag}-{r}", tokens, table, sampling, computed))
    return mk.StepInput(out)


def run_decode(args: argparse.Namespace, records: harness.Records) -> None:
    """Run kernel-only decode timing benchmark across ``args`` cases."""
    cases, max_context = _decode_cases(args)
    rows = max(batch for batch, _ in cases)
    max_num_seqs = args.max_num_seqs or rows
    sub_base = 2 * max_num_seqs + 1
    pages = (
        -(
            -(
                max_context
                + _VERIFY_WIDTH * args.steps * args.repeats
                + _STEP_SLACK_TOKENS
            )
            // k3.logical()
        )
        + 1
    )
    dp = config.active().attention_dp
    blocks = 1 + -(-rows // dp) * pages
    sizes = tuple(sorted({b for b, _ in cases if b > 1} | {1}))
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(sub_base) if args.dspark else sub_base,
        batch_sizes=sizes,
    )
    pools = k3.Pools(blocks, pages, 1, pattern=True)
    draft, schedule = _build_draft_and_schedule(args, cases, max_context, max_num_seqs)
    dec = k3.decoder(
        ops,
        pools,
        max_num_seqs=max_num_seqs,
        max_model_len=pages * k3.logical(),
        draft=draft,
        schedule=schedule,
    )
    harness.sync()
    dec.warmup()
    harness.log(
        f"warm: {len(pools.tensors)} pools of {blocks} blocks, batch sizes {sizes}"
    )
    results, streams = [], {}
    for batch, context in cases:
        label: int | str = (
            context if not isinstance(context, list) else f"mix {args.mix}"
        )
        tag = f"b{batch}c{context if not isinstance(context, list) else 'mix'}"
        dec.step(
            _rows_for_decode(args, pools, pages, draft, (batch, context, True, tag))
        )
        bridge.materialize(
            dec.step(
                _rows_for_decode(
                    args, pools, pages, draft, (batch, context, False, tag)
                )
            ).tokens
        )
        times, outs = [], []
        for repeat in range(args.repeats):
            with _trace(args.trace, tag, last=repeat == args.repeats - 1):
                started = time.perf_counter()
                for _ in range(args.steps):
                    step_in = _rows_for_decode(
                        args, pools, pages, draft, (batch, context, False, tag)
                    )
                    outs.append(dec.step(step_in).tokens)
                bridge.materialize(outs[-1])
                times.append(1e3 * (time.perf_counter() - started) / args.steps)
        ms = statistics.median(times)
        dspark = draft is not None and _drafts_at(args, batch) > 0
        if dspark:
            tok_mat = torch.cat([o[:batch] for o in outs]).cpu()
            per_step = float((tok_mat >= 0).sum()) / len(outs) / batch
            rate = 1e3 * per_step * batch / ms
            line = (
                f"B{batch} ctx {label}: DSpark x{_drafts_at(args, batch)} "
                f"{ms:.2f} ms per step, "
                f"{per_step:.2f} tokens per request step -> {rate:.0f} tok/s"
            )
        else:
            rate = batch * 1e3 / ms
            line = f"B{batch} ctx {label}: {ms:.2f} ms per step, {rate:.1f} tok/s"
        harness.log(f"{line}  (repeats {[round(t, 2) for t in times]})")
        records.add(
            case=tag,
            batch=batch,
            context=label,
            dspark=dspark,
            temperature=args.temperature,
            ms=round(ms, 3),
            tokens_per_s=round(rate, 1),
            repeats=[round(t, 3) for t in times],
        )
        results.append((batch, label, ms))
        if args.save_tokens:
            streams[tag] = torch.stack([o[:batch, 0] for o in outs]).cpu().numpy()
        dec.forget([f"{tag}-{r}" for r in range(batch)])
    if args.save_tokens and harness.rank() == 0:
        savez_fn = cast("Callable[..., None]", np.savez)
        savez_fn(args.save_tokens, **streams)
    summary = "; ".join(f"B{b}@{c}={ms:.2f}ms" for b, c, ms in results)
    harness.log(f"summary: {summary}")


def _parse_prefill_cases(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    """Parse ``--prompts`` specifications into ``(length, count, prefix)`` tuples."""
    cases: list[tuple[int, int, int]] = []
    for part in filter(None, args.prompts.split(",")):
        prompt, _, prefix_str = part.partition("@")
        length, count = (int(x) for x in prompt.split("x"))
        prefix = int(prefix_str) if prefix_str else 0
        if length * count > k3_program.MAX_STEP_TOKENS or count > k3_prefill.SLOTS:
            harness.fail(
                f"prefill case {part}: N * LEN must fit one 8192-token step, "
                f"N <= {k3_prefill.SLOTS}"
            )
        if prefix % k3.PAGE:
            harness.fail(
                f"prefill case {part}: a prefix-cache hit is page-aligned; "
                f"@PREFIX must be a multiple of {k3.PAGE}"
            )
        cases.append((length, count, prefix))
    return cases


def _prefill_tables(
    cases: list[tuple[int, int, int]],
    max_seqs: int,
    max_model_len: int,
) -> tuple[k3.Pools, list[tuple[np.ndarray, ...]]]:
    """Allocate stand-in prefill pools and per-slot block tables."""
    totals = [length + prefix for length, _, prefix in cases]
    slot_layout: list[tuple[int, list[int]]] = []
    for slot in range(max_seqs):
        longest = max(
            total
            for total, (_, count, _) in zip(totals, cases, strict=True)
            if count > slot
        )
        columns = sorted(
            {
                column
                for length, count, prefix in cases
                if count > slot
                for column in ([prefix // k3.PAGE - 1] if prefix else [])
                + [(prefix + length - 1) // k3.PAGE]
            }
        )
        slot_layout.append((-(-longest // k3.LOGICAL), columns))
    pools = k3.Pools(
        1 + sum(mla + _KDA_GROUPS * len(columns) for mla, columns in slot_layout),
        max_model_len // k3.LOGICAL,
        max_model_len // k3.PAGE,
        pattern=True,
    )
    tables: list[tuple[np.ndarray, ...]] = []
    for mla, columns in slot_layout:
        kda: list[np.ndarray] = []
        for _ in range(_KDA_GROUPS):
            row = np.zeros(max_model_len // k3.PAGE, np.int32)
            for column in columns:
                row[column] = pools.fresh(1)[0]
            kda.append(row)
        tables.append(pools.tables(pools.fresh(mla), kda))
    return pools, tables


def run_prefill(args: argparse.Namespace, records: harness.Records) -> None:
    """Run kernel-only prefill timing benchmark across ``args`` prompt cases."""
    cases = _parse_prefill_cases(args)
    max_seqs = max(count for _, count, _ in cases)
    totals = [length + prefix for length, _, prefix in cases]
    max_model_len = -(-max(totals) // k3.LOGICAL) * k3.LOGICAL
    prefiller = cast(
        "k3_prefill.KimiPrefiller",
        k3_program.build(
            k3.engine_config(
                "prefill",
                max_num_seqs=max_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=k3_program.MAX_STEP_TOKENS,
            ),
            args.presharded,
            tp_rank=torch.distributed.get_rank(),
        ),
    )
    prefiller.load_weights()
    pools, tables = _prefill_tables(cases, max_seqs, max_model_len)
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    prefiller.warmup()
    harness.sync()
    rng = np.random.default_rng(0)
    corpus = np.load(args.token_file).astype(np.int32) if args.token_file else None
    largest = prefiller.program.windows[-1]
    for length, count, prefix in cases:
        tag = f"p{length}x{count}" + (f"c{prefix}" if prefix else "")
        pieces = [
            piece
            for r in range(count)
            for piece in k3_prefill.plan_pieces(r, prefix, prefix + length, [], largest)
        ]
        windows = "+".join(
            str(window)
            for window, _ in k3_prefill.pack_windows(
                pieces,
                prefiller.program.windows,
                prefiller.program.slots,
                prefiller.program.capacity,
            )
        )
        times: list[float] = []
        for repeat in range(args.repeats + 1):
            rows = [
                mk.RequestInput(
                    req_id=f"{tag}-{repeat}-{r}",
                    computed=prefix,
                    scheduled=length,
                    token_ids=_prompt_tokens(rng, corpus, prefix + length),
                    num_tokens=prefix + length,
                    num_prompt_tokens=prefix + length,
                    block_tables=tables[r],
                )
                for r in range(count)
            ]
            with _trace(args.trace, tag, last=repeat == args.repeats):
                started = time.perf_counter()
                bridge.materialize(prefiller.step(mk.StepInput(rows)).tokens)
                elapsed = 1e3 * (time.perf_counter() - started)
            if repeat:
                times.append(elapsed)
        ms = statistics.median(times)
        rate = length * count * 1e3 / ms
        on_prefix = f" on a {prefix} prefix" if prefix else ""
        harness.log(
            f"prefill {count} x {length} tokens{on_prefix} (window {windows}): "
            f"{ms:.1f} ms per step, {rate:.0f} tok/s  "
            f"(repeats {[round(t, 1) for t in times]})"
        )
        records.add(
            case=tag,
            length=length,
            count=count,
            context=prefix,
            windows=windows,
            ms=round(ms, 2),
            tokens_per_s=round(rate, 1),
            repeats=[round(t, 2) for t in times],
        )
