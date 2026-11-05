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
"""Kernel-only timing of the serving ops (``bench decode`` / ``bench prefill``).

**Decode** times ``KimiDecoder.step`` over vLLM's 24 unified pools for every
``--batches`` x ``--contexts`` case (``--mix``: one batch of rows at different
contexts), or with ``--dspark [CHECKPOINT]`` the DSpark verify + draft step
(serving's draft schedule: seven drafts through batch four, three through
eight; ``--acceptance`` fixes the accepted drafts per position so drafts
compare at equal acceptance). Each request starts at its context over
synthetic KV pages and a clean KDA state, with distinct tokens per row (they
set MoE routing); ``--steps`` steps are enqueued back to back and timed with
one host read, median of ``--repeats``. DSpark's own tokens per step on
synthetic contexts are not representative.

**Prefill** times :class:`KimiPrefiller.step` per ``--prompts`` case: ``LENxN``
is one step of ``N`` prompts of ``LEN`` new tokens, ``LENxN@PREFIX`` appends
them to a page-aligned cached prefix. Each line names the compiled windows the
case packs into.

**DSA microbenchmarks** (``bench dsa-micro``): the costs a Kimi-K3.1 sparse
attention path adds to a decode step, timed one JAX body at a time by
``dsa_micro.time_detail``'s device-loop slope (see ``dsa_micro``).

``--trace DIR`` (decode and prefill): rank 0 saves an xprof trace of each
case's last repeat.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from tools.megakernel import dsa_micro, harness, k3


def add_arguments(parser, profile: bool = False) -> None:
    sub = parser.add_subparsers(dest="role", required=True)
    decode = sub.add_parser("decode", help="the decode ops")
    decode.add_argument("--batches", default="1,8")
    decode.add_argument(
        "--contexts", default="1152", help="per-request context (all rows of a batch)"
    )
    decode.add_argument(
        "--mix",
        default="",
        help="also mixed batches, rows by context: e.g. 1152x4,131072x4; "
        "several batches separated by '/'",
    )
    decode.add_argument(
        "--compiled",
        default="",
        help="compiled batch sizes (default: the cases' own); a smaller batch "
        "runs padded in the next larger one",
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
        help="time DSpark verify + draft steps with this speculator (batches "
        "up to eight); without the flag every step is plain",
    )
    decode.add_argument(
        "--drafts",
        type=int,
        choices=(3, 7),
        default=0,
        help="with --dspark: this many drafts at every batch up to eight "
        "(default: serving's schedule)",
    )
    decode.add_argument(
        "--acceptance",
        default="",
        help="with --dspark: synthetic per-position acceptance rates of the "
        "seven drafts (comma separated; three drafts take the first three), "
        "e.g. 0.92,0.83,0.73,0.65,0.57,0.50,0.43; default: the drafts' own",
    )
    decode.add_argument(
        "--spec-drafts",
        type=int,
        choices=(0, 3, 7),
        default=0,
        help="with --dspark: this many drafts at every batch, any size (e.g. "
        "attention DP2's batched verify + draft); default: --drafts or "
        "serving's schedule",
    )
    decode.add_argument(
        "--save-tokens",
        default="",
        help="file: rank 0 saves each case's timed greedy tokens [steps, batch] (npz)",
    )
    decode.add_argument(
        "--token-file",
        default="",
        help="int32 .npy of real text: each row first decodes a prompt of its "
        "context from it (its own offset; the text repeats past its end), so "
        "the caches hold real latents and index keys; default: synthetic "
        "pattern pools at the context",
    )
    decode.add_argument(
        "--ingest-cache",
        default="",
        help="with --token-file: a directory (network storage) for each case's "
        "ingested pools, index keys and KDA states: the first run saves them "
        "after the prompts' replay, later runs with the same key reload them "
        "instead of replaying (plain decode; the same token streams)",
    )
    decode.add_argument(
        "--ingest-key",
        choices=("full", "base"),
        default="full",
        help="with --ingest-cache: full keys the saved state by the kernel config "
        "and decode code too; base leaves them out so flag arms of one tree "
        "share one ingest, valid only for arms whose ingested state is "
        "bit-identical to the saving run's",
    )
    decode.add_argument(
        "--fragment-pages",
        action="store_true",
        help="each row's pages in a fixed random order of the pool's blocks "
        "(serving's free-list order) instead of ascending",
    )
    prefill = sub.add_parser("prefill", help="the prefill stack")
    prefill.add_argument(
        "--prompts",
        default="8192x1,2048x4",
        help="cases LENxN[@PREFIX]: N prompts of LEN new tokens per step, each "
        "appended to a cached prefix of PREFIX tokens (a page-aligned hit)",
    )
    prefill.add_argument("--repeats", type=int, default=3)
    prefill.add_argument(
        "--token-file",
        default="",
        help="a .npy of int32 token ids: each prompt is a random slice of it "
        "(real text) instead of uniform random ids",
    )
    if not profile:
        dsa_micro.add_arguments(
            sub.add_parser("dsa-micro", help="the DSA decode microbenchmarks")
        )
    for role in (decode, prefill):
        role.add_argument(
            "--trace",
            default="",
            help="directory: rank 0 saves an xprof trace of each case",
        )
        if profile:
            role.add_argument(
                "--llo", default="", help="directory for the compiled LLO of every op"
            )


def _prompt_tokens(rng, corpus, size):
    """``size`` token ids: uniform random, or a random slice of ``corpus``."""
    if corpus is None:
        return rng.integers(1000, 100000, size, dtype=np.int32)
    if size > corpus.size:
        raise SystemExit(
            f"--token-file holds {corpus.size} tokens, a prompt needs {size}"
        )
    start = int(rng.integers(0, corpus.size - size + 1))
    return corpus[start : start + size]


@contextmanager
def _trace(directory: str, tag: str, last: bool):
    """Rank 0 saves an xprof trace of the last repeat under ``directory``
    (``TRACE_RANKS``, e.g. ``0,8,16,24``: those ranks, under ``rank<r>/``)."""
    ranks = [int(r) for r in os.environ.get("TRACE_RANKS", "0").split(",")]
    if not (directory and last and harness.rank() in ranks):
        yield
        return
    from torch_tpu._internal.profiler import _impl as profiler

    path = os.path.join(directory, tag)
    if ranks != [0]:
        path = os.path.join(directory, f"rank{harness.rank()}", tag)
    profiler.start_trace(path)
    try:
        yield
    finally:
        profiler.stop_trace()
        harness.log(f"trace of {tag}: {path}")


def run(args, records: harness.Records) -> None:
    if args.role == "dsa-micro":
        dsa_micro.run(args, records)
    else:
        (run_decode if args.role == "decode" else run_prefill)(args, records)


def run_decode(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import config
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import Control, DecodeOps

    batches = [int(b) for b in args.batches.split(",") if b]
    contexts = [int(c) for c in args.contexts.split(",")]
    mixes = [
        [
            int(part.split("x")[0])
            for part in filter(None, one.split(","))
            for _ in range(int(part.split("x")[1]))
        ]
        for one in filter(None, args.mix.split("/"))
    ]
    mix_labels = list(filter(None, args.mix.split("/")))
    cases = [(batch, context) for context in contexts for batch in batches] + [
        (len(mix), mix) for mix in mixes
    ]
    max_context = max(contexts + [c for mix in mixes for c in mix])
    rows = max(batch for batch, _ in cases)
    max_num_seqs = args.max_num_seqs or rows
    sub_base = 2 * max_num_seqs + 1
    pages = -(-(max_context + 8 * args.steps * args.repeats + 64) // k3.logical()) + 1
    # Under attention DP a rank's pools hold only its own group's requests.
    dp = config.active().attention_dp
    mla_blocks = 1 + -(-rows // dp) * pages
    if args.ingest_cache and (not args.token_file or args.dspark):
        raise SystemExit(
            "--ingest-cache takes --token-file and plain decode (no --dspark)"
        )
    # --ingest-cache: three KDA state blocks a row (one per KDA group) after the
    # MLA pages, where each row's state is exported and imported.
    blocks = mla_blocks + (3 * rows if args.ingest_cache else 0)
    sizes = (
        tuple(int(b) for b in args.compiled.split(","))
        if args.compiled
        else tuple(sorted({b for b, _ in cases if b > 1} | {1}))
    )
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(sub_base) if args.dspark else sub_base,
        batch_sizes=sizes,
        **k3.decode_ops_options(),
    )
    corpus = np.load(args.token_file).astype(np.int32) if args.token_file else None
    pools = k3.Pools(blocks, pages, 1, pattern=corpus is None)
    # Block ids by (row group, page): ascending, or one fixed shuffle of the blocks.
    block_ids = 1 + np.arange(mla_blocks - 1, dtype=np.int32)
    if args.fragment_pages:
        block_ids = np.random.default_rng(0).permutation(block_ids)

    # Serving's DSpark schedule: seven drafts through batch four, three
    # through eight, none above. --spec-drafts: that many at every batch;
    # --drafts: at every batch up to eight.
    def drafts_at(batch: int) -> int:
        if args.spec_drafts:
            return args.spec_drafts
        if args.drafts:
            return args.drafts if batch <= 8 else 0
        return 7 if batch <= 4 else 3 if batch <= 8 else 0

    draft = None
    if args.dspark and any(drafts_at(batch) for batch, _ in cases):
        from vllm_torchtpu.megakernel.kimi_k3.draft import draft_class

        rates = None
        if args.acceptance:
            seven = tuple(float(r) for r in args.acceptance.split(","))
            if len(seven) != 7:
                raise SystemExit("--acceptance takes seven rates")
            rates = {7: seven, 3: seven[:3]}
        draft = draft_class(args.dspark)(
            max_position=max_context + 8 * args.steps * args.repeats + 256,
            checkpoint=args.dspark,
            presharded=args.presharded,
            synthetic_rates_by_width=rates,
        )
    schedule = None
    if draft is not None and args.spec_drafts:
        schedule = ((1, max_num_seqs, args.spec_drafts),)
    elif draft is not None and args.drafts:
        spans = [(1, min(8, max_num_seqs), args.drafts)]
        if max_num_seqs > 8:
            spans.append((9, max_num_seqs, 0))
        schedule = tuple(spans)
    elif draft is not None:
        spans = [(1, min(4, max_num_seqs), 7)]
        if max_num_seqs > 4:
            spans.append((5, min(8, max_num_seqs), 3))
        if max_num_seqs > 8:
            spans.append((9, max_num_seqs, 0))
        schedule = tuple(spans)
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=max_num_seqs,
        max_model_len=pages * k3.logical(),
        draft=draft,
        schedule=schedule,
    )
    harness.sync()
    decoder.warmup()
    harness.log(
        f"warm: {len(pools.tensors)} pools of {blocks} blocks, batch sizes {sizes}"
    )
    sampling = (args.temperature, 0.95 if args.temperature > 0 else 1.0, 0, 7)

    def kda_blocks(r):
        return tuple(int(mla_blocks + 3 * r + g) for g in range(3))

    def rows_for(batch, context, first, tag, resumed=None):
        """The step's requests: on the first, each reset at its context with
        a clean KDA state; then continuing (the device holds their inputs).
        ``resumed``: the rows' first tokens of a reloaded ingest, each row
        resuming after its context from its state blocks."""
        per_row = context if isinstance(context, list) else [context] * batch
        out = []
        for r in range(batch):
            table = pools.scheduled(
                [block_ids[r // dp * pages + p] for p in range(pages)], r
            )
            speculating = draft is not None and drafts_at(batch) and not first
            query = drafts_at(batch) + 1 if speculating else 1
            tokens = [1000 + 37 * r] if first else np.zeros(query, np.int32)
            computed = per_row[r] - 1 if first else 0
            if resumed is not None:
                out.append(
                    k3.request(
                        f"{tag}-{r}",
                        [int(resumed[r])],
                        table,
                        sampling,
                        per_row[r],
                        kda=kda_blocks(r),
                    )
                )
                continue
            if first and corpus is not None:  # a real prompt of the row's context
                start = 7919 * r % corpus.size
                tokens = np.resize(np.roll(corpus, -start), per_row[r])
                computed = 0
            out.append(k3.request(f"{tag}-{r}", tokens, table, sampling, computed))
        return mk.StepInput(out)

    ingest = None
    if args.ingest_cache:
        token_file = Path(args.token_file)
        ingest = k3.IngestCache(
            args.ingest_cache,
            {
                "presharded": str(args.presharded),
                "layout": (Path(args.presharded) / "layout.json").read_text(),
                "token_file": [
                    str(token_file.resolve()),
                    token_file.stat().st_size,
                    hashlib.sha256(np.ascontiguousarray(corpus).tobytes()).hexdigest(),
                ],
                "pages": pages,
                "blocks": blocks,
                "rows": rows,
                "dp": dp,
                "fragment_pages": bool(args.fragment_pages),
                "block_ids": hashlib.sha256(block_ids.tobytes()).hexdigest(),
                "max_num_seqs": max_num_seqs,
                "sizes": sizes,
                "temperature": args.temperature,
            },
            scope=args.ingest_key,
        )
        harness.log(
            f"ingest cache: {ingest.scope} key, {ingest.root}"
            + (
                " (valid only for arms whose ingested state is bit-identical"
                " to the saving run's)"
                if ingest.scope == "base"
                else ""
            )
        )

    def first_step(batch, context, tag):
        """The case's first step and the warm step after it, or with a saved
        ingest one step that resumes the rows (the fresh run's second)."""
        per_row = context if isinstance(context, list) else [context] * batch
        if ingest is not None and ingest.ready(tag):
            started = time.perf_counter()
            resumed = ingest.load(tag, pools)
            harness.log(
                f"{tag}: ingest loaded from {ingest.root} in "
                f"{time.perf_counter() - started:.1f} s"
            )
            bridge.materialize(
                decoder.step(rows_for(batch, context, True, tag, resumed)).tokens
            )
            return
        started = time.perf_counter()
        first = decoder.step(rows_for(batch, context, True, tag)).tokens
        if ingest is not None:
            first = first[:batch, 0].cpu().numpy()
            harness.log(
                f"{tag}: {sum(per_row)} prompt tokens ingested in "
                f"{time.perf_counter() - started:.1f} s"
            )
            started = time.perf_counter()
            # Each row's state after its context into its own blocks (as a
            # transfer leaves it).
            from vllm_torchtpu.megakernel.kimi_k3.pools import kda_blocks_at_length

            for r in range(batch):
                tables = (
                    rows_for(batch, context, False, tag, first).requests[r].block_tables
                )
                kda = kda_blocks_at_length(tables, decoder._kda_groups, per_row[r])
                slot = decoder.slot_map.allocate(f"{tag}-{r}")
                decoder._importer.export_state(*decoder.ops.slots, kda, slot)
            ingest.save(tag, pools, first)
            harness.log(
                f"{tag}: ingest saved to {ingest.root} in "
                f"{time.perf_counter() - started:.1f} s"
            )
        bridge.materialize(decoder.step(rows_for(batch, context, False, tag)).tokens)

    results = []
    streams = {}
    for batch, context in cases:
        mixed = isinstance(context, list)
        label = "mix " + mix_labels[mixes.index(context)] if mixed else context
        tag = f"b{batch}mix{mixes.index(context)}" if mixed else f"b{batch}c{context}"
        # First step: rows reset at the context.
        first_step(batch, context, tag)
        times, outs = [], []
        for repeat in range(args.repeats):
            with _trace(args.trace, tag, repeat == args.repeats - 1):
                started = time.perf_counter()
                for _ in range(args.steps):
                    outs.append(
                        decoder.step(rows_for(batch, context, False, tag)).tokens
                    )
                bridge.materialize(outs[-1])
                times.append(1e3 * (time.perf_counter() - started) / args.steps)
        ms = statistics.median(times)
        dspark = draft is not None and drafts_at(batch) > 0
        if dspark:
            tokens = torch.cat([o[:batch] for o in outs]).cpu()
            per_step = float((tokens >= 0).sum()) / len(outs) / batch
            rate = 1e3 * per_step * batch / ms
            line = (
                f"B{batch} ctx {label}: DSpark x{drafts_at(batch)} "
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
        decoder.forget([f"{tag}-{r}" for r in range(batch)])
    if args.save_tokens and harness.rank() == 0:
        np.savez(args.save_tokens, **streams)
    harness.log("summary: " + "; ".join(f"B{b}@{c}={ms:.2f}ms" for b, c, ms in results))


def run_prefill(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import prefill as k3_prefill
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program

    cases = []
    for part in filter(None, args.prompts.split(",")):
        prompt, _, prefix = part.partition("@")
        length, count = (int(x) for x in prompt.split("x"))
        prefix = int(prefix) if prefix else 0
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
    max_seqs = max(count for _, count, _ in cases)
    totals = [length + prefix for length, _, prefix in cases]
    max_model_len = -(-max(totals) // k3.LOGICAL) * k3.LOGICAL
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=max_seqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=k3_program.MAX_STEP_TOKENS,
        ),
        args.presharded,
        tp_rank=torch.distributed.get_rank(),
    )
    prefiller.load_weights()
    # Per request slot: real MLA pages for the longest case that uses it (the
    # window reads and writes them), but a KDA state block only at the
    # columns a step touches -- a hit's and the final one; the rest of the
    # row is the null block, so a long-context case costs its MLA pages only.
    slot_layout = []
    for slot in range(max_seqs):
        longest = max(
            total for total, (_, count, _) in zip(totals, cases) if count > slot
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
        1 + sum(mla + 3 * len(columns) for mla, columns in slot_layout),
        max_model_len // k3.LOGICAL,
        max_model_len // k3.PAGE,
        pattern=True,
    )
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    prefiller.warmup()
    harness.sync()
    # Each request gets its own blocks once; repeats reuse them under new ids.
    tables = []
    for mla, columns in slot_layout:
        kda = []
        for _ in range(3):
            row = np.zeros(max_model_len // k3.PAGE, np.int32)
            for column in columns:
                row[column] = pools.fresh(1)[0]
            kda.append(row)
        tables.append(pools.tables(pools.fresh(mla), kda))
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
        times = []
        for repeat in range(args.repeats + 1):  # the first is a warm-up of this shape
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
            with _trace(args.trace, tag, repeat == args.repeats):
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
