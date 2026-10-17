# SPDX-License-Identifier: Apache-2.0
"""Kernel-only timing of the serving ops (``bench decode`` / ``bench prefill``).

**Decode** times ``KimiDecoder.step`` over vLLM's 24 unified pools for every
``--batches`` x ``--contexts`` case (``--mix``: one batch of rows at different
contexts), or with ``--dspark [CHECKPOINT]`` the DSpark verify + draft step of
one request. Each request starts at its context over synthetic KV pages and a
clean KDA state, with distinct tokens per row (they set MoE routing); ``--steps``
steps are enqueued back to back and timed with one host read, median of
``--repeats``. DSpark's tokens per step on synthetic contexts are not
representative.

**Prefill** times :class:`KimiPrefiller.step` per ``--prompts`` case: ``LENxN``
is one step of ``N`` prompts of ``LEN`` new tokens, ``LENxN@PREFIX`` appends
them to a page-aligned cached prefix. Each line names the compiled windows the
case packs into.

``--trace DIR`` (both): rank 0 saves an xprof trace of each case's last repeat.
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager

import numpy as np

from tools.megakernel import harness, k3


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
        help="time B1 as DSpark verify + draft with this speculator; without "
        "the flag B1 is the plain step",
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
    """Rank 0 saves an xprof trace of the last repeat under ``directory``."""
    if not (directory and last and harness.rank() == 0):
        yield
        return
    from torch_tpu._internal.profiler import _impl as profiler

    path = os.path.join(directory, tag)
    profiler.start_trace(path)
    try:
        yield
    finally:
        profiler.stop_trace()
        harness.log(f"trace of {tag}: {path}")


def run(args, records: harness.Records) -> None:
    (run_decode if args.role == "decode" else run_prefill)(args, records)


def run_decode(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import config
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import Control, DecodeOps

    batches = [int(b) for b in args.batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    mix = [
        int(part.split("x")[0])
        for part in filter(None, args.mix.split(","))
        for _ in range(int(part.split("x")[1]))
    ]
    cases = [(batch, context) for context in contexts for batch in batches] + (
        [(len(mix), mix)] if mix else []
    )
    max_context = max(contexts + mix)
    rows = max(batch for batch, _ in cases)
    max_num_seqs = args.max_num_seqs or rows
    sub_base = 2 * max_num_seqs + 1
    pages = -(-(max_context + 8 * args.steps * args.repeats + 64) // k3.logical()) + 1
    # Under attention DP a rank's pools hold only its own group's requests.
    dp = config.active().attention_dp
    blocks = 1 + -(-rows // dp) * pages
    sizes = tuple(sorted({b for b, _ in cases if b > 1} | {1}))
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(sub_base) if args.dspark else sub_base,
        batch_sizes=sizes,
    )
    pools = k3.Pools(blocks, pages, 1, pattern=True)
    draft = None
    if args.dspark and any(batch == 1 for batch, _ in cases):
        from vllm_torchtpu.megakernel.kimi_k3.draft import DsparkDraft

        draft = DsparkDraft(
            max_position=max_context + 8 * args.steps * args.repeats + 256,
            checkpoint=args.dspark,
            presharded=args.presharded,
        )
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=max_num_seqs,
        max_model_len=pages * k3.logical(),
        draft=draft,
        # The bench speculates on batch-one rows only (7 drafts).
        schedule=((1, 1, 7), (2, max(2, max_num_seqs), 0)) if draft else None,
    )
    harness.sync()
    decoder.warmup()
    harness.log(
        f"warm: {len(pools.tensors)} pools of {blocks} blocks, batch sizes {sizes}"
    )
    sampling = (args.temperature, 0.95 if args.temperature > 0 else 1.0, 0, 7)

    def rows_for(batch, context, first, tag):
        """The step's requests: on the first, each reset at its context with
        a clean KDA state; then continuing (the device holds their inputs)."""
        per_row = context if isinstance(context, list) else [context] * batch
        out = []
        for r in range(batch):
            table = pools.scheduled([1 + r // dp * pages + p for p in range(pages)], r)
            query = 8 if batch == 1 and draft is not None and not first else 1
            tokens = [1000 + 37 * r] if first else np.zeros(query, np.int32)
            computed = per_row[r] - 1 if first else 0
            out.append(k3.request(f"{tag}-{r}", tokens, table, sampling, computed))
        return mk.StepInput(out)

    results = []
    streams = {}
    for batch, context in cases:
        label = context if not isinstance(context, list) else "mix " + args.mix
        tag = f"b{batch}c{context if not isinstance(context, list) else 'mix'}"
        # First step: rows reset at the context.
        decoder.step(rows_for(batch, context, True, tag))
        bridge.materialize(decoder.step(rows_for(batch, context, False, tag)).tokens)
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
        dspark = batch == 1 and draft is not None
        if dspark:
            tokens = torch.cat([o[:1] for o in outs]).cpu()
            per_step = float((tokens >= 0).sum()) / len(outs)
            rate = 1e3 * per_step / ms
            line = (
                f"B1 ctx {label}: DSpark {ms:.2f} ms per step, "
                f"{per_step:.2f} tokens per step (synthetic) -> {rate:.0f} tok/s"
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
