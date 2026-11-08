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

# Source provenance: inferact/k3-megakernel@30c9ba966b tools/megakernel/check.py
"""Numerics checks of the K3 programs.

Runs ``check spec``, ``check handoff``, ``check packing``, ``check greedy``,
and ``check masked``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch

from .. import prefill as k3_prefill
from .. import program as k3_program
from ..decode_ops import Control, DecodeOps, decode_batch_sizes
from ..draft import draft_class
from ..pools import kda_blocks_at_length
from ..program import VOCAB
from ..runtime_shim import bridge
from ..runtime_shim import program as mk
from . import harness, k3

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping, Sequence

    from ..decode import KimiDecoder
    from ..prefill import KimiPrefiller


class _TokenizerLike(Protocol):
    """Protocol for tokenizers used in megakernel verification tools."""

    def encode(
        self, text: str, *, disallowed_special: tuple[()] = ()
    ) -> Sequence[int] | np.ndarray:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: Sequence[int] | np.ndarray) -> str:
        """Decode token IDs into text."""
        ...


_VOCAB_WORD_BITS: int = 32
_DEFAULT_SEED_A: int = 7
_DEFAULT_SEED_B: int = 8
_MAX_SPEC_CHURN_MS: float = 1000.0
_PACKING_PAGES: tuple[int, ...] = (1, 2, 1, 4, 1, 3, 1, 2, 1, 1, 5, 2, 1, 3, 2, 2)
_PACKING_PREFIX_PAGES: tuple[int, ...] = (2, 4, 8, 3)
_MAGIC_4: int = 4
_LAST_WINDOW_IDX: int = 2
_BATCH_STEPS_112: int = 111


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for numerics check subcommands.

    Args:
        parser: Target CLI argument parser to configure.
    """
    subs = parser.add_subparsers(dest="check", required=True)

    spec = subs.add_parser("spec", help="decoder speculation numerics and step times")
    spec.add_argument("--prompt-tokens", type=int, default=128)
    spec.add_argument("--generate", type=int, default=64)
    spec.add_argument("--single-drafts", type=int, default=7)
    spec.add_argument("--small-batch-drafts", type=int, default=3)
    spec.add_argument("--dspark", type=str, default=None)
    spec.add_argument(
        "--large-batches",
        type=str,
        default="",
        help="comma-separated batch sizes to check past 8",
    )
    spec.add_argument(
        "--require-verifier-parity",
        action="store_true",
        help="require sequential verifier parity on large batches",
    )

    handoff = subs.add_parser("handoff", help="prefill-to-decode handoff numerics")
    handoff.add_argument("--generate", type=int, default=64)
    handoff.add_argument("--single-drafts", type=int, default=7)
    handoff.add_argument("--speculate-steps", type=int, default=128)
    handoff.add_argument("--dspark", type=str, default=None)

    packing = subs.add_parser("packing", help="chunked packing numerics against alone")
    packing.add_argument("--dspark", type=str, default=None)

    greedy = subs.add_parser("greedy", help="dump greedy decode tokens to a JSON file")
    greedy.add_argument("--output", type=str, required=True)
    greedy.add_argument("--generate", type=int, default=128)
    greedy.add_argument("--prompt-tokens", type=int, default=128)

    masked = subs.add_parser("masked", help="decoder bitmask grammar checks")
    masked.add_argument("--generate", type=int, default=32)
    masked.add_argument("--prompt-tokens", type=int, default=128)
    masked.add_argument("--single-drafts", type=int, default=7)
    masked.add_argument("--dspark", type=str, default=None)

    for sub in (spec, handoff, packing, greedy, masked):
        sub.add_argument(
            "--model",
            type=str,
            required=True,
            help="checkpoint path for tokenizer and prompt generation",
        )


def run(args: argparse.Namespace, records: harness.Records) -> None:
    """Dispatch verification execution to the selected check subcommand.

    Args:
        args: Parsed CLI namespace containing subcommand name and flags.
        records: Structured run record collector.
    """
    dispatch = {
        "spec": run_spec,
        "handoff": run_handoff,
        "packing": run_packing,
        "greedy": run_greedy,
        "masked": run_masked,
    }
    dispatch[args.check](args, records)


def _agree(a: Sequence[int], b: Sequence[int]) -> int:
    """Count how many leading tokens two sequences have in common.

    Args:
        a: First sequence of token IDs.
        b: Second sequence of token IDs.

    Returns:
        Number of identical leading elements.
    """
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def run_greedy(args: argparse.Namespace, records: harness.Records) -> None:
    """Run greedy decode over reference prompts and save emitted tokens.

    Args:
        args: Parsed CLI flags containing prompt sizes and output paths.
        records: Structured run record collector.
    """
    count, generate = args.prompt_tokens, args.generate
    all_ids, raw_tokenizer = k3.fixed_prompt(count + 4000, args.model)
    tokenizer = cast("_TokenizerLike", raw_tokenizer)
    fixed = np.asarray(all_ids[:count], np.int32)

    def encode(text: str) -> np.ndarray:
        """Encode given text into token ID array."""
        return np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)

    prompts = {
        "fixed": fixed,
        "lighthouse": encode("The history of the lighthouse begins"),
        "robot": encode("Write a short story about a robot who learns to paint."),
        "transistor": encode("Explain step by step how a transistor amplifies."),
        "primes": encode("List the prime numbers below fifty and say why each."),
    }
    longest = max(map(len, prompts.values()))
    pages = -(-(longest + generate + 64) // k3.logical()) + 1
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(16),
        batch_sizes=decode_batch_sizes(len(prompts)),
    )
    pools = k3.Pools((len(prompts) + 1) * pages, pages, 1)
    decoder = k3.decoder(
        ops, pools, max_num_seqs=len(prompts), max_model_len=pages * k3.logical()
    )
    tables = {
        name: pools.scheduled(
            np.arange(1 + n * pages, 1 + (n + 1) * pages, dtype=np.int32), n
        )
        for n, name in enumerate(prompts)
    }

    decoder.warmup()
    harness.sync()

    def decode_step(
        requests: list[mk.RequestInput], finished: tuple[str, ...] = ()
    ) -> torch.Tensor:
        """Submit one decode step input to the decoder and return emitted tokens."""
        return decoder.step(mk.StepInput(requests, finished)).tokens

    outs: dict[str, list[torch.Tensor]] = {k: [] for k in prompts}
    first_rows = [k3.request(name, prompts[name], tables[name]) for name in prompts]
    step_out = decode_step(first_rows)[: len(prompts)]
    for i, name in enumerate(prompts):
        outs[name].append(step_out[i : i + 1])

    following_rows = [k3.request(name, [0], tables[name]) for name in prompts]
    for _ in range(generate - 1):
        step_out = decode_step(following_rows)[: len(prompts)]
        for i, name in enumerate(prompts):
            outs[name].append(step_out[i : i + 1])

    decoder.forget(tuple(prompts))

    result = {}
    for name, tensors in outs.items():
        tokens = [
            int(t) for t in torch.cat(tensors).cpu().reshape(-1).tolist() if t >= 0
        ]
        result[name] = {"tokens": tokens, "text": tokenizer.decode(tokens)}
        harness.log(f"{name}: {result[name]['text']!r}")

    records.add(case="greedy", results=result)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def _check_forced_tokens(
    decoder: KimiDecoder,
    tokens: Sequence[int],
    table: np.ndarray,
    label: str,
) -> bool:
    """Check that single-bit masks strictly force the expected tokens."""
    word_bits = _VOCAB_WORD_BITS
    rows = []
    first = True
    for tok in tokens:
        mask = np.zeros((1, VOCAB // word_bits), dtype=np.int32)
        mask[0, tok // word_bits] = 1 << (tok % word_bits)
        req = k3.request("f", [tok] if first else [0], table)
        first = False
        rows.append(decoder.step(mk.StepInput([req], mask=mask)).tokens[0])
    decoder.forget(("f",))
    emitted = [int(t[0].cpu()) for t in rows]
    matches = emitted == list(tokens)
    harness.log(f"{label}: forced tokens match exactly: {matches}")
    return matches


def _check_mixed_batch(
    decoder: KimiDecoder,
    tokens: Sequence[int],
    tables: dict[str, np.ndarray],
) -> bool:
    """Check batch with mixed unconstrained and masked streams."""
    word_bits = _VOCAB_WORD_BITS
    tok = tokens[0]
    mask_b = np.zeros(VOCAB // word_bits, dtype=np.int32)
    mask_b[tok // word_bits] = 1 << (tok % word_bits)
    mask = np.stack(
        [
            np.full((VOCAB // word_bits,), -1, dtype=np.int32),
            mask_b,
        ]
    )

    req_a = k3.request("ma", [0], tables["ma"])
    req_b = k3.request("mb", [0], tables["mb"])
    out = decoder.step(mk.StepInput([req_a, req_b], mask=mask)).tokens[:2]
    decoder.forget(("ma", "mb"))
    res_b = int(out[1, 0].cpu())
    ok = res_b == tok
    harness.log(f"mixed batch: masked stream matched forced token: {ok}")
    return ok


def run_masked(args: argparse.Namespace, records: harness.Records) -> None:
    """Drive decoder with vocabulary bitmasks (structured outputs).

    Args:
        args: Parsed CLI namespace.
        records: Structured run record collector.
    """
    count = args.prompt_tokens
    all_ids, raw_tokenizer = k3.fixed_prompt(count + 2000, args.model)
    tokenizer = cast("_TokenizerLike", raw_tokenizer)
    ids = np.asarray(all_ids[:count], np.int32)

    def encode(text: str) -> np.ndarray:
        """Tokenize input text into an int32 array without special filtering."""
        return np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)

    prompts = {
        "A": ids,
        "B": encode("The history of the lighthouse begins"),
    }
    pages = -(-(count + 128) // k3.logical()) + 1
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(8),
        batch_sizes=decode_batch_sizes(2),
    )
    pools = k3.Pools(8 * pages, pages, 1)
    draft = None
    if args.dspark:
        draft = draft_class(args.dspark)(
            max_position=count + 256,
            checkpoint=args.dspark,
            presharded=args.presharded,
        )

    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=2,
        max_model_len=pages * k3.logical(),
        draft=draft,
    )
    decoder.warmup()
    harness.sync()

    tables = {
        name: pools.scheduled(
            np.arange(1 + n * pages, 1 + (n + 1) * pages, dtype=np.int32), n
        )
        for n, name in enumerate(("A", "B", "ref", "ma", "mb", "f"))
    }

    ref_req = k3.request("ref", ids, tables["ref"])
    ref_outs = [decoder.step(mk.StepInput([ref_req])).tokens[:1]]
    ref_outs.extend(
        decoder.step(mk.StepInput([k3.request("ref", [0], tables["ref"])])).tokens[:1]
        for _ in range(16)
    )
    decoder.forget(("ref",))
    reference = [
        int(t) for t in torch.cat(ref_outs).cpu().reshape(-1).tolist() if t >= 0
    ]

    ok_forced = _check_forced_tokens(
        decoder, reference[:8], tables["f"], "forced single"
    )
    ok_mixed = _check_mixed_batch(decoder, reference[:8], tables)

    ok_spec = True
    if draft is not None:
        ok_spec = _masked_around_speculation(
            decoder,
            [ids, prompts["B"]],
            [tables["A"], tables["B"]],
            ("A", "B"),
            reference,
        )

    all_ok = ok_forced and ok_mixed and ok_spec
    records.add(case="masked", ok=all_ok)
    if not all_ok:
        harness.fail("masked grammar check")


def _masked_around_speculation(
    decoder: KimiDecoder,
    prompts: list[np.ndarray],
    tables: list[np.ndarray],
    names: tuple[str, str],
    plain_ref: list[int],
) -> bool:
    """Verify that masking turns off speculation cleanly."""
    word_bits = _VOCAB_WORD_BITS
    ones = np.full((2, VOCAB // word_bits), -1, dtype=np.int32)

    def run_window_steps(
        tag: str,
        windows: Sequence[tuple[str, np.ndarray | None, int]],
    ) -> tuple[list[list[int]], list[list[int]], bool, bool]:
        """Execute windows of speculation and masking steps."""
        streams: list[list[int]] = [[], []]
        bounds: list[list[int]] = []
        stepped_ok = True
        resumed = False
        for window, (label, mask, steps) in enumerate(windows):
            before = dict(decoder.stats)
            first = window == 0
            for _ in range(steps):
                requests = [
                    k3.request(tag + n, ids if first else np.zeros(4, np.int32), table)
                    for n, ids, table in zip(names, prompts, tables, strict=True)
                ]
                rows = decoder.step(mk.StepInput(requests, mask=mask)).tokens[:2]
                rows_np = rows.cpu().numpy()
                for n in range(2):
                    tokens = [int(t) for t in rows_np[n] if t >= 0]
                    streams[n].extend(tokens)
                    resumed |= window == _LAST_WINDOW_IDX and len(tokens) > 1
                first = False
            bounds.append([len(st) for st in streams])
            speculated = decoder.stats["batched"] - before.get("batched", 0)
            plain = decoder.stats["plain"] - before.get("plain", 0)
            expected = plain == steps if label == "masked" else speculated > 0
            stepped_ok &= expected
            harness.log(
                f"dspark {tag}, {label} window: {speculated} speculative and "
                f"{plain} plain steps, streams end at {bounds[-1]} "
                f"({'ok' if expected else 'WRONG'})"
            )
        decoder.forget(tuple(tag + n for n in names))
        return streams, bounds, resumed, stepped_ok

    masked_windows = (
        ("speculate", None, 6),
        ("masked", ones, 6),
        ("speculate", None, 10),
    )
    pure_windows = (
        ("speculate", None, 6),
        ("speculate", None, 6),
        ("speculate", None, 10),
    )
    mixed, bounds, resumed, stepped_ok = run_window_steps("s", masked_windows)
    pure, _, _, _ = run_window_steps("n", pure_windows)
    vs_plain = [_agree(st, plain_ref) for st in mixed]
    pure_vs_plain = [_agree(st, plain_ref) for st in pure]
    vs_pure = [_agree(a, b) for a, b in zip(mixed, pure, strict=True)]
    harness.log(
        f"dspark, agreement (tokens): masked run vs plain {vs_plain}, pure DSpark "
        f"vs plain {pure_vs_plain}, masked run vs pure DSpark {vs_pure}; the "
        f"masked window spans tokens {bounds[0]} to {bounds[1]}; speculation "
        f"resumed after it: {resumed}"
    )
    return stepped_ok and resumed and all(a > 0 for a in vs_plain)


def _tokens_of(emitted: Sequence[torch.Tensor]) -> list[int]:
    """Extract flattened nonnegative token IDs from tensor sequence."""
    return [
        int(t) for t in torch.cat(list(emitted)).cpu().reshape(-1).tolist() if t >= 0
    ]


def _spec_churn(
    batched: KimiDecoder,
    prompts: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
    verify_rows_fn: Callable[[int], int],
) -> tuple[float, float]:
    """Requests joining and leaving across speculation widths right after warmup."""
    short = {n: prompts[n][:24] for n in prompts}
    order = [
        ("A",),
        ("A", "B"),
        tuple("ABCD"),
        tuple("ABCDEFGH"),
        tuple("ABC"),
        ("C",),
        ("C", "D"),
        tuple("CDEFG"),
        tuple("ABCDEFGH"),
        ("H",),
    ]
    live: set[str] = set()
    worst_join = 0.0
    worst_steady = 0.0
    log = []
    for names in order:
        leaving = tuple("c" + n for n in sorted(live - set(names)))
        times = []
        for step in range(4):
            group = []
            for n in names:
                if n in live:
                    q = verify_rows_fn(len(names))
                    group.append(k3.request("c" + n, np.zeros(q), tables[n]))
                else:
                    group.append(k3.request("c" + n, short[n], tables[n]))
            t0 = time.perf_counter()
            out = batched.step(mk.StepInput(group, leaving if step == 0 else ())).tokens
            bridge.materialize(out)
            times.append(1e3 * (time.perf_counter() - t0))
            live = set(names)
        worst_join = max(worst_join, times[0])
        worst_steady = max(worst_steady, *times[1:])
        log.append(f"{''.join(names)} " + "/".join(f"{t:.0f}" for t in times))
    batched.forget(tuple("c" + n for n in live))
    harness.log("churn after warmup, ms per step: " + "; ".join(log))
    return worst_join, worst_steady


@dataclasses.dataclass(frozen=True)
class _SpecGroupConfig:
    """Config bundle for spec group execution."""

    names: Sequence[str]
    tag: str
    sampling: tuple[float, float, int, int]
    steps: int


def _run_spec_group(
    batched: KimiDecoder,
    cfg: _SpecGroupConfig,
    prompts: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
    verify_rows_fn: Callable[[int], int],
) -> tuple[list[list[int]], list[float]]:
    """Run steps for a specific group of requests."""
    first = [
        k3.request(cfg.tag + n, prompts[n], tables[n], cfg.sampling) for n in cfg.names
    ]
    k = verify_rows_fn(len(cfg.names))
    following = [
        k3.request(cfg.tag + n, np.zeros(k), tables[n], cfg.sampling) for n in cfg.names
    ]
    count_ = len(cfg.names)
    outs = [batched.step(mk.StepInput(first)).tokens[:count_]]
    outs.extend(
        batched.step(mk.StepInput(following)).tokens[:count_] for _ in range(cfg.steps)
    )
    stacked = torch.stack(outs).cpu().numpy()
    streams = [
        [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(count_)
    ]
    rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(count_)]
    return streams, rates


@dataclasses.dataclass(frozen=True)
class _PlainGroupConfig:
    """Config bundle for plain group evaluation."""

    names: Sequence[str]
    streams: Sequence[Sequence[int]]
    prompts: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]
    references: Mapping[str, Sequence[int]]


def _plain_group(
    plain: KimiDecoder,
    cfg: _PlainGroupConfig,
) -> tuple[list[int], list[int]]:
    """Check plain batch greedy agreement."""
    first = [k3.request("g" + m, cfg.prompts[m], cfg.tables[m]) for m in cfg.names]
    following = [k3.request("g" + m, [0], cfg.tables[m]) for m in cfg.names]
    outs = [plain.step(mk.StepInput(first)).tokens[: len(cfg.names)]]
    outs.extend(
        plain.step(mk.StepInput(following)).tokens[: len(cfg.names)]
        for _ in range(_BATCH_STEPS_112)
    )
    plain.forget(tuple("g" + m for m in cfg.names))
    stacked = torch.stack(outs).cpu().numpy()
    tokens = [[int(t) for t in stacked[:, r, 0]] for r in range(len(cfg.names))]
    agree = [
        _agree(t, cfg.references[m]) for t, m in zip(tokens, cfg.names, strict=True)
    ]
    follow = [_agree(st, t) for st, t in zip(cfg.streams, tokens, strict=True)]
    harness.log(
        f"plain batch-{len(cfg.names)} greedy: "
        + "; ".join(
            f"{m} agrees for {a} of 112 (batched follows for {f})"
            for m, a, f in zip(cfg.names, agree, follow, strict=True)
        )
    )
    return agree, follow


@dataclasses.dataclass(frozen=True)
class _PipelinedFixture:
    """Fixture tables and prompts for pipelined latency measurement."""

    names: Sequence[str]
    prompts: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]


def _pipelined_step_ms(
    decoder: KimiDecoder,
    req_and_query: tuple[int, int],
    fix: _PipelinedFixture,
) -> float:
    """Measure pipelined step latency."""
    requests, query = req_and_query
    first = [
        k3.request(f"q{r}", fix.prompts[n], fix.tables[n])
        for r, n in enumerate(fix.names[:requests])
    ]
    decoder.step(mk.StepInput(first))
    following = [
        k3.request(f"q{r}", np.zeros(query), fix.tables[n])
        for r, n in enumerate(fix.names[:requests])
    ]
    for _ in range(4):
        out = decoder.step(mk.StepInput(following)).tokens
    bridge.materialize(out)
    started = time.perf_counter()
    for _ in range(40):
        out = decoder.step(mk.StepInput(following)).tokens
    bridge.materialize(out)
    decoder.forget(tuple(f"q{r}" for r in range(requests)))
    return 1e3 * (time.perf_counter() - started) / 40


@dataclasses.dataclass(frozen=True)
class _ArrivalFixture:
    """Fixture bundle for arrival verification."""

    prompts: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]
    rows: int
    verify_rows_fn: Callable[[int], int]


def _arriving_arrival(
    batched: KimiDecoder,
    name: str,
    joining_mode: str,
    fix: _ArrivalFixture,
) -> list[int]:
    """Test transferred arrival first token alone vs joining speculation."""
    prompts, tables, rows, verify_rows_fn = (
        fix.prompts,
        fix.tables,
        fix.rows,
        fix.verify_rows_fn,
    )
    n, req = len(prompts[name]), "t" + name
    if joining_mode == "alone":
        arrival = k3.request(req, prompts[name][-1:], tables[name], computed=n - 1)
        out = batched.step(mk.StepInput([arrival])).tokens[:1]
        batched.forget((req,))
        return _tokens_of([out])
    batched.step(mk.StepInput([k3.request("tA", prompts["A"], tables["A"])]))
    for _ in range(4):
        batched.step(mk.StepInput([k3.request("tA", np.zeros(rows), tables["A"])]))
    k = verify_rows_fn(2)
    tokens = np.concatenate((prompts[name][-1:], np.full(k - 1, -1)))
    arrival = k3.request(req, tokens, tables[name], computed=n - 1)
    arrival = dataclasses.replace(arrival, num_tokens=n)
    out = batched.step(
        mk.StepInput([k3.request("tA", np.zeros(k), tables["A"]), arrival])
    ).tokens[1:2]
    batched.forget(("tA", req))
    return _tokens_of([out])


def _spec_single_stream_loop(
    single: KimiDecoder,
    rows: int,
    prompts: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
    reference: Sequence[int],
) -> tuple[list[int], int, float, list[int], float, bool, bool]:
    """Run greedy DSpark alone, sampling, and measure single-stream stats."""

    def run_alone(
        req: str,
        sampling: tuple[float, float, int, int],
        n_steps: int,
        first_step_mode: str = "init",
    ) -> list[torch.Tensor]:
        """Decode ``n_steps`` single-stream steps and return output tensors."""
        outs = []
        if first_step_mode == "init":
            outs.append(
                single.step(
                    mk.StepInput([k3.request(req, prompts["A"], tables["A"], sampling)])
                ).tokens[:1]
            )
        outs.extend(
            single.step(
                mk.StepInput([k3.request(req, np.zeros(rows), tables["A"], sampling)])
            ).tokens[:1]
            for _ in range(n_steps)
        )
        return outs

    a_tokens = _tokens_of(run_alone("a", (0.0, 1.0, 0, 0), 60))
    same = _agree(a_tokens, reference)
    per_step = (len(a_tokens) - 1) / 60

    outs_b = []
    out = single.step(
        mk.StepInput(
            [
                k3.request("a", [0], tables["A"]),
                k3.request("b", prompts["B"], tables["B"]),
            ]
        )
    ).tokens
    outs_b.append(out[1:2])
    for _ in range(23):
        out = single.step(
            mk.StepInput(
                [
                    k3.request("a", [0], tables["A"]),
                    k3.request("b", [0], tables["B"]),
                ]
            )
        ).tokens
        outs_b.append(out[1:2])
    b_tokens = _tokens_of(outs_b)
    single.forget(("a",))

    runs = []
    for req, seed in (
        ("s1", _DEFAULT_SEED_A),
        ("s2", _DEFAULT_SEED_A),
        ("s3", _DEFAULT_SEED_B),
    ):
        runs.append(_tokens_of(run_alone(req, (0.7, 0.9, 50, seed), 12)))
        single.forget((req,))
    reproducible = runs[0] == runs[1]
    varies = runs[0] != runs[2]

    run_alone("t", (0.0, 1.0, 0, 0), 4)
    if single.control is not None:
        bridge.materialize(single.control)
    started = time.perf_counter()
    for _ in range(40):
        out = single.step(
            mk.StepInput([k3.request("t", np.zeros(rows), tables["A"])])
        ).tokens
    bridge.materialize(out)
    step_ms = 1e3 * (time.perf_counter() - started) / 40
    single.forget(("t",))
    return (
        a_tokens,
        same,
        per_step,
        b_tokens,
        float(round(step_ms, 3)),
        reproducible,
        varies,
    )


@dataclasses.dataclass(frozen=True)
class _BatchedFlowFixture:
    """Fixture bundle for batched speculation flows."""

    prompts: dict[str, np.ndarray]
    tables: dict[str, np.ndarray]
    rows: int
    reference: Sequence[int]
    references: Mapping[str, Sequence[int]]
    verify_rows: Callable[[int], int]


@dataclasses.dataclass(frozen=True)
class _SpecBatchedResults:
    """Results from batched speculation execution."""

    pair_rates: list[float]
    four_rates: list[float]
    eight_rates: list[float]
    same_pa: int
    same_pb: int
    same_more: int
    four_agree: list[int]
    plain_four: list[int]
    follow_four: list[int]
    eight_agree: list[int]
    plain_eight: list[int]
    follow_eight: list[int]
    five_agree: list[int]
    three_agree: list[int]
    b_reproducible: bool
    b_varies: bool
    timing: dict[int, tuple[float, float]]


def _spec_run_batched_flows(
    batched: KimiDecoder,
    plain: KimiDecoder,
    fix: _BatchedFlowFixture,
) -> _SpecBatchedResults:
    """Measure speculative acceptance, reference agreement, and step latencies."""
    prompts, tables, rows = fix.prompts, fix.tables, fix.rows
    reference, references, verify_rows = fix.reference, fix.references, fix.verify_rows
    pair_cfg = _SpecGroupConfig(("A", "B"), "p", (0.0, 1.0, 0, 0), 48)
    pair_streams, pair_rates = _run_spec_group(
        batched, pair_cfg, prompts, tables, verify_rows
    )
    batched.step(mk.StepInput([], ("pB",)))
    alone = [
        batched.step(
            mk.StepInput([k3.request("pA", np.zeros(rows), tables["A"])])
        ).tokens[:1]
        for _ in range(20)
    ]
    pa_more = pair_streams[0] + _tokens_of(alone)
    batched.forget(("pA",))
    same_pa = _agree(pair_streams[0], reference)
    same_pb = _agree(pair_streams[1], references["B"])
    same_more = _agree(pa_more, reference)

    four = ("A", "B", "C", "D")
    four_cfg = _SpecGroupConfig(four, "f", (0.0, 1.0, 0, 0), 32)
    four_streams, four_rates = _run_spec_group(
        batched, four_cfg, prompts, tables, verify_rows
    )
    batched.forget(tuple("f" + n for n in four))
    four_agree = [
        _agree(st, references[n]) for n, st in zip(four, four_streams, strict=True)
    ]
    p_cfg_four = _PlainGroupConfig(four, four_streams, prompts, tables, references)
    plain_four, follow_four = _plain_group(plain, p_cfg_four)

    eight = tuple("ABCDEFGH")
    eight_cfg = _SpecGroupConfig(eight, "e", (0.0, 1.0, 0, 0), 32)
    eight_streams, eight_rates = _run_spec_group(
        batched, eight_cfg, prompts, tables, verify_rows
    )
    batched.forget(tuple("e" + n for n in eight))
    eight_agree = [
        _agree(st, references[n]) for n, st in zip(eight, eight_streams, strict=True)
    ]
    p_cfg_eight = _PlainGroupConfig(eight, eight_streams, prompts, tables, references)
    plain_eight, follow_eight = _plain_group(plain, p_cfg_eight)

    five = tuple("ABCDE")
    five_cfg = _SpecGroupConfig(five, "v", (0.0, 1.0, 0, 0), 16)
    five_streams, _ = _run_spec_group(batched, five_cfg, prompts, tables, verify_rows)
    batched.forget(tuple("v" + n for n in five))
    five_agree = [
        _agree(st, references[n]) for n, st in zip(five, five_streams, strict=True)
    ]

    three = ("A", "B", "C")
    three_cfg = _SpecGroupConfig(three, "t", (0.0, 1.0, 0, 0), 16)
    three_streams, _ = _run_spec_group(batched, three_cfg, prompts, tables, verify_rows)
    batched.forget(tuple("t" + n for n in three))
    three_agree = [
        _agree(st, references[n]) for n, st in zip(three, three_streams, strict=True)
    ]

    b_runs = []
    for tag, seed in (
        ("x", _DEFAULT_SEED_A),
        ("y", _DEFAULT_SEED_A),
        ("z", _DEFAULT_SEED_B),
    ):
        sub_cfg = _SpecGroupConfig(("A", "B"), tag, (0.7, 0.9, 50, seed), 12)
        streams, _ = _run_spec_group(batched, sub_cfg, prompts, tables, verify_rows)
        batched.forget((f"{tag}A", f"{tag}B"))
        b_runs.append(streams)
    b_reproducible = b_runs[0] == b_runs[1]
    b_varies = b_runs[0] != b_runs[2]

    pipelined_fix = _PipelinedFixture(eight, prompts, tables)
    timing = {}
    for req_count, rates in ((2, pair_rates), (4, four_rates), (8, eight_rates)):
        k_val = verify_rows(req_count)
        s_ms = _pipelined_step_ms(batched, (req_count, k_val), pipelined_fix)
        p_ms = _pipelined_step_ms(plain, (req_count, 1), pipelined_fix)
        timing[req_count] = (round(s_ms, 3), round(p_ms, 3))

    return _SpecBatchedResults(
        pair_rates=pair_rates,
        four_rates=four_rates,
        eight_rates=eight_rates,
        same_pa=same_pa,
        same_pb=same_pb,
        same_more=same_more,
        four_agree=four_agree,
        plain_four=plain_four,
        follow_four=follow_four,
        eight_agree=eight_agree,
        plain_eight=plain_eight,
        follow_eight=follow_eight,
        five_agree=five_agree,
        three_agree=three_agree,
        b_reproducible=b_reproducible,
        b_varies=b_varies,
        timing=timing,
    )


def _build_spec_prompts(
    tokenizer: _TokenizerLike, count: int, all_ids: Sequence[int]
) -> dict[str, np.ndarray]:
    """Build named prompts for speculation check."""
    ids = np.asarray(all_ids[:count], np.int32)

    def encode(text: str) -> np.ndarray:
        """Encode prompt text as int32 numpy array."""
        return np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)

    return {
        "A": ids,
        "B": encode("The history of the lighthouse begins"),
        "C": np.asarray(all_ids[count : count + 2000], np.int32),
        "D": encode("Write a short story about a robot who learns to paint."),
        "E": encode("Explain, step by step, how a transistor amplifies a signal."),
        "F": np.asarray(all_ids[count + 2000 :], np.int32),
        "G": encode("List the prime numbers below fifty and say why each is prime."),
        "H": encode("Once upon a time, in a small village by the sea,"),
    }


def run_spec(args: argparse.Namespace, records: harness.Records) -> None:
    """Run comprehensive speculation check against plain greedy references."""
    count, generate, rows = args.prompt_tokens, args.generate, args.single_drafts + 1
    large = sorted(int(n) for n in args.large_batches.split(",") if n)
    if args.require_verifier_parity and not large:
        harness.fail("--require-verifier-parity requires --large-batches")
    most = max([8, *large])
    all_ids, raw_tokenizer = k3.fixed_prompt(count + 4000, args.model)
    tokenizer = cast("_TokenizerLike", raw_tokenizer)
    prompts = _build_spec_prompts(tokenizer, count, all_ids)
    ids = prompts["A"]
    longest = max(map(len, prompts.values()))
    pages = -(-(longest + 4 * generate + 64) // k3.logical()) + 1
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(2 * most + 1),
        batch_sizes=decode_batch_sizes(most),
    )
    pools = k3.Pools((10 + max([0, *large])) * pages + 1, pages, 1)
    draft = draft_class(args.dspark)(
        max_position=longest + 8 * generate + 64,
        checkpoint=args.dspark,
        presharded=args.presharded,
    )

    single = k3.decoder(
        ops,
        pools,
        max_num_seqs=most,
        max_model_len=pages * k3.logical(),
        draft=draft,
        schedule=((1, 1, args.single_drafts), (2, 8, 0)),
    )
    plain = k3.decoder(
        ops,
        pools,
        max_num_seqs=most,
        max_model_len=pages * k3.logical(),
        draft=draft,
    )
    batched = k3.decoder(
        ops,
        pools,
        max_num_seqs=most,
        max_model_len=pages * k3.logical(),
        draft=draft,
        schedule=(
            (1, 1, args.single_drafts),
            (2, 4, args.small_batch_drafts),
            (5, 8, 3),
        ),
    )
    harness.sync()

    tables = {
        name: pools.scheduled(
            np.arange(1 + n * pages, 1 + (n + 1) * pages, dtype=np.int32), n
        )
        for n, name in enumerate(("A", "B", "ref", "C", "D", "E", "F", "G", "H"))
    }

    def plain_reference(prompt: Sequence[int] | np.ndarray, n: int) -> list[int]:
        """Compute plain greedy reference tokens."""
        outs = [
            plain.step(mk.StepInput([k3.request("ref", prompt, tables["ref"])])).tokens[
                :1
            ]
        ]
        outs.extend(
            plain.step(mk.StepInput([k3.request("ref", [0], tables["ref"])])).tokens[:1]
            for _ in range(n - 1)
        )
        plain.forget(("ref",))
        return _tokens_of(outs)

    single.warmup()
    plain.warmup()
    reference = plain_reference(ids, generate)
    b_reference = plain_reference(prompts["B"], 24)

    a_tokens, same, per_step, b_tokens, step_ms, reproducible, varies = (
        _spec_single_stream_loop(single, rows, prompts, tables, reference)
    )
    same_b = _agree(b_tokens, b_reference)

    batched.warmup()

    def verify_rows(req_count: int) -> int:
        """Return verification width."""
        return batched.spec_tokens[req_count] + 1

    references = {
        n: reference if n == "A" else plain_reference(prompts[n], generate)
        for n in prompts
    }

    churn_join, churn_steady = _spec_churn(batched, prompts, tables, verify_rows)

    flow_fix = _BatchedFlowFixture(
        prompts, tables, rows, reference, references, verify_rows
    )
    res = _spec_run_batched_flows(batched, plain, flow_fix)

    arr_fix = _ArrivalFixture(prompts, tables, rows, verify_rows)
    arrivals = {
        n: (
            _arriving_arrival(batched, n, "alone", arr_fix),
            _arriving_arrival(batched, n, "joining", arr_fix),
        )
        for n in ("B", "D")
    }

    large_results: dict[str, dict[str, object]] = {}
    records.add(
        case="spec",
        single_drafts=args.single_drafts,
        small_batch_drafts=args.small_batch_drafts,
        dspark_vs_plain=[same, min(len(a_tokens), len(reference))],
        tokens_per_step=per_step,
        batch2_vs_plain=[same_b, len(b_tokens)],
        seed_reproducible=reproducible,
        seed_varies=varies,
        pipelined_ms=step_ms,
        batched_pair_agree=[res.same_pa, res.same_pb],
        batched_pair_tokens_per_step=[round(r, 2) for r in res.pair_rates],
        batched_then_alone=res.same_more,
        batched_four_agree=res.four_agree,
        plain_four_agree=res.plain_four,
        batched_four_follow_plain=res.follow_four,
        batched_four_tokens_per_step=[round(r, 2) for r in res.four_rates],
        batched_three_agree=res.three_agree,
        batched_eight_agree=res.eight_agree,
        plain_eight_agree=res.plain_eight,
        batched_eight_follow_plain=res.follow_eight,
        batched_eight_tokens_per_step=[round(r, 2) for r in res.eight_rates],
        batched_five_agree=res.five_agree,
        batched_seed_reproducible=res.b_reproducible,
        batched_seed_varies=res.b_varies,
        batched_ms={str(b): t for b, t in res.timing.items()},
        churn_worst_ms=[round(churn_join), round(churn_steady)],
        transferred_arrival_first_tokens=arrivals,
        large_batches=large_results,
    )
    ok = reproducible and varies and same > 0
    ok = (
        ok
        and res.b_reproducible
        and res.b_varies
        and res.same_pa > 0
        and res.same_pb > 0
    )
    ok = ok and all(
        a > 0
        for a in res.four_agree + res.three_agree + res.eight_agree + res.five_agree
    )
    ok = ok and all(a[:1] == j[:1] for a, j in arrivals.values())
    if churn_steady > _MAX_SPEC_CHURN_MS or churn_join > _MAX_SPEC_CHURN_MS:
        ok = False
    if not ok:
        harness.fail("spec check")


@dataclasses.dataclass(frozen=True)
class _HandoffContext:
    """Context bundle for handoff draft ring checks."""

    lengths: dict[str, int]
    prompts: dict[str, list[int]]
    first_token: dict[str, int]
    logical: dict[str, list[int]]
    decode_states: dict[str, list[list[int]]]


def _handoff_rings(
    args: argparse.Namespace,
    records: harness.Records,
    decoder: KimiDecoder,
    ctx: _HandoffContext,
    row_fn: Callable[..., mk.RequestInput],
) -> bool:
    """Check that imported draft rings match expected prompt positions."""
    if decoder.draft is None:
        msg = "decoder draft must not be None"
        raise RuntimeError(msg)
    window = decoder.draft.config.window
    steps, ok = args.speculate_steps, True
    for req, n in ctx.lengths.items():
        first = row_fn(
            f"spec-{req}",
            (n, 1),
            ctx.prompts[req] + [ctx.first_token[req]],
            n,
            (ctx.logical[req], ctx.decode_states[req]),
        )
        kda_groups = vars(decoder)["_kda_groups"]
        blocks = kda_blocks_at_length(first.block_tables, kda_groups, n)
        importer = vars(decoder)["_importer"]
        _, positions = importer.import_ring(
            decoder.draft.config, torch.from_numpy(blocks).to("tpu"), n
        )
        held = int((positions >= 0).sum().cpu())
        ok &= held == min(n, window)
        counts = {}
        for arm, ring in (("ring", True), ("cold", False)):
            vars(decoder)["_ring"] = ring
            req_id = f"spec-{arm}-{req}"
            outs = [
                decoder.step(
                    mk.StepInput([dataclasses.replace(first, req_id=req_id)])
                ).tokens[:1]
            ]
            following = mk.RequestInput(
                req_id=req_id,
                computed=n + 1,
                scheduled=args.single_drafts + 1,
                token_ids=np.zeros(n + args.single_drafts + 2, np.int32),
                num_tokens=n + args.single_drafts + 2,
                num_prompt_tokens=n,
                block_tables=first.block_tables,
            )
            outs.extend(
                decoder.step(mk.StepInput([following])).tokens[:1]
                for _ in range(steps - 1)
            )
            decoder.step(mk.StepInput([], (req_id,)))
            emitted = torch.cat(outs).cpu().numpy()
            counts[arm] = (emitted >= 0).sum(axis=1)
        vars(decoder)["_ring"] = True
        windows = [w for w in (16, 64) if w < steps]
        windows.append(steps)
        rates = {
            arm: [round(float(counts[arm][:w].mean()), 2) for w in windows]
            for arm in counts
        }
        records.add(
            case=f"handoff-ring-{req}",
            prompt_tokens=n,
            ring_positions=held,
            tokens_per_step_ring=rates["ring"],
            tokens_per_step_cold=rates["cold"],
        )
    return ok


@dataclasses.dataclass(frozen=True)
class _HandoffTableSetup:
    """Setup inputs for handoff tables."""

    ids: Sequence[int]
    lengths: dict[str, int]
    hit: dict[str, tuple[str, int]]
    max_model_len: int


def _build_handoff_prompts_and_tables(
    cfg: _HandoffTableSetup,
) -> tuple[
    dict[str, list[int]], k3.Pools, dict[str, list[int]], dict[str, list[list[int]]]
]:
    """Build prompts, pools, and block tables for handoff verification."""
    ids, lengths, hit, max_model_len = cfg.ids, cfg.lengths, cfg.hit, cfg.max_model_len
    prompts: dict[str, list[int]] = {}
    at = 0
    for req, n in lengths.items():
        if req in hit:
            donor, cached = hit[req]
            prompts[req] = prompts[donor][:cached] + list(ids[at : at + n - cached])
            at += n - cached
        else:
            prompts[req] = list(ids[at : at + n])
            at += n
    decode_pages = max_model_len // k3.LOGICAL + 1
    pools = k3.Pools(
        1
        + 3 * decode_pages
        + sum(-(-n // k3.LOGICAL) + 3 * -(-n // k3.PAGE) for n in lengths.values())
        + 8,
        decode_pages,
        max_model_len // k3.PAGE,
    )
    logical: dict[str, list[int]] = {}
    states: dict[str, list[list[int]]] = {}
    for req, n in lengths.items():
        blocks = -(-n // k3.LOGICAL)
        if req in hit:
            donor, cached = hit[req]
            logical[req] = logical[donor][: cached // k3.LOGICAL] + pools.fresh(
                blocks - cached // k3.LOGICAL
            )
        else:
            logical[req] = pools.fresh(blocks)
        states[req] = [pools.fresh(-(-n // k3.PAGE)) for _ in range(3)]
    for req, (donor, cached) in hit.items():
        for g in range(3):
            states[req][g][cached // k3.PAGE - 1] = states[donor][g][
                cached // k3.PAGE - 1
            ]
    return prompts, pools, logical, states


@dataclasses.dataclass(frozen=True)
class _HandoffPrefillFixture:
    """Fixture bundle for handoff prefill execution."""

    prompts: dict[str, list[int]]
    lengths: dict[str, int]
    hit: dict[str, tuple[str, int]]
    logical: dict[str, list[int]]
    states: dict[str, list[list[int]]]
    pools: k3.Pools


def _handoff_prefill_execution(
    prefiller: KimiPrefiller,
    fix: _HandoffPrefillFixture,
) -> dict[str, int]:
    """Run prefiller through chunked schedule and record first tokens."""
    prompts, lengths, hit = fix.prompts, fix.lengths, fix.hit
    logical, states, pools = fix.logical, fix.states, fix.pools

    def row(
        req_id: str,
        computed_tokens: tuple[int, int],
        tokens: Sequence[int],
        prompt_len: int,
        tables_tuple: tuple[list[int], list[list[int]]],
    ) -> mk.RequestInput:
        """Build a chunked prefill ``RequestInput`` from bounds and block tables."""
        computed, scheduled = computed_tokens
        mla_table, kda_tables = tables_tuple
        tokens_arr = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=scheduled,
            token_ids=tokens_arr,
            num_tokens=len(tokens_arr),
            num_prompt_tokens=prompt_len,
            block_tables=pools.tables(mla_table, kda_tables),
        )

    done = {req: hit[req][1] if req in hit else 0 for req in lengths}
    first_token: dict[str, int] = {}
    schedule = [
        [("A", 4096), ("C", 700)],
        *[[("A", end)] for end in range(8192, k3.LOGICAL + 1, 4096)],
        [("A", lengths["A"])],
        [("B", lengths["B"]), ("D", k3.PAGE)],
        [("E", 100)],
        [("F", lengths["F"])],
    ]
    for step in schedule:
        rows_in = [
            row(
                req,
                (done[req], total - done[req]),
                prompts[req],
                len(prompts[req]),
                (logical[req], states[req]),
            )
            for req, total in step
        ]
        step_tokens = (
            prefiller.step(mk.StepInput(rows_in))
            .tokens[: len(rows_in), 0]
            .cpu()
            .tolist()
        )
        for (req, total), tok in zip(step, step_tokens, strict=True):
            done[req] = total
            if total == len(prompts[req]):
                first_token[req] = int(tok)
    bridge.materialize(pools.tensors[0])
    return first_token


def _handoff_verify_continue(
    decoder: KimiDecoder,
    generate: int,
    fixtures: tuple[
        dict[str, list[int]],
        dict[str, int],
        dict[str, list[int]],
        dict[str, list[list[int]]],
        dict[str, list[int]],
        k3.Pools,
    ],
    lengths: dict[str, int],
    records: harness.Records,
) -> bool:
    """Run continued decoding from prefilled state and verify against reference."""
    prompts, first_token, logical, decode_states, reference_tables, pools = fixtures

    def make_req(
        req_id: str,
        computed_and_sched: tuple[int, int],
        toks: Sequence[int],
        p_len: int,
        tab: tuple[list[int], list[list[int]]],
    ) -> mk.RequestInput:
        """Build a handoff verification ``RequestInput`` from bounds and tables."""
        computed, scheduled = computed_and_sched
        t_arr = np.asarray(toks, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=scheduled,
            token_ids=t_arr,
            num_tokens=len(t_arr),
            num_prompt_tokens=p_len,
            block_tables=pools.tables(tab[0], tab[1]),
        )

    def continue_from(req_id: str, first_row: mk.RequestInput) -> list[int]:
        """Decode subsequent tokens until completion."""
        outs = [decoder.step(mk.StepInput([first_row])).tokens[:1]]
        computed = first_row.end
        for _ in range(generate - 1):
            following = mk.RequestInput(
                req_id=req_id,
                computed=computed,
                scheduled=1,
                token_ids=np.zeros(computed + 1, np.int32),
                num_tokens=computed + 1,
                num_prompt_tokens=first_row.num_prompt_tokens,
                block_tables=first_row.block_tables,
            )
            outs.append(decoder.step(mk.StepInput([following])).tokens[:1])
            computed += 1
        decoder.step(mk.StepInput([], (req_id,)))
        return [int(t) for t in torch.cat(outs).cpu().reshape(-1).tolist() if t >= 0]

    all_first_match = True
    for req, n in lengths.items():
        pd_tokens = [first_token[req]]
        pd_tokens.extend(
            continue_from(
                f"pd-{req}",
                make_req(
                    f"pd-{req}",
                    (n, 1),
                    prompts[req] + [first_token[req]],
                    n,
                    (logical[req], decode_states[req]),
                ),
            )[: generate - 1]
        )
        ref = continue_from(
            f"ref-{req}",
            make_req(
                f"ref-{req}",
                (0, n),
                prompts[req],
                n,
                (reference_tables[req], [[], [], []]),
            ),
        )
        same = _agree(pd_tokens, ref)
        all_first_match &= pd_tokens[0] == ref[0]
        records.add(
            case=f"handoff-{req}",
            prompt_tokens=n,
            first_token_matches=pd_tokens[0] == ref[0],
            agree=[same, len(ref)],
        )
    return all_first_match


def run_handoff(args: argparse.Namespace, records: harness.Records) -> None:
    """Run prefill-to-decode handoff numerics verification."""
    generate, max_model_len = args.generate, 32768
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    all_ids, _ = k3.fixed_prompt(24000, args.model)
    lengths = {
        "A": k3.LOGICAL + 3808,
        "B": k3.LOGICAL + 1500,
        "C": 700,
        "D": k3.PAGE,
        "E": 100,
        "F": k3.LOGICAL + 400,
    }
    hit = {"B": ("A", k3.LOGICAL), "F": ("A", k3.LOGICAL)}
    prompts, pools, logical, states = _build_handoff_prompts_and_tables(
        _HandoffTableSetup(all_ids, lengths, hit, max_model_len)
    )
    decode_pages = max_model_len // k3.LOGICAL + 1
    decode_states = {
        req: [s + pools.fresh(-(-(n + 1) // k3.PAGE) - len(s)) for s in states[req]]
        for req, n in lengths.items()
    }
    reference_tables = {req: pools.fresh(decode_pages) for req in lengths}

    draft = Path(args.dspark) if args.dspark else None
    prefiller: KimiPrefiller | None = cast(
        "KimiPrefiller",
        k3_program.build(
            k3.engine_config(
                "prefill",
                max_num_seqs=4,
                max_model_len=max_model_len,
                max_num_batched_tokens=4096,
            ),
            args.presharded,
            draft,
            tp_rank=rank,
        ),
    )
    if prefiller is None:
        msg = "prefiller build returned None"
        raise RuntimeError(msg)
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()

    first_token = _handoff_prefill_execution(
        prefiller, _HandoffPrefillFixture(prompts, lengths, hit, logical, states, pools)
    )

    prefiller.program.weights.clear()
    vars(prefiller.program)["_expert_banks"].clear()
    if prefiller.ring is not None:
        prefiller.ring.weights = ()
    prefiller = None
    gc.collect()
    harness.sync()

    decoder: KimiDecoder = cast(
        "KimiDecoder",
        k3_program.build(
            k3.engine_config(
                "decode",
                max_num_seqs=2,
                max_model_len=max_model_len,
                max_num_batched_tokens=512,
                num_speculative_tokens=args.single_drafts if draft else 0,
                num_speculative_tokens_per_batch_size=(
                    (1, 1, args.single_drafts),
                    (2, 2, 0),
                )
                if draft
                else None,
            ),
            args.presharded,
            draft,
            tp_rank=rank,
        ),
    )
    decoder.load_weights()
    decoder.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()

    fixtures = (prompts, first_token, logical, decode_states, reference_tables, pools)
    all_first_match = _handoff_verify_continue(
        decoder, generate, fixtures, lengths, records
    )

    def make_req_fixture(
        req_id: str,
        bounds: tuple[int, int],
        toks: Sequence[int],
        p_len: int,
        tables_tuple: tuple[list[int], list[list[int]]],
    ) -> mk.RequestInput:
        """Build a handoff draft-ring ``RequestInput`` from bounds and tables."""
        computed, sched = bounds
        log, st = tables_tuple
        t_arr = np.asarray(toks, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=sched,
            token_ids=t_arr,
            num_tokens=len(t_arr),
            num_prompt_tokens=p_len,
            block_tables=pools.tables(log, st),
        )

    rings_ok = True
    if draft is not None:
        ctx = _HandoffContext(
            lengths=lengths,
            prompts=prompts,
            first_token=first_token,
            logical=logical,
            decode_states=decode_states,
        )
        rings_ok = _handoff_rings(args, records, decoder, ctx, make_req_fixture)
    if not all_first_match:
        harness.fail("a prefill first token differs from the decode-only reference")
    if not rings_ok:
        harness.fail("an imported draft ring does not hold the prompt's last positions")


def _packing_cases(page: int, window: int, slots: int) -> list[tuple[int, int]]:
    """Determine packing lengths and prefix hit tokens for prompts."""
    pages = list(_PACKING_PAGES)
    if len(pages) != slots or sum(pages) != window // page:
        pages = [window // page // slots] * slots
    cases = []
    for i, n in enumerate(pages):
        trim = 0 if i % _MAGIC_4 == 1 else (37 * i + 11) % page
        length = 1 if i == _MAGIC_4 else n * page - trim
        hit = i - (slots - len(_PACKING_PREFIX_PAGES))
        cases.append((length, _PACKING_PREFIX_PAGES[hit] * page if hit >= 0 else 0))
    return cases


def _device_tpu(values: object, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Convert numpy sequence to TPU device tensor."""
    return torch.from_numpy(np.asarray(values)).to(dtype).to("tpu")


def _squares(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Compute squared error sums."""
    a, b = a.to(torch.float32), b.to(torch.float32)
    return [float(((a - b) ** 2).sum().cpu()), float((a**2).sum().cpu())]


@dataclasses.dataclass(frozen=True)
class _PackingArmsFixture:
    """Fixture bundle for packing arm execution."""

    cases: list[tuple[int, int]]
    prompts: list[list[int]]
    donors: dict[int, tuple[list[int], list[list[int]]]]
    hits: list[int]
    pools: k3.Pools
    window: int
    slots: int


def _packing_execute_arms(
    prefiller: KimiPrefiller,
    fix: _PackingArmsFixture,
) -> dict[
    str, tuple[list[int], list[tuple[list[int], list[list[int]]]], list[np.ndarray]]
]:
    """Execute packing arms: alone, packed4, and packed."""
    cases, prompts, donors = fix.cases, fix.prompts, fix.donors
    hits, pools, window, slots = fix.hits, fix.pools, fix.window, fix.slots

    def tables(
        end: int, donor: object = None, cached: int = 0
    ) -> tuple[list[int], list[list[int]]]:
        """Allocate and initialize block tables."""
        state_width = (2 * window) // k3.PAGE
        mla = pools.fresh(-(-end // k3.LOGICAL))
        kda = [[0] * state_width for _ in range(3)]
        for g in range(3):
            kda[g][(end - 1) // k3.PAGE] = pools.fresh(1)[0]
            if donor is not None:
                column = cached // k3.PAGE - 1
                donor_kda = cast("tuple[list[int], list[list[int]]]", donor)[1]
                kda[g][column] = donor_kda[g][column]
        return mla, kda

    def row(
        req_id: str,
        tokens: Sequence[int],
        computed: int,
        table: tuple[list[int], list[list[int]]],
    ) -> mk.RequestInput:
        """Construct RequestInput for packing step."""
        tokens_arr = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=len(tokens_arr) - computed,
            token_ids=tokens_arr,
            num_tokens=len(tokens_arr),
            num_prompt_tokens=len(tokens_arr),
            block_tables=pools.tables(*table),
        )

    def step(
        rows_in: list[mk.RequestInput], copies: Sequence[tuple[int, int]] = ()
    ) -> torch.Tensor:
        """Submit a packed prefill step with block copies and return tokens."""
        return prefiller.step(mk.StepInput(rows_in, copies=tuple(copies))).tokens

    runs = {}
    for arm, limit in (("alone", 1), ("packed4", 4), ("packed", slots)):
        blocks = [
            tables(prefix + length, donors.get(i), prefix)
            for i, (length, prefix) in enumerate(cases)
        ]
        rows_list = [
            row(f"{arm}-{i}", prompts[i], prefix, blocks[i])
            for i, (_, prefix) in enumerate(cases)
        ]
        copies = {i: (donors[i][0][0], blocks[i][0][0]) for i in hits}
        if arm == "alone":
            first = []
            for i, one in enumerate(rows_list):
                tokens = step([one], [copies[i]] if i in copies else ())
                first.append(int(tokens[0, 0].cpu()))
        else:
            pieces = [
                k3_prefill.plan_pieces(i, prefix, prefix + n, [], window, k3.PAGE)[0]
                for i, (n, prefix) in enumerate(cases)
            ]
            plan = k3_prefill.pack_windows(
                pieces,
                prefiller.program.windows,
                limit,
                prefiller.program.capacity,
            )
            if arm == "packed" and (len(plan) != 1 or len(plan[0][1]) != len(cases)):
                harness.fail(f"the {len(cases)} prompts did not pack into one window")
            prefiller.slots = limit
            tokens = step(rows_list, list(copies.values()))
            prefiller.slots = slots
            first = [int(t) for t in tokens[: len(rows_list), 0].cpu().tolist()]
        kda_groups = vars(prefiller)["_kda_groups"]
        finals = [
            kda_blocks_at_length(r.block_tables, kda_groups, r.end) for r in rows_list
        ]
        runs[arm] = (first, blocks, finals)
    bridge.materialize(pools.tensors[0])
    return runs


@dataclasses.dataclass(frozen=True)
class _PackingEvalFixture:
    """Fixture bundle for evaluating packed L2 deviations."""

    runs: dict[
        str,
        tuple[
            list[int],
            list[tuple[list[int], list[list[int]]]],
            list[np.ndarray],
        ],
    ]
    cases: list[tuple[int, int]]
    tokenizer: _TokenizerLike
    pools: k3.Pools
    prefiller: KimiPrefiller
    records: harness.Records


def _compute_packing_sums(
    runs: dict[
        str,
        tuple[
            list[int],
            list[tuple[list[int], list[list[int]]]],
            list[np.ndarray],
        ],
    ],
    cases: list[tuple[int, int]],
    pools: k3.Pools,
    prefiller: KimiPrefiller,
) -> list[object]:
    """Compute and gather squared sums across ranks."""

    def owned_pages(
        table: tuple[list[int], list[list[int]]], prefix: int, length: int
    ) -> list[int]:
        """Return logical pages owned by this rank."""
        cp = k3.LOGICAL // k3.PAGE
        pages = range(prefix // k3.PAGE, (prefix + length) // k3.PAGE)
        return [table[0][k // cp] for k in pages if k % cp == prefiller.cp_rank]

    writer = vars(prefiller)["_writer"]
    states = {
        arm: writer.read_states(_device_tpu(np.concatenate(runs[arm][2])))
        for arm in runs
    }
    rings = {}
    if prefiller.ring is not None:
        for arm in runs:
            rings[arm] = [
                writer.read_rings(
                    prefiller.ring.config,
                    _device_tpu(runs[arm][2][i]),
                    [prefix + length],
                )[0][0]
                for i, (length, prefix) in enumerate(cases)
            ]

    def compare(arm: str, i: int, length: int, prefix: int) -> dict[str, object]:
        """Compare squared sums against alone execution."""
        entry: dict[str, object] = {}
        pages = [owned_pages(runs[a][1][i], prefix, length) for a in ("alone", arm)]
        if pages[0]:
            index = [_device_tpu(p, torch.int64) for p in pages]
            per_layer = [
                _squares(pool[index[0]], pool[index[1]]) for pool in pools.tensors
            ]
            entry["latents"] = np.sum(per_layer, axis=0).tolist()
            entry["latents0"] = per_layer[0]
        for name, part in (("recurrent", 0), ("conv", 1)):
            entry[name] = _squares(
                states["alone"][part][0, :, i], states[arm][part][0, :, i]
            )
        if rings:
            entry["ring"] = _squares(rings["alone"][i], rings[arm][i])
        return entry

    arms = ("packed4", "packed")
    sums = {
        arm: [compare(arm, i, n, prefix) for i, (n, prefix) in enumerate(cases)]
        for arm in arms
    }
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, sums)
    return gathered


def _packing_evaluate_l2(fix: _PackingEvalFixture) -> None:
    """Evaluate relative L2 deviations across packed arms against alone."""
    runs, cases = fix.runs, fix.cases
    tokenizer, pools = fix.tokenizer, fix.pools
    prefiller, records = fix.prefiller, fix.records

    gathered = _compute_packing_sums(runs, cases, pools, prefiller)
    names = ("latents0", "latents", "recurrent", "conv", "ring")

    def relative(arm: str, i: int, name: str) -> float | None:
        """Compute relative L2 metric across gathered ranks."""
        num = sum(
            cast("dict[str, list[dict[str, tuple[float, float]]]]", g)[arm][i].get(
                name, (0.0, 0.0)
            )[0]
            for g in gathered
            if g is not None
        )
        den = sum(
            cast("dict[str, list[dict[str, tuple[float, float]]]]", g)[arm][i].get(
                name, (0.0, 0.0)
            )[1]
            for g in gathered
            if g is not None
        )
        return math.sqrt(num / den) if den else None

    arms = ("packed4", "packed")
    alone = runs["alone"][0]
    worst = {arm: dict.fromkeys(names, 0.0) for arm in arms}
    matches = {
        arm: sum(a == b for a, b in zip(alone, runs[arm][0], strict=True))
        for arm in arms
    }
    for i, (length, prefix) in enumerate(cases):
        line = []
        for arm in arms:
            errors = {n: relative(arm, i, n) for n in names}
            filtered_errors = {n: v for n, v in errors.items() if v is not None}
            for name, value in filtered_errors.items():
                worst[arm][name] = max(worst[arm][name], value)
            token = runs[arm][0][i]
            line.append(
                f"{arm} {'same' if token == alone[i] else 'DIFFERS'} "
                f"{tokenizer.decode([token])!r} "
                + " ".join(f"{k} {v:.1e}" for k, v in filtered_errors.items())
            )
            records.add(
                case=f"packing-{arm}-{i}",
                length=length,
                prefix=prefix,
                first_token_matches=token == alone[i],
                **{f"rel_l2_{k}": round(v, 6) for k, v in filtered_errors.items()},
            )
    for arm in arms:
        harness.log(f"{arm}: {matches[arm]} of {len(cases)} first tokens match alone")
    if matches["packed"] < matches["packed4"] - 1 or any(
        worst["packed"][n] > 2 * worst["packed4"][n] + 0.02 for n in names
    ):
        harness.fail("one-window packing departs further than the reference packing")


def run_packing(args: argparse.Namespace, records: harness.Records) -> None:
    """Run chunked packing numerics check against alone baselines."""
    window, slots = k3_program.MAX_STEP_TOKENS, k3_prefill.SLOTS
    max_model_len = 2 * window
    cases = _packing_cases(k3.PAGE, window, slots)
    hits = [i for i, (_, prefix) in enumerate(cases) if prefix]
    ids, raw_tokenizer = k3.fixed_prompt(sum(n + p for n, p in cases), args.model)
    tokenizer = cast("_TokenizerLike", raw_tokenizer)
    prompts: list[list[int]] = []
    at = 0
    for length, prefix in cases:
        prompts.append(ids[at : at + prefix + length])
        at += prefix + length
    mla_width, state_width = max_model_len // k3.LOGICAL, max_model_len // k3.PAGE
    per_request = -(-max_model_len // k3.LOGICAL) + 3
    pools = k3.Pools(
        1 + (len(hits) + 3 * len(cases)) * per_request, mla_width, state_width
    )
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    draft = Path(args.dspark) if args.dspark else None
    prefiller: KimiPrefiller = cast(
        "KimiPrefiller",
        k3_program.build(
            k3.engine_config(
                "prefill",
                max_num_seqs=slots,
                max_model_len=max_model_len,
                max_num_batched_tokens=window,
            ),
            args.presharded,
            draft,
            tp_rank=rank,
        ),
    )
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    prefiller.warmup()
    harness.sync()

    def make_table(end: int) -> tuple[list[int], list[list[int]]]:
        """Allocate fresh MLA and KDA block tables covering ``end`` tokens."""
        mla = pools.fresh(-(-end // k3.LOGICAL))
        kda = [[0] * state_width for _ in range(3)]
        for g in range(3):
            kda[g][(end - 1) // k3.PAGE] = pools.fresh(1)[0]
        return mla, kda

    donors = {i: make_table(cases[i][1]) for i in hits}
    donor_rows = [
        mk.RequestInput(
            req_id=f"donor-{i}",
            computed=0,
            scheduled=cases[i][1],
            token_ids=np.asarray(prompts[i][: cases[i][1]], np.int32),
            num_tokens=cases[i][1],
            num_prompt_tokens=cases[i][1],
            block_tables=pools.tables(donors[i][0], donors[i][1]),
        )
        for i in hits
    ]
    prefiller.step(mk.StepInput(donor_rows))

    p_fix = _PackingArmsFixture(cases, prompts, donors, hits, pools, window, slots)
    runs = _packing_execute_arms(prefiller, p_fix)

    _packing_evaluate_l2(
        _PackingEvalFixture(runs, cases, tokenizer, pools, prefiller, records)
    )
