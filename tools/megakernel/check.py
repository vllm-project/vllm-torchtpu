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
"""Numerics checks of the K3 programs (``check spec`` / ``check handoff`` /
``check packing`` / ``check greedy`` / ``check resume`` / ``check masked`` /
``check dsa-keys`` / ``check dsa-encode`` / ``check dsa-scores`` /
``check arena-order`` / ``check dsa-select``).

**spec** drives the decoder (``KimiDecoder``) as the runner does, over stand-in
unified pools, and checks greedy DSpark against plain greedy decode (the streams
agree until a near-tie), requests joining and leaving across the single and
batched speculation widths (each stream against its plain reference), seeded
sampling (one seed reproduces, another differs) and that no step after warmup
compiles; it reports tokens per step and pipelined step times.
``--require-verifier-parity`` requires complete output equality against
one-token steps of the same verifier shape for every ``--large-batches`` case.

**handoff** runs the prefill program over chunked prompts and prefix-cache hits
into stand-in pools, binds the decode program to the same pools and decodes
each request from the prefill's first token, against a decode-only replay: the
first tokens must match (later ones may part at a near-tie). With ``--dspark``
each imported draft ring must hold the prompt's last positions, and decoding
from it is compared with a cold ring in tokens per step.

**packing** prefills sixteen prompts of mixed lengths, four continuing cached
prefixes, into separate pool blocks three times: each alone, packed at most
four to a window, and all in one window. Each packing's first tokens, latent
pages and final KDA states (with ``--dspark``, draft rings) are compared with
the prompts alone (relative L2). A packing places rows on other ranks, so the
reductions run in another order and the deep stack amplifies that: the
one-window packing must stay as close as the four-to-a-window reference.

**greedy** replays the fixed prompt and four short prompts on the decoder, then
writes each request's greedy tokens and text to ``--output`` (JSON): two kernel
configs' files compare a layout change (e.g. attention DP) on real text.
``--batch`` sets the decoder's batch size (B8 by default); above B8 windows of
the fixed prompt fill the batch.

**resume** decodes a request, preempts it (the decoder is told, as the runner
tells it of a preempted request) and resumes it as vLLM does -- a prefix hit at
a KDA block boundary, the rest recomputed in 512-token chunks -- twice: from a
hit inside the prompt, then from one past it that only the first recompute
wrote. The outputs must equal an uninterrupted decode's; a control resumed
without the preemption signal must not.

**masked** drives the decoder with grammar bitmasks (structured outputs): an
all-ones mask must reproduce the unmasked tokens exactly, random masks must
never yield a disallowed token, and a mask allowing a single token must force
it, greedy or sampled, wherever in the vocabulary the token lives.

**dsa-keys** (Kimi-K3.1: ``--dsa-config`` with ``--synthetic-indexer
rope-probe``, BF16 pages) decodes the fixed prompt, then recomputes every
committed index key from its latent row -- under the rope-probe indexer the key
is the LayerNorm of the row's rope lanes and their negation -- and compares the
FP8 bytes and scale exponents; each rank checks the positions it owns.
``--batch`` as in **greedy**.

**dsa-encode** runs the decode kernel's FP8 encodings (the index key's and,
with the query floor, the index query's) as one collective-free Pallas call on
every rank's own chip over every finite BF16 value and every BF16 amax, and
compares them with XLA's on the host CPU.

**dsa-scores** (Kimi-K3.1, ``--synthetic-indexer random``) decodes the fixed
prompt with the decoder's check outputs on (``--batch`` as in **dsa-keys**),
then checks the last step's indexer against the JAX reference, the parts named
by ``--parts``: ``inputs`` requires each MLA layer's input and q-down latent to
be equal on every rank, and each row's committed index key to be the reference
key of that input within one FP8 step (the in-kernel LayerNorm may round
differently, so a few keys may differ by a step). Each rank checks its own.
``queries`` and ``scores`` check the scorer's query values and head weights,
and each row's scores over its own pages (``run_dsa_scores``).

**arena-order** (no weights) expands stand-in pages of the ``--kv-cache-dtype``
pool layout into a prefill arena (``expand_prefix_pages``) on every rank's own
chip, each token's rope lanes holding its page slot and position, and checks
that arena row ``c`` is token ``c % P`` of slot ``order[1 + c // P]``: the
order Kimi-K3.1's prefill selection names arena columns by.

**dsa-select** (Kimi-K3.1, ``--synthetic-indexer random``) decodes
``--long-rows`` prompts of distinct lengths above ``INDEX_TOPK`` (cut from the
fixed prompt) beside short windows that fill ``--batch``, with the decoder's
check outputs on, then checks the last step's select, the parts named by
``--parts``: ``select`` requires each long row's masks over the DCP group plus
its new-token flag to be ``dsa_reference.decode_selection`` of the kernel's
own scores exactly, ``INDEX_TOPK`` tokens, and each short row's masks empty
with its flag set; ``attend`` requires each sampled row's merged attention
output to be ``dsa_reference.latent_attention`` of its absorbed queries over
the tokens the masks select (a short row: every token) and, when flagged, the
new token (``run_dsa_select``). ``--tie-keys`` repeats a few index keys over
the cache after the first step, so that rows tie at their top-k.
"""

from __future__ import annotations

import dataclasses
import gc
import time

import numpy as np

from tools.megakernel import harness, k3


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="check", required=True)
    spec = sub.add_parser(
        "spec", help="DSpark against plain decode, batch transitions, seeded sampling"
    )
    spec.add_argument("--prompt-tokens", type=int, default=4000)
    spec.add_argument("--generate", type=int, default=200)
    spec.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument("--small-batch-drafts", type=int, choices=(3, 7), default=7)
    spec.add_argument(
        "--large-batches",
        default="",
        help="also speculate with three drafts at these request counts (the "
        "fixture prompts reused), e.g. 16,32 (32: attention DP2 only), against "
        "plain steps of the same requests",
    )
    spec.add_argument(
        "--require-verifier-parity",
        action="store_true",
        help="require every generated token in --large-batches to match "
        "one-token steps of the same verifier shape",
    )
    spec.add_argument(
        "--dspark", required=True, help="the DSpark speculator checkpoint"
    )
    handoff = sub.add_parser(
        "handoff", help="prefill -> pools -> decode against a decode-only replay"
    )
    handoff.add_argument("--generate", type=int, default=32)
    handoff.add_argument("--single-drafts", type=int, choices=(3, 7), default=7)
    handoff.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: hand the draft ring over too",
    )
    handoff.add_argument("--speculate-steps", type=int, default=64)
    packing = sub.add_parser(
        "packing", help="prompts packed into one prefill window against each alone"
    )
    packing.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: compare the prefill's rings too",
    )
    greedy = sub.add_parser(
        "greedy", help="real prompts replayed on the decoder, then greedy tokens"
    )
    greedy.add_argument("--prompt-tokens", type=int, default=5000)
    greedy.add_argument("--generate", type=int, default=128)
    greedy.add_argument("--output", required=True)
    greedy.add_argument(
        "--batch",
        type=int,
        default=8,
        help="the decoder's batch size: the five prompts (the first --batch of "
        "them below B5); a batch above B8 is filled with windows of the fixed "
        "prompt, a request per row",
    )
    greedy.add_argument(
        "--margins",
        action="store_true",
        help="also record each step's top-2 logit gap per request (near-ties)",
    )
    resume = sub.add_parser(
        "resume", help="preempted and resumed decodes against an uninterrupted one"
    )
    resume.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="the prompt (default: one DCP block and three pages, plus 37)",
    )
    masked = sub.add_parser(
        "masked", help="grammar-masked (structured outputs) vs unmasked decode"
    )
    masked.add_argument("--prompt-tokens", type=int, default=4000)
    masked.add_argument("--generate", type=int, default=48)
    masked.add_argument(
        "--dspark",
        default=None,
        help="the DSpark speculator checkpoint: run every case on a speculating "
        "decoder (masked steps fall back to plain ones) and check speculation "
        "around a run of masked steps",
    )
    keys = sub.add_parser(
        "dsa-keys", help="K3.1 index keys (rope-probe) against their latent rows"
    )
    keys.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="the prompt (default: one DCP block and 16 tokens, so that every "
        "rank owns a position)",
    )
    keys.add_argument("--generate", type=int, default=16)
    keys.add_argument(
        "--batch",
        type=int,
        default=8,
        help="the decoder's batch size; a batch above B8 is filled with windows "
        "of the fixed prompt, a request per row",
    )
    scores = sub.add_parser(
        "dsa-scores", help="K3.1's indexer inputs, queries and scores against JAX"
    )
    scores.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="the first request's prompt (default: one DCP block and 16 tokens, so "
        "that every rank owns a position)",
    )
    scores.add_argument("--generate", type=int, default=4)
    scores.add_argument(
        "--batch",
        type=int,
        default=1,
        help="the decoder's batch size; a batch above B1 is filled with windows "
        "of the fixed prompt, a request per row",
    )
    scores.add_argument(
        "--parts",
        default=",".join(DSA_SCORE_PARTS),
        help="what to check, comma-separated: " + ", ".join(DSA_SCORE_PARTS),
    )
    select = sub.add_parser(
        "dsa-select", help="K3.1's top-k select across the DCP group against JAX"
    )
    select.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="the first long row's prompt, the n-th 977 n tokens shorter (default: "
        "one DCP block and 16 tokens)",
    )
    select.add_argument("--generate", type=int, default=4)
    select.add_argument("--batch", type=int, default=8, help="the decoder's batch size")
    select.add_argument(
        "--token-file",
        default="",
        help="a .npy of int32 token ids: the prompts' text instead of the fixed "
        "prompt (longer real contexts)",
    )
    select.add_argument(
        "--long-rows",
        type=int,
        default=2,
        help="rows with a prompt from the fixed text; windows of it fill the batch",
    )
    select.add_argument(
        "--parts",
        default=",".join(DSA_SELECT_PARTS),
        help="what to check, comma-separated: " + ", ".join(DSA_SELECT_PARTS),
    )
    select.add_argument(
        "--tie-keys",
        type=int,
        default=0,
        help="after the first step, give every index-cache token slot one of this "
        "many keys (the same on every rank), so the long rows tie at their top-k "
        "(default 0: the decoded keys)",
    )
    sub.add_parser(
        "dsa-encode", help="K3.1's in-kernel FP8 encodings under Mosaic against XLA"
    )
    sub.add_parser(
        "arena-order", help="the prefill arena's token order within each page"
    )
    for check in (spec, handoff, packing, greedy, resume, masked, keys, scores, select):
        check.add_argument(
            "--model",
            required=True,
            help="the Kimi-K3 checkpoint (the tokenizer and the fixed prompt's text)",
        )


def run(args, records: harness.Records) -> None:
    {
        "spec": run_spec,
        "handoff": run_handoff,
        "packing": run_packing,
        "greedy": run_greedy,
        "resume": run_resume,
        "masked": run_masked,
        "dsa-keys": run_dsa_keys,
        "dsa-encode": run_dsa_encode,
        "dsa-scores": run_dsa_scores,
        "arena-order": run_arena_order,
        "dsa-select": run_dsa_select,
    }[args.check](args, records)


def batch_prompts(prompts: list, ids, batch: int, fill_above: int = 8) -> list:
    """A check's ``prompts`` for a decoder of ``batch`` rows: cut to the
    batch and, above ``fill_above`` rows, filled to it with distinct windows of
    the token list ``ids`` (window ``n`` from token ``61 n``, ``32 + n`` tokens
    long) so every row decodes a request of its own."""
    if batch <= fill_above:
        return prompts[:batch]
    extra = batch - len(prompts)
    if len(ids) < 62 * extra + 32:
        harness.fail(f"B{batch} needs a prompt of {62 * extra + 32} tokens")
    return prompts + [
        np.asarray(ids[61 * n : 62 * n + 32], np.int32) for n in range(1, extra + 1)
    ]


def run_greedy(args, records: harness.Records) -> None:
    import json

    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps

    all_ids, tokenizer = k3.fixed_prompt(args.prompt_tokens, args.model)
    prompts = [np.asarray(all_ids, np.int32)] + [
        np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)
        for text in (
            "Write a short story about a robot who learns to paint.",
            "Explain, step by step, how a transistor amplifies a signal.",
            "List the prime numbers below fifty and say why each is prime.",
            "Once upon a time, in a small village by the sea,",
        )
    ]
    prompts = batch_prompts(prompts, all_ids, args.batch)
    pages = -(-(args.prompt_tokens + args.generate + 64) // k3.logical()) + 1
    # One batch size for every config (under attention DP the five prompts
    # need B8): the replay's single rows and the batch run the same kernel.
    ops = DecodeOps(
        args.presharded,
        num_slots=2 * args.batch + 1,
        batch_sizes=(args.batch,),
        logit_margins=args.margins,
        **k3.decode_ops_options(),
    )
    pools = k3.Pools(len(prompts) * pages + 1, pages, 1)
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=args.batch,
        max_model_len=pages * k3.logical(),
    )
    tops = []  # with --margins: each step's [rows, 2] (top id, top-2 gap)
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]

    def step(requests):
        tokens = decoder.step(mk.StepInput(requests)).tokens
        if args.margins:
            tops.append(ops.logit_margins_out.cpu().numpy())
        return tokens

    started = time.perf_counter()
    outs = [
        step([k3.request(f"r{n}", ids, tables[n]) for n, ids in enumerate(prompts)])
    ]
    harness.log(f"prompts replayed in {time.perf_counter() - started:.1f} s")
    for _ in range(args.generate - 1):
        outs.append(
            step([k3.request(f"r{n}", [0], tables[n]) for n in range(len(prompts))])
        )
    streams = torch.stack([o[: len(prompts), 0] for o in outs]).cpu().numpy().T
    result = [
        {
            "prompt_tokens": len(ids),
            "tokens": stream.tolist(),
            "text": tokenizer.decode(stream.tolist()),
        }
        for ids, stream in zip(prompts, streams, strict=True)
    ]
    if args.margins:
        rows = request_rows(len(prompts), tops[0].shape[0], ops.attention_dp)
        for n, item in enumerate(result):
            top = np.stack([t[rows[n]] for t in tops])
            item["margins"] = [round(float(m), 4) for m in top[:, 1]]
            item["margin_id_mismatches"] = int(
                (top[:, 0].astype(np.int64) != np.asarray(item["tokens"])).sum()
            )
    if harness.rank() == 0:
        with open(args.output, "w") as stream:
            json.dump(result, stream, indent=1)
    for n, item in enumerate(result):
        harness.log(
            f"r{n} ({item['prompt_tokens']} prompt tokens): {item['text'][:160]!r}"
        )
    records.add(
        check="greedy",
        batch=args.batch,
        requests=len(prompts),
        generate=args.generate,
    )


def request_rows(requests: int, batch: int, attention_dp: int) -> list[int]:
    """The plain step's batch row of each request the checks admit in order:
    request ``n`` itself, or under attention DP group ``n % dp``'s row ``n //
    dp`` (``Pools.scheduled``; the decoder fills a group's rows in order,
    ``decode_ops.dp_group_rows``)."""
    if attention_dp == 1:
        return list(range(requests))
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import dp_group_rows

    order, per = dp_group_rows(batch, attention_dp), batch // attention_dp
    return [
        int(order[(n % attention_dp) * per + n // attention_dp])
        for n in range(requests)
    ]


def key_steps(got, got_exponent, want, want_exponent):
    """``(same, beyond, difference)`` per encoded index key (numpy ``got`` and
    ``want`` uint8 ``[..., 128]``, their exponent bytes ``[...]``): ``same``
    where the bytes and the exponent are equal, ``difference`` the largest
    dequantized difference, and ``beyond`` where it exceeds 32 units of the
    coarser scale (one FP8 step in its top binade) or is not finite. JAX runs
    on the default device."""
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    got, want = np.asarray(got), np.asarray(want)
    got_exponent, want_exponent = np.asarray(got_exponent), np.asarray(want_exponent)
    difference = np.abs(
        np.asarray(
            dsa_cache.dequantize_index_keys(jnp.asarray(got), jnp.asarray(got_exponent))
            - dsa_cache.dequantize_index_keys(
                jnp.asarray(want), jnp.asarray(want_exponent)
            )
        )
    ).max(-1)
    same = (want == got).all(-1) & (want_exponent == got_exponent)
    # In float64: finite for every exponent byte, so infinite keys exceed it.
    step = 32 * np.exp2(
        np.maximum(got_exponent, want_exponent).astype(np.float64) - 127
    )
    return same, ~(difference <= step), difference


def index_key_mismatches(latent, index, blocks, slots, page, eps):
    """``(differing, beyond_a_step, worst)`` over the keys at ``(blocks, slots)``
    of one layer under the rope-probe indexer: ``latent`` is the layer's BF16
    pool, ``index`` the layer's pages of the index-key cache over all blocks
    (``[blocks, tokens + 32, 128]``, numpy). A key differs if its FP8 bytes or
    scale exponent are not those of the LayerNorm of ``[rope, -rope]`` of its
    latent row's rope lanes; ``beyond_a_step`` counts those off by more than 32
    units of the coarser scale (one FP8 step in its top binade) or not finite
    (a NaN byte or an infinite scale), ``worst`` is the largest dequantized
    difference (NaN if any is). The reference runs on the host CPU: a worker's
    TPU devices belong to the torch_tpu runtime."""
    import jax
    import jax.numpy as jnp
    import torch

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    block, token = (torch.as_tensor(a, dtype=torch.int64) for a in (blocks, slots))
    rope = latent[block, token // 2, token % 2, 512:576]  # two tokens a latent row
    rope = torch.cat((rope, -rope), dim=-1).float().numpy()
    got = index[blocks, slots]
    got_exponent = index[blocks, page + slots // 128, slots % 128]
    with jax.default_device(jax.devices("cpu")[0]):
        key = ref.key_from_projection(
            jnp.asarray(rope).astype(jnp.bfloat16), jnp.ones(128), jnp.zeros(128), eps
        )
        want, want_exponent = dsa_cache.quantize_index_keys(key)
        same, beyond, difference = key_steps(got, got_exponent, want, want_exponent)
    return (
        int((~same).sum()),
        int(beyond.sum()),
        float(difference.max(initial=0.0)),
    )


def run_dsa_keys(args, records: harness.Records) -> None:
    """Kimi-K3.1 index keys under the rope-probe synthetic indexer (BF16
    pages): a committed key is the FP8 encoding of LayerNorm([k_pe, -k_pe])
    of its committed latent row's rope lanes. Every rank checks the positions
    it owns."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    if k3.indexer_mode() != "rope-probe" or k3.CACHE_DTYPE not in ("auto", "bfloat16"):
        harness.fail(
            "check dsa-keys needs --dsa-config, --synthetic-indexer rope-probe "
            "and BF16 pages"
        )
    prompt_tokens = args.prompt_tokens or k3.logical() + 16
    ids, _ = k3.fixed_prompt(prompt_tokens, args.model)
    prompts = batch_prompts([np.asarray(ids, np.int32)], ids, args.batch)
    pages = -(-(prompt_tokens + args.generate + 64) // k3.logical()) + 1
    ops = DecodeOps(
        args.presharded,
        num_slots=2 * args.batch + 1,
        batch_sizes=(args.batch,),
        **k3.decode_ops_options(),
    )
    pools = k3.Pools(len(prompts) * pages + 1, pages, 1)
    decoder = k3.decoder(
        ops, pools, max_num_seqs=args.batch, max_model_len=pages * k3.logical()
    )
    # Fail before the long decode if the host reference cannot run.
    index_key_mismatches(
        torch.zeros((1, 128, 2, 640), dtype=torch.bfloat16),
        np.zeros((1, *dsa_cache.index_block_shape(k3.PAGE)), np.uint8),
        np.zeros(1, np.int64),
        np.zeros(1, np.int64),
        k3.PAGE,
        k3.DSA[0].norm_eps,
    )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]
    decoder.step(
        mk.StepInput([k3.request(f"r{n}", p, tables[n]) for n, p in enumerate(prompts)])
    )
    for _ in range(args.generate - 1):
        decoder.step(
            mk.StepInput(
                [k3.request(f"r{n}", [0], tables[n]) for n in range(len(prompts))]
            )
        )
    # Positions written: each prompt and its generated inputs. The fixtures'
    # pages are owned by mesh position (no owner table).
    page, group = k3.PAGE, k3.logical() // k3.PAGE
    rank = ops.rank
    blocks, slots = [], []
    for prompt, table in zip(prompts, tables, strict=True):
        total = len(prompt) + args.generate - 1
        positions = np.asarray(
            [p for p in range(total) if (p // page) % group == rank], np.int64
        )
        blocks.append(table[positions // page // group])
        slots.append(positions % page)
    blocks, slots = np.concatenate(blocks), np.concatenate(slots)
    if not len(slots):
        harness.fail(f"rank {rank} owns no position: raise --prompt-tokens")
    eps = k3.DSA[0].norm_eps
    bad = beyond = 0
    worst = 0.0
    index = ops.index_cache.cpu().numpy()
    for m in range(len(layout.MLA)):
        latent = ops.unified_pools[ops.pool_map[m]].cpu()
        differ, over, largest = index_key_mismatches(
            latent, index[:, m], blocks, slots, page, eps
        )
        bad, beyond = bad + differ, beyond + over
        worst = float(np.maximum(worst, largest))  # NaN-propagating
    checked = len(slots) * len(layout.MLA)
    message = (
        f"dsa-keys: {checked - bad}/{checked} keys byte-identical, {beyond} more "
        f"than one FP8 step off, worst dequantized difference {worst:.3g}"
    )
    harness.log_host(f"r{rank:02d} {message}")  # every rank reports its own
    records.add(
        check="dsa-keys",
        batch=args.batch,
        requests=len(prompts),
        checked=checked,
        mismatched=bad,
        beyond=beyond,
        worst=worst,
    )
    # LayerNorm reductions may round differently in-kernel than in XLA: allow
    # rare one-step FP8 differences, never a wrong slot (a zero or foreign key).
    if bad > checked // 100 or beyond:
        harness.fail(message)


ENCODE_ROWS = 512  # keys per grid step of the encode check (whole BF16 tiles)
ARENA_SLOTS = 8  # gathered page slots of the arena-order check


def arena_order_pages(page_rows: int, packing: int, dtype) -> np.ndarray:
    """``[ARENA_SLOTS, page_rows, packing, 640]`` stand-in pool pages: token
    ``r * packing + i`` of slot ``s`` at ``[s, r, i]``, its rope lanes
    512:516 holding ``s`` and its position's three base-8 digits (exact in
    FP8 and BF16), every other lane zero."""
    pages = np.zeros((ARENA_SLOTS, page_rows, packing, 640), np.float32)
    token = np.arange(page_rows * packing).reshape(page_rows, packing)
    pages[..., 512] = np.arange(ARENA_SLOTS)[:, None, None]
    for d in range(3):
        pages[..., 513 + d] = token // 8 ** (2 - d) % 8
    return pages.astype(dtype)


def arena_order_mismatches(
    rope: np.ndarray, order: np.ndarray, page_tokens: int
) -> int:
    """Arena rows of the first ``order[0]`` pages whose rope lanes 0:4
    (``rope``, ``[rows, 4]``) do not name token ``c % page_tokens`` of slot
    ``order[1 + c // page_tokens]``."""
    rows = int(order[0]) * page_tokens
    c = np.arange(rows)
    token = c % page_tokens
    want = np.stack(
        [order[1 + c // page_tokens]] + [token // 8 ** (2 - d) % 8 for d in range(3)],
        -1,
    )
    return int((rope[:rows].astype(np.float32) != want).any(-1).sum())


def run_arena_order(args, records: harness.Records) -> None:
    """``expand_prefix_pages`` under Mosaic on every rank's own chip over
    :func:`arena_order_pages` in a shuffled slot order with two pages unused:
    each arena row's rope lanes (key lanes 128:132 of every head) against
    :func:`arena_order_mismatches`' order."""
    import jax.numpy as jnp
    import ml_dtypes
    import torch
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.layout import cache_layout

    cache = cache_layout(k3.CACHE_DTYPE)
    page_rows, packing, _ = cache.block_shape
    page_tokens = page_rows * packing
    dtype = getattr(ml_dtypes, cache.dtype)
    pages = arena_order_pages(page_rows, packing, dtype)
    order = np.asarray([ARENA_SLOTS - 2, 5, 2, 7, 0, 3, 6, 1, 4], np.int32)
    up = np.zeros((512, 3 * 256), ml_dtypes.bfloat16)

    def body(pages, order, up):
        keys, _ = kernels.expand_prefix_pages(pages[0], order[0], up[0])
        return (keys[0][:, :, 128:132].astype(jnp.float32)[None],)

    mesh, rank = bridge.kernel_mesh("tp", 32)
    op = bridge.register_op(
        f"pallas::k31_arena_order_check_{cache.dtype}",
        mesh,
        body,
        (P("tp"),) * 3,
        (P("tp"),),
    )
    if pages.dtype.itemsize == 1:  # torch takes no FP8 numpy arrays: their bytes
        device_pages = bridge.upload(pages.view(np.uint8)).view(
            getattr(torch, cache.dtype)
        )
    else:
        device_pages = bridge.upload(pages)
    keys = op(device_pages, bridge.upload(order), bridge.upload(up))[0]
    keys = keys[0].cpu().numpy()  # [heads, arena rows, 4]
    bad = sum(arena_order_mismatches(head, order, page_tokens) for head in keys)
    rows = int(order[0]) * page_tokens * keys.shape[0]
    message = f"arena-order: {rows - bad}/{rows} arena rows in page token order"
    harness.log_host(f"r{rank:02d} {message}")  # every rank reports its own
    records.add(check="arena-order", rows=rows, misplaced=bad)
    if bad:
        harness.fail(message)


def reference_encodings(keys):
    """The in-kernel encodings of BF16 ``keys`` [n, 128] computed by XLA on the
    host CPU: ``dsa_cache.index_key_rows``' codes and exponents, and
    ``encode_index_keys``' values with the query floor and its exponent in
    every lane (writable numpy copies; BF16 values as FP32)."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    with jax.default_device(jax.devices("cpu")[0]):
        keys = jnp.asarray(keys)
        codes, exponents = dsa_cache.index_key_rows(keys)
        values, exponent = dsa_cache.encode_index_keys(
            keys, floor=dsa_cache.QUERY_AMAX_FLOOR
        )
        return (
            np.array(codes, np.float32),
            np.array(exponents, np.float32),
            np.array(values),
            np.array(jnp.broadcast_to(exponent, values.shape)),
        )


def encode_mismatches(keys, codes, exponents, values, value_exponents):
    """``(key rows, query rows)`` of BF16 ``keys`` [n, 128] whose in-kernel
    encodings differ from XLA's on the host CPU: the key's ``codes`` and
    ``exponents`` (``dsa_cache.index_key_rows``) against
    ``quantize_index_keys``; the query ``values`` and ``value_exponents``
    (``encode_index_keys`` with the query floor, the exponent in every lane)
    against ``quantize_index_queries``."""
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    with jax.default_device(jax.devices("cpu")[0]):
        keys = jnp.asarray(keys)
        got_bytes, got_exponent = (
            np.asarray(a)
            for a in dsa_cache.index_key_bytes(
                jnp.asarray(codes, jnp.bfloat16), jnp.asarray(exponents, jnp.bfloat16)
            )
        )
        want_bytes, want_exponent = (
            np.asarray(a) for a in dsa_cache.quantize_index_keys(keys)
        )
        want_values, want_scale = (
            np.asarray(a) for a in dsa_cache.quantize_index_queries(keys)
        )
        exponent = np.asarray(value_exponents, np.int32)
        scale = np.asarray(dsa_cache.power_of_two(jnp.asarray(exponent[:, 0]) - 127))
    key_rows = (got_bytes != want_bytes).any(-1) | (got_exponent != want_exponent)
    value_rows = (
        np.ascontiguousarray(values, np.float32).view(np.uint32)
        != want_values.view(np.uint32)
    ).any(-1)  # bit for bit: -0 differs from +0
    query_rows = (
        value_rows | (scale != want_scale) | (exponent != exponent[:, :1]).any(-1)
    )
    return int(key_rows.sum()), int(query_rows.sum())


def run_dsa_encode(args, records: harness.Records) -> None:
    """The decode kernel's FP8 encodings under Mosaic, each rank on its own
    chip with no collectives: ``dsa_cache.index_key_rows`` and
    ``encode_index_keys`` with the query floor over every finite BF16 value
    and every BF16 amax, against XLA on the host CPU."""
    import jax
    import jax.numpy as jnp
    import torch
    from jax.experimental import pallas as pl
    from jax.sharding import PartitionSpec as P

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    keys = np.concatenate((ref.finite_bf16_keys(), ref.every_amax_keys()))
    rows = -(-len(keys) // ENCODE_ROWS) * ENCODE_ROWS
    keys = np.concatenate((keys, np.zeros((rows - len(keys), 128), keys.dtype)))

    def kernel(keys_ref, codes_ref, exponents_ref, values_ref, value_exponents_ref):
        codes_ref[...], exponents_ref[...] = dsa_cache.index_key_rows(keys_ref[...])
        values, exponent = dsa_cache.encode_index_keys(
            keys_ref[...], floor=dsa_cache.QUERY_AMAX_FLOOR
        )
        values_ref[...] = values
        value_exponents_ref[...] = jnp.broadcast_to(exponent, values.shape)

    block = pl.BlockSpec((ENCODE_ROWS, 128), lambda i: (i, 0))
    shapes = (
        jax.ShapeDtypeStruct((rows, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, 128), jnp.float32),
        jax.ShapeDtypeStruct((rows, 128), jnp.int32),
    )

    def body(local):
        outputs = pl.pallas_call(
            kernel,
            grid=(rows // ENCODE_ROWS,),
            in_specs=[block],
            out_specs=[block] * 4,
            out_shape=shapes,
            name="k31_encode_check",
        )(local[0])
        return tuple(output[None] for output in outputs)

    mesh, rank = bridge.kernel_mesh("tp", 32)
    op = bridge.register_op(
        "pallas::k31_encode_check", mesh, body, (P("tp"),), (P("tp"),) * 4
    )

    def host(tensor):
        tensor = tensor[0].cpu()
        return (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()

    outputs = [host(tensor) for tensor in op(bridge.upload(keys))]
    key_rows, query_rows = encode_mismatches(keys, *outputs)
    message = (
        f"dsa-encode: keys {rows - key_rows}/{rows} rows exact, "
        f"queries {rows - query_rows}/{rows} rows exact"
    )
    harness.log_host(f"r{rank:02d} {message}")  # every rank reports its own
    records.add(check="dsa-encode", rows=rows, key_rows=key_rows, query_rows=query_rows)
    if key_rows or query_rows:
        harness.fail(message)


DSA_SCORE_PARTS = ("inputs", "queries", "scores")
DSA_SELECT_PARTS = ("select", "attend")
ATTEND_LAYERS = (0, 12, 23)  # the MLA layers the attend part samples
TIE_SEED = 7  # tie_index_keys' keys


def dp_requests(rank: int, rows: int, group: int) -> tuple[list[int], int, int]:
    """``(requests, member, dp)``: the checks' requests ``rank``'s attention
    DP group decodes, in group-row order, and ``rank``'s member index in its
    ``group``-rank DCP group. ``Pools.scheduled`` puts request ``n`` in DP
    group ``n % dp`` and the decoder places a group's requests in its rows
    in order, so group row ``q`` holds request ``g + q dp``. At DP1: every
    request, and the rank."""
    from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config

    dp = kernel_config.active().attention_dp
    return list(range(rank // group, rows, dp)), rank % group, dp


def row_tokens(position: int, rank: int, page: int, group: int):
    """``(cached, new)`` of a row at ``position`` on ``rank`` (local page
    ``j`` is global page ``j group + rank``, local or global tables): the
    local indices of its cached tokens below ``position``, ascending, and the
    new token's local index if this rank owns its page, else None."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    tokens = np.arange(page)
    cached = [
        j * page + tokens[(j * group + rank) * page + tokens < position]
        for j in range(dsa_decode.local_pages(position, rank, page, group))
    ]
    new = None
    if position // page % group == rank:
        new = position // page // group * page + position % page
    return (np.concatenate(cached) if cached else np.zeros(0, np.int64)), new


def masked_tokens(position: int, rank: int, page: int, group: int):
    """The local indices of a row's score entries the scores contract
    (``dsa_decode``) holds at -inf: ``[L, L rounded up to 128)`` with ``L =
    dsa_decode.local_length(position, rank, page, group)``. Every later entry
    is arbitrary (unwritten; stale VMEM on TPU), so the select bounds each row
    by ``L``, not by the context length; with a row split into parts of ``P``
    local tokens (``P`` from ``dsa_decode.score_shape``), part ``p``'s bound
    is ``clip(L - p P, 0, P)``."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    length = int(dsa_decode.local_length(position, rank, page, group))
    return np.arange(length, -(-length // 128) * 128)


def key_step_mismatch(got, got_exponent, want, want_exponent):
    """``(differs, beyond)`` of one committed index key (``got`` [128] uint8
    and its exponent byte) against the expected encoding (:func:`key_steps`):
    it differs unless both are equal, and is beyond when the dequantized keys
    are more than one FP8 step apart or not finite."""
    same, beyond, _ = key_steps(got, got_exponent, want, want_exponent)
    return bool(not same), bool(beyond)


def query_pair_mismatches(queries, weights, want_queries, want_weights):
    """``(differing, beyond)`` over (row, head) pairs of the kernel's FP8
    query values ``[n, 32, 128]`` and head weights ``[n, 32]`` against the
    reference's: a pair differs unless both are bit-identical, and is beyond
    when ``|q w - q_ref w_ref|`` exceeds an eighth (one E4M3 step at its
    coarsest) of the pair's largest ``|q_ref w_ref|``; a power-of-two scale
    moved between ``q`` and ``w`` stays within."""
    queries = np.asarray(queries, np.float64)
    want_queries = np.asarray(want_queries, np.float64)
    weights = np.asarray(weights, np.float64)
    want_weights = np.asarray(want_weights, np.float64)
    got = queries * weights[..., None]
    want = want_queries * want_weights[..., None]
    same = (queries == want_queries).all(-1) & (weights == want_weights)
    bound = np.abs(want).max(-1, keepdims=True) / 8
    beyond = ~(np.abs(got - want) <= bound).all(-1)
    return int((~same).sum()), int(beyond.sum())


SCORE_TOLERANCE = 1e-5  # max |s - ref| / max |ref| per layer and row


def index_keys_at(index, table, layer, tokens, page):
    """FP32 keys ``[n, 128]`` of a row's local ``tokens`` in MLA layer
    ``layer``'s pages of the index cache ``index`` ``[blocks, 24, page + 32,
    128]`` (numpy; ``table`` the row's local block table)."""
    import jax.numpy as jnp

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    tokens = np.asarray(tokens, np.int64)
    blocks, slots = np.asarray(table)[tokens // page], tokens % page
    key_bytes = index[blocks, layer, slots]
    exponents = index[blocks, layer, page + slots // 128, slots % 128]
    return np.asarray(
        dsa_cache.dequantize_index_keys(jnp.asarray(key_bytes), jnp.asarray(exponents))
    )


def score_errors(scores, queries, weights, keys):
    """``max |s - ref| / max |ref|`` (FP64) of a row's kernel ``scores [n]``
    against ``dsa_reference.scores`` of its query values ``[32, 128]``, head
    weights ``[32]`` and ``keys [n, 128]``; ``inf`` for a non-finite score."""
    import jax.numpy as jnp

    from tools.megakernel import dsa_reference as ref

    want = np.asarray(
        ref.scores(
            jnp.asarray(queries, jnp.float32)[None],
            jnp.asarray(weights, jnp.float32)[None],
            jnp.asarray(keys),
        )
    )[0].astype(np.float64)
    got = np.asarray(scores, np.float64)
    if not np.isfinite(got).all():
        return float("inf")
    largest = max(float(np.abs(want).max(initial=0.0)), 1e-30)
    return float(np.abs(got - want).max(initial=0.0)) / largest


def selection_mismatches(masks, new_selected, want_masks, want_new):
    """``(missing, extra)``: selected tokens of the reference (``want``) the
    kernel's members' ``masks`` lack, and the kernel's the reference lacks,
    over every member and row; the new token's flags count as one token."""
    missing = extra = 0
    for got, want in zip(masks, want_masks, strict=True):
        got, want = np.asarray(got, bool), np.asarray(want, bool)
        missing += int((want & ~got).sum())
        extra += int((got & ~want).sum())
    got_new, want_new = np.asarray(new_selected, bool), np.asarray(want_new, bool)
    return missing + int((want_new & ~got_new).sum()), extra + int(
        (got_new & ~want_new).sum()
    )


def attention_error(merged, want):
    """``max |merged - want| / max |want|`` in FP64 over the whole arrays
    (:func:`dsa_micro.fold_errors`' output term); ``inf`` if ``merged`` is not
    finite."""
    from tools.megakernel import dsa_micro

    merged, want = np.asarray(merged, np.float64), np.asarray(want, np.float64)
    if not np.isfinite(merged).all():
        return float("inf")
    flat, lse = (1, -1, merged.shape[-1]), np.zeros((1, 1))
    return float(
        dsa_micro.fold_errors(merged.reshape(flat), lse, want.reshape(flat), lse)[0][0]
    )


def attend_summary(merged, want, tolerance: float, samples) -> dict:
    """``attend_samples`` (the heads), ``attend_worst`` (the largest
    :func:`attention_error` of a head), ``attend_worst_at`` (that head's
    sample), ``attend_over`` (heads above ``tolerance``) and the long rows'
    heads whose new token the select took (``attend_flagged``) or left
    (``attend_unflagged``) of sampled merged outputs ``merged [..., 512]``
    against the reference's ``want``; ``samples`` holds a head's ``(rank,
    layer, row, head, long, flagged)`` in ``merged``'s order."""
    merged = np.asarray(merged).reshape(-1, np.shape(merged)[-1])
    want = np.asarray(want).reshape(merged.shape)
    if len(samples) != len(merged):
        raise ValueError(f"{len(samples)} samples for {len(merged)} heads")
    errors = [attention_error(got[None], ref[None]) for got, ref in zip(merged, want)]
    worst = "none"
    if errors:
        rank, layer, row, head, long, flagged = samples[int(np.argmax(errors))]
        worst = (
            f"r{rank:02d} layer {layer} row {row} head {head} "
            f"{'long' if long else 'short'} {'flagged' if flagged else 'unflagged'}"
        )
    return {
        "attend_samples": len(errors),
        "attend_worst": max(errors, default=0.0),
        "attend_worst_at": worst,
        "attend_over": sum(not error <= tolerance for error in errors),
        "attend_flagged": sum(s[4] and s[5] for s in samples),
        "attend_unflagged": sum(s[4] and not s[5] for s in samples),
    }


def tie_index_keys(index, page: int, patterns: int):
    """numpy: ``index`` ``[blocks, layers, page + 32, 128]`` uint8 with every
    token slot ``s`` of every block and layer holding the ``s % patterns``-th
    of ``patterns`` seeded keys (bytes and exponent, the same on every rank):
    a row's cached tokens then score at most ``patterns`` values."""
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    rng = np.random.default_rng(TIE_SEED)
    keys = rng.standard_normal((patterns, 128), np.float32) * np.exp2(
        rng.integers(-4, 5, (patterns, 1))
    ).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        key_bytes, exponent = (
            np.asarray(a)
            for a in dsa_cache.quantize_index_keys(
                jnp.asarray(keys.astype(ml_dtypes.bfloat16))
            )
        )
    slots = np.arange(page)
    out = np.array(index, np.uint8)
    out[:, :, slots] = key_bytes[slots % patterns]
    out[:, :, page + slots // 128, slots % 128] = exponent.reshape(-1)[slots % patterns]
    return out


def needs_tie_quota(values, k: int) -> bool:
    """Whether a row's candidate scores ``values`` take a tie quota at their
    ``k``-th highest: more than ``k`` of them reach it (the select's tie
    round)."""
    values = np.asarray(values)
    if len(values) <= k:
        return False
    tau = np.sort(values)[::-1][k - 1]
    return bool((values >= tau).sum() > k)


def select_row_summary(masks, new_selected, want_masks, want_new, *, sparse: bool):
    """One row's selection against the reference's: ``masks`` and
    ``want_masks`` per member, bool ``[T]`` (its local tokens; the new token
    never among them), ``new_selected`` and ``want_new`` the new token's
    flags. ``missing`` and ``extra`` as :func:`selection_mismatches`;
    ``count_bad`` 1 when a ``sparse`` row's masks and flag do not hold
    ``INDEX_TOPK`` tokens; ``short_bad`` 1 when a row within the top-k has a
    selected token in a mask (it folds every token) or its flag clear."""
    from vllm_torchtpu.megakernel.kimi_k3 import dsa

    missing, extra = selection_mismatches(masks, new_selected, want_masks, want_new)
    held = sum(int(np.asarray(mask, bool).sum()) for mask in masks)
    return {
        "missing": missing,
        "extra": extra,
        "count_bad": int(sparse and held + bool(new_selected) != dsa.INDEX_TOPK),
        "short_bad": int(not sparse and (held > 0 or not new_selected)),
    }


def _index_debug_decode(args, prompts, pages, after_first=None):
    """Decode ``prompts`` (a request per row of ``args.batch``) for
    ``args.generate`` steps with the decoder's check outputs on: ``(ops,
    tables, debug)``, ``debug`` the last step's outputs by
    ``dsa_decode.DEBUG_OUTPUTS`` name (numpy; BF16 as FP32).
    ``after_first(ops)`` runs after the first step. The checks key a rank's
    pages by its mesh position: a decoder with an owner table is refused."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode

    ops = DecodeOps(
        args.presharded,
        num_slots=2 * args.batch + 1,
        batch_sizes=(args.batch,),
        index_debug=True,
        **k3.decode_ops_options(),
    )
    pools = k3.Pools(len(prompts) * pages + 1, pages, 1)
    decoder = k3.decoder(
        ops, pools, max_num_seqs=args.batch, max_model_len=pages * k3.logical()
    )
    if ops.owner_table is not None:
        harness.fail(
            "the DSA checks own pages by mesh position, not by an owner table "
            f"{ops.owner_table}"
        )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]
    decoder.step(
        mk.StepInput([k3.request(f"r{n}", p, tables[n]) for n, p in enumerate(prompts)])
    )
    if after_first is not None:
        after_first(ops)
    for _ in range(args.generate - 1):
        decoder.step(
            mk.StepInput(
                [k3.request(f"r{n}", [0], tables[n]) for n in range(len(prompts))]
            )
        )
    debug = {
        name: (tensor.float() if tensor.dtype == torch.bfloat16 else tensor)
        .cpu()
        .numpy()
        for name, tensor in zip(
            dsa_decode.DEBUG_OUTPUTS, ops.index_debug_outputs, strict=True
        )
    }
    return ops, tables, debug


def run_dsa_scores(args, records: harness.Records) -> None:
    """Kimi-K3.1's indexer in the decode kernel against the JAX reference
    (``--synthetic-indexer random``), from the last step's check outputs
    (``DecodeOps(index_debug=True)``); that step must score (a row's context
    above ``INDEX_TOPK``), and every row decodes a request of its own.
    ``inputs``: each MLA layer's input ``x`` and q-down latent ``q_c`` equal
    on every rank, and each row's committed key the reference key of ``x`` (0
    beyond one FP8 step, at most max(16, 1%) differing); every rank checks its
    own positions. ``queries``: the scorer's FP8 query values and head weights
    equal on every rank, against ``dsa_reference.index_query`` of the kernel's
    ``x`` and ``q_c`` with the synthetic heads (0 pairs beyond one step, at
    most 5% differing). The kernel forms its queries from its own q-down
    latent, so this part also checks the ``q_c`` output: one never written
    puts every pair beyond. ``scores``: the kernel's scores of each row's
    local tokens below its position and of its new token on the owner,
    against ``dsa_reference.scores`` of the kernel's own queries and weights
    and the committed keys (within ``SCORE_TOLERANCE``), and its
    :func:`masked_tokens` -inf."""
    import hashlib
    import json

    import jax
    import jax.numpy as jnp
    import torch

    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3 import dsa, layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache, dsa_decode

    parts = set(args.parts.split(","))
    if not parts <= set(DSA_SCORE_PARTS):
        harness.fail(f"--parts takes {', '.join(DSA_SCORE_PARTS)}, not {args.parts}")
    if k3.DSA is None or k3.indexer_mode() not in ("random", None):
        harness.fail(
            "check dsa-scores needs --dsa-config, and --synthetic-indexer random "
            "or a pre-shard with the indexer weights"
        )
    prompt_tokens = args.prompt_tokens or k3.logical() + 16
    ids, _ = k3.fixed_prompt(prompt_tokens, args.model)
    # Attention DP groups score apart (request n is in group n % dp): each
    # gets a long prompt of its own, 977 tokens shorter per group.
    dp = dp_requests(0, args.batch, k3.logical() // k3.PAGE)[2]
    longs = [np.asarray(ids[: len(ids) - 977 * g], np.int32) for g in range(dp)]
    prompts = batch_prompts(longs, ids, args.batch, fill_above=dp)
    # The last step's inputs sit at these positions; the op's check outputs are
    # that step's.
    positions = [len(prompt) + args.generate - 2 for prompt in prompts]
    if min(positions[:dp]) + 1 <= dsa.INDEX_TOPK:
        harness.fail(
            f"check dsa-scores needs a last step that scores: a context above "
            f"{dsa.INDEX_TOPK} tokens (raise --prompt-tokens or --generate)"
        )
    pages = -(-(prompt_tokens + args.generate + 64) // k3.logical()) + 1
    ops, tables, debug = _index_debug_decode(args, prompts, pages)
    index = ops.index_cache.cpu().numpy()
    rank, rows, layers = ops.rank, len(prompts), len(layout.MLA)
    page, group = k3.PAGE, k3.logical() // k3.PAGE
    eps, hidden = k3.DSA[0].norm_eps, debug["x"].shape[-1]
    weights = IndexerWeights(args.presharded, rank, layers, hidden)
    # Attention DP: the debug rows are the group's; its pages are on its ranks.
    requests, member, dp = dp_requests(rank, rows, group)
    held, data_group = len(requests), rank // group
    tables = [np.asarray(table) % index.shape[0] for table in tables]

    def replicated(*arrays):
        """Whether every rank of this rank's DP group holds the same
        ``arrays`` (a collective)."""
        mine = hashlib.md5(
            b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
        ).hexdigest()
        every = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(every, (data_group, mine))
        return len({h for g, h in every if g == data_group}) == 1

    summary, ok = {}, True
    with jax.default_device(jax.devices("cpu")[0]):
        if "inputs" in parts:
            weight, bias = weights.key_norm()
            checked = differing = beyond = 0
            for m in range(layers):
                wk = jnp.asarray(weights.key_projection(m))
                for q, b in enumerate(requests):
                    position = positions[b]
                    if position // page % group != member:
                        continue
                    x = jnp.asarray(debug["x"][m, q : q + 1], jnp.bfloat16)
                    key = ref.index_key(
                        x, wk, jnp.asarray(weight[m]), jnp.asarray(bias[m]), eps
                    )
                    want, want_exponent = (
                        np.asarray(a)[0] for a in dsa_cache.quantize_index_keys(key)
                    )
                    block, slot = tables[b][position // page // group], position % page
                    differs, far = key_step_mismatch(
                        index[block, m, slot],
                        index[block, m, page + slot // 128, slot % 128],
                        want,
                        want_exponent,
                    )
                    checked, differing = checked + 1, differing + differs
                    beyond += far
            same = replicated(debug["x"][:, :held], debug["q_c"][:, :held])
            summary.update(
                keys=checked,
                keys_differing=differing,
                keys_beyond=beyond,
                inputs_replicated=same,
            )
            ok = ok and same and not beyond and differing <= max(16, checked // 100)
        if "queries" in parts:
            pairs = differing = beyond = 0
            for m in range(layers):
                wq = weights.query_heads(m)
                wv = weights.weight_rows(m).T
                want_q, want_w = ref.index_query(
                    jnp.asarray(debug["q_c"][m, :held], jnp.bfloat16),
                    jnp.asarray(wq),
                    jnp.asarray(debug["x"][m, :held], jnp.bfloat16),
                    jnp.asarray(np.ascontiguousarray(wv)),
                )
                found = query_pair_mismatches(
                    debug["queries"][m, :held],
                    debug["weights"][m, :held, :, 0],
                    np.asarray(want_q),
                    np.asarray(want_w),
                )
                pairs, differing = pairs + held * dsa.INDEX_HEADS, differing + found[0]
                beyond += found[1]
            same = replicated(
                debug["queries"][:, :held], debug["weights"][:, :held, :, 0]
            )
            summary.update(
                pairs=pairs,
                pairs_differing=differing,
                pairs_beyond=beyond,
                queries_replicated=same,
            )
            ok = ok and same and not beyond and differing <= pairs // 20
        if "scores" in parts:
            worst, over, masked_bad, scored = 0.0, 0, 0, 0
            for m in range(layers):
                by_row = dsa_decode.scores_by_row(debug["scores"][m], args.batch // dp)
                for q, b in enumerate(requests):
                    position = positions[b]
                    cached, new = row_tokens(position, member, page, group)
                    tokens = cached if new is None else np.append(cached, new)
                    if len(tokens):
                        keys = index_keys_at(index, tables[b], m, tokens, page)
                        error = score_errors(
                            by_row[q, tokens],
                            debug["queries"][m, q],
                            debug["weights"][m, q, :, 0],
                            keys,
                        )
                        worst, scored = max(worst, error), scored + 1
                        over += not error <= SCORE_TOLERANCE
                    masked = masked_tokens(position, member, page, group)
                    masked_bad += int((by_row[q, masked] != -np.inf).sum())
            summary.update(
                score_rows=scored,
                scores_over=over,
                score_worst=worst,
                masked_not_inf=masked_bad,
            )
            ok = ok and not over and not masked_bad
    summary["pass"] = bool(ok)
    harness.log_host(f"r{rank:02d} dsa-scores {json.dumps(summary)}")
    records.add(check="dsa-scores", batch=args.batch, parts=sorted(parts), **summary)
    if not ok:
        harness.fail(f"dsa-scores: {summary}")


class IndexerWeights:
    """The indexer weights of the checks' JAX reference, per MLA layer: the
    synthetic indexer's (``--synthetic-indexer random``, drawn from its seed)
    or, without one, the pre-shard's own, every rank's index head read from
    its rank directory (``rank_loader.RankDir.read_indexer``)."""

    def __init__(self, presharded, rank: int, layers: int, hidden: int):
        import ml_dtypes

        from vllm_torchtpu.megakernel.kimi_k3 import dsa

        self.dsa, self.layers, self.hidden = dsa, layers, hidden
        self.bf16 = ml_dtypes.bfloat16
        self.seed = None if k3.DSA[1] is None else k3.DSA[1][1]
        if self.seed is not None:
            return
        from vllm_torchtpu.megakernel.kimi_k3.kernels.rank_loader import RankDir

        own = RankDir(presharded, rank).read_indexer(
            names=("index_wk", "index_knorm_weight", "index_knorm_bias")
        )
        heads = [
            RankDir(presharded, r).read_indexer(names=("index_wq", "index_wv"))
            for r in range(dsa.INDEX_HEADS)
        ]
        self.wk = own["index_wk"]
        self.norm = (own["index_knorm_weight"], own["index_knorm_bias"])
        self.wq = np.concatenate([h["index_wq"] for h in heads], axis=-1)
        self.wv = np.concatenate([h["index_wv"] for h in heads], axis=1)

    def key_norm(self):
        """``(weight, bias)`` ``[layers, 1, 128]``."""
        if self.seed is None:
            return self.norm
        return self.dsa.synthetic_key_norm(self.seed, self.layers, self.bf16)

    def key_projection(self, m: int):
        """``[hidden, 128]``: ``wk`` transposed."""
        if self.seed is None:
            return self.wk[m]
        return self.dsa.synthetic_key_projection(self.seed, m, self.hidden, self.bf16)

    def query_heads(self, m: int):
        """``[q_lora, 32 * 128]``: every head's ``wq_b`` rows, transposed."""
        if self.seed is None:
            return self.wq[m]
        return self.dsa.synthetic_query_heads(
            self.seed, m, self.dsa.QUERY_NORM_LANES, self.bf16
        )

    def weight_rows(self, m: int):
        """``[32, hidden]``: ``weights_proj``."""
        if self.seed is None:
            return self.wv[m]
        return self.dsa.synthetic_weight_rows(self.seed, m, self.hidden, self.bf16)


def run_dsa_select(args, records: harness.Records) -> None:
    """Kimi-K3.1's select and sparse attention in the decode kernel against
    the JAX reference (``--synthetic-indexer random``), from the last step's
    check outputs: rows ``n < --long-rows`` decode the fixed prompt's first
    ``prompt_tokens - 977 n`` tokens, the others short windows of it. Every
    rank gathers each MLA layer's long rows' scores and masks over its local
    tokens below ``dsa_decode.local_length`` (past it the mask words are not
    written in a step) from every member. ``select``: checks them, with its
    own new-token flag, against ``dsa_reference.decode_selection`` of those
    scores (:func:`select_row_summary`); each short row's own masks must be
    empty and its flag set. ``attend``: in the ``ATTEND_LAYERS``, for the long
    rows and the first short row, every rank gathers the cache-dtype latent
    rows each member's masks select (a short row: every token below its
    position), and checks its heads' merged outputs against
    ``dsa_reference.latent_attention`` of its absorbed queries over them and,
    if flagged, the new token's BF16 latent row as the kernel merges it (the
    ``new_latent`` output, not its committed cache-dtype row)
    (:func:`attend_summary`, within ``dsa_micro.FOLD_TOLERANCE``). Every
    summary counts ``tie_rows``, the long rows whose top-k took a tie quota
    (:func:`needs_tie_quota`); ``--tie-keys`` makes them (:func:`tie_index_keys`
    after the first step)."""
    import json

    import torch

    from tools.megakernel import dsa_micro
    from tools.megakernel import dsa_reference as ref
    from vllm_torchtpu.megakernel.kimi_k3 import dsa, layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_decode, dsa_select

    parts = set(args.parts.split(","))
    if not parts <= set(DSA_SELECT_PARTS):
        harness.fail(f"--parts takes {', '.join(DSA_SELECT_PARTS)}, not {args.parts}")
    if k3.DSA is None or k3.indexer_mode() not in ("random", None):
        harness.fail(
            "check dsa-select needs --dsa-config, and --synthetic-indexer random "
            "or a pre-shard with the indexer weights"
        )
    prompt_tokens = args.prompt_tokens or k3.logical() + 16
    if args.token_file:
        ids = np.load(args.token_file).astype(np.int32)[:prompt_tokens].tolist()
    else:
        ids, _ = k3.fixed_prompt(prompt_tokens, args.model)
    if len(ids) < prompt_tokens:
        harness.fail(
            f"check dsa-select: the prompt text holds {len(ids)} tokens, fewer "
            f"than --prompt-tokens {prompt_tokens}"
        )
    long_rows = min(args.long_rows, args.batch)
    lengths = [prompt_tokens - 977 * n for n in range(long_rows)]
    if not lengths or lengths[-1] + args.generate - 1 <= dsa.INDEX_TOPK:
        harness.fail(
            f"check dsa-select needs long rows whose last step's context exceeds "
            f"{dsa.INDEX_TOPK} tokens: {lengths} prompt tokens"
        )
    prompts = batch_prompts(
        [np.asarray(ids[:n], np.int32) for n in lengths], ids, args.batch, fill_above=1
    )
    positions = [len(prompt) + args.generate - 2 for prompt in prompts]
    pages = -(-(prompt_tokens + args.generate + 64) // k3.logical()) + 1
    page, group = k3.PAGE, k3.logical() // k3.PAGE

    def tie(ops):
        tied = tie_index_keys(ops.index_cache.cpu().numpy(), page, args.tie_keys)
        ops.index_cache.set_(torch.from_numpy(tied).to(ops.index_cache.device))

    ops, tables, debug = _index_debug_decode(
        args, prompts, pages, after_first=tie if args.tie_keys else None
    )
    rank, layers = ops.rank, len(layout.MLA)
    # Attention DP: the debug rows are the group's; its DCP group is its ranks.
    requests, member, dp = dp_requests(rank, len(prompts), group)
    data_group, row_of = rank // group, {b: q for q, b in enumerate(requests)}
    longs = [b for b in requests if b < long_rows]
    shorts = [b for b in requests if b >= long_rows]
    if not longs:
        harness.fail(f"check dsa-select: DP group {data_group} decodes no long row")
    tables = [np.asarray(table) % ops.index_cache.shape[0] for table in tables]
    lengths = [
        [int(dsa_decode.local_length(p, o, page, group)) for p in positions]
        for o in range(group)
    ]
    mine, totals = [], dict.fromkeys(("missing", "extra", "count_bad", "short_bad"), 0)
    for m in range(layers):
        by_row = dsa_decode.scores_by_row(debug["scores"][m], args.batch // dp)
        masks = dsa_select.unpack_mask(
            debug["mask"][m], args.batch // dp, by_row.shape[1]
        )
        masks = {b: masks[row_of[b], : lengths[member][b]] for b in requests}
        mine.append(
            {b: (by_row[row_of[b], : lengths[member][b]], masks[b]) for b in longs}
        )
        for b in shorts:
            found = select_row_summary(
                [masks[b]],
                debug["new_selected"][m, row_of[b], 0] != 0,
                [np.zeros_like(masks[b])],
                True,
                sparse=False,
            )
            totals = {key: totals[key] + found[key] for key in totals}
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (data_group, member, mine))
    # By DCP member (the page owner) in this rank's DP group, not by process.
    every = {o: theirs for g, o, theirs in gathered if g == data_group}
    if sorted(every) != list(range(group)):
        harness.fail(
            f"check dsa-select runs on the {group} DCP ranks, not {sorted(every)}"
        )
    tokens = max(lengths[o][b] for o in range(group) for b in longs)
    tie_rows = 0
    peaks = []  # per long row and layer: the most selected tokens one member holds
    for m in range(layers):
        local = []
        for o in range(group):
            scores = np.full((len(longs), tokens), np.nan, np.float32)
            for i, b in enumerate(longs):
                scores[i, : lengths[o][b]] = every[o][m][b][0]
            local.append(scores)
        for b in longs:
            values = np.concatenate([every[o][m][b][0] for o in range(group)])
            tie_rows += needs_tie_quota(values, dsa.INDEX_TOPK)
        if "select" not in parts:
            continue
        want, want_new = ref.decode_selection(
            local, [positions[b] for b in longs], page_tokens=page, group=group
        )
        for i, b in enumerate(longs):
            got = []
            for o in range(group):
                held = np.zeros(tokens, bool)
                held[: lengths[o][b]] = every[o][m][b][1]
                got.append(held)
            peaks.append(max(int(h.sum()) for h in got))
            found = select_row_summary(
                got,
                debug["new_selected"][m, row_of[b], 0] != 0,
                [want[o][i] for o in range(group)],
                want_new[i],
                sparse=True,
            )
            totals = {key: totals[key] + found[key] for key in totals}
    summary = {"rows": layers * len(requests), **totals}
    ok = not any(totals.values())
    if peaks:
        # Per-member shares of the top-k: what a one-round merge's per-member
        # candidate count must cover.
        peak = np.asarray(peaks)
        summary.update(
            member_peak_max=int(peak.max()),
            member_peak_p50=float(np.percentile(peak, 50)),
            member_peak_p90=float(np.percentile(peak, 90)),
            member_peak_p99=float(np.percentile(peak, 99)),
            **{
                f"rows_peak_over_{n}": int((peak > n).sum())
                for n in (96, 128, 192, 256, 384, 512)
            },
        )
    if "attend" in parts:
        dtypes = {ops.unified_pools[ops.pool_map[m]].dtype for m in ATTEND_LAYERS}
        if len(dtypes) != 1:
            harness.fail(f"check dsa-select: MLA pools of dtypes {dtypes}")
        fp8 = dtypes.pop() != torch.bfloat16
        kv = "fp8" if fp8 else "bf16"
        sampled = [*longs, *shorts[:1]]
        sent = {}
        for m in ATTEND_LAYERS:
            pool = ops.unified_pools[ops.pool_map[m]].cpu()
            bits = pool.view(torch.uint8 if fp8 else torch.int16).numpy()
            bits = bits.view(np.uint8 if fp8 else np.uint16)
            for b in sampled:
                cached, _ = row_tokens(positions[b], member, page, group)
                held = np.flatnonzero(mine[m][b][1]) if b < long_rows else cached
                sent[m, b] = dsa_micro.latent_values(bits, tables, b, held, kv)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (data_group, member, sent))
        received = {o: theirs for g, o, theirs in gathered if g == data_group}
        merged, want, samples = [], [], []
        for m in ATTEND_LAYERS:
            for b in sampled:
                rows = np.concatenate([received[o][m, b] for o in range(group)])
                handoff = debug["new_latent"][m, row_of[b], :3].reshape(-1)
                new = handoff[: dsa_micro.LATENT_LANES]
                flagged = bool(debug["new_selected"][m, row_of[b], 0] != 0)
                absorbed = debug["absorbed"][m, :, row_of[b]]
                want.append(
                    ref.latent_attention(absorbed, rows, new if flagged else None)[0]
                )
                merged.append(debug["merged"][m, :, row_of[b]])
                samples += [
                    (rank, m, b, h, b < long_rows, flagged)
                    for h in range(len(absorbed))
                ]
        found = attend_summary(
            np.stack(merged),
            np.stack(want),
            dsa_micro.FOLD_TOLERANCE[kv][0],
            samples,
        )
        summary.update(found)
        ok = ok and not found["attend_over"]
    summary["tie_rows"] = tie_rows
    summary["pass"] = bool(ok)
    harness.log_host(f"r{rank:02d} dsa-select {json.dumps(summary)}")
    records.add(check="dsa-select", batch=args.batch, parts=sorted(parts), **summary)
    if not summary["pass"]:
        harness.fail(f"dsa-select: {summary}")


def run_resume(args, records: harness.Records) -> None:
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps

    page, block = k3.PAGE, k3.logical()
    prompt_len = args.prompt_tokens or block + 3 * page + 37
    if prompt_len <= block + page:
        harness.fail(f"--prompt-tokens must exceed {block + page}")
    first, second, total = page + 10, 2 * page + 30, 2 * page + 62
    ids, tokenizer = k3.fixed_prompt(prompt_len, args.model)
    prompt = np.asarray(ids, np.int32)
    pages = -(-(prompt_len + total + 64) // block) + 1
    columns = -(-(prompt_len + total + 64) // page)
    ops = DecodeOps(args.presharded, num_slots=2 * 8 + 1, batch_sizes=(8,))
    pools = k3.Pools(6 * (pages + 3 * columns) + 1, pages, columns)
    decoder = k3.decoder(ops, pools, max_num_seqs=8, max_model_len=pages * block)
    harness.sync()
    decoder.warmup()
    sequence: dict[str, np.ndarray] = {}

    def allocate(hit=0, keep=None):
        """The block tables vLLM allocates for a recompute from a ``hit``-token
        prefix hit of the request holding ``keep``: MLA pages kept through
        the logical block holding the hit (a partial one copied on write),
        fresh after it; KDA state columns null before the hit's, its kept,
        fresh after it (align mode)."""
        kept = -(-hit // block)
        mla = (list(keep[0][:kept]) if hit else []) + pools.fresh(pages - kept)
        kda = []
        for g in range(3):
            table = [0] * columns
            if hit:
                table[hit // page - 1] = keep[1][g][hit // page - 1]
            table[hit // page :] = pools.fresh(columns - hit // page)
            kda.append(table)
        return mla, kda

    def row(req, computed, scheduled, tables):
        known = len(sequence[req])
        return mk.RequestInput(
            req_id=req,
            computed=computed,
            scheduled=scheduled,
            token_ids=sequence[req],
            num_tokens=known,
            num_prompt_tokens=prompt_len,
            block_tables=pools.tables(pools.scheduled(tables[0], 0), tables[1]),
        )

    def step(rows):
        out = decoder.step(mk.StepInput(rows)).tokens[0].cpu().tolist()
        return [int(t) for t in out if t >= 0][:1]

    def decode(req, until, tables):
        """Decode ``req`` until it holds ``until`` outputs, admitting it (its
        prompt in one step) unless the decoder knows it."""
        while len(sequence[req]) < prompt_len + until:
            known = len(sequence[req])
            computed = 0 if req not in decoder.slot_map else known - 1
            out = step([row(req, computed, known - computed, tables)])
            sequence[req] = np.append(sequence[req], out).astype(np.int32)

    def resume(req, hit, tables, forward=True):
        """``req`` preempted, then resumed from a ``hit``-token prefix hit: the
        rest recomputed in 512-token chunks, the last one decoding its next
        output. ``forward``: the runner hands the decoder the preemption."""
        if forward:
            decoder.step(mk.StepInput([], (req,)))
        tables = allocate(hit, tables)
        computed, known = hit, len(sequence[req])
        while computed < known:
            chunk = min(512, known - computed)
            out = step([row(req, computed, chunk, tables)])
            computed += chunk
        sequence[req] = np.append(sequence[req], out).astype(np.int32)
        return tables

    started = time.perf_counter()
    sequence["ref"] = prompt.copy()
    decode("ref", total, allocate())
    reference = sequence["ref"][prompt_len:].tolist()
    harness.log(
        f"reference: {total} outputs in {time.perf_counter() - started:.1f} s: "
        f"{tokenizer.decode(reference[:24])!r}"
    )
    # Preempted twice: resumed from a hit inside the prompt (a state the first
    # replay wrote), then from one past it (a state only a recompute wrote).
    hits = (block, (prompt_len + first) // page * page)
    sequence["pre"] = prompt.copy()
    tables = allocate()
    decode("pre", first, tables)
    tables = resume("pre", hits[0], tables)
    decode("pre", second, tables)
    tables = resume("pre", hits[1], tables)
    decode("pre", total, tables)
    resumed = sequence["pre"][prompt_len:].tolist()
    # The control resumes without the decoder hearing of the preemption.
    sequence["stale"] = prompt.copy()
    tables = allocate()
    decode("stale", first, tables)
    tables = resume("stale", hits[0], tables, forward=False)
    decode("stale", first + 16, tables)
    stale = sequence["stale"][prompt_len:].tolist()
    decoder.forget(("ref", "pre", "stale"))
    agree, stale_agree = _agree(resumed, reference), _agree(stale, reference)
    harness.log(
        f"{prompt_len}-token prompt, preempted after {first} and {second} outputs, "
        f"resumed from {hits[0]} and {hits[1]} tokens: the first {agree} of "
        f"{total} outputs agree with the uninterrupted decode; without the "
        f"preemption signal the first {stale_agree} of {len(stale)}"
    )
    records.add(
        check="resume",
        prompt_tokens=prompt_len,
        preempted_after=[first, second],
        hits=list(hits),
        agree=[agree, total],
        stale_agree=[stale_agree, len(stale)],
    )
    if agree < total:
        harness.fail("a resumed request's outputs differ from the uninterrupted decode")
    if stale_agree == len(stale):
        harness.fail("the control resumed without the preemption signal still agrees")


def _agree(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def run_masked(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import DecodeOps

    all_ids, tokenizer = k3.fixed_prompt(args.prompt_tokens, args.model)
    prompts = [
        np.asarray(all_ids, np.int32),
        np.asarray(
            tokenizer.encode(
                "Write a short story about a robot who learns to paint.",
                disallowed_special=(),
            ),
            np.int32,
        ),
    ]
    words = k3_program.VOCAB // 32
    # Speculative steps take up to eight tokens each.
    pages = -(-(args.prompt_tokens + 8 * args.generate + 64) // k3.logical()) + 1
    ops = DecodeOps(args.presharded, num_slots=2 * 4 + 1, batch_sizes=(1, 2, 4))
    pools = k3.Pools(16 * pages + 1, pages, 1)
    draft = None
    if args.dspark:
        from vllm_torchtpu.megakernel.kimi_k3.draft import DsparkDraft

        draft = DsparkDraft(
            max_position=args.prompt_tokens + 8 * args.generate + 64,
            checkpoint=args.dspark,
            presharded=args.presharded,
        )
    decoder = k3.decoder(
        ops,
        pools,
        max_num_seqs=4,
        max_model_len=pages * k3.logical(),
        draft=draft,
        # Seven drafts alone, three for two to four requests.
        schedule=((1, 1, 7), (2, 4, 3)) if draft else None,
    )
    harness.sync()
    decoder.warmup()
    tables = [
        pools.scheduled(np.arange(1 + n * pages, 1 + (n + 1) * pages), n)
        for n in range(len(prompts))
    ]
    names = tuple(f"m{n}" for n in range(len(prompts)))

    def decode(tag, steps, masks, sampling=(0.0, 1.0, 0, 0)):
        """``steps`` decode steps of both prompts; ``masks``: step -> the
        rows' ``[2, words]`` int32 bitmasks (None: unmasked). Returns the
        steps' first tokens, ``[steps, 2]``."""
        outs = [
            decoder.step(
                mk.StepInput(
                    [
                        k3.request(tag + n, ids, table, sampling)
                        for n, ids, table in zip(names, prompts, tables, strict=True)
                    ],
                    mask=masks(0),
                )
            ).tokens[:2, :1]
        ]
        for step in range(1, steps):
            outs.append(
                decoder.step(
                    mk.StepInput(
                        [
                            k3.request(tag + n, [0], table, sampling)
                            for n, table in zip(names, tables, strict=True)
                        ],
                        mask=masks(step),
                    )
                ).tokens[:2, :1]
            )
        decoder.forget(tuple(tag + n for n in names))
        return torch.stack(outs).cpu().numpy().reshape(-1, 2)

    steps = args.generate
    ones = np.full((2, words), -1, np.int32)
    # The plain reference. A speculating decoder would verify the requests'
    # first step (their anchors); an all-ones mask keeps it plain like the
    # masked runs', so masked and unmasked plain ops are compared alone.
    reference = decode(
        "u", steps, lambda s: ones if draft is not None and s == 0 else None
    )
    unmasked = decode("a", steps, lambda _: ones)
    same = int((unmasked == reference).sum())
    harness.log(
        f"all-ones masks: {same} of {reference.size} tokens identical to unmasked"
    )

    # Random masks: every token the op chooses must be allowed.
    rng = np.random.default_rng(0)
    random_masks = [
        rng.integers(-(2**31), 2**31, (2, words), np.int32) for _ in range(steps)
    ]
    chosen = decode("r", steps, lambda s: random_masks[s])
    violations = sum(
        not (random_masks[s][n][token // 32] >> (token % 32)) & 1
        for s in range(steps)
        for n, token in enumerate(chosen[s])
        if token >= 0
    )
    harness.log(f"random masks: {violations} disallowed tokens of {chosen.size}")

    # A mask allowing a single token forces it, greedy or sampled, wherever
    # the token lives in the (vocabulary-sharded) range.
    forced_ok = True
    for sampling, label in (
        ((0.0, 1.0, 0, 0), "greedy"),
        ((0.7, 0.9, 0, 42), "sampled"),
    ):
        for token in (17, 81920, k3_program.VOCAB - 13):
            mask = np.zeros((2, words), np.int32)
            mask[:, token // 32] = np.int32(1 << (token % 32))
            out = decode("f", 1, lambda _, mask=mask: mask, sampling)
            ok = bool((out[0] == token).all())
            forced_ok &= ok
            harness.log(
                f"forced token {token} ({label}): {out[0].tolist()} "
                f"({'ok' if ok else 'WRONG'})"
            )

    # A masked request must not move its unmasked batch-mate (row mapping).
    blocked = np.full((2, words), -1, np.int32)
    blocked[0, reference[0, 0] // 32] &= ~np.int32(1 << (reference[0, 0] % 32))
    mixed = decode("b", 1, lambda _: blocked)
    row_maps = int(mixed[0, 1]) == int(reference[0, 1])
    blocks = bool(
        int(mixed[0, 0]) != int(reference[0, 0])
        and (blocked[0, mixed[0, 0] // 32] >> (mixed[0, 0] % 32)) & 1
    )
    harness.log(
        f"mixed batch: unmasked row unchanged {row_maps}; masked row blocks "
        f"{int(reference[0, 0])} -> {int(mixed[0, 0])} ({'ok' if blocks else 'WRONG'})"
    )
    spec_ok = True
    if draft is not None:
        spec_ok = _masked_around_speculation(decoder, prompts, tables, names, reference)
    records.add(
        check="masked",
        dspark=bool(draft),
        generate=steps,
        all_ones_identical=[same, int(reference.size)],
        random_violations=int(violations),
        forced_ok=forced_ok,
        mixed_rows_ok=bool(row_maps and blocks),
        speculation_ok=spec_ok,
    )
    ok = same == reference.size and not violations and forced_ok and row_maps and blocks
    if not (ok and spec_ok):
        harness.fail("masked check")


def _masked_around_speculation(decoder, prompts, tables, names, reference) -> bool:
    """Both prompts speculating (three drafts each, as vLLM schedules them),
    then a run of masked steps (all-ones masks: the batch steps plain), then
    speculating again; and the same steps speculating throughout. The masked
    steps must not speculate, speculation must resume after them (a
    multi-token step), and the greedy streams must follow the plain
    reference (until a near-tie: the verify and plain ops reduce in
    different orders) about as far as the pure DSpark streams do."""
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program

    ones = np.full((2, k3_program.VOCAB // 32), -1, np.int32)
    plain_ref = [[int(t) for t in reference[:, n]] for n in range(2)]

    def run(tag, windows):
        streams, bounds = [[], []], []
        first, resumed, stepped_ok = True, False, True
        for window, (label, mask, steps) in enumerate(windows):
            before = dict(decoder.stats)
            for _ in range(steps):
                requests = [
                    k3.request(tag + n, ids if first else np.zeros(4, np.int32), table)
                    for n, ids, table in zip(names, prompts, tables, strict=True)
                ]
                rows = decoder.step(mk.StepInput(requests, mask=mask)).tokens[:2]
                rows = rows.cpu().numpy()
                for n in range(2):
                    tokens = [int(t) for t in rows[n] if t >= 0]
                    streams[n] += tokens
                    resumed |= window == 2 and len(tokens) > 1
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
    mixed, bounds, resumed, stepped_ok = run("s", masked_windows)
    pure, _, _, _ = run("n", pure_windows)
    vs_plain = [_agree(st, ref) for st, ref in zip(mixed, plain_ref, strict=True)]
    pure_vs_plain = [_agree(st, ref) for st, ref in zip(pure, plain_ref, strict=True)]
    vs_pure = [_agree(a, b) for a, b in zip(mixed, pure, strict=True)]
    harness.log(
        f"dspark, agreement (tokens): masked run vs plain {vs_plain}, pure DSpark "
        f"vs plain {pure_vs_plain}, masked run vs pure DSpark {vs_pure}; the "
        f"masked window spans tokens {bounds[0]} to {bounds[1]}; speculation "
        f"resumed after it: {resumed}"
    )
    return stepped_ok and resumed and all(a > 0 for a in vs_plain)


def run_spec(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.decode_ops import (
        Control,
        DecodeOps,
        decode_batch_sizes,
    )
    from vllm_torchtpu.megakernel.kimi_k3.draft import draft_class

    count, generate, rows = args.prompt_tokens, args.generate, args.single_drafts + 1
    large = sorted(int(n) for n in args.large_batches.split(",") if n)
    if args.require_verifier_parity and not large:
        harness.fail("--require-verifier-parity requires --large-batches")
    most = max([8, *large])
    all_ids, tokenizer = k3.fixed_prompt(count + 4000, args.model)
    ids = np.asarray(all_ids[:count], np.int32)

    def encode(text):
        return np.asarray(tokenizer.encode(text, disallowed_special=()), np.int32)

    prompts = {
        "A": ids,
        "B": encode("The history of the lighthouse begins"),
        "C": np.asarray(all_ids[count : count + 2000], np.int32),  # the text after A's
        "D": encode("Write a short story about a robot who learns to paint."),
        "E": encode("Explain, step by step, how a transistor amplifies a signal."),
        "F": np.asarray(all_ids[count + 2000 :], np.int32),  # the text after C's
        "G": encode("List the prime numbers below fifty and say why each is prime."),
        "H": encode("Once upon a time, in a small village by the sea,"),
    }
    # C and F retain their 2000-token fixtures even when --prompt-tokens
    # shortens A. All fixtures must fit both the pool and the draft's RoPE.
    longest_prompt = max(map(len, prompts.values()))
    pages = -(-(longest_prompt + 4 * generate + 64) // k3.logical()) + 1
    # One copy of the weights for the decoders below (max_num_seqs 8, or the
    # largest --large-batches count).
    ops = DecodeOps(
        args.presharded,
        num_slots=Control.total_slots(2 * most + 1),
        batch_sizes=decode_batch_sizes(most),
        **k3.decode_ops_options(),
    )
    pools = k3.Pools((10 + max([0, *large])) * pages + 1, pages, 1)
    draft = draft_class(args.dspark)(
        max_position=longest_prompt + 8 * generate + 64,
        checkpoint=args.dspark,
        presharded=args.presharded,
    )

    def decoder(schedule=None):
        # Every decoder over these ops takes its slot count (attention DP's
        # group shares of the KDA banks follow it).
        return k3.decoder(
            ops,
            pools,
            max_num_seqs=most,
            max_model_len=pages * k3.logical(),
            draft=draft,
            schedule=schedule,
        )

    # Single-request speculation at the requested width; plain batched steps.
    single = decoder(((1, 1, args.single_drafts), (2, 8, 0)))
    plain = decoder()
    # Select both singleton and batched widths, including their transitions.
    batched = decoder(
        ((1, 1, args.single_drafts), (2, 4, args.small_batch_drafts), (5, 8, 3))
    )
    harness.sync()
    # Under attention DP the fixtures alternate between the groups.
    tables = {
        name: pools.scheduled(
            np.arange(1 + n * pages, 1 + (n + 1) * pages, dtype=np.int32), n
        )
        for n, name in enumerate(("A", "B", "ref", "C", "D", "E", "F", "G", "H"))
    }

    def tokens_of(emitted):
        return [int(t) for t in torch.cat(emitted).cpu().reshape(-1).tolist() if t >= 0]

    def row(req, tokens, table, sampling=(0.0, 1.0, 0, 0)):
        return k3.request(req, tokens, table, sampling)

    def decode(decoder, requests, finished=()):
        return decoder.step(mk.StepInput(requests, finished)).tokens

    def plain_reference(prompt, n):
        outs = [decode(plain, [row("ref", prompt, tables["ref"])])[:1]]
        for _ in range(n - 1):
            outs.append(decode(plain, [row("ref", [0], tables["ref"])])[:1])
        plain.forget(("ref",))
        return tokens_of(outs)

    single.warmup()
    plain.warmup()
    reference = plain_reference(ids, generate)
    b_reference = plain_reference(prompts["B"], 24)
    harness.log("references done")

    def run_alone(req, sampling, n_steps, first=True):
        outs = []
        if first:
            outs.append(decode(single, [row(req, ids, tables["A"], sampling)])[:1])
        for _ in range(n_steps):
            outs.append(
                decode(
                    single,
                    [row(req, np.zeros(rows), tables["A"], sampling)],
                )[:1]
            )
        return outs

    a_tokens = tokens_of(run_alone("a", (0.0, 1.0, 0, 0), 60))
    same = _agree(a_tokens, reference)
    per_step = (len(a_tokens) - 1) / 60
    harness.log(
        f"greedy DSpark vs plain greedy: first {same} of "
        f"{min(len(a_tokens), len(reference))} agree; "
        f"{per_step:.2f} tokens per speculative step"
    )

    outs_a, outs_b = [], []
    out = decode(
        single,
        [
            row("a", [0], tables["A"]),
            row("b", prompts["B"], tables["B"]),
        ],
    )
    outs_a.append(out[:1])
    outs_b.append(out[1:2])
    for _ in range(23):
        out = decode(
            single,
            [row("a", [0], tables["A"]), row("b", [0], tables["B"])],
        )
        outs_a.append(out[:1])
        outs_b.append(out[1:2])
    b_tokens = tokens_of(outs_b)
    outs_a.append(decode(single, [row("a", np.zeros(rows), tables["A"])], ("b",))[:1])
    outs_a += run_alone("a", (0.0, 1.0, 0, 0), 20, first=False)
    a_more = tokens_of(outs_a)
    same_b = _agree(b_tokens, b_reference)
    harness.log(
        "B (batch 2 with A) vs its plain reference: "
        f"first {same_b} of {len(b_tokens)} agree"
    )
    harness.log(f"A with B, then alone: {tokenizer.decode(a_more[:120])!r}")
    harness.log(f"B: {tokenizer.decode(b_tokens)!r}")
    single.forget(("a",))

    runs = []
    for req, seed in (("s1", 7), ("s2", 7), ("s3", 8)):
        runs.append(tokens_of(run_alone(req, (0.7, 0.9, 50, seed), 12)))
        single.forget((req,))
    reproducible, varies = runs[0] == runs[1], runs[0] != runs[2]
    harness.log(
        f"sampled: same seed identical {reproducible}, other seed differs {varies}; "
        f"{tokenizer.decode(runs[0][:60])!r}"
    )

    run_alone("t", (0.0, 1.0, 0, 0), 4)
    bridge.materialize(single.control)
    before = dict(single.stats)
    started, host = time.perf_counter(), 0.0
    for _ in range(40):
        t0 = time.perf_counter()
        out = decode(single, [row("t", np.zeros(rows), tables["A"])])
        host += time.perf_counter() - t0
    bridge.materialize(out)
    step_ms = 1e3 * (time.perf_counter() - started) / 40
    harness.log(
        f"pipelined greedy DSpark step: {step_ms:.2f} ms "
        f"(host {1e3 * host / 40:.2f} ms per step); "
        f"ops over 40 steps: { {k: single.stats[k] - before[k] for k in before} }"
    )
    single.forget(("t",))

    # ---------------- 5. batched speculation
    batched.warmup()

    def verify_rows(requests):
        return batched.spec_tokens[requests] + 1

    references = {
        n: reference if n == "A" else plain_reference(prompts[n], generate)
        for n in prompts
    }

    def churn():
        """Requests joining and leaving across the single and batched paths
        right after warmup, timing every step: a draft state carries the
        buffer layouts its producer gave it, and a batched op compiles per
        layout mix of its states, so a step that compiles takes seconds.
        Returns the slowest join step and the slowest steady step (ms)."""
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
        live, worst_join, worst_steady, log = set(), 0.0, 0.0, []
        for names in order:
            leaving = tuple("c" + n for n in sorted(live - set(names)))
            times = []
            for step in range(4):
                group = []
                for n in names:
                    if n in live:
                        q = verify_rows(len(names))
                        group.append(row("c" + n, np.zeros(q), tables[n]))
                    else:
                        group.append(row("c" + n, short[n], tables[n]))
                t0 = time.perf_counter()
                out = decode(batched, group, leaving if step == 0 else ())
                bridge.materialize(out)
                times.append(1e3 * (time.perf_counter() - t0))
                live = set(names)
            worst_join = max(worst_join, times[0])
            worst_steady = max(worst_steady, *times[1:])
            log.append(f"{''.join(names)} " + "/".join(f"{t:.0f}" for t in times))
        batched.forget(tuple("c" + n for n in live))
        harness.log(
            "churn after warmup, ms per step (join/steady...): " + "; ".join(log)
        )
        return worst_join, worst_steady

    def run_group(names, tag, sampling, steps):
        """The group's streams and tokens per step: their prompts, then
        ``steps`` batched steps of ``k`` scheduled tokens each."""
        first = [row(tag + n, prompts[n], tables[n], sampling) for n in names]
        k = verify_rows(len(names))
        following = [row(tag + n, np.zeros(k), tables[n], sampling) for n in names]
        count_ = len(names)
        outs = [decode(batched, first)[:count_]]
        for _ in range(steps):
            outs.append(decode(batched, following)[:count_])
        stacked = torch.stack(outs).cpu().numpy()  # [steps + 1, rows, 8]
        streams = [
            [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(count_)
        ]
        rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(count_)]
        return streams, rates

    def report(label, names, streams, rates):
        agree = [
            _agree(st, references[n]) for n, st in zip(names, streams, strict=True)
        ]
        harness.log(
            f"batched greedy, {label}: "
            + "; ".join(
                f"{n} agrees for {a} of {min(len(st), len(references[n]))} tokens at "
                f"{r:.2f} tokens per step"
                for n, a, st, r in zip(names, agree, streams, rates, strict=True)
            )
        )
        return agree

    churn_join, churn_steady = churn()
    steps = 48
    pair_streams, pair_rates = run_group(("A", "B"), "p", (0.0, 1.0, 0, 0), steps)
    decode(batched, [], ("pB",))
    alone = [
        decode(batched, [row("pA", np.zeros(rows), tables["A"])])[:1] for _ in range(20)
    ]
    pa_more = pair_streams[0] + tokens_of(alone)
    batched.forget(("pA",))
    same_pa, same_pb = report(
        f"2 x {verify_rows(2)} rows", ("A", "B"), pair_streams, pair_rates
    )
    same_more = _agree(pa_more, reference)
    harness.log(f"A alone again after the pair: {same_more} agree")
    four = ("A", "B", "C", "D")
    four_streams, four_rates = run_group(four, "f", (0.0, 1.0, 0, 0), 32)
    batched.forget(tuple("f" + n for n in four))
    four_agree = report(f"4 x {verify_rows(4)} rows", four, four_streams, four_rates)

    def arriving(name, joining):
        """``name``'s first token arriving as a transferred request does: all
        but its last prompt token computed (its pages hold them from the
        groups above). Joining another request's speculation, it arrives with
        that step's draft slots already scheduled (vLLM pads a request
        starting to decode to the speculative width)."""
        n, req = len(prompts[name]), "t" + name
        if not joining:
            arrival = k3.request(req, prompts[name][-1:], tables[name], computed=n - 1)
            out = decode(batched, [arrival])[:1]
            batched.forget((req,))
            return tokens_of([out])
        decode(batched, [row("tA", prompts["A"], tables["A"])])
        for _ in range(4):
            decode(batched, [row("tA", np.zeros(rows), tables["A"])])
        k = verify_rows(2)
        tokens = np.concatenate((prompts[name][-1:], np.full(k - 1, -1)))
        arrival = k3.request(req, tokens, tables[name], computed=n - 1)
        arrival = dataclasses.replace(arrival, num_tokens=n)
        out = decode(batched, [row("tA", np.zeros(k), tables["A"]), arrival])[1:2]
        batched.forget(("tA", req))
        return tokens_of([out])

    arrivals = {n: (arriving(n, False), arriving(n, True)) for n in ("B", "D")}
    harness.log(
        "transferred arrival, first token alone vs joining a speculation with "
        "draft slots: " + "; ".join(f"{n} {a} vs {j}" for n, (a, j) in arrivals.items())
    )

    def plain_group(names, streams, n=112):
        """The group in plain steps: how far batch composition alone moves
        each greedy stream from its single-request reference, and how far
        the batched speculative ``streams`` follow these."""
        first = [row("g" + m, prompts[m], tables[m]) for m in names]
        following = [row("g" + m, [0], tables[m]) for m in names]
        outs = [decode(plain, first)[: len(names)]]
        for _ in range(n - 1):
            outs.append(decode(plain, following)[: len(names)])
        plain.forget(tuple("g" + m for m in names))
        stacked = torch.stack(outs).cpu().numpy()
        tokens = [[int(t) for t in stacked[:, r, 0]] for r in range(len(names))]
        agree = [_agree(t, references[m]) for t, m in zip(tokens, names, strict=True)]
        follow = [_agree(st, t) for st, t in zip(streams, tokens, strict=True)]
        harness.log(
            f"plain batch-{len(names)} greedy: "
            + "; ".join(
                f"{m} agrees for {a} of {n} (the batched stream follows it for "
                f"{f} of {min(n, len(st))})"
                for m, a, f, st in zip(names, agree, follow, streams, strict=True)
            )
        )
        return agree, follow

    plain_four, follow_four = plain_group(four, four_streams)
    eight = tuple("ABCDEFGH")
    eight_streams, eight_rates = run_group(eight, "e", (0.0, 1.0, 0, 0), 32)
    batched.forget(tuple("e" + n for n in eight))
    eight_agree = report(
        f"8 x {verify_rows(8)} rows", eight, eight_streams, eight_rates
    )
    plain_eight, follow_eight = plain_group(eight, eight_streams)
    five = tuple("ABCDE")
    five_streams, five_rates = run_group(five, "v", (0.0, 1.0, 0, 0), 16)
    batched.forget(tuple("v" + n for n in five))
    five_agree = report(
        f"5 (padded to 8) x {verify_rows(5)} rows", five, five_streams, five_rates
    )
    three = ("A", "B", "C")
    three_streams, three_rates = run_group(three, "t", (0.0, 1.0, 0, 0), 16)
    batched.forget(tuple("t" + n for n in three))
    three_agree = report(
        f"3 (padded to 4) x {verify_rows(3)} rows", three, three_streams, three_rates
    )
    sampled_runs = []
    for tag, seed in (("x", 7), ("y", 7), ("z", 8)):
        streams, _ = run_group(("A", "B"), tag, (0.7, 0.9, 50, seed), 12)
        batched.forget((f"{tag}A", f"{tag}B"))
        sampled_runs.append(streams)
    b_reproducible = sampled_runs[0] == sampled_runs[1]
    b_varies = sampled_runs[0] != sampled_runs[2]
    harness.log(
        f"batched sampled: same seed identical {b_reproducible}, other seed "
        f"differs {b_varies}; A {tokenizer.decode(sampled_runs[0][0][:40])!r}"
    )

    def pipelined(decoder, requests, query, n_steps=40):
        """ms per step of the group's first ``requests`` prompts (distinct
        tokens per row, as serving routes them; the rates above are theirs)."""
        names = eight[:requests]
        first = [row(f"q{r}", prompts[n], tables[n]) for r, n in enumerate(names)]
        decode(decoder, first)
        following = [
            row(f"q{r}", np.zeros(query), tables[n]) for r, n in enumerate(names)
        ]
        for _ in range(4):
            out = decode(decoder, following)
        bridge.materialize(out)
        started = time.perf_counter()
        for _ in range(n_steps):
            out = decode(decoder, following)
        bridge.materialize(out)
        decoder.forget(tuple(f"q{r}" for r in range(requests)))
        return 1e3 * (time.perf_counter() - started) / n_steps

    timing = {}
    for requests, rates in ((2, pair_rates), (4, four_rates), (8, eight_rates)):
        k = verify_rows(requests)
        spec_ms, plain_ms = (
            pipelined(batched, requests, k),
            pipelined(plain, requests, 1),
        )
        timing[requests] = (round(spec_ms, 3), round(plain_ms, 3))
        harness.log(
            f"pipelined batched DSpark step ({requests} x {k} rows): {spec_ms:.2f} ms; "
            f"plain batch-{requests} step {plain_ms:.2f} ms; tokens per ms at the "
            f"greedy rates above {sum(rates) / spec_ms:.3f} vs "
            f"{requests / plain_ms:.3f}"
        )
    large_results = {}
    if large:
        # Larger batches: each request a fixture prompt (its first 256 tokens:
        # prompts replay one token a step) on its own pages, against plain
        # steps of the same requests.
        # The batched decoder's schedule carries three drafts past eight.
        spec_large, plain_large = batched, plain
        letters = tuple(prompts)
        source = [letters[i % len(letters)] for i in range(most)]
        group_tables = [
            pools.scheduled(
                np.arange(1 + (9 + i) * pages, 1 + (10 + i) * pages, dtype=np.int32),
                9 + i,
            )
            for i in range(most)
        ]

        def group_streams(dec, n, query, steps, tag):
            first = [
                row(f"{tag}{i}", prompts[source[i]][:256], group_tables[i])
                for i in range(n)
            ]
            following = [
                row(f"{tag}{i}", np.zeros(query), group_tables[i]) for i in range(n)
            ]
            outs = [decode(dec, first)[:n]]
            for _ in range(steps):
                outs.append(decode(dec, following)[:n])
            dec.forget(tuple(f"{tag}{i}" for i in range(n)))
            stacked = torch.stack(outs).cpu().numpy()
            streams = [
                [int(t) for t in stacked[:, r].reshape(-1) if t >= 0] for r in range(n)
            ]
            rates = [float((stacked[1:, r] >= 0).sum(axis=1).mean()) for r in range(n)]
            return streams, rates

        def group_ms(dec, n, query, steps=40):
            decode(
                dec,
                [
                    row(f"w{i}", prompts[source[i]][:256], group_tables[i])
                    for i in range(n)
                ],
            )
            following = [
                row(f"w{i}", np.zeros(query), group_tables[i]) for i in range(n)
            ]
            for _ in range(4):
                out = decode(dec, following)
            bridge.materialize(out)
            started = time.perf_counter()
            for _ in range(steps):
                out = decode(dec, following)
            bridge.materialize(out)
            dec.forget(tuple(f"w{i}" for i in range(n)))
            return 1e3 * (time.perf_counter() - started) / steps

        for n in large:
            k = spec_large.spec_tokens[n] + 1
            spec_streams, spec_rates = group_streams(spec_large, n, k, 32, f"s{n}_")
            plain_streams, _ = group_streams(plain_large, n, 1, 112, f"p{n}_")
            follow = [
                _agree(st, pt)
                for st, pt in zip(spec_streams, plain_streams, strict=True)
            ]
            verify_follow = None
            if args.require_verifier_parity:
                verify_streams, _ = group_streams(
                    spec_large,
                    n,
                    k - 1,
                    max(map(len, spec_streams)) - 1,
                    f"v{n}_",
                )
                expected = max(map(len, spec_streams))
                if any(len(stream) != expected for stream in verify_streams):
                    harness.fail("the sequential verifier must emit one token per step")
                verify_follow = [
                    _agree(st, vt)
                    for st, vt in zip(spec_streams, verify_streams, strict=True)
                ]
                exact = all(
                    agree == len(stream)
                    for agree, stream in zip(verify_follow, spec_streams, strict=True)
                )
                harness.log(
                    f"batch-{n} same-verifier sequential reference: "
                    f"{sum(verify_follow)}/{sum(map(len, spec_streams))} "
                    f"tokens match; exact {exact}"
                )
                if not exact:
                    harness.fail(
                        f"batch-{n} speculative output differs from sequential "
                        f"steps of its verifier: {verify_follow} versus "
                        f"{list(map(len, spec_streams))} tokens"
                    )
            spec_ms, plain_ms = group_ms(spec_large, n, k), group_ms(plain_large, n, 1)
            large_results[str(n)] = dict(
                follow=follow,
                verifier_follow=verify_follow,
                generated_tokens=list(map(len, spec_streams)),
                tokens_per_step=[round(r, 2) for r in spec_rates],
                ms=[round(spec_ms, 3), round(plain_ms, 3)],
            )
            harness.log(
                f"batched greedy, {n} x {k} rows vs plain batch-{n}: the streams "
                f"follow for {sorted(follow)} tokens (of up to "
                f"{min(len(st) for st in spec_streams)}+), "
                f"{np.mean(spec_rates):.2f} tokens per request-step; step "
                f"{spec_ms:.2f} ms vs plain {plain_ms:.2f} ms; tokens per ms "
                f"{sum(spec_rates) / spec_ms:.3f} vs {n / plain_ms:.3f}"
            )
    records.add(
        case="spec",
        single_drafts=args.single_drafts,
        small_batch_drafts=args.small_batch_drafts,
        dspark_vs_plain=[same, min(len(a_tokens), len(reference))],
        tokens_per_step=per_step,
        batch2_vs_plain=[same_b, len(b_tokens)],
        seed_reproducible=reproducible,
        seed_varies=varies,
        pipelined_ms=round(step_ms, 3),
        batched_pair_agree=[same_pa, same_pb],
        batched_pair_tokens_per_step=[round(r, 2) for r in pair_rates],
        batched_then_alone=same_more,
        batched_four_agree=four_agree,
        plain_four_agree=plain_four,
        batched_four_follow_plain=follow_four,
        batched_four_tokens_per_step=[round(r, 2) for r in four_rates],
        batched_three_agree=three_agree,
        batched_eight_agree=eight_agree,
        plain_eight_agree=plain_eight,
        batched_eight_follow_plain=follow_eight,
        batched_eight_tokens_per_step=[round(r, 2) for r in eight_rates],
        batched_five_agree=five_agree,
        batched_seed_reproducible=b_reproducible,
        batched_seed_varies=b_varies,
        batched_ms={str(b): t for b, t in timing.items()},
        churn_worst_ms=[round(churn_join), round(churn_steady)],
        transferred_arrival_first_tokens=arrivals,
        large_batches=large_results,
    )
    ok = reproducible and varies and same > 0
    ok = ok and all(a > 0 for r in large_results.values() for a in r["follow"])
    ok = ok and b_reproducible and b_varies and same_pa > 0 and same_pb > 0
    ok = ok and all(a > 0 for a in four_agree + three_agree + eight_agree + five_agree)
    ok = ok and all(a[:1] == j[:1] for a, j in arrivals.values())
    if churn_steady > 1000 or churn_join > 1000:
        harness.log("a step after warmup compiled (churn above)")
        ok = False
    if not ok:
        harness.fail("spec check")


def run_handoff(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program

    generate, max_model_len = args.generate, 32768
    # The kernel rank stands in for the TP / DCP rank, so the prefiller writes
    # the pools in the mesh order the decoder binds (as the KV transfer does).
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    ids, tokenizer = k3.fixed_prompt(24000, args.model)
    lengths = {
        "A": k3.LOGICAL + 3808,
        "B": k3.LOGICAL + 1500,
        "C": 700,
        "D": k3.PAGE,
        "E": 100,
        "F": k3.LOGICAL + 400,
    }
    hit = {"B": ("A", k3.LOGICAL), "F": ("A", k3.LOGICAL)}
    prompts, at = {}, 0
    for req, n in lengths.items():
        if req in hit:
            donor, cached = hit[req]
            prompts[req] = prompts[donor][:cached] + ids[at : at + n - cached]
            at += n - cached
        else:
            prompts[req] = ids[at : at + n]
            at += n
    schedule = [
        [("A", 4096), ("C", 700)],
        *[[("A", end)] for end in range(8192, k3.LOGICAL + 1, 4096)],
        [("A", lengths["A"])],
        [("B", lengths["B"]), ("D", k3.PAGE)],
        [("E", 100)],
        [("F", lengths["F"])],
    ]
    decode_pages = max_model_len // k3.LOGICAL + 1
    pools = k3.Pools(
        1
        + 3 * decode_pages
        + sum(-(-n // k3.LOGICAL) + 3 * -(-n // k3.PAGE) for n in lengths.values())
        + 8,
        decode_pages,
        max_model_len // k3.PAGE,
    )
    logical, states = {}, {}
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
    # The decoder's KDA tables also cover the token it decodes next, as vLLM
    # allocates them: D (256 tokens) gets a second column the prefill never
    # wrote; the transferred state is in the first.
    decode_states = {
        req: [s + pools.fresh(-(-(n + 1) // k3.PAGE) - len(s)) for s in states[req]]
        for req, n in lengths.items()
    }
    reference_tables = {req: pools.fresh(decode_pages) for req in lengths}

    def row(req_id, computed, scheduled, tokens, prompt_len, mla_table, kda_tables):
        tokens = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=scheduled,
            token_ids=tokens,
            num_tokens=len(tokens),
            num_prompt_tokens=prompt_len,
            block_tables=pools.tables(mla_table, kda_tables),
        )

    from pathlib import Path

    draft = Path(args.dspark) if args.dspark else None
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=4,
            max_model_len=max_model_len,
            max_num_batched_tokens=4096,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()
    done = {req: hit[req][1] if req in hit else 0 for req in lengths}
    first_token = {}
    for step in schedule:
        rows = [
            row(
                req,
                done[req],
                total - done[req],
                prompts[req],
                len(prompts[req]),
                logical[req],
                states[req],
            )
            for req, total in step
        ]
        tokens = (
            prefiller.step(mk.StepInput(rows)).tokens[: len(rows), 0].cpu().tolist()
        )
        for (req, total), token in zip(step, tokens, strict=True):
            done[req] = total
            if total == len(prompts[req]):
                first_token[req] = int(token)
    bridge.materialize(pools.tensors[0])
    harness.log(f"prefill done; first tokens {first_token}")
    # The registered ops keep the prefill program alive: drop its weights by hand.
    prefiller.program.weights.clear()
    prefiller.program._expert_banks.clear()  # the fused MoE's resident experts
    if prefiller.ring is not None:
        prefiller.ring.weights = ()
    del prefiller
    gc.collect()
    harness.sync()

    decoder = k3_program.build(
        k3.engine_config(
            "decode",
            max_num_seqs=2,
            max_model_len=max_model_len,
            max_num_batched_tokens=512,
            num_speculative_tokens=args.single_drafts if draft else 0,
            # The check decodes one request at a time, at the requested width.
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
    )
    decoder.load_weights()
    decoder.bind_kv_caches(pools.caches, pools.groups)
    harness.sync()

    def continue_from(req_id, first_row):
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
        pd = [first_token[req]] + continue_from(
            f"pd-{req}",
            row(
                f"pd-{req}",
                n,
                1,
                prompts[req] + [first_token[req]],
                n,
                logical[req],
                decode_states[req],
            ),
        )[: generate - 1]
        ref = continue_from(
            f"ref-{req}",
            row(
                f"ref-{req}", 0, n, prompts[req], n, reference_tables[req], [[], [], []]
            ),
        )
        same = _agree(pd, ref)
        all_first_match &= pd[0] == ref[0]
        harness.log(
            f"{req} ({n} tokens{', hit ' + str(hit[req][1]) if req in hit else ''}): "
            f"first token {'matches' if pd[0] == ref[0] else 'DIFFERS'}; "
            f"first {same} of {len(ref)} tokens agree; "
            f"P/D {tokenizer.decode(pd[:24])!r}; "
            f"reference {tokenizer.decode(ref[:24])!r}"
        )
        records.add(
            case=f"handoff-{req}",
            prompt_tokens=n,
            first_token_matches=pd[0] == ref[0],
            agree=[same, len(ref)],
        )
    rings_ok = True
    if draft is not None:
        rings_ok = _handoff_rings(
            args,
            records,
            decoder,
            lengths,
            row,
            prompts,
            first_token,
            logical,
            decode_states,
        )
    if not all_first_match:
        harness.fail("a prefill first token differs from the decode-only reference")
    if not rings_ok:
        harness.fail("an imported draft ring does not hold the prompt's last positions")


def _handoff_rings(
    args, records, decoder, lengths, row, prompts, first_token, logical, states
) -> bool:
    """The DSpark half of ``check handoff`` (see the module docstring)."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3.pools import kda_blocks_at_length

    window, steps, ok = decoder.draft.config.window, args.speculate_steps, True
    for req, n in lengths.items():
        first = row(
            f"spec-{req}",
            n,
            1,
            prompts[req] + [first_token[req]],
            n,
            logical[req],
            states[req],
        )
        blocks = kda_blocks_at_length(first.block_tables, decoder._kda_groups, n)
        _, positions = decoder._importer.import_ring(
            decoder.draft.config, torch.from_numpy(blocks).to("tpu"), n
        )
        held = int((positions >= 0).sum().cpu())
        ok &= held == min(n, window)
        counts = {}
        for arm, ring in (("ring", True), ("cold", False)):
            decoder._ring = ring
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
            for _ in range(steps - 1):
                outs.append(decoder.step(mk.StepInput([following])).tokens[:1])
            decoder.step(mk.StepInput([], (req_id,)))
            emitted = torch.cat(outs).cpu().numpy()
            counts[arm] = (emitted >= 0).sum(axis=1)
        decoder._ring = True
        windows = [w for w in (16, 64) if w < steps] + [steps]
        rates = {
            arm: [round(float(counts[arm][:w].mean()), 2) for w in windows]
            for arm in counts
        }
        harness.log(
            f"{req} ({n} tokens): imported ring holds {held} positions "
            f"(expected {min(n, window)}); DSpark tokens per step ring / cold over "
            "the first "
            + "; ".join(
                f"{w} steps: {rates['ring'][i]} / {rates['cold'][i]}"
                for i, w in enumerate(windows)
            )
        )
        records.add(
            case=f"handoff-ring-{req}",
            prompt_tokens=n,
            ring_positions=held,
            tokens_per_step_ring=rates["ring"],
            tokens_per_step_cold=rates["cold"],
        )
    return ok


# Pages per ``check packing`` prompt when the window has two pages a sequence
# (BF16 pools); FP8's 512-token pages leave one page each.
_PACKING_PAGES = (1, 2, 1, 4, 1, 3, 1, 2, 1, 1, 5, 2, 1, 3, 2, 2)
_PACKING_PREFIX_PAGES = (2, 4, 8, 3)


def _packing_cases(page: int, window: int, slots: int) -> list[tuple[int, int]]:
    """``check packing``'s prompts, ``(new tokens, cached prefix tokens)``:
    ``slots`` prompts whose page-padded rows fill one ``window`` exactly, the
    last four on cached prefixes; they end mid-page, on a page or (one) after
    a single token."""
    pages = list(_PACKING_PAGES)
    if len(pages) != slots or sum(pages) != window // page:
        pages = [window // page // slots] * slots
    cases = []
    for i, n in enumerate(pages):
        trim = 0 if i % 4 == 1 else (37 * i + 11) % page
        length = 1 if i == 4 else n * page - trim
        hit = i - (slots - len(_PACKING_PREFIX_PAGES))
        cases.append((length, _PACKING_PREFIX_PAGES[hit] * page if hit >= 0 else 0))
    return cases


def run_packing(args, records: harness.Records) -> None:
    import math
    from pathlib import Path

    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import prefill as k3_prefill
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.pools import kda_blocks_at_length

    window, slots = k3_program.MAX_STEP_TOKENS, k3_prefill.SLOTS
    max_model_len = 2 * window
    cases = _packing_cases(k3.PAGE, window, slots)
    hits = [i for i, (_, prefix) in enumerate(cases) if prefix]
    ids, tokenizer = k3.fixed_prompt(sum(n + p for n, p in cases), args.model)
    prompts, at = [], 0
    for length, prefix in cases:
        prompts.append(ids[at : at + prefix + length])
        at += prefix + length
    mla_width, state_width = max_model_len // k3.LOGICAL, max_model_len // k3.PAGE
    # Per request: its MLA blocks and a state block per KDA group at the one
    # column a step writes (the last); the rest of its rows is the null block.
    per_request = -(-max_model_len // k3.LOGICAL) + 3
    pools = k3.Pools(
        1 + (len(hits) + 3 * len(cases)) * per_request, mla_width, state_width
    )
    # The kernel rank stands in for the TP / DCP rank (see run_handoff).
    rank = bridge.kernel_mesh("tp", k3_program.TP_SIZE)[1]
    draft = Path(args.dspark) if args.dspark else None
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=slots,
            max_model_len=max_model_len,
            max_num_batched_tokens=window,
        ),
        args.presharded,
        draft,
        tp_rank=rank,
    )
    prefiller.load_weights()
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    prefiller.warmup()
    harness.sync()

    def tables(end, donor=None, cached=0):
        """Fresh blocks for a request ``end`` tokens long: ``(mla, kda)``; a
        hit's state block at its last cached column is the donor's."""
        mla = pools.fresh(-(-end // k3.LOGICAL))
        kda = [[0] * state_width for _ in range(3)]
        for g in range(3):
            kda[g][(end - 1) // k3.PAGE] = pools.fresh(1)[0]
            if donor is not None:
                column = cached // k3.PAGE - 1
                kda[g][column] = donor[1][g][column]
        return mla, kda

    def row(req_id, tokens, computed, table):
        tokens = np.asarray(tokens, np.int32)
        return mk.RequestInput(
            req_id=req_id,
            computed=computed,
            scheduled=len(tokens) - computed,
            token_ids=tokens,
            num_tokens=len(tokens),
            num_prompt_tokens=len(tokens),
            block_tables=pools.tables(*table),
        )

    def step(rows, copies=()):
        return prefiller.step(mk.StepInput(rows, copies=tuple(copies))).tokens

    # The cached prefixes, prefilled once. A hit gets a private copy of the
    # donor's MLA block (vLLM's copy-on-write of the block the hit ends in)
    # and shares the donor's state block.
    donors = {i: tables(cases[i][1]) for i in hits}
    step([row(f"donor-{i}", prompts[i][: cases[i][1]], 0, donors[i]) for i in hits])

    # Three arms over fresh blocks each: every prompt alone, the step packed at
    # most four to a window (the reference: how far another packing moves the
    # results) and all of them packed into one window.
    runs = {}
    for arm, limit in (("alone", 1), ("packed4", 4), ("packed", slots)):
        blocks = [
            tables(prefix + length, donors.get(i), prefix)
            for i, (length, prefix) in enumerate(cases)
        ]
        rows = [
            row(f"{arm}-{i}", prompts[i], prefix, blocks[i])
            for i, (_, prefix) in enumerate(cases)
        ]
        copies = {i: (donors[i][0][0], blocks[i][0][0]) for i in hits}
        if arm == "alone":
            first = []
            for i, one in enumerate(rows):
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
            shapes = [f"{len(m)} sequences in a {w}-token window" for w, m in plan]
            harness.log(f"{arm}: " + "; ".join(shapes))
            if arm == "packed" and (len(plan) != 1 or len(plan[0][1]) != len(cases)):
                harness.fail(f"the {len(cases)} prompts did not pack into one window")
            prefiller.slots = limit  # the step packs at most this many a window
            tokens = step(rows, list(copies.values()))
            prefiller.slots = slots
            first = [int(t) for t in tokens[: len(rows), 0].cpu().tolist()]
        finals = [
            kda_blocks_at_length(r.block_tables, prefiller._kda_groups, r.end)
            for r in rows
        ]
        runs[arm] = (first, blocks, finals)
    bridge.materialize(pools.tensors[0])

    # Each rank compares what it holds: the new whole pages it owns (page k on
    # rank k % 32; no padding rows) and its heads of the KDA states and rings.
    # Squared sums per (request, quantity), added over the ranks.
    def device(values, dtype=torch.int32):
        return torch.from_numpy(np.asarray(values)).to(dtype).to("tpu")

    def squares(a, b):
        a, b = a.to(torch.float32), b.to(torch.float32)
        return [float(((a - b) ** 2).sum().cpu()), float((a**2).sum().cpu())]

    def owned_pages(table, prefix, length):
        cp = k3.LOGICAL // k3.PAGE
        pages = range(prefix // k3.PAGE, (prefix + length) // k3.PAGE)
        return [table[0][k // cp] for k in pages if k % cp == prefiller.cp_rank]

    states = {
        arm: prefiller._writer.read_states(device(np.concatenate(runs[arm][2])))
        for arm in runs
    }
    rings = {}
    if prefiller.ring is not None:
        for arm in runs:
            rings[arm] = [
                prefiller._writer.read_rings(
                    prefiller.ring.config, device(runs[arm][2][i]), [prefix + length]
                )[0][0]
                for i, (length, prefix) in enumerate(cases)
            ]

    def compare(arm, i, length, prefix):
        """This rank's squared sums of ``arm`` against the prompt alone:
        latents (all MLA layers, and the first alone), states and ring."""
        entry = {}
        pages = [owned_pages(runs[a][1][i], prefix, length) for a in ("alone", arm)]
        if pages[0]:
            index = [device(p, torch.int64) for p in pages]
            per_layer = [
                squares(pool[index[0]], pool[index[1]]) for pool in pools.tensors
            ]
            entry["latents"] = np.sum(per_layer, axis=0).tolist()
            entry["latents0"] = per_layer[0]
        for name, part in (("recurrent", 0), ("conv", 1)):
            entry[name] = squares(
                states["alone"][part][0, :, i], states[arm][part][0, :, i]
            )
        if rings:
            entry["ring"] = squares(rings["alone"][i], rings[arm][i])
        return entry

    arms = ("packed4", "packed")
    sums = {
        arm: [compare(arm, i, n, prefix) for i, (n, prefix) in enumerate(cases)]
        for arm in arms
    }
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, sums)
    names = ("latents0", "latents", "recurrent", "conv", "ring")

    def relative(arm, i, name):
        num = sum(g[arm][i].get(name, (0.0, 0.0))[0] for g in gathered)
        den = sum(g[arm][i].get(name, (0.0, 0.0))[1] for g in gathered)
        return math.sqrt(num / den) if den else None

    alone = runs["alone"][0]
    worst = {arm: dict.fromkeys(names, 0.0) for arm in arms}
    matches = {arm: sum(a == b for a, b in zip(alone, runs[arm][0])) for arm in arms}
    for i, (length, prefix) in enumerate(cases):
        line = []
        for arm in arms:
            errors = {n: relative(arm, i, n) for n in names}
            errors = {n: v for n, v in errors.items() if v is not None}
            for name, value in errors.items():
                worst[arm][name] = max(worst[arm][name], value)
            token = runs[arm][0][i]
            line.append(
                f"{arm} {'same' if token == alone[i] else 'DIFFERS'} "
                f"{tokenizer.decode([token])!r} "
                + " ".join(f"{k} {v:.1e}" for k, v in errors.items())
            )
            records.add(
                case=f"packing-{arm}-{i}",
                length=length,
                prefix=prefix,
                first_token_matches=token == alone[i],
                **{f"rel_l2_{k}": round(v, 6) for k, v in errors.items()},
            )
        harness.log(
            f"{i:2d}: {length} tokens"
            + (f" on a {prefix}-token prefix" if prefix else "")
            + f", alone {tokenizer.decode([alone[i]])!r}: "
            + "; ".join(line)
        )
    for arm in arms:
        harness.log(
            f"{arm}: {matches[arm]} of {len(cases)} first tokens match alone; "
            "worst relative L2 "
            + ", ".join(f"{k} {v:.1e}" for k, v in worst[arm].items() if v)
        )
    # Another packing reorders the reductions, and 93 layers (top-k routing
    # among them) amplify that; a sequence reading another's rows or state
    # would be far worse than the four-to-a-window reference.
    if matches["packed"] < matches["packed4"] - 1 or any(
        worst["packed"][n] > 2 * worst["packed4"][n] + 0.02 for n in names
    ):
        harness.fail("one-window packing departs further than the reference packing")
