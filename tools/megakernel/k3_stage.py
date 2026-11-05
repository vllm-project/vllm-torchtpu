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
"""The Kimi-K3 prefill as four pipeline stages, one per host
(``kimi_k3/stage.py``), outside vLLM.

    stage bench --prompts 8192x1,2048x4,8192x1@65536 [--repeats 3]
        [--chain DIR] [--reference DIR] [--trace DIR] [--capacity N]
    stage reference --prompts 8192x1,2048x4 --output DIR
    stage dsa-check --prompts 8192x1,2048x4,6144x1@8192 (Kimi-K3.1)
    stage prefix-check --prompts 41316x1@32768 --chain DIR (Kimi-K3.1)

``bench`` times each stage's step on every case (the slowest stage sets the
pipeline's throughput). Stages after the first continue from random
residual streams, or with ``--chain`` from the previous stage's (hand-off
files in DIR, so the four stages run one after another); with
``--reference`` the last stage compares each case's first-token logits with
those ``reference`` saved from the slice program (TP32 / SP32) on the same
token ids. Prefixes (``@PREFIX``) are synthetic cached pages, context
parallel over the stage, in an arena of at least ``--capacity`` tokens.

``dsa-check`` compares each stage's first MLA layer's Kimi-K3.1 selection
with the JAX reference of the indexer and a masked FP32 attention
(:func:`run_dsa_check`). ``prefix-check`` prefills a prompt cold and as a
prefix-cache hit on the serving stages and compares each MLA layer's
selection, scores, attention and latent on the rows both compute
(:func:`run_prefix_check`; ``@HIT`` is the hit's cached tokens).
"""

from __future__ import annotations

import json
import statistics
import time
import zlib
from pathlib import Path

import numpy as np

from tools.megakernel import bench, harness, k3, stage_dsa_micro

STAGES, CORES = 4, 8
VOCAB = 163840
STAGE_WINDOWS = tuple(range(512, 16384 + 1, 512))
DSA_CHECK_ROWS = 64  # sampled rows per core: half at most top-k candidates
DSA_CHECK_FLASH_TILE = 256  # the stage MLA layer's flash tile
_DSA_CHECK_OPS: dict = {}
DSA_DEBUG = (
    "q",
    "w",
    "keys",
    "scores",
    "select",
    "dense",
    "attended",
    "query",
    "key",
    "value",
    "arena_keys",
    "arena_values",
    "key_fp8",
    "value_fp8",
    "arena_latent",
    "window_latent",
    "up",
    "attended_masked",
)


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="stage_command", required=True)
    bench = sub.add_parser("bench", help="each stage's step time per case")
    reference = sub.add_parser(
        "reference", help="the slice program's first-token logits"
    )
    probe = sub.add_parser(
        "mla-probe",
        help="each stage's first MLA layer, row-local vs head-sharded projections, "
        "on the same random input",
    )
    pools = sub.add_parser(
        "pool-reference",
        help="the slice prefill's pools after each case (decode layout), per rank",
    )
    serve = sub.add_parser(
        "serve-check",
        help="the serving stages (KimiPrefillStage) over vLLM-shaped steps: their "
        "pools against pool-reference's",
    )
    dsa_check = sub.add_parser(
        "dsa-check",
        help="Kimi-K3.1: each stage's first MLA layer's index scores, top-k "
        "selection and selected attention against the reference",
    )
    prefix = sub.add_parser(
        "prefix-check",
        help="Kimi-K3.1: the serving stages over a prompt cold and as a prefix-cache "
        "hit, each MLA layer's selection, scores, attention and latent compared",
    )
    prefix.add_argument(
        "--chain", required=True, help="hand-off directory (the stages run in order)"
    )
    prefix.add_argument(
        "--budget", type=int, default=8192, help="tokens a step schedules, at most"
    )
    prefix.add_argument("--max-num-seqs", type=int, default=4)
    prefix.add_argument("--max-model-len", type=int, default=131072)
    prefix.add_argument(
        "--token-ids",
        default=None,
        help="a .npy of one prompt's token ids, for every case of its length "
        "(default: random ids)",
    )
    prefix.add_argument(
        "--cold-steps",
        default=None,
        help="tokens of each cold step, comma-separated (default: --budget's steps)",
    )
    prefix.add_argument(
        "--hit-steps",
        default=None,
        help="tokens of each step after the hit, comma-separated "
        "(default: --budget's steps)",
    )
    prefix.add_argument(
        "--runs",
        default="cold,cold,hit,stale",
        help="the runs in order, the first cold: cold, hit, stale (a hit whose new "
        "blocks hold copies of the first cold run's blocks), copy (a hit on clean "
        "blocks after the same copies onto unused blocks)",
    )
    prefix.add_argument(
        "--shuffle-blocks",
        action="store_true",
        help="hand out block ids in a fixed random order, not ascending",
    )
    micro = sub.add_parser(
        "dsa-micro",
        help="Kimi-K3.1 prefill DSA scoring, selection and bitmask exchange per "
        "core (no weights)",
    )
    stage_dsa_micro.add_arguments(micro)
    for p in (pools, serve):
        p.add_argument("--output" if p is pools else "--reference", required=True)
        p.add_argument(
            "--draft",
            default=None,
            help="the MLA DSpark draft checkpoint: also its context latents' pages",
        )
    serve.add_argument(
        "--budget", type=int, default=8192, help="tokens a step schedules, at most"
    )
    serve.add_argument(
        "--chain", required=True, help="hand-off directory (the stages run in order)"
    )
    for p in (pools, serve):
        p.add_argument(
            "--token-ids",
            default=None,
            help="a .npy of real text's token ids: request r of a LEN case takes ids "
            "[r LEN, (r + 1) LEN), and the log names the text's next token",
        )
    for p in (bench, reference, probe, pools, serve, micro, dsa_check, prefix):
        p.add_argument(
            "--prompts",
            required=True,
            help="LENxN[@PREFIX],...: N sequences of LEN tokens per window",
        )
        p.add_argument("--seed", type=int, default=0)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument(
        "--capacity",
        type=int,
        default=0,
        help="prefix arena tokens, at least (default: the cases' prefixes)",
    )
    bench.add_argument(
        "--serving-windows",
        action="store_true",
        help="run each case in the serving stages' window (powers of two), not "
        "the smallest multiple of 512 tokens",
    )
    bench.add_argument(
        "--serving-prefix",
        action="store_true",
        help="spread each prefix over the cores in blocks of STAGE_SPLIT pages, as "
        "the serving stages own them (prefixes: whole blocks), not page by page",
    )
    bench.add_argument(
        "--chain", default="", help="hand-off directory: run the stages in order"
    )
    bench.add_argument(
        "--reference", default="", help="compare logits with this reference"
    )
    bench.add_argument(
        "--save", default="", help="the last stage saves each case's logits here"
    )
    bench.add_argument(
        "--trace", default="", help="xprof traces of each case's last repeat"
    )
    bench.add_argument(
        "--moe-path",
        choices=("fused", "reference"),
        default="fused",
        help="the stages' routed experts: the fused expert pipeline (windows up "
        "to 8K) or the SparseCore path",
    )
    reference.add_argument("--output", required=True)
    reference.add_argument(
        "--moe-path",
        choices=("default", "reference"),
        default="default",
        help="reference: the slice program's MoE on the SparseCore path the stages run "
        "(make_moe_fused_sp_repacked) instead of the fused kernel",
    )


def _cases(prompts: str) -> list[tuple[int, int, int]]:
    cases = []
    for part in filter(None, prompts.split(",")):
        prompt, _, prefix = part.partition("@")
        length, count = (int(x) for x in prompt.split("x"))
        cases.append((length, count, int(prefix) if prefix else 0))
    return cases


def _tag(length: int, count: int, prefix: int) -> str:
    return f"p{length}x{count}" + (f"c{prefix}" if prefix else "")


def _tokens(seed: int, case: int, length: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed * 1000 + case)
    return [
        rng.integers(0, VOCAB - 1024, length).astype(np.int32) for _ in range(count)
    ]


def _step_cases(prompts: str) -> list[list[tuple[int, int]]]:
    """``bench``'s cases: comma-separated steps, each ``+``-joined pieces
    ``LENxN[@PREFIX]``, as the step's sequences ``[(length, prefix)]``."""
    steps = []
    for step in filter(None, prompts.split(",")):
        pieces = []
        for length, count, prefix in _cases(step.replace("+", ",")):
            pieces += [(length, prefix)] * count
        steps.append(pieces)
    return steps


def _step_tag(pieces: list[tuple[int, int]]) -> str:
    if len(set(pieces)) == 1:
        return _tag(pieces[0][0], len(pieces), pieces[0][1])
    return "+".join(_tag(length, 1, prefix) for length, prefix in pieces)


def _descriptor(
    window: int, slots: int, length: int, count: int, prefix: int, page: int
):
    """The packed window's int32 descriptor and its extent: ``count``
    sequences of ``length`` at page-aligned offsets, each continuing a
    ``prefix``-token cached prefix."""
    return _pieces_descriptor(window, slots, [(length, prefix)] * count, page)


def _pieces_descriptor(
    window: int, slots: int, pieces: list[tuple[int, int]], page: int
):
    """:func:`_descriptor` of sequences ``pieces`` (``(length, prefix)``
    each), packed in order at page-aligned offsets as serving packs them."""
    offsets, bases, offset, base = [], [], 0, 0
    for length, prefix in pieces:
        offsets.append(offset)
        bases.append(base)
        offset += -(-length // page) * page
        base += prefix
    unused = slots - len(pieces)
    desc = (
        (offsets + [window] * unused)
        + ([length for length, _ in pieces] + [0] * unused)
        + [0] * slots
        + (bases + [base] * unused)
        + ([prefix for _, prefix in pieces] + [0] * unused)
    )
    return np.asarray(desc, np.int32), offsets[-1] + pieces[-1][0]


def _arena(
    core: int, ranks: int, count: int, prefix: int, page: int, slots: int, capacity: int
):
    """This core's context parallel prefix operands (``CpPrefix``): its
    pages of each sequence's prefix (global page k on core ``k % ranks``),
    as block ids 1.. of a stand-in pool, and each sequence's base and token
    count in them."""
    return _pieces_arena(core, ranks, [prefix] * count, page, slots, capacity)


def _pieces_arena(
    core: int,
    ranks: int,
    prefixes: list[int],
    page: int,
    slots: int,
    capacity: int,
    chunk: int = 1,
):
    """:func:`_arena` of sequences with cached ``prefixes``; ``chunk`` pages
    a core owns in a row (serving's stages: a block of ``STAGE_SPLIT``
    pages, block k on core ``k % ranks``)."""
    own, bases, counts, cursor = [], [], [], 0
    for prefix in prefixes:
        if prefix % (page * chunk):
            harness.fail(f"prefix {prefix} is not a multiple of {page * chunk} tokens")
        mine = len(range(core, prefix // (page * chunk), ranks)) * chunk
        bases.append(cursor)
        counts.append(mine * page)
        own += list(range(1 + len(own), 1 + len(own) + mine))
        cursor += mine * page
    unused = slots - len(prefixes)
    ids = [len(own)] + own + [0] * (capacity // page - len(own))
    local = bases + [cursor] * unused + counts + [0] * unused
    return np.asarray([ids], np.int32), np.asarray([local], np.int32), len(own)


def _dsa_options() -> dict:
    """``StagePrefillProgram`` keywords for the configured model (none for Kimi-K3)."""
    return {} if k3.DSA is None else {"dsa": k3.DSA[0], "synthetic_indexer": k3.DSA[1]}


def _engine_dsa_options() -> dict:
    """``MegakernelEngineConfig`` keywords for the configured model: the
    serving stages take the pre-shard's indexer without a synthetic one."""
    if k3.DSA is None:
        return {}
    return {**_dsa_options(), "presharded_indexer": k3.DSA[1] is None}


def _index_pool(blocks: int, layers: int, page: int, generator):
    """A stand-in index-key cache for a decode-layout stand-in pool (``[blocks,
    layers, page + 32, 128]`` uint8): random FP8 key bytes (no NaN) and their
    exponent rows near unit scale."""
    import torch

    shape = (blocks, layers, page, 128)
    keys = torch.randint(0, 0x7F, shape, generator=generator, dtype=torch.uint8)
    keys |= torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8) << 7
    scales = torch.zeros((blocks, layers, 32, 128), dtype=torch.uint8)
    scales[:, :, : page // 128] = torch.randint(
        120,
        128,
        (blocks, layers, page // 128, 128),
        generator=generator,
        dtype=torch.uint8,
    )
    return torch.cat((keys, scales), dim=2).to("tpu")


def _give(path: Path, tensor) -> int:
    """Hand ``tensor`` on through ``path``; returns its bytes' CRC."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp.npy")
    values = tensor.to(torch.bfloat16).cpu().view(torch.int16).numpy()
    np.save(staging, values)
    staging.rename(path)
    return zlib.crc32(values.tobytes())


def _take(path: Path):
    import torch

    while not path.exists():
        time.sleep(0.1)
    return torch.from_numpy(np.load(path)).view(torch.bfloat16).to("tpu")


def _stage_mesh():
    import torch
    from torch_tpu._internal.pallas import get_pallas_mesh

    rank = harness.rank()
    if harness.host_world() is not None:  # this host's world is its stage
        return (
            harness.host_world(),
            rank,
            get_pallas_mesh(("k3_sp",)),
            torch.distributed.group.WORLD,
        )
    groups = [
        torch.distributed.new_group(list(range(s * CORES, (s + 1) * CORES)))
        for s in range(STAGES)
    ]
    stage, core = divmod(rank, CORES)
    return (
        stage,
        core,
        get_pallas_mesh(("k3_sp",), process_group=groups[stage]),
        groups[stage],
    )


def run(args, records: harness.Records) -> None:
    if args.stage_command == "reference":
        run_reference(args, records)
    elif args.stage_command == "mla-probe":
        run_mla_probe(args, records)
    elif args.stage_command == "pool-reference":
        run_pool_reference(args, records)
    elif args.stage_command == "serve-check":
        run_serve_check(args, records)
    elif args.stage_command == "dsa-micro":
        stage_dsa_micro.run(args, records)
    elif args.stage_command == "dsa-check":
        run_dsa_check(args, records)
    elif args.stage_command == "prefix-check":
        run_prefix_check(args, records)
    else:
        run_bench(args, records)


def run_mla_probe(args, records: harness.Records) -> None:
    """Each stage's first MLA layer on a random residual stream through the
    stage layer (``prefill_stage.make_mla_layer_stage``) and through the
    slice layer at the stage's heads (norm-gather, head-sharded projections,
    output-projection reduce-scatter), the latter's output gate and
    projection sliced from the stage's all-head weights: the relative L2
    of the two layers' residual updates and latents."""
    import torch

    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.stage import StagePrefillProgram

    cases = _cases(args.prompts)
    stage, core, mesh, group = _stage_mesh()
    page = k3.PAGE
    slots = max(count for _, count, _ in cases)
    capacity = max([count * prefix for _, count, prefix in cases] + [page * CORES])
    capacity = -(-capacity // (page * CORES)) * page * CORES
    program = StagePrefillProgram(stage, mesh, core, capacity, slots, **_dsa_options())
    program.load_weights(args.presharded)
    plan = program.plan
    layer = plan["first"] + 3  # the stage's first MLA layer
    cache = layout.cache_layout(k3.CACHE_DTYPE)
    pool_blocks = 2 + max(
        [count * (prefix // page) // CORES + count for _, count, prefix in cases]
    )
    generator = torch.Generator().manual_seed(core)
    pool = (
        (torch.randn((pool_blocks, *cache.block_shape), generator=generator) * 0.5)
        .to(getattr(torch, cache.dtype))
        .to("tpu")
    )
    mla = program.weights["mla"]
    common = program.weights["common"]
    for length, count, prefix in cases:
        tag = _tag(length, count, prefix)
        rows = -(-length // page) * page
        window = min(w for w in STAGE_WINDOWS if w >= rows * (count - 1) + length)
        cslots = program.window_slots(window)
        desc, extent = _descriptor(window, cslots, length, count, prefix, page)
        ids, local, _ = _arena(
            core, CORES, count, prefix, page, cslots, capacity // CORES + cslots * page
        )
        op = _mla_probe_op(mesh, window, cslots, capacity, layer)
        gen = torch.Generator().manual_seed(1000 * stage + core)
        local_rows = window // CORES
        prefix_rows = torch.randn((1, local_rows, 7168), generator=gen).to(
            torch.bfloat16
        )
        blocks = torch.randn((1, 8, local_rows, 7168), generator=gen).to(torch.bfloat16)
        slot = layer - plan["first"]
        new_out, new_latent, old_out, old_latent = op(
            prefix_rows.to("tpu"),
            blocks.to("tpu"),
            common["attn_res"][:, slot].contiguous(),
            common["attn_norm"][:, slot].contiguous(),
            torch.tensor([extent], dtype=torch.int32).to("tpu"),
            torch.from_numpy(desc).to("tpu"),
            pool,
            torch.from_numpy(ids).to("tpu"),
            torch.from_numpy(local).to("tpu"),
            *(mla[name][:, :1].contiguous() for name in kernels.MLA_FAMILIES),
        )
        base = prefix_rows.float()
        new_delta = new_out.float().cpu() - base
        old_delta = old_out.float().cpu() - base
        real = torch.from_numpy(
            kernels.sp_token_index(
                core, np.arange(local_rows), kernels.window_stripe(local_rows), CORES
            )
            < extent
        )
        new_delta, old_delta = new_delta[0, real], old_delta[0, real]
        update = float((new_delta - old_delta).norm() / old_delta.norm())
        latent = float(
            (new_latent.float() - old_latent.float()).norm() / old_latent.float().norm()
        )
        harness.log_host(
            f"{tag} stage {stage} core {core} MLA layer {layer}: residual update "
            f"rel-L2 {update:.5f}, latents {latent:.5f}"
        )
        records.add(case=tag, stage=stage, core=core, update=update, latent=latent)
        torch.distributed.barrier(group=group)


def run_dsa_check(args, records: harness.Records) -> None:
    """Kimi-K3.1's selection in each stage's first MLA layer
    (``prefill_stage.make_mla_layer_stage(debug=True)``) on a random
    residual stream, a random latent prefix pool and a pool of seeded index
    keys (``k3.synthetic_index_keys``), on :data:`DSA_CHECK_ROWS` of each
    core's rows (:func:`dsa_check_rows`): the rows' index scores against
    ``dsa_reference.scores`` on the layer's own queries, head weights and
    keys (relative to the row's largest); their selection against the top
    ``index_topk`` (``dsa_reference.select_topk``) of their scores over
    their valid candidates (:func:`selection_verdict`); and the attention's
    largest error against an FP32 attention over the selection
    (:func:`masked_attention`), each row's relative to its largest output,
    bounded by twice the dense flash kernel's (no selection) against the FP32
    dense attention on the same rows, rounded up to one significant digit
    (:func:`attention_bound`). Selected rows average fewer keys, so their
    outputs and their BF16 rounding are larger than the dense rows'."""
    import jax
    import torch

    from tools.megakernel import dsa_reference
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.dsa import STAGE_DSA_FAMILIES
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.stage import StagePrefillProgram

    if k3.DSA is None:
        harness.fail("stage dsa-check: Kimi-K3.1 only (--dsa-config)")
    topk = k3.DSA[0].index_topk
    cases = _cases(args.prompts)
    stage, core, mesh, group = _stage_mesh()
    page = k3.PAGE
    slots = max(count for _, count, _ in cases)
    capacity = max([count * prefix for _, count, prefix in cases] + [page * CORES])
    capacity = -(-capacity // (page * CORES)) * page * CORES
    program = StagePrefillProgram(stage, mesh, core, capacity, slots, **_dsa_options())
    program.load_weights(args.presharded)
    plan = program.plan
    layer = plan["first"] + 3  # the stage's first MLA layer
    cache = layout.cache_layout(k3.CACHE_DTYPE)
    pool_blocks = 2 + max(
        [count * (prefix // page) // CORES + count for _, count, prefix in cases]
    )
    generator = torch.Generator().manual_seed(core)
    pool = (
        (torch.randn((pool_blocks, *cache.block_shape), generator=generator) * 0.5)
        .to(getattr(torch, cache.dtype))
        .to("tpu")
    )
    index = torch.from_numpy(
        k3.synthetic_index_keys(pool_blocks, page, harness.rank(), 1, args.seed)
    ).to("tpu")
    families = {**program.weights["mla"], **program.weights["dsa"]}
    names = kernels.MLA_FAMILIES + STAGE_DSA_FAMILIES
    slot = layer - plan["first"]
    common = program.weights["common"]
    cpu = jax.devices("cpu")[0]
    for length, count, prefix in cases:
        tag = _tag(length, count, prefix)
        rows = -(-length // page) * page
        window = min(w for w in STAGE_WINDOWS if w >= rows * (count - 1) + length)
        cslots = program.window_slots(window)
        desc, extent = _descriptor(window, cslots, length, count, prefix, page)
        ids, local, _ = _arena(
            core, CORES, count, prefix, page, cslots, capacity // CORES + cslots * page
        )
        op, dense_op = _dsa_check_ops(mesh, window, cslots, capacity, layer, names)
        gen = torch.Generator().manual_seed(1000 * stage + core)
        local_rows = window // CORES
        prefix_rows = torch.randn((1, local_rows, 7168), generator=gen).to(
            torch.bfloat16
        )
        blocks = torch.randn((1, 8, local_rows, 7168), generator=gen).to(torch.bfloat16)
        length_t = torch.tensor([extent], dtype=torch.int32).to("tpu")
        desc_t = torch.from_numpy(desc).to("tpu")
        outputs = op(
            prefix_rows.to("tpu"),
            blocks.to("tpu"),
            common["attn_res"][:, slot].contiguous(),
            common["attn_norm"][:, slot].contiguous(),
            length_t,
            desc_t,
            pool,
            torch.from_numpy(ids).to("tpu"),
            torch.from_numpy(local).to("tpu"),
            index,
            torch.zeros((1,), dtype=torch.int32).to("tpu"),
            *(families[name][:, :1].contiguous() for name in names),
        )
        debug = dict(zip(DSA_DEBUG, outputs[4:], strict=True))
        dense_attended = dense_op(
            *(debug[name] for name in ("query", "key", "value")),
            length_t,
            desc_t,
            debug["arena_keys"],
            debug["arena_values"],
        )[0]
        positions = kernels.sp_token_index(
            core, np.arange(local_rows), kernels.window_stripe(local_rows), CORES
        )
        valid = [
            valid_columns(desc, extent, cslots, capacity, int(p)) for p in positions
        ]
        picks = dsa_check_rows([v.size for v in valid], topk)
        few = int(sum(valid[r].size <= topk for r in picks))

        host = {name: _on_host(debug[name][0]) for name in DSA_DEBUG}
        q, w, scores = (host[name][picks] for name in ("q", "w", "scores"))
        keys = host["keys"]
        with jax.default_device(cpu):
            expected = np.asarray(dsa_reference.scores(q, w, keys))
        rel = max(
            float(
                np.abs(scores[i, valid[r]] - expected[i, valid[r]]).max()
                / max(np.abs(expected[i, valid[r]]).max(), 1e-30)
            )
            for i, r in enumerate(picks)
        )
        window_rows = positions[picks]
        if int(host["dense"][0]):
            selected = [valid[r] for r in picks]
        else:
            words = host["select"][window_rows].view(np.uint32)
            bits = (words[..., None] >> np.arange(32, dtype=np.uint32)) & 1
            selected = [np.flatnonzero(b.reshape(-1)) for b in bits]
        compact = np.full((len(picks), max(valid[r].size for r in picks)), -np.inf)
        for i, r in enumerate(picks):
            compact[i, : valid[r].size] = scores[i, valid[r]]
        lengths = np.asarray([valid[r].size for r in picks], np.int32)
        with jax.default_device(cpu):
            chosen = np.asarray(
                dsa_reference.select_topk(
                    compact.astype(np.float32), lengths, min(topk, compact.shape[1])
                )
            )
        reference = [valid[r][c[c >= 0]] for r, c in zip(picks, chosen, strict=True)]
        verdicts = [
            selection_verdict(sel, valid[r], ref, scores[i], topk)
            for i, (r, sel, ref) in enumerate(
                zip(picks, selected, reference, strict=True)
            )
        ]
        query = host["query"][:, window_rows]
        arena_keys, arena_values = host["arena_keys"][0], host["arena_values"][0]
        attended = host["attended"][:, window_rows]
        dense_got = _on_host(dense_attended[0])[:, window_rows]
        masked_got = host["attended_masked"][
            :, window_rows
        ]  # the default path (masked flash kernel)
        errors, largest = [0.0, 0.0], [0.0, 0.0]  # selected, dense
        relative, emulated = [0.0, 0.0], [0.0, 0.0]
        for i, r in enumerate(picks):
            for which, (columns, got) in enumerate(
                ((selected[i], attended), (valid[r], dense_got))
            ):
                operands = (
                    query[:, i],
                    columns,
                    capacity,
                    arena_keys,
                    arena_values,
                    host["key"],
                    host["value"],
                )
                want = masked_attention(*operands)
                error = float(np.abs(got[:, i] - want).max())
                error = error if np.isfinite(error) else float("inf")
                errors[which] = max(errors[which], error)
                scale = float(np.abs(want).max())
                largest[which] = max(largest[which], scale)
                relative[which] = max(relative[which], error / max(scale, 1e-30))
                kernel_like = masked_attention(*operands, bf16_probabilities=True)
                emulated[which] = max(
                    emulated[which], float(np.abs(got[:, i] - kernel_like).max())
                )
        # The sparse attention reads window keys as the pool stores them (FP8):
        # its error against that reference.
        fp8_window = 0.0
        for i, r in enumerate(picks):
            want = masked_attention(
                query[:, i],
                selected[i],
                capacity,
                arena_keys,
                arena_values,
                host["key_fp8"],
                host["value_fp8"],
            )
            error = float(np.abs(attended[:, i] - want).max())
            error = error if np.isfinite(error) else float("inf")
            fp8_window = max(fp8_window, error / max(float(np.abs(want).max()), 1e-30))
        # References from the latents in f64 (no rounded keys or values): the
        # sparse kernel's with the window latents FP8-rounded (as it reads them),
        # the dense kernel's with them as BF16.
        import ml_dtypes

        latents = (
            host["arena_latent"].astype(np.float64),
            host["window_latent"].astype(np.float64),
        )
        rounded = (
            host["window_latent"].astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
        )
        up64 = host["up"].astype(np.float64)
        exact = [
            0.0,
            0.0,
            0.0,
        ]  # sparse (this mode's attention), dense, masked (the default path)
        masked = 0.0  # the default path against the FP32 reference
        for i, r in enumerate(picks):
            want = masked_attention(
                query[:, i],
                selected[i],
                capacity,
                arena_keys,
                arena_values,
                host["key"],
                host["value"],
            )
            error = float(np.abs(masked_got[:, i] - want).max())
            masked = max(
                masked,
                (error if np.isfinite(error) else float("inf"))
                / max(float(np.abs(want).max()), 1e-30),
            )
            for which, (columns, got, window) in enumerate(
                (
                    (selected[i], attended, rounded),
                    (valid[r], dense_got, latents[1]),
                    (selected[i], masked_got, latents[1]),
                )
            ):
                want = exact_attention(
                    query[:, i], columns, capacity, latents[0], window, up64
                )
                error = float(np.abs(got[:, i] - want).max())
                exact[which] = max(
                    exact[which],
                    (error if np.isfinite(error) else float("inf"))
                    / max(float(np.abs(want).max()), 1e-30),
                )
        sparse_error, dense_error = errors
        bound = attention_bound(relative[1])
        from vllm_torchtpu.megakernel.kimi_k3.config import active

        sparse = active().mla_sparse_prefill != "off"
        equal = verdicts.count("equal")
        tied = verdicts.count("tie")
        harness.log_host(
            f"{tag} stage {stage} core {core} MLA layer {layer}: {few}/{len(picks)} "
            f"rows of <= {topk} candidates; scores max rel {rel:.2e}; selection "
            f"{equal}/{len(picks)} rows equal, {tied} tie-only; attention error over "
            f"the row's largest output {relative[0]:.2e} (exact f64 from latents: "
            f"sparse {exact[0]:.2e}, dense {exact[1]:.2e}, masked {exact[2]:.2e}; "
            "masked flash vs FP32 "
            f"{masked:.2e}; FP8-window reference "
            f"{fp8_window:.2e}; dense kernel vs FP32 "
            f"{relative[1]:.2e}, bound {bound:.0e}); max abs {sparse_error:.2e} "
            f"selected, {dense_error:.2e} dense; largest FP32 output "
            f"{largest[0]:.2e} selected, {largest[1]:.2e} dense; vs a BF16-"
            f"probability reference {emulated[0]:.2e} selected, {emulated[1]:.2e} "
            "dense"
        )
        result = dict(
            case=tag,
            stage=stage,
            core=core,
            layer=layer,
            rows=len(picks),
            dense_rows=few,
            scores_rel=rel,
            equal=equal,
            tie_only=tied,
            attention=sparse_error,
            dense_attention=dense_error,
            bound=bound,
            largest=largest,
            relative=relative,
            bf16_probability_error=emulated,
            fp8_window=fp8_window,
            exact=exact,
            masked=masked,
            masked_ok=bool(masked <= bound),
            ok=bool(
                rel <= 1e-5
                and equal + tied == len(picks)
                and (fp8_window if sparse else relative[0]) <= bound
            ),
        )
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, result)
        for item in gathered:
            records.add(**item)
        harness.log(
            f"dsa-check {tag}: {sum(item['ok'] for item in gathered)}/{len(gathered)} "
            "cores within the scores, selection and attention bounds; the default path "
            "(masked flash) within the attention bound on "
            f"{sum(item['masked_ok'] for item in gathered)}"
        )
        torch.distributed.barrier(group=group)


def _on_host(tensor) -> np.ndarray:
    """A device tensor on the host (floating point as FP32)."""
    return (tensor.float() if tensor.is_floating_point() else tensor).cpu().numpy()


def valid_columns(desc, extent: int, count: int, capacity: int, row: int):
    """The ``[arena | window]`` columns window row ``row`` admits under the
    ``[5 * count]`` descriptor ``desc`` (``dsa_prefill.candidate_valid``'s
    rule): its sequence's cached prefix, then its window rows up to itself."""
    for s in range(count):
        offset, end = int(desc[s]), int(desc[s] + desc[count + s])
        base, cached = int(desc[3 * count + s]), int(desc[4 * count + s])
        if offset <= row < min(end, extent):
            return np.concatenate(
                (np.arange(base, base + cached), capacity + np.arange(offset, row + 1))
            )
    return np.zeros(0, np.int64)


def dsa_check_rows(candidates, topk: int, rows: int = DSA_CHECK_ROWS):
    """Up to ``rows`` local rows with candidates, spread evenly: half of
    them (or all there are) of at most ``topk`` candidates, the rest above."""
    counts = np.asarray(candidates)
    groups = [
        np.flatnonzero((counts > 0) & (counts <= topk)),
        np.flatnonzero(counts > topk),
    ]
    want = [min(rows // 2, groups[0].size)]
    want.append(min(rows - want[0], groups[1].size))
    want[0] = min(rows - want[1], groups[0].size)
    picked = [
        g[np.linspace(0, g.size - 1, n).astype(int)] if n else g[:0]
        for g, n in zip(groups, want, strict=True)
    ]
    return np.unique(np.concatenate(picked))


def selection_verdict(selected, valid, reference, scores, topk: int) -> str:
    """``equal`` when the row's ``selected`` columns are its ``reference``
    top ``topk`` (all its ``valid`` columns when there are at most
    ``topk``); ``tie`` when they differ only among ties (``topk`` valid
    columns, every one scoring at least the reference's lowest);
    ``differs`` otherwise."""
    selected = np.sort(selected)
    if valid.size <= topk:
        return "equal" if np.array_equal(selected, np.sort(valid)) else "differs"
    if np.array_equal(selected, np.sort(reference)):
        return "equal"
    if (
        selected.size == topk
        and np.isin(selected, valid).all()
        and scores[selected].min() >= scores[reference].min()
    ):
        return "tie"
    return "differs"


def masked_attention(
    query,
    columns,
    capacity,
    arena_keys,
    arena_values,
    keys,
    values,
    bf16_probabilities=False,
):
    """One window row's FP32 base-2 softmax attention at every head
    (``[H, 128]``; queries pre-scaled by ``SCALE_LOG2``) over its
    ``[arena | window]`` ``columns``: arena keys ``[H, C, 256]`` and values
    ``[H, 128, C]``, window keys ``[H, T, 256]`` and values ``[H, T, 128]``.
    ``bf16_probabilities``: the probabilities rounded to BF16 before both the
    value product and the denominator, as the flash kernel takes them."""
    held, new = columns[columns < capacity], columns[columns >= capacity] - capacity
    k = np.concatenate((arena_keys[:, held], keys[:, new]), axis=1).astype(np.float32)
    v = np.concatenate(
        (np.swapaxes(arena_values[:, :, held], 1, 2), values[:, new]), axis=1
    ).astype(np.float32)
    logits = np.einsum("hd,hcd->hc", query.astype(np.float32), k)
    weights = np.exp2(logits - logits.max(axis=1, keepdims=True))
    if bf16_probabilities:
        import ml_dtypes

        weights = weights.astype(ml_dtypes.bfloat16).astype(np.float32)
    return np.einsum("hc,hcd->hd", weights, v) / weights.sum(axis=1, keepdims=True)


def exact_attention(query, columns, capacity, arena_latent, window_latent, up):
    """One window row's base-2 softmax attention at every head (``[H,
    128]``) in f64 from the latents of its ``columns`` (``[arena | window]``)
    and the heads' up-projection ``up`` (``[512, H * 256]``: per head W_UK
    128 columns, then W_UV 128), with no rounded keys or values."""
    held, new = columns[columns < capacity], columns[columns >= capacity] - capacity
    latent = np.concatenate((arena_latent[held], window_latent[new]), axis=0)
    heads = up.shape[1] // 256
    w = up.reshape(512, heads, 256)
    q = query.astype(np.float64)
    absorbed = np.einsum("dhk,hk->dh", w[:, :, :128], q[:, :128])
    logits = latent[:, :512] @ absorbed + latent[:, 512:576] @ q[:, 128:192].T  # [C, H]
    p = np.exp2(logits - logits.max(axis=0, keepdims=True))
    weighted = p.T @ latent[:, :512]  # [H, 512]
    return np.einsum("hd,dhv->hv", weighted, w[:, :, 128:]) / p.sum(axis=0)[:, None]


def attention_bound(dense_error: float) -> float:
    """Twice ``dense_error`` (the dense kernel's), rounded up to one
    significant digit."""
    bound = 2 * dense_error
    if bound <= 0:
        return 0.0
    step = 10.0 ** np.floor(np.log10(bound))
    return float(np.ceil(bound / step - 1e-9) * step)


def _dsa_check_ops(mesh, window: int, slots: int, capacity: int, layer: int, names):
    """``run_dsa_check``'s ops: the stage MLA layer with its debug outputs
    (flattened in :data:`DSA_DEBUG` order after the layer's four), and the
    flash kernel without a selection on its inputs."""
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_stage

    stage_layer = prefill_stage.make_mla_layer_stage(
        mesh,
        seq_len=window,
        segments=slots,
        prefix_capacity=capacity,
        dsa=k3.DSA[0],
        rope_probe=k3.indexer_mode() == "rope-probe",
        flash_tile=DSA_CHECK_FLASH_TILE,
        debug=True,
    )

    def checked(
        prefix_rows,
        blocks,
        folded,
        norm,
        length_,
        segments,
        pool_,
        ids_,
        local_,
        index_cache,
        index_slot,
        *families,
    ):
        families = dict(zip(names, families, strict=True))
        arena = kernels.CpPrefix((pool_,), ids_, local_)
        slot = jnp.zeros((1,), jnp.int32)
        active = jnp.full((1,), layer // 12 + 1, jnp.int32)
        *outputs, debug = stage_layer(
            prefix_rows,
            blocks,
            folded,
            norm,
            active,
            families,
            segments,
            length_,
            slot,
            arena,
            (index_cache, index_slot),
        )
        return (*outputs, *(debug[name] for name in DSA_DEBUG))

    def dense(query, key, value, length_, segments, arena_keys, arena_values):
        attended = kernels.mla_flash_prefill_packed(
            query[0],
            key[0],
            value[0],
            length_,
            segments,
            count=slots,
            tile=DSA_CHECK_FLASH_TILE,
            arena=(arena_keys[0], arena_values[0]),
        )
        return (attended[None],)

    sharded, replicated = P("k3_sp"), P()
    suffix = f"w{window}_c{slots}_a{capacity}"
    if suffix in _DSA_CHECK_OPS:
        return _DSA_CHECK_OPS[suffix]
    _DSA_CHECK_OPS[suffix] = (
        bridge.register_global_op(
            f"pallas::k3_stage_dsa_check_{suffix}",
            mesh,
            checked,
            (sharded,) * 4
            + (replicated,) * 2
            + (sharded,) * 4
            + (replicated,)
            + (sharded,) * len(names),
            (sharded,) * (4 + len(DSA_DEBUG)),
        ),
        bridge.register_op(
            f"pallas::k3_stage_dsa_dense_{suffix}",
            mesh,
            dense,
            (sharded,) * 3 + (replicated,) * 2 + (sharded,) * 2,
            (sharded,),
        ),
    )
    return _DSA_CHECK_OPS[suffix]


# prefix-check's run kinds: name, a prefix-cache hit, its copies.
PREFIX_KINDS = {
    "cold": ("cold", False, None),
    "hit": ("hit", True, None),
    "stale": ("hit-stale", True, "stale"),
    "copy": ("hit-copy", True, "copy"),
}


def prefix_runs(spec: str) -> list[tuple[str, bool, str | None]]:
    """``--runs``: each run's name (a repeat: ``cold-again``, then a number),
    whether it is a hit, and what its copies do."""
    runs, seen = [], {}
    for kind in spec.split(","):
        name, warm, copies = PREFIX_KINDS[kind]
        seen[kind] = seen.get(kind, 0) + 1
        if seen[kind] > 1:
            again = kind == "cold" and seen[kind] == 2
            name = "cold-again" if again else f"{name}-{seen[kind]}"
        runs.append((name, warm, copies))
    if runs[0][1]:
        raise ValueError("the first run must be cold")
    return runs


PROBE_FIELDS = (
    "output",
    "latent",
    "key_bytes",
    "exponent",
    "scores",
    "select",
    "dense",
    "attended",
    "unions",
    "chunks",
)
# Query tiles (rows) over which the probe counts the union of the rows' selections.
UNION_TILES = (64, 128, 256, 512)
# Query tiles (rows) and aligned key chunks (tokens) of selection_chunks.
CHUNK_ROWS = (1, 8, 64, 256)
CHUNK_TOKENS = (1, 4, 8, 32, 128, 512)


def selection_chunks(select, capacity: int):
    """``[sum_Q 2 * len(CHUNK_TOKENS) * T / Q]`` int32 from a window's
    selection words (``[T, W]`` int32): per query tile size ``Q`` of
    :data:`CHUNK_ROWS`, over the arena columns then over the window
    columns, per chunk size ``g`` of :data:`CHUNK_TOKENS`, each tile's count
    of aligned ``g``-column chunks that any of its rows selects (``g`` 1:
    the union's size)."""
    import jax
    import jax.numpy as jnp

    words = jax.lax.bitcast_convert_type(select, jnp.uint32)
    rows, width = words.shape
    arena = capacity // 32
    out = []
    for tile in CHUNK_ROWS:
        union = jax.lax.reduce(
            words.reshape(rows // tile, tile, width),
            jnp.uint32(0),
            jax.lax.bitwise_or,
            (1,),
        )
        for part in (union[:, :arena], union[:, arena:]):
            for g in CHUNK_TOKENS:
                if g == 1:
                    count = jax.lax.population_count(part).astype(jnp.int32)
                elif g < 32:
                    mask = jnp.uint32((1 << g) - 1)
                    count = sum(
                        (((part >> jnp.uint32(g * i)) & mask) != 0).astype(jnp.int32)
                        for i in range(32 // g)
                    )
                else:
                    count = (part != 0).reshape(part.shape[0], -1, g // 32).any(axis=2)
                out.append(jnp.sum(count, axis=1, dtype=jnp.int32))
    return jnp.concatenate(out).astype(jnp.int32)


def chunk_summary(chunks, rows: int, length: int) -> dict:
    """:func:`selection_chunks`' output over the tiles wholly below
    ``length``: per ``Q{tile}_g{g}``, the tiles' chunk counts over every
    column (``all``: mean, p50, p90, max) and the arena's and window's
    means."""
    stats, at = {}, 0
    for tile in CHUNK_ROWS:
        tiles, whole = rows // tile, length // tile
        parts = []
        for _ in range(2):
            parts.append(
                [
                    chunks[at + i * tiles : at + (i + 1) * tiles][:whole]
                    for i in range(len(CHUNK_TOKENS))
                ]
            )
            at += len(CHUNK_TOKENS) * tiles
        if not whole:
            continue
        for i, g in enumerate(CHUNK_TOKENS):
            arena, window = parts[0][i], parts[1][i]
            total = arena + window
            stats[f"Q{tile}_g{g}"] = [
                round(float(total.mean()), 1),
                float(np.percentile(total, 50)),
                float(np.percentile(total, 90)),
                int(total.max()),
                round(float(arena.mean()), 1),
                round(float(window.mean()), 1),
            ]
    return stats


def selection_unions(select, capacity: int):
    """``[sum_Q 2 T / Q + T]`` int32 from a window's selection words
    (``[T, (capacity + T) / 32]`` int32): per query tile size ``Q`` of
    :data:`UNION_TILES`, the size of the union of each tile's selections
    over the arena columns, then over the window columns; then each row's
    selected count."""
    import jax
    import jax.numpy as jnp

    rows, words = select.shape
    arena = capacity // 32
    out = []
    for tile in UNION_TILES:
        union = jax.lax.reduce(
            select.reshape(rows // tile, tile, words),
            jnp.int32(0),
            jax.lax.bitwise_or,
            (1,),
        )
        bits = jax.lax.population_count(union)
        out += [bits[:, :arena].sum(axis=1), bits[:, arena:].sum(axis=1)]
    out.append(jax.lax.population_count(select).sum(axis=1))
    return jnp.concatenate(out).astype(jnp.int32)


def union_summary(unions, rows: int, length: int, start: int) -> str:
    """:func:`selection_unions`' output as text: per tile size, the mean and
    largest union over the tiles wholly below ``length`` (arena, window and
    their sum), and the rows' mean selected count."""
    parts, at = [], 0
    for tile in UNION_TILES:
        tiles = rows // tile
        arena, window = unions[at : at + tiles], unions[at + tiles : at + 2 * tiles]
        at += 2 * tiles
        whole = length // tile
        if not whole:
            continue
        total = arena[:whole] + window[:whole]
        parts.append(
            f"Q{tile} arena {arena[:whole].mean():.0f}/{arena[:whole].max()} "
            f"window {window[:whole].mean():.0f}/{window[:whole].max()} "
            f"all {total.mean():.0f}/{total.max()}"
        )
    selected = unions[at : at + length]
    return (
        f"rows {length} at {start}: selected per row {selected.mean():.0f}; "
        + "; ".join(parts)
    )


def selected_tokens(words, start: int, capacity: int) -> np.ndarray:
    """The global tokens a window row's selection ``words`` (int32 view of
    the packed ``[arena | window]`` bits) keeps, for a window of one
    sequence continuing ``start`` cached tokens: arena column ``c`` is
    token ``c``, window column ``capacity + w`` token ``start + w``."""
    words = np.asarray(words).view(np.uint32)
    bits = (words[:, None] >> np.arange(32, dtype=np.uint32)) & 1
    columns = np.flatnonzero(bits.reshape(-1))
    window = columns[columns >= capacity] - capacity
    return np.concatenate((columns[columns < capacity], start + window))


def candidate_scores(scores, start: int, row: int, capacity: int) -> np.ndarray:
    """Window row ``row``'s scores over its valid candidates by global
    token (as :func:`selected_tokens`): the cached ``[0, start)``, then the
    window up to itself."""
    return np.concatenate((scores[:start], scores[capacity : capacity + row + 1]))


def ulps(a, b) -> int:
    """The largest distance between two FP32 arrays in units of the last
    place (0: bitwise equal)."""

    def ordered(x):
        bits = np.asarray(x, np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(1 << 31) - bits, bits)

    return int(np.abs(ordered(a) - ordered(b)).max()) if np.size(a) else 0


def _prefix_probe_op(mesh, window: int, slots: int, capacity: int, names):
    """``prefix-check``'s op: a stage MLA layer with its debug outputs
    (``make_mla_layer_stage(debug=True)``) on the stream entering it, reduced
    to the window row ``target`` (``[row, rank, local row]``): the layer's
    output row, latent, key bytes and exponent, the row's scores over every
    candidate, its selection words, the window's dense flag and every head's
    attended row (``[ranks, heads, 128]``), all replicated."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_stage
    from vllm_torchtpu.megakernel.kimi_k3.stage import DECODE_PAGE_ROWS

    key = f"prefix_w{window}_c{slots}_a{capacity}"
    if key in _DSA_CHECK_OPS:
        return _DSA_CHECK_OPS[key]
    layer_op = prefill_stage.make_mla_layer_stage(
        mesh,
        seq_len=window,
        segments=slots,
        prefix_capacity=capacity,
        dsa=k3.DSA[0],
        rope_probe=k3.indexer_mode() == "rope-probe",
        debug=True,
    )

    def probe(stream, blocks, folded, norm, active, length, segments, pool, ids,
              local, index_cache, index_slot, slot, target, *families):  # fmt: skip
        pages = pool.reshape(-1, DECODE_PAGE_ROWS, *pool.shape[2:])
        output, latent, key_bytes, exponents, debug = layer_op(
            stream,
            blocks,
            folded,
            norm,
            active,
            dict(zip(names, families, strict=True)),
            segments,
            length,
            slot,
            kernels.CpPrefix((pages,), ids, local),
            (index_cache, index_slot),
        )
        return jax.shard_map(
            pick,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 8 + (P(),),
            out_specs=(P(),) * len(PROBE_FIELDS),
            check_vma=False,
        )(output, latent, key_bytes, exponents, debug["scores"], debug["select"],
          debug["dense"], debug["attended"], target)  # fmt: skip

    def pick(output, latent, key_bytes, exponents, scores, select, dense, attended,
             target):  # fmt: skip
        # Inside the map: the owner's rows by an all-gather, the window's
        # (the same on every rank) from this rank's copy.
        row, rank, mine = target[0], target[1], target[2]

        def owner(local):
            return jax.lax.all_gather(local, "k3_sp")[rank]

        return (
            owner(output[0, mine]),
            latent[0, row],
            key_bytes[0, row],
            exponents[0, row][None],
            owner(scores[0, mine]),
            select[0, row],
            dense[0],
            jax.lax.all_gather(attended[0, :, row].astype(jnp.float32), "k3_sp"),
            selection_unions(select[0], capacity),
            selection_chunks(select[0], capacity),
        )

    sharded, replicated = P("k3_sp"), P()
    _DSA_CHECK_OPS[key] = bridge.register_global_op(
        f"pallas::k3_stage_prefix_check_{key}",
        mesh,
        probe,
        (sharded,) * 4
        + (replicated,) * 3
        + (sharded,) * 4
        + (replicated,) * 3
        + (sharded,) * len(names),
        (replicated,) * len(PROBE_FIELDS),
    )
    return _DSA_CHECK_OPS[key]


def _probe_windows(program, mesh, saved: list) -> None:
    """Wrap ``program.run`` (a ``StagePrefillProgram``): each group op also
    returns the stream entering its MLA layer, and after every window each
    stage MLA layer runs again with its debug outputs on the stream entering
    it (the final op's: the stream entering that op) for the window's last
    row; appends ``(start, row, layer, values)`` to ``saved``: ``values`` by
    :data:`PROBE_FIELDS`, then the latent and key bytes the stage op itself
    kept for the row (``kept_latent``, ``kept_key``) and, after layer 92,
    the window's first sequence's ``logits``."""
    import torch

    from vllm_torchtpu.megakernel.kimi_k3.dsa import STAGE_DSA_FAMILIES
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels

    plan, run = program.plan, program.run
    program.tap_offset = lambda first: 3  # the stream entering each group's MLA
    names = kernels.MLA_FAMILIES + STAGE_DSA_FAMILIES
    slots = plan["slots"]

    def device(values):
        return torch.tensor(values, dtype=torch.int32).to("tpu")

    def probed(window, page_tokens, length, segments, pools, ids, local, **kwargs):
        out = run(window, page_tokens, length, segments, pools, ids, local, **kwargs)
        prefix, blocks, states, logits, _, taps = out
        count = program.window_slots(window)
        desc = segments.cpu().numpy()
        if desc[count] != int(length.cpu().numpy()[0]) or desc[0] != 0:
            raise RuntimeError("stage prefix-check: one sequence per window")
        row, start = int(desc[count]) - 1, int(desc[4 * count])
        rows = window // CORES
        stripe = kernels.window_stripe(rows)
        for rank in range(CORES):
            owned = kernels.sp_token_index(rank, np.arange(rows), stripe, CORES)
            if row in owned:
                target = device([row, rank, int(np.flatnonzero(owned == row)[0])])
        op = _prefix_probe_op(mesh, window, count, program.capacity, names)
        families = {**program.weights["mla"], **program.weights["dsa"]}
        common = program.weights["common"]
        layers = [
            (4 * g + 3, g - slots["mla"][0], taps[4 * g + 3]) for g in plan["groups"]
        ]
        if plan["final"]:
            layers.append((92, 23 - slots["mla"][0], prefix))
        for j, (layer, slot, stream) in enumerate(layers):
            at = layer - slots["layer"][0]
            values = op(
                stream,
                blocks,
                common["attn_res"][:, at].contiguous(),
                common["attn_norm"][:, at].contiguous(),
                device([(layer - 1) // 12 + 1]),
                length,
                segments,
                pools[j],
                ids,
                local,
                kwargs["index"],
                program._slot(kwargs["index_slots"][j]),
                device([slot]),
                target,
                *(families[name] for name in names),
            )
            values = dict(zip(PROBE_FIELDS, map(_on_host, values), strict=True))
            if program.rank == 0:
                harness.log_host(
                    f"selection union stage {program.stage} layer {layer}: "
                    + union_summary(values["unions"], window, row + 1, start)
                )
                harness.log_host(
                    "chunk-union "
                    + json.dumps(
                        dict(
                            stage=program.stage,
                            layer=layer,
                            start=start,
                            rows=row + 1,
                            window=window,
                            dense=int(values["dense"].reshape(-1)[0]),
                            stats=chunk_summary(values["chunks"], window, row + 1),
                        ),
                        separators=(",", ":"),
                    )
                )
            values["kept_latent"] = _on_host(states[4][0, j, row])
            values["kept_key"] = _on_host(states[5][0, j, row])
            if logits is not None and layer == 92:
                values["logits"] = _on_host(logits[0])
            saved.append((start, row, layer, values))
        return out

    program.run = probed


def explicit_steps(sizes: str, start: int, length: int) -> list[tuple[int, int]]:
    """``(computed, scheduled)`` of steps of the comma-separated ``sizes``
    from token ``start``, which must end at ``length``."""
    steps, done = [], start
    for size in (int(x) for x in sizes.split(",")):
        steps.append((done, size))
        done += size
    if done != length:
        harness.fail(f"stage prefix-check: steps {sizes} end at {done}, not {length}")
    return steps


def prefix_tables(length, hit, steps, block, state, widths, cold, take):
    """A ``prefix-check`` run's block tables (MLA, then the three KDA
    groups): new blocks (``take()``) in its MLA columns and in each step's
    KDA state column (the state after the step); with a ``hit`` (whole MLA
    blocks), the ``cold`` run's MLA columns before it and its KDA state
    column for it. Returns the tables and the new MLA and KDA blocks."""
    new: tuple[list, list] = ([], [])

    def fresh(kind):
        new[kind].append(take())
        return new[kind][-1]

    mla = np.zeros(widths[0], np.int32)
    for c in range(-(-length // block)):
        mla[c] = cold[0][c] if c < hit // block else fresh(0)
    kda = []
    for g in range(3):
        table = np.zeros(widths[1], np.int32)
        if hit:
            table[hit // state - 1] = cold[1 + g][hit // state - 1]
        for computed, scheduled in steps:
            table[(computed + scheduled - 1) // state] = fresh(1)
        kda.append(table)
    return (mla, *kda), new


def _written(tensors, index, tables, layers, length, hit, core, state, page):
    """This core's bytes a run left from token ``hit`` on: per stage MLA
    layer, the latent rows, index keys and exponents of its chunks there
    (``[tokens, ...]``), and the final KDA states' recurrent and conv rows
    per ``(pool, group)`` of the ``layers[1]`` KDA layers."""
    import torch

    split = state // page
    mla, kda = layers
    tokens, latents, keys, exponents = [], [], [], []
    for k in range(hit // state, -(-length // state)):
        if k % CORES != core:
            continue
        b = int(tables[0][k // CORES])
        keep = k * state + np.arange(state) < length
        tokens.append(k * state + np.flatnonzero(keep))
        latents.append(
            np.stack(
                [
                    tensors[m][b].cpu().view(torch.uint8).numpy().reshape(state, -1)
                    for m in range(mla)
                ]
            )[:, keep, :576]
        )
        held = index[b].cpu().numpy().reshape(mla, split, page + 32, 128)
        keys.append(held[:, :, :page].reshape(mla, state, 128)[:, keep])
        scales = held[:, :, page : page + page // 128]
        exponents.append(scales.reshape(mla, state)[:, keep])
    column = (length - 1) // state
    states = {
        (m, g): tensors[m][int(tables[1 + g][column])]
        .cpu()
        .view(torch.uint8)
        .numpy()
        .reshape(split, 128, -1)[:, :81]
        for m in range(mla)
        for g in range(3)
        if 3 * m + g < kda
    }

    def join(parts, shape):
        if not parts:
            return np.zeros((mla, 0, *shape), np.uint8)
        return np.concatenate(parts, axis=1)

    return dict(
        tokens=np.concatenate(tokens) if tokens else np.zeros(0, np.int64),
        latents=join(latents, (576,)),
        keys=join(keys, (128,)),
        exponents=join(exponents, ()),
        states=states,
    )


def equal_tokens(a, b) -> int:
    """Tokens (axis 1 of ``[layers, tokens, ...]``) whose bytes are equal in
    every layer."""
    same = a == b
    return int(same.all(axis=tuple(range(2, same.ndim))).all(0).sum())


def compare_probe_rows(cold, other, start, row, capacity) -> dict:
    """One MLA layer's probe values for a window row in two runs:
    selection by global token, scores over the valid candidates, attended
    output, latent and output row."""
    chosen = [selected_tokens(v["select"], start, capacity) for v in (cold, other)]
    scores = [
        candidate_scores(v["scores"], start, row, capacity) for v in (cold, other)
    ]
    only = [np.setdiff1d(chosen[0], chosen[1]), np.setdiff1d(chosen[1], chosen[0])]
    attended = np.abs(cold["attended"] - other["attended"]).max()
    return dict(
        selected=int(chosen[0].size),
        selection_equal=bool(not only[0].size and not only[1].size),
        only_cold=only[0][:8].tolist(),
        only_other=only[1][:8].tolist(),
        only_scores=[  # [cold, other] scores of each run's own tokens
            [[float(scores[r][t]) for t in only[o][:8]] for r in (0, 1)] for o in (0, 1)
        ],
        lowest_selected=[
            float(s[c].min()) if c.size else 0.0
            for s, c in zip(scores, chosen, strict=True)
        ],
        scores_ulps=ulps(scores[0], scores[1]),
        attended_max=float(attended),
        attended_scale=float(np.abs(cold["attended"]).max()),
        attended_ulps=ulps(cold["attended"], other["attended"]),
        latent_equal=bool(np.array_equal(cold["latent"], other["latent"])),
        key_equal=bool(np.array_equal(cold["key_bytes"], other["key_bytes"])),
        output_max=float(np.abs(cold["output"] - other["output"]).max()),
        dense=[int(cold["dense"].reshape(-1)[0]), int(other["dense"].reshape(-1)[0])],
    )


def run_prefix_check(args, records: harness.Records) -> None:
    """Kimi-K3.1's prefill under a prefix-cache hit, on the serving stages
    (``KimiPrefillStage``, as ``serve-check``): each case's prompt
    (``LENx1@HIT``, ``HIT`` whole MLA blocks) prefilled cold, cold again,
    then twice as a hit of its first ``HIT`` tokens over the first cold
    run's MLA blocks and KDA state (as vLLM's align mode shares them), the
    second with its new blocks holding stale bytes (copies of the cold run's
    first blocks), in vLLM-like steps of at most ``--budget`` tokens. After
    every window each stage MLA layer runs again with its debug outputs for
    the window's last row (:func:`_probe_windows`), and each run's rows are
    compared with the first cold run's window starting at the same token
    (:func:`compare_probe_rows`); then the latents, index keys and final KDA
    states each run left from token ``HIT`` on (:func:`_written`), and the
    first tokens."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.pools import stage_index_cache
    from vllm_torchtpu.megakernel.kimi_k3.stage import (
        HANDOFF_SNAPSHOTS,
        HIDDEN_STATES,
        SNAPSHOT,
    )

    if k3.DSA is None:
        harness.fail("stage prefix-check: Kimi-K3.1 only (--dsa-config)")
    state = layout.STAGE_SPLIT * k3.PAGE  # a core's chunk: a KDA state block
    block = CORES * state  # an MLA block
    cases, schedules = _cases(args.prompts), []
    for length, count, hit in cases:
        cold = [s for (s,) in _schedule([length], args.budget, state)]
        if args.cold_steps:
            cold = explicit_steps(args.cold_steps, 0, length)
        if count != 1 or hit % block or hit not in {c + s for c, s in cold}:
            harness.fail(
                f"stage prefix-check: {length}x{count}@{hit}: one prompt per case "
                f"and a hit of whole {block}-token blocks that a cold step ends at"
            )
        warm = [
            (hit + c, s) for ((c, s),) in _schedule([length - hit], args.budget, state)
        ]
        if args.hit_steps:
            warm = explicit_steps(args.hit_steps, hit, length)
        schedules.append((cold, warm))
    if max(length for length, _, _ in cases) > args.max_model_len:
        harness.fail("stage prefix-check: a prompt above --max-model-len")
    stage, core, mesh, group = _stage_mesh()
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        block_size=state,
        kv_cache_dtype=k3.CACHE_DTYPE,
        cp_kv_cache_interleave_size=state,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.budget,
        async_scheduling=False,
        pipeline_parallel_size=STAGES,
        prefill_context_parallel_size=CORES,
        **_engine_dsa_options(),
    )
    serving = k3_program.build_stage(
        config, Path(args.presharded), stage=stage, mesh=mesh, core=core
    )
    serving.load_weights()
    blocks = 1
    runs_spec = prefix_runs(args.runs)
    for (length, _, hit), (cold, warm) in zip(cases, schedules, strict=True):
        columns = -(-length // block)
        for _, is_hit, copies in runs_spec:
            blocks += (
                (2 if copies == "copy" else 1)
                * (columns - hit // block + 3 * len(warm))
                if is_hit
                else columns + 3 * len(cold)
            )
    tensors, caches, groups, mla, kda = _stage_pools(serving.program, k3.PAGE, blocks)
    serving.bind_kv_caches(caches, groups)
    index = stage_index_cache(caches, mla)
    saved: list = []
    _probe_windows(serving.program, mesh, saved)
    widths = (args.max_model_len // block, args.max_model_len // state)
    ids = np.arange(1, blocks)
    if args.shuffle_blocks:
        ids = np.random.default_rng(args.seed).permutation(ids)
    allocated = iter(ids.tolist())
    chain = Path(args.chain)
    names = [HIDDEN_STATES] + [SNAPSHOT.format(i) for i in range(HANDOFF_SNAPSHOTS)]

    def handoff(case_run, s, to, name):
        return chain / f"{case_run}-step{s}-stage{to}-core{core}-{name}.npy"

    for case, ((length, _, hit), (cold_steps, warm_steps)) in enumerate(
        zip(cases, schedules, strict=True)
    ):
        tag = _tag(length, 1, hit)
        (tokens,) = _tokens(args.seed, case, length, 1)
        if args.token_ids:
            tokens = np.load(args.token_ids).astype(np.int32)
            if tokens.size < length:
                harness.fail(f"stage prefix-check: {tokens.size} ids for {tag}")
            tokens = tokens[:length]
        runs: dict = {}
        for name, warm, copies in runs_spec:
            steps = warm_steps if warm else cold_steps
            tables, new = prefix_tables(
                length,
                hit if warm else 0,
                steps,
                block,
                state,
                widths,
                runs["cold"]["tables"] if warm else None,
                lambda: next(allocated),
            )
            shared = _shared_crcs(
                tensors, index, runs["cold"]["tables"], (len(mla), len(kda)), hit,
                block, state, core,
            ) if warm else None  # fmt: skip
            if copies:  # the cold run's first MLA block and first KDA state
                cold_tables = runs["cold"]["tables"]
                kept = cold_tables[1][(sum(cold_steps[0]) - 1) // state]
                onto = new
                if copies == "copy":
                    onto = ([next(allocated) for _ in new[0]],
                            [next(allocated) for _ in new[1]])  # fmt: skip
                serving._copy_blocks(
                    [(int(cold_tables[0][0]), b) for b in onto[0]]
                    + [(int(kept), b) for b in onto[1]]
                )
            handoffs: dict = {"in": {}, "out": {}}
            saved.clear()
            first = -1
            for s, (computed, scheduled) in enumerate(steps):
                row = mk.RequestInput(
                    req_id=f"{tag}-{name}",
                    computed=computed,
                    scheduled=scheduled,
                    token_ids=tokens,
                    num_tokens=length,
                    num_prompt_tokens=length,
                    block_tables=tables,
                )
                step_input = mk.StepInput(requests=[row])
                if stage > 0:
                    paths = [handoff(f"{tag}-{name}", s, stage, n) for n in names]
                    step_input.hidden = {
                        n: _take(p)[0] for n, p in zip(names, paths, strict=True)
                    }
                    handoffs["in"][computed] = [
                        zlib.crc32(np.load(p).tobytes()) for p in paths
                    ]
                    for p in paths:  # the chain holds a step in flight per stage
                        p.unlink()
                output = serving.step(step_input)
                if output.hidden is not None:
                    while (
                        s
                        and handoff(
                            f"{tag}-{name}", s - 1, stage + 1, names[0]
                        ).exists()
                    ):
                        time.sleep(0.1)
                    handoffs["out"][computed] = [
                        _give(
                            handoff(f"{tag}-{name}", s, stage + 1, n),
                            output.hidden[n][None],
                        )
                        for n in names
                    ]
                if output.tokens is not None:
                    first = max(first, int(output.tokens.cpu().numpy().reshape(-1)[0]))
            runs[name] = dict(
                tables=tables,
                rows={(start, r, layer): v for start, r, layer, v in saved},
                written=_written(
                    tensors,
                    index,
                    tables,
                    (len(mla), len(kda)),
                    length,
                    hit,
                    core,
                    state,
                    k3.PAGE,
                ),
                first=first,
                handoffs=handoffs,
                shared=(
                    (
                        shared,
                        _shared_crcs(
                            tensors,
                            index,
                            runs["cold"]["tables"],
                            (len(mla), len(kda)),
                            hit,
                            block,
                            state,
                            core,
                        ),
                    )
                    if warm
                    else None
                ),  # fmt: skip
            )
            harness.log(f"prefix-check {tag} {name}: {len(steps)} steps")
        _report_prefix_runs(
            runs, tag, stage, core, serving.program.capacity, records, runs_spec
        )
        torch.distributed.barrier(group=group)


def _shared_crcs(tensors, index, tables, layers, hit, block, state, core) -> list:
    """CRCs of what a hit shares with the first cold run, as this core holds
    it: every stage MLA layer's pages and index keys of the cached MLA blocks,
    and the KDA state each layer's groups resume from."""
    import torch

    mla, kda = layers
    crcs = []
    for b in (int(tables[0][c]) for c in range(hit // block)):
        for m in range(mla):
            crcs.append(zlib.crc32(tensors[m][b].cpu().view(torch.uint8).numpy()))
        crcs.append(zlib.crc32(index[b].cpu().numpy()))
    for m in range(mla):
        for g in range(3):
            if 3 * m + g < kda:
                b = int(tables[1 + g][hit // state - 1])
                crcs.append(zlib.crc32(tensors[m][b].cpu().view(torch.uint8).numpy()))
    return crcs


def layer_differences(a, b, tokens) -> list:
    """Per layer of two ``[layers, tokens, ...]`` arrays over ``tokens``: the
    lowest differing token and how many differ (-1, 0: none)."""
    out = []
    for x, y in zip(a, b, strict=True):
        moved = x != y
        moved = moved.any(axis=tuple(range(1, moved.ndim))) if moved.ndim > 1 else moved
        where = np.flatnonzero(moved)
        out.append((int(tokens[where].min()) if where.size else -1, int(where.size)))
    return out


def merge_layer_differences(parts: list) -> list:
    """Every core's :func:`layer_differences` of one stage, layer by layer."""
    merged = []
    for layer in zip(*parts, strict=True):
        firsts = [f for f, n in layer if n]
        merged.append((min(firsts) if firsts else -1, sum(n for _, n in layer)))
    return merged


def _report_prefix_runs(runs, tag, stage, core, capacity, records, specs) -> None:
    """Each run against the first cold run: the probe rows (from each
    stage's core 0, whose values every core shares) and every core's
    written bytes, gathered and logged by rank 0."""
    import torch

    cold = runs["cold"]
    # What the first hit saw before it ran: the shared blocks' reference.
    cold_shared = next(
        (r["shared"][0] for r in runs.values() if r["shared"] is not None), None
    )
    results = []
    for name, run in runs.items():
        if name == "cold":
            continue
        rows = []
        for key in (
            sorted(run["rows"].keys() & cold["rows"].keys()) if core == 0 else ()
        ):
            start, row, layer = key
            a, b = cold["rows"][key], run["rows"][key]
            item = compare_probe_rows(a, b, start, row, capacity)
            item.update(
                start=start,
                row=row,
                layer=layer,
                recomputed=all(
                    np.array_equal(v["latent"], v["kept_latent"])
                    and np.array_equal(v["key_bytes"], v["kept_key"])
                    for v in (a, b)
                ),
            )
            if "logits" in a:
                item["logits_max"] = float(np.abs(a["logits"] - b["logits"]).max())
            rows.append(item)
        mine, theirs = cold["written"], run["written"]
        same = {
            part: equal_tokens(mine[part], theirs[part])
            for part in ("latents", "keys", "exponents")
        }
        moved = np.flatnonzero((mine["latents"] != theirs["latents"]).any(axis=(0, 2)))
        first_moved = int(mine["tokens"][moved[0]]) if moved.size else -1
        states = sum(
            bool(np.array_equal(mine["states"][k], theirs["states"][k]))
            for k in mine["states"]
        )
        results.append(
            dict(
                run=name,
                stage=stage,
                core=core,
                rows=rows,
                tokens=int(mine["tokens"].size),
                same=same,
                states=(states, len(mine["states"])),
                first=(cold["first"], run["first"]),
                first_moved=first_moved,
                layers={
                    part: layer_differences(mine[part], theirs[part], mine["tokens"])
                    for part in ("latents", "keys", "exponents")
                },
                states_moved=sorted(
                    key
                    for key in mine["states"]
                    if not np.array_equal(mine["states"][key], theirs["states"][key])
                ),
                handoffs={
                    side: sorted(
                        start
                        for start in run["handoffs"][side]
                        if run["handoffs"][side][start]
                        != cold["handoffs"][side].get(start)
                    )
                    for side in ("in", "out")
                },
                sent=run["handoffs"]["out"],
                received=run["handoffs"]["in"],
                shared_moved=(
                    None
                    if run["shared"] is None
                    else [
                        sum(
                            x != y
                            for x, y in zip(run["shared"][i], cold_shared, strict=True)
                        )
                        for i in (0, 1)
                    ]
                ),
            )
        )
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, results)
    if harness.rank() != 0:
        return
    for name, _, _ in specs[1:]:
        mine = [r for ranked in gathered for r in ranked if r["run"] == name]
        _log_run_differences(mine, name, tag)
        rows = [dict(row, stage=r["stage"]) for r in mine for row in r["rows"]]
        for row in rows:
            records.add(
                case=tag,
                run=name,
                **{
                    k: v
                    for k, v in row.items()
                    if k not in ("only_cold", "only_other", "only_scores")
                },
            )
        differs = [row for row in rows if not row["selection_equal"]]
        tokens = sum(r["tokens"] for r in mine)
        same = {
            p: sum(r["same"][p] for r in mine) for p in ("latents", "keys", "exponents")
        }
        states = [sum(r["states"][i] for r in mine) for i in (0, 1)]
        # With a world per host, only the last stage's world has the first tokens.
        first = next((r["first"] for r in mine if r["stage"] == STAGES - 1), ("-", "-"))
        harness.log(
            f"prefix-check {tag} {name} vs cold: {len(rows)} rows (window starts "
            f"{sorted({row['start'] for row in rows})}): selection equal in "
            f"{len(rows) - len(differs)}, scores within "
            f"{max((row['scores_ulps'] for row in rows), default=0)} ulp, attended "
            f"within {max((row['attended_max'] for row in rows), default=0):.2e} "
            f"({max((row['attended_ulps'] for row in rows), default=0)} ulp), "
            f"latents equal in {sum(row['latent_equal'] for row in rows)}, output "
            f"rows within {max((row['output_max'] for row in rows), default=0):.2e}; "
            f"recomputed rows equal the stage op's in "
            f"{sum(row['recomputed'] for row in rows)}"
        )
        harness.log(
            f"prefix-check {tag} {name} vs cold: written from the hit on, "
            f"{same['latents']}/{tokens} tokens' latents, {same['keys']}/{tokens} "
            f"keys, {same['exponents']}/{tokens} exponents and {states[0]}/"
            f"{states[1]} final KDA states equal; first token {first[1]} (cold "
            f"{first[0]}); per-core lowest differing latent token "
            f"{sorted({r['first_moved'] for r in mine if r['first_moved'] >= 0})[:8]}"
            + "".join(
                f"; logits within {row['logits_max']:.2e} (window start {row['start']})"
                for row in rows
                if "logits_max" in row
            )
        )
        for start in sorted({row["start"] for row in rows}):
            moved = [
                row
                for row in sorted(rows, key=lambda r: (r["stage"], r["layer"]))
                if row["start"] == start
                and not (
                    row["selection_equal"]
                    and row["latent_equal"]
                    and row["attended_ulps"] == 0
                    and row["output_max"] == 0
                )
            ]
            if moved:
                row = moved[0]
                harness.log(
                    f"prefix-check {tag} {name}: window start {start}: first "
                    f"difference at stage {row['stage']} layer {row['layer']} "
                    f"(selection equal {row['selection_equal']}, scores within "
                    f"{row['scores_ulps']} ulp, attended within "
                    f"{row['attended_max']:.2e} of {row['attended_scale']:.2e}, "
                    f"latent equal {row['latent_equal']}, output within "
                    f"{row['output_max']:.2e})"
                )
        for row in differs[:4]:
            harness.log(
                f"prefix-check {tag} {name}: stage {row['stage']} layer "
                f"{row['layer']} window start {row['start']} row {row['row']}: of "
                f"{row['selected']} selected, cold only {row['only_cold']} (scores "
                f"cold, {name}: {row['only_scores'][0]}), {name} only "
                f"{row['only_other']} ({row['only_scores'][1]}); "
                f"lowest selected score cold {row['lowest_selected'][0]}, {name} "
                f"{row['lowest_selected'][1]}; scores within {row['scores_ulps']} ulp"
            )
        records.add(
            case=tag,
            run=name,
            summary=True,
            rows=len(rows),
            selection_differs=len(differs),
            tokens=tokens,
            same=same,
            states=states,
            first=list(first),
        )


def _log_run_differences(mine: list, name: str, tag: str) -> None:
    """One run's differences from the first cold run, stage by stage: each
    layer's lowest differing token and count, KDA states, hand-offs, the
    bytes shared with the cold run, and any hand-off changed on its way."""
    shared = [r["shared_moved"] for r in mine if r["shared_moved"] is not None]
    if shared:
        harness.log(
            f"prefix-check {tag} {name}: shared blocks' CRCs that moved, summed over "
            f"cores: before the run {sum(s[0] for s in shared)}, after it "
            f"{sum(s[1] for s in shared)}"
        )
    sent = {(r["stage"], r["core"]): r["sent"] for r in mine}
    garbled = [
        (r["stage"], r["core"], start)
        for r in mine
        for start, crcs in r["received"].items()
        if sent.get((r["stage"] - 1, r["core"]), {}).get(start) != crcs
    ]
    harness.log(
        f"prefix-check {tag} {name}: hand-offs changed between stages: {garbled[:8]}"
    )
    for stage in range(STAGES):
        own = [r for r in mine if r["stage"] == stage]
        moved = {
            part: merge_layer_differences([r["layers"][part] for r in own])
            for part in ("latents", "keys", "exponents")
        }
        states = sum(len(r["states_moved"]) for r in own)
        inputs = sorted({s for r in own for s in r["handoffs"]["in"]})
        outputs = sorted({s for r in own for s in r["handoffs"]["out"]})
        if (
            any(n for rows in moved.values() for _, n in rows)
            or states
            or inputs
            or outputs
        ):
            harness.log(
                f"prefix-check {tag} {name}: stage {stage}: "
                f"latents {moved['latents']}, "
                f"keys {moved['keys']}, exponents {moved['exponents']}, KDA states "
                f"moved {states}, hand-off in moved at {inputs}, out at {outputs}"
            )


def run_bench(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.stage import (
        STAGE_WINDOWS as SERVING_WINDOWS,
    )
    from vllm_torchtpu.megakernel.kimi_k3.stage import StagePrefillProgram

    cases = _step_cases(args.prompts)
    stage, core, mesh, group = _stage_mesh()
    page = k3.PAGE
    chunk = layout.STAGE_SPLIT if args.serving_prefix else 1
    windows = SERVING_WINDOWS if args.serving_windows else STAGE_WINDOWS
    slots = max(len(pieces) for pieces in cases)
    capacity = max(
        [args.capacity]
        + [sum(prefix for _, prefix in pieces) for pieces in cases]
        + [page * CORES]
    )
    capacity = -(-capacity // (page * CORES)) * page * CORES
    program = StagePrefillProgram(
        stage, mesh, core, capacity, slots, args.moe_path, **_dsa_options()
    )
    program.load_weights(args.presharded)
    harness.log(
        f"stage {stage}: layers {program.plan['first']}..{program.plan['last'] - 1}, "
        f"groups {program.plan['groups']}, prefix arena {capacity} tokens"
    )
    cache = layout.cache_layout(k3.CACHE_DTYPE)
    pool_blocks = 2 + max(
        _pieces_arena(c, CORES, [p for _, p in pieces], page, 0, 0, chunk)[2]
        + len(pieces)
        for pieces in cases
        for c in range(CORES)
    )
    generator = torch.Generator().manual_seed(core)
    pool = (
        (torch.randn((pool_blocks, *cache.block_shape), generator=generator) * 0.5)
        .to(getattr(torch, cache.dtype))
        .to("tpu")
    )
    layers = program.plan["mla_layers"]
    indexed = {}
    if k3.DSA is not None:
        indexed = dict(
            index=_index_pool(pool_blocks, layers, page, generator),
            index_slots=range(layers),
        )
    chain = Path(args.chain) if args.chain else None
    reference = Path(args.reference) if args.reference else None
    for case, pieces in enumerate(cases):
        tag = _step_tag(pieces)
        lengths = [length for length, _ in pieces]
        _, extent = _pieces_descriptor(0, len(pieces), pieces, page)
        window = min(w for w in windows if w >= extent)
        kernels.check_window(window, "stage window", CORES)
        cslots = program.window_slots(window)
        desc, extent = _pieces_descriptor(window, cslots, pieces, page)
        ids, local, _ = _pieces_arena(
            core,
            CORES,
            [prefix for _, prefix in pieces],
            page,
            cslots,
            capacity // CORES + cslots * page * chunk,
            chunk,
        )
        local_rows = window // CORES
        order = kernels.sp_order(window, CORES)
        tokens = np.full(window, 0, np.int32)
        rng = np.random.default_rng(args.seed * 1000 + case)
        for offset, length in zip(desc[:cslots], lengths):
            tokens[offset : offset + length] = rng.integers(0, VOCAB - 1024, length)
        device = dict(
            length=torch.tensor([extent], dtype=torch.int32).to("tpu"),
            segments=torch.from_numpy(desc).to("tpu"),
            ids=torch.from_numpy(ids).to("tpu"),
            local=torch.from_numpy(local).to("tpu"),
            tokens=torch.from_numpy(
                tokens[order[core * local_rows : (core + 1) * local_rows]][None]
            ).to("tpu"),
        )
        valid_blocks = 2 * stage
        times = []
        for repeat in range(args.repeats + 1):
            inputs = {}
            if stage > 0:
                if chain is not None:
                    inputs["prefix"] = _take(
                        chain / f"{tag}-{repeat}-stage{stage}-core{core}-prefix.npy"
                    )
                    blocks = _take(
                        chain / f"{tag}-{repeat}-stage{stage}-core{core}-blocks.npy"
                    )
                    full = torch.zeros(
                        (1, 8, local_rows, 7168), dtype=torch.bfloat16
                    ).to("tpu")
                    inputs["blocks"] = torch.cat(
                        (blocks, full[:, valid_blocks:]), dim=1
                    )
                else:
                    gen = torch.Generator().manual_seed(1000 * stage + core)
                    inputs["prefix"] = (
                        (torch.randn((1, local_rows, 7168), generator=gen))
                        .to(torch.bfloat16)
                        .to("tpu")
                    )
                    inputs["blocks"] = (
                        (torch.randn((1, 8, local_rows, 7168), generator=gen))
                        .to(torch.bfloat16)
                        .to("tpu")
                    )
            torch.distributed.barrier(group=group)
            last = repeat == args.repeats
            with bench._trace(args.trace, f"{tag}-stage{stage}", last):
                started = time.perf_counter()
                prefix_out, blocks_out, _, logits, routes, _ = program.run(
                    window,
                    page,
                    device["length"],
                    device["segments"],
                    [pool] * layers,
                    device["ids"],
                    device["local"],
                    tokens=device["tokens"],
                    **indexed,
                    **inputs,
                )
                bridge.materialize(logits if logits is not None else prefix_out)
                elapsed = 1e3 * (time.perf_counter() - started)
            if repeat:
                times.append(elapsed)
            if chain is not None and stage < STAGES - 1:
                base = chain / f"{tag}-{repeat}-stage{stage + 1}-core{core}"
                _give(base.with_name(base.name + "-prefix.npy"), prefix_out)
                _give(
                    base.with_name(base.name + "-blocks.npy"),
                    blocks_out[:, : valid_blocks + 2],
                )
            if last and logits is not None and reference is not None and core == 0:
                _compare(reference / f"{tag}.npy", logits, len(pieces), tag)
            if last and logits is not None and args.save and core == 0:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                np.save(
                    Path(args.save) / f"{tag}.npy",
                    logits.float().cpu().numpy()[: len(pieces)],
                )
            if last and core == 0 and routes:
                busiest = max(int(r.reshape(-1)[0].cpu()) for r in routes)
                harness.log_host(f"{tag} stage {stage}: busiest chip routes {busiest}")
        ms = statistics.median(times)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, (stage, core, ms, _hbm_peak()))
        stages = sorted({s for s, *_ in gathered})
        per_stage = [max(ms for s, _, ms, _ in gathered if s == st) for st in stages]
        peaks = [
            max(peak for s, _, _, (peak, _) in gathered if s == st) for st in stages
        ]
        limit = min(limit for *_, (_, limit) in gathered)
        harness.log(
            f"stage HBM peak {tag}: "
            + " / ".join(f"{p / 2**30:.1f}" for p in peaks)
            + f" GiB of {limit / 2**30:.1f} (the busiest core per stage, so far)"
        )
        rate = sum(lengths) * 1e3 / max(per_stage)
        harness.log(
            f"stage steps {tag} (window {window}, stages {stages}): "
            + " / ".join(f"{t:.1f}" for t in per_stage)
            + f" ms; pipeline {rate:.0f} tok/s, one window through the stages "
            f"{sum(per_stage):.1f} ms (repeats on stage 0 core 0: "
            f"{[round(t, 1) for t in times]})"
        )
        records.add(
            case=tag,
            length=sum(lengths),
            count=len(pieces),
            context=max(prefix for _, prefix in pieces),
            pieces=pieces,
            window=window,
            stages=stages,
            stage_ms=[round(t, 2) for t in per_stage],
            ms=round(max(per_stage), 2),
            tokens_per_s=round(rate, 1),
            latency_ms=round(sum(per_stage), 2),
            capacity=capacity,
            hbm_peak_gib=[round(p / 2**30, 2) for p in peaks],
        )


def _hbm_peak() -> tuple[int, int]:
    """This worker's device's ``(peak bytes in use, bytes limit)`` (zeros
    when its runtime reports none)."""
    import jax

    from vllm_torchtpu.megakernel import bridge

    try:
        _, position = bridge.kernel_mesh("tp")
        device = sorted(jax.devices(), key=lambda d: int(d.id))[position]
        stats = device.memory_stats() or {}
    except Exception:  # noqa: BLE001 -- diagnostics only
        stats = {}
    return int(stats.get("peak_bytes_in_use", 0)), int(stats.get("bytes_limit", 0))


def _compare(path: Path, logits, count: int, tag: str) -> None:
    expected = np.load(path)
    got = logits.float().cpu().numpy()[:count]
    for s in range(count):
        e, g = expected[s].astype(np.float64), got[s].astype(np.float64)
        rel = np.linalg.norm(g - e) / max(np.linalg.norm(e), 1e-30)
        pe = np.exp(e - e.max())
        pe /= pe.sum()
        pg = np.exp(g - g.max())
        pg /= pg.sum()
        kl = float(np.sum(pe * (np.log(pe + 1e-30) - np.log(pg + 1e-30))))
        top_e, top_g = np.argsort(-e)[:2], np.argsort(-g)[:2]
        harness.log_host(
            f"{tag} sequence {s}: logits rel-L2 {rel:.4f}, KL {kl:.5f}, argmax "
            f"{int(e.argmax())} (slice) vs {int(g.argmax())} (stages); top-2 margin "
            f"{e[top_e[0]] - e[top_e[1]]:.4f} (slice, {top_e.tolist()}) "
            f"{g[top_g[0]] - g[top_g[1]]:.4f} (stages, {top_g.tolist()}); "
            "slice's top two in the "
            f"stages {g[top_e[0]] - g[top_e[1]]:+.4f}, largest logit difference "
            f"{np.abs(g - e).max():.4f}"
        )


def _mla_probe_op(mesh, window: int, slots: int, capacity: int, layer: int):
    """``run_mla_probe``'s op: both MLA layers on the same operands."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_stage

    heads = 4 * 384
    new_layer = prefill_stage.make_mla_layer_stage(
        mesh, seq_len=window, segments=slots, prefix_capacity=capacity
    )
    norm_gather = kernels.make_norm_gather_sp(mesh, seq_len=window)
    old_layer = kernels.make_indexed_mla_layer_sp_ring(
        mesh,
        seq_len=window,
        segments=slots,
        prefix_capacity=capacity,
        flash_tile=256,
    )

    def head_slices(gate, out):
        rank = jax.lax.axis_index("k3_sp")
        gate = jax.lax.dynamic_slice_in_dim(gate[0], rank * heads, heads, axis=2)
        out = jax.lax.dynamic_slice_in_dim(out[0], rank * heads, heads, axis=1)
        return gate[None], out[None]

    def both(
        prefix_rows,
        blocks,
        folded,
        norm,
        length_,
        segments,
        pool_,
        ids_,
        local_,
        *families,
    ):
        families = dict(zip(kernels.MLA_FAMILIES, families, strict=True))
        arena = kernels.CpPrefix((pool_,), ids_, local_)
        slot = jnp.zeros((1,), jnp.int32)
        active = jnp.full((1,), layer // 12 + 1, jnp.int32)
        new_out, new_latent = new_layer(
            prefix_rows,
            blocks,
            folded,
            norm,
            active,
            families,
            segments,
            length_,
            slot,
            arena,
        )
        gate, out = jax.shard_map(
            head_slices,
            mesh=mesh,
            in_specs=(P("k3_sp"), P("k3_sp")),
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )(families["m_gate"], families["m_o"])
        sliced = dict(families, m_gate=gate, m_o=out)
        _, gathered = norm_gather(prefix_rows, blocks, folded, norm, active)
        old_out, old_latent = old_layer(
            gathered, prefix_rows, sliced, segments, length_, slot, arena
        )
        return new_out, new_latent, old_out, old_latent

    sharded, replicated = P("k3_sp"), P()
    return bridge.register_global_op(
        f"pallas::k3_stage_mla_probe_w{window}",
        mesh,
        both,
        (sharded,) * 4
        + (replicated,) * 2
        + (sharded,) * 3
        + (sharded,) * len(kernels.MLA_FAMILIES),
        (sharded,) * 4,
    )


def run_reference(args, records: harness.Records) -> None:
    """The slice program's first-token logits for each case, saved as
    ``OUTPUT/<case>.npy`` ([count, VOCAB] FP32) by rank 0."""
    import torch

    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program

    cases = _cases(args.prompts)
    if any(prefix for _, _, prefix in cases):
        harness.fail("stage reference: fresh prompts only")
    if args.moe_path == "reference":
        from vllm_torchtpu.megakernel.kimi_k3.kernels import (
            prefill_megakernel as kernels,
        )

        kernels.make_indexed_moe_sp_ring = kernels._make_indexed_moe_sp_ring_reference
    max_seqs = max(count for _, count, _ in cases)
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=max_seqs,
            max_model_len=k3.LOGICAL,
            max_num_batched_tokens=k3_program.MAX_STEP_TOKENS,
        ),
        args.presharded,
        tp_rank=torch.distributed.get_rank(),
    )
    prefiller.load_weights()
    pools = k3.Pools(4, 1, 1)
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    program, page = prefiller.program, k3.PAGE
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for case, (length, count, prefix) in enumerate(cases):
        tag = _tag(length, count, prefix)
        rows = -(-length // page) * page
        extent_rows = rows * (count - 1) + length
        window = min(w for w in program.windows if w >= extent_rows)
        cslots = program.window_slots(window)
        desc, extent = _descriptor(window, cslots, length, count, 0, page)
        tokens = np.full(window, 0, np.int32)
        for s, seq in enumerate(_tokens(args.seed, case, length, count)):
            tokens[s * rows : s * rows + length] = seq
        local_rows = window // 32
        order = program.sp_order(window)
        rank = prefiller.rank
        shard = torch.from_numpy(
            tokens[order[rank * local_rows : (rank + 1) * local_rows]][None]
        ).to("tpu")
        recurrent, conv = program.stack_states([program.zero_state()] * cslots)
        ids = torch.zeros((1, 1 + prefiller.cp_capacity // page), dtype=torch.int32).to(
            "tpu"
        )
        local = torch.zeros((1, 2 * cslots), dtype=torch.int32).to("tpu")
        logits, *_ = program.run_packed(
            shard,
            torch.tensor([extent], dtype=torch.int32).to("tpu"),
            torch.from_numpy(desc).to("tpu"),
            recurrent,
            conv,
            prefiller._writer.mla_pools(),
            ids,
            local,
            index=prefiller._writer.index,
        )
        values = logits.float().cpu().numpy()[:count]
        if rank == 0:
            np.save(output / f"{tag}.npy", values)
            harness.log(
                f"reference {tag} (window {window}): "
                f"argmax {values.argmax(axis=-1).tolist()}"
            )
        records.add(case=tag, window=window)


def _requests(
    case: int, length: int, count: int, seed: int, token_ids: str | None = None
):
    """A case's requests: ``(req_id, token ids)``; with ``token_ids`` (a .npy
    of real text), request ``r`` takes its ``[r * length, (r + 1) * length)``."""
    if token_ids:
        ids = np.load(token_ids).astype(np.int32)
        if ids.size <= count * length:
            harness.fail(f"{token_ids}: {ids.size} ids for {count} x {length}")
        return [
            (f"c{case}r{r}", ids[r * length : (r + 1) * length]) for r in range(count)
        ]
    return [
        (f"c{case}r{r}", tokens)
        for r, tokens in enumerate(_tokens(seed, case, length, count))
    ]


def _next_tokens(length: int, count: int, token_ids: str | None) -> list[int]:
    """The real text's token after each request of ``_requests`` (none for
    random ids)."""
    if not token_ids:
        return []
    ids = np.load(token_ids)
    return [int(ids[(r + 1) * length]) for r in range(count)]


def run_pool_reference(args, records: harness.Records) -> None:
    """The slice prefill (``KimiPrefiller``) over each case in one step, its
    requests' pages and final KDA states read back from the decode-layout
    pools: ``OUTPUT/rank<q>.npz`` per TP rank, keyed ``<req>/mla<m>/<column>``
    (this rank's page in that MLA block column, ``[page, 576]`` BF16 bits)
    and ``<req>/kda<k>/{recurrent,conv}`` (the heads of kernel rank
    ``kernel``, as the pool reader returns them), and each request's first
    token on rank 0."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.pools import kda_blocks_at_length

    cases = _cases(args.prompts)
    if any(prefix for _, _, prefix in cases) or any(
        length * count > k3_program.MAX_STEP_TOKENS for length, count, _ in cases
    ):
        harness.fail("pool-reference: fresh prompts of one step each")
    prefiller = k3_program.build(
        k3.engine_config(
            "prefill",
            max_num_seqs=max(count for _, count, _ in cases),
            max_model_len=k3.LOGICAL,
            max_num_batched_tokens=k3_program.MAX_STEP_TOKENS,
        ),
        args.presharded,
        None if args.draft is None else Path(args.draft),
        tp_rank=torch.distributed.get_rank(),
    )
    prefiller.load_weights()
    blocks = sum(
        count * (2 + 3 * (length // k3.PAGE + 1)) for length, count, _ in cases
    )
    drafts = (
        0 if prefiller.draft_latents is None else prefiller.draft_latents.config.layers
    )
    pools = k3.Pools(blocks + 2, 1, k3.LOGICAL // k3.PAGE, draft_layers=drafts)
    prefiller.bind_kv_caches(pools.caches, pools.groups)
    writer = prefiller._writer
    assert writer is not None
    rank, kernel = torch.distributed.get_rank(), prefiller.rank
    saved: dict[str, np.ndarray] = {"kernel": np.asarray(kernel)}
    for case, (length, count, _) in enumerate(cases):
        rows = []
        for req_id, tokens in _requests(case, length, count, args.seed, args.token_ids):
            mla = pools.fresh(1)
            kda = [pools.fresh(-(-length // k3.PAGE)) for _ in range(3)]
            rows.append(
                mk.RequestInput(
                    req_id=req_id,
                    computed=0,
                    scheduled=length,
                    token_ids=np.asarray(tokens, np.int32),
                    num_tokens=length,
                    num_prompt_tokens=length,
                    block_tables=pools.tables(mla, kda),
                )
            )
        first = prefiller.step(mk.StepInput(requests=rows)).tokens
        assert first is not None
        if rank == 0:
            saved[f"case{case}/first"] = first.cpu().numpy().reshape(-1)[:count]
        for row in rows:
            column = row.block_tables[0][0]
            pages = [(f"mla{m}", pools.tensors[writer.mla_raw[m]]) for m in range(24)]
            pages += [(f"draft{d}", pools.tensors[24 + d]) for d in range(drafts)]
            for name, tensor in pages:
                page = tensor[int(column)].float().cpu()
                saved[f"{row.req_id}/{name}/0"] = (
                    page.reshape(-1, 640)[:, :576]
                    .to(torch.bfloat16)
                    .view(torch.int16)
                    .numpy()
                )
            if writer.index is not None:
                held = writer.index[int(column)].cpu().numpy()
                for m in range(24):
                    saved[f"{row.req_id}/index{m}/0"] = held[m]
            finals = kda_blocks_at_length(
                row.block_tables, prefiller._kda_groups, length
            )
            recurrent, conv = writer.read_states(torch.from_numpy(finals).to("tpu"))
            recurrent, conv = recurrent.cpu().numpy(), conv.float().cpu().numpy()
            for k in range(69):
                saved[f"{row.req_id}/kda{k}/recurrent"] = recurrent[0, k, 0]
                saved[f"{row.req_id}/kda{k}/conv"] = conv[0, k, 0]
        harness.log(
            f"pool reference {_tag(length, count, 0)}: first tokens "
            f"{first.cpu().numpy().reshape(-1)[:count].tolist()} (text "
            f"{_next_tokens(length, count, args.token_ids)})"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f"rank{rank:02d}.npz", **saved)
    records.add(cases=args.prompts, rank=rank)


def _stage_pools(program, page: int, blocks: int, draft_layers: int = 0):
    """Stand-in pools of a stage's layers as vLLM allocates them (stage
    block: four decode blocks; pool i holds the stage's MLA layer i and its
    KDA layers 3i..3i+2; then ``draft_layers`` MLA DSpark draft pools; with
    Kimi-K3.1 the index-key cache under the MLA layers' index names) and the
    KV cache groups: the MLA layers' (``8 * 4 * page``-token blocks) and
    three KDA groups (``4 * page``)."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import layout, pools

    first, last = program.plan["first"], program.plan["last"]
    mla = [i for i in layout.MLA if first <= i < last]
    kda = [i for i in layout.KDA if first <= i < last]
    stage = layout.stage_cache_layout(k3.CACHE_DTYPE, tuple(mla), tuple(kda))
    dtype = getattr(torch, stage.dtype)
    tensors = [
        torch.zeros((blocks, *stage.block_shape), dtype=dtype).to("tpu")
        for _ in range(len(mla) + draft_layers)
    ]
    mla_names = [f"model.layers.{i}.self_attn.mla_attn.attn" for i in mla]
    mla_names += [
        f"model.layers.{len(layout.MLA) + len(layout.KDA) + d}.self_attn.mla_attn.attn"
        for d in range(draft_layers)
    ]
    kda_names = [f"model.layers.{i}.self_attn" for i in kda]
    caches = {name: tensors[m] for m, name in enumerate(mla_names)}
    caches.update({name: tensors[k // 3] for k, name in enumerate(kda_names)})
    if k3.DSA is not None:
        index_names = [f"model.layers.{i}{pools.INDEX_CACHE}" for i in mla]
        caches.update(
            pools.allocate_index_cache(
                blocks, index_names, layout.STAGE_SPLIT * page, k3.CACHE_DTYPE, "tpu"
            )
        )
    tokens = stage.tokens_per_block
    groups = [mk.KVGroup(tuple(mla_names), CORES * tokens, 4)]
    groups += [
        mk.KVGroup(tuple(n for k, n in enumerate(kda_names) if k % 3 == g), tokens, 64)
        for g in range(3)
    ]
    return tensors, caches, groups, mla, kda


def _schedule(
    lengths: list[int], budget: int, chunk: int
) -> list[list[tuple[int, int]]]:
    """vLLM-like steps over fresh requests of ``lengths``: each step's
    ``(computed, scheduled)`` per request, at most ``budget`` tokens, chunks
    ending on ``chunk``-token boundaries unless they end the prompt."""
    done, steps = [0] * len(lengths), []
    while any(d < n for d, n in zip(done, lengths, strict=True)):
        left, step = budget, []
        for i, n in enumerate(lengths):
            take = min(n - done[i], left)
            if done[i] + take < n:
                take -= (done[i] + take) % chunk
            step.append((done[i], max(take, 0)))
            done[i] += max(take, 0)
            left -= max(take, 0)
        steps.append(step)
    return steps


def run_serve_check(args, records: harness.Records) -> None:
    """The serving stages over each case's requests, in vLLM-like steps of
    at most ``--budget`` tokens (hand-offs through ``--chain`` files), then
    each core's pools against ``pool-reference``'s: every page it owns of
    its stage's MLA layers (the decoder's pages ``4 k .. 4 k + 3`` of chunk
    ``k``, DCP ranks ``4 core + i``) and every KDA state's head shard ``g``
    (TP32 rank ``4 core + g``), as relative L2; and the first tokens. With
    Kimi-K3.1 (``--dsa-config``) also the index keys of those pages
    (:func:`_index_checks`)."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.pools import (
        kda_blocks_at_length,
        stage_index_cache,
    )
    from vllm_torchtpu.megakernel.kimi_k3.stage import (
        HANDOFF_SNAPSHOTS,
        HIDDEN_STATES,
        SNAPSHOT,
    )

    cases = _cases(args.prompts)
    stage, core, mesh, group = _stage_mesh()
    split = layout.STAGE_SPLIT
    config = MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        block_size=split * k3.PAGE,
        kv_cache_dtype=k3.CACHE_DTYPE,
        cp_kv_cache_interleave_size=split * k3.PAGE,
        max_num_seqs=max(count for _, count, _ in cases),
        max_model_len=k3.LOGICAL,
        max_num_batched_tokens=args.budget,
        async_scheduling=False,
        pipeline_parallel_size=STAGES,
        prefill_context_parallel_size=CORES,
        **_engine_dsa_options(),
    )
    serving = k3_program.build_stage(
        config,
        Path(args.presharded),
        stage=stage,
        mesh=mesh,
        core=core,
        draft=None if args.draft is None else Path(args.draft),
    )
    serving.load_weights()
    blocks = 2 + sum(count * (1 + 3) for _, count, _ in cases)
    drafts = 0 if serving.draft is None else serving.draft.config.layers
    tensors, caches, groups, mla, kda = _stage_pools(
        serving.program, k3.PAGE, blocks, drafts
    )
    serving.bind_kv_caches(caches, groups)
    index_cache = stage_index_cache(caches, mla)
    serving.warmup()
    reader = _shard_reader(mesh)
    chain = Path(args.chain)
    references: dict = {}
    next_block = 1
    for case, (length, count, prefix) in enumerate(cases):
        if prefix:
            harness.fail("serve-check: fresh prompts only")
        requests = _requests(case, length, count, args.seed, args.token_ids)
        tables = []
        for _ in requests:
            mla_table = np.zeros(4, np.int32)
            mla_table[0] = next_block
            kda_tables = []
            for _ in range(3):
                table = np.zeros(64, np.int32)
                table[(length - 1) // (split * k3.PAGE)] = (
                    next_block + 1 + len(kda_tables)
                )
                kda_tables.append(table)
            next_block += 4
            tables.append((mla_table, *kda_tables))
        steps = _schedule([length] * count, args.budget, split * k3.PAGE)
        first = np.full(count, -1)
        for s, step in enumerate(steps):
            rows = [
                mk.RequestInput(
                    req_id=req_id,
                    computed=computed,
                    scheduled=scheduled,
                    token_ids=np.asarray(tokens, np.int32),
                    num_tokens=length,
                    num_prompt_tokens=length,
                    block_tables=tables[r],
                )
                for r, ((req_id, tokens), (computed, scheduled)) in enumerate(
                    zip(requests, step, strict=True)
                )
            ]
            step_input = mk.StepInput(requests=rows)
            base = chain / f"case{case}-step{s}"
            if stage > 0:
                names = [HIDDEN_STATES] + [
                    SNAPSHOT.format(i) for i in range(HANDOFF_SNAPSHOTS)
                ]
                names += list(serving.handoff_taps)
                step_input.hidden = {
                    name: _take(
                        base.with_name(
                            f"{base.name}-stage{stage}-core{core}-{name}.npy"
                        )
                    )[0]
                    for name in names
                }
            output = serving.step(step_input)
            if output.hidden is not None:
                for name, tensor in output.hidden.items():
                    _give(
                        base.with_name(
                            f"{base.name}-stage{stage + 1}-core{core}-{name}.npy"
                        ),
                        tensor[None],
                    )
            if output.tokens is not None:
                tokens = output.tokens.cpu().numpy().reshape(-1)[:count]
                first = np.where(tokens >= 0, tokens, first)
        tag = _tag(length, count, prefix)
        worst, latent_pages = [], {}
        if not references:
            for path in sorted(Path(args.reference).glob("rank*.npz")):
                data = np.load(path)
                tp = int(path.stem.removeprefix("rank"))
                references[("tp", tp)] = data
                references[("kernel", int(data["kernel"]))] = data
        for r, (req_id, _) in enumerate(requests):
            mla_table, *kda_tables = tables[r]
            latent_pools = [
                (m, f"mla{layout.MLA.index(layer)}", f"mla {layer}")
                for m, layer in enumerate(mla)
            ] + [(len(mla) + d, f"draft{d}", f"draft {d}") for d in range(drafts)]
            for m, key_name, what in latent_pools:
                pool = tensors[m][int(mla_table[0])].float().cpu()
                pages = pool.reshape(split, -1, 640)[:, :, :576].numpy()
                latent_pages[req_id, m] = pages
                for i in range(split):
                    q = split * core + i
                    if q * k3.PAGE >= length:
                        continue
                    key = f"{req_id}/{key_name}/0"
                    expected = torch.from_numpy(references[("tp", q)][key])
                    expected = expected.view(torch.bfloat16).float().numpy()
                    worst.append((_rel(pages[i], expected), f"{what} page {q}"))
            finals = kda_blocks_at_length(tables[r], serving._kda_groups, length)
            for k, layer in enumerate(kda):
                ids = split * int(finals[k]) + np.arange(split)
                recurrent, conv = reader(tensors[k // 3], ids)
                index = layout.KDA.index(layer)
                for g in range(split):
                    reference = references[("kernel", split * core + g)]
                    for name, got in (("recurrent", recurrent[g]), ("conv", conv[g])):
                        expected = reference[f"{req_id}/kda{index}/{name}"]
                        worst.append(
                            (_rel(got, expected), f"kda {layer} shard {g} {name}")
                        )
        worst.sort(reverse=True)
        harness.log_host(
            f"{tag} stage {stage} core {core}: {len(worst)} pool regions, worst rel-L2 "
            + ", ".join(f"{v:.4f} ({what})" for v, what in worst[:3])
        )
        by_layer: dict = {}
        for v, what in worst:
            kind, layer = what.split()[:2]
            if kind in ("mla", "kda"):
                key = (kind, int(layer))
                by_layer[key] = max(by_layer.get(key, 0.0), v)
        harness.log_host(
            f"{tag} stage {stage} core {core}: worst rel-L2 per layer "
            + " ".join(
                f"{kind}{layer}:{v:.3f}"
                for (kind, layer), v in sorted(
                    by_layer.items(), key=lambda item: (item[0][1], item[0][0])
                )
            )
        )
        latents = [(v, what) for v, what in worst if what.startswith("mla ")]
        if latents:
            harness.log_host(
                f"{tag} stage {stage} core {core}: worst latent rel-L2 "
                f"{latents[0][0]:.4f} ({latents[0][1]})"
            )
        if index_cache is not None:
            _index_checks(
                index_cache,
                tables,
                requests,
                latent_pages,
                mla,
                length,
                core,
                tag,
                stage,
                records,
            )
            _index_against_reference(
                index_cache, tables, requests, references, mla, length, core, tag, stage
            )
        if serving.last and core == 0:
            expected = np.load(Path(args.reference) / "rank00.npz")[f"case{case}/first"]
            harness.log_host(
                f"{tag} first tokens {first.tolist()} (slice {expected.tolist()}, text "
                f"{_next_tokens(length, count, args.token_ids)})"
            )
        records.add(
            case=tag, stage=stage, core=core, worst=worst[0][0] if worst else 0.0
        )
        torch.distributed.barrier(group=group)


def _index_against_reference(
    index, tables, requests, references, mla, length, core, tag, stage
) -> None:
    """The index keys a core committed of each request's pages against
    ``pool-reference``'s (the slice prefill's, ``<req>/index<m>/0`` of TP
    rank ``4 core + i``): prompt tokens whose key bytes and exponent are equal,
    within one FP8 step (``check.key_steps``) and beyond, and the largest
    dequantized difference. Skipped for a Kimi-K3 reference."""
    import jax

    from tools.megakernel.check import key_steps
    from vllm_torchtpu.megakernel.kimi_k3 import layout

    split, page = layout.STAGE_SPLIT, k3.PAGE
    tokens = same = beyond = 0
    largest = 0.0
    for r, (req_id, _) in enumerate(requests):
        held = index[int(tables[r][0][0])].cpu().numpy()
        held = held.reshape(len(mla), split, page + 32, 128)
        for m, layer in enumerate(mla):
            key = f"{req_id}/index{layout.MLA.index(layer)}/0"
            for i in range(split):
                q = split * core + i
                prompt = min(max(length - q * page, 0), page)
                if not prompt or key not in references[("tp", q)]:
                    continue
                want = references[("tp", q)][key]
                got = held[m, i]
                exponents = (
                    got[page:].reshape(-1)[:prompt],
                    want[page:].reshape(-1)[:prompt],
                )
                with jax.default_device(jax.devices("cpu")[0]):
                    equal, far, difference = key_steps(
                        got[:prompt], exponents[0], want[:prompt], exponents[1]
                    )
                tokens += prompt
                same += int(equal.sum())
                beyond += int(far.sum())
                largest = max(largest, float(difference.max()))
    if tokens:
        harness.log_host(
            f"{tag} stage {stage} core {core}: index keys vs the slice's: "
            f"{same}/{tokens} "
            f"prompt tokens equal, {tokens - same - beyond} within one FP8 step, "
            f"{beyond} beyond; largest difference {largest:.3e}"
        )


def _index_checks(
    index, tables, requests, latent_pages, mla, length, core, tag, stage, records
) -> None:
    """The index keys a core committed of each request's pages (its
    sub-pages ``i`` of the request's MLA block, decode pages ``4 core + i``):
    every prompt token keyed (a non-zero exponent byte) and nothing past the
    prompt; under rope-probe each key against the FP8 encoding of the
    LayerNorm of ``[rope, -rope]`` of the latent row the same pool holds
    (``check.key_steps``: equal, off by at most one FP8 step, or beyond). The
    reference runs on the host CPU: a worker's TPU devices belong to the
    torch_tpu runtime."""
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from tools.megakernel import dsa_reference as ref
    from tools.megakernel.check import key_steps
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import dsa_cache

    split, page = layout.STAGE_SPLIT, k3.PAGE
    probe = k3.indexer_mode() == "rope-probe"
    tokens = missing = stray = equal = beyond = 0
    for r, (req_id, _) in enumerate(requests):
        block = int(tables[r][0][0])
        held = index[block].cpu().numpy().reshape(len(mla), split, page + 32, 128)
        for m in range(len(mla)):
            for i in range(split):
                positions = (split * core + i) * page + np.arange(page)
                prompt = positions < length
                got = held[m, i, :page]
                exponent = held[m, i, page : page + page // 128].reshape(-1)
                tokens += int(prompt.sum())
                missing += int((exponent[prompt] == 0).sum())
                stray += int((got[~prompt].any(-1) | (exponent[~prompt] != 0)).sum())
                if not probe or not prompt.any():
                    continue
                rope = latent_pages[req_id, m][i][prompt, 512:576]
                rope = np.concatenate((rope, -rope), axis=-1)
                with jax.default_device(jax.devices("cpu")[0]):
                    key = ref.key_from_projection(
                        jnp.asarray(rope.astype(ml_dtypes.bfloat16)),
                        jnp.ones(128),
                        jnp.zeros(128),
                        k3.DSA[0].norm_eps,
                    )
                    want, want_exponent = dsa_cache.quantize_index_keys(key)
                    same, far, _ = key_steps(
                        got[prompt], exponent[prompt], want, want_exponent
                    )
                equal += int(same.sum())
                beyond += int(far.sum())
    harness.log_host(
        f"index keys {tag} stage {stage} core {core}: {tokens - missing}/{tokens} "
        f"prompt tokens keyed, {stray} slots past the prompt not zero"
    )
    record = dict(
        case=tag,
        stage=stage,
        core=core,
        keyed=tokens - missing,
        prompt=tokens,
        stray=stray,
    )
    if probe:
        harness.log_host(
            f"index keys {tag} stage {stage} core {core}: {equal}/{tokens} equal, "
            f"{tokens - equal - beyond} one step, {beyond} beyond"
        )
        record.update(equal=equal, one_step=tokens - equal - beyond, beyond=beyond)
    records.add(**record)


def _rel(got, expected) -> float:
    got = np.asarray(got, np.float64).reshape(-1)
    expected = np.asarray(expected, np.float64).reshape(-1)[: got.size]
    got = got[: expected.size]
    return float(np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-30))


def _shard_reader(mesh):
    """``(stage pool, decode-block ids [n]) -> (recurrent [n, 3, 128, 128],
    conv [n, 3 * 3 * 3 * 128])``: KDA states of a stage pool's decode blocks
    on this core, read as the decoder's importer reads them."""
    import jax.numpy as jnp
    import torch
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels import pool_adapters
    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3.layout import LAYOUT
    from vllm_torchtpu.megakernel.kimi_k3.pools import CONV_ROWS, SSM_ROWS

    rows = LAYOUT.block_shape[0]

    def body(pool, ids):
        decode = pool.reshape(-1, rows, *pool.shape[2:])
        recurrent = pool_adapters.gather_region(
            decode, ids[0], tok0=0, ntok=SSM_ROWS, out_dtype=jnp.float32, out_lanes=128
        )
        conv = pool_adapters.gather_region(
            decode, ids[0], tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
        )
        n = ids.shape[1]
        return (
            recurrent[:, : 3 * 128].reshape(n, 3, 128, 128)[None],
            conv.reshape(n, -1)[:, : 3 * 3 * 3 * 128].astype(jnp.float32)[None],
        )

    op = bridge.register_op(
        "pallas::k3_stage_check_read_shards",
        mesh,
        body,
        (P("k3_sp"), P("k3_sp")),
        (P("k3_sp"), P("k3_sp")),
    )

    def read(pool, ids):
        ids = torch.from_numpy(np.asarray(ids, np.int32)[None]).to("tpu")
        recurrent, conv = op(pool, ids)
        return recurrent[0].cpu().numpy(), conv[0].cpu().numpy()

    return read
