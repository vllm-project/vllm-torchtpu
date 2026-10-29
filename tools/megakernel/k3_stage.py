# SPDX-License-Identifier: Apache-2.0
"""The Kimi-K3 prefill as four pipeline stages, one per host
(``kimi_k3/stage.py``), outside vLLM.

    stage bench --prompts 8192x1,2048x4,8192x1@65536 [--repeats 3]
        [--chain DIR] [--reference DIR] [--trace DIR]
    stage reference --prompts 8192x1,2048x4 --output DIR

``bench`` times each stage's step on every case (the slowest stage sets the
pipeline's throughput). Stages after the first continue from random
residual streams, or with ``--chain`` from the previous stage's (hand-off
files in DIR, so the four stages run one after another); with
``--reference`` the last stage compares each case's first-token logits with
those ``reference`` saved from the slice program (TP32 / SP32) on the same
token ids. Prefixes (``@PREFIX``) are synthetic cached pages, context
parallel over the stage.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from tools.megakernel import bench, harness, k3

STAGES, CORES = 4, 8
VOCAB = 163840
STAGE_WINDOWS = tuple(range(512, 16384 + 1, 512))


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
    for p in (bench, reference, probe, pools, serve):
        p.add_argument(
            "--prompts",
            required=True,
            help="LENxN[@PREFIX],...: N sequences of LEN tokens per window",
        )
        p.add_argument("--seed", type=int, default=0)
    bench.add_argument("--repeats", type=int, default=3)
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


def _descriptor(
    window: int, slots: int, length: int, count: int, prefix: int, page: int
):
    """The packed window's int32 descriptor and its extent: ``count``
    sequences of ``length`` at page-aligned offsets, each continuing a
    ``prefix``-token cached prefix."""
    rows = -(-length // page) * page
    offsets = [i * rows for i in range(count)]
    lengths = [length] * count
    bases = [i * prefix for i in range(count)]
    unused = slots - count
    desc = (
        (offsets + [window] * unused)
        + (lengths + [0] * unused)
        + [0] * slots
        + (bases + [count * prefix] * unused)
        + ([prefix] * count + [0] * unused)
    )
    return np.asarray(desc, np.int32), offsets[-1] + length


def _arena(
    core: int, ranks: int, count: int, prefix: int, page: int, slots: int, capacity: int
):
    """This core's context parallel prefix operands (``CpPrefix``): its
    pages of each sequence's prefix (global page k on core ``k % ranks``),
    as block ids 1.. of a stand-in pool, and each sequence's base and token
    count in them."""
    own, bases, counts, cursor = [], [], [], 0
    pages = prefix // page
    for _ in range(count):
        mine = len(range(core, pages, ranks))
        bases.append(cursor)
        counts.append(mine * page)
        own += list(range(1 + len(own), 1 + len(own) + mine))
        cursor += mine * page
    unused = slots - count
    ids = [len(own)] + own + [0] * (capacity // page - len(own))
    local = bases + [cursor] * unused + counts + [0] * unused
    return np.asarray([ids], np.int32), np.asarray([local], np.int32), len(own)


def _give(path: Path, tensor) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp.npy")
    np.save(staging, tensor.to(torch.bfloat16).cpu().view(torch.int16).numpy())
    staging.rename(path)


def _take(path: Path):
    import torch

    while not path.exists():
        time.sleep(0.1)
    return torch.from_numpy(np.load(path)).view(torch.bfloat16).to("tpu")


def _stage_mesh():
    import torch
    from torch_tpu._internal.pallas import get_pallas_mesh

    rank = harness.rank()
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
    program = StagePrefillProgram(stage, mesh, core, capacity, slots)
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


def run_bench(args, records: harness.Records) -> None:
    import torch

    from vllm_torchtpu.megakernel import bridge
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as kernels
    from vllm_torchtpu.megakernel.kimi_k3.stage import StagePrefillProgram

    cases = _cases(args.prompts)
    stage, core, mesh, group = _stage_mesh()
    page = k3.PAGE
    slots = max(count for _, count, _ in cases)
    capacity = max([count * prefix for _, count, prefix in cases] + [page * CORES])
    capacity = -(-capacity // (page * CORES)) * page * CORES
    program = StagePrefillProgram(stage, mesh, core, capacity, slots, args.moe_path)
    program.load_weights(args.presharded)
    harness.log(
        f"stage {stage}: layers {program.plan['first']}..{program.plan['last'] - 1}, "
        f"groups {program.plan['groups']}"
    )
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
    chain = Path(args.chain) if args.chain else None
    reference = Path(args.reference) if args.reference else None
    for case, (length, count, prefix) in enumerate(cases):
        tag = _tag(length, count, prefix)
        rows = -(-length // page) * page
        window = min(w for w in STAGE_WINDOWS if w >= rows * (count - 1) + length)
        kernels.check_window(window, "stage window", CORES)
        cslots = program.window_slots(window)
        desc, extent = _descriptor(window, cslots, length, count, prefix, page)
        ids, local, _ = _arena(
            core, CORES, count, prefix, page, cslots, capacity // CORES + cslots * page
        )
        local_rows = window // CORES
        order = kernels.sp_order(window, CORES)
        tokens = np.full(window, 0, np.int32)
        for s, seq in enumerate(_tokens(args.seed, case, length, count)):
            tokens[s * rows : s * rows + length] = seq
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
                    [pool] * program.plan["mla_layers"],
                    device["ids"],
                    device["local"],
                    tokens=device["tokens"],
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
                _compare(reference / f"{tag}.npy", logits, count, tag)
            if last and logits is not None and args.save and core == 0:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                np.save(
                    Path(args.save) / f"{tag}.npy", logits.float().cpu().numpy()[:count]
                )
            if last and core == 0 and routes:
                busiest = max(int(r.reshape(-1)[0].cpu()) for r in routes)
                harness.log_host(f"{tag} stage {stage}: busiest chip routes {busiest}")
        ms = statistics.median(times)
        gathered = [None] * (STAGES * CORES)
        torch.distributed.all_gather_object(gathered, (stage, core, ms))
        per_stage = [
            max(ms for s, _, ms in gathered if s == st) for st in range(STAGES)
        ]
        rate = length * count * 1e3 / max(per_stage)
        harness.log(
            f"stage steps {tag} (window {window}): "
            + " / ".join(f"{t:.1f}" for t in per_stage)
            + f" ms; pipeline {rate:.0f} tok/s, one window through the stages "
            f"{sum(per_stage):.1f} ms (repeats on stage 0 core 0: "
            f"{[round(t, 1) for t in times]})"
        )
        records.add(
            case=tag,
            length=length,
            count=count,
            context=prefix,
            window=window,
            stage_ms=[round(t, 2) for t in per_stage],
            ms=round(max(per_stage), 2),
            tokens_per_s=round(rate, 1),
            latency_ms=round(sum(per_stage), 2),
        )


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
        harness.log_host(
            f"{tag} sequence {s}: logits rel-L2 {rel:.4f}, KL {kl:.5f}, argmax "
            f"{int(e.argmax())} (slice) vs {int(g.argmax())} (stages)"
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
        )
        values = logits.float().cpu().numpy()[:count]
        if rank == 0:
            np.save(output / f"{tag}.npy", values)
            harness.log(
                f"reference {tag} (window {window}): "
                f"argmax {values.argmax(axis=-1).tolist()}"
            )
        records.add(case=tag, window=window)


def _requests(case: int, length: int, count: int, seed: int):
    """A case's requests: ``(req_id, token ids)``."""
    return [
        (f"c{case}r{r}", tokens)
        for r, tokens in enumerate(_tokens(seed, case, length, count))
    ]


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
        for req_id, tokens in _requests(case, length, count, args.seed):
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
            finals = kda_blocks_at_length(
                row.block_tables, prefiller._kda_groups, length
            )
            recurrent, conv = writer.read_states(torch.from_numpy(finals).to("tpu"))
            recurrent, conv = recurrent.cpu().numpy(), conv.float().cpu().numpy()
            for k in range(69):
                saved[f"{row.req_id}/kda{k}/recurrent"] = recurrent[0, k, 0]
                saved[f"{row.req_id}/kda{k}/conv"] = conv[0, k, 0]
        harness.log(f"pool reference {_tag(length, count, 0)}: first tokens saved")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f"rank{rank:02d}.npz", **saved)
    records.add(cases=args.prompts, rank=rank)


def _stage_pools(program, page: int, blocks: int, draft_layers: int = 0):
    """Stand-in pools of a stage's layers as vLLM allocates them (stage
    block: four decode blocks; pool i holds the stage's MLA layer i and its
    KDA layers 3i..3i+2; then ``draft_layers`` MLA DSpark draft pools) and
    the KV cache groups: the MLA layers' (``8 * 4 * page``-token blocks) and
    three KDA groups (``4 * page``)."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.kimi_k3 import layout

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
    (TP32 rank ``4 core + g``), as relative L2; and the first tokens."""
    import torch

    from vllm_torchtpu.megakernel import program as mk
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.kimi_k3 import layout
    from vllm_torchtpu.megakernel.kimi_k3 import program as k3_program
    from vllm_torchtpu.megakernel.kimi_k3.pools import (
        kda_blocks_at_length,
    )
    from vllm_torchtpu.megakernel.kimi_k3.stage import (
        DRAFT_TAP,
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
    serving.warmup()
    reader = _shard_reader(mesh)
    chain = Path(args.chain)
    references: dict = {}
    next_block = 1
    for case, (length, count, prefix) in enumerate(cases):
        if prefix:
            harness.fail("serve-check: fresh prompts only")
        requests = _requests(case, length, count, args.seed)
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
                names += [DRAFT_TAP] if serving.tapped else []
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
        worst = []
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
        if serving.last and core == 0:
            expected = np.load(Path(args.reference) / "rank00.npz")[f"case{case}/first"]
            harness.log_host(
                f"{tag} first tokens {first.tolist()} (slice {expected.tolist()})"
            )
        records.add(
            case=tag, stage=stage, core=core, worst=worst[0][0] if worst else 0.0
        )
        torch.distributed.barrier(group=group)


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
