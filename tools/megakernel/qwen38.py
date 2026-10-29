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
"""The Qwen3.8 prefill stage without a vLLM server.

    python -m tools.megakernel qwen38 bench --prompts 8192x1,2048x4,8192x1@32768
    python -m tools.megakernel qwen38 probe --prompt "The capital of France is"
    python -m tools.megakernel qwen38 decode --presharded DIR --batches 1,4,16

``probe`` loads the checkpoint (``--layers N``: each stage's first N layers)
and runs one prompt, printing every layer's output range on each stage's
first core: stage 0 embeds the prompt, each later stage takes the previous
one's rows through files in ``--handoff`` (random rows without it), the last
stage prints its greedy token. ``--check L,...`` compares those layers'
mixer and MoE updates with ``qwen38_check``'s NumPy layer on the same input.

Each host runs its pipeline stage (vLLM's PP4 partition, its 8 cores one PCP
group) over pools shaped as vLLM allocates them for PP4 x PCP8 (the unified
HND pools, ``--block-size 768``, 256-token kernel pages), with synthetic
weights (``--seed``; ``--checkpoint`` loads real ones). Stages after the first
take random rows as their hand-off (``--chain``: the previous stage's output
rows, through files), so every stage times its own step: ``LENxN`` is ``N``
prompts of ``LEN`` new tokens, ``@PREFIX`` after that many cached tokens (on
fresh pages); the median of ``--repeats`` steps after a warm-up, per stage.

``decode`` runs the decoder (TP32, DCP8 attention) on the rank cache
``--presharded`` over pools shaped as vLLM allocates them for it, each batch's
requests decoding ``--steps`` tokens from position ``--context`` (zero KV and
states before it), and reports the median step time per batch.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np

from tools.megakernel import harness
from tools.megakernel.bench import trace

STAGES = 4
CORES = 8
INTERLEAVE = 256
MANAGER_BLOCK = 768  # --block-size 768: three 256-token kernel pages
PAGE = 256
POOL_PAGE = (8, 64, 4, PAGE)  # [2 * kv heads, head_dim / 4, 4 FP8 per word, page]


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="qwen_command", required=True)
    bench = sub.add_parser("bench", help="each stage's step time")
    bench.add_argument("--prompts", default="8192x1")
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--layers", type=int, default=0, help="per stage (0: all)")
    bench.add_argument("--max-num-batched-tokens", type=int, default=32768)
    bench.add_argument("--max-num-seqs", type=int, default=64)
    bench.add_argument("--max-model-len", type=int, default=139264)
    bench.add_argument(
        "--trace",
        default="",
        help="save an xprof trace of each case's last repeat here (TRACE_RANKS)",
    )
    bench.add_argument(
        "--checkpoint", default="", help="the checkpoint's weights instead of synthetic"
    )
    bench.add_argument(
        "--chain",
        default="",
        help="directory through which each later stage takes the previous one's rows",
    )
    bench.add_argument(
        "--token-file",
        default="",
        help="prompt tokens (.npy, tiled to each length) instead of random ids",
    )

    decode = sub.add_parser("decode", help="the decoder's step time")
    decode.add_argument("--presharded", required=True, help="the decoder's rank cache")
    decode.add_argument("--batches", default="1,2,4,8,16")
    decode.add_argument("--context", type=int, default=8192)
    decode.add_argument("--steps", type=int, default=32)
    decode.add_argument("--max-model-len", type=int, default=139264)

    probe = sub.add_parser("probe", help="each layer's output range, real weights")
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--prompt", default="The capital of France is")
    probe.add_argument(
        "--token-file", default="", help="the prompt as token ids (.npy) instead"
    )
    probe.add_argument("--layers", type=int, default=0, help="per stage (0: all)")
    probe.add_argument("--handoff", default="", help="directory for the stages' rows")
    probe.add_argument(
        "--check",
        default="",
        help="layers (comma-separated) to compare with qwen38_check's NumPy layer",
    )
    probe.add_argument("--max-num-batched-tokens", type=int, default=32768)
    probe.add_argument("--max-num-seqs", type=int, default=64)
    probe.add_argument("--max-model-len", type=int, default=139264)


def environment() -> dict[str, str]:
    """What the server's recipe sets for the pools' layout."""
    return {
        "VLLM_KV_CACHE_LAYOUT": "HND",
        "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL": "1",
    }


class Pools:
    """vLLM's PP4 x PCP8 unified pools of one stage: raw pool ``i`` holds the
    ``i``-th layer of every KV cache group (three GDN groups, one attention
    group), as vLLM's hybrid allocator lays them out."""

    def __init__(self, layers: range, blocks: int, max_model_len: int) -> None:
        import torch

        from vllm_torchtpu.megakernel.program import KVGroup
        from vllm_torchtpu.megakernel.qwen38.program import is_full_attention

        attention = [i for i in layers if is_full_attention(i)]
        gdn = [
            [i for i in layers if not is_full_attention(i) and i % 4 == g]
            for g in range(3)
        ]
        count = max(len(attention), *(len(group) for group in gdn))
        self.raw = [
            torch.zeros((blocks * 3, *POOL_PAGE), dtype=torch.float8_e4m3fn).to("tpu")
            for _ in range(count)
        ]
        self.caches = {}
        for g, group in enumerate(gdn):
            for i, layer in enumerate(group):
                self.caches[f"model.layers.{layer}.linear_attn"] = [self.raw[i]]
        for i, layer in enumerate(attention):
            self.caches[f"model.layers.{layer}.self_attn.attn"] = self.raw[i]
        mamba_block = max_model_len * CORES
        self.groups = [
            KVGroup(
                tuple(f"model.layers.{i}.linear_attn" for i in group), mamba_block, 1
            )
            for group in gdn
        ] + [
            KVGroup(
                tuple(f"model.layers.{i}.self_attn.attn" for i in attention),
                MANAGER_BLOCK,
                -(-max_model_len // (PAGE * CORES)),
            )
        ]
        self.blocks = blocks
        self._next = 1  # block 0 is the null block

    def fresh(self, count: int) -> list[int]:
        if self._next + count > self.blocks:
            raise RuntimeError("the bench's pools are full")
        ids = list(range(self._next, self._next + count))
        self._next += count
        return ids

    def tables(self, tokens: int) -> tuple[np.ndarray, ...]:
        """A request's block-table rows: one state block per GDN group and
        the attention group's kernel pages (three per manager block)."""
        widths = [g.table_width for g in self.groups]
        rows = [np.zeros(w, np.int32) for w in widths]
        for g in range(3):
            rows[g][0] = self.fresh(1)[0]
        pages = -(-tokens // (PAGE * CORES))
        for k, block in enumerate(self.fresh(-(-pages // 3))):
            for j in range(3):
                if 3 * k + j < widths[3]:
                    rows[3][3 * k + j] = 3 * block + j
        return tuple(rows)


def _stage(layers: int = 0):
    import torch
    from torch_tpu._internal.pallas import get_pallas_mesh
    from vllm.distributed.utils import get_pp_indices

    from vllm_torchtpu.megakernel.qwen38.program import NUM_LAYERS

    rank = harness.rank()
    groups = [
        torch.distributed.new_group(list(range(s * CORES, (s + 1) * CORES)))
        for s in range(STAGES)
    ]
    stage = rank // CORES
    mesh = get_pallas_mesh(("pcp",), process_group=groups[stage])
    first, last = get_pp_indices(NUM_LAYERS, stage, STAGES)
    span = range(first, min(last, first + layers) if layers else last)
    return stage, mesh, span, groups[stage]


def _config(args):
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig

    return MegakernelEngineConfig(
        role="prefill",
        tensor_parallel_size=1,
        decode_context_parallel_size=1,
        block_size=MANAGER_BLOCK,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",
        cp_kv_cache_interleave_size=INTERLEAVE,
        pipeline_parallel_size=STAGES,
        prefill_context_parallel_size=CORES,
    )


def probe(args) -> None:
    import torch
    from transformers import AutoTokenizer

    from vllm_torchtpu.megakernel.program import RequestInput, StepInput
    from vllm_torchtpu.megakernel.qwen38.program import HIDDEN_SIZE
    from vllm_torchtpu.megakernel.qwen38.rows import Span, plan_rows
    from vllm_torchtpu.megakernel.qwen38.stage import QwenPrefillStage

    if args.token_file:
        tokens = np.load(args.token_file).tolist()
    else:
        tokens = AutoTokenizer.from_pretrained(args.checkpoint)(args.prompt)[
            "input_ids"
        ]
    stage_index, mesh, layers, _ = _stage(args.layers)
    core = harness.rank() % CORES
    stage = QwenPrefillStage(
        _config(args),
        Path(args.checkpoint),
        layers=layers,
        first_stage=stage_index == 0,
        last_stage=stage_index == STAGES - 1,
        mesh=mesh,
        core=core,
    )
    started = time.perf_counter()
    stage.load_weights()
    harness.log_host(
        f"stage {stage_index} layers {layers}: weights in "
        f"{time.perf_counter() - started:.1f} s"
    )
    # The null block, a state block per GDN group, the prompt's attention blocks.
    pages = -(-len(tokens) // (PAGE * CORES))
    pools = Pools(layers, 4 + -(-pages // 3), args.max_model_len)
    stage.bind_kv_caches(pools.caches, pools.groups)
    request = RequestInput(
        req_id="probe",
        computed=0,
        scheduled=len(tokens),
        token_ids=np.asarray(tokens, np.int32),
        num_tokens=len(tokens),
        num_prompt_tokens=len(tokens),
        block_tables=pools.tables(len(tokens)),
    )
    step = StepInput(requests=[request])
    rows = stage.rows(step)
    plan = plan_rows([Span(0, len(tokens))], CORES, INTERLEAVE, rows)
    meta = stage._metadata([request], plan)
    stage._seed_states([request])
    if stage.first_stage:
        h = stage._embed([request], plan)
    elif args.handoff:
        path = Path(args.handoff) / f"stage{stage_index}-core{core}.npy"
        h = stage._land(_take(path).to("tpu"))
    else:
        generator = torch.Generator().manual_seed(stage_index)
        h = stage._land(
            torch.randn((rows, HIDDEN_SIZE), generator=generator)
            .to(torch.bfloat16)
            .to("tpu")
        )
    live = plan.request[plan.core_rows(core)] >= 0

    def report(label, h):
        value = h.float().cpu().numpy()[live]
        if core == 0 and value.size:
            harness.log_host(
                f"stage {stage_index} {label}: max {np.nanmax(np.abs(value)):.4g} "
                f"nan {int(np.isnan(value).sum())} inf {int(np.isinf(value).sum())} "
                f"rows {value.shape[0]}"
            )

    report("input", h)
    checks = {int(x) for x in args.check.split(",") if x}
    positions = plan.position[plan.core_rows(core)][live]
    ckpt = None
    for layer in layers:
        if layer in checks:
            before = h.float().cpu().numpy()[live]
            # The stage's one-op layer on the same input first (a fresh request's
            # pool writes are the same both times).
            fused = stage._run_layer(layer, rows, h, meta).float().cpu().numpy()
            mixed, h = _split_layer(stage, layer, rows, h, meta)
            split = h.float().cpu().numpy()
            if core == 0:
                harness.log_host(
                    f"layer {layer}: one op vs mixer + MoE ops: max |diff| "
                    f"{np.max(np.abs(fused - split)):.4g} over all rows, "
                    f"{np.max(np.abs(fused[live] - split[live])):.4g} over live rows"
                )
            report(f"layer {layer} mixer", mixed)
            if core == 0 and live.any():
                from tools.megakernel import qwen38_check

                ckpt = ckpt or qwen38_check.Checkpoint(Path(args.checkpoint))
                full = (layer + 1) % 4 == 0
                got_mixed = mixed.float().cpu().numpy()[live]
                got_out = h.float().cpu().numpy()[live]
                # Each part from the stage's own input to it.
                want_mixed = qwen38_check.mixer(
                    ckpt, layer, before, positions, stage.dims, full
                )
                want_out = qwen38_check.sparse(ckpt, layer, got_mixed, stage.dims)

                def error(got, want, base):
                    scale = np.max(np.abs(want - base))
                    rows = np.max(np.abs(got - want), axis=-1) / max(scale, 1e-30)
                    return " ".join(f"{r:.2g}" for r in rows)

                harness.log_host(
                    f"check layer {layer} (|mixer| "
                    f"{np.max(np.abs(want_mixed - before)):.3g}, |moe| "
                    f"{np.max(np.abs(want_out - got_mixed)):.3g}): rel err per row: "
                    f"mixer {error(got_mixed, want_mixed, before)}; "
                    f"MoE {error(got_out, want_out, got_mixed)}"
                )
        else:
            h = stage._run_layer(layer, rows, h, meta)
        report(f"layer {layer}", h)
    if stage.last_stage:
        token = int(stage._first_tokens([request], [request], plan, h).cpu()[0, 0])
        if core == 0:
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
            harness.log_host(f"greedy token {token} {tokenizer.decode([token])!r}")
            if checks:
                from tools.megakernel import qwen38_check

                ckpt = ckpt or qwen38_check.Checkpoint(Path(args.checkpoint))
                last = h.float().cpu().numpy()[live][-1:]
                logits = qwen38_check.logits(ckpt, last, stage.dims)[0]
                top = np.argsort(-logits)[:5]
                harness.log_host(
                    "reference top-5 from this hidden: "
                    + ", ".join(f"{int(t)} {tokenizer.decode([int(t)])!r}" for t in top)
                )
    elif args.handoff:
        path = Path(args.handoff) / f"stage{stage_index + 1}-core{core}.npy"
        _give(path, h)


def _give(path: Path, rows) -> None:
    """Hand the BF16 ``rows`` to the next stage, as their 16-bit patterns
    (renamed into place: never read half-written)."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp.npy")
    np.save(staging, rows.to(torch.bfloat16).cpu().view(torch.int16).numpy())
    staging.rename(path)


def _take(path: Path):
    """The previous stage's BF16 rows, once :func:`_give` has written them."""
    import torch

    while not path.exists():
        time.sleep(0.1)
    return torch.from_numpy(np.load(path)).view(torch.bfloat16)


def _split_layer(stage, layer, rows, h, meta):
    """``stage._run_layer`` as two ops: ``(after the mixer, after the MoE)``."""
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel.bridge import register_global_op
    from vllm_torchtpu.megakernel.qwen38 import layers as qlayers
    from vllm_torchtpu.megakernel.qwen38.program import is_full_attention

    full = is_full_attention(layer)
    g = stage._geometry(rows)
    w = stage.weights.layers[layer]
    names = tuple(sorted(w))
    sharded = {"conv", "a_log", "dt_bias"} | {
        f"{w}{part}" for w in ("w13", "w2") for part in ("", "_channel", "_ratio")
    }
    weight_specs = tuple(P("pcp") if n in sharded else P() for n in names)
    state = (
        meta["block_tables"] if full else meta["state_indices"][stage._gdn_slot(layer)]
    )

    plan = stage.layer_plan(layer, rows, meta)

    def mixer(h, pool, *args):
        weights = dict(zip(names, args[: len(names)], strict=True))
        positions, qsl, seq_lens, distribution, state = args[len(names) :][:5]
        part = {
            "positions": positions,
            "query_start_loc": qsl,
            "seq_lens": seq_lens,
            "distribution": distribution,
            "block_tables": state,
            "state_indices": state,
        }
        shared = args[len(names) + 5 :]
        if full:
            return qlayers.attention_mixer(
                g, stage.mesh, h, weights, pool, part, shared
            )
        return qlayers.gdn_mixer(g, stage.mesh, h, weights, pool, part, shared)

    def moe(h, *args):
        weights = dict(zip(names, args, strict=True))
        return (qlayers.moe_block(g, stage.mesh, h, weights),)

    mixer_op = register_global_op(
        f"pallas::qwen38_probe_mixer_{layer}_r{rows}",
        stage.mesh,
        mixer,
        (P("pcp"), P("pcp"))
        + weight_specs
        + (P("pcp"), P(), P(), P(), P())
        + ((P("pcp"),) * 6 + (P(),) if full else (P(),) * len(plan)),
        (P("pcp"), P("pcp")),
        donate_argnums=(1,),
    )
    moe_op = register_global_op(
        f"pallas::qwen38_probe_moe_{layer}_r{rows}",
        stage.mesh,
        moe,
        (P("pcp"),) + weight_specs,
        (P("pcp"),),
    )
    pool = stage._pools[layer]
    h, new_pool = mixer_op(
        h,
        pool,
        *(w[n] for n in names),
        meta["positions"],
        meta["query_start_loc"],
        meta["seq_lens"],
        meta["distribution"],
        state,
        *plan,
    )
    pool.set_(new_pool)
    (out,) = moe_op(h, *(w[n] for n in names))
    return h, out


def decode(args) -> None:
    import torch

    from vllm_torchtpu.megakernel.bridge import materialize
    from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
    from vllm_torchtpu.megakernel.program import KVGroup, RequestInput, StepInput
    from vllm_torchtpu.megakernel.qwen38.decode import QwenDecoder
    from vllm_torchtpu.megakernel.qwen38.program import NUM_LAYERS, is_full_attention

    batches = [int(b) for b in args.batches.split(",")]
    config = MegakernelEngineConfig(
        role="decode",
        tensor_parallel_size=32,
        decode_context_parallel_size=CORES,
        block_size=MANAGER_BLOCK,
        max_num_seqs=max(batches),
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(batches),
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",
        cp_kv_cache_interleave_size=PAGE,
    )
    decoder = QwenDecoder(config, Path(args.presharded), tp_rank=harness.rank())
    started = time.perf_counter()
    decoder.load_weights()
    harness.log(
        f"rank cache weights in {time.perf_counter() - started:.1f} s; {_hbm()}"
    )
    # vLLM's pools for the decoder: raw pool i holds attention layer i and the
    # i-th layer of each GDN group; state blocks are manager blocks of three
    # kernel pages, attention tables list kernel pages.
    attention = [i for i in range(NUM_LAYERS) if is_full_attention(i)]
    gdn = [
        [i for i in range(NUM_LAYERS) if not is_full_attention(i) and i % 4 == g]
        for g in range(3)
    ]
    requests = max(batches)
    pages = -(-(args.context + args.steps + 1) // (PAGE * CORES))
    width = -(-args.max_model_len // (PAGE * CORES))
    state_pages = 3 * (1 + 3 * requests)
    per_block = MANAGER_BLOCK // PAGE
    kernel_pages = -(-(state_pages + requests * pages) // per_block) * per_block
    page_shape = (2, decoder.layout.config.head_dim // 4, 4, PAGE)
    raw = [
        torch.zeros((kernel_pages, *page_shape), dtype=torch.float8_e4m3fn).to("tpu")
        for _ in attention
    ]
    caches = {
        f"model.layers.{i}.self_attn.attn": raw[o] for o, i in enumerate(attention)
    }
    for group in gdn:
        for o, i in enumerate(group):
            caches[f"model.layers.{i}.linear_attn"] = [raw[o]]
    groups = [
        KVGroup(
            tuple(f"model.layers.{i}.linear_attn" for i in group),
            args.max_model_len * CORES,
            1,
        )
        for group in gdn
    ] + [
        KVGroup(
            tuple(f"model.layers.{i}.self_attn.attn" for i in attention), PAGE, width
        )
    ]
    decoder.bind_kv_caches(caches, groups)
    decoder.warmup()
    harness.sync()
    rng = np.random.default_rng(0)

    def tables(r: int) -> tuple[np.ndarray, ...]:
        kv = np.zeros(width, np.int32)
        kv[:pages] = state_pages + r * pages + np.arange(pages)
        return (*(np.asarray([1 + 3 * r + g], np.int32) for g in range(3)), kv)

    for batch in batches:
        tokens = rng.integers(0, 200000, (batch, args.context + args.steps + 1)).astype(
            np.int32
        )
        times, finished = [], ()
        for step in range(args.steps):
            position = args.context + step
            rows = [
                RequestInput(
                    req_id=f"b{batch}-{r}",
                    computed=position,
                    scheduled=1,
                    token_ids=tokens[r],
                    num_tokens=position + 1,
                    num_prompt_tokens=args.context,
                    block_tables=tables(r),
                )
                for r in range(batch)
            ]
            harness.sync()
            begin = time.perf_counter()
            output = decoder.step(StepInput(requests=rows, finished=finished))
            assert output.tokens is not None
            materialize(output.tokens)
            finished = ()
            if step:  # the first imports the requests' states
                times.append((time.perf_counter() - begin) * 1e3)
        decoder.step(
            StepInput(requests=[], finished=[f"b{batch}-{r}" for r in range(batch)])
        )
        median = statistics.median(times)
        last = output.tokens[:batch, 0].tolist()
        harness.log(
            f"decode b{batch} @{args.context}: step {median:.2f} ms = "
            f"{batch * 1e3 / median:.0f} tok/s; last tokens {last}"
        )
    harness.log(_hbm())


def run(args) -> None:
    if args.qwen_command == "probe":
        return probe(args)
    if args.qwen_command == "decode":
        return decode(args)
    import torch

    from vllm_torchtpu.distributed.pp_wave import HIDDEN_STATES
    from vllm_torchtpu.megakernel.bridge import materialize
    from vllm_torchtpu.megakernel.program import RequestInput, StepInput
    from vllm_torchtpu.megakernel.qwen38.program import HIDDEN_SIZE
    from vllm_torchtpu.megakernel.qwen38.stage import QwenPrefillStage

    cases = []
    for part in filter(None, args.prompts.split(",")):
        prompt, _, prefix = part.partition("@")
        length, count = (int(x) for x in prompt.split("x"))
        cases.append((length, count, int(prefix) if prefix else 0))
    stage_index, mesh, layers, group = _stage(args.layers)
    core = harness.rank() % CORES
    config = _config(args)
    stage = QwenPrefillStage(
        config,
        Path(args.checkpoint),
        layers=layers,
        first_stage=stage_index == 0,
        last_stage=stage_index == STAGES - 1,
        mesh=mesh,
        core=core,
        synthetic_seed=None if args.checkpoint else args.seed,
    )
    started = time.perf_counter()
    stage.load_weights()
    harness.log(
        f"{'checkpoint' if args.checkpoint else 'synthetic'} weights in "
        f"{time.perf_counter() - started:.1f} s; {_hbm()}"
    )

    def blocks_per_request(tokens: int) -> int:  # three GDN states + KV blocks
        pages = -(-tokens // (PAGE * CORES))
        return 3 + -(-pages // 3)

    blocks = 2 + (args.repeats + 1) * sum(
        count * blocks_per_request(length + prefix) for length, count, prefix in cases
    )
    pools = Pools(layers, blocks, args.max_model_len)
    stage.bind_kv_caches(pools.caches, pools.groups)
    stage.warmup()
    harness.sync()
    rng = np.random.default_rng(0)
    text = np.load(args.token_file).astype(np.int32) if args.token_file else None

    def prompt(tokens: int, request: int) -> np.ndarray:
        """Random ids, or the token file from a request-dependent offset (real
        text routes its MoE unlike random ids)."""
        if text is None:
            return rng.integers(0, 200000, tokens).astype(np.int32)
        start = request * 997 % text.size
        return np.resize(np.roll(text, -start), tokens)

    generator = torch.Generator().manual_seed(stage_index)
    for length, count, prefix in cases:
        tag = f"p{length}x{count}" + (f"c{prefix}" if prefix else "")
        times = []
        for repeat in range(args.repeats + 1):
            requests = [
                RequestInput(
                    req_id=f"{tag}-{repeat}-{r}",
                    computed=prefix,
                    scheduled=length,
                    token_ids=prompt(prefix + length, r),
                    num_tokens=prefix + length,
                    num_prompt_tokens=prefix + length,
                    block_tables=pools.tables(prefix + length),
                )
                for r in range(count)
            ]
            step = StepInput(requests=requests)
            harness.sync()
            chain = Path(args.chain) / f"{tag}-{repeat}-stage{{}}-core{core}.npy"
            if stage_index > 0 and args.chain:
                hidden = _take(Path(str(chain).format(stage_index)))
            elif stage_index > 0:
                # Distinct rows: identical ones would all route to the same experts.
                rows = stage.rows(step)
                hidden = torch.randn((rows, HIDDEN_SIZE), generator=generator)
            if stage_index > 0:
                step.hidden = {HIDDEN_STATES: hidden.to(torch.bfloat16).to("tpu")}
                materialize(step.hidden[HIDDEN_STATES])
            if args.chain:
                # The stage's cores take their rows at different times: start together.
                torch.distributed.barrier(group=group)
            with trace(args.trace, tag, repeat == args.repeats):
                begin = time.perf_counter()
                output = stage.step(step)
                if output.tokens is not None:
                    materialize(output.tokens)
                else:
                    assert output.hidden is not None
                    materialize(output.hidden[HIDDEN_STATES])
                elapsed = (time.perf_counter() - begin) * 1e3
            if args.chain and output.hidden is not None:
                _give(
                    Path(str(chain).format(stage_index + 1)),
                    output.hidden[HIDDEN_STATES],
                )
            if repeat:  # the first is a warm-up of this shape
                times.append(elapsed)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, statistics.median(times))
        if harness.rank() == 0:
            per_stage = [
                statistics.median(gathered[s * CORES : (s + 1) * CORES])
                for s in range(STAGES)
            ]
            tokens = length * count
            harness.log(
                f"{tag}: stage step ms "
                + " / ".join(f"{t:.1f}" for t in per_stage)
                + f"; slowest {max(per_stage):.1f} ms = {tokens / max(per_stage):.1f}K "
                "tok/s through a full pipeline"
            )
    harness.log(_hbm())


def _hbm() -> str:
    """This core's HBM in use and at its peak, of its limit."""
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    gib = [
        stats.get(key, 0) / 2**30
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
    ]
    return "HBM {:.1f} GiB in use, peak {:.1f}, of {:.1f}".format(*gib)
