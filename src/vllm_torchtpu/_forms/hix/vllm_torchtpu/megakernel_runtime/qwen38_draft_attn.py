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
"""q38 L1-H3 (H3-DRAFT-ATTN-D14, VLLM_TPU_MK_DRAFT_ATTN): the MTP draft's fused-loop QSA attention as one
L = 1 megakernel call per pass (`megakernel/wholestep/qsa_draft.py`).

`build_draft_attn_call` returns `(fn, in_specs, out_specs, donate)` at `B` rows (one row per request, the
fused loop's pass): `fn(block, pool, shared, main_table, positions, starts, context, trig, dcp_rank, *weights)`
derives the kernel's extras with the stock addressing (`pooled_qsa.shared_blocks` of the carried target
selection, the request's context / lengths, the draft's main table and head-pair slot, the RoPE rows),
runs the kernel, then writes the step's main K / V through the stock writer
(`pooled_qsa.qsa_cache_writes(index_state=False)`: DRAFT-1's draft write, the same bytes for the same
K / V projections). Returns `(attention, pool)`: the gated attention before o_proj (the d14mtp tail's
`attn` operand) and the updated pool.

Draft step 0 (rows over a verify window) and the all-token draft call stay on the stock op; the kernel
itself also serves `T = 4` (tests/megakernel/test_l1h3_draft_attn.py), the wiring does not yet.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.wholestep import qsa_draft as qd
from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
from vllm_torchtpu.megakernel.wholestep.qsa_full_c import KR_NEW

from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable
from .qwen38_segment_call import segment_tiles

logger = init_logger(__name__)

RATIO = 4
# The SMEM table window is [requests, columns] int32; the deviceless AOT (l1h3-aot-a3) compiled B <= 64 at 512 columns
# (262144 tokens of 512-token blocks). Larger tables refuse (stock draft attention).
MAX_TABLE_COLS = 512
ATTACH_LINE = "Qwen3.8 MTP draft attention kernel attached (q38 L1-H3): loop rows buckets %s, main-KV DCP %d"
# q38 ATTNWIRE: the kind's row limit (the megakernel geometry check B <= M <= 64; l1h3-aot-a4 compiled B <= 64). A draft whose
# loop buckets go past it refuses (stock draft attention), loudly.
MAX_LOOP_ROWS = 64
AGREE_LINE = ("Qwen3.8 MTP draft attention agreement (q38 ATTNWIRE): attached %d, refused %d of %d TP ranks; loop rows buckets %s "
              "(%s)")
_LOGGED: set[str] = set()


def _once(key: str, msg: str, *args: Any) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


def draft_indexed(max_groups: int, pages_per_seq: int, dcp: int) -> dict[str, Any]:
    """The kind's block options: D14's served union path over HBM tables with the request loop."""
    return dict(max_groups=max_groups, pages_per_seq=pages_per_seq, hbm_tables=True, per_request_scores=True,
                union_attn=True, loop_requests=True, union_phys=False, dcp=dcp)


def build_draft_attn_call(
    mesh: Any,
    g: Any,
    B: int,
    *,
    budget: int,
    max_groups: int,
    pages_per_seq: int,
    dcp: int,
    main_slot: int,
    k_scale: float,
    v_scale: float,
    norm_eps: float,
    rotary_dim: int,
    token_topk: int,
    write: bool = True,
    interpret: bool = False,
    tiles: Any = None,
) -> tuple[Any, tuple, tuple, tuple]:
    R, T = mesh.size, 1
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=segment_tiles(B) if tiles is None else tiles, interpret=interpret)
    spec = qd.draft_attn_spec(g, R, budget=budget, indexed=draft_indexed(max_groups, pages_per_seq, dcp))
    step, info = kernel.make_step(mesh, spec, ex)
    p = info["plan"]
    names = tuple(p.w_names)
    extras = kernel.extra_inputs(spec, info["exec"])
    assert [x.name for x in extras] == qd.EXTRA_NAMES, [x.name for x in extras]
    M, Hc = ex.M, spec.config.H
    nreq = M // T
    KU = -(-(T * budget) // 128) * 128
    tp, rep = P("tp"), P()

    def req_rows(starts):
        """Kernel row q = request q (T = 1): its token row `starts[q]` (rows of idle requests: none)."""
        n = len(starts) - 1
        live = (starts[1:] > starts[:-1])[:nreq]
        idx = jnp.clip(starts[:-1], 0, B - 1).astype(jnp.int32)[:nreq]
        pad_n = max(0, nreq - n)
        return jnp.pad(live, (0, pad_n)), jnp.pad(idx, (0, pad_n))

    def per_rank_inputs(block, pool, shared, main_table, positions, starts, context, trig, rank):
        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import shared_blocks

        n = len(context)
        live, idx = req_rows(starts)
        ctx = jnp.pad(context.astype(jnp.int32), (0, max(0, nreq - n)))[:nreq]
        lens = jnp.where(live, ctx + 1, 0)
        sel = jnp.pad(shared.astype(jnp.int32), ((0, max(0, nreq - shared.shape[0])), (0, 0)), constant_values=-1)[:nreq]
        blocks = shared_blocks(sel, lens, RATIO, token_topk)
        sh, rows = qd.shared_rows(blocks, lens, ctx, live, T=T, KU=KU, ratio=RATIO, page_size=pool.shape[1],
                                  dcp=dcp, dcp_rank=rank.reshape(()))
        main = main_table.reshape(n, -1).astype(jnp.int32)
        main = jnp.pad(main, ((0, max(0, nreq - n)), (0, max(0, pages_per_seq - main.shape[1]))))[:nreq, :pages_per_seq]
        pos = jnp.where(live, positions.astype(jnp.int32)[idx], 0)
        hid = jnp.where(live[:, None], block[idx], 0).astype(block.dtype)
        rope = jnp.pad(trig[pos].astype(jnp.float32), ((0, M - nreq), (0, 0)))
        return (
            jnp.pad(hid, ((0, M - nreq), (0, Hc - hid.shape[1]))),  # the core width (`qd.core_width`)
            pool,
            lens // RATIO,
            lens % RATIO,
            main[None],
            jnp.asarray([max(main_slot, 0)], jnp.int32),
            jnp.asarray([k_scale, v_scale], jnp.float32),
            rope,
            ctx,
            sh,
            rows,
        )

    x_specs = tuple(tp if x.memory_space == "hbm" else rep for x in extras)
    inputs = jax.shard_map(per_rank_inputs, mesh=mesh, in_specs=(tp, tp, tp, rep, rep, rep, rep, rep, tp),
                           out_specs=(tp,) + x_specs, check_vma=False)

    def per_rank_writes(pool, kv, kr, att, nk, main_table, positions, starts, context, trig, rank):
        from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import qsa_cache_writes
        from vllm_torchtpu.kernels.qwen4_exp.qsa import produce_kv

        n = len(context)
        live, idx = req_rows(starts)
        tok = jnp.where(live, idx, B)  # request-major kernel rows -> the step's token rows (idle: dropped)

        def to_tokens(x):
            return jnp.zeros((B,) + x.shape[1:], x.dtype).at[tok].set(x[:nreq], mode="drop")

        out = to_tokens(att[0])
        if not write:
            return out, pool
        k, v = produce_kv(to_tokens(kr[0]), to_tokens(kv[0, :, kv.shape[-1] // 2 :]), nk, positions, trig,
                          norm_eps, rotary_dim)
        pool, _, _, _ = qsa_cache_writes(
            pool, k, v, None, positions, None, trig, starts, context,
            jnp.zeros((n,), jnp.int32), main_table.reshape(n, -1), None,
            compress_ratio=RATIO, norm_eps=norm_eps, rotary_dim=rotary_dim, k_scale=k_scale, v_scale=v_scale,
            main_slot=main_slot, max_query_len=1, dcp=(rank.reshape(1), dcp) if dcp > 1 else None,
            index_state=False,
        )
        return out, pool

    writes = jax.shard_map(per_rank_writes, mesh=mesh, in_specs=(tp, tp, tp, tp, rep, rep, rep, rep, rep, rep, tp),
                           out_specs=(tp, tp), check_vma=False)
    pos0 = jnp.zeros((M,), jnp.int32)
    n_w = len(names)

    def fn(block, pool, shared, main_table, positions, starts, context, trig, rank, nk, *weights):
        hid, *xs = inputs(block, pool, shared, main_table, positions, starts, context, trig, rank)
        _, outs = step(pos0, hid, dict(zip(names, weights)), {}, *xs)
        return writes(pool, outs[KV_NEW], outs[KR_NEW], outs[qd.ATTN_OUT], nk, main_table, positions, starts,
                      context, trig, rank)

    in_specs = (tp, tp, tp, rep, rep, rep, rep, rep, tp, rep) + (tp,) * n_w
    return fn, in_specs, (tp, tp), (1,)


def pack_draft_attn_operands(attn: Any, g: Any, spec: Any, ex: ExecSpec) -> list[torch.Tensor]:
    """This rank's kind operands in `plan.w_names` order: qkvg (the draft's q_proj per-head `[q | gate]`
    columns, then its K and V head), the smalls (q / k norms), the unused final norm."""
    import torch.nn.functional as F

    from .qwen38_wholestep_weights import own_operand, qkvg_operand

    p = planner.plan(spec, ex)
    dev = attn.q_norm.weight.device
    out: list[torch.Tensor] = []
    for name in p.w_names:
        if name == "smalls":
            sm = torch.zeros((1, 1, p.smalls_width), dtype=torch.float32, device=dev)
            for key, val in (("qsa_qn", attn.q_norm.weight.data), ("qsa_kn", attn.k_norm.weight.data)):
                o, width = p.smalls_layout[key]
                sm[0, 0, o : o + val.numel()] = val.reshape(-1).float()
            out.append(sm)
        elif name == "final_norm":  # unused (no residual out is read); the core width
            out.append(torch.zeros((1, spec.config.H), dtype=torch.float32, device=dev))
        else:
            assert name == "qkvg", name
            t = qkvg_operand(attn, g, ex.R)[None]
            nt, tk = p.nt[name], p.tk[name]
            if t.shape[-2] < nt * tk:
                t = F.pad(t, (0, 0, 0, nt * tk - t.shape[-2]))
            out.append(t.reshape(-1, tk, t.shape[-1]))
    return [own_operand(t) for t in out]


class DraftAttnKind:
    """The kind's torch op; one kernel build per rows count (the loop's request capacity)."""

    def __init__(self, mesh: Any, g: Any, opts: dict[str, Any], buckets: tuple[int, ...] | None = None) -> None:
        self.mesh, self.g, self.opts = mesh, g, opts
        # q38 ATTNWIRE: the rows counts the draft's fused-loop passes run at (attach_draft_attn's loop buckets).
        self.buckets = None if buckets is None else tuple(sorted(int(b) for b in buckets))
        self._calls: dict[int, Any] = {}

        # As the whole step's op (qwen38_wholestep.py): no mutated argument; the donated pool comes back as an
        # output and the module copies it into the cache owner.
        @torch.library.custom_op("qwen38_mk::draft_attn", mutates_args=())
        def op(block: torch.Tensor, args: list[torch.Tensor]) -> list[torch.Tensor]:
            return list(self.callable_for(int(block.shape[0]))(block, *args))

        @op.register_fake
        def _(block, args):
            nh = self.g.NH // self.mesh.size
            return [block.new_empty((block.shape[0], nh * self.g.HD)), torch.empty_like(args[0])]

        self.op = op

    def callable_for(self, B: int) -> Any:
        if self.buckets is not None and B not in self.buckets:
            # q38 ATTNWIRE: a loop pass at rows the attach did not plan for; fail-stop, never a silent build.
            raise ValueError(f"q38 ATTNWIRE: MTP draft attention kernel asked for {B} rows; the draft's loop buckets "
                             f"are {self.buckets}")
        if B not in self._calls:
            _once(f"build{B}", "Qwen3.8 MTP draft attention kernel built at %d rows (q38 L1-H3)", B)
            fn, in_specs, outs, donate = build_draft_attn_call(self.mesh, self.g, B, **self.opts)
            typed = make_typed_step_wrapper(f"qwen38_mk_draft_attn_b{B}", fn, len(in_specs), len(outs))
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::draft_attn_b{B}", typed, mesh=self.mesh, input_partition_specs=in_specs,
                output_partition_specs=outs, donate_argnums=donate)
        return self._calls[B]


class DraftAttn(nn.Module):
    """The draft layer's fused-loop attention (`Qwen4ExpMultiTokenPredictor.mk_attn`); owns the packed
    operands as buffers."""

    kind: DraftAttnKind

    def __init__(self, attn: Any, g: Any, mesh: Any, kind: DraftAttnKind, B: int) -> None:
        super().__init__()
        object.__setattr__(self, "kind", kind)
        ex = ExecSpec(R=mesh.size, B=B, T=1, ctx=128, tiles=segment_tiles(B))
        o = kind.opts
        spec = qd.draft_attn_spec(g, mesh.size, budget=o["budget"],
                                  indexed=draft_indexed(o["max_groups"], o["pages_per_seq"], o["dcp"]))
        ops = pack_draft_attn_operands(attn, g, spec, ex)
        self.pack_rows = B  # q38 ATTNWIRE: the rows count the operands were packed at (max of the driver's buckets)
        self.n_w = len(ops)
        for i, t in enumerate(ops):
            self.register_buffer(f"w_{i}", t, persistent=False)

    def forward(self, block: torch.Tensor, positions: torch.Tensor, metadata: Any, attn: Any) -> torch.Tensor:
        from typing import cast

        from vllm.forward_context import get_forward_context

        per_layer = cast(dict, get_forward_context().attn_metadata)
        main_table = per_layer[attn.main_cache_name].block_tables
        n = block.shape[0]
        shared = attn._selection_rows[:n]
        from vllm_torchtpu.hixglue import logical_cols  # q38 HIX-GLUE SL: no-op when widths match

        shared = logical_cols(shared, attn.max_selected_tokens)
        rank = getattr(attn, "dcp_rank", None)
        if rank is None:
            rank = torch.zeros((1,), dtype=torch.int32, device=block.device)
        w = [getattr(self, f"w_{i}") for i in range(self.n_w)]
        pool = attn.cache_owners[0]
        out, new_pool = self.kind.op(block, [pool, shared, main_table.to(torch.int32), positions.to(torch.int32),
                                             metadata.query_start_loc.to(torch.int32),
                                             metadata.context_lens.to(torch.int32), attn.rope_cache, rank,
                                             attn.k_norm.weight, *w])
        pool.copy_(new_pool)
        return out


def loop_buckets(runner: Any, driver: Any) -> tuple[int, ...]:
    """q38 ATTNWIRE: the rows counts of the draft's fused-loop passes, i.e. the drafter's loop bucket (eagle3
    `_loop_bucket`: the request count padded by the runner's token paddings) of every request count up to max_num_reqs.
    A runner without those attributes (test stubs) falls back to the driver's buckets."""
    bucket = getattr(getattr(runner, "drafter", None), "_loop_bucket", None)
    n = int(getattr(runner, "max_num_reqs", 0) or 0)
    if bucket is None or n <= 0 or not getattr(runner, "num_tokens_paddings", None):
        return tuple(sorted(int(b) for b in driver.buckets))
    return tuple(sorted({int(bucket(r)) for r in range(1, n + 1)}))


def _gather(vec: list) -> list:
    """q38 ATTNWIRE: every TP rank's `vec`, in rank order, over the TP group's CPU group (one blocking all_gather; the
    pattern of spec_decode/draft_asm._gather); `[vec]` when the process has no TP group. A failure raises."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        grp = get_tp_group()
        world = int(getattr(grp, "world_size", 1))
    except Exception:  # noqa: BLE001 - no TP group (tests, single chip)
        return [list(vec)]
    if world <= 1:
        return [list(vec)]
    t = torch.tensor([int(v) for v in vec], dtype=torch.int64)
    out = [torch.zeros_like(t) for _ in range(world)]
    try:
        torch.distributed.all_gather(out, t, group=grp.cpu_group)
    except Exception as exc:
        raise RuntimeError("q38 ATTNWIRE: the TP ranks could not agree on the MTP draft attention kernel (CPU-group "
                           "all_gather failed); stopping rather than letting the ranks run different collectives") from exc
    return [o.tolist() for o in out]


def agree_draft_attn(runner: Any, attached: bool) -> bool:
    """q38 ATTNWIRE: all-rank agreement on the draft attention kernel (lead safety rule). The kernel carries TP / DCP
    collectives inside every draft program, so every TP rank takes it or none. One CPU-group all_gather (`_gather`)
    of [flag, attached, fingerprint of the kind's loop buckets, pack rows and options]; any difference raises on every
    rank (fail-stop). Logs the engagement line with the attached / refused rank counts once. Returns `attached`."""
    import hashlib

    model = getattr(getattr(getattr(runner, "drafter", None), "draft_model", None), "model", None)
    mk = getattr(model, "mk_attn", None) if attached else None
    fp = 0
    if mk is not None:
        raw = repr((mk.kind.buckets, int(mk.pack_rows), sorted(mk.kind.opts.items()))).encode()
        fp = int(hashlib.sha1(raw).hexdigest()[:15], 16)
    rows = _gather([int(bool(envs.VLLM_TPU_MK_DRAFT_ATTN)), int(bool(attached)), fp])
    if any(list(r) != list(rows[0]) for r in rows):
        raise RuntimeError(
            "q38 ATTNWIRE: the TP ranks disagree on the MTP draft attention kernel (flag, attached, fingerprint per rank: "
            f"{rows}); stopping rather than letting the ranks run different collectives")
    n_att = sum(int(r[1]) for r in rows)
    buckets = mk.kind.buckets if mk is not None else ()
    _once("agree", AGREE_LINE, n_att, len(rows) - n_att, len(rows), buckets, "kernel" if attached else "stock")
    return attached


def _draft_kv_scale_guard(attn: Any, opts: dict) -> None:
    """q38 KV-SCALE-GUARD v2: the draft attention's kernel (megakernel/wholestep/qsa_draft.py, union_attn) scales the FP8 K / V before
    the dot like the whole step, so the same guard applies; its own per-rank line carries the ' [draft]' site tag."""
    from .qwen38_kv_scale_guard import check_kv_scales

    check_kv_scales(opts["k_scale"], opts["v_scale"], rank=int(getattr(attn, "tp_rank", -1)),
                    where=str(getattr(attn, "prefix", "")) + " (draft)", site="draft")


def attach_draft_attn(runner: Any, driver: Any) -> bool:
    """Route the MTP draft's fused-loop attention to the kind (VLLM_TPU_MK_DRAFT_ATTN); before the draft's
    first trace. Refuses (stock draft attention, a warning) unless the d14mtp tail is attached, the draft
    attends the shared selection without indexer state, and its layer is pooled. q38 ATTNWIRE: every rank then
    agrees on the outcome (`agree_draft_attn`; fail-stop on any difference)."""
    return agree_draft_attn(runner, _attach(runner, driver))


def _attach(runner: Any, driver: Any) -> bool:
    draft = getattr(getattr(runner, "drafter", None), "draft_model", None)
    model = getattr(draft, "model", None)
    if getattr(model, "mk_tail", None) is None:
        logger.warning("Qwen3.8 MTP draft attention (q38 L1-H3): needs the attached MTP tail (mtp=1); stock")
        return False
    attn = model.layers[0].self_attn
    if not (getattr(attn, "_shared_ops", None) is not None and getattr(attn, "_shared_no_index_state", False)
            and getattr(attn, "_selection_source", None) is not None and getattr(attn, "pooled", False)):
        logger.warning("Qwen3.8 MTP draft attention (q38 L1-H3): needs shared selection, "
                       "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE and a pooled draft layer; stock")
        return False
    g, mesh = driver.g, driver.mesh
    dcp = int(getattr(attn, "main_kv_dcp", 1) or 1)
    budget = attn.token_topk // attn.compress_ratio
    # The draft runs on every step, not only whole-step ones: its table must cover the model's whole context
    # (a column past the kernel's table would address a wrong page), so size it by max_model_len.
    ctx_cap = int(getattr(runner, "max_model_len", 0) or getattr(driver, "whole_cap", 32768))
    max_groups = max(1024, -(-(ctx_cap // RATIO) // 1024) * 1024)
    page = int(attn.cache_owners[0].shape[1])  # tokens per pool block (one main-table column)
    pps = -(-max_groups * RATIO // page)
    if pps > MAX_TABLE_COLS:
        logger.warning("Qwen3.8 MTP draft attention (q38 L1-H3): %d main-table columns for %d tokens exceed the "
                       "AOT-checked %d; stock", pps, ctx_cap, MAX_TABLE_COLS)
        return False
    try:
        lb = loop_buckets(runner, driver)
    except Exception:
        logger.exception("Qwen3.8 MTP draft attention (q38 ATTNWIRE): the draft's loop buckets are unknown; stock")
        return False
    if max(lb) > MAX_LOOP_ROWS:
        logger.warning("Qwen3.8 MTP draft attention REFUSED (q38 ATTNWIRE): loop bucket %d rows exceeds the kernel's %d-row "
                       "bound (loop buckets %s); the draft attention stays stock", max(lb), MAX_LOOP_ROWS, lb)
        return False
    kw = dict(attn._select_keywords)  # the stock op's keywords (`bind_pool`)
    opts = dict(budget=budget, max_groups=max_groups, pages_per_seq=pps, dcp=dcp,
                main_slot=int(kw.get("main_slot", -1)),
                k_scale=float(getattr(attn, "_k_scale_float", 1.0)) if getattr(attn, "fp8_kv", False) else 1.0,
                v_scale=float(getattr(attn, "_v_scale_float", 1.0)) if getattr(attn, "fp8_kv", False) else 1.0,
                norm_eps=float(kw["norm_eps"]), rotary_dim=int(kw["rotary_dim"]), token_topk=int(kw["token_topk"]))
    assert int(kw["compress_ratio"]) == RATIO, kw["compress_ratio"]
    _draft_kv_scale_guard(attn, opts)  # q38 KV-SCALE-GUARD v2: outside the try below, so an inexact pair fails engine start
    try:
        kind = DraftAttnKind(mesh, g, opts, buckets=lb)
        model.mk_attn = DraftAttn(attn, g, mesh, kind, max(driver.buckets))
    except Exception:
        logger.exception("Qwen3.8 MTP draft attention (q38 L1-H3): packing failed; stock")
        return False
    logger.info(ATTACH_LINE, lb, dcp)
    return True
