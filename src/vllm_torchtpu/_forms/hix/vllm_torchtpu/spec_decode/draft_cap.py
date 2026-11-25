# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-CAP inside DH-D (VLLM_TPU_DRAFT_ASM_CAP; card DRAFT-CAP, reports/decode_levers_v6_delta.md,
design scratch/l4/draft1b/draft_cap_design.md).

DH-D (spec_decode/draft_one.py, VLLM_TPU_DRAFT_ASM_ONE) runs draft step 0, the carry and the K-1 loop steps as
one program per loop bucket p. Its draft passes read the verify step's attention metadata, whose request
dimension is the chunk's request capacity (max-num-seqs, 64 in serving): every pass runs the ragged decode
attention over the capacity's request slots and all-gathers a capacity-slot DCP partial, and step 0's
``draft_rows`` (the seed's gather indices padded with -1 to the capacity) put o_proj, the HCs, the MoE and the
final mix on capacity rows, while p (the padded live request count R <= p) is 16 / 16-32 / 48 at c16-c48.

With the flag on, the program at loop bucket p runs every draft pass over R_d = min(p, capacity) request
slots, which is p in serving (static per program instance; R <= p on every step, so no request is dropped):

  * persistent per-R_d buffers for seq_lens [R_d], query_start_loc [R_d + 1] and every distinct per-group
    block table (the draft layer's main, compressed-key and raw-ring groups, each request-major and flat,
    [R_d * cols]) are what the program's forward context holds; the program fills them by in-program
    ``copy_`` from the first R_d request slots of the verify metadata (step 0, ``md0``) and of the loop
    metadata (``md1``), through DH-D's own in-place switch (``draft_one.dh_one_body``'s ``bufs``). Block
    tables cannot change within a propose, so the table prefixes are copied once, before step 0;
  * slots past the live requests are padding either way: query_start_loc entries past R hold the total
    token count, so step 0's KV writes and every live row's attention are unchanged;
  * step 0's ``draft_rows`` are the seed's gather indices [p] (seed output 1, the first p entries of the
    capacity rows DH-D passes otherwise; ``carry_rows`` then takes them whole).

Effects (C): the ragged decode attention runs over R_d slots, the DCP partial all-gather is [R_d, ., .]
(``qsa_op._dcp_finish``), step 0's MoE runs at m_{R_d}, and pooled QSA's row-wise path still holds
(len(rows) = R_d <= len(context) = R_d). The program count stays DH-D's (one per loop bucket, compiling only
its token buckets) at smaller shapes. Stage A, DRAFT-ASM and the base path keep the capacity.

The sliced shapes enter the DH-D warm keys and so DRAFT-ASM's agreement fingerprint: ranks with different
VLLM_TPU_DRAFT_ASM_CAP values end on the base path, never on different programs.
"""

import dataclasses
import types
from typing import Any

import torch
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.compilation import bucket_filter
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode import draft_asm, draft_one

logger = init_logger(__name__)

# Distinct per-group block tables one program takes (the draft QSA layer's main, compressed-key and raw-ring
# groups, plus one spare); a draft whose metadata has more keeps DH-D off (reason "one_cap_tables").
TABLE_SLOTS = 4


def cap_rows(p: int, capacity: int) -> int:
    """R_d of the program at loop bucket `p` for a chunk of request capacity `capacity`."""
    return min(p, capacity)


def prefix_md(md: tuple, r_d: int) -> tuple:
    """The first `r_d` request slots of (seq_lens, query_start_loc, request_distribution): views, taken inside
    the program (request_distribution counts requests and is unchanged)."""
    return (md[0][:r_d], md[1][: r_d + 1], md[2])


def copy_table_prefixes(tables: tuple, table_bufs: tuple) -> None:
    """Each table buffer [R_d * cols] takes the first R_d request rows of its request-major flat table
    [capacity * cols], in place."""
    for src, buf in zip(tables, table_bufs):
        if buf is not None:
            buf.copy_(src[: buf.shape[0]])


def dh_cap_body(
    loop: Any,
    *,
    md0: tuple,
    md1: tuple,
    bufs: tuple,
    tables: tuple,
    table_bufs: tuple,
    **kw: Any,
) -> tuple[torch.Tensor, ...]:
    """DH-D's program body (``draft_one.dh_one_body``) at R_d = len(bufs[0]) request slots: the per-group table
    prefixes into `table_bufs`, then DH-D's switch over the R_d prefixes of `md0` and `md1`. `md0` / `md1` /
    `tables` are the capacity-sized metadata, read only. Returns (token0, .., token_{K-1}, source)."""
    r_d = bufs[0].shape[0]
    copy_table_prefixes(tables, table_bufs)
    return draft_one.dh_one_body(loop, md0=prefix_md(md0, r_d), md1=prefix_md(md1, r_d), bufs=bufs, **kw)


class _DraftCapProgram(draft_one._DraftOneProgram):
    """DH-D's program at one loop bucket with R_d request slots (`dh_cap_body`): DH-D's arguments (step 0's
    ``draft_rows`` [R_d]; the metadata buffers [R_d] / [R_d + 1]), then the capacity-sized block table of each
    draft cache group and its [R_d * cols] buffer (unused slots None)."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        draft_rows: torch.Tensor,
        positions_step1: torch.Tensor,
        seq_lens0: torch.Tensor,
        query_start_loc0: torch.Tensor,
        request_distribution0: torch.Tensor,
        seq_lens1: torch.Tensor,
        query_start_loc1: torch.Tensor,
        request_distribution1: torch.Tensor,
        seq_lens_buf: torch.Tensor,
        query_start_loc_buf: torch.Tensor,
        request_distribution_buf: torch.Tensor,
        tail_bonus: torch.Tensor,
        num_reqs: torch.Tensor,
        table0: torch.Tensor | None = None,
        table1: torch.Tensor | None = None,
        table2: torch.Tensor | None = None,
        table3: torch.Tensor | None = None,
        table_buf0: torch.Tensor | None = None,
        table_buf1: torch.Tensor | None = None,
        table_buf2: torch.Tensor | None = None,
        table_buf3: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return dh_cap_body(
            self,
            input_ids=input_ids,
            positions=positions,
            target_hidden=hidden_states,
            draft_rows=draft_rows,
            positions_step1=positions_step1,
            md0=(seq_lens0, query_start_loc0, request_distribution0),
            md1=(seq_lens1, query_start_loc1, request_distribution1),
            bufs=(seq_lens_buf, query_start_loc_buf, request_distribution_buf),
            tables=(table0, table1, table2, table3),
            table_bufs=(table_buf0, table_buf1, table_buf2, table_buf3),
            tail_bonus=tail_bonus,
            num_reqs=num_reqs,
            max_reqs=self.max_reqs,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


def _cap_program_class(p: int) -> type:
    """A subclass of `_DraftCapProgram` whose forward has a code object of its own (as
    ``draft_one._program_class``: vLLM's compile wrapper and Dynamo key compiled code on it)."""
    base = _DraftCapProgram.forward
    code = base.__code__.replace(co_name=f"forward_cap_p{p}")
    fwd = types.FunctionType(code, base.__globals__, f"forward_cap_p{p}", base.__defaults__, base.__closure__)
    fwd.__qualname__ = f"DraftCapProgramP{p}.forward"
    fwd.__annotations__ = dict(base.__annotations__)
    return type(f"DraftCapProgramP{p}", (_DraftCapProgram,), {"forward": fwd})


def build_draft_cap_program(
    *, vllm_config: Any, prefix: str, mtp: Any, num_steps: int, dynamic_arg_dims: dict, dvp: bool = False,
    max_reqs: int, p: int,
) -> torch.nn.Module:
    """``draft_one.build_draft_one_program`` for the R_d program class of loop bucket `p`."""
    dims = {k: v for k, v in dict(dynamic_arg_dims).items() if k in draft_one.ONE_DYNAMIC_ARG_DIMS}
    cls = support_torch_compile(dynamic_arg_dims=dims)(_cap_program_class(p))
    return cls(
        vllm_config=vllm_config, prefix=prefix, mtp=mtp, num_steps=num_steps, dvp=dvp, max_reqs=max_reqs,
    )


def cap_buffers(ctl: Any, r_d: int, md0: tuple, tables: list) -> tuple[tuple, tuple]:
    """Persistent R_d buffers: (seq_lens [R_d], query_start_loc [R_d + 1], request_distribution) and one
    [R_d * cols] buffer per table, allocated once (warm-up) in the controller's ``one_bufs``."""
    key = ("cap", r_d, tuple((tuple(t.shape), t.dtype) for t in (*md0, *tables)))
    got = ctl.one_bufs.get(key)
    if got is None:
        cap = md0[0].shape[0]
        bufs = (
            torch.zeros_like(md0[0][:r_d]),
            torch.zeros_like(md0[1][: r_d + 1]),
            torch.zeros_like(md0[2]),
        )
        tbufs = tuple(torch.zeros_like(t[: r_d * (t.shape[0] // cap)]) for t in tables)
        got = (bufs, tbufs)
        ctl.one_bufs[key] = got
    return got


def cap_module(ctl: Any, p_rows: int):
    """``DraftOne._one_module`` with the R_d program class: one instance per loop bucket, compiling only its token
    buckets (compilation.bucket_filter), held in the controller's ``one_modules``."""
    module = ctl.one_modules.get(p_rows)
    if module is None:
        p = ctl.p
        try:
            with (
                set_model_tag("eagle_head"),
                draft_asm.set_vllm_model_wrapper_context(mesh=p.runner.mesh, vllm_config=p.draft_vllm_config),
                draft_asm.set_current_vllm_config(p.draft_vllm_config),
            ):
                k = p.speculative_config.num_speculative_tokens
                module = build_draft_cap_program(
                    vllm_config=p.draft_vllm_config,
                    prefix=f"draft_cap_k{k}_p{p_rows}",
                    mtp=p.draft_model,
                    num_steps=k - 1,
                    dynamic_arg_dims=getattr(p.draft_model, "_dynamic_arg_dims", {}),
                    dvp=getattr(p, "_dvp", False),
                    max_reqs=ctl.max_reqs,
                    p=p_rows,
                )
        except Exception as exc:
            # Fail-stop: every rank runs the DH-D program on the same steps.
            raise RuntimeError(
                "q38 DRAFT-CAP: could not build the DH-D R_d program; set VLLM_TPU_DRAFT_ASM_CAP=0 to run without it"
            ) from exc
        serves = ctl._one_serves(p_rows)
        bucket_filter.assign(module, lambda n, serves=serves: n in serves)
        ctl.one_modules[p_rows] = module
    return module


def ready(ctl: Any, st: Any, target_md: dict, distinct: list, md0: tuple, md1: tuple, p_rows: int,
          input_ids: torch.Tensor, positions: torch.Tensor, target_hidden: torch.Tensor):
    """``DraftOne._one_ready``'s tail with DRAFT-CAP on: (module, metadata, args), or None and a reason (more
    distinct block tables than TABLE_SLOTS, or a table that is not request-major over the capacity; off warm-up,
    a key the warm-up never recorded)."""
    seed = st.seed
    cap = md0[0].shape[0]
    tables: list = []
    for m in distinct:
        t = m.block_tables
        if not isinstance(t, torch.Tensor) or t.dim() != 1 or t.shape[0] % cap:
            return None, "one_cap_tables"
        if all(t is not u for u in tables):
            tables.append(t)
    if len(tables) > TABLE_SLOTS:
        return None, "one_cap_tables"
    if not tables:
        return None, "one_cap_tables"
    r_d = cap_rows(p_rows, cap)
    bufs, tbufs = cap_buffers(ctl, r_d, md0, tables)
    rows = seed[1] if seed[1].shape[0] == r_d else seed[3]  # [p] gather indices; [cap] rows when p > cap
    pad = (None,) * (TABLE_SLOTS - len(tables))
    args = (input_ids, positions, target_hidden, rows, seed[draft_one.SEED_POS1], *md0, *md1, *bufs,
            seed[draft_one.SEED_BONUS], st.num_reqs_dev, *tables, *pad, *tbufs, *pad)
    if st.mode != "warm" and draft_one._key(*args) not in ctl.warm["one"]:
        return None, "one_cold"
    replaced: dict[int, Any] = {}
    md = {}
    for name, m in target_md.items():
        if id(m) not in replaced:
            tb = next(b for t, b in zip(tables, tbufs) if t is m.block_tables)
            replaced[id(m)] = dataclasses.replace(
                m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2], block_tables=tb
            )
        md[name] = replaced[id(m)]
    logged = ctl.__dict__.setdefault("cap_logged", set())
    if st.mode == "fold" and r_d not in logged:
        logged.add(r_d)
        logger.info("q38 DRAFT-CAP: on at loop bucket %d: R_d %d of capacity %d request slots, %d block tables "
                    "(TP rank %d)", p_rows, r_d, cap, len(tables), ctl._rank())
    return (cap_module(ctl, p_rows), md, args), ""
