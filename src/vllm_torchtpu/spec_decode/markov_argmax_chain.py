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
"""The greedy Markov chain of a vocab-parallel DSpark draft as one torch op
(kernels/collectives/markov_argmax_chain.py; VLLM_TPU_DSPARK_MARKOV_CHAIN).

`build` runs once, from the draft's load, and is collective over the TP group:
it builds the op's mesh (the EP mesh, which must span the same ranks as the TP
group and the whole world, as tp_all_reduce's does), and checks the op on every
rank against the gathered greedy chain computed locally from the replicated
Markov head. The draft step then calls the returned `MarkovArgmaxChain` in
place of its per-slot loop of `markov_argmax_shard`.
"""

import dataclasses
import faulthandler
from typing import Any

import jax
import torch
import torch.nn.functional as F
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import markov_argmax_chain as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# A wrong rank or mesh hangs the self-test's barrier: dump every stack and exit.
# Generous, since the first call also compiles each kernel variant.
SELFTEST_TIMEOUT_S = 900
# Of the TensorCore's VMEM, what the kernel may declare.
_VMEM_SHARE = 0.75
# Row counts whose math rows (padded to 8) are 8, 16 and 24, beside max_rows.
_SELFTEST_ROWS = (1, 9, 17)


def _make_op(mesh):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    specs = (P(axis),) * 5

    def body(logits, w2t, w1, prev, scalars):
        return kernel.markov_argmax_chain(logits, w2t, w1, prev, scalars, n)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        logits: jax.Array,
        w2t: jax.Array,
        w1: jax.Array,
        prev: jax.Array,
        scalars: jax.Array,
    ) -> jax.Array:
        args = (logits, w2t, w1, prev, scalars)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::markov_argmax_chain",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(
        lambda logits, w2t, w1, prev, scalars: logits.new_empty(
            logits.shape[:2], dtype=torch.int32
        )
    )
    return op


@dataclasses.dataclass(frozen=True)
class MarkovArgmaxChain:
    op: Any
    w2t: torch.Tensor  # this rank's Markov shard, [rank_dim, padded vocab]
    w1: torch.Tensor  # the replicated Markov embedding, [vocab, rank_dim]
    scalars: torch.Tensor  # int32 [1, 3]: mesh index, first id, ids in the shard
    max_rows: int  # the most rows a call takes (kernel.max_rows)

    def __call__(self, logits: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """int64 ``[rows, slots]`` greedy draft ids from this rank's
        ``[rows, slots, vocab shard]`` logits and each row's previous token."""
        ids = self.op(logits, self.w2t, self.w1, prev.to(torch.int32), self.scalars)
        return ids.to(torch.int64)


def _refusal() -> str | None:
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if ep is None:
        return "no EP group"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if tp_ranks != tuple(int(r) for r in ep.ranks):
        return f"TP ranks {tp_ranks} != EP ranks {tuple(ep.ranks)}"
    world = torch.distributed.get_world_size()
    if len(tp_ranks) != world:
        return f"the TP group has {len(tp_ranks)} of {world} ranks"
    return None


def _gather(obj) -> list:
    tp = parallel_state.get_tp_group()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def _reference(logits: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, prev):
    """The gathered greedy chain over full-vocab ``[rows, slots, vocab]``
    logits: per slot, argmax of bf16 logits + bf16 bias."""
    ids = []
    for s in range(logits.shape[1]):
        prev = (logits[:, s] + F.linear(w1[prev], w2)).argmax(dim=-1)
        ids.append(prev)
    return torch.stack(ids, dim=1)


def _selftest(
    chain: MarkovArgmaxChain, w2: torch.Tensor, slots: int, start: int, width: int
) -> list[int]:
    """Rows whose ids differ from the gathered chain's. Every rank draws the
    same full-vocab logits, runs the op on its shard and checks all ids."""
    failures = []
    vocab = chain.w1.shape[0]
    w2 = w2[:vocab]  # the LM-head-style projection may carry pad rows
    generator = torch.Generator().manual_seed(0)
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in (
            *(r for r in _SELFTEST_ROWS if r < chain.max_rows),
            chain.max_rows,
        ):
            full = torch.randn(rows, slots, vocab, generator=generator)
            prev = torch.randint(0, vocab, (rows,), generator=generator)
            full = full.to(dtype=w2.dtype, device=w2.device)
            prev = prev.to(w2.device)
            shard = full[:, :, start : start + width]
            if shard.shape[-1] < width:
                shard = F.pad(shard, (0, width - shard.shape[-1]))
            got = chain(shard.contiguous(), prev).cpu()
            want = _reference(full, chain.w1, w2, prev).cpu()
            if not torch.equal(got, want):
                failures.append(rows)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def build(
    w1: torch.Tensor,
    w2: torch.Tensor,
    vocab_start: int,
    valid: int,
    width: int,
    slots: int,
) -> MarkovArgmaxChain | None:
    """The chain op for this rank's vocab shard of the replicated bf16 Markov
    head (``w1`` embedding ``[vocab, r]``, ``w2`` projection ``[vocab, r]``):
    ``width`` logits columns, the first ``valid`` of them the ids from
    ``vocab_start``. None when the TP group cannot run it. Collective over the
    TP group."""
    refusal = _refusal()
    if refusal is None and not (w1.dtype == w2.dtype == torch.bfloat16):
        refusal = f"the Markov head is {w1.dtype} / {w2.dtype}, not bf16"
    if refusal is None and w1.shape[0] % 16:
        refusal = f"the Markov embedding's {w1.shape[0]} rows are not 16-row tiles"
    tp_size = parallel_state.get_tp_group().world_size
    budget = int(pltpu.get_tpu_info().vmem_capacity_bytes * _VMEM_SHARE)
    rows_cap = kernel.max_rows(tp_size, slots, w1.shape[1], width, budget)
    if refusal is None and rows_cap == 0:
        refusal = "no row count fits the kernel's VMEM budget"
    mesh = None if refusal else build_ep_mesh()  # collective
    mesh_index = -1 if mesh is None else ep_mesh_index()
    seen = _gather(dict(refusal=refusal, mesh_index=mesh_index))
    if any(s["refusal"] for s in seen):
        logger.info("markov_argmax_chain: not built: %s", [s["refusal"] for s in seen])
        return None
    tp = parallel_state.get_tp_group()
    if sorted(s["mesh_index"] for s in seen) != list(range(tp.world_size)):
        raise RuntimeError(f"markov_argmax_chain: mesh indices are not 0..n-1: {seen}")
    rows = w2[vocab_start : vocab_start + valid]
    lanes = kernel.padded_vocab(width)
    w2t = F.pad(rows, (0, 0, 0, lanes - valid)).t().contiguous()
    scalars = torch.tensor([[mesh_index, vocab_start, valid]], dtype=torch.int32)
    chain = MarkovArgmaxChain(
        op=_make_op(mesh),
        w2t=w2t,
        w1=w1,
        scalars=scalars.to(w2.device),
        max_rows=rows_cap,
    )
    failures = _gather(_selftest(chain, w2, slots, vocab_start, width))
    if any(failures):
        raise RuntimeError(
            f"markov_argmax_chain: self-test failed (rows per rank): {failures}"
        )
    logger.info(
        "markov_argmax_chain: built over %d ranks (mesh index %d, ids from %d, "
        "up to %d rows)",
        int(mesh.size),
        mesh_index,
        vocab_start,
        rows_cap,
    )
    return chain
