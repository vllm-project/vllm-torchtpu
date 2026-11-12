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
"""The DSpark greedy Markov chain as one torch op (TPU_DSPARK_MARKOV_KERNEL).

`arm` runs once when the draft loads, collectively over the TP group: it builds
the op over the EP mesh (the mesh of every other Pallas kernel in the step,
which must span the TP group), keeps the Markov rows the kernel reads, and
runs the kernel once on integer-valued inputs whose answer every rank can
compute exactly. A rank that disagrees fails the self-test and nothing arms.
`FusedMarkov.draft` then replaces the K-step loop of
`markov_argmax_shard` calls in the compiled draft step.
"""

import dataclasses

import jax
import numpy as np
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import markov_argmax as kernel
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Hold markov_w1 in f32 for the kernel (a copy) instead of reading the bf16
# weight in place.
W1_F32 = False
_SELFTEST_ROWS = 2


@dataclasses.dataclass(frozen=True)
class FusedMarkov:
    op: object
    rank: torch.Tensor  # [1, 1] int32 mesh index
    start: torch.Tensor  # [1, 1] int32 first global id of this rank's shard
    w1: torch.Tensor  # [vocab, r]
    w2p: torch.Tensor  # [padded_vocab(vs), r] bf16
    vs: int

    def draft(self, logits: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        """``[rows, k]`` int64 greedy drafts from this rank's ``[rows, k, vs]``
        vocab shard of the draft logits and each row's anchor id ``prev0``."""
        ids = self.op(
            logits.contiguous(),
            prev0.to(torch.int32),
            self.w1,
            self.w2p,
            self.rank,
            self.start,
        )
        return ids.to(torch.int64)


def _make_op(mesh, vs: int):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    specs = (P(axis),) * 6

    def body(logits, prev0, w1, w2p, rank, start):
        return kernel.markov_argmax(logits, prev0, w1, w2p, rank, start, n, vs)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        logits: jax.Array,
        prev0: jax.Array,
        w1: jax.Array,
        w2p: jax.Array,
        rank: jax.Array,
        start: jax.Array,
    ) -> jax.Array:
        args = (logits, prev0, w1, w2p, rank, start)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::dspark_markov_argmax",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=P(axis),
    )
    op.register_fake(
        lambda logits, prev0, w1, w2p, rank, start: logits.new_empty(
            logits.shape[:2], dtype=torch.int32
        )
    )
    return op


def _bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def selftest_inputs(n: int, vs: int, vocab: int, rank_dim: int, k: int):
    """Integer-valued inputs (every sum exact in f32) and their drafts.

    Returns (logits [n, rows, k, vs], w1 [vocab, r], w2 [n * vs, r], prev0
    [rows], expected [rows, k]); the expectation rounds bias and score to bf16
    as the kernel does, and breaks ties to the lowest id."""
    rows = _SELFTEST_ROWS
    v = np.arange(vocab, dtype=np.int32)[:, None]
    c = np.arange(rank_dim, dtype=np.int32)[None, :]
    w1 = (((v * 7 + c * 3) % 5) - 2).astype(np.float32) / 4
    g = np.arange(n * vs, dtype=np.int32)[:, None]
    w2 = (((g * 11 + c * 5) % 5) - 2).astype(np.float32) / 8
    rng = np.random.default_rng(1234)
    logits = rng.integers(-3, 4, size=(n, rows, k, vs)).astype(np.float32)
    prev0 = rng.integers(0, vocab, size=(rows,)).astype(np.int64)
    prev = prev0.copy()
    expected = []
    for i in range(k):
        bias = _bf16(w1[prev] @ w2.T)  # [rows, n * vs]
        scores = _bf16(logits[:, :, i, :].transpose(1, 0, 2).reshape(rows, -1) + bias)
        prev = scores.argmax(-1)  # first max = lowest id
        expected.append(prev)
    return logits, w1, w2, prev0, np.stack(expected, 1)


def _gather(obj) -> list:
    tp = parallel_state.get_tp_group()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def _refusal() -> str | None:
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if tp.world_size == 1:
        return "TP is 1"
    if ep is None:
        return "no EP group"
    if tuple(int(r) for r in tp.ranks) != tuple(int(r) for r in ep.ranks):
        return "the TP and EP groups differ"
    if len(tp.ranks) != torch.distributed.get_world_size():
        return "the TP group is not the whole world"
    return None


def _shape_refusal(vocab: int) -> str | None:
    if vocab % 16:
        return f"the vocab ({vocab}) is not a multiple of the 16-row tile"
    return None


def arm(
    w1: torch.Tensor,
    w2_shard: torch.Tensor,
    vs: int,
    vocab_start: int,
    k: int,
) -> FusedMarkov | None:
    """Build and self-test the op; None when it cannot serve this deployment.
    Collective over the TP group.

    w1 [vocab, r] is markov_w1 (replicated), w2_shard [>= vs, r] this rank's
    rows of markov_w2 from ``vocab_start`` on, vs the shard width."""
    tp = parallel_state.get_tp_group()
    refusal = _refusal() or _shape_refusal(w1.shape[0])
    mesh = None if refusal else build_ep_mesh()  # collective
    mesh_index = -1 if mesh is None else ep_mesh_index()
    seen = _gather(dict(refusal=refusal, mesh_index=mesh_index))
    if any(s["refusal"] for s in seen):
        logger.warning(
            "TPU_DSPARK_MARKOV_KERNEL=1 not armed: %s", [s["refusal"] for s in seen]
        )
        return None
    if sorted(s["mesh_index"] for s in seen) != list(range(tp.world_size)):
        raise RuntimeError(f"dspark markov kernel: mesh indices {seen}")
    n = int(mesh.size)
    device = w2_shard.device
    vocab, rank_dim = w1.shape
    vp = kernel.padded_vocab(vs)
    op = _make_op(mesh, vs)
    rank_t = torch.tensor([[mesh_index]], dtype=torch.int32, device=device)
    tp_rank = tp.rank_in_group

    # Self-test on this deployment's shapes: catches a wrong mesh index,
    # vocab start or exchange before serving. The answer needs no peer.
    lg, tw1, tw2, p0, expected = selftest_inputs(n, vs, vocab, rank_dim, k)
    w1_dtype = torch.float32 if W1_F32 else torch.bfloat16
    t_start = torch.tensor([[tp_rank * vs]], dtype=torch.int32, device=device)
    t_w2 = torch.zeros((vp, rank_dim), dtype=torch.bfloat16)
    t_w2[:vs] = torch.from_numpy(tw2[tp_rank * vs : (tp_rank + 1) * vs])
    test = FusedMarkov(
        op,
        rank_t,
        t_start,
        torch.from_numpy(tw1).to(torch.bfloat16).to(w1_dtype).to(device),
        t_w2.to(device),
        vs,
    )
    got = test.draft(
        torch.from_numpy(lg[tp_rank]).to(torch.bfloat16).to(device),
        torch.from_numpy(p0).to(device),
    ).cpu().numpy()
    ok = bool(np.array_equal(got, expected))
    del test
    results = _gather(ok)
    if not all(results):
        logger.warning(
            "TPU_DSPARK_MARKOV_KERNEL=1 not armed: self-test failed on ranks %s",
            [i for i, r in enumerate(results) if not r],
        )
        return None

    w2p = torch.zeros((vp, rank_dim), dtype=torch.bfloat16, device=device)
    w2p[: min(vs, w2_shard.shape[0])] = w2_shard[:vs].to(torch.bfloat16)
    fused = FusedMarkov(
        op,
        rank_t,
        torch.tensor([[vocab_start]], dtype=torch.int32, device=device),
        w1.detach().to(w1_dtype).contiguous(),
        w2p,
        vs,
    )
    logger.info(
        "TPU_DSPARK_MARKOV_KERNEL=1: fused DSpark Markov kernel armed over %d "
        "ranks (mesh index %d, vocab start %d, shard %d -> %d cols, k=%d, "
        "w1 %s); self-test passed",
        n,
        mesh_index,
        vocab_start,
        vs,
        vp,
        k,
        w1_dtype,
    )
    return fused
