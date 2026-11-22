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
"""Torch op for the DSpark greedy Markov chain kernel
(kernels/collectives/markov_argmax.py), over the EP mesh like
`pallas_all_reduce`.

`build` runs at draft load, on every TP rank: it builds the op, runs it on
inputs whose answer every rank computes on the host (at 8 rows and at the
largest row bucket that fits VMEM), and returns None (the caller keeps the XLA
loop) unless every rank got that answer. A rank that hangs in the self-test
(a wrong mesh) exits the process after `SELFTEST_TIMEOUT_S`, as
`pallas_all_reduce`'s self-test does; it cannot fall back alone.
"""

import dataclasses
import faulthandler
from typing import Any

import jax
import numpy as np
import torch
import torch.nn.functional as F
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.collectives import markov_argmax as kernel
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

SELFTEST_TIMEOUT_S = 300
_SELFTEST_ROWS = 8
_SELFTEST_STEPS = 7


@dataclasses.dataclass(frozen=True)
class MarkovChainOp:
    """The kernel bound to one draft model's Markov head shard."""

    op: Any
    header: torch.Tensor  # int32 [3]: mesh index, shard start, real columns
    words: torch.Tensor  # int32 [vocab, D / 2]: markov_w1 viewed as words
    w2_even: torch.Tensor  # bf16 [D / 2, W] (`markov_argmax.split_halves`)
    w2_odd: torch.Tensor
    width: int  # W
    n: int  # ranks

    def fits(self, steps: int, rows: int) -> bool:
        """Whether a chain of ``steps`` over ``rows`` fits the kernel's VMEM."""
        return kernel.fits(self.n, steps, rows, self.width, self.words.shape[1])

    def __call__(
        self, slot_logits: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """int32 ``[rows, steps]`` drafts from ``slot_logits`` bf16
        ``[steps, rows, shard]`` (this rank's shard) and ``anchors [rows]``;
        call when `fits`."""
        steps, _, shard = slot_logits.shape
        logits = F.pad(slot_logits, (0, self.width - shard))
        # The kernel DMAs markov_w1 rows at these ids, unchecked (padded rows
        # carry the mask token).
        anchors = anchors.to(torch.int32).clamp(0, self.words.shape[0] - 1)
        scalars = torch.cat([self.header, anchors])
        out = self.op(scalars, logits, self.words, self.w2_even, self.w2_odd)
        # A copy, not a view of the kernel's [rows, 128] output: a compiled
        # program returning the view makes its consumer run a separate
        # as_strided program to materialize it.
        return out[:, :steps].contiguous()


def _refusal() -> str | None:
    if not envs.TPU_DSPARK_MARKOV_KERNEL:
        return "TPU_DSPARK_MARKOV_KERNEL=0"
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if tp.world_size == 1:
        return "TP is 1"
    if ep is None:
        return "no EP group"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if tp_ranks != tuple(int(r) for r in ep.ranks):
        return f"TP ranks {tp_ranks} != EP ranks {tuple(ep.ranks)}"
    if len(tp_ranks) != torch.distributed.get_world_size():
        return "the TP group is not the whole world"
    return None


def _make_op(mesh):
    axis = mesh.axis_names[0]
    n = int(mesh.size)
    specs = (P(axis),) * 5

    def body(scalars, logits, words, w2_even, w2_odd):
        return kernel.markov_argmax(scalars, logits, words, w2_even, w2_odd, n)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        scalars: jax.Array,
        logits: jax.Array,
        words: jax.Array,
        w2_even: jax.Array,
        w2_odd: jax.Array,
    ) -> jax.Array:
        args = (scalars, logits, words, w2_even, w2_odd)
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
        lambda scalars, logits, *args: torch.empty(
            (logits.shape[1], kernel.LANES), dtype=torch.int32, device=logits.device
        )
    )
    return op


def _bind(op, w1, w2_shard, start: int, valid: int, mesh_index: int, n: int, device):
    width = kernel.padded_width(w2_shard.shape[0])
    w2 = F.pad(w2_shard[:valid], (0, 0, 0, width - valid))
    bound = MarkovChainOp(
        op=op,
        header=torch.tensor(
            [mesh_index, start, valid], dtype=torch.int32, device=device
        ),
        # Built on the host into buffers of their own (+vocab x D x 2 bytes):
        # on the device torch_tpu kept the int32 view of the bf16 table, even
        # cloned, as strided slices it re-ran every call (8.6 ms, j-434d06f7).
        words=_on_host(w1, lambda w: w.view(torch.int32), device),
        w2_even=_on_host(w2, lambda w: w[:, 0::2].T, device),
        w2_odd=_on_host(w2, lambda w: w[:, 1::2].T, device),
        width=width,
        n=n,
    )
    if device.type == "tpu":
        synchronize_tensors([bound.header, bound.words, bound.w2_even, bound.w2_odd])
    return bound


def _on_host(x: torch.Tensor, fn, device) -> torch.Tensor:
    return fn(x.cpu().contiguous()).contiguous().to(device)


def _selftest(
    op, n: int, rank: int, mesh_index: int, dim: int, rows: int, device
) -> bool:
    """Run a chain of ``rows`` whose values are exact in bf16 (small integers)
    on every rank and compare it with the host's."""
    shard, steps = 1000, _SELFTEST_STEPS
    vocab = n * shard
    rng = np.random.default_rng(1234)
    logits = rng.integers(-8, 9, (steps, rows, vocab)).astype(np.float32)
    # Four +-1 entries per markov_w1 row and quarters in markov_w2: every bias
    # (|b| <= 1 in steps of 1/4) and score is exact in bf16.
    w1 = np.zeros((vocab, dim), np.float32)
    cols = rng.integers(0, dim, (vocab, 4))
    np.put_along_axis(w1, cols, rng.choice([-1.0, 1.0], (vocab, 4)), axis=1)
    w2 = rng.integers(-1, 2, (vocab, dim)).astype(np.float32) / 4
    anchors = rng.integers(0, vocab, rows)
    prev, want = anchors, []
    for i in range(steps):
        prev = (logits[i] + w1[prev] @ w2.T).argmax(-1)  # first index on ties
        want.append(prev)
    want = np.stack(want, 1)

    def t(x, dtype=torch.bfloat16):
        return torch.from_numpy(np.ascontiguousarray(x)).to(dtype).to(device)

    lo = rank * shard
    bound = _bind(op, t(w1), t(w2[lo : lo + shard]), lo, shard, mesh_index, n, device)
    got = bound(t(logits[:, :, lo : lo + shard]), t(anchors, torch.int32))
    return bool(np.array_equal(got.cpu().numpy(), want))


def _gather(obj) -> list:
    tp = parallel_state.get_tp_group()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def build(
    markov_w1: torch.Tensor,
    markov_w2_shard: torch.Tensor,
    vocab_start: int,
    valid: int,
    max_rows: int,
) -> MarkovChainOp | None:
    """The kernel bound to this rank's Markov head shard (``markov_w2_shard``
    ``[shard, D]``, of which the first ``valid`` rows are real), or None.
    ``max_rows``: the most rows a chain runs (the drafter's largest row
    bucket). Collective over the TP group."""
    device = markov_w1.device
    refusal = _refusal()
    mesh = None if refusal else build_ep_mesh()  # collective
    mesh_index = -1 if mesh is None else ep_mesh_index()
    seen = _gather(dict(refusal=refusal, mesh_index=mesh_index))
    if any(s["refusal"] for s in seen):
        logger.info("DSpark Markov kernel off: %s", seen[0]["refusal"])
        return None
    tp = parallel_state.get_tp_group()
    if sorted(s["mesh_index"] for s in seen) != list(range(tp.world_size)):
        logger.warning("DSpark Markov kernel off: mesh indices %s", seen)
        return None
    op = _make_op(mesh)
    n, dim = tp.world_size, markov_w1.shape[1]
    width = kernel.padded_width(markov_w2_shard.shape[0])
    # 8 rows, and the largest row count the drafter runs on the kernel
    # (`MarkovChainOp.fits`).
    test_rows = [_SELFTEST_ROWS] + [
        rows
        for rows in range(max_rows, _SELFTEST_ROWS, -1)
        if kernel.fits(n, _SELFTEST_STEPS, rows, width, dim // 2)
    ][:1]
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        passed = all(
            _selftest(op, n, tp.rank_in_group, mesh_index, dim, rows, device)
            for rows in test_rows
        )
    except Exception as e:  # noqa: BLE001 - any failure keeps the XLA loop
        logger.warning("DSpark Markov kernel self-test raised: %r", e)
        passed = False
    finally:
        faulthandler.cancel_dump_traceback_later()
    if not all(_gather(passed)):
        logger.warning("DSpark Markov kernel off: self-test failed on some rank")
        return None
    logger.info(
        "DSpark Markov chain runs as one kernel over %d ranks (self-test rows %s).",
        n,
        test_rows,
    )
    return _bind(
        op, markov_w1, markov_w2_shard, vocab_start, valid, mesh_index, n, device
    )
