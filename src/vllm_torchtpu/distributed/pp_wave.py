# SPDX-License-Identifier: Apache-2.0
"""Hand-off of activations between pipeline stages over ICI.

One hand-off is a single collective-permute that all stages run together:
each stage sends one buffer to the next stage and receives one from the
previous stage (the last stage's buffer goes to stage 0, which ignores it).
Because every stage takes part in every hand-off, launch number p on one
stage pairs with launch number p on every other stage.

What travels is the sum of hidden states and residual, one tensor in the
model dtype. The receiving stage's first norm adds hidden states and
residual in float32, normalizes that sum, and keeps it rounded to the model
dtype as the new residual. Given the rounded sum and a zero residual, the
norm keeps the identical residual but normalizes a sum rounded once more
than in a single-stage model: a relative difference of at most 2^-8 per
element for bfloat16. Sending the sum halves the bytes on the link; sending
both tensors would reproduce the single-stage arithmetic exactly.

A model may need side data to travel with the activations: a sparse-attention
model whose later layers reuse the top-k indices an earlier layer chose has to
carry that table across a stage boundary, because the layer that would
recompute it lives on another chip. Such a tensor rides the same launch as a
further tensor. A stage that does not recompute a tensor passes on what it
received.

A forward here is one runner forward over one token chunk; the runner splits
a scheduler step into as many forwards as its token budget needs, and each
forward is one hand-off. Launch numbers count from 0 within a burst.

Stage 0 launches once right after each forward it runs; every other stage
launches once right before each forward, because it needs the incoming
activations first, and that launch carries the output of the stage's
previous forward (zeros when there is none). Prologue launches of zeros fill
the numbers before a stage's first real one: one on stage 0, r on stage r.
So the output of stage 0's forward m goes out on launch m+1, stage r receives
it on launch m+r, right before its own forward m, and it reaches the last of
S stages on launch m+S-1. A stage that is ahead of the others queues its
launches on its device; its host never waits.

Three stages, two forwards, then two pushes that carry forward 1 to the
last stage. Each row is one launch number and each entry the buffer that
stage sends on it:

    launch  stage 0             stage 1               stage 2
    0       prologue: zeros     prologue: zeros       prologue: zeros
    1       after fwd 0: out 0  before fwd 0: zeros   prologue: zeros
    2       after fwd 1: out 1  before fwd 1: out 0   before fwd 0: zeros
    3       after push: zeros   before push: out 1    before fwd 1: zeros
    4       after push: zeros   before push: zeros    before push: zeros

On launch p stage r receives what stage r-1 sent on launch p: stage 1 gets
stage 0's output 0 on launch 1, stage 2 gets stage 1's output 0 on launch 2
and its output 1 on launch 3.

Launches pair up only while every stage keeps launching, which they do
while the engine keeps dispatching steps. When the engine stops to wait
for a result, forwards can be stuck part-way, because the launches that
would carry them further are never issued. A push (``push``) is a forward
that carries nothing: every stage issues the one launch it would issue for
a forward, sending what it still holds and then zeros, so every forward in
flight moves one stage further and the numbering stays as it is. The
engine side (``pp_push``) sends as many pushes as the waited-for step still
needs, counting scheduler steps. A stage that is ahead keeps its unpaired
launches queued on its device until the others catch up; once nothing is
in flight the engine settles (``settle``): each stage launches whatever it
still holds, then enough zeros to bring all stages to the same count, so
the devices are quiet while the engine is idle. The forwards between two
settles form a burst; the next burst starts with the prologue again.

Two rules keep the launches pairing across stages: every launch runs the
same compiled program (one buffer padded to the largest token bucket), and
every launch is submitted to the device right away, even on a stage that
never reads the result.
"""

from typing import Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed.pp_shift import get_or_create_pp_mesh, pp_permute_op
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.annotation import TraceAnnotation

logger = init_logger(__name__)

# The intermediate tensors every stage passes on.
HIDDEN_STATES = "hidden_states"
RESIDUAL = "residual"


def pp_rank_flags(parallel_config: Any) -> tuple[bool, bool]:
    """(is_first_stage, is_last_stage) for this worker."""
    if parallel_config.pipeline_parallel_size <= 1:
        return True, True
    from vllm.distributed.parallel_state import get_pp_group

    group = get_pp_group()
    return bool(group.is_first_rank), bool(group.is_last_rank)


def _launch_now(op, carried: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Run one hand-off as its own device program, submitted immediately.

    vLLM runs torch_tpu in fused-eager mode, where ops are deferred, fused
    with their neighbours and submitted only when a result is read. Pending
    producers are flushed first, then the launch alone; neither flush waits.
    """
    from torch_tpu._internal.sync import sync

    sync.synchronize(None, wait=False)
    out = op(*carried)
    if isinstance(out, torch.Tensor):
        out = (out,)
    else:
        out = tuple(out)
    sync.synchronize(list(out), wait=False)
    return out


class PPWave:
    def __init__(
        self,
        device: torch.device,
        max_rows: int,
        template: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ):
        from vllm.distributed.parallel_state import get_pp_group

        group = get_pp_group()
        self.rank = int(group.rank_in_group)
        self.size = int(group.world_size)
        self.last = self.size - 1
        extras = [key for key in template if key not in (HIDDEN_STATES, RESIDUAL)]
        if (
            set(template) != {HIDDEN_STATES, RESIDUAL, *extras}
            or template[HIDDEN_STATES] != template[RESIDUAL]
            or any(len(template[key][0]) != 1 for key in (HIDDEN_STATES, *extras))
        ):
            raise ValueError(
                "the pipeline hand-off carries hidden_states and residual, "
                "two [tokens, hidden] tensors of one dtype, plus any number "
                "of [tokens, width] side tensors; the model's intermediate "
                f"tensors are {template}"
            )
        # The first is the activation sum; the rest are side data a stage
        # passes on unchanged. Every stage builds from the same template,
        # so the order matches across stages.
        self.carried = (HIDDEN_STATES, *extras)
        self.max_rows = int(max_rows)
        self.op = pp_permute_op(get_or_create_pp_mesh(), num_tensors=len(self.carried))
        self.zeros = tuple(
            torch.zeros(
                (self.max_rows, *template[key][0]),
                dtype=template[key][1],
                device=device,
            )
            for key in self.carried
        )
        self.zero = self.zeros[0]
        self.launches = 0
        self.bursts = 0
        self.burst_open = False
        self.burst_launches = 0
        self.forwards = 0
        self.pending: tuple[torch.Tensor, ...] | None = None
        logger.info(
            "PP wave: rank %d/%d, hand-off carries %s",
            self.rank,
            self.size,
            ", ".join(
                f"{key}[{self.max_rows}, {template[key][0][0]}] {template[key][1]}"
                for key in self.carried
            ),
        )

    def _launch(self, carried: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        with TraceAnnotation("PP:Handoff"):
            out = _launch_now(self.op, carried)
        self.launches += 1
        self.burst_launches += 1
        return out

    def _pad(
        self, carried: tuple[torch.Tensor, ...], rows: int
    ) -> tuple[torch.Tensor, ...]:
        if rows == self.max_rows:
            return carried
        return tuple(torch.cat([t, z[rows:]]) for t, z in zip(carried, self.zeros))

    def warmup(self) -> None:
        """One launch outside any burst on every stage, carrying a known
        value; checks that it arrived on the next stage."""
        assert not self.burst_open and self.launches == 0
        out = self._launch(tuple(torch.full_like(z, self.rank + 1) for z in self.zeros))
        if self.rank > 0:
            want = float(self.rank)
            for key, x in zip(self.carried, out):
                got = (float(x[0, 0].item()), float(x[-1, -1].item()))
                if any(g != want for g in got):
                    raise RuntimeError(
                        f"PP wave rank {self.rank}: hand-off self-test tensor "
                        f"{key!r} received {got}, expected {want}"
                    )
        else:
            _ = out[0][0, 0].item()
        self.burst_launches = 0

    def _open_burst(self) -> None:
        """Stage r's launch before its forward m is number m+r and stage 0's
        launch after its forward m is number m+1; empty launches fill the
        numbers before the first real one."""
        if self.burst_open:
            return
        self.burst_open = True
        self.bursts += 1
        self.burst_launches = 0
        self.forwards = 0
        with TraceAnnotation("PP:Prologue"):
            for _ in range(max(self.rank, 1)):
                self._launch(self.zeros)

    def _check_count(self, extra: int = 0) -> None:
        expected = self.forwards + max(self.rank, 1) + extra
        if self.burst_launches != expected:
            raise RuntimeError(
                f"PP wave rank {self.rank}: burst of {self.forwards} forwards "
                f"issued {self.burst_launches} launches, expected {expected}"
            )

    def start_forward(self, rows: int) -> IntermediateTensors | None:
        """Called before forward number m of the burst with its padded token
        count. Returns the incoming activations (None on the first stage)."""
        self._open_burst()
        if self.rank == 0:
            return None
        send = self.pending if self.pending is not None else self.zeros
        self.pending = None
        out = self._launch(send)
        # Copies: the next launch overwrites the received buffers. The
        # residual is zero: the sum arrived as the hidden states.
        tensors = {key: x[:rows].clone() for key, x in zip(self.carried, out)}
        tensors[RESIDUAL] = self.zero[:rows].clone()
        return IntermediateTensors(tensors)

    def end_forward(self, tensors: dict[str, torch.Tensor] | None, rows: int) -> None:
        """Called after the forward with its outgoing activations (None on
        the last stage)."""
        self.forwards += 1
        if self.rank == self.last:
            self._check_count()
            return
        assert tensors is not None
        missing = [key for key in self.carried if key not in tensors]
        if missing:
            raise RuntimeError(
                f"PP wave rank {self.rank}: the forward produced no "
                f"{missing} to hand on"
            )
        # Summed in the model dtype; see the module docstring.
        total = tensors[HIDDEN_STATES] + tensors[RESIDUAL]
        outgoing = (total, *(tensors[key] for key in self.carried[1:]))
        padded = self._pad(outgoing, rows)
        if self.rank == 0:
            self._launch(padded)
        else:
            self.pending = padded
        self._check_count()

    def push(self) -> None:
        """A forward that carries nothing: the one launch this stage issues
        for a forward, sending what it still holds (zeros otherwise), so
        every forward in flight moves one stage further."""
        self._open_burst()
        with TraceAnnotation("PP:Push"):
            if self.rank == 0:
                self._launch(self.zeros)
            else:
                send = self.pending if self.pending is not None else self.zeros
                self.pending = None
                self._launch(send)
        self.forwards += 1
        if 0 < self.rank < self.last:
            self.pending = self.zeros
        self._check_count()

    def settle(self) -> None:
        """Bring every stage to the same launch count and close the burst;
        called once nothing is in flight."""
        if not self.burst_open:
            return
        with TraceAnnotation("PP:Settle"):
            if 0 < self.rank < self.last:
                assert self.pending is not None
                self._launch(self.pending)
                self.pending = None
            zero_launches = self.size - 2 - self.rank if self.rank < self.last else 0
            for _ in range(zero_launches):
                self._launch(self.zeros)
        # Every stage now holds forwards + size - 1 launches.
        self._check_count(self.size - 1 - max(self.rank, 1))
        if self.bursts <= 5 or self.bursts % 100 == 0:
            logger.debug(
                "PP wave rank %d: burst %d settled after %d forwards, %d "
                "launches (%d total)",
                self.rank,
                self.bursts,
                self.forwards,
                self.burst_launches,
                self.launches,
            )
        self.burst_open = False
        self.forwards = 0
        self.burst_launches = 0
