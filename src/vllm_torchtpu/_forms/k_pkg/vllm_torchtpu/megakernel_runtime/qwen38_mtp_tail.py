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
"""The MTP draft tail (o_proj, both HCs, MoE of the draft layer) as a torch op
that serves any rows count.

The draft program reaches the tail with every token count it is traced and
warmed up at, not only the megakernel buckets: the fused draft loop and the
all-token step see up to the scheduler's token budget. An L=1 segment kernel
built at such a count does not fit VMEM (256 rows: 77 MiB of 64, of which
11.5 MiB register spill slots and two f32[256, 10240] accumulators). Counts up
to `chunk` (the largest bucket) build the kernel at that count, as before;
larger counts run the `chunk`-row kernel over row chunks in one JAX call. The
tail is row-independent, so chunking does not change results.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from .exported_bridge import make_typed_step_wrapper, sharded_jax_callable
from .qwen38_segment_call import build_segment_call

logger = init_logger(__name__)

# Chunk counts above this mean the draft reached the tail with a prefill-sized
# batch; it still runs, but routing such steps to the stock layer would be cheaper.
WARN_CHUNKS = 8


def build_tail_call(
    mesh: Any, g: Any, B: int, layout: Any, chunk: int, write: bool = True, **kwargs: Any
) -> tuple[Any, tuple, tuple, tuple]:
    """`build_segment_call` for the L=1 tail at `B` rows per rank, in `chunk`-row
    kernel calls when `B > chunk`."""
    if B <= chunk:
        return build_segment_call(mesh, g, 1, B, layout, write, **kwargs)
    call_c, in_specs, out_specs, donate = build_segment_call(
        mesh, g, 1, chunk, layout, write, **kwargs
    )
    assert not donate and len(out_specs) == 1, "the L=1 tail has no GDN pool"
    n = -(-B // chunk)
    tp, chunked = P("tp"), P(None, "tp")

    def _split(x):
        x = jnp.pad(x, ((0, n * chunk - B),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape(n, chunk, *x.shape[1:])

    split = jax.shard_map(_split, mesh=mesh, in_specs=tp, out_specs=chunked, check_vma=False)
    merge = jax.shard_map(
        lambda r: r.reshape(n * chunk, *r.shape[2:])[:B],
        mesh=mesh,
        in_specs=chunked,
        out_specs=tp,
        check_vma=False,
    )

    def call(hidden, attn, *weights):
        def one(xa):
            return call_c(xa[0], xa[1], *weights)[0]

        return (merge(jax.lax.map(one, (split(hidden), split(attn)))),)

    return call, in_specs, out_specs, donate


class MtpTailKind:
    """The draft tail's torch op; resolves the rows count on the concrete input.

    Like the main model's `(1, False)` segment kind, but its own op, so a rows
    count past the buckets never builds a kernel at that count.
    """

    def __init__(self, mesh: Any, g: Any, layout: Any, write: bool, chunk: int) -> None:
        self.mesh, self.g, self.layout, self.write, self.chunk = mesh, g, layout, write, chunk
        self._calls: dict[int, Any] = {}
        self.tag = "mtp_tail" + ("" if write else "_shadow")

        @torch.library.custom_op(f"qwen38_mk::{self.tag}", mutates_args=())
        def op(
            hidden: torch.Tensor, attn: torch.Tensor, weights: list[torch.Tensor]
        ) -> list[torch.Tensor]:
            return list(self.callable_for(int(hidden.shape[0]))(hidden, attn, *weights))

        @op.register_fake
        def _(hidden, attn, weights):
            return [torch.empty_like(hidden)]

        self.op = op

    def callable_for(self, B: int) -> Any:
        if B not in self._calls:
            n = -(-B // self.chunk)
            log = logger.warning if n > WARN_CHUNKS else logger.info
            log(
                "Qwen3.8 MTP tail op built at %d rows (%s); largest so far %d",
                B,
                "one call" if B <= self.chunk else f"{n} x {self.chunk}-row chunks",
                max([B, *self._calls]),
            )
            full, in_specs, outs, donate = build_tail_call(
                self.mesh, self.g, B, self.layout, self.chunk, self.write
            )
            typed = make_typed_step_wrapper(
                f"qwen38_mk_{self.tag}_b{B}", full, len(in_specs), len(outs)
            )
            self._calls[B] = sharded_jax_callable(
                f"qwen38_mk::{self.tag}_b{B}",
                typed,
                mesh=self.mesh,
                input_partition_specs=in_specs,
                output_partition_specs=outs,
                donate_argnums=donate,
            )
        return self._calls[B]
