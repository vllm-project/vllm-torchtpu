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
"""Pallas all-reduce for small TP activations (additional_config tp_small_all_reduce).

At decode a TP all-reduce moves a bf16 [rows, hidden] activation with rows of
a few tens. With ``--additional-config '{"tp_small_all_reduce": true}'``,
GroupCoordinator.all_reduce sends every 2-D bf16 all-reduce of the TP group
to one jax_op, which runs `kernels.experimental.tp_all_reduce` (one kernel:
reduce-scatter, f32 sum in rank order, all-gather; exactly rows x hidden
travel, no row padding) when the tensor is at most
``tp_small_all_reduce_max_bytes`` (512 KiB), and lax.psum otherwise. The
choice is made inside the op from the concrete row count, so the torch graph
stays the same for every token bucket (reading the token dimension in Python
would specialize the trace per bucket). The keys are additional_config, so
they are part of the compile-cache key.

`arm` runs once at the end of TPUModelRunner.load_model: the op's mesh is the
EP mesh (`build_ep_mesh`), so every JAX op in a module shares one device
order, and it needs the TP group to be the EP group and the whole world (DP 1,
PCP 1). It is collective: the ranks compare what they armed, then run the op
on exact inputs at several row counts before the first model step.
"""

import faulthandler

import jax
import numpy as np
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state

from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh, ep_mesh_index, get_ep_group
from vllm_torchtpu.kernels.experimental import tp_all_reduce
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SELFTEST_TIMEOUT_S = 600

# --additional-config keys.
ENABLE_KEY = "tp_small_all_reduce"
MAX_BYTES_KEY = "tp_small_all_reduce_max_bytes"
DEFAULT_MAX_BYTES = 512 * 1024

# Set by `arm`: (op, n ranks, group name) or None.
_ARMED: tuple | None = None
_MAX_BYTES = DEFAULT_MAX_BYTES


def eligible(x: torch.Tensor, n: int) -> bool:
    """Whether x goes through the op: dtype, rank and hidden size only, all
    static. The row count is the traced token dimension (a SymInt), so it is
    never read here: branching on it would specialize the graph per bucket.
    The op picks the kernel or psum from the concrete rows (`use_kernel`)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        return False
    return tp_all_reduce.supported(n, int(x.shape[1]))


def use_kernel(rows: int, hidden: int) -> bool:
    """Whether a concrete bf16 [rows, hidden] all-reduce takes the kernel."""
    return 0 < rows * hidden * 2 <= _MAX_BYTES


def _make_op(mesh):
    axis = mesh.axis_names[0]
    n = int(mesh.shape[axis])

    def body(x):
        rows, hidden = x.shape
        if not use_kernel(rows, hidden):
            return lax.psum(x, axis)
        return tp_all_reduce.all_reduce(x, n, axis)

    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=P(axis), out_specs=P(axis), check_vma=False
    )

    def fn(x: jax.Array) -> jax.Array:
        return sharded(jax.device_put(x, NamedSharding(mesh, P(axis))))

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::tp_small_all_reduce",
        fn,
        mesh=mesh,
        input_partition_specs=(P(axis),),
        output_partition_specs=P(axis),
    )
    op.register_fake(lambda x, *args, **kwargs: torch.empty_like(x))
    return op


def maybe_all_reduce(x: torch.Tensor, group_name: str) -> torch.Tensor | None:
    """The op's all-reduce of x (kernel or psum, by its rows), or None when
    the caller should use its own collective."""
    if _ARMED is None:
        return None
    op, n, armed_group = _ARMED
    if group_name != armed_group or not eligible(x, n):
        return None
    return op(x.contiguous())


def _patch_group_all_reduce() -> None:
    """Route GroupCoordinator.all_reduce through `maybe_all_reduce`. At the
    Python level, ahead of vLLM's all_reduce custom op, so the traced graph
    holds the kernel's op in place of the collective."""
    cls = parallel_state.GroupCoordinator
    if getattr(cls.all_reduce, "_tp_small_all_reduce", False):
        return
    stock = cls.all_reduce

    def all_reduce_(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size > 1:
            out = maybe_all_reduce(input_, self.unique_name)
            if out is not None:
                return out
        return stock(self, input_)

    all_reduce_._tp_small_all_reduce = True
    cls.all_reduce = all_reduce_


def _topology():
    """(mesh, mesh index) over the TP group, or a refusal reason. Collective."""
    tp = parallel_state.get_tp_group()
    ep = get_ep_group()
    if tp.world_size == 1:
        return "TP group has one rank"
    if ep is None:
        return "no EP group (needs --enable-expert-parallel for the shared mesh)"
    tp_ranks = tuple(int(r) for r in tp.ranks)
    if tp_ranks != tuple(int(r) for r in ep.ranks):
        return f"TP ranks {tp_ranks} != EP ranks (needs dp 1, pcp 1)"
    if len(tp_ranks) != torch.distributed.get_world_size():
        return f"the TP group has {len(tp_ranks)} of the world's ranks"
    mesh = build_ep_mesh()
    index = ep_mesh_index()
    assert mesh is not None and index is not None
    return mesh, index


def _gather(obj) -> list:
    tp = parallel_state.get_tp_group()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, obj, group=tp.cpu_group)
    return seen


def selftest_rows(hidden: int) -> list[int]:
    """Row counts the self-test runs: the decode sizes and the largest that
    still takes the kernel."""
    top = _MAX_BYTES // (2 * hidden)
    return sorted({r for r in (1, 4, 5, 8, 15, 16, 17, 32, 40, top) if 0 < r <= top})


def _selftest(op, n, index, hidden, device) -> list[tuple[int, int]]:
    """The exact sums of `tp_all_reduce.selftest_inputs` at every row count of
    `selftest_rows`; returns this rank's failing (rows, hidden)."""
    failures = []
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in selftest_rows(hidden):
            x = tp_all_reduce.selftest_inputs(n, rows, hidden)
            mine = torch.from_numpy(x[index]).to(torch.bfloat16).to(device)
            out = op(mine).to(torch.float32).cpu().numpy()
            if not np.array_equal(out, x.sum(0)):
                failures.append((rows, hidden))
    finally:
        faulthandler.cancel_dump_traceback_later()
    return failures


def arm_from_config(vllm_config, device: torch.device) -> bool:
    """Arm when additional_config asks for it; returns whether it is armed.
    Every rank sees the same config, so every rank arms or none does."""
    global _MAX_BYTES
    additional = vllm_config.additional_config
    cfg = additional if isinstance(additional, dict) else {}
    enabled = cfg.get(ENABLE_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{ENABLE_KEY} must be a bool")
    if not enabled:
        return False
    _MAX_BYTES = int(cfg.get(MAX_BYTES_KEY, DEFAULT_MAX_BYTES))
    if _MAX_BYTES <= 0:
        raise ValueError(f"additional_config.{MAX_BYTES_KEY} must be positive")
    return arm(vllm_config.model_config.get_hidden_size(), device)


def arm(hidden_size: int, device: torch.device) -> bool:
    """Arm the op for the TP group; returns whether it is armed. Collective."""
    global _ARMED
    topo = _topology()
    if isinstance(topo, str):
        logger.warning("tp_small_all_reduce: not armed: %s", topo)
        _gather(dict(index=-1))
        return False
    mesh, index = topo
    n = int(mesh.size)
    seen = _gather(dict(index=index))
    indices = sorted(s["index"] for s in seen)
    if indices != list(range(len(seen))):
        raise RuntimeError(f"tp_small_all_reduce: mesh indices {indices} != 0..n-1")
    if not tp_all_reduce.supported(n, hidden_size):
        logger.warning(
            "tp_small_all_reduce: not armed: hidden %d over %d ranks is not "
            "lane aligned",
            hidden_size,
            n,
        )
        return False
    op = _make_op(mesh)
    failures = _selftest(op, n, index, hidden_size, device)
    seen = _gather(dict(index=index, failures=failures))
    if any(s["failures"] for s in seen):
        raise RuntimeError(f"tp_small_all_reduce: self-test failed: {seen}")
    _ARMED = (op, n, parallel_state.get_tp_group().unique_name)
    _patch_group_all_reduce()
    logger.info(
        "tp_small_all_reduce: armed over %d ranks (mesh index %d); bf16 2-D "
        "tensors of at most %d bytes go to the Pallas kernel",
        n,
        index,
        _MAX_BYTES,
    )
    return True
