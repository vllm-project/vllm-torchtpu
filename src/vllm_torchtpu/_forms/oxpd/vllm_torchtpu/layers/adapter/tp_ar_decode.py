# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX: the decode TP8 all-reduce of [rows, hidden] bf16 as one Pallas call (H8) instead of XLA's ring.

Why (labels M / C / I as in reports/lane_arfix.md):
- The decode step issues 55 TP all-reduces of bf16 [rows, 2560] (M, in-engine captures j-3c0d2c76 at 16 rows and
  j-b91368df at 64 rows): 49 in the target forward (48 attention output projections + the vocab-parallel embedding)
  and 6 in the MTP draft modules. XLA runs each as a synchronous TensorCore ring all-reduce, 14.5 us at 16 rows and
  18.1 us at 64 rows (M), i.e. a latency floor of about 14 ring steps; 0.80 / 0.99 ms per step (M).
- DEC-FB-2's hierarchical kernel (``kernels/collectives/tp8_hier_all_reduce.py``, H8: die-to-die pair sum, a 4-chip
  exchange on half the bytes, an all-gather) measured 11.66 us device time per call at 64 rows (M, T39 j-79670eeb
  trace rows64_h8; 11.84 / 15.40 / 22.77 us long-minus-short at 64 / 128 / 256 rows).
- T39's XLA arm (38-42 us) was NOT the engine's all-reduce: its trace shows it offloaded to the SparseCore
  (``SparseCore Offload Type: OFFLOAD_COLLECTIVE``), because the bench created the JAX client before
  ``env_override`` raised the SparseCore offload threshold (M + C). Its "no keep" ratios are against that arm.

What this does: with ``VLLM_TPU_DECODE_TP_AR=h8`` (bf16 pair-sum wire, PR 2821's choice) or ``=h8x`` (f32 wire,
bitwise the single-rounding f32 sum), the attention output all-reduces of the Qwen3.8 decode forward
(``sequence_parallel.reduce`` on the replicated path and ``Qwen4ExpQSA.forward_with_indices(reduce=True)``, the
MTP draft's row path) call one Pallas kernel per rank, built as a torch_tpu ``jax_op`` over the EP mesh (the same
construction and DeviceAssignment as the fused MoE decode kernel). Inside the op the row count is static: rows that
H8 does not take (not a multiple of 16, or above ``VLLM_TPU_DECODE_TP_AR_MAX_ROWS``, default 256) use XLA's
all-reduce (``lax.psum``) in the same program, so no torch branch depends on the bucket.

Flag off (unset or empty, the default): ``arm`` returns at once, ``_ARMED`` stays None, ``use_kernel`` is False and
both call sites run their original ``tensor_model_parallel_all_reduce(x)`` line, so the traced graph is unchanged.
The switch is not registered in envs.py; ``tpu_compiler._tpu_compile_env_factors`` keys the compile cache with it
only when it is set, so unset leaves every compile key unchanged.

Fail-closed: a set flag that cannot arm (no EP mesh, TP world != 8, the EP group is not the TP group, ranks
disagree) raises at load rather than serving stock silently. Every rank arms the same way (signature compared over
the TP group): a rank on the XLA path while its peers wait in the kernel would hang the slice.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm_torchtpu.kernels.collectives.tp8_ar_decode_fn import (  # noqa: F401  (ENV / modes re-exported)
    ENV,
    MAX_ROWS_ENV,
    MODES,
    N_MEMBERS,
    PAD16_ENV,
    compile_factors,
    global_fn,
    max_rows,
    mode,
    pad16,
    pairs_from_device_ids,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (op, hidden) once armed; None means every call takes the stock XLA all-reduce.
_ARMED: tuple[Any, int] | None = None


def _check_same_on_every_rank(signature: tuple, group) -> None:
    gathered: list = [None] * int(group.world_size)
    torch.distributed.all_gather_object(gathered, signature, group=group.cpu_group)
    if any(g != gathered[0] for g in gathered):
        raise RuntimeError(f"{ENV}: TP ranks disagree on the decode all-reduce arming: {sorted(set(map(str, gathered)))}")


def arm(model_runner: Any = None) -> bool:
    """Build the op once after load (every rank calls it). Returns True when armed; False only when the flag is
    off. A set flag that cannot arm raises."""
    global _ARMED
    m = mode()
    if not m:
        return False
    rows_cap = max_rows()
    pad = pad16()
    from jax.experimental.pallas import tpu as pltpu  # noqa: F401  (import check before any op is built)
    from jax.sharding import PartitionSpec
    from torch_tpu._internal.pallas import jax_op
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.distributed.ep_mesh import (
        EP_AXIS_NAME,
        _ep_mesh_order,
        build_ep_mesh,
        ep_device_ids,
        get_ep_group,
    )
    from vllm_torchtpu.kernels.closure_op_name import closure_op_name

    tp = get_tp_group()
    ep = get_ep_group()
    why = []
    if int(tp.world_size) != N_MEMBERS:
        why.append(f"TP world {tp.world_size} != {N_MEMBERS}")
    if ep is None:
        why.append("EP is off (the op runs over the EP mesh)")
    elif sorted(int(r) for r in ep.ranks) != sorted(int(r) for r in tp.ranks):
        why.append(f"EP ranks {tuple(ep.ranks)} are not the TP ranks {tuple(tp.ranks)}")
    mesh = None
    if not why:
        mesh = build_ep_mesh()
        if mesh is None or mesh.shape[EP_AXIS_NAME] != N_MEMBERS:
            why.append(f"EP mesh {None if mesh is None else dict(mesh.shape)}")
    if why:
        raise RuntimeError(f"{ENV}={m} set but the decode TP all-reduce kernel cannot arm: {'; '.join(why)}")
    # Mesh order exactly as build_ep_mesh decides it (its devices may be abstract, without coordinates).
    unordered = ep_device_ids()
    device_ids = tuple(int(unordered[e]) for e in _ep_mesh_order(unordered))
    seen = tuple(getattr(d, "id", None) for d in mesh.devices.flat)
    if all(i is not None for i in seen) and tuple(int(i) for i in seen) != device_ids:
        raise RuntimeError(f"{ENV}: EP mesh device ids {seen} != the EP mesh order {device_ids}")
    pairs = pairs_from_device_ids(device_ids)
    wire = MODES[m]
    signature = (m, wire, rows_cap, pad, pairs, device_ids)
    _check_same_on_every_rank(signature, tp)
    spec = PartitionSpec(EP_AXIS_NAME)
    fn = global_fn(mesh, axis=EP_AXIS_NAME, pairs=pairs, wire=wire, max_rows_=rows_cap, pad16_=pad)
    name = closure_op_name("pallas::q38_tp_ar_decode", (mesh, pairs, wire, rows_cap, pad))
    # pyrefly: ignore[no-matching-overload]
    op = jax_op(name, fn, mesh=mesh, input_partition_specs=(spec,), output_partition_specs=spec)

    def _fake(x):
        return torch.empty_like(x)

    op.register_fake(_fake)
    hidden = _hidden_size(model_runner)
    _ARMED = (op, hidden)
    logger.info(
        "q38 AR-FIX decode TP all-reduce ARMED: %s=%s (H8, wire %s), rows %% 16 == 0 and <= %d take the kernel, "
        "others XLA psum in the same op; %s=%d (pad other row counts to 16 for H8); hidden %d; mesh device ids %s; "
        "pairs %s; op %s",
        ENV, m, wire, rows_cap, PAD16_ENV, int(pad), hidden, device_ids, pairs, name,
    )
    return True


def _hidden_size(model_runner: Any) -> int:
    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8

    cfg = getattr(getattr(model_runner, "model_config", None), "hf_text_config", None)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    if hidden <= 0:
        raise RuntimeError(f"{ENV}: no hidden_size on the runner's model config")
    h8.shard_table(hidden)  # raises for a hidden H8 cannot split
    return hidden


def use_kernel(x: torch.Tensor) -> bool:
    """True when the armed kernel takes ``x`` (bf16 [rows, hidden]). Static per graph: flag, rank, dtype, hidden;
    never the row count (rows are decided inside the op). False whenever the flag is off."""
    armed = _ARMED
    return armed is not None and x.dim() == 2 and x.dtype == torch.bfloat16 and int(x.shape[-1]) == armed[1]


def kernel_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """The armed op on ``x`` (callers check ``use_kernel`` first)."""
    # No logging or Python state here: this runs inside Dynamo's traced forward (a side effect would break the
    # graph). Engagement is the arm-time line plus the kernel name in the compiled program / xprof
    # (tp8_hier_all_reduce_fwd_<wire> in place of all-reduce).
    return _ARMED[0](x)  # type: ignore[index]
