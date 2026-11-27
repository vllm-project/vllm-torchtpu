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
"""o_proj of a sequence-parallel MiniMax-M3 step on the rank's token shard.

The usual SP attention exit reduce-scatters o_proj's per-rank partial sums,
bf16 [T, hidden]. Here the attention output (bf16 [T, heads/TP x head_dim])
goes through an all-to-all instead, so each rank holds every head of its
token rows, and o_proj contracts the full input dim on those rows: the same
MXU work, 1/6 of the bytes over ICI (TPU_MINIMAX_M3_SP_A2A_OPROJ).

With the flag on, every path (decode too) runs on the rescaled weight
prepare() leaves: elements shifted into e4m3's subnormal range lose mantissa
bits, so decode outputs change slightly as well.
"""

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.utils import replace_parameter

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import pallas_all_reduce
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.minimax_m3.step_halves import zero_of

# Set on the o_proj modules whose quant method prepares the SP operands, and
# once prepared.
PREPARE_ATTR = "_minimax_m3_sp_o_proj"
PREPARED_ATTR = "_minimax_m3_sp_o_proj_prepared"
_FP8_MAX = 448.0


def mode() -> str:
    return envs.TPU_MINIMAX_M3_SP_A2A_OPROJ


def mark(o_proj: nn.Module) -> None:
    if mode() != "off":
        setattr(o_proj, PREPARE_ATTR, True)


def applies(o_proj: nn.Module, num_tokens: int) -> bool:
    """Whether this o_proj takes the all-to-all path on an SP step of
    ``num_tokens`` tokens: prepared after loading from an e4m3 [K/TP, N]
    weight with a per-channel scale, the step at least
    TPU_MINIMAX_M3_SP_A2A_OPROJ_MIN_TOKENS, and the TP rank tensor it
    indexes blocks by is armed (else the reduce-scatter path runs on the
    prepared weight)."""
    if mode() == "off" or not getattr(o_proj, PREPARED_ATTR, False):
        return False
    if num_tokens < envs.TPU_MINIMAX_M3_SP_A2A_OPROJ_MIN_TOKENS:
        return False
    return pallas_all_reduce.armed_rank(get_tp_group().ranks) is not None


def _all_gather_cpu(t: torch.Tensor) -> torch.Tensor:
    tp = get_tp_group()
    local = t.cpu().contiguous()
    seen = [torch.empty_like(local) for _ in range(tp.world_size)]
    torch.distributed.all_gather(seen, local, group=tp.cpu_group)
    return torch.stack(seen)


def prepare(o_proj: nn.Module) -> None:
    """After loading (collective over TP): rescale this rank's e4m3 shard
    onto the per-channel scale shared by all ranks (both are powers of two,
    so only elements pushed below e4m3's range change) and make it the
    weight every path uses. "replicate" keeps every rank's rescaled shard,
    rotated so this rank's comes first: the weight becomes [TP, K/TP, N],
    block j rank (rank + j) % TP's, and the plain path reads block 0.

    Every trace reads the same parameters (compilation/shape_variants.py
    hands buckets between the SP and the plain trace only then)."""
    if not (
        o_proj.weight.dtype == torch.float8_e4m3fn
        and getattr(o_proj, WEIGHT_FLIPPED_ATTR, False)
    ):
        return
    device = o_proj.weight.device
    scale = o_proj.weight_scale.data
    common = _all_gather_cpu(scale.float()).amax(dim=0)
    shift = (scale.float().cpu() / common).to(device)
    w = (o_proj.weight.data.float() * shift).to(torch.float8_e4m3fn)
    if mode() == "replicate":
        # gloo carries no 8-bit float: send the bytes as int32.
        w = _all_gather_cpu(w.cpu().view(torch.int32)).view(torch.float8_e4m3fn)
        w = torch.roll(w, -get_tp_group().rank_in_group, dims=0)
    replace_parameter(o_proj, "weight", nn.Parameter(w.to(device), requires_grad=False))
    replace_parameter(
        o_proj,
        "weight_scale",
        nn.Parameter(common.to(scale.dtype).to(device), requires_grad=False),
    )
    setattr(o_proj, PREPARED_ATTR, True)


def _rank() -> torch.Tensor:
    """This rank's TP index as a tensor (int32 [1, 1]), so every rank traces
    the same program."""
    rank = pallas_all_reduce.armed_rank(get_tp_group().ranks)
    if rank is None:
        raise RuntimeError(
            "TPU_MINIMAX_M3_SP_A2A_OPROJ reads rank-indexed blocks by the rank of the "
            "armed TP all-reduce (TPU_TP_PALLAS_ALL_REDUCE_MAX_BYTES), on a mesh in "
            "rank order"
        )
    return rank


def _rank_block(x: torch.Tensor) -> torch.Tensor:
    """Block ``rank`` of ``x``'s leading axis."""
    return torch.index_select(x, 0, _rank().view(1)).squeeze(0)


def _rotate_blocks(x: torch.Tensor) -> torch.Tensor:
    """x [TP, ...] by source rank -> block j is rank (rank + j) % TP's, the
    order of a replicated weight."""
    tp = x.shape[0]
    order = torch.arange(tp, dtype=torch.int32, device=x.device) + _rank().view(1)
    return torch.index_select(x, 0, torch.remainder(order, tp))


def gathers_with_attention(attention: nn.Module, num_tokens: int) -> bool:
    """Whether `attention`'s MSA op gathers its o_proj's weight blocks under
    its attention (the ``weight_block`` of the 4-bit DCP op over several KV
    heads) for an SP step of ``num_tokens`` tokens, in place of
    `sp_o_proj`'s all-gather, which XLA issued only after the attention."""
    o_proj = attention.o_proj
    return (
        mode() == "gather"
        and getattr(attention.msa_op, "gathers_weight", False)
        and applies(o_proj, num_tokens)
        and o_proj.weight.dim() == 2
        # The op gathers over the whole world, in TP rank order.
        and get_tp_group().world_size == torch.distributed.get_world_size()
    )


def weight_block(o_proj: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's e4m3 [K/TP, N] weight block and its TP index (int32 [1]),
    for an op that gathers every rank's (`gathers_with_attention`)."""
    return local_weight(o_proj), _rank().view(1)


def local_weight(o_proj: nn.Module) -> torch.Tensor:
    """This rank's e4m3 [K/TP, N] block of a prepared o_proj's weight."""
    w = o_proj.weight
    return w if w.dim() == 2 else w[0]


def local_rows(x: torch.Tensor) -> torch.Tensor:
    """This rank's row shard of x [T, ...] (the reduce-scatter's layout)."""
    tp = get_tp_group().world_size
    return _rank_block(x.view(tp, x.shape[0] // tp, *x.shape[1:]))


def _full_weight(
    o_proj: nn.Module, tp: int, after: torch.Tensor | None = None
) -> torch.Tensor:
    """e4m3 [TP, K/TP, N]: every rank's rescaled block, gathered once
    ``after`` is computed (its all-gather then cannot start before)."""
    w = o_proj.weight
    if w.dim() == 3:
        return w
    if after is not None:
        w = _after_op(w, after)
    w = tensor_model_parallel_all_gather(w, 0)
    return w.view(tp, -1, w.shape[-1])


def all_to_all_rows(x: torch.Tensor) -> torch.Tensor:
    """Row block ``d`` of ``x`` goes to TP rank ``d``; row block ``r`` of the
    result came from rank ``r``."""
    tp = get_tp_group()
    out = funcol.all_to_all_single(x.contiguous(), None, None, tp.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


def _quantize_rows_pow2(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """e4m3 rows of bf16 ``x`` and their power-of-two f32 scales."""
    amax = jnp.max(jnp.abs(x.astype(jnp.float32)), axis=-1, keepdims=True)
    scale = jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(amax, 1e-30) / _FP8_MAX)))
    return (x.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn), scale


def _o_proj_e4m3(q: jax.Array, r: jax.Array, w: jax.Array, s: jax.Array) -> jax.Array:
    """q e4m3 [E, T, I] with power-of-two row scales r [E, T, 1], every rank's
    heads of T rows; w e4m3 [E, I, N] with s [N]: bf16 [T, N]. The rows are
    shifted onto their largest scale (exact but for underflow), so one e4m3 x
    e4m3 dot contracts (E, I)."""
    top = jnp.max(r, axis=0)
    q = (q.astype(jnp.float32) * (r / top)).astype(jnp.float8_e4m3fn)
    acc = jax.lax.dot_general(
        q, w, (((0, 2), (0, 1)), ((), ())), preferred_element_type=jnp.float32
    )
    return (acc * top * s.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


# Built at import: torch_tpu cannot create a jax_op inside a traced forward.
_quantize_op = pallas.jax_op(
    "pallas::minimax_m3_sp_quantize_rows_pow2", _quantize_rows_pow2
)
_quantize_op.register_fake(
    lambda x: (
        x.new_empty(x.shape, dtype=torch.float8_e4m3fn),
        x.new_empty((*x.shape[:-1], 1), dtype=torch.float32),
    )
)


def _after(x: jax.Array, dep: jax.Array) -> jax.Array:
    """x, once dep is computed: an XLA optimization barrier over both, no
    copy."""
    return jax.lax.optimization_barrier((x, dep))[0]


_after_op = pallas.jax_op("pallas::minimax_m3_sp_after", _after)
_after_op.register_fake(lambda x, dep: torch.empty_like(x))
_o_proj_e4m3_op = pallas.jax_op("pallas::minimax_m3_sp_o_proj_e4m3", _o_proj_e4m3)
_o_proj_e4m3_op.register_fake(
    lambda q, r, w, s: q.new_empty((q.shape[1], w.shape[2]), dtype=torch.bfloat16)
)


def sp_o_proj(
    o_proj: nn.Module,
    attn: torch.Tensor,
    pieces: int = 1,
    after: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    weight_after: torch.Tensor | None = None,
) -> torch.Tensor:
    """o_proj(attn) summed over TP, for this rank's row shard: ``attn`` is
    this rank's heads for all T rows, bf16 [T, K/TP]; returns [T/TP, N].
    TPU_MINIMAX_M3_SP_A2A_E4M3 sends the heads as e4m3 rows (power-of-two
    row scales) and contracts e4m3 x e4m3, one all-to-all and o_proj per
    row piece of the shard (``pieces``), so a piece's o_proj can run while
    the next piece crosses; the bf16 path ignores ``pieces``. The projection
    of the received rows runs after ``after`` is computed (a data
    dependence), so the all-to-all can cross under the work producing it.
    ``weight``: every rank's blocks [TP, K/TP, N], already gathered
    (`gathers_with_attention`); else they are gathered here, not before
    ``weight_after`` is computed (the step's last gathered attention input:
    the weight's all-gather then shares the links with the attention, not
    with the input gathers that attention waits for)."""
    tp = get_tp_group().world_size
    rows, k = attn.shape
    w = _full_weight(o_proj, tp, weight_after) if weight is None else weight
    # A replicated weight starts at this rank's block: rotate the heads alike.
    order = _rotate_blocks if o_proj.weight.dim() == 3 else (lambda x: x)
    if envs.TPU_MINIMAX_M3_SP_A2A_E4M3:
        q, r = _quantize_op(attn)
        # Piece i: row block i of every destination's shard.
        q = q.view(tp, pieces, -1, k)
        r = r.view(tp, pieces, -1, 1)
        out = []
        for i in range(pieces):
            qi, ri = q[:, i], r[:, i]
            piece_rows = qi.shape[1]
            qi = order(all_to_all_rows(qi.reshape(-1, k)).view(tp, piece_rows, k))
            ri = order(all_to_all_rows(ri.reshape(-1, 1)).view(tp, piece_rows, 1))
            if after is not None:
                ri = ri - zero_of(after)
            # The weight is the op's first reader of o_proj's parameters, as in
            # the plain trace (placeholders follow first use; shape_variants
            # compares).
            out.append(_o_proj_e4m3_op(qi, ri, w, o_proj.weight_scale))
        return out[0] if pieces == 1 else torch.cat(out)
    heads = order(all_to_all_rows(attn).view(tp, rows // tp, k))
    if after is not None:
        heads = heads - zero_of(after).to(heads.dtype)
    weight = w.to(torch.bfloat16) * o_proj.weight_scale.to(torch.bfloat16)
    return torch.einsum("eti,eio->to", heads, weight)
