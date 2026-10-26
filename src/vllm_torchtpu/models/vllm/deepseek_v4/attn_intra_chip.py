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
"""Intra-chip TP for DSv4 attention weights, with KV and attention kept DP.

`VllmDeepseekV4MLAAttentionIntraChip` splits wq_b, wo_a, and wo_b across the two
chiplets within a chip, only applicable for chips like Ironwood that have two
chiplets per chip. Each chiplet keeps half the heads of `wq_b` and half the
o-groups of `wo_a`/`wo_b`.
To address the weights being sharded, during runtime the chiplets exchange their
activations back and forth via manually inserted collectives.

T0 and T1 are the tokens of chiplet 0 and chiplet 1. Specifically, each chiplet
does the following:
  1. All-gather `qr` with the neighboring chiplet.
     Chiplet 0 has [T0, T1] and Chiplet 1 has [T0, T1]
  2. Run WQ_B projection on gathered values.
     Now each chiplet has q for half of the heads, for both chiplets' tokens.
     Chiplet 0 has [T0_(0-63h), T1_(0-63h)] and Chiplet 1 has
     [T0_(64-127h), T1_(64-127h)].
  3. Swap the tokens post the projection.
     Chiplet 0 has [T0_(0-127h)] and Chiplet 1 has [T1_(0-127h)]
  4. Perform attention operations for the layer, doing (QK_T)V to get output.
  5. Swap the attention output, as `wo_a`/`wo_b` only contain some of the heads.
     Gather positions, since each chiplet applies inverse RoPE to both chiplets'
     tokens.
     Before Swap: Chiplet 0 has [Out0_(0-127h)], Chiplet 1 has [Out1_(0-127h)]
     After Swap: Chiplet 0 has [Out0_(0-63h), Out1_(0-63h)],
     Chiplet 1 has [Out0_(64-127h), Out1_(64-127h)]
  6. Each chiplet performs fused inverse-RoPE + `wo_a` on 8 groups.
     Chiplet 0 has [Out0_(0-7g), Out1_(0-7g)],
     Chiplet 1 has [Out0_(8-15g), Out1_(8-15g)]
     Then each chiplet does `wo_b` projection, resulting in partials.
     Chiplet 0 has [Out0_P0, Out1_P0], Chiplet 1 has [Out0_P1, Out1_P1]
  7. Each chiplet performs reduce-scatter to get final output.
     Chiplet 0 has [Out0_P0 + Out0_P1], Chiplet 1 has [Out1_P0 + Out1_P1]

Experimental: `DeepseekV4DecoderLayer` uses this class when
TPU_DSV4_ATTN_INTRA_CHIP_TP=1. TPU_DSV4_PAIR_SWAP fuses step 5 into step 6: the
peer's rows are fetched by remote DMA inside the `wo_a` kernel.
"""

from __future__ import annotations

import functools

import jax
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.deepseek_v4.o_projection import (
    PAIR_MAX_TOKENS,
    fused_reverse_rope_wo_a_projection_pair_swap,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
    VllmDeepseekV4MLAAttention,
    _pallas_op_cache,
)

logger = init_logger(__name__)


def _chip_pair_group() -> parallel_state.GroupCoordinator:
    """The intra-chip group, which must exist and pair exactly two cores."""
    from vllm_torchtpu.distributed.intra_chip import get_intra_chip_group

    group = get_intra_chip_group()
    if group is None or group.world_size != 2:
        raise RuntimeError(
            "TPU_DSV4_ATTN_INTRA_CHIP_TP needs a 2-core intra-chip group, got "
            f"{None if group is None else group.world_size}; see the intra-chip "
            "grouping warning above"
        )
    return group


def _load_current_chiplet_slice(layer: torch.nn.Module, rank: int, size: int) -> None:
    """Make `layer` load slice `rank` of `size` from the full checkpoint tensors.

    vLLM's loaders slice by `tp_rank`/`tp_size`, kept on the layer (unquantized
    weights) and on each quantized parameter (block-fp8 weights and scales).
    Both are read from the global TP group at construction, which is size 1
    here, so they are set to this core's place in the chip pair instead.
    """
    # torch's Module.__setattr__ stub only accepts modules and tensors.
    # pyrefly: ignore[bad-argument-type]
    layer.tp_rank, layer.tp_size = rank, size
    for param in layer.parameters():
        if hasattr(param, "tp_rank"):
            # vLLM adds tp_rank/tp_size to quantized parameters at runtime.
            # pyrefly: ignore[missing-attribute]
            param.tp_rank, param.tp_size = rank, size


def _swap_with_peer_chiplet(group, mine: torch.Tensor, theirs: torch.Tensor):
    """Send `theirs` to the peer core; return what the peer sent us.

    A 2-rank all-gather that keeps the peer's half.
    """
    rank = group.rank_in_group
    gathered = group.all_gather(theirs.contiguous(), dim=0)
    return gathered.view(2, *theirs.shape)[1 - rank]


def _on_pair_mesh(kernel, mesh: Mesh, *args):
    """Run `kernel(*shards)` on every core of `mesh`, each core's operands its
    own shard (the rank-local tensors under the op's specs)."""
    (ax,) = mesh.axis_names
    spec = P(ax)
    args = tuple(jax.device_put(a, NamedSharding(mesh, spec)) for a in args)
    return jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(spec,) * len(args),
        out_specs=spec,
        check_vma=False,
    )(*args)


def _wo_a_pair_swap_fn(mesh: Mesh, head_dim: int):
    kernel = functools.partial(
        fused_reverse_rope_wo_a_projection_pair_swap, head_dim=head_dim
    )

    def fn(
        o: jax.Array,
        positions: jax.Array,
        cos_sin_cache: jax.Array,
        wo_a: jax.Array,
        wo_a_scale: jax.Array,
        peer: jax.Array,
        rank: jax.Array,
    ) -> jax.Array:
        return _on_pair_mesh(
            kernel, mesh, o, positions, cos_sin_cache, wo_a, wo_a_scale, peer, rank
        )

    return fn


def _peer_device_id(my_device: int) -> int:
    """The other core of `my_device`'s chip."""
    from vllm_torchtpu.distributed.chip_topology import get_chip_topology

    topology = get_chip_topology()
    chip = next(ids for ids in topology.chips if my_device in ids)
    (peer,) = [d for d in chip if d != my_device]
    return peer


class _PairSwap:
    """The chip-pair wo_a op, which addresses the peer core directly.

    It runs on the EP mesh -- the one the fused MoE kernel uses, since every
    Pallas kernel in a program must share one device assignment -- and takes
    the peer's mesh index and my index in the pair as device data. Built at
    weight load: the mesh's first construction is a collective.
    """

    def __init__(self, group, device, head_dim: int) -> None:
        from vllm_torchtpu.distributed.ep_mesh import build_ep_mesh
        from vllm_torchtpu.distributed.mesh_utils import _get_tpu_global_device_id

        mesh = build_ep_mesh()
        if mesh is None:
            raise RuntimeError("chip-pair kernels need the expert-parallel mesh")
        ids = [int(d.id) for d in mesh.devices.flat]
        me = _get_tpu_global_device_id()
        peer = ids.index(_peer_device_id(me))
        as_i32 = functools.partial(torch.tensor, dtype=torch.int32, device=device)
        logger.info_once(
            "chip-pair wo_a in-kernel o exchange armed (device %d, peer mesh "
            "index %d, pair rank %d)",
            me,
            peer,
            group.rank_in_group,
        )
        self.peer = as_i32([peer])
        self.rank = as_i32([group.rank_in_group])
        name = f"pallas::deepseek_v4_wo_a_pair_swap_d{head_dim}"
        op = _pallas_op_cache.get(name)
        if op is None:
            (ax,) = mesh.axis_names
            op = pallas.jax_op(
                name,
                _wo_a_pair_swap_fn(mesh, head_dim),
                mesh=mesh,
                input_partition_specs=(P(ax),) * 7,
                output_partition_specs=P(ax),
            )
            op.register_fake(
                lambda o, pos, _cache, wo_a, *_: o.new_empty(
                    (pos.shape[0], wo_a.shape[1])
                )
            )
            _pallas_op_cache[name] = op
        self._op = op

    def __call__(self, o, positions_all, cos_sin_cache, wo_a, wo_a_scale):
        """wo_a over [core 0 tokens; core 1 tokens] with my heads, the peer's
        tokens fetched from its `o` inside the kernel."""
        return self._op(
            o, positions_all, cos_sin_cache, wo_a, wo_a_scale, self.peer, self.rank
        )


class VllmDeepseekV4MLAAttentionIntraChip(VllmDeepseekV4MLAAttention):
    """DSv4 attention with `wq_b`/`wo_a`/`wo_b` split across a chip's two cores."""

    # Set with object.__setattr__ in __init__, outside nn.Module's registry.
    _chip_pair: parallel_state.GroupCoordinator

    def __init__(self, vllm_config, prefix: str = "", *args, **kwargs) -> None:
        # The split halves a layer that holds all heads; under vLLM TP>1 the
        # heads are already split, so the halves would be of a shard.
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise RuntimeError(
                "TPU_DSV4_ATTN_INTRA_CHIP_TP requires tensor parallel size 1, "
                f"got {tp_size}"
            )
        # Every layer exchanges activations with the peer core, so both cores
        # must run each forward together at the same padded token count. The
        # runner keeps DP ranks in that lockstep only under expert parallel
        # (TPUModelRunner._dp_lockstep_enabled); otherwise each rank pads to its
        # own bucket, an idle one may skip the forward, and the exchange hangs
        # or pairs tensors from different steps.
        from vllm_torchtpu.utils import get_dp_size

        parallel_config = vllm_config.parallel_config
        if not (
            get_dp_size(parallel_config) > 1 and parallel_config.enable_expert_parallel
        ):
            raise RuntimeError(
                "TPU_DSV4_ATTN_INTRA_CHIP_TP requires DP lockstep (data parallel "
                "size > 1 with expert parallel), so both cores of a chip run every "
                "forward at the same token count"
            )
        super().__init__(vllm_config, prefix, *args, **kwargs)
        group = _chip_pair_group()
        rank, size = group.rank_in_group, group.world_size
        quant_config = vllm_config.quant_config
        # Rebuilt half-size; the replicated ones from the base are dropped.
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
            tp_rank=rank,
            tp_size=size,
        )
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
            tp_rank=rank,
            tp_size=size,
        )
        # torch's Module.__setattr__ stub only accepts modules and tensors.
        # pyrefly: ignore[bad-argument-type]
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_groups // size
        # RowParallelLinear takes no tp_rank/tp_size, so it is built at current
        # chiplet's half of the input rows with TP off and pointed at its slice
        # below.
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank // size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
            disable_tp=True,
        )
        for layer in (self.wq_b, self.wo_a, self.wo_b):
            _load_current_chiplet_slice(layer, rank, size)
        object.__setattr__(self, "_chip_pair", group)
        self._pair_tokens = 2 * vllm_config.scheduler_config.max_num_batched_tokens
        logger.info_once(
            "ATTN_INTRA_CHIP_TP: wq_b/wo_a/wo_b sharded across the chip pair"
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        # The base reshapes wo_a by the local head and group counts, which are
        # halved for these weights but not for attention, so swap them in here.
        size = self._chip_pair.world_size
        heads, groups = self.n_local_heads, self.n_local_groups
        self.n_local_heads, self.n_local_groups = heads // size, groups // size
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            self.n_local_heads, self.n_local_groups = heads, groups
        if envs.TPU_DSV4_PAIR_SWAP:
            if self._pair_tokens > PAIR_MAX_TOKENS:
                raise ValueError(
                    "TPU_DSV4_PAIR_SWAP covers one token tile: 2 x "
                    f"max_num_batched_tokens = {self._pair_tokens} > "
                    f"{PAIR_MAX_TOKENS}"
                )
            swap = _PairSwap(
                self._chip_pair, self.wo_b.weight.device, int(self.head_dim)
            )
            object.__setattr__(self, "_pair_swap", swap)

    def _project_q(self, qr: torch.Tensor) -> torch.Tensor:
        # Each core projects both cores' tokens onto its half of the heads, then
        # the cores swap so each holds all heads for its own tokens.
        group = self._chip_pair
        rank = group.rank_in_group
        num_tokens = qr.shape[0]
        qr_pair = group.all_gather(qr.contiguous(), dim=0)
        if qr_pair.shape[0] != 2 * num_tokens:
            raise RuntimeError(
                f"Chip pair gathered {qr_pair.shape[0]} qr rows, expected "
                f"2 x {num_tokens}; both cores must run at the same token count"
            )
        q_half = self.wq_b(qr_pair).view(2, num_tokens, -1)
        peer = _swap_with_peer_chiplet(group, q_half[rank], q_half[1 - rank])
        pieces = [q_half[rank], peer] if rank == 0 else [peer, q_half[rank]]
        return torch.cat(pieces, dim=-1)

    def _o_proj(self, o, positions):
        group = self._chip_pair
        rank = group.rank_in_group
        num_tokens = o.shape[0]
        half = self.n_local_heads // 2
        positions_all = group.all_gather(
            positions.to(torch.int32).contiguous(), dim=0
        ).to(positions.dtype)
        if envs.TPU_DSV4_PAIR_SWAP:
            z = self._pair_swap(
                o.reshape(num_tokens, self.n_local_heads, self.head_dim),
                positions_all,
                self.rotary_emb.cos_sin_cache,
                self.wo_a.weight,
                self.wo_a.weight_scale,
            )
        else:
            o = o.reshape(num_tokens, 2, half, self.head_dim)
            peer = _swap_with_peer_chiplet(group, o[:, rank], o[:, 1 - rank])
            pieces = [o[:, rank], peer] if rank == 0 else [peer, o[:, rank]]
            o_half = torch.cat(pieces, dim=0).contiguous()
            z = self.o_proj_op(
                o_half,
                positions_all,
                self.rotary_emb.cos_sin_cache,
                self.wo_a.weight,
                self.wo_a.weight_scale,
            )
        partial = self.wo_b(z)
        if isinstance(partial, tuple):
            partial = partial[0]
        return group.reduce_scatter(partial.contiguous(), dim=0)
