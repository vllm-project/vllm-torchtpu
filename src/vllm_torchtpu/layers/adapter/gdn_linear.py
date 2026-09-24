# SPDX-License-Identifier: Apache-2.0
"""GDN projections with whole-head TP sharding and Q/K replication."""

import torch
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    adjust_block_scale_shard,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)

from vllm_torchtpu.kernels.gdn.head_geometry import GdnHeadGeometry


class GdnColumnParallelLinear(MergedColumnParallelLinear):
    """Merged Q/K/V[/Z] projection using the same loader for every TP size.

    Checkpoint widths describe unique heads. Allocation widths include Q/K
    replicas, as in vLLM's QKVParallelLinear. Keeping these separate is also
    necessary for the Qwen3.5 mapper's fused QKV checkpoint plus separate Z.
    Parameter creation, quantization and forward remain in the parent layer.
    """

    def __init__(
        self,
        *,
        input_size: int,
        geometry: GdnHeadGeometry,
        d_k: int,
        d_v: int,
        include_z: bool,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        self.checkpoint_sizes = [geometry.num_kq_heads * d_k] * 2 + [
            geometry.num_v_heads * d_v
        ] * (1 + include_z)
        self.replication_factors = [geometry.kq_replication_factor] * 2 + [1] * (
            1 + include_z
        )
        super().__init__(
            input_size=input_size,
            output_sizes=[
                size * replicas
                for size, replicas in zip(
                    self.checkpoint_sizes, self.replication_factors
                )
            ],
            bias=False,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def _parameter_slice(
        self, param: BasevLLMParameter, size: int, offset: int
    ) -> tuple[int, int]:
        """Translate output-channel coordinates to this parameter's storage."""
        if isinstance(param, BlockQuantScaleParameter):
            return adjust_block_scale_shard(self.weight_block_size, size, offset)
        return size, offset

    def weight_loader(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        if isinstance(param, PerTensorScaleParameter):
            # Per-tensor weight/input scales use logical segment ids, not
            # channel slices. vLLM already handles scalar-to-fused loading.
            return super().weight_loader_v2(param, loaded_weight, loaded_shard_id)

        output_dim = param.output_dim
        if not isinstance(loaded_shard_id, int):
            shard_ids = (
                tuple(range(len(self.checkpoint_sizes)))
                if loaded_shard_id is None
                else loaded_shard_id
            )
            offset = 0
            for shard_id in shard_ids:
                size, storage_offset = self._parameter_slice(
                    param, self.checkpoint_sizes[shard_id], offset
                )
                self.weight_loader(
                    param,
                    loaded_weight.narrow(output_dim, storage_offset, size),
                    shard_id,
                )
                offset += self.checkpoint_sizes[shard_id]
            return

        size = self.output_partition_sizes[loaded_shard_id]
        offset = sum(self.output_partition_sizes[:loaded_shard_id])
        size, offset = self._parameter_slice(param, size, offset)
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            shard_size=size,
            shard_offset=offset,
            # Use vLLM's replicated-column loader (the FA K/V operation).
            # GDN Q/K use R replicas, V/Z use R=1; the actual GDN segment id
            # has already determined the destination slice above.
            shard_id="k",
            num_heads=self.replication_factors[loaded_shard_id],
        )

    # vLLM selects the entry point by quantization method, but both entries
    # must apply precisely the same head ownership and checkpoint splitting.
    weight_loader_v2 = weight_loader


class GdnInterleavedColumnParallelLinear(MergedColumnParallelLinear):
    """Shard checkpoint groups while keeping complete shared Q/K heads.

    Each group contains Q/K and its associated V/Z channels, or B/A channels.
    Only the latter channels split when several TP ranks share a Q/K head.
    The rank-local parameter uses canonical Q/K/V/Z or B/A segment ordering;
    checkpoint interleaving never reaches the compute path.
    """

    def __init__(
        self,
        *,
        geometry: GdnHeadGeometry,
        group_sizes: list[int],
        replicated_segments: list[bool],
        input_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self.geometry = geometry
        self.group_sizes = group_sizes
        self.replicated_segments = replicated_segments
        self.local_group_sizes = [
            size if replicated else size // geometry.kq_replication_factor
            for size, replicated in zip(group_sizes, replicated_segments)
        ]
        super().__init__(
            input_size=input_size,
            output_sizes=[
                sum(self.local_group_sizes)
                * geometry.local_num_kq_heads
                * geometry.parallel_size
            ],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        self.validate_shard_id(loaded_shard_id)
        if isinstance(param, PerTensorScaleParameter):
            return super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        geometry = self.geometry
        sizes = self.group_sizes
        local_sizes = self.local_group_sizes
        if isinstance(param, BlockQuantScaleParameter):
            block = self.weight_block_size[0]
            if any(size % block for size in sizes + local_sizes):
                raise ValueError(
                    "GDN interleaved scale blocks must align "
                    "with rank-local head segments."
                )
            sizes = [size // block for size in sizes]
            local_sizes = [size // block for size in local_sizes]
        if isinstance(param, ChannelQuantScaleParameter):
            loaded_weight = loaded_weight.reshape(-1, *param.shape[1:])
        source = loaded_weight.movedim(param.output_dim, 0)
        groups = source.reshape(geometry.num_kq_heads, sum(sizes), *source.shape[1:])
        first = geometry.kq_shard_index(self.tp_rank) * geometry.local_num_kq_heads
        groups = groups.narrow(0, first, geometry.local_num_kq_heads)
        replica = self.tp_rank % geometry.kq_replication_factor
        pieces = [
            piece.narrow(1, 0 if shared else replica * width, width)
            for piece, width, shared in zip(
                groups.split(sizes, dim=1), local_sizes, self.replicated_segments
            )
        ]
        # Flatten each segment's head groups before concatenating segments.
        # Scales take exactly the same permutation as their weight rows.
        local = torch.cat([piece.flatten(0, 1) for piece in pieces], dim=0).movedim(
            0, param.output_dim
        )
        assert local.shape == param.shape
        param.data.copy_(local)

    weight_loader_v2 = weight_loader
