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
"""Head geometry shared by the GDN TP and PCP kernel paths."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class GdnHeadGeometry:
    """Validated rank-local GDN head geometry.

    Value heads are sharded across every parallel rank. Query/key heads are sharded
    when there are enough of them and replicated across adjacent ranks when
    their count is smaller than the parallel size.
    """

    num_kq_heads: int
    num_v_heads: int
    parallel_size: int
    num_unique_kq_shards: int
    local_num_kq_heads: int
    local_num_v_heads: int
    kq_replication_factor: int

    def kq_shard_index(self, rank: int) -> int:
        if not 0 <= rank < self.parallel_size:
            raise ValueError(
                f"rank={rank} must be in [0, {self.parallel_size}).")
        return rank // self.kq_replication_factor

    def local_conv_dim(self, d_k: int, d_v: int) -> int:
        if d_k <= 0 or d_v <= 0:
            raise ValueError(
                f"GDN head dimensions must be positive, got {d_k=} {d_v=}.")
        return (2 * self.local_num_kq_heads * d_k +
                self.local_num_v_heads * d_v)

    def local_state_shapes(
            self,
            shapes: tuple[tuple[int, ...], ...],
            d_k: int,
            d_v: int,
            *,
            conv_dim_axis: int = -1
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply head ownership, preserving tap/speculation and layout axes."""
        conv_shape, recurrent_shape = shapes
        conv_shape = list(conv_shape)
        conv_shape[conv_dim_axis] = self.local_conv_dim(d_k, d_v)
        return (tuple(conv_shape), (self.local_num_v_heads,
                                    *recurrent_shape[1:]))


def derive_gdn_head_geometry(
    num_kq_heads: int,
    num_v_heads: int,
    parallel_size: int,
) -> GdnHeadGeometry:
    """Derive the native sharding/replication relationship for GDN TP and PCP."""
    if num_kq_heads <= 0 or num_v_heads <= 0 or parallel_size <= 0:
        raise ValueError("GDN head counts and parallel size must be positive: "
                         f"{num_kq_heads=} {num_v_heads=} {parallel_size=}.")
    if num_v_heads % parallel_size:
        raise ValueError(f"num_v_heads={num_v_heads} must be divisible by "
                         f"parallel_size={parallel_size}.")

    num_unique_kq_shards = min(num_kq_heads, parallel_size)
    if num_kq_heads % num_unique_kq_shards:
        raise ValueError(
            f"num_kq_heads={num_kq_heads} must be divisible by "
            f"parallel_size={parallel_size} when Q/K heads are sharded.")
    if parallel_size % num_unique_kq_shards:
        raise ValueError(
            f"parallel_size={parallel_size} must be divisible by "
            f"num_kq_heads={num_kq_heads} when Q/K heads are replicated.")

    local_num_kq_heads = num_kq_heads // num_unique_kq_shards
    local_num_v_heads = num_v_heads // parallel_size
    if local_num_v_heads % local_num_kq_heads:
        raise ValueError(
            "Each rank-local Q/K head must own an integer number of V heads: "
            f"{local_num_kq_heads=} {local_num_v_heads=}.")

    return GdnHeadGeometry(
        num_kq_heads=num_kq_heads,
        num_v_heads=num_v_heads,
        parallel_size=parallel_size,
        num_unique_kq_shards=num_unique_kq_shards,
        local_num_kq_heads=local_num_kq_heads,
        local_num_v_heads=local_num_v_heads,
        kq_replication_factor=parallel_size // num_unique_kq_shards,
    )
