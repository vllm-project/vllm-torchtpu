import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ceil up to the nearest multiple of b.
def _align_to(a, b):
    return pl.cdiv(a, b) * b


@dataclasses.dataclass(frozen=True)
class Config:
    input_size: int
    hidden_size: int
    reduce_group_size: int
    topk_dtype: Any
    in_dtype: Any
    core_axis_name: str
    subcore_axis_name: str
    tpu_info: pltpu.TpuInfo

    def __post_init__(self):
        # Only supports either bf16 or fp32 for now.
        assert self.in_dtype in (jnp.bfloat16, jnp.float32)

    @property
    def sc_info(self):
        sc_info = self.tpu_info.sparse_core
        assert sc_info is not None
        return sc_info

    @property
    def num_tot_cores(self) -> int:
        return self.sc_info.num_cores * self.sc_info.num_subcores

    @property
    def num_row_subchunks(self) -> int:
        base_block_size = self.sc_info.num_lanes * self.num_row_partitions
        input_size_block = pl.cdiv(self.input_size, base_block_size)
        return max(1, min(4, input_size_block))

    @property
    def output_size(self) -> int:
        return self.input_size // self.reduce_group_size

    @property
    def in_dtype_bytes(self) -> int:
        return jax.dtypes.itemsize_bits(self.in_dtype) // 8

    @property
    def padded_input_size(self) -> int:
        align_val = self.num_row_partitions * self.reduce_group_size
        return _align_to(self.input_size, align_val)

    @property
    def should_fallback(self) -> bool:
        # For a small {input + output} both likely fit in TensorCore VMEM, where a
        # plain TC gather-reduce beats routing through SparseCore and HBM.
        if self.tpu_info.sparse_core is None:
            return True
        vmem_capacity_threshold = self.tpu_info.vmem_capacity_bytes * 0.6
        x_size = self.input_size * self.hidden_size * self.in_dtype_bytes
        # TODO(kyuyeunk): Improve fallback calculation logic.
        return x_size * 2 < vmem_capacity_threshold

    @property
    def row_partition_size(self) -> int:
        return self.padded_input_size // self.num_row_partitions

    @property
    def row_partition_size_padded(self) -> int:
        return _align_to(self.row_partition_size, self.row_chunk_size)

    @property
    def row_chunk_size(self) -> int:
        """Number of rows handled per row-pipeline block."""
        return self.sc_info.num_lanes * self.num_row_subchunks

    @property
    def num_col_chunks(self) -> int:
        return self.col_size // self.col_chunk_size

    @property
    def row_shift(self) -> int:
        """log2 of how many source rows pack into one uint32 gather element.

    The SparseCore indirect DMA requires 32-bit elements: bfloat16 packs two
    source rows per uint32 (shift 1), float32 is 1:1 (shift 0).
    """
        input_packing = 32 // jax.dtypes.itemsize_bits(self.in_dtype)
        return input_packing.bit_length() - 1

    @property
    def num_row_partitions(self) -> int:
        """Calculates the number of row partitions."""
        return self.num_tot_cores // self.num_column_partitions

    @property
    def num_column_partitions(self) -> int:
        """Calculates the number of row partitions."""
        # DMA constraint requires column partition to be multiple of lane size.
        # Prefer to use a large number of column partitions, as long as each
        # partition's size is not too small for DMA pipeline efficiency and each
        # partition's size can divide the hidden size.

        # Each column partition will do DMA pipelining on col_size.
        num_lanes = self.tpu_info.num_lanes
        preferred_num_stages = 4
        num_column_partitions = 1
        while (self.num_tot_cores % (num_column_partitions * 2) == 0
               and self.hidden_size %
               (num_lanes * num_column_partitions * 2) == 0
               and self.hidden_size // (num_column_partitions * 2 * num_lanes)
               >= preferred_num_stages):
            num_column_partitions *= 2
        return num_column_partitions

    @property
    def aligned_hidden_size(self) -> int:
        """Calculates the aligned hidden size."""
        num_lanes = self.tpu_info.num_lanes
        return _align_to(self.hidden_size,
                         num_lanes * self.num_column_partitions)

    @property
    def col_size(self) -> int:
        """Calculates the column size."""
        return self.aligned_hidden_size // self.num_column_partitions

    @property
    def col_chunk_size(self) -> int:
        """Picks the column chunk size the inner pipeline gathers at a time.

    The chunk is the largest divisor of ``col_size`` whose gather double-buffer
    still fits comfortably in SparseCore VMEM.
    """
        match self.tpu_info.generation:
            case 6:
                target_bytes = int(256 * 1024 * 0.95)
            case 7:
                target_bytes = int(512 * 1024 * 0.95)
            case _:
                target_bytes = int(128 * 1024 * 0.95)

        # uint32 gather buffer, double-buffered by emit_pipeline.
        num_simd_lanes = self.sc_info.num_lanes
        num_lanes = self.tpu_info.num_lanes
        bytes_per_col = num_simd_lanes * 4 * 2
        max_safe_col = _align_to(target_bytes // bytes_per_col, num_lanes)

        start_col = _align_to(min(self.col_size, max_safe_col), num_lanes)
        for chunk in range(start_col, num_lanes - 1, -num_lanes):
            if self.col_size % chunk == 0:
                return chunk
        return num_lanes
