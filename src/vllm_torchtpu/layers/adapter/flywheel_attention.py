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
"""--attention-backend flywheel: the FlyWheel kernels
(layers/core/flywheel_attention_interface.py) in the attention layers.

layers.register_layers registers both classes by path, so this module, and
flywheel_tpu with it, loads only once the backend is selected.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import jax.numpy as jnp
import numpy as np
import torch
from flywheel_tpu.flash_attn_interface import PAGE_SIZE_MULTIPLE
from flywheel_tpu.pallas.copy_utils import merged_cache_heads
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.kv_cache_layout import KVCacheLayout as VllmKVCacheLayout

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import (
    QWEN_GDN_ARCHITECTURES,
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.layers.adapter.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.attention_backend import (
    HostStepInputs,
    TPUAttentionBackend,
)
from vllm_torchtpu.layers.core import flywheel_attention_interface as fw

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class FlywheelMetadata(NamedTuple):
    """AttentionMetadata.backend_metadata of one attention KV cache group in
    one step: prepare_step's host arrays, which the runner uploads and the
    attention impl appends to the kernel call."""

    # i32[3 * bound]: the KV writer's (cache_row_start, new_row_start, length)
    # pieces, and i32[1]: their count.
    kv_write_slices: np.ndarray | torch.Tensor
    kv_write_num_slices: np.ndarray | torch.Tensor
    # i32[1]: the step's index into step_kernels(bucket).
    kernel_index: np.ndarray | torch.Tensor


class FlywheelAttentionBackend(TPUAttentionBackend):
    """The runner side of --attention-backend flywheel."""

    has_step_metadata = True

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Reject every configuration this backend cannot serve, listing each
        unsupported feature in one error."""
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        problems: list[str] = []
        if (
            cache_config.user_specified_block_size
            and cache_config.block_size % PAGE_SIZE_MULTIPLE
        ):
            problems.append(
                f"block size {cache_config.block_size} is not a multiple of "
                f"{PAGE_SIZE_MULTIPLE}; pass a --block-size that is, or leave "
                "it unset"
            )
        if model_config.use_mla:
            problems.append("MLA models are not supported")
        if model_config.is_hybrid:
            if model_config.architecture not in QWEN_GDN_ARCHITECTURES:
                problems.append(
                    "hybrid models other than Qwen GDN ones are not supported"
                )
            elif unified_kv_layout_enabled_for_architecture(model_config.architecture):
                problems.append(
                    "GDN on the unified block pool is not supported; set "
                    "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0"
                )
        if model_config.get_head_size() not in fw._DECODE_HEAD_DIMS:
            problems.append(
                f"head_dim {model_config.get_head_size()} is not one of the "
                f"decode kernel's {fw._DECODE_HEAD_DIMS}"
            )
        if model_config.dtype != torch.bfloat16:
            problems.append(f"model dtype {model_config.dtype} is not bfloat16")
        if (
            cache_config.cache_dtype
            not in PallasAttentionBackend.supported_kv_cache_dtypes
        ):
            problems.append(
                f"kv cache dtype {cache_config.cache_dtype!r} is not bfloat16 or fp8"
            )
        if vllm_config.speculative_config is not None:
            problems.append("speculative decoding is not supported")
        if vllm_config.kv_transfer_config is not None:
            problems.append("KV transfer (disaggregated serving) is not supported")
        if parallel_config.decode_context_parallel_size > 1:
            problems.append("decode context parallelism is not supported")
        if parallel_config.prefill_context_parallel_size > 1:
            problems.append("prefill context parallelism is not supported")
        if parallel_config.pipeline_parallel_size > 1:
            problems.append("pipeline parallelism is not supported")
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            problems.append("VLLM_TPU_BLOCK_MAJOR_KV is not supported; unset it")
        if problems:
            raise ValueError(
                "The flywheel attention backend cannot serve this configuration: "
                + "; ".join(problems)
                + "."
            )

    def conv1d_gdn_kernel(self) -> Callable[..., Any]:
        return fw.flywheel_conv1d_gdn

    def prepare_step(self, step: HostStepInputs) -> dict[int, FlywheelMetadata]:
        """Pick the step's kernel and build each attention group's KV write
        slice table. The decode kernel appends the leading decode rows itself,
        so the writer takes only the rows after them."""
        kernel_index = np.array(
            [
                fw.step_kernel_index(
                    step.num_decode,
                    step.query_start_loc,
                    step.seq_lens,
                    step.num_tokens,
                )
            ],
            dtype=np.int32,
        )
        metadata = {}
        for gid, page_size in step.page_sizes.items():
            # Sized by the largest bucket, unlike tpu-inference's (one
            # executable per bucket): HostStepInputs.max_num_tokens.
            slices = np.zeros(
                (
                    3,
                    fw.kv_write_slice_bound(
                        step.max_num_tokens, step.max_num_seqs, page_size
                    ),
                ),
                dtype=np.int32,
            )
            num_slices = fw.build_kv_write_slices(
                step.query_start_loc[step.num_decode :],
                step.seq_lens[step.num_decode :],
                step.block_tables[gid][step.num_decode :],
                page_size=page_size,
                out=slices,
            )
            metadata[gid] = FlywheelMetadata(
                slices.reshape(-1), np.array([num_slices], np.int32), kernel_index
            )
        return metadata

    def warmup_metadata(
        self, page_sizes: Mapping[int, int], *, max_num_seqs: int, max_num_tokens: int
    ) -> dict[int, FlywheelMetadata]:
        """All zeros: the writer copies nothing and the kernel index picks the
        decode branch."""
        zero = np.zeros(1, np.int32)
        return {
            gid: FlywheelMetadata(
                np.zeros(
                    3
                    * fw.kv_write_slice_bound(max_num_tokens, max_num_seqs, page_size),
                    np.int32,
                ),
                zero,
                zero,
            )
            for gid, page_size in page_sizes.items()
        }


class FlywheelPallasAttentionBackendImpl(PallasAttentionBackendImpl):
    """RPA's attention impl, running the FlyWheel kernels instead."""

    _kernel_entry = staticmethod(fw.flywheel_kernel_entry)
    _kernel_op_prefix = "pallas::flywheel_attention"


class FlywheelPallasAttentionBackend(PallasAttentionBackend):
    """The vLLM attention backend of --attention-backend flywheel."""

    @staticmethod
    def get_name() -> str:
        return "FLYWHEEL"

    @staticmethod
    def get_impl_cls() -> type[FlywheelPallasAttentionBackendImpl]:
        return FlywheelPallasAttentionBackendImpl

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[VllmKVCacheLayout, ...]:
        return (VllmKVCacheLayout.LBNHC,)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str | torch.dtype = "auto",
    ) -> tuple[int, ...]:
        # tpu-inference's: K and V rows interleave per KV head (bf16) or head
        # pair (fp8), without RPA's (head, K|V) split, which XLA copies whole
        # at one KV head per chip.
        dtype = cache_dtype_str
        if isinstance(dtype, str):  # "auto" probes only num_blocks' axis
            dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(dtype.lower(), torch.bfloat16)
        cache_heads = merged_cache_heads(
            num_kv_heads, jnp.dtype(str(dtype).removeprefix("torch."))
        )
        return (num_blocks, block_size, cache_heads, fw.align_head_dim(head_size))

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> list[int | MultipleOf]:
        return [MultipleOf(PAGE_SIZE_MULTIPLE)]

    @staticmethod
    def get_page_size(vllm_config: "VllmConfig") -> int:
        """RPA's page size, rounded up to whole pages the kernels can index."""
        page_size = PallasAttentionBackend.get_page_size(vllm_config)
        return cdiv(page_size, PAGE_SIZE_MULTIPLE) * PAGE_SIZE_MULTIPLE
