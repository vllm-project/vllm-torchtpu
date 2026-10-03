"""Shared vLLM backend contract for V4.1 attention and cache adapters."""

from vllm.models.deepseek_v41.compressor import CompressorBackend


class AttentionBackend(CompressorBackend):
    @staticmethod
    def get_name():
        return "TORCH_TPU_DEEPSEEK_V41"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, *args, **kwargs
    ):
        return num_blocks, num_kv_heads, block_size, head_size
