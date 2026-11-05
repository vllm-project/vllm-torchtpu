# SPDX-License-Identifier: Apache-2.0
"""KV cache group size rule (see _patch_vllm_kv_cache_group_size)."""


def kv_cache_group_size(bucket_sizes: list[int], rule: str = "gcd") -> int:
    """Layers per KV cache group for buckets of same-type layers.

    "min" is upstream vLLM: the smallest bucket, widened to the largest when
    that is under 1.5x it. A lone layer of its own kind (Qwen3.8's PLE)
    then forces one group per layer, so every cached block takes one block
    ID per layer. "gcd" sizes groups from the buckets of more than one layer
    (their GCD) and pads the small ones up to it: Qwen3.8 gets 12 (main KV,
    compressed keys and raw rings one group each, GDN three, PLE one with 11
    padding slots), as Qwen3.5 gets 15.
    """
    import math

    if not bucket_sizes:
        raise ValueError("KV cache grouping needs at least one layer bucket")
    if rule == "min":
        size = min(bucket_sizes)
        if max(bucket_sizes) < size * 1.5:
            size = max(bucket_sizes)
        return size
    if rule != "gcd":
        raise ValueError(f"Unknown TPU_KV_CACHE_GROUP_SIZE_RULE {rule!r}")
    multi = [n for n in bucket_sizes if n > 1]
    return math.gcd(*(multi or bucket_sizes))
