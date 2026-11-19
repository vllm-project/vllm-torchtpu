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


def kv_cache_layer_buckets(kv_cache_spec: dict) -> list[list[str]]:
    """Same-type layer buckets of a uniform-page KV cache plan, as the TPU group-size
    patch builds them (vllm_torchtpu._patch_vllm_kv_cache_group_size): layers with
    equal specs share a bucket, and a bucket absorbs another spec its first spec's
    ``merge`` accepts, except across a ``q38_kv_bucket`` boundary.

    ``q38_kv_bucket`` (a class attribute, None by default): specs whose values differ
    never share a bucket. q38 F2 marks the producer's MTP draft specs with it
    (models/vllm/qwen4_exp/config.non_prefix_cacheable): vLLM's merge accepts a
    subclass with equal fields, so without it the draft joined the target's buckets
    (13 layers against 36 GDN), the GCD rule fell to one layer per group, and a
    pipeline stage's unified pool grew several-fold (F2 smoke attempt 3, j-58a40b86)."""
    from collections import defaultdict

    same_type_layers = defaultdict(list)
    for layer_name, layer_spec in kv_cache_spec.items():
        same_type_layers[layer_spec].append(layer_name)
    layer_buckets: list[list[str]] = []
    spec_buckets: list[list] = []
    for layer_spec, layer_names in same_type_layers.items():
        tag = getattr(layer_spec, "q38_kv_bucket", None)
        for names, specs in zip(layer_buckets, spec_buckets):
            if getattr(specs[0], "q38_kv_bucket", None) != tag:
                continue
            try:
                type(specs[0]).merge([*specs, layer_spec])
            except (AssertionError, ValueError):
                continue
            names.extend(layer_names)
            specs.append(layer_spec)
            break
        else:
            layer_buckets.append(list(layer_names))
            spec_buckets.append([layer_spec])
    return layer_buckets
