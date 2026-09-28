# SPDX-License-Identifier: Apache-2.0
"""Deviceless duck-typed tensor fakes for the Raiden pool-manifest tests.

Moved out of tpu_connector_v2_test_utils.py so that collecting the live
Raiden tests no longer imports the TPUConnectorV2 cluster at module scope.
FakeStorage and FakeTensor move as a pair: FakeTensor's default constructor
builds a FakeStorage, and FakeStorage._next_ptr is mutable class state that
must exist exactly once (two copies would hand out colliding data_ptr()
values and silently weaken the dead-storage check).
"""

import types


class FakeStorage:
    """Duck-typed untyped_storage() stand-in with a unique data pointer."""

    _next_ptr = 0x1000

    def __init__(self):
        FakeStorage._next_ptr += 0x100000000
        self._ptr = FakeStorage._next_ptr

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    """Duck-typed tensor for pool-manifest tests (no torch required)."""

    def __init__(
        self,
        shape,
        element_size,
        dtype="torch.fake",
        storage=None,
        storage_offset_elems=0,
    ):
        self.shape = tuple(shape)
        self._element_size = int(element_size)
        self.dtype = dtype
        self._storage = storage if storage is not None else FakeStorage()
        self._storage_offset = int(storage_offset_elems)
        numel = 1
        for dim in self.shape:
            numel *= dim
        self.nbytes = numel * self._element_size

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return self._storage_offset


def kimi_pool_manifest(heads, pool=None):
    """The stock full K3 unified pool for a local KDA head count."""
    from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
    from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

    state_shape = ((3, 3, heads, 128), (heads, 128, 128))
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=heads * 128 * 128 * 4,
        conv_bytes=3 * 3 * heads * 128 * 2,
        token_bytes=2560,
    )
    rows = layout.required_tokens
    if pool is None:
        pool = FakeTensor((4, rows, 2, 640), 2, dtype="torch.bfloat16")
    state_layer, fa_layer = "model.layers.1.kda", "model.layers.4.attn"
    groups = [
        types.SimpleNamespace(
            layer_names=[state_layer],
            kv_cache_spec=types.SimpleNamespace(
                shapes=state_shape,
                dtypes=("torch.float32", "torch.float32"),
                page_size_bytes=rows * 2560,
            ),
        ),
        types.SimpleNamespace(
            layer_names=[fa_layer],
            kv_cache_spec=types.SimpleNamespace(
                block_size=rows * 2, page_size_bytes=rows * 2560
            ),
        ),
    ]
    return rpm.build_kimi_k3_pool_manifest(
        named_kv_caches={state_layer: [pool], fa_layer: pool},
        kv_cache_groups=groups,
        raw_tensors=[pool],
        mamba_group_ordinal_by_layer={state_layer: 0},
    )


def glm_named_kv_caches(
    *,
    num_blocks=16,
    block_size_tokens=1024,
    nope_shape=None,
    rope_shape=None,
    idx_shape=None,
):
    """Named per-layer caches shaped like a GLM-5.2 materialization.

    Each MLA layer holds a (nope, rope) uint8 tensor pair and
    one  uint8 DSA indexer cache. Nope packs one token per
    [packing, width] row, rope packs `packing` tokens per row.
    """
    nope_shape = nope_shape or (num_blocks, block_size_tokens, 4, 128)
    rope_shape = rope_shape or (num_blocks, block_size_tokens // 4, 4, 128)
    idx_shape = idx_shape or (num_blocks, block_size_tokens // 4, 4, 256)
    named = {
        f"model.layers.{idx}.self_attn.mla_attn": (
            FakeTensor(nope_shape, 1, dtype="torch.uint8"),
            FakeTensor(rope_shape, 1, dtype="torch.uint8"),
        )
        for idx in range(2)
    }
    named["model.layers.0.self_attn.indexer"] = FakeTensor(
        idx_shape, 1, dtype="torch.uint8"
    )
    return named


# DSv4 page geometry, mirroring _allocate_ds_v4_kv_caches: a CSA layer pages
# 1024 tokens at compress_ratio 4, and VllmDeepseekV4SWACache._swa_block_size
# gives 1024 // 4 // 2 = 128 so the SWA page lands on the same byte size as the
# CSA NoPE page it overlays.
DSV4_CSA_BLOCK_TOKENS = 1024
DSV4_CSA_COMPRESS_RATIO = 4
DSV4_SWA_BLOCK_TOKENS = DSV4_CSA_BLOCK_TOKENS // DSV4_CSA_COMPRESS_RATIO // 2


def dsv4_kv_cache_groups():
    """The three real DSv4 attention groups: CSA, indexer, then SWA.

    Page sizes differ per group by design, so no single connector-wide page
    size can validate all three. torch and vLLM are imported here rather than
    at module scope so the fake-tensor helpers above stay importable without
    them.
    """
    import torch
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    )

    def mla(head_size):
        return MLAAttentionSpec(
            block_size=DSV4_CSA_BLOCK_TOKENS,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
            tokens_per_state=DSV4_CSA_COMPRESS_RATIO,
            model_version="deepseek_v4",
        )

    return [
        KVCacheGroupSpec(
            layer_names=["model.layers.0.self_attn"], kv_cache_spec=mla(1024)
        ),
        KVCacheGroupSpec(
            layer_names=["model.layers.0.self_attn.indexer"], kv_cache_spec=mla(256)
        ),
        KVCacheGroupSpec(
            layer_names=["model.layers.1.self_attn"],
            kv_cache_spec=SlidingWindowMLASpec(
                block_size=DSV4_SWA_BLOCK_TOKENS,
                num_kv_heads=1,
                head_size=1024,
                dtype=torch.uint8,
                sliding_window=512,
                model_version="deepseek_v4",
            ),
        ),
    ]


class FakeKVCacheGroup:
    """Duck-typed KVCacheGroupSpec (layer names plus one shared spec)."""

    def __init__(self, layer_names, spec):
        self.layer_names = list(layer_names)
        self.kv_cache_spec = spec


class FakeDsv4Spec:
    """Duck-typed DSv4 KV cache spec: only the fields the manifest reads.

    Field names track vLLM's. A fake left on a renamed field keeps the suite
    green while production reads an attribute that no longer exists.
    """

    def __init__(self, *, num_states, tokens_per_state=1, sliding_window=None):
        self.num_states = int(num_states)
        self.tokens_per_state = int(tokens_per_state)
        self.sliding_window = sliding_window
        # What the scheduler pages this group at, one row per
        # `tokens_per_state` tokens.
        self.block_size = self.num_states * self.tokens_per_state


class FakeUniformTypeSpecs:
    """Duck-typed `UniformTypeKVCacheSpecs`, how vLLM hands DSv4 its groups.

    The wrapper reports the uncompressed `block_size` and carries no
    `tokens_per_state`, so anything reading role geometry off it sees every
    layer as an uncompressed page. Callers have to unwrap per layer.
    """

    def __init__(self, specs):
        self.kv_cache_specs = dict(specs)
        self.block_size = max(spec.block_size for spec in self.kv_cache_specs.values())


def _dsv4_group(layer_names, spec):
    """A DSv4 cache group: per-layer specs under a uniform-type wrapper."""
    return FakeKVCacheGroup(
        layer_names, FakeUniformTypeSpecs({name: spec for name in layer_names})
    )


def dsv4_materialization(*, num_blocks=16, tokens_per_page=256):
    """A DSv4 materialization with the runner's overlay plan applied.

    Two CSA layers, two indexer layers, one HCA layer, two SWA layers and the
    four compressor state caches. Every SWA and state cache aliases an
    already-allocated array, so the returned caches exercise the
    block-ID-disjoint aliasing DSv4 admission has to describe.

    Returns `(named_kv_caches, kv_cache_groups)`.
    """
    tokens = int(tokens_per_page)

    def cache(rows, width):
        return FakeTensor((num_blocks, rows, 4, width), 1, dtype="torch.uint8")

    csa_nope = [cache(tokens, 128) for _ in range(2)]
    csa_rope = [cache(tokens // 4, 128) for _ in range(2)]
    indexer = [cache(tokens // 4, 256) for _ in range(2)]
    hca = cache(tokens * 2, 128)

    csa_names = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
    idx_names = [f"model.layers.{i}.self_attn.attn.indexer.k_cache" for i in range(2)]
    hca_name = "model.layers.2.self_attn.attn"
    swa_names = [f"model.layers.{i}.self_attn.swa_cache" for i in range(2)]

    named = {}
    for name, nope, rope in zip(csa_names, csa_nope, csa_rope):
        named[name] = (nope, rope)
    for name, array in zip(idx_names, indexer):
        named[name] = array
    named[hca_name] = hca
    # SWA overlays the CSA NoPE arrays by position within its cache group.
    for name, host in zip(swa_names, csa_nope):
        named[name] = host

    csa_state = [f"{name}.compressor.state_cache" for name in csa_names]
    idx_state = [
        f"model.layers.{i}.self_attn.attn.indexer.compressor.state_cache"
        for i in range(2)
    ]
    hca_state = f"{hca_name}.compressor.state_cache"
    # CSA/indexer states share their own compressed-KV array; the HCA state is
    # far too big for the HCA page, so it overlays a CSA NoPE array instead.
    for name, host in zip(csa_state, csa_nope):
        named[name] = (host, csa_rope[csa_nope.index(host)])
    for name, host in zip(idx_state, indexer):
        named[name] = host
    named[hca_state] = csa_nope[0]

    csa_spec = FakeDsv4Spec(num_states=tokens, tokens_per_state=4)
    idx_spec = FakeDsv4Spec(num_states=tokens, tokens_per_state=4)
    hca_spec = FakeDsv4Spec(num_states=tokens, tokens_per_state=1)
    # SWA keeps raw latents, so its page holds an eighth of the CSA group's
    # tokens at the same byte size, and it pages at that smaller size.
    swa_spec = FakeDsv4Spec(
        num_states=tokens // 2, tokens_per_state=1, sliding_window=tokens // 2
    )
    # Each state role has its own window, so vLLM puts it in its own group.
    groups = [
        _dsv4_group(csa_names, csa_spec),
        _dsv4_group(idx_names, idx_spec),
        _dsv4_group([hca_name], hca_spec),
        _dsv4_group(swa_names, swa_spec),
        _dsv4_group(csa_state, csa_spec),
        _dsv4_group(idx_state, idx_spec),
        _dsv4_group([hca_state], hca_spec),
    ]
    return named, groups
