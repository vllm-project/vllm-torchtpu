# SPDX-License-Identifier: Apache-2.0
"""Raiden pool-manifest builder for the Qwen3.5 hybrid KV cache.

Raiden's admission surface is model-agnostic (storages / pools / regions, see
``tpu_sync.api.torch.pool_layout``). Everything model- and kernel-specific
about the Qwen3.5 hybrid layout is derived HERE, on the vLLM side, from the
**live materialization** (the typed KV cache tensors bound into the forward
context plus their kv-cache-group specs) — never from pinned constants.

Binding resolution follows the storage-pointer rule used by the working ZMQ
transfer path (``_register_named_region``):

- every logical cache shares storage with a raw unified tensor →
  ``aliased_raw``: pools reference the raw storages with interior offsets;
- no typed cache shares storage with a raw tensor → ``private_typed``: one
  pool per typed tensor (legacy materializations only);
- a mix is a materialization bug and is rejected.

Current unified-pool materializations expose a GDN layer as ``[pool]`` rather
than persistent ``(conv, ssm)`` tensors.  For manifest construction we derive
two metadata-only logical views from the Mamba spec.  This does not execute a
torch operation or add tensors to the model graph.

Registering storages the kernels never touch (e.g. the raw unified pages while
the typed caches are private) hard-fails before any raiden manager is
constructed.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.gdn_pool_layout import (pooled_gdn_conv_state_bytes,
                                           pooled_gdn_ssm_state_bytes,
                                           pooled_gdn_state_dtypes,
                                           pooled_gdn_state_itemsize)

from .tags import (TAG_DSA_IDX, TAG_FA, TAG_GDN_CONV, TAG_GDN_SSM,
                   TAG_MLA_NOPE, TAG_MLA_ROPE, class_tag, layer_tag)

BINDING_PRIVATE_TYPED = "private_typed"
BINDING_ALIASED_RAW = "aliased_raw"


class DeadStorageError(RuntimeError):
    """Pools reference storage the model kernels never read or write."""


class ManifestError(ValueError):
    """The live materialization cannot be described as a pool manifest."""


@dataclasses.dataclass(frozen=True)
class RegionSpec:
    """Mirror of raiden's RegionSpec (kept local so deviceless tests do not
    need the raiden extension)."""

    name: str
    offset_bytes: int
    stride_bytes: int
    unit_bytes: int
    num_units: int
    units_per_stride: int = 1

    @property
    def live_bytes(self) -> int:
        return self.unit_bytes * self.num_units * self.units_per_stride

    @property
    def extent_end_bytes(self) -> int:
        """Exclusive end of the last live byte relative to the pool base."""
        if self.num_units <= 0:
            return self.offset_bytes
        return (self.offset_bytes + (self.num_units - 1) * self.stride_bytes +
                self.unit_bytes * self.units_per_stride)

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PoolEntry:
    tag: str
    layer_name: str
    storage_index: int
    base_offset_bytes: int
    block_stride_bytes: int
    num_blocks: int
    regions: tuple[RegionSpec, ...]
    dtype_tag: str

    @property
    def live_bytes_per_block(self) -> int:
        return sum(region.live_bytes for region in self.regions)

    def to_pool_dict(self) -> dict[str, Any]:
        """Shape consumed by raiden's ``register_pools`` (coerce_pool_spec)."""
        return {
            "tag": self.tag,
            "storage_index": self.storage_index,
            "base_offset_bytes": self.base_offset_bytes,
            "block_stride_bytes": self.block_stride_bytes,
            "num_blocks": self.num_blocks,
            "regions": [region.to_dict() for region in self.regions],
            "dtype_tag": self.dtype_tag,
        }


@dataclasses.dataclass
class PoolManifest:
    binding: str
    storages: list[Any]
    pools: list[PoolEntry]

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pool in self.pools:
            counts[pool.tag] = counts.get(pool.tag, 0) + 1
        return counts

    def geometry_by_tag(self) -> dict[str, dict[str, int]]:
        """Uniform (num_blocks, stride, live) per class tag; raises on
        divergence. Per-layer tags fold into their class tag."""
        result: dict[str, dict[str, int]] = {}
        for pool in self.pools:
            geometry = {
                "num_blocks": pool.num_blocks,
                "block_stride_bytes": pool.block_stride_bytes,
                "live_bytes_per_block": pool.live_bytes_per_block,
            }
            tag = class_tag(pool.tag)
            existing = result.get(tag)
            if existing is None:
                result[tag] = geometry
            elif existing != geometry:
                raise ManifestError(
                    f"pool geometry diverges within tag {tag}: "
                    f"{existing} vs {geometry} ({pool.layer_name})")
        return result

    def pools_of_class(self, tag: str) -> list[PoolEntry]:
        """Pools whose class tag is ``tag``, in manifest order."""
        return [pool for pool in self.pools if class_tag(pool.tag) == tag]

    def pool_dicts(self) -> list[dict[str, Any]]:
        return [pool.to_pool_dict() for pool in self.pools]

    def dtype_tags(self) -> list[str]:
        return [pool.dtype_tag for pool in self.pools]


# ---------------------------------------------------------------------------
# Duck-typed tensor accessors (work for torch tensors and test fakes).
# ---------------------------------------------------------------------------


def _nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(tensor.numel()) * int(tensor.element_size())


def _element_size(tensor: Any) -> int:
    if hasattr(tensor, "element_size"):
        return int(tensor.element_size())
    itemsize = getattr(getattr(tensor, "dtype", None), "itemsize", None)
    if itemsize is None:
        raise ManifestError(f"cannot determine element size of {tensor!r}")
    return int(itemsize)


def _storage_ptr(tensor: Any) -> int | None:
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        return None


def _storage_offset_bytes(tensor: Any) -> int:
    if hasattr(tensor, "storage_offset"):
        return int(tensor.storage_offset()) * _element_size(tensor)
    return 0


def _dtype_tag(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    text = str(dtype)
    return text.removeprefix("torch.")


@dataclasses.dataclass(frozen=True)
class _PooledStateView:
    """Tensor-like metadata for one logical state inside a unified pool.

    The descriptor deliberately implements only the accessors used below.  In
    particular, constructing it never calls ``view``/``as_strided`` on an XLA
    tensor, so Raiden manifest admission cannot grow the device graph.
    """

    pool: Any
    shape: tuple[int, ...]
    dtype: Any
    itemsize: int
    storage_offset_elems: int
    nbytes: int

    def element_size(self) -> int:
        return self.itemsize

    def untyped_storage(self) -> Any:
        return self.pool.untyped_storage()

    def storage_offset(self) -> int:
        return self.storage_offset_elems


def _raw_index_for(tensor: Any, raw_tensors: Sequence[Any]) -> int | None:
    tensor_ptr = _storage_ptr(tensor)
    for idx, raw in enumerate(raw_tensors):
        if raw is tensor:
            return idx
        raw_ptr = _storage_ptr(raw)
        if (tensor_ptr is not None and raw_ptr is not None
                and tensor_ptr == raw_ptr):
            return idx
    return None


def layer_index_from_name(layer_name: str) -> int | None:
    parts = str(layer_name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Region derivation (the Qwen3.5 / TPU-kernel model logic, kept on the model
# side of the model-agnostic Raiden admission surface).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GdnHeadGeometry:
    """Per-rank linear-attention geometry (already TP/PCP-divided)."""

    local_key_heads: int
    local_value_heads: int
    key_head_dim: int
    value_head_dim: int
    # Model's linear_conv_kernel_dim; the pooled conv region holds
    # kernel_size - 1 taps (no spec-decode widening, see gdn_pool_layout).
    conv_kernel_size: int = 4

    @property
    def conv_dim(self) -> int:
        return (2 * self.local_key_heads * self.key_head_dim +
                self.local_value_heads * self.value_head_dim)


def _fa_regions(*, block_size_tokens: int, token_stride_bytes: int,
                num_kv_heads: int, head_size: int,
                itemsize: int) -> tuple[RegionSpec, ...]:
    unit_bytes = 2 * head_size * itemsize  # one K+V head pair per token
    live_per_token = unit_bytes * num_kv_heads
    if live_per_token > token_stride_bytes:
        raise ManifestError(
            "full-attention live bytes per token exceed the physical token "
            f"stride: live={live_per_token} stride={token_stride_bytes}")
    return (RegionSpec(
        name="fa_payload",
        offset_bytes=0,
        stride_bytes=token_stride_bytes,
        unit_bytes=unit_bytes,
        num_units=block_size_tokens,
        units_per_stride=num_kv_heads,
    ), )


def _gdn_conv_regions(*, conv_shape: Sequence[int], itemsize: int,
                      geometry: GdnHeadGeometry) -> tuple[RegionSpec, ...]:
    # Since the mamba state relayout removal the conv cache is declared
    # (blocks, taps, 1, dim); the singleton is layout-neutral, so normalize
    # it away rather than duplicating the derivation below.
    if len(conv_shape) == 4 and conv_shape[2] == 1:
        conv_shape = (conv_shape[0], conv_shape[1], conv_shape[3])
    elif len(conv_shape) == 4 and conv_shape[1] == 1:
        conv_shape = (conv_shape[0], conv_shape[2], conv_shape[3])
    if len(conv_shape) != 3:
        raise ManifestError("GDN conv state must be (blocks, taps, dim) or "
                            f"(blocks, taps, 1, dim): got {conv_shape}")
    taps = int(conv_shape[1])
    local_dim = int(conv_shape[2])
    expected_dim = geometry.conv_dim
    if local_dim != expected_dim:
        raise ManifestError(
            "GDN conv state dim does not match the local head geometry: "
            f"tensor dim={local_dim} derived={expected_dim} ({geometry})")
    row_stride = local_dim * itemsize
    key_bytes = geometry.key_head_dim * itemsize
    value_bytes = geometry.value_head_dim * itemsize
    qk_bytes = 2 * geometry.local_key_heads * key_bytes
    if tpu_envs.TPU_GDN_CONV_QK_PAIR_LAYOUT:
        # QK pair-blocked layout: the rank-local Q and K rows share whole
        # pool tokens ([Q-pair, K-pair] per token), so the transferable
        # unit is the combined QK block followed by the V block — two
        # contiguous extents per tap.  Full-width states store the rank
        # blocks in rank order, which the pooled kernel's rows_perm maps
        # back to the logical [Q | K | V] row order.
        return (
            RegionSpec(name="gdn_conv_qk",
                       offset_bytes=0,
                       stride_bytes=row_stride,
                       unit_bytes=qk_bytes,
                       num_units=taps,
                       units_per_stride=1),
            RegionSpec(name="gdn_conv_v",
                       offset_bytes=qk_bytes,
                       stride_bytes=row_stride,
                       unit_bytes=value_bytes,
                       num_units=taps,
                       units_per_stride=geometry.local_value_heads),
        )
    q_offset = 0
    k_offset = geometry.local_key_heads * key_bytes
    v_offset = 2 * geometry.local_key_heads * key_bytes
    return (
        RegionSpec(name="gdn_conv_q",
                   offset_bytes=q_offset,
                   stride_bytes=row_stride,
                   unit_bytes=key_bytes,
                   num_units=taps,
                   units_per_stride=geometry.local_key_heads),
        RegionSpec(name="gdn_conv_k",
                   offset_bytes=k_offset,
                   stride_bytes=row_stride,
                   unit_bytes=key_bytes,
                   num_units=taps,
                   units_per_stride=geometry.local_key_heads),
        RegionSpec(name="gdn_conv_v",
                   offset_bytes=v_offset,
                   stride_bytes=row_stride,
                   unit_bytes=value_bytes,
                   num_units=taps,
                   units_per_stride=geometry.local_value_heads),
    )


def _gdn_ssm_regions(*, ssm_shape: Sequence[int],
                     itemsize: int) -> tuple[RegionSpec, ...]:
    if len(ssm_shape) != 4:
        raise ManifestError(
            f"GDN ssm state must be (blocks, heads, d, d): got {ssm_shape}")
    heads = int(ssm_shape[1])
    head_bytes = int(ssm_shape[2]) * int(ssm_shape[3]) * itemsize
    return (RegionSpec(
        name="gdn_ssm",
        offset_bytes=0,
        stride_bytes=head_bytes,
        unit_bytes=head_bytes,
        num_units=heads,
        units_per_stride=1,
    ), )


# ---------------------------------------------------------------------------
# Manifest builder.
# ---------------------------------------------------------------------------


def _group_spec_for_layer(kv_cache_groups: Sequence[Any],
                          layer_name: str) -> Any:
    for group in kv_cache_groups:
        if layer_name in group.layer_names:
            return group.kv_cache_spec
    raise ManifestError(f"layer {layer_name} is in no kv cache group")


def _pooled_gdn_state_views(*, pool: Any, spec: Any,
                            raw_tensors: Sequence[Any], layer_name: str,
                            gdn_geometry: GdnHeadGeometry) -> tuple[Any, Any]:
    """Derive logical ``(conv, ssm)`` descriptors from a unified raw pool.

    The state geometry is tied to the pooled GDN kernel's byte model
    (``gdn_pool_layout`` — the same helpers ``_build_v3_pool_state_plan``
    uses), NOT to the layer's declared MambaSpec shapes or Conv dtype.  The
    pooled kernel never reads those declarations, so only the kernel model
    describes the geometry and Conv bytes that actually move; a declaration
    that drifts from it (e.g. a Conv-dtype/shape override made for the typed
    materialization path) must not leak into transfer spans.  The spec
    contributes the SSM dtype and manager page pitch, plus the identity check
    that this is a two-state GDN layer.
    """
    raw_index = _raw_index_for(pool, raw_tensors)
    if raw_index is None:
        raise ManifestError(
            f"pooled GDN cache {layer_name} must share raw pool storage")
    raw = raw_tensors[raw_index]
    pool_nbytes = _nbytes(pool)
    raw_nbytes = _nbytes(raw)
    pool_offset_bytes = _storage_offset_bytes(pool)
    if (pool_nbytes != raw_nbytes
            or pool_offset_bytes != _storage_offset_bytes(raw)):
        raise ManifestError(
            f"pooled GDN cache {layer_name} must cover the complete raw pool")

    shapes = tuple(getattr(spec, "shapes", ()))
    dtypes = tuple(getattr(spec, "dtypes", ()))
    if len(shapes) != 2 or len(dtypes) != 2:
        raise ManifestError(
            f"pooled GDN cache {layer_name} requires conv and SSM specs")

    taps = gdn_geometry.conv_kernel_size - 1
    if taps <= 0:
        raise ManifestError(
            f"pooled GDN cache {layer_name} has no conv taps: "
            f"conv_kernel_size={gdn_geometry.conv_kernel_size}")
    conv_shape = (taps, gdn_geometry.conv_dim)
    conv_dtype, ssm_dtype = pooled_gdn_state_dtypes(dtypes)
    try:
        conv_itemsize = pooled_gdn_state_itemsize(conv_dtype)
        ssm_itemsize = pooled_gdn_state_itemsize(ssm_dtype)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    conv_bytes = pooled_gdn_conv_state_bytes(
        kernel_size=gdn_geometry.conv_kernel_size,
        conv_dim=gdn_geometry.conv_dim)
    ssm_shape = (gdn_geometry.local_value_heads, gdn_geometry.value_head_dim,
                 gdn_geometry.key_head_dim)
    ssm_bytes = pooled_gdn_ssm_state_bytes(
        num_v_heads=gdn_geometry.local_value_heads,
        head_k_dim=gdn_geometry.key_head_dim,
        head_v_dim=gdn_geometry.value_head_dim,
        dtype=ssm_dtype)

    try:
        manager_page_bytes = int(spec.page_size_bytes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ManifestError(
            f"pooled GDN cache {layer_name} has no manager page size") from exc
    if manager_page_bytes <= 0 or raw_nbytes % manager_page_bytes != 0:
        raise ManifestError(
            f"raw pool bytes {raw_nbytes} for {layer_name} must be divisible "
            f"by manager page bytes {manager_page_bytes}")
    if conv_bytes + ssm_bytes > manager_page_bytes:
        raise ManifestError(
            f"GDN states for {layer_name} exceed one manager page: "
            f"states={conv_bytes + ssm_bytes} page={manager_page_bytes}")
    num_blocks = raw_nbytes // manager_page_bytes

    states: list[_PooledStateView] = []
    # The PR #106 pooled ABI places SSM first and conv immediately after it,
    # while the public/canonical manifest order remains conv then SSM.
    for shape, dtype, itemsize, state_bytes, offset_bytes in ((conv_shape,
                                                               conv_dtype,
                                                               conv_itemsize,
                                                               conv_bytes,
                                                               ssm_bytes),
                                                              (ssm_shape,
                                                               ssm_dtype,
                                                               ssm_itemsize,
                                                               ssm_bytes, 0)):
        absolute_offset_bytes = pool_offset_bytes + offset_bytes
        if (manager_page_bytes % itemsize != 0
                or absolute_offset_bytes % itemsize != 0):
            raise ManifestError(
                f"pooled GDN state for {layer_name} is not aligned to "
                f"dtype {dtype}: page={manager_page_bytes} "
                f"offset={absolute_offset_bytes}")
        states.append(
            _PooledStateView(
                pool=pool,
                shape=(num_blocks, *shape),
                dtype=dtype,
                itemsize=itemsize,
                storage_offset_elems=absolute_offset_bytes // itemsize,
                nbytes=num_blocks * state_bytes,
            ))
    return states[0], states[1]


class _StorageTable:
    """Ordered, pointer-deduplicated storage list."""

    def __init__(self) -> None:
        self.storages: list[Any] = []
        self._index_by_ptr: dict[int, int] = {}

    def index_for(self, tensor: Any) -> int:
        ptr = _storage_ptr(tensor)
        if ptr is None:
            ptr = id(tensor)
        index = self._index_by_ptr.get(ptr)
        if index is None:
            index = len(self.storages)
            self.storages.append(tensor)
            self._index_by_ptr[ptr] = index
        return index


def _binding_for(typed_tensors: Sequence[tuple[str, Any]],
                 raw_tensors: Sequence[Any]) -> tuple[str, dict[int, int]]:
    """Returns (binding, {typed position -> raw index}) after consistency
    checks."""
    raw_index_by_pos: dict[int, int] = {}
    for pos, (_, tensor) in enumerate(typed_tensors):
        raw_index = _raw_index_for(tensor, raw_tensors)
        if raw_index is not None:
            raw_index_by_pos[pos] = raw_index
    if not raw_index_by_pos:
        return BINDING_PRIVATE_TYPED, {}
    if len(raw_index_by_pos) == len(typed_tensors):
        return BINDING_ALIASED_RAW, raw_index_by_pos
    aliased = [typed_tensors[pos][0] for pos in sorted(raw_index_by_pos)]
    private = [
        name for pos, (name, _) in enumerate(typed_tensors)
        if pos not in raw_index_by_pos
    ]
    raise ManifestError(
        "mixed KV cache binding: some typed caches alias raw storage "
        f"({aliased[:3]}…) while others are private ({private[:3]}…)")


def build_qwen35_pool_manifest(
    *,
    named_kv_caches: Mapping[str, Any],
    kv_cache_groups: Sequence[Any],
    raw_tensors: Sequence[Any],
    gdn_geometry: GdnHeadGeometry,
    mamba_group_ordinal_by_layer: Mapping[str, int] | None = None,
    per_layer_tags: bool = False,
) -> PoolManifest:
    """Builds the canonical pool manifest from the live materialization.

    Canonical pool order (must match on both transfer peers because pool
    indices travel on the wire): model layer order; within a GDN layer, conv
    before ssm.

    ``per_layer_tags`` suffixes every tag with the layer index
    (``fa.l3``, ``gdn.conv.g0.l0`` ...) so pools pair up by layer: a
    pipeline-parallel producer registers only its own layers, and the
    controller matches each of them to the same layer on the destination.

    ``mamba_group_ordinal_by_layer`` (opt-in, state-reshard deployments):
    suffixes GDN pool tags with the layer's mamba kv-cache-group ordinal
    (``gdn.conv.g0`` ...). Each group has its own block table, so state
    transfers must address slots per group; the suffix is pure vLLM policy —
    raiden keeps treating tags as opaque. Both peers must configure it
    identically (enforced by the manifest identity check at plan time).
    """
    if not named_kv_caches:
        raise ManifestError("named_kv_caches is empty")

    ordered_layers = sorted(named_kv_caches.keys(),
                            key=lambda name:
                            (layer_index_from_name(name) is None,
                             layer_index_from_name(name) or 0, str(name)))

    # Flatten to (pool key, typed/logical tensor) in canonical order.
    flat: list[tuple[str, str, Any]] = []  # (tag, layer_name, tensor)
    for layer_name in ordered_layers:
        cache = named_kv_caches[layer_name]
        if isinstance(cache, (list, tuple)):
            if len(cache) == 1:
                spec = _group_spec_for_layer(kv_cache_groups, layer_name)
                states = _pooled_gdn_state_views(pool=cache[0],
                                                 spec=spec,
                                                 raw_tensors=raw_tensors,
                                                 layer_name=layer_name,
                                                 gdn_geometry=gdn_geometry)
            elif len(cache) == 2:
                states = cache
            else:
                raise ManifestError(
                    f"GDN layer {layer_name} must have a unified pool or "
                    f"(conv, ssm) states: "
                    f"got {len(cache)}")
            suffix = ""
            if mamba_group_ordinal_by_layer is not None:
                ordinal = mamba_group_ordinal_by_layer.get(layer_name)
                if ordinal is None:
                    raise ManifestError(
                        f"GDN layer {layer_name} has no mamba group ordinal")
                suffix = f".g{int(ordinal)}"
            flat.append((TAG_GDN_CONV + suffix, layer_name, states[0]))
            flat.append((TAG_GDN_SSM + suffix, layer_name, states[1]))
        else:
            flat.append((TAG_FA, layer_name, cache))
    if per_layer_tags:
        tagged = []
        for tag, layer_name, tensor in flat:
            layer_index = layer_index_from_name(layer_name)
            if layer_index is None:
                raise ManifestError(
                    f"per-layer pool tags need a layer index in {layer_name}")
            tagged.append((layer_tag(tag, layer_index), layer_name, tensor))
        flat = tagged

    binding, raw_index_by_pos = _binding_for(
        [(layer_name, tensor) for _, layer_name, tensor in flat], raw_tensors)

    storages = _StorageTable()
    pools: list[PoolEntry] = []
    for pos, (tag, layer_name, tensor) in enumerate(flat):
        nbytes = _nbytes(tensor)
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        itemsize = _element_size(tensor)
        dtype_tag = _dtype_tag(tensor)

        if class_tag(tag) == TAG_FA:
            spec = _group_spec_for_layer(kv_cache_groups, layer_name)
            block_size = int(spec.block_size)
            num_kv_heads = int(spec.num_kv_heads)
            head_size = int(spec.head_size)
            if len(shape) < 2:
                raise ManifestError(
                    f"full-attention cache {layer_name} needs a paged shape: "
                    f"got {shape}")
            total_tokens = shape[0] * shape[1]
            if total_tokens % block_size != 0:
                raise ManifestError(
                    f"full-attention cache {layer_name} token capacity "
                    f"{total_tokens} is not divisible by the logical block "
                    f"size {block_size}")
            num_blocks = total_tokens // block_size
            if nbytes % total_tokens != 0:
                raise ManifestError(
                    f"full-attention cache {layer_name} nbytes {nbytes} is "
                    f"not divisible by token capacity {total_tokens}")
            token_stride = nbytes // total_tokens
            live_stride = nbytes // num_blocks
            regions = _fa_regions(block_size_tokens=block_size,
                                  token_stride_bytes=token_stride,
                                  num_kv_heads=num_kv_heads,
                                  head_size=head_size,
                                  itemsize=itemsize)
        else:
            if not shape:
                raise ManifestError(f"GDN state {layer_name} has no shape")
            num_blocks = shape[0]
            if num_blocks <= 0 or nbytes % num_blocks != 0:
                raise ManifestError(
                    f"GDN state {layer_name} nbytes {nbytes} is not "
                    f"divisible by num_blocks {num_blocks}")
            live_stride = nbytes // num_blocks
            if class_tag(tag).startswith(TAG_GDN_CONV):
                regions = _gdn_conv_regions(conv_shape=shape,
                                            itemsize=itemsize,
                                            geometry=gdn_geometry)
            else:
                regions = _gdn_ssm_regions(ssm_shape=shape, itemsize=itemsize)

        if binding == BINDING_PRIVATE_TYPED:
            storage_index = storages.index_for(tensor)
            base_offset = 0
            stride = live_stride
        else:
            raw = raw_tensors[raw_index_by_pos[pos]]
            storage_index = storages.index_for(raw)
            raw_nbytes = _nbytes(raw)
            if raw_nbytes % num_blocks != 0:
                raise ManifestError(
                    f"raw storage bytes {raw_nbytes} are not divisible by "
                    f"num_blocks {num_blocks} for {layer_name}")
            stride = raw_nbytes // num_blocks
            base_offset = (_storage_offset_bytes(tensor) -
                           _storage_offset_bytes(raw))
            if base_offset < 0 or base_offset >= stride:
                raise ManifestError(
                    f"typed cache {layer_name} offset {base_offset} is "
                    f"outside one raw page of {stride} bytes")

        last_live_byte = base_offset + max(region.extent_end_bytes
                                           for region in regions)
        if last_live_byte > stride:
            raise ManifestError(
                f"logical cache {layer_name} extends through byte "
                f"{last_live_byte}, beyond its physical page of {stride} "
                "bytes")

        pools.append(
            PoolEntry(
                tag=tag,
                layer_name=layer_name,
                storage_index=storage_index,
                base_offset_bytes=base_offset,
                block_stride_bytes=stride,
                num_blocks=num_blocks,
                regions=regions,
                dtype_tag=dtype_tag,
            ))

    manifest = PoolManifest(binding=binding,
                            storages=storages.storages,
                            pools=pools)
    # All pools must agree on the block count: the vLLM BlockPool hands out
    # one global block-id space across groups.
    block_counts = {pool.num_blocks for pool in manifest.pools}
    if len(block_counts) != 1:
        raise ManifestError(
            f"pools disagree on num_blocks: {sorted(block_counts)}")
    manifest.geometry_by_tag()
    return manifest


def materialize_storages(manifest: PoolManifest) -> None:
    """Forces device-buffer materialization of every manifest storage.

    torch_tpu tensors created with ``torch.empty`` can carry no materialized
    device buffer until first use;
    wrapping such a tensor in a raiden manager hangs its constructor
    (AwaitBuffer never resolves). A one-block read materializes the buffer
    without touching cache contents.

    Note the flip side: the raiden manager pins the storages' *current*
    device buffers. Torch-level ops that swap a tensor's buffer (e.g. eager
    ``copy_``) after admission leave the manager reading stale storage. The
    model's cache-update kernels write in place, so admission at
    register_runner time is safe; do not rebind typed caches afterwards.
    """
    for storage in manifest.storages:
        if hasattr(storage, "cpu"):
            try:
                storage[:1].cpu()
            except Exception:  # pragma: no cover - non-torch fakes
                pass


def verify_storage_binding(
    manifest: PoolManifest,
    named_kv_caches: Mapping[str, Any],
    raw_tensors: Sequence[Any],
) -> int:
    """The dead-storage check (hard fail).

    The storages referenced by the manifest must exactly cover the typed KV
    caches bound into the forward context — for ``private_typed`` they must BE
    those tensors (and never the raw unified pages); for ``aliased_raw`` every
    typed cache must resolve into a referenced raw storage.

    Returns the number of verified pool storages for logging.
    """
    typed_ptrs: set[int] = set()
    for cache in named_kv_caches.values():
        tensors = cache if isinstance(cache, (list, tuple)) else (cache, )
        for tensor in tensors:
            ptr = _storage_ptr(tensor)
            if ptr is not None:
                typed_ptrs.add(ptr)
    raw_ptrs = {
        ptr
        for ptr in (_storage_ptr(raw) for raw in raw_tensors)
        if ptr is not None
    }
    storage_ptrs = {
        ptr
        for ptr in (_storage_ptr(storage) for storage in manifest.storages)
        if ptr is not None
    }

    if manifest.binding == BINDING_PRIVATE_TYPED:
        dead = storage_ptrs & raw_ptrs
        if dead:
            raise DeadStorageError(
                "pool storages reference raw unified pages while the typed "
                "KV caches are private tensors — the kernels never touch "
                f"those bytes ({len(dead)} storages).")
        if storage_ptrs != typed_ptrs:
            missing = len(typed_ptrs - storage_ptrs)
            extra = len(storage_ptrs - typed_ptrs)
            raise DeadStorageError(
                "pool storages do not match the typed KV cache storages: "
                f"{missing} typed storages unreferenced, {extra} pool "
                "storages unknown")
    else:
        if not storage_ptrs <= raw_ptrs:
            raise DeadStorageError(
                "aliased_raw pool storages must be raw unified tensors")
        if not typed_ptrs <= raw_ptrs:
            raise DeadStorageError(
                "aliased_raw binding requires every typed KV cache to share "
                "raw unified storage")
    return len(manifest.storages)


_GLM_REGION_NAMES = {
    TAG_MLA_NOPE: "mla_nope_rows",
    TAG_MLA_ROPE: "mla_rope_rows",
    TAG_DSA_IDX: "dsa_rows",
}


def _glm_row_region(tag: str, *, row_bytes: int,
                    num_rows: int) -> tuple[RegionSpec, ...]:
    """One dense region of whole packed rows covering the full page."""
    return (RegionSpec(
        name=_GLM_REGION_NAMES[tag],
        offset_bytes=0,
        stride_bytes=row_bytes,
        unit_bytes=row_bytes,
        num_units=num_rows,
        units_per_stride=1,
    ), )


def build_glm_mla_pool_manifest(
    *,
    named_kv_caches: Mapping[str, Any],
    raw_tensors: Sequence[Any],
    block_size_tokens: int,
) -> PoolManifest:
    """Builds the Pool Manifest for GLM MLA models.

    Per-layer TP-replicated cache classes: the split sparse MLA latent
    (nope, rope) tensor pair plus the DSA indexer K cache  (TAG_DSA_IDX).
    The packed [blocks, rows, packing, width] layout is declared as
    row-granular regions; a nope cache carries one row per token,
    every other class packs `packing` tokens per row.
    """
    if block_size_tokens <= 0:
        raise ManifestError("block_size_tokens must be positive")
    flat: list[tuple[str, str, str, Any]] = []
    for layer_name in sorted(named_kv_caches):
        cache = named_kv_caches[layer_name]
        if isinstance(cache, (list, tuple)):
            if len(cache) != 2:
                raise ManifestError(
                    f"GLM MLA admission expects (nope, rope) cache pairs; "
                    f"layer {layer_name} has a {len(cache)}-tuple")
            nope, rope = cache
            flat.append((TAG_MLA_NOPE, layer_name, _dtype_tag(nope), nope))
            flat.append((TAG_MLA_ROPE, layer_name, _dtype_tag(rope), rope))
            continue
        dtype_tag = _dtype_tag(cache)
        flat.append((TAG_DSA_IDX, layer_name, dtype_tag, cache))
    if not any(tag == TAG_MLA_NOPE for tag, _, _, _ in flat):
        raise ManifestError(
            "GLM MLA admission found no latent (mla.nope) caches")

    binding, _ = _binding_for([(layer_name, tensor)
                               for _, layer_name, _, tensor in flat],
                              raw_tensors)
    if binding != BINDING_PRIVATE_TYPED:
        raise ManifestError(
            "GLM MLA admission requires private typed cache tensors; got "
            f"binding {binding!r}")

    storages = _StorageTable()
    pools: list[PoolEntry] = []
    for tag, layer_name, dtype_tag, tensor in flat:
        nbytes = _nbytes(tensor)
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        if len(shape) != 4:
            raise ManifestError(
                f"cache {layer_name} must be [blocks, rows, packing, width]: "
                f"got shape {shape}")
        num_blocks, rows, packing, width = shape
        itemsize = _element_size(tensor)
        if num_blocks <= 0 or nbytes % num_blocks != 0:
            raise ManifestError(
                f"cache {layer_name} nbytes {nbytes} is not divisible by "
                f"num_blocks {num_blocks}")
        live_stride = nbytes // num_blocks
        if tag == TAG_MLA_NOPE:
            # One packed row per token in the nope layout.
            if rows != block_size_tokens:
                raise ManifestError(
                    f"cache {layer_name} nope rows {rows} do not match the "
                    f"KV block size {block_size_tokens}")
        elif rows * packing != block_size_tokens:
            raise ManifestError(
                f"cache {layer_name} rows*packing {rows}*{packing} does not "
                f"match the KV block size {block_size_tokens}")
        # The packing axis must fill one 32-bit word (see the layout
        # fingerprint's tile assertion).
        if packing * itemsize != 4:
            raise ManifestError(
                f"cache {layer_name} packing {packing} does not fill one "
                f"32-bit word at itemsize {itemsize}")
        if width % 128:
            raise ManifestError(
                f"cache {layer_name} width {width} is not lane-aligned")
        row_bytes = packing * width * itemsize
        if live_stride != rows * row_bytes:
            raise ManifestError(
                f"cache {layer_name} block stride {live_stride} does not "
                f"match {rows} rows of {row_bytes} bytes")
        pools.append(
            PoolEntry(
                tag=tag,
                layer_name=layer_name,
                storage_index=storages.index_for(tensor),
                base_offset_bytes=0,
                block_stride_bytes=live_stride,
                num_blocks=num_blocks,
                regions=_glm_row_region(tag,
                                        row_bytes=row_bytes,
                                        num_rows=rows),
                dtype_tag=dtype_tag,
            ))

    manifest = PoolManifest(binding=binding,
                            storages=storages.storages,
                            pools=pools)
    block_counts = {pool.num_blocks for pool in manifest.pools}
    if len(block_counts) != 1:
        raise ManifestError(
            f"pools disagree on num_blocks: {sorted(block_counts)}")
    manifest.geometry_by_tag()  # raises on per-tag geometry divergence
    return manifest
