# SPDX-License-Identifier: Apache-2.0
"""Byte-level oracle for Stage-3 prefill -> decode resharding under HND.

Every source unit's FA pages and GDN states are synthesized in the
rank-local layout its kernels write, with a unique 32-bit marker per word.
The producer scheduler and workers lower them through the real connector
paths, and the decode scheduler and leader name the transfer through
theirs. A port of the Raiden planner's checks gathers the declarations per
tag, splits compact destination offsets at destination pages, validates
exactly-once prefix coverage per destination unit and applies the copies.
The result must equal destination buffers built independently from the
destination's TP/DCP head and token ownership, including the final DCP
stripe padded with the last page, and each FA copy of a replicated source
head must come from the source replica it is dealt to.
"""

import dataclasses
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc
from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
from vllm_torchtpu.distributed.kv_transfer.raiden.tags import class_tag, split_layer_tag

from .raiden_test_utils import FakeTensor
from .test_tpu_connector import (
    _FakeRaidenControllerFacade,
    _make_raiden_scheduler,
    _make_raiden_worker,
)

KERNEL_PAGE = 128
PACKING = 4
TAPS = 3
D_K = D_V = 128
BF16 = 2
LANE_BYTES = KERNEL_PAGE * BF16  # one BF16 conv row of carrier lanes
GRANULE = PACKING * KERNEL_PAGE
SRC_STATE_BLOCK = 11
DST_STATE_BLOCK = 77
SRC_FILL = 0xEE
DST_FILL = 0xDD
CONTROLLER = "localhost:27000"


@dataclasses.dataclass(frozen=True)
class Topology:
    pp: int
    pcp: int
    ptp: int
    dtp: int
    dcp: int
    kv_heads: int
    key_heads: int
    value_heads: int
    layers_per_stage: int = 2
    src_block: int = 384
    dst_block: int = 256
    head_dim: int = 32
    qk_pair: bool = True

    @property
    def units(self) -> int:
        return self.pp * self.pcp * self.ptp

    @property
    def topology_name(self) -> str:
        if self.pp > 1:
            return f"pp{self.pp}pcp{self.pcp}_prefill"
        return f"pcp{self.pcp}tp{self.ptp}_prefill"

    def stage_layers(self, stage: int) -> list[int]:
        first = stage * self.layers_per_stage
        return list(range(first, first + self.layers_per_stage))

    def layers(self) -> list[int]:
        return list(range(self.pp * self.layers_per_stage))


def is_fa(layer: int) -> bool:
    return layer % 2 == 1


# A hybrid model with 4 KV heads and 16 key / 128 value GDN heads, and a small
# single-host shape.
PP4_PCP8 = Topology(
    pp=4, pcp=8, ptp=1, dtp=32, dcp=8, kv_heads=4, key_heads=16, value_heads=128
)
PCP16_TP2 = Topology(
    pp=1,
    pcp=16,
    ptp=2,
    dtp=32,
    dcp=8,
    kv_heads=4,
    key_heads=16,
    value_heads=128,
    layers_per_stage=4,
    dst_block=384,
)
PP2_PCP2 = Topology(
    pp=2,
    pcp=2,
    ptp=1,
    dtp=4,
    dcp=2,
    kv_heads=2,
    key_heads=16,
    value_heads=32,
    qk_pair=False,
)
PP2_PCP2_QK_PAIR = dataclasses.replace(PP2_PCP2, qk_pair=True)
# The same model at half width: 2 KV heads, 8 key / 64 value GDN heads.
HALF_PCP8_TP2 = Topology(
    pp=1,
    pcp=8,
    ptp=2,
    dtp=16,
    dcp=8,
    kv_heads=2,
    key_heads=8,
    value_heads=64,
    layers_per_stage=4,
    dst_block=384,
)
HALF_PP2_PCP8 = Topology(
    pp=2, pcp=8, ptp=1, dtp=16, dcp=8, kv_heads=2, key_heads=8, value_heads=64
)
# One KV head replicated over the producer's TP2 and the decode's two DCP
# groups: each source replica ships the FA copy of one group.
PCP4_TP2_ONE_KV_HEAD = Topology(
    pp=1,
    pcp=4,
    ptp=2,
    dtp=4,
    dcp=2,
    kv_heads=1,
    key_heads=8,
    value_heads=32,
    layers_per_stage=4,
)

TOKENS = [1, 127, 128, 129, 1000, 3072, 8192, 8191, 12345]

# ---------------------------------------------------------------------------
# Ownership rules, written from each side's documented sharding.
# ---------------------------------------------------------------------------


def _tp_kv_heads(heads: int, degree: int, rank: int) -> list[int]:
    """FA KV heads of a TP rank; heads replicate when fewer than ranks."""
    if heads >= degree:
        width = heads // degree
        return list(range(rank * width, (rank + 1) * width))
    return [rank // (degree // heads)]


def _gdn_heads(
    keys: int, values: int, degree: int, rank: int
) -> tuple[list[int], list[int]]:
    """GDN key and value heads of a head-axis rank: value heads shard
    contiguously, key heads shard or replicate across adjacent ranks."""
    width = values // degree
    value_heads = list(range(rank * width, (rank + 1) * width))
    if keys >= degree:
        key_width = keys // degree
        return list(range(rank * key_width, (rank + 1) * key_width)), value_heads
    return [rank // (degree // keys)], value_heads


# ---------------------------------------------------------------------------
# Marker payloads.
# ---------------------------------------------------------------------------


def _slab_words(topo: Topology) -> int:
    return topo.head_dim * KERNEL_PAGE // 4


def _fa_page(topo: Topology, layer: int, page: int, heads: list[int]) -> np.ndarray:
    """One kernel page of the given KV heads: K then V slab per head."""
    word = np.arange(_slab_words(topo), dtype=np.uint32)
    return np.concatenate(
        [
            np.uint32((layer << 25) | (page << 15) | (head << 12) | (kv << 11)) + word
            for head in heads
            for kv in (0, 1)
        ]
    ).view(np.uint8)


def _head_row(
    layer: int, tap: int, kind: int, head: int, head_bytes: int
) -> np.ndarray:
    word = np.arange(head_bytes // 4, dtype=np.uint32)
    return np.uint32((layer << 25) | (tap << 23) | (kind << 21) | (head << 14)) + word


def _conv_state(
    layer: int, key_heads: list[int], value_heads: list[int], qk_pair: bool
) -> np.ndarray:
    """Per tap: stored Q/K block then V. The QK pair-blocked layout stores
    [Q row, K row] pairs of the pooled kernel's logical row width."""
    taps = []
    for tap in range(TAPS):
        q = np.concatenate([_head_row(layer, tap, 0, h, D_K * BF16) for h in key_heads])
        k = np.concatenate([_head_row(layer, tap, 1, h, D_K * BF16) for h in key_heads])
        v = np.concatenate(
            [_head_row(layer, tap, 2, h, D_V * BF16) for h in value_heads]
        )
        if qk_pair:
            row_bytes = 4 * (q.size + k.size + v.size)
            width = math.gcd(row_bytes, LANE_BYTES, 4 * q.size) // 4
            qk = np.stack((q.reshape(-1, width), k.reshape(-1, width)), axis=1).reshape(
                -1
            )
        else:
            qk = np.concatenate((q, k))
        taps.append(np.concatenate((qk, v)))
    return np.concatenate(taps).view(np.uint8)


def _ssm_state(layer: int, value_heads: list[int]) -> np.ndarray:
    word = np.arange(D_K * D_V * BF16 // 4, dtype=np.uint32)
    return np.concatenate(
        [
            np.uint32((layer << 25) | (1 << 24) | (head << 13)) + word
            for head in value_heads
        ]
    ).view(np.uint8)


# ---------------------------------------------------------------------------
# Manifests.
# ---------------------------------------------------------------------------


def _manifest(
    topo: Topology,
    layers: list[int],
    *,
    fa_heads: int,
    key_heads: int,
    value_heads: int,
    block_tokens: int,
    per_layer_tags: bool,
) -> rpm.PoolManifest:
    geometry = sol.SolFaGeometry(2 * fa_heads, topo.head_dim // PACKING, KERNEL_PAGE)
    gdn = rpm.GdnHeadGeometry(
        local_key_heads=key_heads,
        local_value_heads=value_heads,
        key_head_dim=D_K,
        value_head_dim=D_V,
    )
    conv_regions = rpm._gdn_conv_regions(
        conv_shape=(64, TAPS, gdn.conv_dim), itemsize=BF16, geometry=gdn
    )
    ssm_regions = rpm._gdn_ssm_regions(
        ssm_shape=(64, value_heads, D_V, D_K), itemsize=BF16
    )
    ssm_bytes = value_heads * D_K * D_V * BF16
    stride = block_tokens * geometry.bytes_per_token

    def tag(base: str, layer: int) -> str:
        return f"{base}.l{layer}" if per_layer_tags else base

    pools = []
    for layer in layers:
        if is_fa(layer):
            pools.append(
                rpm.PoolEntry(
                    tag("fa", layer),
                    f"model.layers.{layer}.self_attn.attn",
                    0,
                    0,
                    stride,
                    64,
                    sol.sol_fa_regions(
                        block_size_tokens=block_tokens, geometry=geometry
                    ),
                    "float8_e4m3fn",
                )
            )
            continue
        name = f"model.layers.{layer}.linear_attn"
        pools.append(
            rpm.PoolEntry(
                tag("gdn.conv.g0", layer),
                name,
                0,
                ssm_bytes,
                stride,
                64,
                conv_regions,
                "bfloat16",
            )
        )
        pools.append(
            rpm.PoolEntry(
                tag("gdn.ssm.g0", layer),
                name,
                0,
                0,
                stride,
                64,
                ssm_regions,
                "bfloat16",
            )
        )
    return rpm.PoolManifest(
        binding=rpm.BINDING_ALIASED_RAW,
        storages=[
            FakeTensor(
                (64, 2 * fa_heads, topo.head_dim // PACKING, PACKING, KERNEL_PAGE), 1
            )
        ],
        pools=pools,
    )


def _pool_layers(manifest: rpm.PoolManifest) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for pool in manifest.pools:
        result.setdefault(pool.tag, []).append(int(pool.layer_name.split(".")[2]))
    return result


def _live_bytes(manifest: rpm.PoolManifest) -> dict[str, int]:
    return {pool.tag: pool.live_bytes_per_block for pool in manifest.pools}


def _kind(tag: str) -> str:
    base = class_tag(tag)
    return "fa" if base == "fa" else base.split(".")[1]


def _unit(fields: dict) -> tuple:
    return tuple(sorted(fields.items()))


def _unit_rank(unit: tuple) -> int:
    return int(dict(unit)["job_replica_id"].rsplit("rank", 1)[1])


_PAYLOAD = {
    "minor_to_major": [4, 3, 2, 1, 0],
    "tiles": [[4, 128], [4, 1]],
    "element_size_in_bits": 8,
    "fa_kv_layout": sol.SOL_LAYOUT_TOKEN,
    "fa_kernel_page_tokens": KERNEL_PAGE,
}

_ENV_OVERRIDES: list = []


@pytest.fixture(autouse=True)
def _restore_tpu_envs():
    yield
    while _ENV_OVERRIDES:
        _ENV_OVERRIDES.pop().stop()


def _set_env(name: str, value) -> None:
    """Overrides one tpu_envs value until the end of the test."""
    override = patch.object(tc.tpu_envs, name, value)
    override.start()
    _ENV_OVERRIDES.append(override)


def _environment(monkeypatch, topo: Topology, per_layer_tags: bool) -> None:
    monkeypatch.setattr(tc, "_raiden_seq_on_lane_layout", lambda: True)
    _set_env("TPU_KV_RESHARD_TRANSPORT", "raiden")
    _set_env("TPU_RAIDEN_TRANSFER_PARALLELISM", topo.units)
    _set_env("TPU_RAIDEN_DST_SHARDS", topo.dtp)
    _set_env("TPU_RAIDEN_CONTROLLER_ADDRESS", CONTROLLER)
    _set_env("TPU_RAIDEN_POOL_TAGS_PER_LAYER", per_layer_tags)
    _set_env("TPU_GDN_CONV_QK_PAIR_LAYOUT", topo.qk_pair)


def _set_model(config, topo: Topology) -> None:
    hf = config.model_config.hf_config
    hf.num_key_value_heads = topo.kv_heads
    hf.linear_num_key_heads = topo.key_heads
    hf.linear_num_value_heads = topo.value_heads
    hf.linear_key_head_dim = D_K
    hf.linear_value_head_dim = D_V


# ---------------------------------------------------------------------------
# Producer: scheduler finish, then every worker's registration.
# ---------------------------------------------------------------------------


def _produce(monkeypatch, topo: Topology, num_tokens: int, per_layer_tags: bool):
    _set_env("TPU_RAIDEN_ENGINE_ID", "producer")
    scheduler = _make_raiden_scheduler(
        is_producer=True, tp_size=topo.ptp, pcp_size=topo.pcp, block_size=topo.src_block
    )
    config = scheduler.vllm_config
    config.parallel_config.pipeline_parallel_size = topo.pp
    config.parallel_config.cp_kv_cache_interleave_size = KERNEL_PAGE
    config.kv_transfer_config.kv_connector_extra_config = {
        "destination_decode_context_parallel_size": topo.dcp
    }
    scheduler._stage3_mamba_group_indices = [1]
    assert scheduler.get_finished_count() == topo.units
    request = MagicMock(
        request_id="request",
        num_computed_tokens=num_tokens + 1,
        num_prompt_tokens=num_tokens + 1,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params={},
    )
    blocks = math.ceil(num_tokens / (topo.src_block * topo.pcp))
    # Shuffled physical ids plus one trailing block past the prefix.
    scheduler_ids = [900 - 17 * i for i in range(blocks + 1)]
    delay, params = scheduler.request_finished(
        request, scheduler_ids, mamba_block_ids=[[5, SRC_STATE_BLOCK]]
    )
    assert delay and params["src_parallelism"] == topo.units
    assert params["num_tokens"] == num_tokens
    send = scheduler.reqs_to_send["request"]
    assert list(send.local_block_ids) == scheduler_ids[:blocks]

    registrations, pools, live, memory = {}, {}, {}, {}
    kv_local = max(1, topo.kv_heads // topo.ptp)
    gdn_degree = topo.pcp * topo.ptp
    num_pages = math.ceil(num_tokens / KERNEL_PAGE)
    kps_per_block = topo.src_block // KERNEL_PAGE
    for stage in range(topo.pp):
        for p in range(topo.pcp):
            for t in range(topo.ptp):
                worker = _make_raiden_worker(
                    tp_rank=t,
                    tp_size=topo.ptp,
                    pcp_size=topo.pcp,
                    pp_size=topo.pp,
                    block_size=topo.src_block,
                    interleave_size=KERNEL_PAGE,
                )
                transfer_config = worker.vllm_config.kv_transfer_config
                transfer_config.kv_connector_extra_config = {
                    "destination_decode_context_parallel_size": topo.dcp
                }
                _set_model(worker.vllm_config, topo)
                assert worker._raiden_qwen35_admission_topology() == topo.topology_name
                geometry = worker._gdn_head_geometry()
                worker._validate_gdn_transfer_geometry(geometry)
                manifest = _manifest(
                    topo,
                    topo.stage_layers(stage),
                    fa_heads=kv_local,
                    key_heads=geometry.local_num_kq_heads,
                    value_heads=geometry.local_num_v_heads,
                    block_tokens=topo.src_block,
                    per_layer_tags=per_layer_tags,
                )
                worker._raiden_manifest = manifest
                worker._raiden_layout_fingerprint_payload = dict(_PAYLOAD)
                worker._stage3_state_group_count = 1
                facade = _FakeRaidenControllerFacade()
                worker._raiden_controller_facade = facade
                worker._raiden_controller_address = CONTROLLER
                with (
                    patch(
                        "vllm.distributed.parallel_state.get_pp_group",
                        return_value=SimpleNamespace(rank_in_group=stage),
                    ),
                    patch("vllm_torchtpu.distributed.pcp.get_pcp_rank", return_value=p),
                ):
                    rank = worker._local_raiden_transfer_rank()
                    expected_rank = (
                        stage * topo.pcp + p if topo.pp > 1 else p * topo.ptp + t
                    )
                    assert rank == expected_rank
                    worker._validate_stage3_transfer_parallelism(topo.units)
                    unit = _unit(worker._raiden_work_unit_fields(rank))
                    worker._raiden_work_unit = unit
                    metadata = tc.TPUConnectorMetadata()
                    metadata.reqs_to_send["request"] = send
                    worker._register_stage3_request_blocks(metadata)
                (call,) = facade.register_request_blocks_calls
                assert call["unit"] == unit
                registrations[unit] = call["pool_spans"]
                pools[unit] = _pool_layers(manifest)
                live[unit] = _live_bytes(manifest)

                # Rank-local source memory: owned kernel pages packed dense
                # over the scheduler block prefix; the stage's GDN states.
                fa_heads = _tp_kv_heads(topo.kv_heads, topo.ptp, t)
                owned = list(range(p, num_pages, topo.pcp))
                key_heads, value_heads = _gdn_heads(
                    topo.key_heads, topo.value_heads, gdn_degree, t * topo.pcp + p
                )
                for layer in topo.stage_layers(stage):
                    if not is_fa(layer):
                        memory[(unit, layer, "conv", SRC_STATE_BLOCK)] = _conv_state(
                            layer, key_heads, value_heads, topo.qk_pair
                        )
                        memory[(unit, layer, "ssm", SRC_STATE_BLOCK)] = _ssm_state(
                            layer, value_heads
                        )
                        continue
                    page_bytes = kv_local * 2 * topo.head_dim * KERNEL_PAGE
                    for local, page in enumerate(owned):
                        block, slot = divmod(local, kps_per_block)
                        key = (unit, layer, "fa", send.local_block_ids[block])
                        buffer = memory.setdefault(
                            key,
                            np.full(
                                kps_per_block * page_bytes, SRC_FILL, dtype=np.uint8
                            ),
                        )
                        buffer[slot * page_bytes : (slot + 1) * page_bytes] = _fa_page(
                            topo, layer, page, fa_heads
                        )
    return params, registrations, pools, live, memory


# ---------------------------------------------------------------------------
# Decode: scheduler allocation and the leader's single transfer request.
# ---------------------------------------------------------------------------


def _decode_request(monkeypatch, topo: Topology, params: dict, per_layer_tags: bool):
    _set_env("TPU_RAIDEN_ENGINE_ID", "decode")
    num_tokens = params["num_tokens"]
    scheduler = _make_raiden_scheduler(
        is_producer=False, tp_size=topo.dtp, block_size=topo.dst_block
    )
    scheduler.vllm_config.parallel_config.decode_context_parallel_size = topo.dcp
    scheduler._stage3_mamba_group_indices = [1]
    assert scheduler.get_finished_count() == topo.dtp
    pages = math.ceil(num_tokens / (topo.dst_block * topo.dcp))
    order = np.random.default_rng(num_tokens).permutation(pages)
    dst_ids = [40 + 9 * int(index) for index in order]
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (dst_ids, [3, DST_STATE_BLOCK])
    # The decode prompt has one token more than the transferred prefix.
    request = MagicMock(
        request_id="decode-request",
        num_prompt_tokens=num_tokens + 1,
        num_computed_tokens=0,
        kv_transfer_params=dict(params),
    )
    scheduler.update_state_after_alloc(request, blocks, num_tokens)
    load = scheduler.reqs_to_load["decode-request"]
    assert list(load.local_block_ids) == dst_ids
    assert load.mamba_state_block_ids == [DST_STATE_BLOCK]

    worker = _make_raiden_worker(
        is_producer=False, tp_rank=0, tp_size=topo.dtp, block_size=topo.dst_block
    )
    worker.vllm_config.parallel_config.decode_context_parallel_size = topo.dcp
    _set_model(worker.vllm_config, topo)
    assert worker._raiden_qwen35_admission_topology() == (
        f"tp{topo.dtp}dcp{topo.dcp}_decode"
    )
    geometry = worker._gdn_head_geometry()
    manifest = _manifest(
        topo,
        topo.layers(),
        fa_heads=max(1, topo.kv_heads // topo.dtp),
        key_heads=geometry.local_num_kq_heads,
        value_heads=geometry.local_num_v_heads,
        block_tokens=topo.dst_block,
        per_layer_tags=per_layer_tags,
    )
    worker._raiden_manifest = manifest
    worker._stage3_state_group_count = 1
    monkeypatch.setattr(worker, "_new_raiden_id", _unit)
    monkeypatch.setattr(worker, "_raiden_hbm_memory_type", lambda: 3)
    facade = _FakeRaidenControllerFacade()
    worker._stage3_source_facades[CONTROLLER] = facade
    worker._raiden_work_unit = _unit(worker._raiden_work_unit_fields(0))
    outcome = worker._submit_stage3_load_as_leader(
        req_meta=load,
        source_req_id=load.source_req_id,
        destination_req_id="decode-request",
        uuid=load.uuid,
        num_tokens=load.num_tokens,
        local_blocks=list(load.local_block_ids),
        dst_controller_address="localhost:27001",
        dst_units=worker._raiden_destination_work_units(),
    )
    assert outcome[0] == "accepted", outcome
    (transfer,) = facade.start_transfer_calls
    return transfer, manifest, dst_ids


# ---------------------------------------------------------------------------
# Planner port: gather, split, validate coverage, apply.
# ---------------------------------------------------------------------------


def _split_global_space(entry, dst_live: int) -> list:
    """dst_space_version=1 -> page-indexed spans at destination pages."""
    if entry.dst_space_version == 0:
        return list(entry.spans)
    spans = []
    for span in entry.spans:
        assert span.count == 1 and span.src_stride_bytes == 0
        assert span.dst_stride_bytes == 0 and span.dst_block_index == 0
        remaining, offset = span.size_bytes, span.dst_offset_bytes
        src = span.src_offset_bytes
        while remaining:
            page, in_page = divmod(offset, dst_live)
            take = min(remaining, dst_live - in_page)
            spans.append(
                dataclasses.replace(
                    span,
                    src_offset_bytes=src,
                    dst_block_index=page,
                    dst_offset_bytes=in_page,
                    size_bytes=take,
                )
            )
            offset, src, remaining = offset + take, src + take, (remaining - take)
    return spans


def _validate_coverage(
    declared, *, num_dst: int, num_blocks: int, src_live: int, dst_live: int, tag: str
) -> None:
    """The planner's ValidateSpanCoverage rules for one tag."""
    coverage = [[[] for _ in range(num_blocks)] for _ in range(num_dst)]
    declared_total = covered_total = replicated_total = 0
    for entry, spans in declared:
        declared_total += entry.declared_bytes
        for span in spans:
            ordinal = -1 if span.dst_unit_ordinal is None else span.dst_unit_ordinal
            assert ordinal < num_dst, tag
            assert span.dst_block_index < num_blocks, tag
            assert (
                span.src_offset_bytes
                + (span.count - 1) * span.src_stride_bytes
                + span.size_bytes
            ) <= src_live, tag
            assert (
                span.dst_offset_bytes
                + (span.count - 1) * span.dst_stride_bytes
                + span.size_bytes
            ) <= dst_live, tag
            for repeat in range(span.count):
                start = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                interval = (start, start + span.size_bytes)
                if ordinal < 0:
                    for d in range(num_dst):
                        coverage[d][span.dst_block_index].append(interval)
                    replicated_total += span.size_bytes
                else:
                    coverage[ordinal][span.dst_block_index].append(interval)
                covered_total += span.size_bytes
    extents = None
    for d in range(num_dst):
        dst_extents = []
        for index in range(num_blocks):
            intervals = sorted(coverage[d][index])
            assert intervals, f"{tag}: block {index} uncovered at {d}"
            covered_until = 0
            for start, end in intervals:
                problem = "overlap" if start < covered_until else "gap"
                assert start == covered_until, (
                    f"{tag}: coverage {problem} at byte {min(start, covered_until)}"
                    f" of block {index} at destination {d}"
                )
                covered_until = end
            if index != num_blocks - 1:
                assert covered_until == dst_live, (tag, d, index)
            dst_extents.append(covered_until)
        if extents is None:
            extents = dst_extents
        assert dst_extents == extents, f"{tag}: non-uniform extents at {d}"
    assert declared_total == covered_total, (tag, declared_total, covered_total)
    assert num_dst * sum(extents) == (covered_total + (num_dst - 1) * replicated_total)


def _plan_and_apply(
    transfer, registrations, src_pools, src_live, src_memory, dst_manifest
) -> dict:
    src_units = list(transfer["src_units"])
    dst_units = list(transfer["dst_units"])
    assert set(src_units) == set(registrations)
    dst_pools = _pool_layers(dst_manifest)
    dst_live = _live_bytes(dst_manifest)
    tags = list(transfer["transfer_pool_tags"])
    counts = list(transfer["dst_block_counts"])
    flat_ids = list(transfer["dst_device_block_ids"])
    assert len(tags) == len(counts) and sum(counts) == len(flat_ids)
    assert sorted(tags) == sorted(dst_pools)
    dst_memory: dict = {}
    cursor = 0
    for tag, count in zip(tags, counts):
        dst_ids = flat_ids[cursor : cursor + count]
        cursor += count
        layers = dst_pools[tag]
        holders = [unit for unit in src_units if tag in src_pools[unit]]
        assert holders, f"no source unit registers pools for {tag}"
        live_values = {src_live[unit][tag] for unit in holders}
        assert len(live_values) == 1, (tag, live_values)
        declared = []
        for unit in src_units:
            for entry in registrations[unit]:
                if entry.tag != tag or not entry.spans:
                    continue
                assert tag in src_pools[unit], f"{unit} declares {tag} without a pool"
                assert len(src_pools[unit][tag]) == len(layers)
                declared.append(
                    (unit, entry, _split_global_space(entry, dst_live[tag]))
                )
        assert declared, f"no declarations for {tag}"
        _validate_coverage(
            [(e, s) for _, e, s in declared],
            num_dst=len(dst_units),
            num_blocks=len(dst_ids),
            src_live=live_values.pop(),
            dst_live=dst_live[tag],
            tag=tag,
        )
        kind = _kind(tag)
        for unit, entry, spans in declared:
            for span in spans:
                targets = (
                    range(len(dst_units))
                    if span.dst_unit_ordinal is None
                    else [span.dst_unit_ordinal]
                )
                src_block = entry.block_ids[span.src_block_ordinal]
                dst_block = dst_ids[span.dst_block_index]
                for k, layer in enumerate(layers):
                    src_layer = src_pools[unit][tag][k]
                    source = src_memory[(unit, src_layer, kind, src_block)]
                    for repeat in range(span.count):
                        src = span.src_offset_bytes + repeat * span.src_stride_bytes
                        dst = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                        payload = source[src : src + span.size_bytes]
                        assert payload.size == span.size_bytes
                        for d in targets:
                            buffer = dst_memory.setdefault(
                                (d, layer, kind, dst_block),
                                np.full(dst_live[tag], DST_FILL, dtype=np.uint8),
                            )
                            buffer[dst : dst + span.size_bytes] = payload
    return dst_memory


# ---------------------------------------------------------------------------
# Expected destination memory.
# ---------------------------------------------------------------------------


def _check_destination(
    topo: Topology, num_tokens: int, dst_ids: list[int], dst_memory: dict
) -> None:
    num_pages = math.ceil(num_tokens / KERNEL_PAGE)
    stripes = math.ceil(num_pages / topo.dcp)
    fa_local = max(1, topo.kv_heads // topo.dtp)
    page_bytes = fa_local * 2 * topo.head_dim * KERNEL_PAGE
    dst_live = topo.dst_block // KERNEL_PAGE * page_bytes
    for d in range(topo.dtp):
        heads = _tp_kv_heads(topo.kv_heads, topo.dtp, d)
        lane = d % topo.dcp
        pages = list(range(lane, num_pages, topo.dcp))
        # The final stripe pads missing lanes with the last page.
        pages += [num_pages - 1] * (stripes - len(pages))
        key_heads, value_heads = _gdn_heads(
            topo.key_heads, topo.value_heads, topo.dtp, d
        )
        for layer in topo.layers():
            if not is_fa(layer):
                np.testing.assert_array_equal(
                    dst_memory[(d, layer, "conv", DST_STATE_BLOCK)],
                    _conv_state(layer, key_heads, value_heads, topo.qk_pair),
                )
                np.testing.assert_array_equal(
                    dst_memory[(d, layer, "ssm", DST_STATE_BLOCK)],
                    _ssm_state(layer, value_heads),
                )
                continue
            expected = np.concatenate(
                [_fa_page(topo, layer, page, heads) for page in pages]
            )
            assert len(dst_ids) == math.ceil(expected.size / dst_live)
            for index, block in enumerate(dst_ids):
                actual = dst_memory[(d, layer, "fa", block)]
                chunk = expected[index * dst_live : (index + 1) * dst_live]
                np.testing.assert_array_equal(actual[: chunk.size], chunk)
                assert (actual[chunk.size :] == DST_FILL).all()
    written = {(d, layer) for d, layer, _, _ in dst_memory}
    assert written == {(d, layer) for d in range(topo.dtp) for layer in topo.layers()}


def _check_fa_copy_dealing(topo: Topology, registrations: dict) -> None:
    """Each decode DCP group holding a KV head takes one full FA copy, dealt
    round-robin over the source TP replicas of that head."""
    src_replicas = topo.ptp // min(topo.kv_heads, topo.ptp)
    dst_replicas = topo.dtp // min(topo.kv_heads, topo.dtp)
    for unit, entries in registrations.items():
        replica = _unit_rank(unit) % topo.ptp % src_replicas
        for entry in entries:
            if class_tag(entry.tag) != "fa":
                continue
            for span in entry.spans:
                copy = span.dst_unit_ordinal % dst_replicas // topo.dcp
                assert copy % src_replicas == replica, (unit, span)


def _run_oracle(monkeypatch, topo: Topology, num_tokens: int, per_layer_tags: bool):
    _environment(monkeypatch, topo, per_layer_tags)
    params, registrations, src_pools, src_live, src_memory = _produce(
        monkeypatch, topo, num_tokens, per_layer_tags
    )
    _check_fa_copy_dealing(topo, registrations)
    transfer, dst_manifest, dst_ids = _decode_request(
        monkeypatch, topo, params, per_layer_tags
    )
    assert transfer["num_tokens"] == num_tokens
    dst_memory = _plan_and_apply(
        transfer, registrations, src_pools, src_live, src_memory, dst_manifest
    )
    _check_destination(topo, num_tokens, dst_ids, dst_memory)
    return registrations, transfer


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_pp4_pcp8_to_tp32_dcp8_oracle(monkeypatch, num_tokens):
    registrations, transfer = _run_oracle(monkeypatch, PP4_PCP8, num_tokens, True)
    assert len(transfer["src_units"]) == 32
    # Each stage's units declare only that stage's layers.
    for unit, entries in registrations.items():
        rank = _unit_rank(unit)
        stage = rank // PP4_PCP8.pcp
        layers = {split_layer_tag(entry.tag)[1] for entry in entries}
        assert layers <= set(PP4_PCP8.stage_layers(stage))
        # The GDN state of every unit reaches four destinations, and the
        # two destinations sharing a key head both receive its Q/K rows.
        for entry in entries:
            if class_tag(entry.tag).startswith("gdn."):
                assert {span.dst_unit_ordinal for span in entry.spans} == set(
                    range(4 * (rank % 8), 4 * (rank % 8) + 4)
                )


@pytest.mark.parametrize("per_layer_tags", [False, True])
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_pcp16_tp2_to_tp32_dcp8_oracle(monkeypatch, num_tokens, per_layer_tags):
    _run_oracle(monkeypatch, PCP16_TP2, num_tokens, per_layer_tags)


@pytest.mark.parametrize(
    "topo", [PP2_PCP2, PP2_PCP2_QK_PAIR], ids=["split-qkv", "qk-pair"]
)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_pp2_pcp2_to_tp4_dcp2_oracle(monkeypatch, topo, num_tokens):
    _run_oracle(monkeypatch, topo, num_tokens, True)


@pytest.mark.parametrize(
    "topo,per_layer_tags",
    [(HALF_PCP8_TP2, False), (HALF_PCP8_TP2, True), (HALF_PP2_PCP8, True)],
    ids=["pcp8tp2-class-tags", "pcp8tp2", "pp2pcp8"],
)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_half_scale_to_tp16_dcp8_oracle(monkeypatch, topo, per_layer_tags, num_tokens):
    registrations, _ = _run_oracle(monkeypatch, topo, num_tokens, per_layer_tags)
    # PCP8 x TP2 maps GDN state 1:1 (TP-major); a PP2 x PCP8 rank fans out
    # to two destinations that both replicate its key head.
    for unit, entries in registrations.items():
        rank = _unit_rank(unit)
        if topo.pp > 1:
            p = rank % topo.pcp
            expected = {2 * p, 2 * p + 1}
        else:
            p, t = divmod(rank, topo.ptp)
            expected = {t * topo.pcp + p}
        for entry in entries:
            if class_tag(entry.tag).startswith("gdn."):
                assert {span.dst_unit_ordinal for span in entry.spans} == expected


@pytest.mark.parametrize("num_tokens", TOKENS)
def test_replicated_fa_head_deals_copies_oracle(monkeypatch, num_tokens):
    registrations, _ = _run_oracle(monkeypatch, PCP4_TP2_ONE_KV_HEAD, num_tokens, False)
    senders = {
        _unit_rank(unit) % PCP4_TP2_ONE_KV_HEAD.ptp
        for unit, entries in registrations.items()
        for entry in entries
        if class_tag(entry.tag) == "fa" and entry.spans
    }
    assert senders == {0, 1}


def test_oracle_rejects_a_misrouted_span(monkeypatch):
    """The coverage port fails on a destination gap or overlap."""
    topo = PP2_PCP2_QK_PAIR
    _environment(monkeypatch, topo, True)
    params, registrations, src_pools, src_live, src_memory = _produce(
        monkeypatch, topo, 1000, True
    )
    transfer, dst_manifest, _ = _decode_request(monkeypatch, topo, params, True)
    unit = next(iter(registrations))
    entries = list(registrations[unit])
    index = next(
        i for i, entry in enumerate(entries) if class_tag(entry.tag) == "gdn.ssm.g0"
    )
    entry = entries[index]
    first = entry.spans[0]
    entries[index] = dataclasses.replace(
        entry,
        spans=(
            dataclasses.replace(
                first, dst_unit_ordinal=entry.spans[1].dst_unit_ordinal
            ),
        )
        + entry.spans[1:],
    )
    registrations[unit] = entries
    with pytest.raises(AssertionError, match="overlap|uncovered"):
        _plan_and_apply(
            transfer, registrations, src_pools, src_live, src_memory, dst_manifest
        )


@pytest.mark.parametrize(
    "keys,values,degree",
    [(16, 128, 8), (16, 32, 2), (16, 32, 4), (16, 128, 32), (4, 32, 16)],
)
def test_stored_conv_order_matches_kernel_rows_perm(keys, values, degree):
    """The oracle's stored conv order is the pooled kernel's rows_perm."""
    pytest.importorskip("jax")
    from vllm_torchtpu.kernels import pool_adapters

    key_heads, value_heads = _gdn_heads(keys, values, degree, 0)
    conv_dim = 2 * len(key_heads) * D_K + len(value_heads) * D_V
    stored = _conv_state(3, key_heads, value_heads, True).view(np.uint32)
    logical = _conv_state(3, key_heads, value_heads, False).view(np.uint32)
    lanes = LANE_BYTES // BF16
    width = math.gcd(math.gcd(conv_dim, lanes), len(key_heads) * D_K)
    perm = pool_adapters._conv_qk_pair_rows_perm(
        TAPS, conv_dim, len(value_heads), D_V, width
    )
    row_words = width * BF16 // 4
    stored_rows = stored.reshape(-1, row_words)
    logical_rows = logical.reshape(-1, row_words)
    if perm is None:
        np.testing.assert_array_equal(stored_rows, logical_rows)
    else:
        np.testing.assert_array_equal(stored_rows[list(perm)], logical_rows)


# ---------------------------------------------------------------------------
# GDN fan-out lowering on its own.
# ---------------------------------------------------------------------------


def _state_regions(kind: str, keys: int, values: int, degree: int, qk_pair: bool):
    _set_env("TPU_GDN_CONV_QK_PAIR_LAYOUT", qk_pair)
    key_heads, value_heads = _gdn_heads(keys, values, degree, 0)
    if kind == "ssm":
        return rpm._gdn_ssm_regions(
            ssm_shape=(4, len(value_heads), D_V, D_K), itemsize=BF16
        )
    gdn = rpm.GdnHeadGeometry(
        local_key_heads=len(key_heads),
        local_value_heads=len(value_heads),
        key_head_dim=D_K,
        value_head_dim=D_V,
    )
    return rpm._gdn_conv_regions(
        conv_shape=(4, TAPS, gdn.conv_dim), itemsize=BF16, geometry=gdn
    )


@pytest.mark.parametrize(
    "src,dst,keys,values,qk_pair",
    [
        (8, 32, 16, 128, True),  # PP4 x PCP8 -> TP32: Q/K replication 1 -> 2.
        (2, 4, 16, 32, True),  # Destinations keep four key heads.
        (2, 4, 16, 32, False),
        (4, 8, 16, 32, False),
        (4, 16, 4, 32, True),  # Q/K replication 1 -> 4.
        (8, 16, 4, 32, True),  # Q/K replicated on both sides, 2 -> 4.
        (8, 16, 8, 64, True),  # Half-scale PP2 x PCP8 -> TP16.
    ],
)
@pytest.mark.parametrize("kind", ["conv", "ssm"])
def test_gdn_fanout_reassembles_destination_states(
    src, dst, keys, values, qk_pair, kind
):
    regions = _state_regions(kind, keys, values, src, qk_pair)
    tag = f"gdn.{kind}.g0.l2"

    def state(degree, rank):
        key_heads, value_heads = _gdn_heads(keys, values, degree, rank)
        if kind == "ssm":
            return _ssm_state(2, value_heads)
        return _conv_state(2, key_heads, value_heads, qk_pair)

    received = [np.full(state(dst, d).size, DST_FILL, np.uint8) for d in range(dst)]
    coverage = [np.zeros(x.size, np.int32) for x in received]
    for rank in range(src):
        payload = state(src, rank)
        reg = sol.lower_gdn_state_shard_spans_sol(
            tag=tag,
            block_id=9,
            transfer_rank=rank,
            parallelism=src,
            regions=regions,
            dst_shards=dst,
            physical_granule_bytes=GRANULE,
            num_key_heads=keys,
            num_value_heads=values,
            conv_lane_bytes=LANE_BYTES,
        )
        fan_out = dst // src
        assert reg.block_ids == (9,)
        assert sol.registration_source_bytes(reg) == payload.size
        assert reg.declared_bytes == fan_out * received[0].size
        assert {span.dst_unit_ordinal for span in reg.spans} == set(
            range(rank * fan_out, (rank + 1) * fan_out)
        )
        for span in reg.spans:
            assert span.size_bytes % GRANULE == 0
            for repeat in range(span.count):
                src_at = span.src_offset_bytes + repeat * span.src_stride_bytes
                dst_at = span.dst_offset_bytes + repeat * span.dst_stride_bytes
                d = span.dst_unit_ordinal
                received[d][dst_at : dst_at + span.size_bytes] = payload[
                    src_at : src_at + span.size_bytes
                ]
                coverage[d][dst_at : dst_at + span.size_bytes] += 1
    for d in range(dst):
        np.testing.assert_array_equal(received[d], state(dst, d))
        assert (coverage[d] == 1).all()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"conv_lane_bytes": 512}, "row pairs differ"),
        ({"physical_granule_bytes": 1024}, "physical granule"),
        ({"conv_lane_bytes": None}, "lane width"),
        ({"num_key_heads": None}, "head counts"),
        ({"dst_shards": 12}, "must divide"),
    ],
)
def test_gdn_fanout_fails_closed(overrides, match):
    args = dict(
        tag="gdn.conv.g0.l0",
        block_id=9,
        transfer_rank=3,
        parallelism=8,
        regions=_state_regions("conv", 16, 128, 8, True),
        dst_shards=32,
        physical_granule_bytes=GRANULE,
        num_key_heads=16,
        num_value_heads=128,
        conv_lane_bytes=LANE_BYTES,
    )
    args.update(overrides)
    with pytest.raises(sol.SeqOnLaneError, match=match):
        sol.lower_gdn_state_shard_spans_sol(**args)
