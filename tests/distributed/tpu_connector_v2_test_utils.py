# SPDX-License-Identifier: Apache-2.0
import enum
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm_torchtpu.distributed.kv_transfer.v2.common import (LayerType,
                                                             TensorLayout)
from vllm_torchtpu.distributed.kv_transfer.v2.layout import (
    HeadSegment, KVCacheRegion, TokenFirstLayoutSpec)
from vllm_torchtpu.distributed.kv_transfer.v2.metadata import (
    ConnectorMetadataV2, HeadMapping, KVParallelLayout, LocalDecodeAllocation,
    PullMeta, RankTransferPlan, SourceBlockRef, StridedSegmentOp, TpKVTopology)
from vllm_torchtpu.distributed.kv_transfer.v2.pcp_policy import (
    PcpReshardingPolicy, PcpTokenTransfer)
from vllm_torchtpu.distributed.kv_transfer.v2.planner import (
    ContiguousHeadTPTransferPlanner, TPTransferPlanner)


def _load_v2_module(monkeypatch: Any, *, stub_zmq: bool = True):
    root = Path(__file__).parents[2]
    v2_dir = root / "src/vllm_torchtpu/distributed/kv_transfer/v2"

    vllm = types.ModuleType("vllm")
    vllm_config = types.ModuleType("vllm.config")
    vllm_config.VllmConfig = type("VllmConfig", (), {})
    vllm_logger = types.ModuleType("vllm.logger")
    vllm_logger._VllmLogger = logging.Logger
    vllm_logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)

    if stub_zmq or importlib.util.find_spec("zmq") is None:
        zmq = types.ModuleType("zmq")

        class FakeZmqContext:

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def destroy(self, linger=0):
                self.linger = linger

        zmq.Context = FakeZmqContext
        monkeypatch.setitem(sys.modules, "zmq", zmq)

    base_name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    for name in [
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    base = types.ModuleType(base_name)
    base.KVConnectorRole = enum.Enum("KVConnectorRole", "SCHEDULER WORKER")
    base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
    base.SupportsHMA = type("SupportsHMA", (), {})

    class KVConnectorBase_V1:

        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._vllm_config = vllm_config
            self._role = role
            self._kv_cache_config = kv_cache_config
            self._connector_metadata = None

    base.KVConnectorBase_V1 = KVConnectorBase_V1
    monkeypatch.setitem(sys.modules, base_name, base)

    for name, package_path in [
        ("vllm_torchtpu", root / "src/vllm_torchtpu"),
        ("vllm_torchtpu.distributed", root / "src/vllm_torchtpu/distributed"),
        ("vllm_torchtpu.distributed.kv_transfer",
         root / "src/vllm_torchtpu/distributed/kv_transfer"),
        ("vllm_torchtpu.distributed.kv_transfer.v2", v2_dir),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        monkeypatch.setitem(sys.modules, name, package)

    tpu_envs = types.ModuleType("vllm_torchtpu.envs")
    tpu_envs.TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL = True
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.envs", tpu_envs)

    def get_loopback_host() -> str:
        return "127.0.0.1"

    def get_kv_port() -> int:
        return 9100

    def get_no_coord_workers() -> int:
        return 0

    def get_default_pull_timeout() -> float:
        return 120.0

    def get_side_channel_port() -> str:
        return "9600"

    def get_node_id() -> int:
        return 0

    dist_utils = types.ModuleType("vllm_torchtpu.distributed.utils")
    dist_utils.get_host_ip = get_loopback_host
    dist_utils.get_kv_ips = get_loopback_host
    dist_utils.get_kv_ports = get_kv_port
    dist_utils.get_kv_transfer_port = get_kv_port
    dist_utils.get_kv_coord_executor_max_workers = get_no_coord_workers
    dist_utils.get_p2p_wait_pull_timeout = get_default_pull_timeout
    dist_utils.get_side_channel_port = get_side_channel_port
    dist_utils.get_node_id = get_node_id
    monkeypatch.setitem(sys.modules, "vllm_torchtpu.distributed.utils",
                        dist_utils)

    for name in tuple(sys.modules):
        if name.startswith("vllm_torchtpu.distributed.kv_transfer.v2."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module_name = "vllm_torchtpu.distributed.kv_transfer.v2.tpu_connector"
    spec = importlib.util.spec_from_file_location(
        module_name,
        v2_dir / "tpu_connector.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def load_v2_module():
    return SimpleNamespace(
        ConnectorMetadataV2=ConnectorMetadataV2,
        ContiguousHeadTPTransferPlanner=ContiguousHeadTPTransferPlanner,
        HeadSegment=HeadSegment,
        HeadMapping=HeadMapping,
        KVCacheRegion=KVCacheRegion,
        KVParallelLayout=KVParallelLayout,
        LayerType=LayerType,
        LocalDecodeAllocation=LocalDecodeAllocation,
        PcpReshardingPolicy=PcpReshardingPolicy,
        PcpTokenTransfer=PcpTokenTransfer,
        PullMeta=PullMeta,
        RankTransferPlan=RankTransferPlan,
        SourceBlockRef=SourceBlockRef,
        StridedSegmentOp=StridedSegmentOp,
        TensorLayout=TensorLayout,
        TokenFirstLayoutSpec=TokenFirstLayoutSpec,
        TPTransferPlanner=TPTransferPlanner,
        TpKVTopology=TpKVTopology,
    )


def qwen35_4tp_layer_layout():
    raw_layout = []
    for raw_idx in range(10):
        base_layer = raw_idx * 4
        raw_layout.append({
            "raw_idx":
            raw_idx,
            "linear_layers":
            tuple(f"model.layers.{base_layer + offset}.linear_attn"
                  for offset in range(3)),
            "full_layer":
            f"model.layers.{base_layer + 3}.self_attn",
        })
    return tuple(raw_layout)


def layout(mod: Any, tp_size: int, pcp_size: int = 1):
    return mod.KVParallelLayout(
        full_attn_pcp_size=pcp_size,
        full_attn_tp_size=tp_size,
        linear_attn_pcp_size=pcp_size,
        linear_attn_tp_size=tp_size,
    )


def fa_pcp_source_layout(mod: Any, pcp_size: int):
    return mod.KVParallelLayout(
        full_attn_pcp_size=pcp_size,
        full_attn_tp_size=pcp_size,
        linear_attn_pcp_size=1,
        linear_attn_tp_size=1,
    )


def head_mapping_rows(mappings):
    return tuple((
        mapping.global_head,
        mapping.source_head,
        mapping.destination_head,
        mapping.segment_name,
    ) for mapping in mappings)


def block_ref_rows(block_refs):
    return tuple((block_ref.logical_block_index, block_ref.block_id)
                 for block_ref in block_refs)


def manual_pull_meta(
    mod: Any,
    *,
    req_id: int = 1,
    p_ranks: tuple[int, ...],
    fa_mapping_rows_by_rank=None,
    mamba_mapping_rows_by_rank=None,
    fa_source_block_ids: tuple[int, ...] = (),
    mamba_source_block_ids: tuple[int, ...] = (),
    fa_block_ref_rows_by_rank=None,
):
    fa_mapping_rows_by_rank = fa_mapping_rows_by_rank or {}
    mamba_mapping_rows_by_rank = mamba_mapping_rows_by_rank or {}
    fa_block_ref_rows_by_rank = fa_block_ref_rows_by_rank or {}

    fa_head_mappings_by_rank = build_head_mappings(mod,
                                                   fa_mapping_rows_by_rank)
    mamba_value_head_mappings_by_rank = build_head_mappings(
        mod, mamba_mapping_rows_by_rank)
    return mod.PullMeta(
        req_id=req_id,
        p_ranks=p_ranks,
        fa_heads_by_rank=heads_by_rank(fa_mapping_rows_by_rank),
        mamba_key_heads_by_rank=heads_by_rank(mamba_mapping_rows_by_rank),
        mamba_value_heads_by_rank=heads_by_rank(mamba_mapping_rows_by_rank),
        fa_head_mappings_by_rank=fa_head_mappings_by_rank,
        mamba_key_head_mappings_by_rank=mamba_value_head_mappings_by_rank,
        mamba_value_head_mappings_by_rank=mamba_value_head_mappings_by_rank,
        fa_source_block_ids=fa_source_block_ids,
        mamba_source_block_ids=mamba_source_block_ids,
        fa_block_refs_by_rank={
            rank:
            tuple(
                mod.SourceBlockRef(
                    logical_block_index=logical_block_index,
                    block_id=block_id,
                ) for logical_block_index, block_id in block_refs)
            for rank, block_refs in fa_block_ref_rows_by_rank.items()
        },
    )


def build_head_mappings(mod: Any, mapping_rows_by_rank):
    return {
        rank:
        tuple(
            mod.HeadMapping(
                global_head=global_head,
                source_head=source_head,
                destination_head=destination_head,
                segment_name=segment_name,
            ) for global_head, source_head, destination_head, segment_name in
            rows)
        for rank, rows in mapping_rows_by_rank.items()
    }


def heads_by_rank(mapping_rows_by_rank):
    return {
        rank: tuple(global_head for global_head, _, _, _ in rows)
        for rank, rows in mapping_rows_by_rank.items()
    }


def full_attention_region(
    mod: Any,
    *,
    rank: int,
    layer_name: str,
    base_addr: int,
    block_size: int,
    token_stride_bytes: int,
    head_stride_bytes: int,
    live_head_bytes: int,
    num_heads: int = 1,
    layout_id: str = "pallas_batched_rpa_token_first_v1",
):
    block_bytes = block_size * token_stride_bytes
    layout_spec = mod.TokenFirstLayoutSpec(
        block_size=block_size,
        block_bytes=block_bytes,
        block_stride_bytes=block_bytes,
        token_stride_bytes=token_stride_bytes,
        head_stride_bytes=head_stride_bytes,
        live_head_bytes=live_head_bytes,
        num_heads=num_heads,
        layout_id=layout_id,
    )
    return mod.KVCacheRegion(
        rank=rank,
        layer_name=layer_name,
        layer_type=mod.LayerType.FULL_ATTN,
        base_addr=base_addr,
        block_size=block_size,
        block_bytes=block_bytes,
        layout=mod.TensorLayout.TOKEN_FIRST,
        num_heads=num_heads,
        token_first_layout=layout_spec,
    )


def build_block_size_case(
    mod: Any,
    *,
    source_block_size: int,
    destination_block_size: int,
):
    layer_name = "model.layers.0.self_attn"
    live_head_bytes = 10
    source_region = full_attention_region(
        mod,
        rank=0,
        layer_name=layer_name,
        base_addr=1_000,
        block_size=source_block_size,
        token_stride_bytes=live_head_bytes,
        head_stride_bytes=live_head_bytes,
        live_head_bytes=live_head_bytes,
    )
    destination_region = full_attention_region(
        mod,
        rank=0,
        layer_name=layer_name,
        base_addr=2_000,
        block_size=destination_block_size,
        token_stride_bytes=live_head_bytes,
        head_stride_bytes=live_head_bytes,
        live_head_bytes=live_head_bytes,
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=1,
        block_size=source_block_size,
        kv_source_layout=layout(mod, tp_size=1),
        kv_caches={0: {
            layer_name: source_region
        }},
        fa_block_ids=(10, 11, 12),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=1),
        block_size=destination_block_size,
        tp_rank=0,
        total_num_kv_heads=1,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )
    destination = mod.LocalDecodeAllocation(
        rank=0,
        block_size=destination_block_size,
        kv_caches={layer_name: destination_region},
        fa_block_ids=(20, 21),
        mamba_block_ids=(),
    )
    return metadata, topology, destination


def build_fa_pcp_case(
        mod: Any,
        *,
        destination_tp_size: int = 2,
        destination_tp_rank: int = 1,
        destination_block_size: int = 4,
        destination_block_ids: tuple[int, ...] = (200, 201, 202, 203),
):
    layer_name = "model.layers.0.self_attn"
    source_pcp_size = 4
    source_block_size = 2
    live_head_bytes = 10
    total_num_kv_heads = 2
    destination_num_heads = total_num_kv_heads // destination_tp_size
    token_stride_bytes = live_head_bytes * total_num_kv_heads
    source_regions_by_rank = {
        rank: {
            layer_name:
            full_attention_region(
                mod,
                rank=rank,
                layer_name=layer_name,
                base_addr=10_000 + rank * 1_000,
                block_size=source_block_size,
                token_stride_bytes=token_stride_bytes,
                head_stride_bytes=live_head_bytes,
                live_head_bytes=live_head_bytes,
                num_heads=total_num_kv_heads,
            )
        }
        for rank in range(source_pcp_size)
    }
    destination = mod.LocalDecodeAllocation(
        rank=destination_tp_rank,
        block_size=destination_block_size,
        kv_caches={
            layer_name:
            full_attention_region(
                mod,
                rank=destination_tp_rank,
                layer_name=layer_name,
                base_addr=100_000,
                block_size=destination_block_size,
                token_stride_bytes=token_stride_bytes,
                head_stride_bytes=live_head_bytes,
                live_head_bytes=live_head_bytes,
                num_heads=destination_num_heads,
            )
        },
        fa_block_ids=destination_block_ids,
        mamba_block_ids=(),
    )
    metadata = mod.ConnectorMetadataV2(
        req_id=3,
        block_size=source_block_size,
        kv_source_layout=fa_pcp_source_layout(mod, pcp_size=source_pcp_size),
        kv_caches=source_regions_by_rank,
        fa_block_ids=(100, 101),
        mamba_block_ids=(),
    )
    topology = mod.TpKVTopology(
        local_layout=layout(mod, tp_size=destination_tp_size),
        block_size=destination_block_size,
        tp_rank=destination_tp_rank,
        total_num_kv_heads=total_num_kv_heads,
        total_num_mamba_key_heads=0,
        total_num_mamba_heads=0,
    )
    return metadata, topology, destination


def mamba_state_regions(
    mod: Any,
    *,
    rank: int,
    tp_rank: int,
    tp_size: int,
    base_addr: int,
    block_stride_bytes: int,
):
    layer_name = "model.layers.0.linear_attn"
    head_dim = 128
    bf16_bytes = 2
    fp32_bytes = 4
    key_heads = 16
    value_heads = 32
    local_key_heads = key_heads // tp_size
    local_value_heads = value_heads // tp_size
    local_conv_dim = (2 * key_heads * head_dim +
                      value_heads * head_dim) // tp_size
    conv_state_bytes = 3 * local_conv_dim * bf16_bytes
    recurrent_head_bytes = head_dim * head_dim * fp32_bytes
    recurrent_state_bytes = local_value_heads * recurrent_head_bytes
    conv_slot_stride_bytes = local_conv_dim * bf16_bytes
    q_heads = tuple(
        range(tp_rank * local_key_heads, (tp_rank + 1) * local_key_heads))
    v_heads = tuple(
        range(tp_rank * local_value_heads, (tp_rank + 1) * local_value_heads))
    q_offset = 0
    k_offset = local_key_heads * head_dim * bf16_bytes
    v_offset = k_offset * 2
    state0_name = f"{layer_name}.state0"
    state1_name = f"{layer_name}.state1"

    return {
        state0_name:
        mod.KVCacheRegion(
            rank=rank,
            layer_name=state0_name,
            layer_type=mod.LayerType.MAMBA_STATE,
            base_addr=base_addr,
            block_size=None,
            block_bytes=conv_state_bytes,
            block_stride_bytes=block_stride_bytes,
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=local_key_heads * 2 + local_value_heads,
            head_segments=(
                mod.HeadSegment(
                    name="q",
                    global_heads=q_heads,
                    local_head_start=0,
                    local_head_count=local_key_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=q_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
                mod.HeadSegment(
                    name="k",
                    global_heads=q_heads,
                    local_head_start=local_key_heads,
                    local_head_count=local_key_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=k_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
                mod.HeadSegment(
                    name="v",
                    global_heads=v_heads,
                    local_head_start=local_key_heads * 2,
                    local_head_count=local_value_heads,
                    head_bytes=head_dim * bf16_bytes,
                    base_offset_bytes=v_offset,
                    stride_bytes=conv_slot_stride_bytes,
                    num_segments=3,
                ),
            ),
            block_id_index=0,
        ),
        state1_name:
        mod.KVCacheRegion(
            rank=rank,
            layer_name=state1_name,
            layer_type=mod.LayerType.MAMBA_STATE,
            base_addr=base_addr + conv_state_bytes,
            block_size=None,
            block_bytes=recurrent_state_bytes,
            block_stride_bytes=block_stride_bytes,
            layout=mod.TensorLayout.BLOCKS_FIRST,
            num_heads=local_value_heads,
            block_id_index=0,
        ),
    }
