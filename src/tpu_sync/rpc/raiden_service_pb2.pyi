from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MemoryType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MEMORY_TYPE_UNSPECIFIED: _ClassVar[MemoryType]
    MEMORY_TYPE_DRAM: _ClassVar[MemoryType]
    MEMORY_TYPE_HBM: _ClassVar[MemoryType]

class HostTilingMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HOST_TILING_MODE_UNSPECIFIED: _ClassVar[HostTilingMode]
    HOST_TILING_MODE_DEFERRED: _ClassVar[HostTilingMode]
    HOST_TILING_MODE_ON_ARRIVAL: _ClassVar[HostTilingMode]
    HOST_TILING_MODE_PRE_TILED: _ClassVar[HostTilingMode]
MEMORY_TYPE_UNSPECIFIED: MemoryType
MEMORY_TYPE_DRAM: MemoryType
MEMORY_TYPE_HBM: MemoryType
HOST_TILING_MODE_UNSPECIFIED: HostTilingMode
HOST_TILING_MODE_DEFERRED: HostTilingMode
HOST_TILING_MODE_ON_ARRIVAL: HostTilingMode
HOST_TILING_MODE_PRE_TILED: HostTilingMode

class RaidenIdProto(_message.Message):
    __slots__ = ("job_name", "job_replica_id", "data_name", "data_replica_idx")
    JOB_NAME_FIELD_NUMBER: _ClassVar[int]
    JOB_REPLICA_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_REPLICA_IDX_FIELD_NUMBER: _ClassVar[int]
    job_name: str
    job_replica_id: str
    data_name: str
    data_replica_idx: int
    def __init__(self, job_name: _Optional[str] = ..., job_replica_id: _Optional[str] = ..., data_name: _Optional[str] = ..., data_replica_idx: _Optional[int] = ...) -> None: ...

class NDSliceDimensionProto(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: int
    end: int
    def __init__(self, start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class NDSliceProto(_message.Message):
    __slots__ = ("dimensions",)
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    dimensions: _containers.RepeatedCompositeFieldContainer[NDSliceDimensionProto]
    def __init__(self, dimensions: _Optional[_Iterable[_Union[NDSliceDimensionProto, _Mapping]]] = ...) -> None: ...

class RegionSpecProto(_message.Message):
    __slots__ = ("name", "offset_bytes", "stride_bytes", "unit_bytes", "num_units", "units_per_stride")
    NAME_FIELD_NUMBER: _ClassVar[int]
    OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    UNIT_BYTES_FIELD_NUMBER: _ClassVar[int]
    NUM_UNITS_FIELD_NUMBER: _ClassVar[int]
    UNITS_PER_STRIDE_FIELD_NUMBER: _ClassVar[int]
    name: str
    offset_bytes: int
    stride_bytes: int
    unit_bytes: int
    num_units: int
    units_per_stride: int
    def __init__(self, name: _Optional[str] = ..., offset_bytes: _Optional[int] = ..., stride_bytes: _Optional[int] = ..., unit_bytes: _Optional[int] = ..., num_units: _Optional[int] = ..., units_per_stride: _Optional[int] = ...) -> None: ...

class PoolSpecProto(_message.Message):
    __slots__ = ("tag", "storage_index", "base_offset_bytes", "block_stride_bytes", "num_blocks", "regions", "dtype_tag")
    TAG_FIELD_NUMBER: _ClassVar[int]
    STORAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    BASE_OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    BLOCK_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    NUM_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_TAG_FIELD_NUMBER: _ClassVar[int]
    tag: str
    storage_index: int
    base_offset_bytes: int
    block_stride_bytes: int
    num_blocks: int
    regions: _containers.RepeatedCompositeFieldContainer[RegionSpecProto]
    dtype_tag: str
    def __init__(self, tag: _Optional[str] = ..., storage_index: _Optional[int] = ..., base_offset_bytes: _Optional[int] = ..., block_stride_bytes: _Optional[int] = ..., num_blocks: _Optional[int] = ..., regions: _Optional[_Iterable[_Union[RegionSpecProto, _Mapping]]] = ..., dtype_tag: _Optional[str] = ...) -> None: ...

class VariableMetadataProto(_message.Message):
    __slots__ = ("name", "shape", "mesh_shape", "layout", "item_size", "layer_idx", "sharding_spec", "global_shard_indices")
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    MESH_SHAPE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    ITEM_SIZE_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    SHARDING_SPEC_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_SHARD_INDICES_FIELD_NUMBER: _ClassVar[int]
    name: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    mesh_shape: _containers.RepeatedScalarFieldContainer[int]
    layout: _containers.RepeatedScalarFieldContainer[int]
    item_size: int
    layer_idx: int
    sharding_spec: _containers.RepeatedScalarFieldContainer[str]
    global_shard_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., mesh_shape: _Optional[_Iterable[int]] = ..., layout: _Optional[_Iterable[int]] = ..., item_size: _Optional[int] = ..., layer_idx: _Optional[int] = ..., sharding_spec: _Optional[_Iterable[str]] = ..., global_shard_indices: _Optional[_Iterable[int]] = ...) -> None: ...

class RegisterWorkUnitRequest(_message.Message):
    __slots__ = ("unit", "shards", "control_plane_rpc_address", "itemsize", "mesh_shape", "layout", "global_shape", "pools", "layout_fingerprint", "page_tokens", "transfer_parallelism", "transfer_rank", "variables", "mesh_axes", "host_subgrid")
    UNIT_FIELD_NUMBER: _ClassVar[int]
    SHARDS_FIELD_NUMBER: _ClassVar[int]
    CONTROL_PLANE_RPC_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    ITEMSIZE_FIELD_NUMBER: _ClassVar[int]
    MESH_SHAPE_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_SHAPE_FIELD_NUMBER: _ClassVar[int]
    POOLS_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_RANK_FIELD_NUMBER: _ClassVar[int]
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    MESH_AXES_FIELD_NUMBER: _ClassVar[int]
    HOST_SUBGRID_FIELD_NUMBER: _ClassVar[int]
    unit: RaidenIdProto
    shards: _containers.RepeatedScalarFieldContainer[str]
    control_plane_rpc_address: str
    itemsize: int
    mesh_shape: _containers.RepeatedScalarFieldContainer[int]
    layout: _containers.RepeatedScalarFieldContainer[int]
    global_shape: _containers.RepeatedScalarFieldContainer[int]
    pools: _containers.RepeatedCompositeFieldContainer[PoolSpecProto]
    layout_fingerprint: str
    page_tokens: int
    transfer_parallelism: int
    transfer_rank: int
    variables: _containers.RepeatedCompositeFieldContainer[VariableMetadataProto]
    mesh_axes: _containers.RepeatedScalarFieldContainer[str]
    host_subgrid: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, unit: _Optional[_Union[RaidenIdProto, _Mapping]] = ..., shards: _Optional[_Iterable[str]] = ..., control_plane_rpc_address: _Optional[str] = ..., itemsize: _Optional[int] = ..., mesh_shape: _Optional[_Iterable[int]] = ..., layout: _Optional[_Iterable[int]] = ..., global_shape: _Optional[_Iterable[int]] = ..., pools: _Optional[_Iterable[_Union[PoolSpecProto, _Mapping]]] = ..., layout_fingerprint: _Optional[str] = ..., page_tokens: _Optional[int] = ..., transfer_parallelism: _Optional[int] = ..., transfer_rank: _Optional[int] = ..., variables: _Optional[_Iterable[_Union[VariableMetadataProto, _Mapping]]] = ..., mesh_axes: _Optional[_Iterable[str]] = ..., host_subgrid: _Optional[_Iterable[int]] = ...) -> None: ...

class ShardPushEntryProto(_message.Message):
    __slots__ = ("dst_peer", "dst_shard_idx", "dst_offset_bytes", "src_offset_bytes", "size_bytes", "src_block_id", "dst_block_id", "src_stride_bytes", "dst_stride_bytes", "count", "layer_idx", "pool_group", "dst_peers", "outer_src_strides_bytes", "outer_dst_strides_bytes", "outer_counts")
    DST_PEER_FIELD_NUMBER: _ClassVar[int]
    DST_SHARD_IDX_FIELD_NUMBER: _ClassVar[int]
    DST_OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    SRC_OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SRC_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    DST_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    SRC_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DST_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    LAYER_IDX_FIELD_NUMBER: _ClassVar[int]
    POOL_GROUP_FIELD_NUMBER: _ClassVar[int]
    DST_PEERS_FIELD_NUMBER: _ClassVar[int]
    OUTER_SRC_STRIDES_BYTES_FIELD_NUMBER: _ClassVar[int]
    OUTER_DST_STRIDES_BYTES_FIELD_NUMBER: _ClassVar[int]
    OUTER_COUNTS_FIELD_NUMBER: _ClassVar[int]
    dst_peer: str
    dst_shard_idx: int
    dst_offset_bytes: int
    src_offset_bytes: int
    size_bytes: int
    src_block_id: int
    dst_block_id: int
    src_stride_bytes: int
    dst_stride_bytes: int
    count: int
    layer_idx: int
    pool_group: int
    dst_peers: _containers.RepeatedScalarFieldContainer[str]
    outer_src_strides_bytes: _containers.RepeatedScalarFieldContainer[int]
    outer_dst_strides_bytes: _containers.RepeatedScalarFieldContainer[int]
    outer_counts: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dst_peer: _Optional[str] = ..., dst_shard_idx: _Optional[int] = ..., dst_offset_bytes: _Optional[int] = ..., src_offset_bytes: _Optional[int] = ..., size_bytes: _Optional[int] = ..., src_block_id: _Optional[int] = ..., dst_block_id: _Optional[int] = ..., src_stride_bytes: _Optional[int] = ..., dst_stride_bytes: _Optional[int] = ..., count: _Optional[int] = ..., layer_idx: _Optional[int] = ..., pool_group: _Optional[int] = ..., dst_peers: _Optional[_Iterable[str]] = ..., outer_src_strides_bytes: _Optional[_Iterable[int]] = ..., outer_dst_strides_bytes: _Optional[_Iterable[int]] = ..., outer_counts: _Optional[_Iterable[int]] = ...) -> None: ...

class ShardPushScheduleProto(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ShardPushEntryProto]
    def __init__(self, entries: _Optional[_Iterable[_Union[ShardPushEntryProto, _Mapping]]] = ...) -> None: ...

class StartTransferRequest(_message.Message):
    __slots__ = ("src_units", "dst_units", "shard_push_schedules", "uuid", "is_sender", "dst_mem_type", "use_block_chunks", "expected_block_count", "req_id", "pool_dtype_tags", "transfer_pool_indices", "parallelism", "skip_d2h", "pool_groups", "skip_tiling", "expected_layer_chunk_counts", "broadcast_round", "broadcast_round_destinations", "wire_pool_indices", "receiver_addrs", "host_tiling_mode")
    class ShardPushSchedulesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: ShardPushScheduleProto
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[ShardPushScheduleProto, _Mapping]] = ...) -> None: ...
    class SkipTilingEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: bool
        def __init__(self, key: _Optional[int] = ..., value: _Optional[bool] = ...) -> None: ...
    class ExpectedLayerChunkCountsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: int
        def __init__(self, key: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
    class WirePoolIndicesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: int
        def __init__(self, key: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
    class ReceiverAddrsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ReceiverAddrsProto
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ReceiverAddrsProto, _Mapping]] = ...) -> None: ...
    SRC_UNITS_FIELD_NUMBER: _ClassVar[int]
    DST_UNITS_FIELD_NUMBER: _ClassVar[int]
    SHARD_PUSH_SCHEDULES_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    IS_SENDER_FIELD_NUMBER: _ClassVar[int]
    DST_MEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    USE_BLOCK_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_BLOCK_COUNT_FIELD_NUMBER: _ClassVar[int]
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    POOL_DTYPE_TAGS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_POOL_INDICES_FIELD_NUMBER: _ClassVar[int]
    PARALLELISM_FIELD_NUMBER: _ClassVar[int]
    SKIP_D2H_FIELD_NUMBER: _ClassVar[int]
    POOL_GROUPS_FIELD_NUMBER: _ClassVar[int]
    SKIP_TILING_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_LAYER_CHUNK_COUNTS_FIELD_NUMBER: _ClassVar[int]
    BROADCAST_ROUND_FIELD_NUMBER: _ClassVar[int]
    BROADCAST_ROUND_DESTINATIONS_FIELD_NUMBER: _ClassVar[int]
    WIRE_POOL_INDICES_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ADDRS_FIELD_NUMBER: _ClassVar[int]
    HOST_TILING_MODE_FIELD_NUMBER: _ClassVar[int]
    src_units: _containers.RepeatedCompositeFieldContainer[RaidenIdProto]
    dst_units: _containers.RepeatedCompositeFieldContainer[RaidenIdProto]
    shard_push_schedules: _containers.MessageMap[int, ShardPushScheduleProto]
    uuid: int
    is_sender: bool
    dst_mem_type: MemoryType
    use_block_chunks: bool
    expected_block_count: int
    req_id: str
    pool_dtype_tags: _containers.RepeatedScalarFieldContainer[str]
    transfer_pool_indices: _containers.RepeatedScalarFieldContainer[int]
    parallelism: int
    skip_d2h: bool
    pool_groups: _containers.RepeatedCompositeFieldContainer[PoolGroupProto]
    skip_tiling: _containers.ScalarMap[int, bool]
    expected_layer_chunk_counts: _containers.ScalarMap[int, int]
    broadcast_round: int
    broadcast_round_destinations: _containers.RepeatedCompositeFieldContainer[BroadcastRoundDestinationsProto]
    wire_pool_indices: _containers.ScalarMap[int, int]
    receiver_addrs: _containers.MessageMap[str, ReceiverAddrsProto]
    host_tiling_mode: HostTilingMode
    def __init__(self, src_units: _Optional[_Iterable[_Union[RaidenIdProto, _Mapping]]] = ..., dst_units: _Optional[_Iterable[_Union[RaidenIdProto, _Mapping]]] = ..., shard_push_schedules: _Optional[_Mapping[int, ShardPushScheduleProto]] = ..., uuid: _Optional[int] = ..., is_sender: _Optional[bool] = ..., dst_mem_type: _Optional[_Union[MemoryType, str]] = ..., use_block_chunks: _Optional[bool] = ..., expected_block_count: _Optional[int] = ..., req_id: _Optional[str] = ..., pool_dtype_tags: _Optional[_Iterable[str]] = ..., transfer_pool_indices: _Optional[_Iterable[int]] = ..., parallelism: _Optional[int] = ..., skip_d2h: _Optional[bool] = ..., pool_groups: _Optional[_Iterable[_Union[PoolGroupProto, _Mapping]]] = ..., skip_tiling: _Optional[_Mapping[int, bool]] = ..., expected_layer_chunk_counts: _Optional[_Mapping[int, int]] = ..., broadcast_round: _Optional[int] = ..., broadcast_round_destinations: _Optional[_Iterable[_Union[BroadcastRoundDestinationsProto, _Mapping]]] = ..., wire_pool_indices: _Optional[_Mapping[int, int]] = ..., receiver_addrs: _Optional[_Mapping[str, ReceiverAddrsProto]] = ..., host_tiling_mode: _Optional[_Union[HostTilingMode, str]] = ...) -> None: ...

class BroadcastRoundDestinationsProto(_message.Message):
    __slots__ = ("round_idx", "dst_units", "dst_peers")
    ROUND_IDX_FIELD_NUMBER: _ClassVar[int]
    DST_UNITS_FIELD_NUMBER: _ClassVar[int]
    DST_PEERS_FIELD_NUMBER: _ClassVar[int]
    round_idx: int
    dst_units: _containers.RepeatedScalarFieldContainer[str]
    dst_peers: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, round_idx: _Optional[int] = ..., dst_units: _Optional[_Iterable[str]] = ..., dst_peers: _Optional[_Iterable[str]] = ...) -> None: ...

class PoolGroupProto(_message.Message):
    __slots__ = ("pool_indices", "dst_device_block_ids", "expected_pushes", "dst_expected_extent_bytes", "order_rank")
    POOL_INDICES_FIELD_NUMBER: _ClassVar[int]
    DST_DEVICE_BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_PUSHES_FIELD_NUMBER: _ClassVar[int]
    DST_EXPECTED_EXTENT_BYTES_FIELD_NUMBER: _ClassVar[int]
    ORDER_RANK_FIELD_NUMBER: _ClassVar[int]
    pool_indices: _containers.RepeatedScalarFieldContainer[int]
    dst_device_block_ids: _containers.RepeatedScalarFieldContainer[int]
    expected_pushes: int
    dst_expected_extent_bytes: _containers.RepeatedScalarFieldContainer[int]
    order_rank: int
    def __init__(self, pool_indices: _Optional[_Iterable[int]] = ..., dst_device_block_ids: _Optional[_Iterable[int]] = ..., expected_pushes: _Optional[int] = ..., dst_expected_extent_bytes: _Optional[_Iterable[int]] = ..., order_rank: _Optional[int] = ...) -> None: ...

class GetMetadataResponse(_message.Message):
    __slots__ = ("metadata",)
    METADATA_FIELD_NUMBER: _ClassVar[int]
    metadata: _containers.RepeatedCompositeFieldContainer[RegisterWorkUnitRequest]
    def __init__(self, metadata: _Optional[_Iterable[_Union[RegisterWorkUnitRequest, _Mapping]]] = ...) -> None: ...

class ControlRequest(_message.Message):
    __slots__ = ("command", "peers", "start_transfer_request", "register_work_unit_request")
    class Command(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        COMMAND_UNSPECIFIED: _ClassVar[ControlRequest.Command]
        COMMAND_START_TRANSFER: _ClassVar[ControlRequest.Command]
        COMMAND_SHUTDOWN: _ClassVar[ControlRequest.Command]
        COMMAND_REGISTER_WORK_UNIT: _ClassVar[ControlRequest.Command]
        COMMAND_GET_METADATA: _ClassVar[ControlRequest.Command]
        COMMAND_REGISTER_TRANSFER_SCHEDULE: _ClassVar[ControlRequest.Command]
    COMMAND_UNSPECIFIED: ControlRequest.Command
    COMMAND_START_TRANSFER: ControlRequest.Command
    COMMAND_SHUTDOWN: ControlRequest.Command
    COMMAND_REGISTER_WORK_UNIT: ControlRequest.Command
    COMMAND_GET_METADATA: ControlRequest.Command
    COMMAND_REGISTER_TRANSFER_SCHEDULE: ControlRequest.Command
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    PEERS_FIELD_NUMBER: _ClassVar[int]
    START_TRANSFER_REQUEST_FIELD_NUMBER: _ClassVar[int]
    REGISTER_WORK_UNIT_REQUEST_FIELD_NUMBER: _ClassVar[int]
    command: ControlRequest.Command
    peers: _containers.RepeatedScalarFieldContainer[str]
    start_transfer_request: StartTransferRequest
    register_work_unit_request: RegisterWorkUnitRequest
    def __init__(self, command: _Optional[_Union[ControlRequest.Command, str]] = ..., peers: _Optional[_Iterable[str]] = ..., start_transfer_request: _Optional[_Union[StartTransferRequest, _Mapping]] = ..., register_work_unit_request: _Optional[_Union[RegisterWorkUnitRequest, _Mapping]] = ...) -> None: ...

class PoolHostAddrsProto(_message.Message):
    __slots__ = ("host_base_addrs", "block_stride_bytes", "num_blocks", "host_slot_by_block")
    class HostSlotByBlockEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: int
        def __init__(self, key: _Optional[int] = ..., value: _Optional[int] = ...) -> None: ...
    HOST_BASE_ADDRS_FIELD_NUMBER: _ClassVar[int]
    BLOCK_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    NUM_BLOCKS_FIELD_NUMBER: _ClassVar[int]
    HOST_SLOT_BY_BLOCK_FIELD_NUMBER: _ClassVar[int]
    host_base_addrs: _containers.RepeatedScalarFieldContainer[int]
    block_stride_bytes: int
    num_blocks: int
    host_slot_by_block: _containers.ScalarMap[int, int]
    def __init__(self, host_base_addrs: _Optional[_Iterable[int]] = ..., block_stride_bytes: _Optional[int] = ..., num_blocks: _Optional[int] = ..., host_slot_by_block: _Optional[_Mapping[int, int]] = ...) -> None: ...

class ReceiverAddrsProto(_message.Message):
    __slots__ = ("pools",)
    class PoolsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: PoolHostAddrsProto
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[PoolHostAddrsProto, _Mapping]] = ...) -> None: ...
    POOLS_FIELD_NUMBER: _ClassVar[int]
    pools: _containers.MessageMap[int, PoolHostAddrsProto]
    def __init__(self, pools: _Optional[_Mapping[int, PoolHostAddrsProto]] = ...) -> None: ...

class ControlResponse(_message.Message):
    __slots__ = ("success", "message", "response_data", "get_metadata_response", "receiver_pool_addrs")
    class ReceiverPoolAddrsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: PoolHostAddrsProto
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[PoolHostAddrsProto, _Mapping]] = ...) -> None: ...
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DATA_FIELD_NUMBER: _ClassVar[int]
    GET_METADATA_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_POOL_ADDRS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    response_data: str
    get_metadata_response: GetMetadataResponse
    receiver_pool_addrs: _containers.MessageMap[int, PoolHostAddrsProto]
    def __init__(self, success: _Optional[bool] = ..., message: _Optional[str] = ..., response_data: _Optional[str] = ..., get_metadata_response: _Optional[_Union[GetMetadataResponse, _Mapping]] = ..., receiver_pool_addrs: _Optional[_Mapping[int, PoolHostAddrsProto]] = ...) -> None: ...
