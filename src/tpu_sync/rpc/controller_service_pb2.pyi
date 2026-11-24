from tpu_sync.rpc import raiden_service_pb2 as _raiden_service_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CoordinateTransferRequest(_message.Message):
    __slots__ = ("src_units", "dst_units", "uuid", "is_sender", "dst_mem_type", "use_block_chunks", "expected_block_count", "req_id", "src_controller_address", "dst_controller_address", "dst_device_block_ids", "transfer_pool_tags", "dst_block_counts", "dst_skip_bytes")
    SRC_UNITS_FIELD_NUMBER: _ClassVar[int]
    DST_UNITS_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    IS_SENDER_FIELD_NUMBER: _ClassVar[int]
    DST_MEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    USE_BLOCK_CHUNKS_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_BLOCK_COUNT_FIELD_NUMBER: _ClassVar[int]
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    SRC_CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DST_CONTROLLER_ADDRESS_FIELD_NUMBER: _ClassVar[int]
    DST_DEVICE_BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_POOL_TAGS_FIELD_NUMBER: _ClassVar[int]
    DST_BLOCK_COUNTS_FIELD_NUMBER: _ClassVar[int]
    DST_SKIP_BYTES_FIELD_NUMBER: _ClassVar[int]
    src_units: _containers.RepeatedCompositeFieldContainer[_raiden_service_pb2.RaidenIdProto]
    dst_units: _containers.RepeatedCompositeFieldContainer[_raiden_service_pb2.RaidenIdProto]
    uuid: int
    is_sender: bool
    dst_mem_type: _raiden_service_pb2.MemoryType
    use_block_chunks: bool
    expected_block_count: int
    req_id: str
    src_controller_address: str
    dst_controller_address: str
    dst_device_block_ids: _containers.RepeatedScalarFieldContainer[int]
    transfer_pool_tags: _containers.RepeatedScalarFieldContainer[str]
    dst_block_counts: _containers.RepeatedScalarFieldContainer[int]
    dst_skip_bytes: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, src_units: _Optional[_Iterable[_Union[_raiden_service_pb2.RaidenIdProto, _Mapping]]] = ..., dst_units: _Optional[_Iterable[_Union[_raiden_service_pb2.RaidenIdProto, _Mapping]]] = ..., uuid: _Optional[int] = ..., is_sender: _Optional[bool] = ..., dst_mem_type: _Optional[_Union[_raiden_service_pb2.MemoryType, str]] = ..., use_block_chunks: _Optional[bool] = ..., expected_block_count: _Optional[int] = ..., req_id: _Optional[str] = ..., src_controller_address: _Optional[str] = ..., dst_controller_address: _Optional[str] = ..., dst_device_block_ids: _Optional[_Iterable[int]] = ..., transfer_pool_tags: _Optional[_Iterable[str]] = ..., dst_block_counts: _Optional[_Iterable[int]] = ..., dst_skip_bytes: _Optional[_Iterable[int]] = ...) -> None: ...

class PoolByteSpan(_message.Message):
    __slots__ = ("src_block_ordinal", "src_offset_bytes", "dst_block_index", "dst_offset_bytes", "size_bytes", "src_stride_bytes", "dst_stride_bytes", "count", "dst_unit_ordinal")
    SRC_BLOCK_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    SRC_OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    DST_BLOCK_INDEX_FIELD_NUMBER: _ClassVar[int]
    DST_OFFSET_BYTES_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SRC_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DST_STRIDE_BYTES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    DST_UNIT_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    src_block_ordinal: int
    src_offset_bytes: int
    dst_block_index: int
    dst_offset_bytes: int
    size_bytes: int
    src_stride_bytes: int
    dst_stride_bytes: int
    count: int
    dst_unit_ordinal: int
    def __init__(self, src_block_ordinal: _Optional[int] = ..., src_offset_bytes: _Optional[int] = ..., dst_block_index: _Optional[int] = ..., dst_offset_bytes: _Optional[int] = ..., size_bytes: _Optional[int] = ..., src_stride_bytes: _Optional[int] = ..., dst_stride_bytes: _Optional[int] = ..., count: _Optional[int] = ..., dst_unit_ordinal: _Optional[int] = ...) -> None: ...

class PoolSpanRegistration(_message.Message):
    __slots__ = ("tag", "block_ids", "spans", "declared_bytes", "dst_space_version")
    TAG_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    SPANS_FIELD_NUMBER: _ClassVar[int]
    DECLARED_BYTES_FIELD_NUMBER: _ClassVar[int]
    DST_SPACE_VERSION_FIELD_NUMBER: _ClassVar[int]
    tag: str
    block_ids: _containers.RepeatedScalarFieldContainer[int]
    spans: _containers.RepeatedCompositeFieldContainer[PoolByteSpan]
    declared_bytes: int
    dst_space_version: int
    def __init__(self, tag: _Optional[str] = ..., block_ids: _Optional[_Iterable[int]] = ..., spans: _Optional[_Iterable[_Union[PoolByteSpan, _Mapping]]] = ..., declared_bytes: _Optional[int] = ..., dst_space_version: _Optional[int] = ...) -> None: ...

class RegisterRequestBlocksRequest(_message.Message):
    __slots__ = ("req_id", "uuid", "unit", "block_ids", "pool_spans")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    POOL_SPANS_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    unit: _raiden_service_pb2.RaidenIdProto
    block_ids: _containers.RepeatedScalarFieldContainer[int]
    pool_spans: _containers.RepeatedCompositeFieldContainer[PoolSpanRegistration]
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ..., unit: _Optional[_Union[_raiden_service_pb2.RaidenIdProto, _Mapping]] = ..., block_ids: _Optional[_Iterable[int]] = ..., pool_spans: _Optional[_Iterable[_Union[PoolSpanRegistration, _Mapping]]] = ...) -> None: ...

class ReleaseRequestBlocksRequest(_message.Message):
    __slots__ = ("req_id", "uuid", "force")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    force: bool
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ..., force: _Optional[bool] = ...) -> None: ...

class CompleteRequestBlocksRequest(_message.Message):
    __slots__ = ("req_id", "uuid", "unit")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    unit: _raiden_service_pb2.RaidenIdProto
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ..., unit: _Optional[_Union[_raiden_service_pb2.RaidenIdProto, _Mapping]] = ...) -> None: ...

class CancelRequestBlocksIfUnclaimedRequest(_message.Message):
    __slots__ = ("req_id", "uuid")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ...) -> None: ...

class GetTransferStatusRequest(_message.Message):
    __slots__ = ("req_id", "uuid")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ...) -> None: ...

class RequestBlockKey(_message.Message):
    __slots__ = ("req_id", "uuid")
    REQ_ID_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    req_id: str
    uuid: int
    def __init__(self, req_id: _Optional[str] = ..., uuid: _Optional[int] = ...) -> None: ...

class GetRequestBlockStatusRequest(_message.Message):
    __slots__ = ("keys",)
    KEYS_FIELD_NUMBER: _ClassVar[int]
    keys: _containers.RepeatedCompositeFieldContainer[RequestBlockKey]
    def __init__(self, keys: _Optional[_Iterable[_Union[RequestBlockKey, _Mapping]]] = ...) -> None: ...

class GetRequestBlockStatusResponse(_message.Message):
    __slots__ = ("statuses",)
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[GetRequestBlockStatusResponse.Status]
        STATUS_UNKNOWN: _ClassVar[GetRequestBlockStatusResponse.Status]
        STATUS_REGISTERED: _ClassVar[GetRequestBlockStatusResponse.Status]
        STATUS_CLAIMED: _ClassVar[GetRequestBlockStatusResponse.Status]
        STATUS_CANCELLED: _ClassVar[GetRequestBlockStatusResponse.Status]
    STATUS_UNSPECIFIED: GetRequestBlockStatusResponse.Status
    STATUS_UNKNOWN: GetRequestBlockStatusResponse.Status
    STATUS_REGISTERED: GetRequestBlockStatusResponse.Status
    STATUS_CLAIMED: GetRequestBlockStatusResponse.Status
    STATUS_CANCELLED: GetRequestBlockStatusResponse.Status
    STATUSES_FIELD_NUMBER: _ClassVar[int]
    statuses: _containers.RepeatedScalarFieldContainer[GetRequestBlockStatusResponse.Status]
    def __init__(self, statuses: _Optional[_Iterable[_Union[GetRequestBlockStatusResponse.Status, str]]] = ...) -> None: ...

class GetTransferStatusResponse(_message.Message):
    __slots__ = ("status",)
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STATUS_UNSPECIFIED: _ClassVar[GetTransferStatusResponse.Status]
        STATUS_NOT_STARTED: _ClassVar[GetTransferStatusResponse.Status]
        STATUS_IN_PROGRESS: _ClassVar[GetTransferStatusResponse.Status]
        STATUS_COMPLETED: _ClassVar[GetTransferStatusResponse.Status]
        STATUS_FAILED: _ClassVar[GetTransferStatusResponse.Status]
    STATUS_UNSPECIFIED: GetTransferStatusResponse.Status
    STATUS_NOT_STARTED: GetTransferStatusResponse.Status
    STATUS_IN_PROGRESS: GetTransferStatusResponse.Status
    STATUS_COMPLETED: GetTransferStatusResponse.Status
    STATUS_FAILED: GetTransferStatusResponse.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: GetTransferStatusResponse.Status
    def __init__(self, status: _Optional[_Union[GetTransferStatusResponse.Status, str]] = ...) -> None: ...

class ControllerRequest(_message.Message):
    __slots__ = ("command", "coordinate_transfer_request", "get_transfer_status_request", "register_request_blocks_request", "release_request_blocks_request", "complete_request_blocks_request", "cancel_request_blocks_if_unclaimed_request", "get_request_block_status_request")
    class Command(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        COMMAND_UNSPECIFIED: _ClassVar[ControllerRequest.Command]
        COMMAND_COORDINATE_TRANSFER: _ClassVar[ControllerRequest.Command]
        COMMAND_GET_TRANSFER_STATUS: _ClassVar[ControllerRequest.Command]
        COMMAND_REGISTER_REQUEST_BLOCKS: _ClassVar[ControllerRequest.Command]
        COMMAND_RELEASE_REQUEST_BLOCKS: _ClassVar[ControllerRequest.Command]
        COMMAND_COMPLETE_REQUEST_BLOCKS: _ClassVar[ControllerRequest.Command]
        COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED: _ClassVar[ControllerRequest.Command]
        COMMAND_GET_REQUEST_BLOCK_STATUS: _ClassVar[ControllerRequest.Command]
    COMMAND_UNSPECIFIED: ControllerRequest.Command
    COMMAND_COORDINATE_TRANSFER: ControllerRequest.Command
    COMMAND_GET_TRANSFER_STATUS: ControllerRequest.Command
    COMMAND_REGISTER_REQUEST_BLOCKS: ControllerRequest.Command
    COMMAND_RELEASE_REQUEST_BLOCKS: ControllerRequest.Command
    COMMAND_COMPLETE_REQUEST_BLOCKS: ControllerRequest.Command
    COMMAND_CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED: ControllerRequest.Command
    COMMAND_GET_REQUEST_BLOCK_STATUS: ControllerRequest.Command
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    COORDINATE_TRANSFER_REQUEST_FIELD_NUMBER: _ClassVar[int]
    GET_TRANSFER_STATUS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    REGISTER_REQUEST_BLOCKS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    RELEASE_REQUEST_BLOCKS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    COMPLETE_REQUEST_BLOCKS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    CANCEL_REQUEST_BLOCKS_IF_UNCLAIMED_REQUEST_FIELD_NUMBER: _ClassVar[int]
    GET_REQUEST_BLOCK_STATUS_REQUEST_FIELD_NUMBER: _ClassVar[int]
    command: ControllerRequest.Command
    coordinate_transfer_request: CoordinateTransferRequest
    get_transfer_status_request: GetTransferStatusRequest
    register_request_blocks_request: RegisterRequestBlocksRequest
    release_request_blocks_request: ReleaseRequestBlocksRequest
    complete_request_blocks_request: CompleteRequestBlocksRequest
    cancel_request_blocks_if_unclaimed_request: CancelRequestBlocksIfUnclaimedRequest
    get_request_block_status_request: GetRequestBlockStatusRequest
    def __init__(self, command: _Optional[_Union[ControllerRequest.Command, str]] = ..., coordinate_transfer_request: _Optional[_Union[CoordinateTransferRequest, _Mapping]] = ..., get_transfer_status_request: _Optional[_Union[GetTransferStatusRequest, _Mapping]] = ..., register_request_blocks_request: _Optional[_Union[RegisterRequestBlocksRequest, _Mapping]] = ..., release_request_blocks_request: _Optional[_Union[ReleaseRequestBlocksRequest, _Mapping]] = ..., complete_request_blocks_request: _Optional[_Union[CompleteRequestBlocksRequest, _Mapping]] = ..., cancel_request_blocks_if_unclaimed_request: _Optional[_Union[CancelRequestBlocksIfUnclaimedRequest, _Mapping]] = ..., get_request_block_status_request: _Optional[_Union[GetRequestBlockStatusRequest, _Mapping]] = ...) -> None: ...

class ControllerResponse(_message.Message):
    __slots__ = ("success", "message", "get_transfer_status_response", "response_data", "get_request_block_status_response")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    GET_TRANSFER_STATUS_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DATA_FIELD_NUMBER: _ClassVar[int]
    GET_REQUEST_BLOCK_STATUS_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    get_transfer_status_response: GetTransferStatusResponse
    response_data: str
    get_request_block_status_response: GetRequestBlockStatusResponse
    def __init__(self, success: _Optional[bool] = ..., message: _Optional[str] = ..., get_transfer_status_response: _Optional[_Union[GetTransferStatusResponse, _Mapping]] = ..., response_data: _Optional[str] = ..., get_request_block_status_response: _Optional[_Union[GetRequestBlockStatusResponse, _Mapping]] = ...) -> None: ...
