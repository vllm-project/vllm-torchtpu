from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MetadataRequest(_message.Message):
    __slots__ = ("client_id",)
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: _Optional[str] = ...) -> None: ...

class RaidenTransferEndpoint(_message.Message):
    __slots__ = ("endpoint", "shards")
    ENDPOINT_FIELD_NUMBER: _ClassVar[int]
    SHARDS_FIELD_NUMBER: _ClassVar[int]
    endpoint: str
    shards: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, endpoint: _Optional[str] = ..., shards: _Optional[_Iterable[int]] = ...) -> None: ...

class MetadataResponse(_message.Message):
    __slots__ = ("endpoints", "transfer_uuid", "transfer_req_id", "block_ids")
    ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_UUID_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_REQ_ID_FIELD_NUMBER: _ClassVar[int]
    BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    endpoints: _containers.RepeatedCompositeFieldContainer[RaidenTransferEndpoint]
    transfer_uuid: int
    transfer_req_id: str
    block_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, endpoints: _Optional[_Iterable[_Union[RaidenTransferEndpoint, _Mapping]]] = ..., transfer_uuid: _Optional[int] = ..., transfer_req_id: _Optional[str] = ..., block_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class ShutdownRequest(_message.Message):
    __slots__ = ("client_id",)
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    def __init__(self, client_id: _Optional[str] = ...) -> None: ...

class ShutdownResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: _Optional[bool] = ...) -> None: ...

class CollectReplicaInfoRequest(_message.Message):
    __slots__ = ("device_id", "expected_count", "info")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    device_id: int
    expected_count: int
    info: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, device_id: _Optional[int] = ..., expected_count: _Optional[int] = ..., info: _Optional[_Iterable[int]] = ...) -> None: ...

class CollectReplicaInfoResponse(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[ReplicaInfoEntry]
    def __init__(self, entries: _Optional[_Iterable[_Union[ReplicaInfoEntry, _Mapping]]] = ...) -> None: ...

class ReplicaInfoEntry(_message.Message):
    __slots__ = ("device_id", "info")
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    INFO_FIELD_NUMBER: _ClassVar[int]
    device_id: int
    info: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, device_id: _Optional[int] = ..., info: _Optional[_Iterable[int]] = ...) -> None: ...
