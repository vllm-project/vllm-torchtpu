from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ControlEnvelope(_message.Message):
    __slots__ = ("message_type", "request_id", "payload", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    message_type: str
    request_id: int
    payload: bytes
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, message_type: _Optional[str] = ..., request_id: _Optional[int] = ..., payload: _Optional[bytes] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ControlResponseEnvelope(_message.Message):
    __slots__ = ("request_id", "status_code", "error_message", "payload")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    request_id: int
    status_code: int
    error_message: str
    payload: bytes
    def __init__(self, request_id: _Optional[int] = ..., status_code: _Optional[int] = ..., error_message: _Optional[str] = ..., payload: _Optional[bytes] = ...) -> None: ...
