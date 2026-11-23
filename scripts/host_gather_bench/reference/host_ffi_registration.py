# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Register typed Host FFI in libtpu through its public PJRT C extension."""

import ctypes as ct
import importlib.util
from functools import cache

import libtpu

# Public ABI prefixes from openxla/xla/xla/pjrt/c/{pjrt_c_api.h,
# pjrt_c_api_ffi_extension.h}. Only append-only fields used here are declared;
# runtime size/major-version checks precede access to extension functions.
_FFI_EXTENSION_TYPE = 5


class _Extension(ct.Structure):
    pass


_Extension._fields_ = [
    ("struct_size", ct.c_size_t),
    ("type", ct.c_int),
    ("next", ct.POINTER(_Extension)),
]


class _Version(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("extension_start", ct.c_void_p),
        ("major", ct.c_int),
        ("minor", ct.c_int),
    ]


class _Api(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("extension_start", ct.POINTER(_Extension)),
        ("version", _Version),
        ("error_destroy", ct.c_void_p),
        ("error_message", ct.c_void_p),
    ]


class _Ffi(ct.Structure):
    _fields_ = [
        ("base", _Extension),
        ("type_register", ct.c_void_p),
        ("user_data_add", ct.c_void_p),
        ("register_handler", ct.c_void_p),
    ]


class _RegisterArgs(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("target_name", ct.c_char_p),
        ("target_name_size", ct.c_size_t),
        ("handler", ct.c_void_p),
        ("platform_name", ct.c_char_p),
        ("platform_name_size", ct.c_size_t),
        ("traits", ct.c_uint32),
    ]


class _ErrorArgs(ct.Structure):
    _fields_ = [
        ("struct_size", ct.c_size_t),
        ("extension_start", ct.c_void_p),
        ("error", ct.c_void_p),
    ]


class _MessageArgs(ct.Structure):
    _fields_ = [
        *_ErrorArgs._fields_,
        ("message", ct.c_void_p),
        ("message_size", ct.c_size_t),
    ]


def register_host_target(plugin: ct.CDLL, target: bytes, handler: ct.c_void_p) -> None:
    """Register with the plugin that compiles device_host subcomputations."""
    plugin.GetPjrtApi.restype = ct.POINTER(_Api)
    api = plugin.GetPjrtApi().contents
    if api.struct_size < ct.sizeof(_Api) or api.version.major != 0:
        raise RuntimeError("Host FFI requires the PJRT C API major version 0")
    extension = api.extension_start
    while extension and extension.contents.type != _FFI_EXTENSION_TYPE:
        extension = extension.contents.next
    if not extension or extension.contents.struct_size < ct.sizeof(_Ffi):
        raise RuntimeError("libtpu must provide PJRT_FFI.register_handler")
    ffi = ct.cast(extension, ct.POINTER(_Ffi)).contents
    if not ffi.register_handler:
        raise RuntimeError("libtpu's PJRT FFI register_handler is unavailable")
    register = ct.CFUNCTYPE(ct.c_void_p, ct.POINTER(_RegisterArgs))(
        ffi.register_handler
    )
    args = _RegisterArgs(
        _RegisterArgs.traits.offset + ct.sizeof(ct.c_uint32),
        target,
        len(target),
        handler,
        b"Host",
        4,
        0,
    )
    error = register(ct.byref(args))
    if error:
        message = _MessageArgs(ct.sizeof(_MessageArgs), None, error, None, 0)
        ct.CFUNCTYPE(None, ct.POINTER(_MessageArgs))(api.error_message)(
            ct.byref(message)
        )
        text = ct.string_at(message.message, message.message_size).decode()
        destroy = _ErrorArgs(ct.sizeof(_ErrorArgs), None, error)
        ct.CFUNCTYPE(None, ct.POINTER(_ErrorArgs))(api.error_destroy)(ct.byref(destroy))
        raise RuntimeError(f"Host FFI registration failed: {text}")


@cache
def _plugin() -> ct.CDLL:
    return ct.CDLL(libtpu.get_library_path())


@cache
def native_library() -> ct.CDLL:
    """Load the packaged extension once; there is no compilation during loading."""
    spec = importlib.util.find_spec("vllm_torchtpu._host_gather_ffi")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install vllm-torchtpu with its Host FFI extension first")
    native = ct.CDLL(spec.origin)
    plugin = _plugin()
    register_host_target(
        plugin,
        b"engram_host_gather_256",
        ct.cast(native.HostGather256, ct.c_void_p),
    )
    native.BindHostGatherTable.argtypes = [ct.c_void_p, ct.c_int64]
    native.BindHostGatherTable.restype = ct.c_int32
    native.ReleaseHostGatherTable.argtypes = [ct.c_int32]
    native.ReleaseHostGatherTable.restype = ct.c_int
    return native
