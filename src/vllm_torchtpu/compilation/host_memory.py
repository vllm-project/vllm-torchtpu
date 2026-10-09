# SPDX-License-Identifier: Apache-2.0
"""Return idle glibc heap pages after a compiler call releases its locals."""

import ctypes
import functools
import gc
import sys
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


@functools.cache
def _glibc_allocator() -> ctypes.CDLL:
    message = (
        "TPU_RECLAIM_COMPILE_MEMORY requires Linux with glibc malloc; "
        "disable it when using another host allocator."
    )
    if sys.platform != "linux":
        raise RuntimeError(message)
    libc = ctypes.CDLL(None)
    try:
        malloc = libc.malloc
        trim = libc.malloc_trim
        version = libc.gnu_get_libc_version
        dladdr = libc.dladdr
    except AttributeError as error:
        raise RuntimeError(message) from error
    version.restype = ctypes.c_char_p
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    dladdr.restype = ctypes.c_int
    allocator_info, trim_info, version_info = _DlInfo(), _DlInfo(), _DlInfo()
    if (
        not dladdr(ctypes.cast(malloc, ctypes.c_void_p), ctypes.byref(allocator_info))
        or not dladdr(ctypes.cast(trim, ctypes.c_void_p), ctypes.byref(trim_info))
        or not dladdr(ctypes.cast(version, ctypes.c_void_p), ctypes.byref(version_info))
        or not allocator_info.dli_fbase
        or allocator_info.dli_fbase != trim_info.dli_fbase
        or allocator_info.dli_fbase != version_info.dli_fbase
        or not version()
    ):
        raise RuntimeError(message)
    return libc


def reclaim_after_compile(
    enabled: Callable[[], bool],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Collect dead compiler cycles, then trim only after its frame is gone."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            allocator = _glibc_allocator() if enabled() else None
            result = function(*args, **kwargs)
            if allocator is not None:
                gc.collect()
                allocator.malloc_trim(0)
            return result

        return wrapped

    return decorate
