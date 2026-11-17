# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: shared-storage views (`pool_alias.view`) in TPU interpret mode.

The jax 0.11.2 TPU interpreter resolves a ref access only through NDIndexer transforms
(`interpret/utils.py:to_range`), so a `ref.bitcast(dtype)` + `Reinterpret(shape)` view fails there, and the kernel
used to fall back to separate typed buffers in interpret mode: the cross-kind dense-bank sharing (`dpool`), the
expert-slot views and the DSA-on-expert-ring alias (`epool`) never executed on CPU.

`install()` wraps the interpreter's `get` / `store` / `swap` callbacks: an access whose transform chain contains a
bitcast or reshape is resolved as a numpy VIEW of the underlying buffer -- the leading NDIndexers select the pool's
row window, the window's bytes are reinterpreted row-major as the view's dtype and shape, and any later NDIndexers
index the view -- so reads and writes go to the pool's own bytes. Each view therefore occupies EXACTLY the byte range
it occupies on the Mosaic path (pool_alias asserts storage_size == raw bytes and the windows are row ranges), and
every overlap between views, intended or not, is executed. What it does not reproduce: the order of elements inside
a view (Mosaic's (8*packing, 128) tiles vs row-major here), so a value written through one view and read through a
DIFFERENTLY SHAPED view of the same bytes reads different elements here than on hardware. The kernel never does that
on purpose (every view is overwritten before it is read); the hardware layout itself is P4's job (A0 TPU job).

Race detection and the interpreter's out-of-bounds / uninitialised-read checks are skipped for such accesses (the
plain NDIndexer path is unchanged). Pinned to jax 0.11.2's interpreter internals.
"""

import numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax._src.pallas.mosaic.interpret import utils as iutils
from jax._src.state import indexing
from jax._src.state import types as stypes

_INSTALLED = {}


def _is_view(transforms):
    return any(
        isinstance(t, (stypes.BitcastTransform, stypes.ReshapeTransform))
        for t in transforms
    )


def _resolve(content, transforms):
    """numpy view of `content` selected by `transforms` (NDIndexer / Bitcast / Reshape / Transpose)."""
    arr = content
    i = 0
    while i < len(transforms):
        t = transforms[i]
        if isinstance(t, stypes.BitcastTransform):
            assert arr.flags.c_contiguous, (
                "bitcast of a non-contiguous window",
                arr.shape,
                arr.strides,
            )
            dt = np.dtype(t.dtype)
            flat = arr.reshape(-1).view(dt)
            # Mosaic semantics: the second-minor dim scales with the itemsize ratio (pool_alias always follows a
            # bitcast with a Reinterpret, which fixes the final shape; this intermediate shape is only a placeholder)
            arr = (
                flat.reshape(-1, arr.shape[-1])
                if arr.ndim >= 2 and flat.size % arr.shape[-1] == 0
                else flat
            )
        elif isinstance(t, stypes.ReshapeTransform):
            assert arr.flags.c_contiguous, (
                "reshape of a non-contiguous window",
                arr.shape,
                arr.strides,
            )
            shp = tuple(t.shape)
            # pool_alias views never pad: ReshapeTransform validates prod(shape) == window elements at trace time
            assert arr.size == int(np.prod(shp)), ("window != view", arr.shape, shp)
            arr = arr.reshape(shp)
        elif isinstance(t, stypes.TransposeTransform):
            arr = arr.transpose(tuple(int(p) for p in t.permutation))
        elif isinstance(t, indexing.NDIndexer):
            rng = iutils.to_range((t,))
            arr = arr[rng]
        else:
            raise NotImplementedError(type(t))
        i += 1
    assert np.shares_memory(arr, content) or arr.size == 0, "view resolved to a copy"
    return arr


def _buffer(shared_memory, device_id, local_core_id, memory_space, buffer_id):
    memory_space = ipc.TPU_MEMORY_SPACE_NAMES[int(memory_space)]
    lc = ipc._local_core_id_or_zero_if_hbm(int(local_core_id), memory_space)
    return shared_memory.mem[(memory_space, int(buffer_id), int(device_id), lc)]


def install():
    """Idempotent. Patches the interpreter's get / store / swap (module globals, resolved at interpret time)."""
    if _INSTALLED:
        return
    o_get, o_store, o_swap = ipc.get, ipc.store, ipc.swap
    r_get, r_store, r_swap = o_get.__wrapped__, o_store.__wrapped__, o_swap.__wrapped__

    def get_(
        token, device_id, local_core_id, memory_space, buffer_id, transforms, *a, **kw
    ):
        transforms = tuple(transforms)
        if not _is_view(transforms):
            return r_get(
                token,
                device_id,
                local_core_id,
                memory_space,
                buffer_id,
                transforms,
                *a,
                **kw,
            )
        sm = ipc._get_shared_memory()
        with sm.lock:
            v = _resolve(
                _buffer(sm, device_id, local_core_id, memory_space, buffer_id).content,
                _ints(transforms),
            ).copy()
        return token, v

    def store_(
        token,
        device_id,
        local_core_id,
        memory_space,
        buffer_id,
        transforms,
        val,
        *a,
        **kw,
    ):
        transforms = tuple(transforms)
        if not _is_view(transforms):
            return r_store(
                token,
                device_id,
                local_core_id,
                memory_space,
                buffer_id,
                transforms,
                val,
                *a,
                **kw,
            )
        sm = ipc._get_shared_memory()
        val = np.asarray(val)
        with sm.lock:
            v = _resolve(
                _buffer(sm, device_id, local_core_id, memory_space, buffer_id).content,
                _ints(transforms),
            )
            assert v.shape == val.shape and v.dtype == val.dtype, (
                v.shape,
                v.dtype,
                val.shape,
                val.dtype,
            )
            v[...] = val
        return token

    def swap_(
        token,
        device_id,
        local_core_id,
        memory_space,
        buffer_id,
        transforms,
        val,
        mask,
        *a,
        **kw,
    ):
        transforms = tuple(transforms)
        if not _is_view(transforms):
            return r_swap(
                token,
                device_id,
                local_core_id,
                memory_space,
                buffer_id,
                transforms,
                val,
                mask,
                *a,
                **kw,
            )
        sm = ipc._get_shared_memory()
        val = np.asarray(val)
        with sm.lock:
            v = _resolve(
                _buffer(sm, device_id, local_core_id, memory_space, buffer_id).content,
                _ints(transforms),
            )
            assert v.shape == val.shape and v.dtype == val.dtype, (
                v.shape,
                v.dtype,
                val.shape,
                val.dtype,
            )
            old = v.copy()
            if mask is None:
                v[...] = val
            else:
                m = np.asarray(mask)
                v[...] = np.where(m, val, old)
        return token, old

    for nm, raw in (("get", get_), ("store", store_), ("swap", swap_)):
        w = ipc.fail_on_exception(raw)
        _INSTALLED[nm] = getattr(ipc, nm)
        setattr(ipc, nm, w)


def _ints(transforms):
    import jax

    return tuple(jax.tree.map(int, transforms))
