# SPDX-License-Identifier: Apache-2.0
"""Deviceless duck-typed tensor fakes for the Raiden pool-manifest tests.

Moved out of tpu_connector_v2_test_utils.py so that collecting the live
Raiden tests no longer imports the TPUConnectorV2 cluster at module scope.
FakeStorage and FakeTensor move as a pair: FakeTensor's default constructor
builds a FakeStorage, and FakeStorage._next_ptr is mutable class state that
must exist exactly once (two copies would hand out colliding data_ptr()
values and silently weaken the dead-storage check).
"""


class FakeStorage:
    """Duck-typed untyped_storage() stand-in with a unique data pointer."""

    _next_ptr = 0x1000

    def __init__(self):
        FakeStorage._next_ptr += 0x100000000
        self._ptr = FakeStorage._next_ptr

    def data_ptr(self):
        return self._ptr


class FakeTensor:
    """Duck-typed tensor for pool-manifest tests (no torch required)."""

    def __init__(self,
                 shape,
                 element_size,
                 dtype="torch.fake",
                 storage=None,
                 storage_offset_elems=0):
        self.shape = tuple(shape)
        self._element_size = int(element_size)
        self.dtype = dtype
        self._storage = storage if storage is not None else FakeStorage()
        self._storage_offset = int(storage_offset_elems)
        numel = 1
        for dim in self.shape:
            numel *= dim
        self.nbytes = numel * self._element_size

    def element_size(self):
        return self._element_size

    def untyped_storage(self):
        return self._storage

    def storage_offset(self):
        return self._storage_offset
