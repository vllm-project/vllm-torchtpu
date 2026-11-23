// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {
constexpr int64_t kRowBytes = 256;
constexpr int64_t kRowWords = kRowBytes / sizeof(int32_t);

struct Table {
  const uint8_t *data;
  int64_t rows;
};

std::mutex tables_mutex;
std::vector<Table> tables;

// Source weights are immutable. The adapter retains them until all executions
// using this handle have completed; handles are never reused by another table.
ffi::Error Gather(ffi::BufferR1<ffi::S32> request,
                  ffi::BufferR1<ffi::S32> staging,
                  ffi::Result<ffi::BufferR1<ffi::S32>> output) {
  if (request.element_count() < 3) {
    return ffi::Error::InvalidArgument("Invalid Host gather request header");
  }
  const int64_t count = request.typed_data()[1];
  if (count <= 0 || count > request.element_count() - 2 ||
      count > staging.element_count() / kRowWords ||
      output->element_count() != staging.element_count()) {
    return ffi::Error::InvalidArgument(
        "Invalid Host gather request or capacity");
  }
  if (staging.typed_data() != output->typed_data()) {
    return ffi::Error::InvalidArgument(
        "Host gather requires an aliased staging buffer");
  }
  Table table;
  {
    std::lock_guard<std::mutex> lock(tables_mutex);
    const int32_t handle = request.typed_data()[0];
    if (handle < 0 || static_cast<size_t>(handle) >= tables.size() ||
        !tables[handle].data) {
      return ffi::Error::InvalidArgument("Host gather table is not registered");
    }
    table = tables[handle];
  }
  const int32_t *ids = request.typed_data() + 2;
  for (int64_t i = 0; i < count; ++i) {
    if (ids[i] < 0 || ids[i] >= table.rows) {
      return ffi::Error::InvalidArgument("Host gather index is out of bounds");
    }
  }
  auto *destination = reinterpret_cast<uint8_t *>(output->typed_data());
  for (int64_t i = 0; i < count; ++i) {
    if (i + 16 < count) {
      const uint8_t *next =
          table.data + static_cast<int64_t>(ids[i + 16]) * kRowBytes;
      for (int64_t offset = 0; offset < kRowBytes; offset += 64)
        __builtin_prefetch(next + offset, 0, 3);
    }
    std::memcpy(destination + i * kRowBytes,
                table.data + static_cast<int64_t>(ids[i]) * kRowBytes,
                kRowBytes);
  }
  return ffi::Error::Success();
}
} // namespace

extern "C" int32_t BindHostGatherTable(const void *data, int64_t rows) {
  if (!data || rows <= 0 || rows > std::numeric_limits<int32_t>::max())
    return -1;
  std::lock_guard<std::mutex> lock(tables_mutex);
  if (tables.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return -1;
  const int32_t handle = tables.size();
  tables.push_back({static_cast<const uint8_t *>(data), rows});
  return handle;
}

extern "C" int ReleaseHostGatherTable(int32_t handle) {
  std::lock_guard<std::mutex> lock(tables_mutex);
  if (handle < 0 || static_cast<size_t>(handle) >= tables.size() ||
      !tables[handle].data)
    return 1;
  tables[handle] = {nullptr, 0};
  return 0;
}

// The alias contract follows JAX examples/ffi cpu_examples.cc. This handler
// writes a donated output and performs no application-side synchronization.
XLA_FFI_DEFINE_HANDLER_SYMBOL(HostGather256, Gather,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Ret<ffi::BufferR1<ffi::S32>>());
