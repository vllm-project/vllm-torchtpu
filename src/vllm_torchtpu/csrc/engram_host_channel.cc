// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Persistent Host gather worker for one Engram layer on one rank.
//
// The device writes `ids = [count, dependency, rows...]`, then a positive
// generation to `request`. The worker gathers `count` rows, each the row's FP8
// bytes followed by its scale bytes, into the pinned `output`, copies them to
// the retained HBM `destination` and only then stores the generation to
// `response` (its negation if the request failed, kStopped once the worker
// exits). A channel with a predecessor starts a request only after the
// predecessor answered `dependency`.
//
// Built against the exact XLA commit of the installed TorchTPU: the tensor
// buffer header is an unsupported private interface.

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "csrc/api/tensor_buffer.h"
#include "torch/custom_class.h"
#include "torch/library.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"

namespace {

constexpr int64_t kPageBytes = 4096;
constexpr int32_t kStopped = INT32_MIN;

torch_tpu::TensorBufferHandle Resolve(const at::Tensor &tensor,
                                      const char *kind) {
  TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1 &&
                  tensor.dim() == 1 && tensor.is_contiguous() &&
                  tensor.storage_offset() == 0 &&
                  tensor.scalar_type() == at::kInt,
              "Engram channel buffers must be whole 1-D int32 TPU tensors");
  auto handle = torch_tpu::GetBaseTensorBuffer(tensor);
  TORCH_CHECK(handle.ok(), handle.status().message());
  TORCH_CHECK(handle->size_bytes() == static_cast<int64_t>(tensor.nbytes()),
              "Engram channel tensor must span its whole allocation");
  auto status = handle->Materialize();
  if (status.ok())
    status = handle->Synchronize();
  TORCH_CHECK(status.ok(), status.message());
  auto buffer = handle->AwaitBuffer();
  TORCH_CHECK(buffer.ok(), buffer.status().message());
  TORCH_CHECK((*buffer)->memory_space() != nullptr &&
                  (*buffer)->memory_space()->kind() == kind,
              "Engram channel buffer is not in ", kind, " memory");
  return std::move(handle).value();
}

// Pins a pinned-Host allocation, and its address, for the channel's lifetime.
struct HostLease {
  explicit HostLease(const at::Tensor &tensor)
      : handle(Resolve(tensor, "pinned_host")) {
    auto reference = (*handle.AwaitBuffer())->AcquireExternalReference();
    TORCH_CHECK(reference.ok(), reference.status().message());
    owner = std::move(reference).value();
    data = static_cast<int32_t *>(owner->OpaqueDeviceMemoryDataPointer());
    bytes = static_cast<int64_t>(tensor.nbytes());
    TORCH_CHECK(data && reinterpret_cast<uintptr_t>(data) % kPageBytes == 0 &&
                    bytes % kPageBytes == 0,
                "Engram Host buffers must be whole pages");
  }

  // A distinct PJRT buffer at the same address. Graphs donate the view: a
  // buffer holding an external reference must not be donated.
  at::Tensor View() {
    auto *source = *handle.AwaitBuffer();
    auto *client = source->client();
    auto view = client->CreateViewOfDeviceBuffer(
        data, source->on_device_shape(), source->memory_space(), [] {});
    TORCH_CHECK(view.ok(), view.status().message());
    auto ready = (*view)->GetReadyFuture().Await();
    TORCH_CHECK(ready.ok(), ready.message());
    auto address = client->UnsafeBufferPointer(view->get());
    TORCH_CHECK(address.ok() && *address == reinterpret_cast<uintptr_t>(data),
                "Engram Host view does not alias its allocation");
    auto made = torch_tpu::MakeTensorBuffer(std::move(view).value());
    TORCH_CHECK(made.ok(), made.status().message());
    view_handle = std::make_unique<torch_tpu::TensorBufferHandle>(
        std::move(made).value());
    auto tensor = view_handle->ToTensor();
    TORCH_CHECK(tensor.ok(), tensor.status().message());
    return std::move(tensor).value();
  }

  torch_tpu::TensorBufferHandle handle;
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> owner;
  std::unique_ptr<torch_tpu::TensorBufferHandle> view_handle;
  int32_t *data;
  int64_t bytes;
};

class HostChannel : public torch::CustomClassHolder {
public:
  HostChannel(at::Tensor weight, at::Tensor scales, at::Tensor ids,
              at::Tensor request, at::Tensor response, at::Tensor output,
              at::Tensor destination,
              std::optional<c10::intrusive_ptr<HostChannel>> predecessor)
      : weight_(std::move(weight)), scales_(std::move(scales)), ids_(ids),
        request_(request), response_(response), output_(output),
        destination_(Resolve(destination, "device")),
        predecessor_(std::move(predecessor).value_or(nullptr)) {
    for (const auto *table : {&weight_, &scales_})
      TORCH_CHECK(table->device().is_cpu() && table->dim() == 2 &&
                      table->is_contiguous() &&
                      table->scalar_type() == at::kByte && table->size(0) > 0,
                  "Engram Host tables must be non-empty contiguous uint8 rows");
    TORCH_CHECK(weight_.size(0) == scales_.size(0),
                "Engram weight and scale row counts differ");
    rows_ = weight_.size(0);
    weight_bytes_ = weight_.size(1);
    row_bytes_ = weight_bytes_ + scales_.size(1);
    capacity_ =
        std::min<int64_t>(ids_.bytes / 4 - 2, output_.bytes / row_bytes_);
    TORCH_CHECK(capacity_ > 0, "Engram channel has no request capacity");
    auto raw =
        xla::PjRtRawBuffer::CreateRawAliasOfBuffer(*destination_.AwaitBuffer());
    TORCH_CHECK(raw.ok(), raw.status().message());
    raw_ = std::move(raw).value();
    TORCH_CHECK(raw_->GetOnDeviceSizeInBytes() ==
                    static_cast<size_t>(output_.bytes),
                "Engram HBM destination must match the gather output");
    ids_view_ = ids_.View();
    request_view_ = request_.View();
    worker_ = std::thread([this] { Run(); });
  }

  ~HostChannel() override { Stop(); }

  at::Tensor ids() const { return ids_view_; }
  at::Tensor request() const { return request_view_; }

  // Empty while healthy; the first failure otherwise.
  std::string error() const {
    return failed_.load(std::memory_order_acquire) ? error_ : std::string();
  }

  // Caller must ensure no executable still waits for a response it needs.
  void Stop() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
      __atomic_store_n(response_.data, kStopped, __ATOMIC_RELEASE);
    }
  }

private:
  void Run() {
    int32_t done = 0;
    while (running_.load(std::memory_order_acquire)) {
      const int32_t generation =
          __atomic_load_n(request_.data, __ATOMIC_ACQUIRE);
      if (generation <= 0 || generation == done) {
        std::this_thread::yield();
        continue;
      }
      // The device publishes the payload before its generation.
      const int32_t count = ids_.data[0];
      const int32_t dependency = ids_.data[1];
      bool ok = !failed_.load(std::memory_order_relaxed);
      if (ok && !AwaitPredecessor(dependency))
        ok = running_.load(std::memory_order_acquire) &&
             Fail("predecessor channel failed or stopped");
      if (ok)
        ok = Serve(count, ids_.data + 2);
      if (!running_.load(std::memory_order_acquire))
        return;
      __atomic_store_n(response_.data, ok ? generation : -generation,
                       __ATOMIC_RELEASE);
      done = generation;
    }
  }

  bool AwaitPredecessor(int32_t dependency) const {
    if (!predecessor_ || dependency == 0)
      return true;
    while (running_.load(std::memory_order_acquire)) {
      const int32_t answer =
          __atomic_load_n(predecessor_->response_.data, __ATOMIC_ACQUIRE);
      if (answer == dependency)
        return true;
      if (answer == -dependency || answer == kStopped)
        return false;
      std::this_thread::yield();
    }
    return false;
  }

  bool Serve(int64_t count, const int32_t *rows) {
    if (count < 0 || count > capacity_)
      return Fail("row count " + std::to_string(count) + " outside [0, " +
                  std::to_string(capacity_) + "]");
    if (count == 0)
      return true;
    for (int64_t i = 0; i < count; ++i)
      if (rows[i] < 0 || rows[i] >= rows_)
        return Fail("row id " + std::to_string(rows[i]) + " outside [0, " +
                    std::to_string(rows_) + ")");
    const auto *weight = static_cast<const uint8_t *>(weight_.const_data_ptr());
    const auto *scales = static_cast<const uint8_t *>(scales_.const_data_ptr());
    const int64_t scale_bytes = row_bytes_ - weight_bytes_;
    auto *output = reinterpret_cast<uint8_t *>(output_.data);
    constexpr int64_t kAhead = 16;
    for (int64_t i = 0; i < count; ++i) {
      if (i + kAhead < count) {
        const int64_t next = rows[i + kAhead];
        for (int64_t offset = 0; offset < weight_bytes_; offset += 64)
          __builtin_prefetch(weight + next * weight_bytes_ + offset, 0, 0);
        __builtin_prefetch(scales + next * scale_bytes, 0, 0);
      }
      uint8_t *row = output + i * row_bytes_;
      std::memcpy(row, weight + rows[i] * weight_bytes_, weight_bytes_);
      std::memcpy(row + weight_bytes_, scales + rows[i] * scale_bytes,
                  scale_bytes);
    }
    const int64_t bytes =
        (count * row_bytes_ + kPageBytes - 1) / kPageBytes * kPageBytes;
    const auto status = raw_->CopyRawHostToDevice(output, 0, bytes).Await();
    return status.ok() || Fail(std::string(status.message()));
  }

  bool Fail(std::string message) {
    if (!failed_.load(std::memory_order_relaxed)) {
      std::fprintf(stderr, "[DSV41] Engram Host channel failed: %s\n",
                   message.c_str());
      error_ = std::move(message);
      failed_.store(true, std::memory_order_release);
    }
    return false;
  }

  at::Tensor weight_, scales_;
  HostLease ids_, request_, response_, output_;
  torch_tpu::TensorBufferHandle destination_;
  xla::PjRtRawBufferRef raw_;
  c10::intrusive_ptr<HostChannel> predecessor_;
  at::Tensor ids_view_, request_view_;
  int64_t rows_, weight_bytes_, row_bytes_, capacity_;
  std::string error_;
  std::atomic<bool> failed_{false};
  std::atomic<bool> running_{true};
  std::thread worker_;
};

} // namespace

TORCH_LIBRARY(engram_host_channel, library) {
  library.class_<HostChannel>("HostChannel")
      .def(torch::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                       at::Tensor, at::Tensor, at::Tensor,
                       std::optional<c10::intrusive_ptr<HostChannel>>>())
      .def("ids", &HostChannel::ids)
      .def("request", &HostChannel::request)
      .def("error", &HostChannel::error)
      .def("stop", &HostChannel::Stop);
}
