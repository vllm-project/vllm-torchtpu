// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// TorchTPU tensor <-> PJRT buffer shim for the Host gather benchmark.
//
// Build with the XLA_COMMIT shipped by the installed TorchTPU wheel; its
// tensor-buffer header is explicitly unsupported and has no stable ABI.
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "csrc/api/tensor_buffer.h"
#include "torch/custom_class.h"
#include "torch/library.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"

namespace {
torch_tpu::TensorBufferHandle Resolve(const at::Tensor &tensor, bool host) {
  TORCH_CHECK(tensor.device().type() == at::DeviceType::PrivateUse1,
              "Expected a TorchTPU tensor");
  TORCH_CHECK(tensor.dim() == 1 && tensor.is_contiguous() &&
                  tensor.storage_offset() == 0 &&
                  tensor.scalar_type() == at::kInt,
              "Expected a complete contiguous int32 base buffer");
  auto resolved = torch_tpu::GetBaseTensorBuffer(tensor);
  TORCH_CHECK(resolved.ok(), resolved.status().message());
  auto handle = std::move(resolved).value();
  TORCH_CHECK(handle.size_bytes() == static_cast<int64_t>(tensor.nbytes()),
              "Tensor must span its entire base allocation");
  for (auto status : {handle.Materialize(), handle.Synchronize()})
    TORCH_CHECK(status.ok(), status.message());
  auto buffer = handle.AwaitBuffer();
  TORCH_CHECK(buffer.ok(), buffer.status().message());
  TORCH_CHECK((*buffer)->memory_space() != nullptr &&
                  (*buffer)->memory_space()->kind() ==
                      (host ? "pinned_host" : "device"),
              "Unexpected buffer memory space");
  return handle;
}

int64_t PinnedHostPointer(const at::Tensor &tensor) {
  auto handle = Resolve(tensor, true);
  auto *buffer = *handle.AwaitBuffer();
  auto pointer = buffer->client()->UnsafeBufferPointer(buffer);
  TORCH_CHECK(pointer.ok(), pointer.status().message());
  TORCH_CHECK(*pointer != 0 && *pointer % 4096 == 0,
              "Expected a page-aligned Host buffer");
  return static_cast<int64_t>(*pointer);
}

// Pins the original allocation even if a compiler replaces the tensor buffer,
// so a background worker never touches freed memory; identity checks then fail.
class PinnedHostBuffer : public torch::CustomClassHolder {
public:
  explicit PinnedHostBuffer(at::Tensor tensor)
      : handle_(Resolve(tensor, true)) {
    auto reference = (*handle_.AwaitBuffer())->AcquireExternalReference();
    TORCH_CHECK(reference.ok(), reference.status().message());
    reference_ = std::move(reference).value();
    pointer_ = PinnedHostPointer(tensor);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(
                    reference_->OpaqueDeviceMemoryDataPointer()) ==
                    static_cast<uintptr_t>(pointer_),
                "External reference and Host pointer disagree");
  }
  int64_t data_ptr() const { return pointer_; }

private:
  torch_tpu::TensorBufferHandle handle_;
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> reference_;
  int64_t pointer_;
};

// Keeps the registered allocation alive and donates a separate mutable PJRT
// view of it. An external reference on the donated buffer itself is forbidden.
class PinnedHostView : public torch::CustomClassHolder {
public:
  explicit PinnedHostView(at::Tensor tensor)
      : owner_(std::make_shared<PinnedHostBuffer>(tensor)) {
    auto original = Resolve(tensor, true);
    auto *source = *original.AwaitBuffer();
    auto *client = source->client();
    auto *pointer = reinterpret_cast<void *>(owner_->data_ptr());
    auto owner = owner_;
    auto view = client->CreateViewOfDeviceBuffer(
        pointer, source->on_device_shape(), source->memory_space(), [owner] {});
    if (!view.ok()) {
      const std::string error(view.status().message());
      view = client->BufferFromHostBuffer(
          pointer, xla::S32, source->on_device_shape().dimensions(),
          std::nullopt, xla::PjRtClient::HostBufferSemantics::kMutableZeroCopy,
          [owner] {}, source->memory_space(), nullptr);
      TORCH_CHECK(view.ok(), "Host view: ", error,
                  "; mutable Host import: ", view.status().message());
    }
    auto ready = (*view)->GetReadyFuture().Await();
    TORCH_CHECK(ready.ok(), ready.message());
    auto address = client->UnsafeBufferPointer(view->get());
    TORCH_CHECK(address.ok() &&
                    *address == static_cast<uintptr_t>(owner_->data_ptr()),
                "Host import copied the allocation instead of aliasing it");
    auto handle = torch_tpu::MakeTensorBuffer(std::move(view).value());
    TORCH_CHECK(handle.ok(), handle.status().message());
    view_ = std::make_unique<torch_tpu::TensorBufferHandle>(
        std::move(handle).value());
  }
  at::Tensor tensor() const {
    auto result = view_->ToTensor();
    TORCH_CHECK(result.ok(), result.status().message());
    return std::move(result).value();
  }

private:
  std::shared_ptr<PinnedHostBuffer> owner_;
  std::unique_ptr<torch_tpu::TensorBufferHandle> view_;
};

// A raw alias of a retained HBM tensor that the worker writes outside the
// graph.
class RawHbmBuffer : public torch::CustomClassHolder {
public:
  explicit RawHbmBuffer(at::Tensor tensor) : handle_(Resolve(tensor, false)) {
    auto raw =
        xla::PjRtRawBuffer::CreateRawAliasOfBuffer(*handle_.AwaitBuffer());
    TORCH_CHECK(raw.ok(), raw.status().message());
    raw_ = std::move(raw).value();
    TORCH_CHECK(raw_->GetOnDeviceSizeInBytes() == tensor.nbytes(),
                "Raw H2D requires an unpadded whole allocation");
  }
  int64_t context() const { return reinterpret_cast<int64_t>(this); }
  // `submitted` receives the steady-clock time at which the copy was enqueued.
  bool upload(const void *source, int64_t bytes, int64_t *submitted) {
    if (bytes > raw_->GetOnDeviceSizeInBytes())
      return false;
    auto done = raw_->CopyRawHostToDevice(source, 0, bytes);
    *submitted = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();
    return done.Await().ok();
  }

private:
  torch_tpu::TensorBufferHandle handle_;
  xla::PjRtRawBufferRef raw_;
};
} // namespace

extern "C" bool CopyRawChannelOutput(void *context, const void *source,
                                     int64_t bytes, int64_t *submitted) {
  return static_cast<RawHbmBuffer *>(context)->upload(source, bytes, submitted);
}

TORCH_LIBRARY(host_gather_bench, library) {
  library.def("pinned_host_pointer(Tensor value) -> int", PinnedHostPointer);
  library.class_<PinnedHostBuffer>("PinnedHostBuffer")
      .def(torch::init<at::Tensor>())
      .def("data_ptr", &PinnedHostBuffer::data_ptr);
  library.class_<PinnedHostView>("PinnedHostView")
      .def(torch::init<at::Tensor>())
      .def("tensor", &PinnedHostView::tensor);
  library.class_<RawHbmBuffer>("RawHbmBuffer")
      .def(torch::init<at::Tensor>())
      .def("context", &RawHbmBuffer::context);
}
