// Zero-copy PyTorch Custom Operator Bridge (`torch.ops.deepsea_moe.*`)
// compiled with GCC 12 (`libstdc++`, C++20) against the Cloud TPU vLLM
// container's `libtorch`, `libpywrap_2_13_0_common.so`, and `libxla_base.so`.
//
// Compiler / proto assembly serialization is delegated across a pure C ABI to
// `libtpuc_serializer.so` (`tpuc_serialize_executable` / `tpuc_free_buffer`),
// run by default in a child process (`lib/tpuc_serialize_cli.py`), while PJRT
// executable loading and execution use the pure C `PJRT_Api` table.

#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <Python.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <torch/library.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "pjrt_c_api.h"

// =============================================================================
// 1. Minimal binary-exact declarations for absl::lts_20260526, xla, torch_tpu
// =============================================================================

namespace absl {
inline namespace lts_20260526 {

enum class StatusToStringMode : int {
  kWithNoExtraData = 0,
  kWithPayload = 1 << 0,
  kWithEverything = ~kWithNoExtraData,
  kDefault = kWithPayload,
};

namespace status_internal {
struct StatusRep {
  void Unref() const;
};
}  // namespace status_internal

class Status {
 public:
  Status() noexcept : rep_(1) {}
  explicit Status(uintptr_t rep) noexcept : rep_(rep) {}
  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;
  Status(Status&& other) noexcept : rep_(other.rep_) { other.rep_ = 1; }
  ~Status() {
    if ((rep_ & 1) == 0) {
      reinterpret_cast<const status_internal::StatusRep*>(rep_)->Unref();
    }
  }
  bool ok() const noexcept { return rep_ == 1; }
  std::string ToString() const {
    if (ok()) return "OK";
    return ToStringSlow(rep_, StatusToStringMode::kDefault);
  }

 private:
  static std::string ToStringSlow(uintptr_t rep, StatusToStringMode mode);
  uintptr_t rep_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr() = delete;
  StatusOr(const StatusOr&) = delete;
  StatusOr& operator=(const StatusOr&) = delete;
  ~StatusOr() {
    if (ok()) {
      value_.~T();
    } else {
      status_.~Status();
    }
  }
  bool ok() const noexcept { return status_.ok(); }
  const Status& status() const noexcept { return status_; }
  T& operator*() & noexcept { return value_; }
  T&& operator*() && noexcept { return std::move(value_); }
  T* operator->() noexcept { return &value_; }
  const T* operator->() const noexcept { return &value_; }

 private:
  Status status_;
  union {
    char dummy_;
    T value_;
  };
};

template <typename T>
class Span {
 public:
  constexpr Span() noexcept : ptr_(nullptr), len_(0) {}
  constexpr Span(const T* ptr, size_t len) noexcept : ptr_(ptr), len_(len) {}
  constexpr const T* data() const noexcept { return ptr_; }
  constexpr size_t size() const noexcept { return len_; }
  constexpr bool empty() const noexcept { return len_ == 0; }
  constexpr const T& operator[](size_t i) const noexcept { return ptr_[i]; }
  friend bool operator==(Span a, Span b) noexcept {
    if (a.len_ != b.len_) return false;
    for (size_t i = 0; i < a.len_; ++i) {
      if (a.ptr_[i] != b.ptr_[i]) return false;
    }
    return true;
  }
  friend bool operator!=(Span a, Span b) noexcept { return !(a == b); }

 private:
  const T* ptr_;
  size_t len_;
};

}  // namespace lts_20260526
}  // namespace absl

namespace xla {

class Shape {
 public:
  std::string ToString(bool print_layout = false) const;
};

class PjRtCApiDeviceDescription {
 public:
  int id() const;
};

class PjRtCApiDevice {
 public:
  const PjRtCApiDeviceDescription& description() const;
};

class PjRtCApiClient {
 public:
  const PJRT_Api* pjrt_c_api() const;
  PJRT_Client* pjrt_c_client() const {
    return reinterpret_cast<PJRT_Client* const*>(this)[7];  // offset 0x38
  }
};

class PjRtBuffer {
 public:
  virtual ~PjRtBuffer() = default;
};

class PjRtCApiBuffer : public PjRtBuffer {
 public:
  PjRtCApiBuffer(PjRtCApiClient* client, PJRT_Buffer* buffer);
  const Shape& on_device_shape() const;
  PjRtCApiDevice* device() const;

  PjRtCApiClient* client() const {
    return reinterpret_cast<PjRtCApiClient* const*>(this)[1];  // offset 0x08
  }
  PJRT_Buffer* c_buffer() const {
    return reinterpret_cast<PJRT_Buffer* const*>(this)[6];  // offset 0x30
  }
};

}  // namespace xla

namespace torch_tpu {

class TensorBufferHandle final {
 public:
  TensorBufferHandle(const TensorBufferHandle&) = delete;
  TensorBufferHandle& operator=(const TensorBufferHandle&) = delete;
  TensorBufferHandle(TensorBufferHandle&& other) noexcept;
  TensorBufferHandle& operator=(TensorBufferHandle&& other) noexcept;
  ~TensorBufferHandle();

  int64_t size_bytes() const;
  absl::Span<const int64_t> dimensions() const;
  int64_t num_elements() const;
  absl::Status Materialize();
  absl::StatusOr<xla::PjRtBuffer*> AwaitBuffer();
  absl::Status Synchronize();
  absl::StatusOr<at::Tensor> ToTensor() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

absl::StatusOr<TensorBufferHandle> GetBaseTensorBuffer(
    const at::Tensor& tensor);
absl::StatusOr<TensorBufferHandle> MakeTensorBuffer(
    std::unique_ptr<xla::PjRtBuffer> buffer);

}  // namespace torch_tpu

// =============================================================================
// 2. C-ABI Serializer Loader & PJRT C-API Execution Helpers
// =============================================================================

namespace deepsea::torch_bridge {
namespace {

struct ScopedGilRelease {
  PyThreadState* state = nullptr;
  ScopedGilRelease() {
    if (Py_IsInitialized() && PyGILState_Check()) {
      state = PyEval_SaveThread();
    }
  }
  ~ScopedGilRelease() {
    if (state != nullptr) {
      PyEval_RestoreThread(state);
    }
  }
};

using SerializeFn = int (*)(const char* asm_runfile, const char* entry_symbol,
                            const char* module_name, int num_params,
                            const char* const* param_shape_strs,
                            int num_results,
                            const char* const* result_shape_strs,
                            int64_t device_id, char** out_payload,
                            size_t* out_payload_len, char** out_error);
using FreeBufFn = void (*)(char* buf);

struct SerializerApi {
  SerializeFn serialize = nullptr;
  FreeBufFn free_buf = nullptr;
};

const SerializerApi& GetSerializerApi() {
  static const SerializerApi api = [] {
    SerializerApi a;
    a.serialize = reinterpret_cast<SerializeFn>(
        dlsym(RTLD_DEFAULT, "tpuc_serialize_executable"));
    a.free_buf =
        reinterpret_cast<FreeBufFn>(dlsym(RTLD_DEFAULT, "tpuc_free_buffer"));
    if (!a.serialize || !a.free_buf) {
      if (const char* bundle_dir = std::getenv("TPUC_BUNDLE_DIR")) {
        std::string so_path =
            std::string(bundle_dir) + "/lib/libtpuc_serializer.so";
        void* h = dlopen(so_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (h) {
          a.serialize = reinterpret_cast<SerializeFn>(
              dlsym(h, "tpuc_serialize_executable"));
          a.free_buf =
              reinterpret_cast<FreeBufFn>(dlsym(h, "tpuc_free_buffer"));
        }
      }
    }
    TORCH_CHECK(
        a.serialize != nullptr && a.free_buf != nullptr,
        "Failed to resolve tpuc_serialize_executable / tpuc_free_buffer "
        "from libtpuc_serializer.so");
    return a;
  }();
  return api;
}

std::string ConsumePjrtError(const PJRT_Api* c_api, PJRT_Error* err) {
  if (err == nullptr) return "OK";
  PJRT_Error_Message_Args msg_args{};
  msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg_args.error = err;
  c_api->PJRT_Error_Message(&msg_args);
  std::string msg(msg_args.message, msg_args.message_size);

  PJRT_Error_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = err;
  c_api->PJRT_Error_Destroy(&destroy_args);
  return msg;
}

inline xla::PjRtCApiBuffer* AsCApiBuffer(xla::PjRtBuffer* b) {
  return static_cast<xla::PjRtCApiBuffer*>(b);
}

// By default `tpuc_serialize_executable` runs in a child process
// (`$TPUC_BUNDLE_DIR/lib/tpuc_serialize_cli.py`). `libtpuc_serializer.so`
// statically links google3 (including TPU runtime libraries) and its first
// call runs `InitGoogle`, which starts ~300 threads; doing that inside the
// process that already runs libtpu intermittently crashed the chip
// (SparseCore halts / UserFatal) and the host (b/575455577).
// `TPUC_SERIALIZE_SUBPROCESS=0` restores the in-process path.
bool SerializeInSubprocess() {
  const char* env = std::getenv("TPUC_SERIALIZE_SUBPROCESS");
  return env == nullptr || std::string_view(env) != "0";
}

std::string SerializeViaSubprocess(
    const char* asm_runfile, const std::string& entry_symbol,
    const std::string& module_name,
    const std::vector<std::string>& param_shapes,
    const std::vector<std::string>& result_shapes, int64_t device_id) {
  const char* bundle_dir = std::getenv("TPUC_BUNDLE_DIR");
  TORCH_CHECK(bundle_dir != nullptr,
              "out-of-process TPU-C serialization needs TPUC_BUNDLE_DIR");
  const char* py = std::getenv("TPUC_SERIALIZE_PYTHON");
  const std::string python = py != nullptr ? py : "python3";
  const std::string cli =
      std::string(bundle_dir) + "/lib/tpuc_serialize_cli.py";
  char out_tmpl[] = "/tmp/tpuc_payload_XXXXXX";
  const int out_fd = mkstemp(out_tmpl);
  TORCH_CHECK(out_fd >= 0, "mkstemp failed");
  close(out_fd);
  const std::string out_path = out_tmpl;

  std::vector<std::string> args = {
      python,      cli,
      asm_runfile, entry_symbol,
      module_name, std::to_string(device_id),
      out_path,    std::to_string(param_shapes.size())};
  for (const auto& s : param_shapes) args.push_back(s);
  for (const auto& s : result_shapes) args.push_back(s);
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);

  // The serializer child itself occasionally segfaults (6 of ~360 runs on
  // CDK, always on the first attempt); each run is independent, so retry.
  constexpr int kMaxAttempts = 3;
  int status = 0;
  std::string payload;
  for (int attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    pid_t pid = 0;
    int rc = posix_spawnp(&pid, python.c_str(), nullptr, nullptr, argv.data(),
                          environ);
    TORCH_CHECK(rc == 0, "posix_spawnp(", python, ") failed: ", rc);
    {
      ScopedGilRelease gil_release;
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
      std::ifstream in(out_path, std::ios::in | std::ios::binary);
      payload.assign(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
      if (!payload.empty()) break;
    }
    fprintf(stderr,
            "[TPU-C] tpuc_serialize_cli(%s) attempt %d/%d failed, status=%d\n",
            entry_symbol.c_str(), attempt, kMaxAttempts, status);
  }
  unlink(out_path.c_str());
  TORCH_CHECK(!payload.empty(), "tpuc_serialize_cli(", entry_symbol,
              ") failed, status=", status);
  return payload;
}

inline std::string ShapeString(xla::PjRtBuffer* b) {
  return AsCApiBuffer(b)->on_device_shape().ToString(/*print_layout=*/true);
}

inline int GetDeviceId(xla::PjRtBuffer* b) {
  xla::PjRtCApiDevice* dev = AsCApiBuffer(b)->device();
  TORCH_CHECK(dev != nullptr, "TPU-C bridge: buffer has null PJRT device");
  return dev->description().id();
}

struct LoadedCExecutable {
  const PJRT_Api* c_api = nullptr;
  xla::PjRtCApiClient* cpp_client = nullptr;
  PJRT_LoadedExecutable* c_exec = nullptr;
  size_t num_outputs = 1;
};

LoadedCExecutable* CompileAndLoadCExecutable(
    xla::PjRtCApiClient* cpp_client, const char* asm_runfile,
    const std::string& entry_symbol, const std::string& module_name,
    const std::vector<std::string>& param_shapes,
    const std::vector<std::string>& result_shapes, int64_t device_id) {
  std::vector<const char*> p_ptrs;
  p_ptrs.reserve(param_shapes.size());
  for (const auto& s : param_shapes) p_ptrs.push_back(s.c_str());

  std::vector<const char*> r_ptrs;
  r_ptrs.reserve(result_shapes.size());
  for (const auto& s : result_shapes) r_ptrs.push_back(s.c_str());

  std::string payload_str;
  const bool use_subprocess = SerializeInSubprocess();
  char* payload = nullptr;
  size_t payload_len = 0;
  if (use_subprocess) {
    payload_str =
        SerializeViaSubprocess(asm_runfile, entry_symbol, module_name,
                               param_shapes, result_shapes, device_id);
  } else {
    const auto& ser = GetSerializerApi();
    char* err_str = nullptr;
    int rc =
        ser.serialize(asm_runfile, entry_symbol.c_str(), module_name.c_str(),
                      static_cast<int>(p_ptrs.size()), p_ptrs.data(),
                      static_cast<int>(r_ptrs.size()), r_ptrs.data(), device_id,
                      &payload, &payload_len, &err_str);
    if (rc != 0) {
      std::string msg = err_str ? err_str : "unknown serializer error";
      if (err_str) ser.free_buf(err_str);
      TORCH_CHECK(false, "tpuc_serialize_executable(", entry_symbol,
                  ") failed: ", msg);
    }
  }

  const PJRT_Api* c_api = cpp_client->pjrt_c_api();
  PJRT_LoadOptions load_opts{};
  load_opts.struct_size = PJRT_LoadOptions_STRUCT_SIZE;

  PJRT_Executable_DeserializeAndLoad_Args load_args{};
  load_args.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
  load_args.client = cpp_client->pjrt_c_client();
  load_args.serialized_executable =
      use_subprocess ? payload_str.data() : payload;
  load_args.serialized_executable_size =
      use_subprocess ? payload_str.size() : payload_len;
  load_args.load_options = &load_opts;

  PJRT_Error* pjrt_err = c_api->PJRT_Executable_DeserializeAndLoad(&load_args);
  if (!use_subprocess) GetSerializerApi().free_buf(payload);
  TORCH_CHECK(pjrt_err == nullptr, "PJRT_Executable_DeserializeAndLoad(",
              entry_symbol, ") failed: ", ConsumePjrtError(c_api, pjrt_err));

  auto* loaded = new LoadedCExecutable();
  loaded->c_api = c_api;
  loaded->cpp_client = cpp_client;
  loaded->c_exec = load_args.loaded_executable;
  loaded->num_outputs = result_shapes.size();
  return loaded;
}

std::unique_ptr<xla::PjRtBuffer> WrapCPjRtBuffer(
    xla::PjRtCApiClient* cpp_client, PJRT_Buffer* c_buf) {
  void* mem = ::operator new(0x280);
  std::memset(mem, 0, 0x280);
  auto* obj = new (mem) xla::PjRtCApiBuffer(cpp_client, c_buf);
  return std::unique_ptr<xla::PjRtBuffer>(obj);
}

std::vector<at::Tensor> ExecuteToTensors(
    LoadedCExecutable* exe, const std::vector<xla::PjRtBuffer*>& args,
    std::string_view op_name) {
  std::vector<PJRT_Buffer*> c_args;
  c_args.reserve(args.size());
  std::vector<int64_t> non_donatable;
  non_donatable.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    c_args.push_back(AsCApiBuffer(args[i])->c_buffer());
    non_donatable.push_back(static_cast<int64_t>(i));
  }

  PJRT_ExecuteOptions opts{};
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  opts.non_donatable_input_indices = non_donatable.data();
  opts.num_non_donatable_input_indices = non_donatable.size();

  PJRT_Buffer* const* arg_list_ptr = c_args.data();
  std::vector<PJRT_Buffer*> out_bufs(exe->num_outputs, nullptr);
  PJRT_Buffer** out_list_ptr = out_bufs.data();

  PJRT_LoadedExecutable_Execute_Args exec_args{};
  exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  exec_args.executable = exe->c_exec;
  exec_args.options = &opts;
  exec_args.argument_lists = &arg_list_ptr;
  exec_args.num_devices = 1;
  exec_args.num_args = c_args.size();
  exec_args.output_lists = &out_list_ptr;

  PJRT_Error* err = exe->c_api->PJRT_LoadedExecutable_Execute(&exec_args);
  TORCH_CHECK(err == nullptr, op_name,
              ": PJRT_LoadedExecutable_Execute failed: ",
              ConsumePjrtError(exe->c_api, err));

  std::vector<at::Tensor> tensors;
  tensors.reserve(exe->num_outputs);
  for (size_t i = 0; i < exe->num_outputs; ++i) {
    auto wrapped = WrapCPjRtBuffer(exe->cpp_client, out_bufs[i]);
    auto handle_or = torch_tpu::MakeTensorBuffer(std::move(wrapped));
    TORCH_CHECK(handle_or.ok(), op_name,
                ": MakeTensorBuffer failed: ", handle_or.status().ToString());
    auto tensor_or = handle_or->ToTensor();
    TORCH_CHECK(tensor_or.ok(), op_name,
                ": ToTensor failed: ", tensor_or.status().ToString());
    tensors.push_back(*std::move(tensor_or));
  }
  return tensors;
}

void ExecuteVoid(LoadedCExecutable* exe,
                 const std::vector<xla::PjRtBuffer*>& args,
                 std::string_view op_name) {
  std::vector<PJRT_Buffer*> c_args;
  c_args.reserve(args.size());
  std::vector<int64_t> non_donatable;
  non_donatable.reserve(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    c_args.push_back(AsCApiBuffer(args[i])->c_buffer());
    non_donatable.push_back(static_cast<int64_t>(i));
  }

  PJRT_ExecuteOptions opts{};
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  opts.non_donatable_input_indices = non_donatable.data();
  opts.num_non_donatable_input_indices = non_donatable.size();

  PJRT_Buffer* const* arg_list_ptr = c_args.data();
  std::vector<PJRT_Buffer*> out_bufs(exe->num_outputs, nullptr);
  PJRT_Buffer** out_list_ptr = out_bufs.data();

  const char* sync_env = std::getenv("TPU_MOE_SYNC_EXECUTE");
  const bool sync = (sync_env == nullptr || std::string_view(sync_env) != "0");
  PJRT_Event* complete_event = nullptr;

  PJRT_LoadedExecutable_Execute_Args exec_args{};
  exec_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  exec_args.executable = exe->c_exec;
  exec_args.options = &opts;
  exec_args.argument_lists = &arg_list_ptr;
  exec_args.num_devices = 1;
  exec_args.num_args = c_args.size();
  exec_args.output_lists = &out_list_ptr;
  if (sync) {
    exec_args.device_complete_events = &complete_event;
  }

  PJRT_Error* err = nullptr;
  PJRT_Error* await_err = nullptr;
  {
    ScopedGilRelease gil_release;
    err = exe->c_api->PJRT_LoadedExecutable_Execute(&exec_args);
    if (err == nullptr && sync && complete_event != nullptr) {
      PJRT_Event_Await_Args await_args{};
      await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
      await_args.event = complete_event;
      await_err = exe->c_api->PJRT_Event_Await(&await_args);

      PJRT_Event_Destroy_Args ev_destroy{};
      ev_destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      ev_destroy.event = complete_event;
      exe->c_api->PJRT_Event_Destroy(&ev_destroy);
    }
  }
  TORCH_CHECK(err == nullptr, op_name,
              ": PJRT_LoadedExecutable_Execute failed: ",
              ConsumePjrtError(exe->c_api, err));
  TORCH_CHECK(await_err == nullptr, op_name,
              ": sync await failed: ", ConsumePjrtError(exe->c_api, await_err));

  for (PJRT_Buffer* b : out_bufs) {
    if (b != nullptr) {
      PJRT_Buffer_Destroy_Args destroy_args{};
      destroy_args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      destroy_args.buffer = b;
      exe->c_api->PJRT_Buffer_Destroy(&destroy_args);
    }
  }
}

}  // namespace

// =============================================================================
// 3. torch.ops.deepsea_moe.* (MoE Fused Expert FFN & Probes)
// =============================================================================

namespace moe {
namespace {

constexpr char kFfnAsm[] =
    "google3/torchtpu_tpuc_kernels/libmoe_fused_ffn_kernel_gf.s";

constexpr char kFfnH2048Asm[] =
    "google3/torchtpu_tpuc_kernels/libmoe_fused_ffn_kernel_h2048_gf.s";

constexpr char kFfnMxuOnlyAsm[] =
    "google3/torchtpu_tpuc_kernels/libmoe_fused_ffn_kernel_mxuonly_gf.s";

constexpr char kMxuAsm[] =
    "google3/platforms/deepsea/software/toolchain/kernels/moe/implementation/"
    "libmoe_mxu_probe_kernel_gf.s";

constexpr char kProbeAsm[] =
    "google3/platforms/deepsea/software/toolchain/kernels/moe/implementation/"
    "libmoe_dma_probe_kernel_gf.s";

struct PinnedBuffers {
  std::vector<torch_tpu::TensorBufferHandle> handles;
  std::vector<xla::PjRtBuffer*> args;
};

PinnedBuffers PinBuffers(std::vector<at::Tensor>& inputs, std::string_view op) {
  PinnedBuffers pinned;
  pinned.handles.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto h = torch_tpu::GetBaseTensorBuffer(inputs[i]);
    TORCH_CHECK(h.ok(), op, ": no buffer for operand ", i);
    const at::IntArrayRef sizes = inputs[i].sizes();
    TORCH_CHECK(h->dimensions() ==
                    absl::Span<const int64_t>(sizes.data(), sizes.size()),
                op, ": operand ", i, " must be a dense base tensor");
    pinned.handles.push_back(*std::move(h));
  }
  // Materialize + AwaitBuffer submits each operand to PJRT; with
  // TPU_MOE_SYNC_EXECUTE (default on) also wait for its producer to finish
  // before the raw PJRT launch (as `moe_bridge.cc` `Buffers()` does).
  const char* sync_env = std::getenv("TPU_MOE_SYNC_EXECUTE");
  const bool sync = (sync_env == nullptr || std::string(sync_env) != "0");
  pinned.args.reserve(pinned.handles.size());
  std::string err_msg;
  size_t err_idx = 0;
  const char* err_step = "";
  {
    ScopedGilRelease gil_release;
    for (size_t i = 0; i < pinned.handles.size(); ++i) {
      absl::Status st = pinned.handles[i].Materialize();
      if (!st.ok()) {
        err_msg = st.ToString();
        err_idx = i;
        err_step = "materialize";
        break;
      }
      auto buf = pinned.handles[i].AwaitBuffer();
      if (!buf.ok()) {
        err_msg = buf.status().ToString();
        err_idx = i;
        err_step = "await";
        break;
      }
      if (sync) {
        absl::Status sync_st = pinned.handles[i].Synchronize();
        if (!sync_st.ok()) {
          err_msg = sync_st.ToString();
          err_idx = i;
          err_step = "synchronize";
          break;
        }
      }
      pinned.args.push_back(*buf);
    }
  }
  TORCH_CHECK(err_msg.empty(), op, ": ", err_step, " ", err_idx, ": ", err_msg);
  return pinned;
}

LoadedCExecutable* GetExe(const char* asm_runfile, const std::string& entry,
                          const std::vector<xla::PjRtBuffer*>& args,
                          const std::string& result_shape) {
  static std::mutex mu;
  static auto* cache = new std::map<std::string, LoadedCExecutable*>();
  xla::PjRtCApiDevice* dev = AsCApiBuffer(args[0])->device();
  TORCH_CHECK(dev != nullptr, entry, ": operand 0 has null PJRT device");
  for (size_t i = 1; i < args.size(); ++i) {
    TORCH_CHECK(AsCApiBuffer(args[i])->device() == dev, entry, ": operand ", i,
                " is on a different PJRT device than operand 0");
  }
  const int device_id = dev->description().id();
  xla::PjRtCApiClient* client = AsCApiBuffer(args[0])->client();
  std::vector<std::string> shapes;
  shapes.reserve(args.size());
  std::string key = std::to_string(reinterpret_cast<uintptr_t>(client)) + "|d" +
                    std::to_string(device_id) + "|" + entry;
  for (xla::PjRtBuffer* b : args) {
    shapes.push_back(ShapeString(b));
    key += "|";
    key += shapes.back();
  }
  std::lock_guard<std::mutex> lock(mu);
  if (auto it = cache->find(key); it != cache->end()) return it->second;
  std::string module = entry + "_d" + std::to_string(device_id) + "_" +
                       std::to_string(cache->size());
  LoadedCExecutable* exe = CompileAndLoadCExecutable(
      client, asm_runfile, entry, module, shapes, {result_shape}, device_id);
  cache->emplace(key, exe);
  return exe;
}

at::Tensor Run(LoadedCExecutable* exe,
               const std::vector<xla::PjRtBuffer*>& args, std::string_view op) {
  auto outs = ExecuteToTensors(exe, args, op);
  return std::move(outs[0]);
}

void RunVoid(LoadedCExecutable* exe, const std::vector<xla::PjRtBuffer*>& args,
             std::string_view op) {
  ExecuteVoid(exe, args, op);
}

at::Tensor DmaProbe(const at::Tensor& w, const at::Tensor& meta) {
  TORCH_CHECK(
      meta.dim() == 1 && meta.size(0) == 16 && meta.scalar_type() == at::kInt,
      "dma_probe: meta int32[16]");
  TORCH_CHECK(w.scalar_type() == at::kInt && w.dim() == 1,
              "dma_probe: w int32[N]");
  std::vector<at::Tensor> inputs = {w, meta};
  auto pinned = PinBuffers(inputs, "dma_probe");
  const auto& args = pinned.args;
  const int device_id = GetDeviceId(args[0]);
  static std::mutex fmu;
  static auto* fast =
      new std::map<std::tuple<uintptr_t, int, int64_t>, LoadedCExecutable*>();
  const auto fkey = std::make_tuple(
      reinterpret_cast<uintptr_t>(AsCApiBuffer(args[0])->client()), device_id,
      w.size(0));
  LoadedCExecutable* exe = nullptr;
  {
    std::lock_guard<std::mutex> lock(fmu);
    if (auto it = fast->find(fkey); it != fast->end()) exe = it->second;
  }
  if (exe == nullptr) {
    exe = GetExe(kProbeAsm, "moe_dma_probe", args, ShapeString(args[1]));
    std::lock_guard<std::mutex> lock(fmu);
    fast->emplace(fkey, exe);
  }
  return Run(exe, args, "dma_probe");
}

at::Tensor MxuProbe(const at::Tensor& x, const at::Tensor& w, at::Tensor& out,
                    const at::Tensor& meta) {
  TORCH_CHECK(
      meta.dim() == 1 && meta.size(0) == 16 && meta.scalar_type() == at::kInt,
      "mxu_probe: meta int32[16]");
  TORCH_CHECK(x.scalar_type() == at::kInt && w.scalar_type() == at::kInt &&
                  out.scalar_type() == at::kFloat,
              "mxu_probe: x/w int32, out f32");
  std::vector<at::Tensor> inputs = {x, w, out, meta};
  auto pinned = PinBuffers(inputs, "mxu_probe");
  const auto& args = pinned.args;
  LoadedCExecutable* exe =
      GetExe(kMxuAsm, "moe_mxu_probe", args, ShapeString(args[3]));
  return Run(exe, args, "mxu_probe");
}

std::string Describe(const at::Tensor& t) {
  std::vector<at::Tensor> inputs = {t};
  auto pinned = PinBuffers(inputs, "describe");
  return ShapeString(pinned.args[0]);
}

void AwaitDispatch(const at::Tensor& t) {
  std::vector<at::Tensor> inputs = {t};
  (void)PinBuffers(inputs, "await_dispatch");
}

void CheckTiling(const std::vector<xla::PjRtBuffer*>& args, int i,
                 std::string_view name, std::string_view tiling) {
  const std::string s = ShapeString(args[i]);
  std::string pat1 = ":" + std::string(tiling) + "}";
  std::string pat2 = ":" + std::string(tiling) + "S(";
  TORCH_CHECK(
      s.find(pat1) != std::string::npos || s.find(pat2) != std::string::npos,
      "expert_ffn: ", name, " is stored as ", s, "; the kernel needs ", tiling);
}

std::pair<LoadedCExecutable*, PinnedBuffers> PrepareExpertFfn(
    const char* asm_runfile, std::string_view op_name, const at::Tensor& x,
    const at::Tensor& w13, const at::Tensor& w2, const at::Tensor& xs_b,
    const at::Tensor& rw_b, const at::Tensor& s13_b, const at::Tensor& s2_b,
    const at::Tensor& meta, at::Tensor& out) {
  TORCH_CHECK(x.scalar_type() == at::kFloat8_e4m3fn &&
                  w13.scalar_type() == at::kFloat8_e4m3fn &&
                  w2.scalar_type() == at::kFloat8_e4m3fn,
              op_name, ": x/w13/w2 are float8_e4m3fn");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, op_name, ": out bf16");
  TORCH_CHECK(
      xs_b.scalar_type() == at::kFloat && rw_b.scalar_type() == at::kFloat &&
          s13_b.scalar_type() == at::kFloat && s2_b.scalar_type() == at::kFloat,
      op_name, ": scales are f32");
  TORCH_CHECK(
      meta.dim() == 1 && meta.size(0) == 256 && meta.scalar_type() == at::kInt,
      op_name, ": meta int32[256]");
  // MoE dims: H = x.size(1), I = w2.size(1). H=4096/I=1024 (Qwen3.5-397B) or
  // H=2048/I=512 (Qwen3.5-35B-A3B); each pair has its own kernel build.
  TORCH_CHECK(x.dim() == 2 && x.size(0) % 32 == 0, op_name,
              ": x [R, H], R % 32 == 0");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3, op_name,
              ": w13 [E, H, 2I] and w2 [E, I, H] must be 3D");
  const int64_t inter = w2.size(1);
  const char* entry = nullptr;
  if (hidden == 4096 && inter == 1024) {
    entry = "moe_fused_expert_ffn_gf";
  } else if (hidden == 2048 && inter == 512) {
    TORCH_CHECK(asm_runfile == kFfnAsm, op_name,
                ": only the full kernel is built for H=2048, I=512");
    asm_runfile = kFfnH2048Asm;
    entry = "moe_fused_expert_ffn_gf_h2048";
  } else {
    TORCH_CHECK(false, op_name, ": unsupported MoE dims H=", hidden,
                ", I=", inter, " (supported: 4096/1024, 2048/512)");
  }
  const int64_t n_experts = w13.size(0);
  TORCH_CHECK(n_experts >= 1 && n_experts <= 80, op_name,
              ": w13.size(0) must be in [1, 80], got ", n_experts);
  TORCH_CHECK(w13.sizes() == at::IntArrayRef({n_experts, hidden, 2 * inter}) &&
                  w2.sizes() == at::IntArrayRef({n_experts, inter, hidden}),
              op_name, ": w13 [E, H, 2I], w2 [E, I, H]; got w13=", w13.sizes(),
              " w2=", w2.sizes());
  TORCH_CHECK(
      xs_b.sizes() == at::IntArrayRef({rows, 128}) &&
          rw_b.sizes() == at::IntArrayRef({rows, 128}) &&
          s13_b.sizes() == at::IntArrayRef({n_experts, 8, 2 * inter}) &&
          s2_b.sizes() == at::IntArrayRef({n_experts, 8, hidden}) &&
          (out.sizes() == at::IntArrayRef({rows, hidden}) ||
           out.sizes() == at::IntArrayRef({rows / 16, hidden / 128, 16, 128})),
      op_name,
      ": xs_b/rw_b [R, 128], s13_b [E, 8, 2I], s2_b [E, 8, H], out [R, H] "
      "(meta[4]=1) or [R/16, H/128, 16, 128] (meta[4]=0). Actual shapes: "
      "xs_b=",
      xs_b.sizes(), " rw_b=", rw_b.sizes(), " s13_b=", s13_b.sizes(),
      " s2_b=", s2_b.sizes(), " out=", out.sizes(), " expected rows=", rows,
      " n_experts=", n_experts, " H=", hidden, " I=", inter);
  std::vector<at::Tensor> inputs = {x,     w13,  w2,   xs_b, rw_b,
                                    s13_b, s2_b, meta, out};
  auto pinned = PinBuffers(inputs, op_name);
  const auto& args = pinned.args;
  xla::PjRtCApiDevice* dev = AsCApiBuffer(args[0])->device();
  TORCH_CHECK(dev != nullptr, op_name, ": operand 0 has null PJRT device");
  for (size_t i = 1; i < args.size(); ++i) {
    TORCH_CHECK(AsCApiBuffer(args[i])->device() == dev, op_name, ": operand ",
                i, " is on a different PJRT device than operand 0");
  }
  const int device_id = dev->description().id();
  static std::mutex fmu;
  static auto* fast =
      new std::map<std::tuple<uintptr_t, uintptr_t, int, int64_t, int64_t>,
                   LoadedCExecutable*>();
  const auto fkey = std::make_tuple(
      reinterpret_cast<uintptr_t>(AsCApiBuffer(args[0])->client()),
      reinterpret_cast<uintptr_t>(asm_runfile), device_id, n_experts,
      rows * 2 + (out.dim() == 4 ? 1 : 0));
  LoadedCExecutable* exe = nullptr;
  {
    std::lock_guard<std::mutex> lock(fmu);
    if (auto it = fast->find(fkey); it != fast->end()) exe = it->second;
  }
  if (exe == nullptr) {
    CheckTiling(args, 0, "x", "T(32,128)(4,1)");
    CheckTiling(args, 1, "w13", "T(32,128)(4,1)");
    CheckTiling(args, 2, "w2", "T(32,128)(4,1)");
    CheckTiling(args, 3, "xs_b", "T(8,128)");
    CheckTiling(args, 4, "rw_b", "T(8,128)");
    CheckTiling(args, 5, "s13_b", "T(8,128)");
    CheckTiling(args, 6, "s2_b", "T(8,128)");
    CheckTiling(args, 8, "out", "T(8,128)(2,1)");
    exe = GetExe(asm_runfile, entry, args, ShapeString(args[7]));
    std::lock_guard<std::mutex> lock(fmu);
    fast->emplace(fkey, exe);
  }
  return {exe, std::move(pinned)};
}

at::Tensor ExpertFfnImpl(const char* asm_runfile, const at::Tensor& x,
                         const at::Tensor& w13, const at::Tensor& w2,
                         const at::Tensor& xs_b, const at::Tensor& rw_b,
                         const at::Tensor& s13_b, const at::Tensor& s2_b,
                         const at::Tensor& meta, at::Tensor& out) {
  auto [exe, pinned] = PrepareExpertFfn(asm_runfile, "expert_ffn", x, w13, w2,
                                        xs_b, rw_b, s13_b, s2_b, meta, out);
  return Run(exe, pinned.args, "expert_ffn");
}

at::Tensor ExpertFfn(const at::Tensor& x, const at::Tensor& w13,
                     const at::Tensor& w2, const at::Tensor& xs_b,
                     const at::Tensor& rw_b, const at::Tensor& s13_b,
                     const at::Tensor& s2_b, const at::Tensor& meta,
                     at::Tensor& out) {
  return ExpertFfnImpl(kFfnAsm, x, w13, w2, xs_b, rw_b, s13_b, s2_b, meta, out);
}

void ExpertFfnOut(const at::Tensor& x, const at::Tensor& w13,
                  const at::Tensor& w2, const at::Tensor& xs_b,
                  const at::Tensor& rw_b, const at::Tensor& s13_b,
                  const at::Tensor& s2_b, const at::Tensor& meta,
                  at::Tensor& out) {
  auto [exe, pinned] = PrepareExpertFfn(kFfnAsm, "expert_ffn_out", x, w13, w2,
                                        xs_b, rw_b, s13_b, s2_b, meta, out);
  RunVoid(exe, pinned.args, "expert_ffn_out");
}

at::Tensor ExpertFfnMxuOnly(const at::Tensor& x, const at::Tensor& w13,
                            const at::Tensor& w2, const at::Tensor& xs_b,
                            const at::Tensor& rw_b, const at::Tensor& s13_b,
                            const at::Tensor& s2_b, const at::Tensor& meta,
                            at::Tensor& out) {
  return ExpertFfnImpl(kFfnMxuOnlyAsm, x, w13, w2, xs_b, rw_b, s13_b, s2_b,
                       meta, out);
}

at::Tensor FusedExpertFfnMeta(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor& meta,
                              at::Tensor&) {
  return at::empty({256}, meta.options());
}

void FusedExpertFfnOutMeta(const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&,
                           const at::Tensor&, const at::Tensor&, at::Tensor&) {}

at::Tensor MxuProbeMeta(const at::Tensor&, const at::Tensor&, at::Tensor&,
                        const at::Tensor& meta) {
  return at::empty({16}, meta.options());
}

at::Tensor DmaProbeMeta(const at::Tensor&, const at::Tensor& meta) {
  return at::empty({16}, meta.options());
}

void AwaitDispatchMeta(const at::Tensor&) {}

}  // namespace

TORCH_LIBRARY(deepsea_moe, m) {
  m.def("dma_probe(Tensor w, Tensor meta) -> Tensor");
  m.def("mxu_probe(Tensor x, Tensor w, Tensor(a!) out, Tensor meta) -> Tensor");
  m.def(
      "expert_ffn(Tensor x, Tensor w13, Tensor w2, Tensor xs_b, "
      "Tensor rw_b, Tensor s13_b, Tensor s2_b, Tensor meta, Tensor(a!) out) "
      "-> Tensor");
  m.def(
      "expert_ffn_out(Tensor x, Tensor w13, Tensor w2, Tensor xs_b, "
      "Tensor rw_b, Tensor s13_b, Tensor s2_b, Tensor meta, Tensor(a!) out) "
      "-> ()");
  m.def(
      "expert_ffn_mxuonly(Tensor x, Tensor w13, Tensor w2, Tensor xs_b, "
      "Tensor rw_b, Tensor s13_b, Tensor s2_b, Tensor meta, Tensor(a!) out) "
      "-> Tensor");
  m.def("describe(Tensor t) -> str");
  m.def("await_dispatch(Tensor t) -> ()");
}

TORCH_LIBRARY_IMPL(deepsea_moe, PrivateUse1, m) {
  m.impl("dma_probe", &DmaProbe);
  m.impl("mxu_probe", &MxuProbe);
  m.impl("expert_ffn", &ExpertFfn);
  m.impl("expert_ffn_out", &ExpertFfnOut);
  m.impl("expert_ffn_mxuonly", &ExpertFfnMxuOnly);
  m.impl("describe", &Describe);
  m.impl("await_dispatch", &AwaitDispatch);
}

TORCH_LIBRARY_IMPL(deepsea_moe, Meta, m) {
  m.impl("dma_probe", &DmaProbeMeta);
  m.impl("mxu_probe", &MxuProbeMeta);
  m.impl("expert_ffn", &FusedExpertFfnMeta);
  m.impl("expert_ffn_out", &FusedExpertFfnOutMeta);
  m.impl("expert_ffn_mxuonly", &FusedExpertFfnMeta);
  m.impl("await_dispatch", &AwaitDispatchMeta);
}

}  // namespace moe
}  // namespace deepsea::torch_bridge

// =============================================================================
// 4. Python C-Extension Entry Point (`import deepsea_torch_bridge`)
// =============================================================================

namespace {

PyMethodDef kModuleMethods[] = {
    {nullptr, nullptr, 0, nullptr},
};

struct PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "deepsea_torch_bridge",
    "Zero-copy TorchTPU <-> Deepsea TPU-C MoE custom operator bridge.",
    -1,
    kModuleMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_deepsea_torch_bridge(void) {
  PyObject* m = PyModule_Create(&kModuleDef);
  if (m == nullptr) {
    return nullptr;
  }
  PyObject* op_name = PyUnicode_FromString("expert_ffn");
  if (op_name != nullptr) {
    PyObject* ops = PyTuple_Pack(1, op_name);
    Py_DECREF(op_name);
    if (ops != nullptr) {
      PyModule_AddObject(m, "ops", ops);
    }
  }
  return m;
}
