// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Persistent Host gather worker for one channel.
//
// The device publishes a request sequence only after its index D2H completes.
// The worker gathers 256 B rows from a CPU table into pinned staging,
// optionally uploads them to a retained HBM buffer, and only then publishes the
// response.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <thread>

// Defined in buffer.cc: a blocking PJRT raw H2D into the retained HBM buffer.
// `submitted` receives the time at which the copy was enqueued.
extern "C" bool CopyRawChannelOutput(void *context, const void *source,
                                     int64_t bytes, int64_t *submitted);

namespace {
constexpr int64_t kWidth = 64; // int32 words per 256 B row
constexpr int64_t kRing = 1 << 14;

int64_t Now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Steady-clock nanoseconds for one handled request.
struct Timing {
  int64_t sequence, seen, gathered, submitted, uploaded, published;
};

struct Channel {
  const int32_t *table;
  int64_t rows;
  const int32_t *ids;
  const int32_t *request;
  int32_t *response;
  int32_t *staging;
  int64_t count;
  void *destination; // Null keeps the rows in pinned Host staging.
  std::atomic<int> max_delay_us{0};
  std::atomic<bool> running{true};
  std::atomic<int64_t> handled{0};
  Timing timings[kRing] = {};
  std::thread worker;

  void Run() {
    std::minstd_rand random(static_cast<uint32_t>(rows ^ count));
    int32_t last = 0;
    while (running.load(std::memory_order_relaxed)) {
      const int32_t sequence = __atomic_load_n(request, __ATOMIC_ACQUIRE);
      if (sequence == last) {
        std::this_thread::yield();
        continue;
      }
      last = sequence;
      const int64_t seen = Now();

      // Validation mode sleeps mid-gather so an early device read sees
      // partial rows.
      const int delay = max_delay_us.load(std::memory_order_relaxed);
      bool valid = true;
      for (int64_t i = 0; valid && i < count; ++i) {
        if (i + 16 < count && ids[i + 16] >= 0 && ids[i + 16] < rows)
          for (int64_t word = 0; word < kWidth; word += 16)
            __builtin_prefetch(table + ids[i + 16] * kWidth + word, 0, 0);
        const int64_t row = ids[i];
        valid = row >= 0 && row < rows;
        if (valid)
          std::memcpy(staging + i * kWidth, table + row * kWidth,
                      kWidth * sizeof(int32_t));
        if (i == count / 2 && delay > 0)
          std::this_thread::sleep_for(
              std::chrono::microseconds(random() % delay));
      }
      const int64_t gathered = Now();

      int64_t submitted = gathered;
      if (valid && destination)
        valid = CopyRawChannelOutput(
            destination, staging, count * kWidth * sizeof(int32_t), &submitted);
      const int64_t uploaded = Now();

      // Record before the response so a reader that sees `handled` also sees
      // the record. The device copies rows only after it sees the response.
      const int64_t n = handled.load(std::memory_order_relaxed);
      timings[n % kRing] = {sequence,  seen,     gathered,
                            submitted, uploaded, Now()};
      handled.store(n + 1, std::memory_order_release);
      __atomic_store_n(response, valid ? sequence : -sequence,
                       __ATOMIC_RELEASE);
    }
  }
};
} // namespace

// buffers: table, ids, request, response, staging. The caller keeps them, and
// the destination, alive until StopChannel returns.
extern "C" Channel *StartChannel(const uintptr_t *buffers, int64_t rows,
                                 int64_t count, void *destination) {
  for (int i = 1; i < 5; ++i)
    if (!buffers[i] || buffers[i] % 4096)
      return nullptr;
  auto *channel = new Channel{reinterpret_cast<const int32_t *>(buffers[0]),
                              rows,
                              reinterpret_cast<const int32_t *>(buffers[1]),
                              reinterpret_cast<const int32_t *>(buffers[2]),
                              reinterpret_cast<int32_t *>(buffers[3]),
                              reinterpret_cast<int32_t *>(buffers[4]),
                              count,
                              destination};
  channel->worker = std::thread([channel] { channel->Run(); });
  return channel;
}

extern "C" void StopChannel(Channel *channel) {
  channel->running.store(false);
  channel->worker.join();
  delete channel;
}

extern "C" void SetChannelDelay(Channel *channel, int max_delay_us) {
  channel->max_delay_us.store(max_delay_us);
}

// Returns the number of handled requests; `out` receives the last `n` records
// as six int64 each, in Timing's field order.
extern "C" int64_t ChannelTimings(Channel *channel, int64_t *out, int64_t n) {
  const int64_t handled = channel->handled.load(std::memory_order_acquire);
  n = std::min({n, handled, kRing});
  for (int64_t i = 0; i < n; ++i) {
    const Timing &t = channel->timings[(handled - n + i) % kRing];
    int64_t *row = out + i * 6;
    row[0] = t.sequence, row[1] = t.seen, row[2] = t.gathered;
    row[3] = t.submitted, row[4] = t.uploaded, row[5] = t.published;
  }
  return handled;
}

// A CPU-only table whose word w holds w ^ salt, so checks need no copy.
extern "C" int32_t *AllocateTable(int64_t rows, uint32_t salt) {
  const size_t bytes = rows * kWidth * sizeof(int32_t);
  void *memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED)
    return nullptr;
  auto *words = static_cast<uint32_t *>(memory);
  for (size_t i = 0; i < bytes / sizeof(int32_t); ++i)
    words[i] = static_cast<uint32_t>(i) ^ salt;
  return static_cast<int32_t *>(memory);
}

extern "C" void FreeTable(int32_t *table, int64_t rows) {
  munmap(table, rows * kWidth * sizeof(int32_t));
}
