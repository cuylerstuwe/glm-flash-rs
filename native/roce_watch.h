// Native implementation portions Copyright 2026 Jay Leaton, Apache-2.0.
// See LICENSES/Apache-2.0.txt and LICENSES/Native-Components-NOTICE.txt.
// The host's view of a failed RoCE wait, taken within microseconds of the failure.
//
// Reading the peer's flag only during a later health check can confuse a late
// peer with a GPU-visibility failure: a flag may arrive after the kernel times out.
//
// ``FailWatch::poll`` runs on the busy-spinning proxy thread, once a loop
// iteration: one acquire load of the failure word, which shares the doorbell's cache line. When the word turns 1 it
// records, at once, the flag word the kernel waited for as host memory holds it and the kernel's last read of it
// (``CTRL_ERR_SEEN``); then it keeps watching that word and records how long after the failure it arrived, if it does.
// Then:
//   host_at_fail == seq, gpu_seen != seq  -> the flag WAS in host memory while the GPU read a stale value:
//                                            a real visibility problem
//   host_at_fail != seq, arrives later    -> the peer was late (no visibility problem)
//   host_at_fail != seq, never arrives    -> never sent
// This watcher has no CUDA, Torch, or verbs dependency.
#pragma once

#include <stdint.h>
#include <time.h>

#include <atomic>

namespace tfroce {

// 32-bit words of the control record that the watch reads (the full list: roce_common.h)
enum : int {
    WATCH_CTRL_ERR_SEQ = 2,
    WATCH_CTRL_ERR_PEER = 3,
    WATCH_CTRL_ERR_HCA = 6,
    WATCH_CTRL_FAILED = 7,
    WATCH_CTRL_ERR_SEEN = 9,
};

inline uint64_t watch_now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}

struct FailWatch {
    enum : uint32_t { IDLE = 0, NOTICED = 1, ARRIVED = 2 };

    // Written by the polling thread; control readers acquire ``state`` before reading the report.
    std::atomic<uint32_t> state{IDLE};
    uint32_t seq = 0, peer = 0, hca = 0;
    uint32_t host_at_fail = 0;       // the flag word in host memory when the failure word was first seen
    uint32_t gpu_seen = 0;           // the kernel's last read of it
    uint64_t noticed_ns = 0;         // CLOCK_MONOTONIC then
    std::atomic<uint64_t> late_ns{0};  // ARRIVED: from noticing to the flag reading ``seq``
    const volatile uint32_t *flag = nullptr;

    // ``ctrl``: the control record; ``flag_at(peer, slot, hca)``: the flag word's host address
    template <class FlagAt>
    void poll(const volatile uint32_t *ctrl, FlagAt flag_at) {
        const uint32_t s = state.load(std::memory_order_relaxed);
        if (s == IDLE) {
            if (__atomic_load_n(const_cast<const uint32_t *>(&ctrl[WATCH_CTRL_FAILED]), __ATOMIC_ACQUIRE) == 0u) {
                return;
            }
            // the kernel wrote the error words, then a system fence, then the failure word: all visible now
            seq = ctrl[WATCH_CTRL_ERR_SEQ];
            peer = ctrl[WATCH_CTRL_ERR_PEER];
            hca = ctrl[WATCH_CTRL_ERR_HCA];
            gpu_seen = ctrl[WATCH_CTRL_ERR_SEEN];
            flag = flag_at(peer, seq & 1u, hca);
            host_at_fail = flag != nullptr ? *flag : 0u;
            noticed_ns = watch_now_ns();
            state.store(NOTICED, std::memory_order_release);
            return;
        }
        if (s == NOTICED && flag != nullptr && host_at_fail != seq && *flag == seq) {
            late_ns.store(watch_now_ns() - noticed_ns, std::memory_order_relaxed);
            state.store(ARRIVED, std::memory_order_release);
        }
    }

    // [state, seq, peer, hca, host_at_fail, gpu_seen, late_ns] (state 0: nothing noticed yet)
    void read(uint64_t out[7]) const {
        out[0] = state.load(std::memory_order_acquire);
        out[1] = seq;
        out[2] = peer;
        out[3] = hca;
        out[4] = host_at_fail;
        out[5] = gpu_seen;
        out[6] = late_ns.load(std::memory_order_relaxed);
    }
};

}  // namespace tfroce
