// Native implementation portions Copyright 2026 Jay Leaton, Apache-2.0.
// See LICENSES/Apache-2.0.txt and LICENSES/Native-Components-NOTICE.txt.
// Shared by roce_native.cpp (host proxy, bindings) and roce.cu (kernel): the control record and the kernel arguments.
// The protocol is adapted from b12x's RoCEnante (Apache-2.0, see roce.cu).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace tfroce {

// 32-bit words of the control record (one FLAG_STRIDE block at the end of the pinned region). The kernel writes all
// of them but CTRL_SEQ's reader (the proxy); the host reads them without touching the device.
enum : int {
    CTRL_SEQ = 0,          // doorbell: the newest sequence staged (proxy polls it)
    CTRL_NBYTES = 1,       // padded bytes of the newest op
    CTRL_ERR_SEQ = 2,      // sequence whose wait timed out
    CTRL_ERR_PEER = 3,     // the peer it waited for
    CTRL_SLOT_NBYTES = 4,  // [4], [5]: padded bytes per slot (the proxy's catch-up after a missed doorbell)
    CTRL_ERR_HCA = 6,      // the HCA whose flag never came
    CTRL_FAILED = 7,       // 1 once any wait timed out (wrap-safe, unlike a sequence number)
    CTRL_COMPLETED = 8,    // the last sequence completed on this rank (doorbell ahead of it: an op is in flight)
    CTRL_ERR_SEEN = 9,     // the failed wait's last read of the flag (what the GPU observed)
    CTRL_WORDS = 10,
};

struct GatherArgs {
    const uint8_t *in;
    uint8_t *out;
    uint64_t nbytes;       // bytes of one rank's shard
    uint64_t recv_base;    // device addresses of the pinned region's parts
    uint64_t flag_base;
    uint64_t send_base;
    uint64_t ctrl_base;
    uint64_t slot_bytes;
    uint32_t *epoch;       // device counters: epoch, this grid size's stage / tail arrivals, poison
    uint32_t *stage_ctr;
    uint32_t *tail_ctr;
    uint32_t *poison;
    uint64_t timeout_ns;
    int world;
    int rank;
    int n_hca;
    int flag_stride;
    int slots;
    // (all disabled by default)
    uint32_t stripe_min;   // padded shards below this many bytes travel on ONE HCA (seq % n_hca), else striped
    uint32_t lean;         // 1: release / acquire ordering instead of the sequentially consistent fences (same order)
    uint64_t *trace;       // optional ring of 4 x u64 per op (globaltimer ns): start, doorbell, flags seen, end
    uint32_t trace_mask;   // ring entries - 1 (a power of two minus one)
    // this rank's shard is copied out while the peers' flags are awaited, and each
    // thread issues every peer's loads before its stores (the same bytes to the same places)
    uint32_t copy2;
};

// the HCAs an op of ``padded`` bytes uses -- ALL when striped, else only ``single_hca``. The kernel
// (which flags to wait for), the proxy (which queue pairs to post on) and the test loop evaluate the same function of
// (padded bytes, seq), so both ranks agree without exchanging anything.
__host__ __device__ inline bool one_hca(uint32_t padded, uint32_t stripe_min, int n_hca) {
    return stripe_min != 0u && padded < stripe_min && n_hca > 1;
}
__host__ __device__ inline int single_hca(uint32_t seq, int n_hca) { return static_cast<int>(seq % uint32_t(n_hca)); }

cudaError_t launch_gather(const GatherArgs &args, int grid, int threads, cudaStream_t stream);

}  // namespace tfroce
