// Native implementation portions Copyright 2026 Jay Leaton, Apache-2.0.
// See LICENSES/Apache-2.0.txt and LICENSES/Native-Components-NOTICE.txt.
// One-shot RoCE all-gather kernel for GLM-5.3-Flash's tensor parallelism.
//
// Adapted from b12x's RoCEnante all-gather (b12x/comm/roce/_allgather_cute.py and _cute_intrinsics.py,
// https://github.com/local-inference-lab/b12x, Copyright 2026 Luke Alonso and the b12x contributors, Apache
// License 2.0; see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md). Changes: rewritten from CuTe DSL into
// CUDA C++; only the dim-0 concatenation (NCCL's all-gather layout); any input/output alignment and any byte
// count (the send slot is padded to 16 bytes, only nbytes are copied out); a wall-clock timeout (%globaltimer)
// instead of a poll count; a host-visible completion word and wrap-safe failure flag; the poison word
// read once per block (no divergent early exit around a barrier).
//
// Protocol (one op, sequence seq = epoch + 1, slot = seq & 1; see roce_native.cpp for the pinned region's layout):
//   1. every block stages its share of the input into the pinned send slot;
//   2. the last block to finish staging publishes the padded byte count and rings the doorbell (ctrl[0] = seq)
//      that the host proxy thread polls; the proxy RDMA-writes the slot to the peer, then a 4-byte seq flag on
//      the same reliable-connected queue pair (so the flag lands after the payload), striped over the HCAs;
//   3. one thread per (peer, HCA) spins on the peer's flag in this rank's pinned region with system-scope
//      acquire loads, until it reads seq or the timeout expires (then: error words + poison, fail-stop);
//   4. every block copies its share of every rank's shard into the output, rank order (the local shard from the
//      input, the peer's from the NIC-written receive slot with system-scope loads);
//   5. the last block to finish advances the device-resident epoch (so a replayed CUDA graph carries on with
//      the next sequence) and mirrors it into the host-visible completion word.
// Data are only moved, never combined: the gathered bytes equal NCCL's.
//
// a timed-out wait claims the failure record with a compare-and-swap on the device poison word (up
// to blocks x HCAs threads may time out; one writes, so the record is never torn), stores what it last read of the
// flag (CTRL_ERR_SEEN) and only then the failure word; the epoch is published only if no wait of this launch failed
// (the device poison, not the host word). The host's view of the same flag at that moment is taken by the proxy
// thread (roce_watch.h), which tells "arrived late" from "delivered but not seen".

#include <cuda_runtime.h>
#include <stdint.h>

#include "roce_common.h"

namespace tfroce {

__device__ __forceinline__ uint32_t ld_relaxed_gpu(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys_u8(const uint8_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_gpu(uint32_t *p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atom_add_relaxed_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void fence_sc_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

// (``lean``): the release / acquire forms of the same orderings
__device__ __forceinline__ void st_release_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t atom_add_acq_rel_sys(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ uint32_t atom_add_acq_rel_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

__device__ __forceinline__ void trace_at(const GatherArgs &a, uint32_t seq, int k, uint64_t t) {
    if (a.trace != nullptr) {
        a.trace[static_cast<uint64_t>(seq & a.trace_mask) * 4u + static_cast<uint32_t>(k)] = t;
    }
}

__device__ __forceinline__ void fence_sc_gpu() { asm volatile("fence.sc.gpu;" ::: "memory"); }

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t atom_cas_gpu(uint32_t *p, uint32_t cmp, uint32_t v) {
    uint32_t old;
    asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;" : "=r"(old) : "l"(p), "r"(cmp), "r"(v) : "memory");
    return old;
}

// Spin until *flag == want (system scope, acquire). false after timeout_ns of wall-clock time without a match;
// ``seen``: the last value read (the failure record keeps it).
__device__ __noinline__ bool wait_flag(const uint32_t *flag, uint32_t want, uint64_t timeout_ns, uint32_t *seen) {
    const uint64_t t0 = globaltimer();
    uint32_t polls = 0;
    while (true) {
        const uint32_t v = ld_acquire_sys(flag);
        if (v == want) {
            *seen = v;
            return true;
        }
        if ((++polls & 1023u) == 0u && globaltimer() - t0 > timeout_ns) {
            *seen = ld_acquire_sys(flag);          // one last look
            return *seen == want;
        }
    }
}

// dst[i] <- src[i] for i in [0, n), a grid-stride loop; vector width chosen from the alignment of both pointers and n.
// ``sys_src``: the source is NIC-written pinned memory, read with system-scope loads.
template <bool SysSrc>
__device__ __forceinline__ void copy_bytes(uint8_t *dst, const uint8_t *src, uint64_t n, uint64_t index,
                                           uint64_t stride) {
    const uint64_t a = reinterpret_cast<uint64_t>(dst) | reinterpret_cast<uint64_t>(src) | n;
    if ((a & 15u) == 0u) {
        const uint64_t packs = n >> 4;
        for (uint64_t i = index; i < packs; i += stride) {
            uint4 v;
            if (SysSrc) {
                v = ld_relaxed_sys_v4(src + (i << 4));
            } else {
                v = reinterpret_cast<const uint4 *>(src)[i];
            }
            reinterpret_cast<uint4 *>(dst)[i] = v;
        }
    } else if ((a & 3u) == 0u) {
        const uint64_t words = n >> 2;
        for (uint64_t i = index; i < words; i += stride) {
            uint32_t v;
            if (SysSrc) {
                v = ld_relaxed_sys(reinterpret_cast<const uint32_t *>(src) + i);
            } else {
                v = reinterpret_cast<const uint32_t *>(src)[i];
            }
            reinterpret_cast<uint32_t *>(dst)[i] = v;
        }
    } else {
        for (uint64_t i = index; i < n; i += stride) {
            uint32_t v;
            if (SysSrc) {
                v = ld_relaxed_sys_u8(src + i);
            } else {
                v = src[i];
            }
            dst[i] = static_cast<uint8_t>(v);
        }
    }
}

// every peer's 16-byte packs of a thread's grid-stride indices (two at a time), all loads issued before
// any store, so the system-scope reads of NIC-written memory overlap instead of each waiting on the previous store.
// The same bytes to the same addresses as the rank-order loop below.
__device__ __forceinline__ void copy_peers_v4(const GatherArgs &a, uint64_t slot, uint64_t index, uint64_t stride) {
    constexpr int MAXW = 8;
    const uint64_t packs = a.nbytes >> 4;
    for (uint64_t i = index; i < packs; i += 2 * stride) {
        const uint64_t j = i + stride;
        uint4 v0[MAXW], v1[MAXW];
#pragma unroll
        for (int s = 0; s < MAXW; s++) {
            if (s < a.world && s != a.rank) {
                const uint8_t *ps = reinterpret_cast<const uint8_t *>(
                    a.recv_base + (static_cast<uint64_t>(s) * a.slots + slot) * a.slot_bytes);
                v0[s] = ld_relaxed_sys_v4(ps + (i << 4));
                if (j < packs) {
                    v1[s] = ld_relaxed_sys_v4(ps + (j << 4));
                }
            }
        }
#pragma unroll
        for (int s = 0; s < MAXW; s++) {
            if (s < a.world && s != a.rank) {
                uint4 *dst = reinterpret_cast<uint4 *>(a.out + static_cast<uint64_t>(s) * a.nbytes);
                dst[i] = v0[s];
                if (j < packs) {
                    dst[j] = v1[s];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(1024) gather_kernel(GatherArgs a) {
    __shared__ uint32_t s_flag;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;
    uint32_t *ctrl = reinterpret_cast<uint32_t *>(a.ctrl_base);

    // A recorded timeout poisons the runtime: later launches do nothing (the host raises on its next check).
    // One read per block, so no thread of the block leaves before a barrier the others wait at.
    if (tid == 0) {
        s_flag = ld_relaxed_gpu(a.poison);
    }
    __syncthreads();
    if (s_flag != 0u) {
        return;
    }
    const uint32_t seq = ld_relaxed_gpu(a.epoch) + 1u;
    const uint64_t slot = seq & 1u;
    uint8_t *send = reinterpret_cast<uint8_t *>(a.send_base + slot * a.slot_bytes);
    const uint32_t padded = static_cast<uint32_t>((a.nbytes + 15u) & ~static_cast<uint64_t>(15u));
    if (blockIdx.x == 0 && tid == 0) {
        trace_at(a, seq, 0, globaltimer());               // (no-op without a trace ring)
    }

    // with copy2, this rank's own shard goes to the output now (off the critical path: the peers are
    // still on their way) unless the input overlaps the output, and the peers' packs leave in one batch of loads
    const uint64_t out_end = reinterpret_cast<uint64_t>(a.out) + static_cast<uint64_t>(a.world) * a.nbytes;
    const uint64_t in_lo = reinterpret_cast<uint64_t>(a.in);
    const bool early_own = a.copy2 != 0u && (in_lo + a.nbytes <= reinterpret_cast<uint64_t>(a.out) || in_lo >= out_end);
    const bool batched = a.copy2 != 0u &&
        ((reinterpret_cast<uint64_t>(a.out) | a.recv_base | a.slot_bytes | a.nbytes) & 15u) == 0u;

    // 1. stage this rank's shard into the pinned send slot
    copy_bytes<false>(send, a.in, a.nbytes, index, stride);
    if (early_own) {
        copy_bytes<false>(a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in, a.nbytes, index, stride);
    }
    __syncthreads();

    // 2. the last block to finish staging rings the proxy's doorbell (one system fence per block after the
    //    barrier orders the block's staging stores before its arrival)
    if (tid == 0) {
        if (a.lean) {
            // the same message-passing order with release / acquire instead of two sequentially
            // consistent system fences: a block's arrival is a release RMW (the barrier made the block's staging
            // stores precede it); the last arrival also acquires every earlier block's; the doorbell is a release
            // store, so the byte counts and every staged byte are visible to the proxy's acquire load of it. One
            // block (a 16 KiB shard): no arrival at all, one release store.
            const bool last = gdim == 1u || ((atom_add_acq_rel_sys(a.stage_ctr, 1u) + 1u) & (gdim - 1u)) == 0u;
            if (last) {
                st_relaxed_sys(ctrl + CTRL_NBYTES, padded);
                st_relaxed_sys(ctrl + CTRL_SLOT_NBYTES + slot, padded);
                st_release_sys(ctrl + CTRL_SEQ, seq);
                trace_at(a, seq, 1, globaltimer());
            }
        } else {
            fence_sc_sys();
            const uint32_t prior = atom_add_relaxed_gpu(a.stage_ctr, 1u);
            if (((prior + 1u) & (gdim - 1u)) == 0u) {
                st_relaxed_sys(ctrl + CTRL_NBYTES, padded);
                st_relaxed_sys(ctrl + CTRL_SLOT_NBYTES + slot, padded);
                fence_sc_sys();
                st_relaxed_sys(ctrl + CTRL_SEQ, seq);
                trace_at(a, seq, 1, globaltimer());
            }
        }
    }

    // 3. wait for every peer's per-HCA flag of this sequence (only the one HCA a small op used)
    const bool one = one_hca(padded, a.stripe_min, a.n_hca);
    const int used = one ? single_hca(seq, a.n_hca) : -1;
    if (tid < static_cast<uint32_t>(a.world * a.n_hca)) {
        const int peer = static_cast<int>(tid) / a.n_hca;
        const int hca = static_cast<int>(tid) - peer * a.n_hca;
        if (peer != a.rank && (used < 0 || hca == used)) {
            const uint32_t *flag = reinterpret_cast<const uint32_t *>(
                a.flag_base + ((static_cast<uint64_t>(peer) * a.slots + slot) * a.n_hca + hca) * a.flag_stride);
            uint32_t seen = 0u;
            if (!wait_flag(flag, seq, a.timeout_ns, &seen) && atom_cas_gpu(a.poison, 0u, 1u) == 0u) {
                // the first timed-out wait of the runtime writes the record; the failure word goes last, after a
                // system fence, so a host that sees it reads a whole record (roce_watch.h, Runtime.check)
                st_relaxed_sys(ctrl + CTRL_ERR_PEER, static_cast<uint32_t>(peer));
                st_relaxed_sys(ctrl + CTRL_ERR_HCA, static_cast<uint32_t>(hca));
                st_relaxed_sys(ctrl + CTRL_ERR_SEQ, seq);
                st_relaxed_sys(ctrl + CTRL_ERR_SEEN, seen);
                fence_sc_sys();
                st_relaxed_sys(ctrl + CTRL_FAILED, 1u);
                fence_sc_sys();
            }
            if (a.trace != nullptr && blockIdx.x == 0) {
                atomicMax(reinterpret_cast<unsigned long long *>(a.trace + static_cast<uint64_t>(seq & a.trace_mask) * 4u + 2u),
                          static_cast<unsigned long long>(globaltimer()));
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        s_flag = ld_relaxed_gpu(a.poison);
    }
    __syncthreads();

    // 4. every rank's shard, in rank order (a timed-out wait leaves the peer slot unreliable: nothing is copied)
    if (s_flag == 0u && batched) {
        if (!early_own) {
            copy_bytes<false>(a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in, a.nbytes, index, stride);
        }
        copy_peers_v4(a, slot, index, stride);
    } else if (s_flag == 0u) {
        for (int src = 0; src < a.world; src++) {
            if (src == a.rank && early_own) {
                continue;
            }
            uint8_t *dst = a.out + static_cast<uint64_t>(src) * a.nbytes;
            if (src == a.rank) {
                copy_bytes<false>(dst, a.in, a.nbytes, index, stride);
            } else {
                const uint8_t *peer_slot = reinterpret_cast<const uint8_t *>(
                    a.recv_base + (static_cast<uint64_t>(src) * a.slots + slot) * a.slot_bytes);
                copy_bytes<true>(dst, peer_slot, a.nbytes, index, stride);
            }
        }
    }

    // 5. the last block to finish publishes the next epoch (not after a failure: later launches stay no-ops)
    if (a.lean) {
        // the next launch reads the epoch after this kernel ended (stream order), so the copy-out needs
        // no fence before it; the arrival is an acq_rel RMW so the last block sees every block's poison CAS
        __syncthreads();
        if (tid == 0) {
            const bool last = gdim == 1u || ((atom_add_acq_rel_gpu(a.tail_ctr, 1u) + 1u) & (gdim - 1u)) == 0u;
            if (last && ld_relaxed_gpu(a.poison) == 0u && ld_relaxed_sys(ctrl + CTRL_FAILED) == 0u) {
                st_release_gpu(a.epoch, seq);
                st_relaxed_sys(ctrl + CTRL_COMPLETED, seq);
            }
            if (last) {
                trace_at(a, seq, 3, globaltimer());
            }
        }
        return;
    }
    fence_sc_gpu();
    __syncthreads();
    if (tid == 0) {
        const uint32_t prior = atom_add_relaxed_gpu(a.tail_ctr, 1u);
        if (((prior + 1u) & (gdim - 1u)) == 0u) {
            fence_sc_gpu();
            if (ld_relaxed_gpu(a.poison) == 0u && ld_relaxed_sys(ctrl + CTRL_FAILED) == 0u) {
                st_release_gpu(a.epoch, seq);
                st_relaxed_sys(ctrl + CTRL_COMPLETED, seq);
            }
            trace_at(a, seq, 3, globaltimer());
        }
    }
}

cudaError_t launch_gather(const GatherArgs &args, int grid, int threads, cudaStream_t stream) {
    gather_kernel<<<grid, threads, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace tfroce
