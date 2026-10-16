// Exact stable grouping for the canonical router: eight distinct experts and
// the shared expert per row. Bitmap order is independent of atomic arrival.
#include <stdint.h>
constexpr int E = 289, WORDS = 256, SLOTS = 9;
__device__ __forceinline__ int scan(int x, int* warps) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int d = 1; d < 32; d <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += v;
    }
    if (lane == 31) warps[warp] = x;
    __syncthreads();
    int offset = 0;
    for (int w = 0; w < warp; ++w) offset += warps[w];
    return x + offset;
}
extern "C" __global__ void group_clear(uint32_t* bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * WORDS) bits[i] = 0;
}
extern "C" __global__ void group_mark(const int* picks, uint32_t* bits, int rows) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
        int e = picks[r * SLOTS + s];
        atomicOr(bits + e * WORDS + r / 32, 1u << (r & 31));
    }
}
extern "C" __global__ void group_prefix(const uint32_t* bits, int* prefix, int* counts) {
    __shared__ int warps[8];
    int e = blockIdx.x, t = threadIdx.x;
    int n = __popc(bits[e * WORDS + t]);
    int sum = scan(n, warps);
    prefix[e * WORDS + t] = sum - n;
    if (t == WORDS - 1) counts[e] = sum;
}
extern "C" __global__ void group_ids(const int* counts, int* place, int* ids, int* count) {
    __shared__ int warps[16];
    int e = threadIdx.x;
    int used = e < E && counts[e] > 0;
    int sum = scan(used, warps);
    if (e < E) place[e] = sum - 1;
    if (used) ids[sum - 1] = e;
    if (e == 511) *count = sum;
}
extern "C" __global__ void group_scatter(const int* picks, const uint32_t* bits,
    const int* prefix, const int* counts, const int* place, const int* count, int* members, int maxm) {
    int e = blockIdx.x, t = threadIdx.x;
    if (counts[e]) {
        int u = place[e], off = prefix[e * WORDS + t];
        uint32_t b = bits[e * WORDS + t];
        while (b) {
            int r = t * 32 + __ffs(b) - 1;
            int slot = 0;
            while (picks[r * SLOTS + slot] != e) ++slot;
            members[u * maxm + off++] = r * 32 + slot;
            b &= b - 1;
        }
        for (int j = counts[e] + t; j < maxm; j += WORDS) members[u * maxm + j] = -1;
    }
    // Compact unused rows do not overlap the used rows above.
    if (e >= *count) for (int j = t; j < maxm; j += WORDS) members[e * maxm + j] = -1;
}
// The same ids/count/members for at most 32 rows in one 512-thread block: one
// row mask per expert replaces the bitmap, prefix and scatter launches.
extern "C" __global__ void group_small(const int* picks, int rows, int* ids, int* count,
    int* members, int maxm) {
    __shared__ int warps[16];
    __shared__ int local[32 * SLOTS];
    __shared__ int total;
    const int e = threadIdx.x;
    for (int i = e; i < rows * SLOTS; i += blockDim.x) local[i] = picks[i];
    __syncthreads();
    uint32_t mask = 0;
    if (e < E)
        for (int r = 0; r < rows; ++r)
#pragma unroll
            for (int s = 0; s < SLOTS; ++s)
                if (local[r * SLOTS + s] == e) mask |= 1u << r;
    const int used = mask != 0;
    const int sum = scan(used, warps);
    if (e == 511) total = sum;
    __syncthreads();
    const int n = total;
    if (e == 0) *count = n;
    if (used) {
        const int u = sum - 1;
        ids[u] = e;
        int off = 0;
        for (uint32_t b = mask; b; b &= b - 1) {
            const int r = __ffs(b) - 1;
            int slot = 0;
            while (local[r * SLOTS + slot] != e) ++slot;
            members[u * maxm + off++] = r * 32 + slot;
        }
        for (int j = off; j < maxm; ++j) members[u * maxm + j] = -1;
    }
    if (e < E && e >= n) for (int j = 0; j < maxm; ++j) members[e * maxm + j] = -1;
}
