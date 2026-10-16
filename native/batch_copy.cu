// Pure byte movement for the speculative verifier's per-round bookkeeping.
// Pointer tables arrive by value as kernel parameters, so a launch carries its
// own table (no staging lifetime) and is valid inside CUDA graph capture.
// Every source/destination/size is a multiple of 16 bytes (checked by the owner).
#include <stdint.h>

#define INDEX_LAYERS 16
struct IndexUndo {
    unsigned long long ring_keys[INDEX_LAYERS];
    unsigned long long ring_gates[INDEX_LAYERS];
    unsigned long long undo_keys[INDEX_LAYERS];
    unsigned long long undo_gates[INDEX_LAYERS];
};
// grid (rows - first, layers, 2): z selects keys/gates. restore != 0 copies
// undo row r back to ring row (position + r) % ring_rows; otherwise saves it.
extern "C" __global__ void index_undo(IndexUndo t, unsigned position, unsigned first,
                                      unsigned ring_rows, unsigned row_bytes,
                                      unsigned restore) {
    unsigned r = first + blockIdx.x, l = blockIdx.y;
    bool gate = blockIdx.z != 0;
    unsigned at = (position + r) % ring_rows;
    char* ring = (char*)(gate ? t.ring_gates[l] : t.ring_keys[l]) + size_t(at) * row_bytes;
    char* undo = (char*)(gate ? t.undo_gates[l] : t.undo_keys[l]) + size_t(r) * row_bytes;
    const uint4* src = (const uint4*)(restore ? undo : ring);
    uint4* dst = (uint4*)(restore ? ring : undo);
    for (unsigned i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) dst[i] = src[i];
}

#define KDA_LAYERS 34
struct KdaCommit {
    unsigned long long conv[KDA_LAYERS];     // retained 3-row tail (in/out)
    unsigned long long proj[KDA_LAYERS];     // the window's projected rows
    unsigned long long tail[KDA_LAYERS];     // committed-tail scratch
    unsigned long long state_src[KDA_LAYERS];
    unsigned long long state_dst[KDA_LAYERS];
};
// grid (1 + state blocks, layers). Block x == 0 rebuilds the tail exactly as the
// three ordered row copies did (row r = row keep + r of [tail; projected rows])
// into the scratch, then copies the scratch over the retained tail. Blocks
// x >= 1 copy the recurrent state.
extern "C" __global__ void kda_commit(KdaCommit t, unsigned keep, unsigned row_bytes,
                                      unsigned proj_row_bytes, unsigned state_bytes) {
    unsigned l = blockIdx.y;
    if (blockIdx.x == 0) {
        for (unsigned row = 0; row < 3; row++) {
            unsigned at = keep + row;
            const char* from = at < 3 ? (const char*)t.conv[l] + size_t(at) * row_bytes
                                      : (const char*)t.proj[l] + size_t(at - 3) * proj_row_bytes;
            const uint4* src = (const uint4*)from;
            uint4* dst = (uint4*)((char*)t.tail[l] + size_t(row) * row_bytes);
            for (unsigned i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) dst[i] = src[i];
        }
        __syncthreads();
        const uint4* src = (const uint4*)t.tail[l];
        uint4* dst = (uint4*)t.conv[l];
        for (unsigned i = threadIdx.x; i < 3 * row_bytes / 16; i += blockDim.x) dst[i] = src[i];
        return;
    }
    const uint4* src = (const uint4*)t.state_src[l];
    uint4* dst = (uint4*)t.state_dst[l];
    unsigned n = state_bytes / 16, stride = (gridDim.x - 1) * blockDim.x;
    for (unsigned i = (blockIdx.x - 1) * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}
