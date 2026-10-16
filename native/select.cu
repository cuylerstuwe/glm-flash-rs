// Byte-exact greedy selection from BF16 logits. No target arithmetic changes.
// Order by value descending, then token ID ascending. Signed zeros are
// numerically equal. All nonfinite values fail closed.
#include <stdint.h>

__device__ __forceinline__ void local_best(const uint16_t *logits, int base, int count,
                                           int *out) {
    __shared__ int keys[256], ids[256], bad[256];
    const int t = threadIdx.x;
    int key = -32768, id = 2147483647, invalid = 0;
    for (int i = t; i < count; i += 256) {
        const uint16_t bits = logits[i];
        const int k = (bits & 0x7fff) == 0 ? 0 : int(int16_t(bits ^ ((bits & 0x8000) ? 0x7fff : 0)));
        invalid |= (bits & 0x7f80) == 0x7f80;
        if (k > key || (k == key && base + i < id)) {
            key = k; id = base + i;
        }
    }
    keys[t] = key; ids[t] = id; bad[t] = invalid;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (t < stride) {
            const int k = keys[t + stride], i = ids[t + stride];
            if (k > keys[t] || (k == keys[t] && i < ids[t])) {
                keys[t] = k; ids[t] = i;
            }
            bad[t] |= bad[t + stride];
        }
        __syncthreads();
    }
    if (!t) { out[0] = keys[0]; out[1] = ids[0]; out[2] = bad[0]; out[3] = 0; }
}

// `stride` ints between one rank's candidate records (4 for a single row).
__device__ __forceinline__ void merge_best(const int *candidates, int stride, int *result) {
    int key = -32768, id = 2147483647, invalid = 0;
    for (int rank = 0; rank < 4; ++rank) {
        const int k = candidates[stride * rank], i = candidates[stride * rank + 1];
        if (k > key || (k == key && i < id)) { key = k; id = i; }
        invalid |= candidates[stride * rank + 2];
    }
    result[0] = id; result[1] = invalid;
}

extern "C" __global__ void greedy_local(const uint16_t *logits, int base,
                                        int count, int *out) {
    local_best(logits, base, count, out);
}

extern "C" __global__ void greedy_merge(const int *candidates, int *result) {
    if (threadIdx.x) return;
    merge_best(candidates, 4, result);
}

// One block per row of a [rows][row_elems] BF16 head; out is [rows][4].
extern "C" __global__ void greedy_local_rows(const uint16_t *logits, int base, int count,
                                             int row_elems, int *out) {
    local_best(logits + size_t(blockIdx.x) * row_elems, base, count, out + 4 * blockIdx.x);
}

// Gathered rank-major [rank][rows][4]; result is [rows][2] (id, invalid).
extern "C" __global__ void greedy_merge_rows(const int *candidates, int rows, int *result) {
    const int row = threadIdx.x;
    if (row >= rows) return;
    merge_best(candidates + 4 * row, 4 * rows, result + 2 * row);
}
