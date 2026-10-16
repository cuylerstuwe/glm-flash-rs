// Image rows replace the embedded prompt rows at their virtual-ID positions, in
// all four hyper-connection copies.
#include <stdint.h>
constexpr int D = 4096, COPIES = 4;
// map: (row in slab, table row) pairs; only rows in [lo, hi) were embedded here.
extern "C" __global__ void vision_rows(uint16_t* x, const int* map, int count,
    const uint16_t* rows, int lo, int hi) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const int r = map[2 * i], k = map[2 * i + 1];
    if (r < lo || r >= hi) return;
    const uint4* src = reinterpret_cast<const uint4*>(rows + (size_t)k * D);
    for (int c = 0; c < COPIES; ++c) {
        uint4* dst = reinterpret_cast<uint4*>(x + ((size_t)r * COPIES + c) * D);
        for (int j = threadIdx.x; j < D / 8; j += blockDim.x) dst[j] = src[j];
    }
}
