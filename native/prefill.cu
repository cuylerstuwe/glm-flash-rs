// Rust prefill support. FP8 weight dequantization into a BF16 copy:
// BF16(FP32(e4m3 weight) * FP32(stored BF16 scale)).
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

extern "C" __global__ void f8_b16_copy(const uint8_t* w, const __nv_bfloat16* scale,
                                      __nv_bfloat16* out, uint64_t count, int k) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint8_t bits = w[i];
    unsigned e = (bits >> 3) & 15, m = bits & 7;
    float x = e == 0 ? float(m) * 0.001953125f : ldexpf(float(8 + m), int(e) - 10);
    if (e == 15 && m == 7) x = __int_as_float(0x7fc00000);
    if (bits & 128) x = -x;
    out[i] = __float2bfloat16_rn(x * __bfloat162float(scale[i / k]));
}

extern "C" __global__ void cast_bf16(const float* x, __nv_bfloat16* y, uint64_t n) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

extern "C" __global__ void shift_conv(const __nv_bfloat16* p, __nv_bfloat16* state,
                                      int rows, int stride, int channels) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;
    // Each thread owns one channel and saves the old rows before overwriting.
    __nv_bfloat16 old[3] = {state[ch], state[channels + ch], state[2 * channels + ch]};
    for (int r = 0; r < 3; ++r) {
        int source = rows + r - 3;
        state[r * channels + ch] = source < 0 ? old[source + 3] : p[source * stride + ch];
    }
}
