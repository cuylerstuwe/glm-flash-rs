// GLM-5.3-Flash's EXL3 routed experts on CUDA: the grouped trellis GEMV and the Hadamard rotations around it.
//
// Format (see exl3.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): a 16x16 tile is 32 little-endian
// 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values
// are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile), so a tile goes from memory
// to the tensor cores without a shuffle or a layout change.
//
// Every output depends only on its own row: the rows of a window share an mma tile but mma keeps rows
// independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a
// fixed order (shared memory, then the epilogue kernels), never with atomics.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace alaska_exl3 {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// The work item of both grouped kernels: MG member tiles (16 rows of X each, row codes in rows_sh[16 * MG], -1 =
// none) of expert e times W_q of one matrix over one K split, for the NT column tiles from nt0;
// Z[mat][split][member row][n]. Warp w of W runs a fixed contiguous range of the split's k tiles; warps are added
// in order 0..W-1. Each member tile has its own accumulators and gets, k tile by k tile, the same mma on the same
// decoded B fragments whatever MG, NT and the block the item runs in, so a row's Z depends only on (W, SK) and its
// own inputs: MG > 1 decodes a weight tile once for several member tiles, it does not change a sum.
template <int NT, int W, int MG>
__device__ __forceinline__ void member_tiles(const half* __restrict__ X, const uint32_t* __restrict__ T, int e,
                                             const int* rows_sh, float (*red)[16 * MG][NT * 16], float* __restrict__ Z,
                                             int K, int N, int P, int SK, int split, int mat, int nt0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    int r0[MG], r1[MG];
    const half* x0[MG];
    const half* x1[MG];
#pragma unroll
    for (int m = 0; m < MG; ++m) {
        r0[m] = rows_sh[16 * m + g];
        r1[m] = rows_sh[16 * m + g + 8];
        x0[m] = X + (size_t)(r0[m] < 0 ? 0 : r0[m]) * K + 2 * t;
        x1[m] = X + (size_t)(r1[m] < 0 ? 0 : r1[m]) * K + 2 * t;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[MG][NT][2][4];
#pragma unroll
    for (int m = 0; m < MG; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;

    for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
        uint32_t words[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
        const int k = kt * 16;
        uint32_t a[MG][4];
#pragma unroll
        for (int m = 0; m < MG; ++m) {
            a[m][0] = load_pair(x0[m] + k, r0[m] >= 0);
            a[m][1] = load_pair(x1[m] + k, r1[m] >= 0);
            a[m][2] = load_pair(x0[m] + k + 8, r0[m] >= 0);
            a[m][3] = load_pair(x1[m] + k + 8, r1[m] >= 0);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
#pragma unroll
            for (int m = 0; m < MG; ++m) {
                mma16816(acc[m][i][0], a[m], b0);
                mma16816(acc[m][i][1], a[m], b1);
            }
        }
        tile += (size_t)NTILES * 32;
    }

    // warps' partial sums through shared memory, added in warp order
#pragma unroll
    for (int m = 0; m < MG; ++m)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][16 * m + g][col] = acc[m][i][h][0];
                red[warp][16 * m + g][col + 1] = acc[m][i][h][1];
                red[warp][16 * m + g + 8][col] = acc[m][i][h][2];
                red[warp][16 * m + g + 8][col + 1] = acc[m][i][h][3];
            }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * MG * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
    }
}

// Program (u, n block, (mat * SK + split) * MT + member tile): 16 members of distinct expert u (rows of X, one
// row per member) times W_q of matrix `mat` over this split's K range, for NT columns (``member_tiles``).
template <int NT, int W>
__device__ __forceinline__ void grouped_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int E) {
    const int u = blockIdx.x;
    if (u >= ucount[0] || uids[u] >= E) return;          // the shared expert (id E) is not EXL3
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int e = uids[u];

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    __shared__ float red[W][16][NT * 16];
    member_tiles<NT, W, 1>(X, T, e, rows_sh, red, Z, K, N, P, SK, split, mat, blockIdx.y * NT);
}

// Long windows (prefill chunks): program (n block, u, mat * SK + split) runs every member tile of distinct
// expert u, MG at a time, in one block. Upstream's grid gives each member tile its own program, far apart in launch
// order, so an expert with more than 16 members read its weights from memory once per member tile, and a window
// of R rows launched R/16 programs per (expert, n block, split), almost all empty. Here the weights come from
// memory once (later passes find the block's slice in L2), no program is empty, and the n blocks of one (expert,
// split) run side by side, so they share X in L2. Same work items, same bits (``member_tiles``).
template <int NT, int W, int MG>
__global__ void __launch_bounds__(W * 32) grouped_loop_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int E) {
    const int u = blockIdx.y;
    if (u >= ucount[0] || uids[u] >= E) return;          // the shared expert (id E) is not EXL3
    const int split = blockIdx.z % SK;
    const int mat = blockIdx.z / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int e = uids[u];

    __shared__ int rows_sh[16 * MG];
    __shared__ float red[W][16 * MG][NT * 16];
    for (int m0 = 0; m0 < maxm; m0 += 16 * MG) {
        if (threadIdx.x < 16 * MG) {
            const int m = m0 + threadIdx.x;
            int code = m < maxm ? members[u * maxm + m] : -1;
            rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
        }
        __syncthreads();
        if (rows_sh[0] < 0) break;                        // members come first: no member left (same for all threads)
        member_tiles<NT, W, MG>(X, T, e, rows_sh, red, Z, K, N, P, SK, split, mat, blockIdx.x * NT);
        __syncthreads();                                  // rows_sh and red are rewritten by the next pass
    }
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Program (member row, 128-block of K, matrix): Xh[mat][row][block] = fp16((x[row] * suh[mat][e]) @ H) for the
// gate and up projections of every routed slot (row, slot) with slot < slots - 1 (the last slot is the shared
// expert). One warp per program.
extern "C" __global__ void rot_in_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// an epilogue program alone reads its row's 128-column block of every
// (matrix, split) partial: 512 bytes, 128-byte aligned (N and D are multiples of 128 floats). Once the warp has summed
// them they are dead (the next writer is a later kernel's partials), so their L2 lines are dropped without a write-back
// instead of being evicted dirty to DRAM. No value changes. ``lines`` = matrices x splits x 4.
__device__ __forceinline__ void discard_partials(const float* Z, int lane, int lines, int P, int p, int N, int blk) {
    __syncwarp();
    for (int i = lane; i < lines; i += 32) {
        const float* zl = Z + ((size_t)(i >> 2) * P + p) * N + blk * 128 + (i & 3) * 32;
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(zl) : "memory");
    }
}

// Program (member row, 128-block of the rank's intermediate width): gate and up outputs summed over the splits in
// order, rotated, scaled by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down
// projection's input rotation: Xd[row][block] = fp16((act * suh_d[e]) @ H).
extern "C" __global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int slots, float limit, int discard) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    if (discard) discard_partials(Z, lane, 2 * SK * 4, P, p, N, blk);
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): the down projection's output summed over the splits in
// order, rotated and scaled by svh: Y[row][slot][:] (fp32, this rank's share of the expert's output).
extern "C" __global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int slots, int discard) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    if (discard) discard_partials(Z, lane, SK * 4, P, p, D, blk);
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

}

extern "C" __global__ void __launch_bounds__(128) exl3_grouped_8_4(
 const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1,
 const int* uids, const int* ucount, const int* members, float* Z,
 int K, int N, int P, int SK, int maxm, int slots, int E) {
 alaska_exl3::grouped_kernel<8,4>(X0,X1,T0,T1,uids,ucount,members,Z,K,N,P,SK,maxm,slots,E);
}
