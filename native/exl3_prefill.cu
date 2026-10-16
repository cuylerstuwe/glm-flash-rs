// GPU device kernels with Rust-owned launch and allocation.
// GLM-5.3-Flash's EXL3 routed experts for fast prefill chunks, as two fused
// grouped GEMMs over the window's distinct experts.
//
//   gateup  Xd[p] = fp16(H(act * suh_d)),  act = limited SwiGLU(H(Xg[p] @ Wg_q) * svh_g, H(Xu[p] @ Wu_q) * svh_u)
//   down    Y[p]  = H(Xd[p] @ Wd_q) * svh_d                                          (fp32, this rank's share)
//
// for every routed pair p = row * slots + slot (Xg, Xu from exl3.cu's rot_in; H = the 128-block Hadamard / sqrt(128)).
// Program (128-column block, distinct expert u): the 4 warps decode each 16x16 trellis tile of a K slab ONCE into
// shared memory (as the lanes' mma B fragments), then every warp multiplies its own member tiles (16 pairs each) by
// the whole slab, so a weight tile is fetched and decoded once per 64 (gate/up) or 128 (down) members instead of once
// per 16-32; the next slab's words are in flight while the current one multiplies.  The down kernel also loads each k tile's A
// fragments (the members' rows) one k tile ahead of their mma (-10% on GB10 at 1024 rows; same values, same bits; for
// gate/up the extra registers cost more than the latency they hide). One warp accumulates the FULL K
// of its rows in one ascending chain of mma (no split-K partials, no Z buffer), and the column block is exactly one
// Hadamard block, so the output rotation, scales and SwiGLU run in registers on the mma accumulators and the result
// is stored once. Upstream's path (``exl3.cu``: K split 4 ways and 4 warps, Z partials, one-warp epilogue programs)
// is kept for decode and verify windows.
//
// Exactness: this is NOT upstream's arithmetic (one K chain instead of 16 partial sums), so it is only used for
// fast-prefill chunks, whose bits no decode window has to reproduce (``fastpf.py``: the boundary rule). It is
// deterministic (fixed chains, fixed butterfly order, no atomics, each output written by one thread) and even row
// independent: a pair's outputs depend only on its own inputs (mma keeps rows apart, the K order is fixed by K).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr int WARPS = 4;                             // warps a block
constexpr int NB = 8;                                // n tiles a block: 128 columns, one Hadamard block

// -- the EXL3 format (the same definitions as exl3.cu) ----------------------------------------------------------------
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

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves: b0 = rows
// (k) 2t, 2t+1 | 2t+8, 2t+9 of column (n) g, b1 the same of column g + 8 (g = lane / 4, t = lane % 4).
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// -- the Hadamard transform on mma accumulators -----------------------------------------------------------------------
// A warp's accumulators for one member tile and the block's 128 columns: v[nt][h][c] holds row g + 8 * (c >> 1),
// column nt * 16 + h * 8 + 2t + (c & 1). Column bit 0 is c & 1, bits 1-2 are t (lane bits 0-1), bit 3 is h, bits 4-6
// are nt. The butterflies run bit 0 to bit 6, each as (lower + upper, lower - upper): the order of exl3.cu's fwht128,
// so an element gets the same sums as there.
__device__ __forceinline__ void fwht_frag(float (&v)[NB][2][4], int lane) {
#pragma unroll
    for (int nt = 0; nt < NB; ++nt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; c += 2) {
                const float a = v[nt][h][c], b = v[nt][h][c + 1];
                v[nt][h][c] = a + b;
                v[nt][h][c + 1] = a - b;
            }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1)
#pragma unroll
        for (int nt = 0; nt < NB; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float o = __shfl_xor_sync(0xffffffffu, v[nt][h][c], m);
                    v[nt][h][c] = (lane & m) ? o - v[nt][h][c] : v[nt][h][c] + o;
                }
#pragma unroll
    for (int nt = 0; nt < NB; ++nt)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float a = v[nt][0][c], b = v[nt][1][c];
            v[nt][0][c] = a + b;
            v[nt][1][c] = a - b;
        }
#pragma unroll
    for (int m = 1; m < NB; m <<= 1)
#pragma unroll
        for (int nt = 0; nt < NB; ++nt) {
            if (nt & m) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const float a = v[nt][h][c], b = v[nt | m][h][c];
                    v[nt][h][c] = a + b;
                    v[nt | m][h][c] = a - b;
                }
        }
}

// -- the mainloop -----------------------------------------------------------------------------------------------------
// A K slab of MATS matrices x KS k tiles x NB n tiles, decoded: per tile, each lane's (b0[0], b0[1], b1[0], b1[1]).
template <int MATS, int KS>
struct Slab {
    uint4 f[MATS * KS * NB][32];
};

// Member tiles warp + WARPS * j (j < MTW) of this pass (rows_sh: pair rows, -1 = none) times the block's 128 columns
// of matrix mat over all of K: acc[j][mat]. K goes in slabs of KS k tiles; slab s is decoded into buffer s & 1 while
// the words of slab s + 1 load, then multiplied (one __syncthreads a slab: a warp writing buffer s & 1 has passed the
// barrier of slab s - 1, which every warp reaches only after multiplying slab s - 2).
template <int MATS, int KS, int MTW, bool PREFETCH>
__device__ __forceinline__ void mainloop(const half* __restrict__ X0, const half* __restrict__ X1,
                                         const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1, int e,
                                         int K, int N, int nb, const int* rows_sh, Slab<MATS, KS>* slabs,
                                         float (&acc)[MTW][MATS][NB][2][4]) {
    static_assert(MATS * KS == WARPS, "warp w decodes row w of a slab: (matrix w / KS, k tile w % KS), all NB n tiles");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4, S = KT / KS;

    int r0[MTW], r1[MTW];
    bool act[MTW];
#pragma unroll
    for (int j = 0; j < MTW; ++j) {
        const int m = warp + WARPS * j;
        r0[j] = rows_sh[16 * m + g];
        r1[j] = rows_sh[16 * m + g + 8];
        act[j] = rows_sh[16 * m] >= 0;             // members come first: an empty tile has no member at all
    }
#pragma unroll
    for (int j = 0; j < MTW; ++j)
#pragma unroll
        for (int mat = 0; mat < MATS; ++mat)
#pragma unroll
            for (int nt = 0; nt < NB; ++nt)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[j][mat][nt][h][c] = 0.f;

    // this warp's tiles of a slab: (matrix warp / KS, k tile warp % KS) x n tiles 0..NB-1, i = warp * NB + nt
    const uint32_t* src = (warp / KS ? T1 : T0) + (((size_t)e * KT + warp % KS) * NTILES + nb * NB) * 32 + lane;
    const size_t slab_step = (size_t)KS * NTILES * 32;
    uint32_t words[NB];
#pragma unroll
    for (int q = 0; q < NB; ++q) words[q] = __ldg(src + q * 32);

    // The A fragments (this warp's member rows) of k tile kt; loaded one k tile ahead of their mma, so the L2
    // latency of the row gathers overlaps the previous k tile's products (the values, hence the bits, are the same).
    uint32_t a_cur[MTW][MATS][4], a_nxt[MTW][MATS][4];
    auto load_a = [&](int kt, uint32_t (&a)[MTW][MATS][4]) {
        const int k = kt * 16 + 2 * t;
#pragma unroll
        for (int j = 0; j < MTW; ++j)
#pragma unroll
            for (int mat = 0; mat < MATS; ++mat) {
                const half* X = mat ? X1 : X0;
                const half* x0 = X + (size_t)(r0[j] < 0 ? 0 : r0[j]) * K + k;
                const half* x1 = X + (size_t)(r1[j] < 0 ? 0 : r1[j]) * K + k;
                const bool ok = act[j];
                a[j][mat][0] = load_pair(x0, ok && r0[j] >= 0);
                a[j][mat][1] = load_pair(x1, ok && r1[j] >= 0);
                a[j][mat][2] = load_pair(x0 + 8, ok && r0[j] >= 0);
                a[j][mat][3] = load_pair(x1 + 8, ok && r1[j] >= 0);
            }
    };
    if (PREFETCH) load_a(0, a_cur);

    for (int s = 0; s < S; ++s) {
        Slab<MATS, KS>& sl = slabs[s & 1];
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            uint32_t b0[2], b1[2];
            decode_tile(words[q], lane, b0, b1);
            sl.f[warp * NB + q][lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
        }
        if (s + 1 < S) {
            src += slab_step;
#pragma unroll
            for (int q = 0; q < NB; ++q) words[q] = __ldg(src + q * 32);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const int kt = s * KS + kk;
            if (PREFETCH && kt + 1 < KT) load_a(kt + 1, a_nxt);
#pragma unroll
            for (int j = 0; j < MTW; ++j) {
                if (!act[j]) continue;                 // warp-uniform
#pragma unroll
                for (int mat = 0; mat < MATS; ++mat) {
                    uint32_t a[4];
                    if (PREFETCH) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) a[c] = a_cur[j][mat][c];
                    } else {
                        const int k = kt * 16 + 2 * t;
                        const half* X = mat ? X1 : X0;
                        const half* x0 = X + (size_t)(r0[j] < 0 ? 0 : r0[j]) * K + k;
                        const half* x1 = X + (size_t)(r1[j] < 0 ? 0 : r1[j]) * K + k;
                        a[0] = load_pair(x0, r0[j] >= 0);
                        a[1] = load_pair(x1, r1[j] >= 0);
                        a[2] = load_pair(x0 + 8, r0[j] >= 0);
                        a[3] = load_pair(x1 + 8, r1[j] >= 0);
                    }
#pragma unroll
                    for (int nt = 0; nt < NB; ++nt) {
                        const uint4 f = sl.f[(mat * KS + kk) * NB + nt][lane];
                        mma16816(acc[j][mat][nt][0], a, f.x, f.y);
                        mma16816(acc[j][mat][nt][1], a, f.z, f.w);
                    }
                }
            }
            if (PREFETCH) {
#pragma unroll
                for (int j = 0; j < MTW; ++j)
#pragma unroll
                    for (int mat = 0; mat < MATS; ++mat)
#pragma unroll
                        for (int c = 0; c < 4; ++c) a_cur[j][mat][c] = a_nxt[j][mat][c];
            }
        }
    }
}

// The pass loop of both kernels: members of expert u, 16 * WARPS * MTW at a time, into rows_sh (pair rows).
__device__ __forceinline__ bool load_rows(int* rows_sh, int count, const int* __restrict__ members, int u, int maxm,
                                          int m0, int slots) {
    __syncthreads();                               // the previous pass is done with rows_sh and the slabs
    for (int i = threadIdx.x; i < count; i += WARPS * 32) {
        const int m = m0 + i;
        const int code = m < maxm ? members[(size_t)u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    return rows_sh[0] >= 0;                        // same for every thread
}

// Program (128-column block nb of the rank's expert width, distinct expert u): gate and up of every member, then
// Xd = fp16(H(act * suh_d)) for the block (exl3.cu gateup_epilogue's formulas, per element in the same order).
template <int KS>
__device__ __forceinline__ void gateup_body(
    const half* __restrict__ XG, const half* __restrict__ XU, const uint32_t* __restrict__ TG,
    const uint32_t* __restrict__ TU, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int maxm, int slots, int E, float limit) {
    constexpr int MTW = 1, ROWS = 16 * WARPS * MTW;
    const int u = blockIdx.y, nb = blockIdx.x;
    if (u >= ucount[0] || uids[u] >= E) return;    // the shared expert (id E) is not EXL3
    const int e = uids[u];
    __shared__ int rows_sh[ROWS];
    __shared__ Slab<2, KS> slabs[2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    for (int m0 = 0; m0 < maxm; m0 += ROWS) {
        if (!load_rows(rows_sh, ROWS, members, u, maxm, m0, slots)) break;
        float acc[MTW][2][NB][2][4];
        mainloop<2, KS, MTW, false>(XG, XU, TG, TU, e, K, N, nb, rows_sh, slabs, acc);
        if (rows_sh[16 * warp] < 0) continue;       // this warp's member tile is empty (warp-uniform)
        float (&gv)[NB][2][4] = acc[0][0];
        float (&uv)[NB][2][4] = acc[0][1];
        fwht_frag(gv, lane);
        fwht_frag(uv, lane);
        const half* sg = svh_g + (size_t)e * N + nb * 128;
        const half* su = svh_u + (size_t)e * N + nb * 128;
        const half* sd = suh_d + (size_t)e * N + nb * 128;
#pragma unroll
        for (int nt = 0; nt < NB; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int col = nt * 16 + h * 8 + 2 * t + (c & 1);
                    const float gg = fminf(bf16r(gv[nt][h][c] * HAD_SCALE * __half2float(sg[col])), limit);
                    const float uu = fminf(fmaxf(bf16r(uv[nt][h][c] * HAD_SCALE * __half2float(su[col])), -limit),
                                           limit);
                    const float a = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    gv[nt][h][c] = a * __half2float(sd[col]);
                }
        fwht_frag(gv, lane);
        const int ra = rows_sh[16 * warp + g], rb = rows_sh[16 * warp + g + 8];
#pragma unroll
        for (int nt = 0; nt < NB; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int n = nb * 128 + nt * 16 + h * 8 + 2 * t;
                if (ra >= 0)
                    *reinterpret_cast<half2*>(xd + (size_t)ra * N + n) = __halves2half2(
                        __float2half_rn(gv[nt][h][0] * HAD_SCALE), __float2half_rn(gv[nt][h][1] * HAD_SCALE));
                if (rb >= 0)
                    *reinterpret_cast<half2*>(xd + (size_t)rb * N + n) = __halves2half2(
                        __float2half_rn(gv[nt][h][2] * HAD_SCALE), __float2half_rn(gv[nt][h][3] * HAD_SCALE));
            }
    }
}

// Program (128-column block nb of the model width, distinct expert u): Y[p][cols] = H(Xd[p] @ Wd_q) * svh_d
// (exl3.cu down_epilogue's formula), for every member p of expert u.
template <int KS>
__device__ __forceinline__ void down_body(
    const half* __restrict__ XD, const uint32_t* __restrict__ TD, const int* __restrict__ uids,
    const int* __restrict__ ucount, const int* __restrict__ members, const half* __restrict__ svh_d,
    float* __restrict__ y, int K, int N, int maxm, int slots, int E) {
    constexpr int MTW = 2, ROWS = 16 * WARPS * MTW;
    const int u = blockIdx.y, nb = blockIdx.x;
    if (u >= ucount[0] || uids[u] >= E) return;
    const int e = uids[u];
    __shared__ int rows_sh[ROWS];
    __shared__ Slab<1, KS> slabs[2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    for (int m0 = 0; m0 < maxm; m0 += ROWS) {
        if (!load_rows(rows_sh, ROWS, members, u, maxm, m0, slots)) break;
        float acc[MTW][1][NB][2][4];
        mainloop<1, KS, MTW, true>(XD, XD, TD, TD, e, K, N, nb, rows_sh, slabs, acc);
        const half* sv = svh_d + (size_t)e * N + nb * 128;
#pragma unroll
        for (int j = 0; j < MTW; ++j) {
            const int m = warp + WARPS * j;
            if (rows_sh[16 * m] < 0) continue;      // warp-uniform
            float (&v)[NB][2][4] = acc[j][0];
            fwht_frag(v, lane);
            const int ra = rows_sh[16 * m + g], rb = rows_sh[16 * m + g + 8];
#pragma unroll
            for (int nt = 0; nt < NB; ++nt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = nt * 16 + h * 8 + 2 * t;
                    const float s0 = __half2float(sv[col]), s1 = __half2float(sv[col + 1]);
                    if (ra >= 0)
                        *reinterpret_cast<float2*>(y + (size_t)ra * N + nb * 128 + col) =
                            make_float2(v[nt][h][0] * HAD_SCALE * s0, v[nt][h][1] * HAD_SCALE * s1);
                    if (rb >= 0)
                        *reinterpret_cast<float2*>(y + (size_t)rb * N + nb * 128 + col) =
                            make_float2(v[nt][h][2] * HAD_SCALE * s0, v[nt][h][3] * HAD_SCALE * s1);
                }
        }
    }
}

// -- fast2 (the default; GLM53_TF_FAST_EXPERTS=v1 runs the kernels above): the weights as the mma's A operand ----------
// ncu on the v1 kernels (8192-row chunks): 2 blocks of 4 warps an SM, 43% of the stalls at the slab barrier. With
// every warp decoding and multiplying they were bound by shared memory instead: each warp re-read every decoded 16 x 16
// fragment (512 B) for its 16 member rows, ~300 KB a 32-k stage and SM, as long as the stage's mma.
// So fast2 computes Y^T = W^T X^T. A warp owns MTL 16-column tiles of the item's 128 columns (of one matrix) for ALL
// the item's BM members and decodes its own trellis tiles straight into registers: a decoded tile, as decode_tile lays
// it out, is exactly an m16n8k16 A fragment, so decoded weights never touch shared memory and each fragment feeds
// BM / 8 mma. Only the member rows are shared: gathered by cp.async into an NSA-deep ring of KS-k-tile stages
// (zero-filled past the last member), read with ldmatrix as B fragments that each feed MTL mma. One __syncthreads a
// stage. Work items (distinct expert u, pass of BM members, 128-column block nb) come from plan_kernel's per-expert
// pass counts; a persistent grid walks them t-major with nb fastest, so the column blocks of a pass read its rows from
// L2 together and an expert's passes read its weights together, and no program is launched for an absent expert.
// Epilogue: the accumulators go to shared memory as fp32 rows (ER rows a round), then one warp a row runs the
// 128-point transforms with butterflies bit 0 to bit 6, (lo + hi, lo - hi) -- fwht128's order, as fwht_frag -- and
// v1's formulas element by element.
//
// Exactness: every output element is the same chain of m16n8k16 products over ascending k tiles as in v1 (A and B
// swapped give the same element bits: checked against v1 bit for bit on GB10, 1024-8192 rows, uniform and skewed
// routing), the same transforms and the same epilogue arithmetic, so fast2 == v1 bit for bit. Row-independent and
// deterministic: the tiling is fixed by the shapes, a pair's outputs depend only on its own row, no atomics on data.
namespace fast2 {

constexpr int MAXU = 1024;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&a)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

// plan[0] = items (passes) in all, plan[1 + u] = the first pass of distinct expert u (plan[1 + maxu] = plan[0]).
extern "C" __global__ void fast_expert_plan(const int* __restrict__ uids, const int* __restrict__ ucount,
                            const int* __restrict__ members, int maxu, int maxm, int E, int BM, int* __restrict__ plan) {
    __shared__ int sc[MAXU];
    const int u = threadIdx.x;
    int items = 0;
    if (u < maxu && u < ucount[0] && uids[u] < E) {
        const int* m = members + (size_t)u * maxm;
        int lo = 0, hi = maxm;                     // members come first: the count is the first -1
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (m[mid] >= 0) lo = mid + 1; else hi = mid;
        }
        items = (lo + BM - 1) / BM;
    }
    sc[u] = items;
    __syncthreads();
    for (int d = 1; d < MAXU; d <<= 1) {
        const int v = u >= d ? sc[u - d] : 0;
        __syncthreads();
        sc[u] += v;
        __syncthreads();
    }
    if (u < maxu) plan[2 + u] = sc[u];
    if (u == 0) {
        plan[1] = 0;
        plan[0] = sc[maxu - 1];
        plan[maxu + 2] = 0; // reset this invocation's ticket in the same stream
    }
}

template <int MATS, int MTL, int NG, int KS, int NSA, int NGR>
struct Cfg {
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int LDA = KS * 16 + 8, LDE = 132, ER = NGR * 8, ROUNDS = NG / NGR;
    static constexpr size_t RING = (size_t)NSA * MATS * BM * LDA * sizeof(half);
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
};

// 128-point transform of one row, 4 values a lane (column 4 * lane + j), butterflies bit 0 to bit 6.
__device__ __forceinline__ void fwht_row(float (&v)[4], int lane) {
    float a = v[0], b = v[1];
    v[0] = a + b; v[1] = a - b;
    a = v[2]; b = v[3];
    v[2] = a + b; v[3] = a - b;
    a = v[0]; b = v[2];
    v[0] = a + b; v[2] = a - b;
    a = v[1]; b = v[3];
    v[1] = a + b; v[3] = a - b;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
}

template <int MATS, int MTL, int NG, int KS, int NSA, int NGR>
__global__ void __launch_bounds__(Cfg<MATS, MTL, NG, KS, NSA, NGR>::THREADS) expert_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ uids, const int* __restrict__ members,
    const int* __restrict__ plan, int K, int N, int maxm, int slots, int maxu, const half* __restrict__ sv0,
    const half* __restrict__ sv1, const half* __restrict__ sd, half* __restrict__ xd, float* __restrict__ y,
    float limit) {
    using C = Cfg<MATS, MTL, NG, KS, NSA, NGR>;
    constexpr int W = C::W, THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0, "n groups in pairs");
    extern __shared__ __align__(16) unsigned char smem[];
    half* ring = reinterpret_cast<half*>(smem);              // [NSA][MATS][BM][LDA]
    float* ep = reinterpret_cast<float*>(smem);              // [MATS][ER][LDE], after the K loop
    __shared__ int rows_sh[BM];
    __shared__ int offs[MAXU + 1];
    __shared__ int cnt_sh;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;   // columns slice * 16 MTL .. + 16 MTL of the item
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N / 128;
    const int total = plan[0] * NBLK;
    for (int i = threadIdx.x; i <= maxu; i += THREADS) offs[i] = plan[1 + i];

    for (int item = blockIdx.x; item < total; item += gridDim.x) {
        const int tt = item / NBLK, nb = item % NBLK;
        __syncthreads();
        int lo = 0, hi = maxu;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (offs[mid] <= tt) lo = mid; else hi = mid;
        }
        const int u = lo, pass = tt - offs[u], e = uids[u];
        if (threadIdx.x == 0) cnt_sh = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < BM; i += THREADS) {
            const int m = pass * BM + i;
            const int code = m < maxm ? members[(size_t)u * maxm + m] : -1;
            rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
            if (code >= 0) atomicMax(&cnt_sh, i + 1);
        }
        __syncthreads();
        const int cnt = cnt_sh;                                 // members of this pass (they come first)

        auto load_a = [&](int s) {
            constexpr int CPR = KS * 2;
            const int k0 = s * KS * 16;
            half* dst = ring + (size_t)(s % NSA) * MATS * BM * LDA;
#pragma unroll
            for (int i = threadIdx.x; i < MATS * BM * CPR; i += THREADS) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                const int row = rows_sh[r];
                const half* src = (m ? X1 : X0) + (size_t)(row < 0 ? 0 : row) * K + k0 + c * 8;
                cp_async16(dst + ((size_t)m * BM + r) * LDA + c * 8, src, row >= 0);
            }
        };
        // this warp's trellis words: column tiles nb * 8 + slice * MTL + l, k tile kt
        const uint32_t* tw = (mat ? T1 : T0) + ((size_t)e * KT * NTILES + nb * NB + slice * MTL) * 32 + lane;
        uint32_t wr[KS][MTL];
        auto load_w = [&](int kt, uint32_t (&w)[MTL]) {
#pragma unroll
            for (int l = 0; l < MTL; ++l) w[l] = __ldg(tw + ((size_t)kt * NTILES + l) * 32);
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

#pragma unroll
        for (int s = 0; s < NSA - 1; ++s) {
            if (s < S) load_a(s);
            cp_commit();
        }
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) load_w(kk, wr[kk]);
        cp_wait<NSA - 2>();
        __syncthreads();
        for (int s = 0; s < S; ++s) {
            if (s + NSA - 1 < S) load_a(s + NSA - 1);
            cp_commit();
            const half* A = ring + ((size_t)(s % NSA) * MATS + mat) * BM * LDA;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t af[MTL][4];
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    uint32_t b0[2], b1[2];
                    decode_tile(wr[kk][l], lane, b0, b1);
                    af[l][0] = b0[0]; af[l][1] = b1[0]; af[l][2] = b0[1]; af[l][3] = b1[1];
                }
                if (s + 1 < S) load_w((s + 1) * KS + kk, wr[kk]);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                         // warp-uniform
                        uint32_t b[4];
                        ldsm4(b, A + (16 * np + (lane & 7) + ((lane >> 4) << 3)) * LDA + kk * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b[0], b[1]);
                            mma16816(acc[l][2 * np + 1], af[l], b[2], b[3]);
                        }
                    }
                }
            }
            cp_wait<NSA - 2>();
            __syncthreads();
        }

        // epilogue, ER rows a round: accumulators -> ep[mat][row][col] (fp32), then one warp a row
#pragma unroll
        for (int rd = 0; rd < C::ROUNDS; ++rd) {
            if (rd) __syncthreads();
            if (ER * rd < cnt) {
#pragma unroll
                for (int n = 0; n < NGR; ++n) {
                    const int ng = rd * NGR + n;
                    float* E = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                    for (int l = 0; l < MTL; ++l) {
                        E[l * 16] = acc[l][ng][0];
                        E[LDE + l * 16] = acc[l][ng][1];
                        E[l * 16 + 8] = acc[l][ng][2];
                        E[LDE + l * 16 + 8] = acc[l][ng][3];
                    }
                }
            }
            __syncthreads();
            for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
                const int row = rows_sh[ER * rd + r];
                const int c0 = 4 * lane;
                float v[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
                fwht_row(v, lane);
                if constexpr (MATS == 2) {
                    float w[4];
                    *reinterpret_cast<float4*>(w) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                    fwht_row(w, lane);
                    const half* sg = sv0 + (size_t)e * N + nb * 128 + c0;
                    const half* su = sv1 + (size_t)e * N + nb * 128 + c0;
                    const half* sdd = sd + (size_t)e * N + nb * 128 + c0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float gg = fminf(bf16r(v[j] * HAD_SCALE * __half2float(sg[j])), limit);
                        const float uu = fminf(fmaxf(bf16r(w[j] * HAD_SCALE * __half2float(su[j])), -limit), limit);
                        const float a = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                        v[j] = a * __half2float(sdd[j]);
                    }
                    fwht_row(v, lane);
                    half2* o = reinterpret_cast<half2*>(xd + (size_t)row * N + nb * 128 + c0);
                    o[0] = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
                    o[1] = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
                } else {
                    const half* sv = sv0 + (size_t)e * N + nb * 128 + c0;
                    float4 o;
                    o.x = v[0] * HAD_SCALE * __half2float(sv[0]);
                    o.y = v[1] * HAD_SCALE * __half2float(sv[1]);
                    o.z = v[2] * HAD_SCALE * __half2float(sv[2]);
                    o.w = v[3] * HAD_SCALE * __half2float(sv[3]);
                    *reinterpret_cast<float4*>(y + (size_t)row * N + nb * 128 + c0) = o;
                }
            }
        }
    }
}


} // namespace fast2
namespace fat {

// the row ring's 16-byte chunk c of row r (CPR chunks a row): XOR-swizzled so the 8 rows of an ldmatrix phase hit
// 8 distinct bank groups
template <int CPR>
__device__ __forceinline__ int swz(int r, int c) {
    static_assert(CPR == 8 || CPR == 4, "rows of 64 or 32 halves");
    return CPR == 8 ? (c ^ (r & 7)) : (c ^ ((r >> 1) & 3));
}

template <int MATS, int MTL, int NG, int KS, int NSA, int NGR, bool SHX>
struct Cfg {
    static constexpr int BM = NG * 8, WPM = NB / MTL, W = MATS * WPM, THREADS = W * 32;
    static constexpr int XM = SHX ? 1 : MATS;                       // matrices of member rows in a stage
    static constexpr int CPR = KS * 2;                              // 16-byte chunks of a row in a stage
    static constexpr int LDA = KS * 16;                             // halves a row (swizzled, no padding)
    static constexpr int LDE = 132, ER = NGR * 8, ROUNDS = NG / NGR;
    static constexpr size_t A_BYTES = (size_t)XM * BM * LDA * sizeof(half);
    static constexpr size_t W_BYTES = (size_t)MATS * KS * NB * 32 * sizeof(uint32_t);
    static constexpr size_t STAGE = A_BYTES + W_BYTES;
    static constexpr size_t RING = (size_t)NSA * STAGE;
    static constexpr size_t EPI = (size_t)MATS * ER * LDE * sizeof(float);
    static constexpr size_t SMEM = RING > EPI ? RING : EPI;
    static_assert(A_BYTES % 16 == 0, "stage alignment");
};

// MINB: CTAs an SM the registers are capped for (2: <= 128 registers at 256 threads)
template <int MATS, int MTL, int NG, int KS, int NSA, int NGR, bool SHX, int MINB>
__device__ __forceinline__ void expert_body(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ uids, const int* __restrict__ members,
    const int* __restrict__ plan, int* __restrict__ ticket, int K, int N, int maxm, int slots, int maxu,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ sd, half* __restrict__ xd,
    float* __restrict__ y, float limit) {
    using C = Cfg<MATS, MTL, NG, KS, NSA, NGR, SHX>;
    constexpr int W = C::W, THREADS = C::THREADS, BM = C::BM, LDA = C::LDA, LDE = C::LDE, ER = C::ER, CPR = C::CPR;
    static_assert(NG % 2 == 0 && NG % NGR == 0 && NGR % 2 == 0, "n groups in pairs");
    extern __shared__ __align__(16) unsigned char smem[];
    float* ep = reinterpret_cast<float*>(smem);              // [MATS][ER][LDE], after the K loop
    __shared__ int rows_sh[BM];
    __shared__ int item_sh, u_sh, cnt_sh;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int mat = warp / C::WPM, slice = warp % C::WPM;   // columns slice * 16 MTL .. + 16 MTL of the item
    const int xmat = SHX ? 0 : mat;                          // the matrix of rows this warp multiplies
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4, S = KT / KS, NBLK = N / 128;
    const int total = plan[0] * NBLK;
    // ldmatrix: this lane's row within a 16-row pair of n groups, and its chunk's swizzle (constant per lane)
    const int lr = (lane & 7) + ((lane >> 4) << 3);
    const int lc = (lane >> 3) & 1;

    int u = 0;                                               // thread 0: the distinct expert of the last item
    int next = blockIdx.x;                                   // static stride (no ticket)
    for (;;) {
        __syncthreads();                                     // the previous item is done with rows_sh / smem
        if (threadIdx.x == 0) {
            const int it = ticket ? atomicAdd(ticket, 1) : next;
            item_sh = it;
            if (it < total) {
                // items come in ascending order per CTA (stride or ticket): walk u forward from the last one
                const int tt = it / NBLK;
                while (u + 1 < maxu && __ldg(plan + 2 + u) <= tt) ++u;
                u_sh = u;
            }
            cnt_sh = 0;
        }
        next += gridDim.x;
        __syncthreads();
        const int item = item_sh;
        if (item >= total) break;
        const int tt = item / NBLK, nb = item % NBLK;
        const int uu = u_sh, pass = tt - plan[1 + uu], e = uids[uu];
        for (int i = threadIdx.x; i < BM; i += THREADS) {
            const int m = pass * BM + i;
            const int code = m < maxm ? members[(size_t)uu * maxm + m] : -1;
            rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
            if (code >= 0) atomicMax(&cnt_sh, i + 1);
        }
        __syncthreads();
        const int cnt = cnt_sh;                              // members of this pass (they come first)

        // stage s: the member rows' k range [s KS 16, (s + 1) KS 16) (zero-filled past the last member), then the
        // item's trellis words of k tiles s KS .. s KS + KS - 1, all MATS matrices, the NB column tiles of block nb
        auto load_stage = [&](int s) {
            unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            half* ra = reinterpret_cast<half*>(st);
            const int k0 = s * KS * 16;
#pragma unroll
            for (int i = threadIdx.x; i < C::XM * BM * CPR; i += THREADS) {
                const int m = i / (BM * CPR), r = (i / CPR) % BM, c = i % CPR;
                const int row = rows_sh[r];
                const half* src = (m ? X1 : X0) + (size_t)(row < 0 ? 0 : row) * K + k0 + c * 8;
                fast2::cp_async16(ra + ((size_t)m * BM + r) * LDA + swz<CPR>(r, c) * 8, src, row >= 0);
            }
            uint32_t* rw = reinterpret_cast<uint32_t*>(st + C::A_BYTES);
            constexpr int WCH = MATS * KS * NB * 8;          // 16-byte chunks: 8 a 16 x 16 tile
#pragma unroll
            for (int i = threadIdx.x; i < WCH; i += THREADS) {
                const int q = i & 7, n = (i >> 3) % NB, kk = (i / (8 * NB)) % KS, m = i / (8 * NB * KS);
                const uint32_t* src = (m ? T1 : T0) + (((size_t)e * KT + s * KS + kk) * NTILES + nb * NB + n) * 32 + q * 4;
                fast2::cp_async16(rw + ((m * KS + kk) * NB + n) * 32 + q * 4, src, true);
            }
        };

        float acc[MTL][NG][4];
#pragma unroll
        for (int l = 0; l < MTL; ++l)
#pragma unroll
            for (int n = 0; n < NG; ++n)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[l][n][c] = 0.f;

#pragma unroll
        for (int s = 0; s < NSA - 1; ++s) {
            if (s < S) load_stage(s);
            fast2::cp_commit();
        }
        fast2::cp_wait<NSA - 2>();
        __syncthreads();
        for (int s = 0; s < S; ++s) {
            if (s + NSA - 1 < S) load_stage(s + NSA - 1);
            fast2::cp_commit();
            const unsigned char* st = smem + (size_t)(s % NSA) * C::STAGE;
            const half* A = reinterpret_cast<const half*>(st) + (size_t)xmat * BM * LDA;
            const uint32_t* Wd = reinterpret_cast<const uint32_t*>(st + C::A_BYTES) + (size_t)mat * KS * NB * 32 +
                                 slice * MTL * 32 + lane;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk) {
                uint32_t af[MTL][4];
#pragma unroll
                for (int l = 0; l < MTL; ++l) {
                    uint32_t b0[2], b1[2];
                    decode_tile(Wd[(kk * NB + l) * 32], lane, b0, b1);
                    af[l][0] = b0[0]; af[l][1] = b1[0]; af[l][2] = b0[1]; af[l][3] = b1[1];
                }
                const int chunk = swz<CPR>(lr, kk * 2 + lc);
#pragma unroll
                for (int np = 0; np < NG / 2; ++np) {
                    if (16 * np < cnt) {                         // warp-uniform
                        uint32_t b[4];
                        fast2::ldsm4(b, A + (16 * np + lr) * LDA + chunk * 8);
#pragma unroll
                        for (int l = 0; l < MTL; ++l) {
                            mma16816(acc[l][2 * np], af[l], b[0], b[1]);
                            mma16816(acc[l][2 * np + 1], af[l], b[2], b[3]);
                        }
                    }
                }
            }
            fast2::cp_wait<NSA - 2>();
            __syncthreads();
        }

        // epilogue: fast2's, unchanged (ER rows a round: accumulators -> ep[mat][row][col] fp32, one warp a row)
#pragma unroll
        for (int rd = 0; rd < C::ROUNDS; ++rd) {
            if (rd) __syncthreads();
            if (ER * rd < cnt) {
#pragma unroll
                for (int n = 0; n < NGR; ++n) {
                    const int ng = rd * NGR + n;
                    float* E = ep + ((size_t)mat * ER + 8 * n + 2 * t) * LDE + slice * 16 * MTL + g;
#pragma unroll
                    for (int l = 0; l < MTL; ++l) {
                        E[l * 16] = acc[l][ng][0];
                        E[LDE + l * 16] = acc[l][ng][1];
                        E[l * 16 + 8] = acc[l][ng][2];
                        E[LDE + l * 16 + 8] = acc[l][ng][3];
                    }
                }
            }
            __syncthreads();
            for (int r = warp; r < ER && ER * rd + r < cnt; r += W) {
                const int row = rows_sh[ER * rd + r];
                const int c0 = 4 * lane;
                float v[4];
                *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(ep + (size_t)r * LDE + c0);
                fast2::fwht_row(v, lane);
                if constexpr (MATS == 2) {
                    float w[4];
                    *reinterpret_cast<float4*>(w) = *reinterpret_cast<const float4*>(ep + ((size_t)ER + r) * LDE + c0);
                    fast2::fwht_row(w, lane);
                    const half* sg = sv0 + (size_t)e * N + nb * 128 + c0;
                    const half* su = sv1 + (size_t)e * N + nb * 128 + c0;
                    const half* sdd = sd + (size_t)e * N + nb * 128 + c0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float gg = fminf(bf16r(v[j] * HAD_SCALE * __half2float(sg[j])), limit);
                        const float uu2 = fminf(fmaxf(bf16r(w[j] * HAD_SCALE * __half2float(su[j])), -limit), limit);
                        const float a = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu2);
                        v[j] = a * __half2float(sdd[j]);
                    }
                    fast2::fwht_row(v, lane);
                    half2* o = reinterpret_cast<half2*>(xd + (size_t)row * N + nb * 128 + c0);
                    o[0] = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
                    o[1] = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
                } else {
                    const half* sv = sv0 + (size_t)e * N + nb * 128 + c0;
                    float4 o;
                    o.x = v[0] * HAD_SCALE * __half2float(sv[0]);
                    o.y = v[1] * HAD_SCALE * __half2float(sv[1]);
                    o.z = v[2] * HAD_SCALE * __half2float(sv[2]);
                    o.w = v[3] * HAD_SCALE * __half2float(sv[3]);
                    *reinterpret_cast<float4*>(y + (size_t)row * N + nb * 128 + c0) = o;
                }
            }
        }
    }
}

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

extern "C" __global__ void __launch_bounds__(256) fast_rot_in(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                      const int* __restrict__ pick, const half* __restrict__ suh0,
                                                      half* __restrict__ out0, int K, int slots, int items) {
    const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= items) return;                     // warp-uniform
    const int KB = K / 128;
    const int p = item / KB, blk = item % KB;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x & 31;
    const int e = pick[row * slots + slot];
    const half* suh = suh0 + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = out0 + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}


} // namespace fat
static_assert(fat::Cfg<2,2,8,4,3,4,true>::SMEM == 49152);
static_assert(fat::Cfg<2,2,8,4,3,4,false>::SMEM == 73728);
static_assert(fat::Cfg<1,2,8,4,3,8,false>::SMEM == 36864);
static_assert(fat::Cfg<1,2,16,2,3,8,false>::SMEM == 33792);
} // anonymous namespace

extern "C" __global__ void __launch_bounds__(256,2) fast_gu_shared(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* uids, const int* members, const int* plan, int* ticket, int K, int N, int maxm, int slots, int maxu, const half* sv0, const half* sv1, const half* sd, half* xd, float* y, float limit) { fat::expert_body<2,2,8,4,3,4,true,2>(X0,X1,T0,T1,uids,members,plan,ticket,K,N,maxm,slots,maxu,sv0,sv1,sd,xd,y,limit); }

extern "C" __global__ void __launch_bounds__(256,1) fast_gu_separate(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* uids, const int* members, const int* plan, int* ticket, int K, int N, int maxm, int slots, int maxu, const half* sv0, const half* sv1, const half* sd, half* xd, float* y, float limit) { fat::expert_body<2,2,8,4,3,4,false,1>(X0,X1,T0,T1,uids,members,plan,ticket,K,N,maxm,slots,maxu,sv0,sv1,sd,xd,y,limit); }

extern "C" __global__ void __launch_bounds__(128,2) fast_down(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* uids, const int* members, const int* plan, int* ticket, int K, int N, int maxm, int slots, int maxu, const half* sv0, const half* sv1, const half* sd, half* xd, float* y, float limit) { fat::expert_body<1,2,8,4,3,8,false,2>(X0,X1,T0,T1,uids,members,plan,ticket,K,N,maxm,slots,maxu,sv0,sv1,sd,xd,y,limit); }

extern "C" __global__ void __launch_bounds__(128,2) fast_down_large(const half* X0, const half* X1, const uint32_t* T0, const uint32_t* T1, const int* uids, const int* members, const int* plan, int* ticket, int K, int N, int maxm, int slots, int maxu, const half* sv0, const half* sv1, const half* sd, half* xd, float* y, float limit) { fat::expert_body<1,2,16,2,3,8,false,2>(X0,X1,T0,T1,uids,members,plan,ticket,K,N,maxm,slots,maxu,sv0,sv1,sd,xd,y,limit); }

extern "C" __global__ void __launch_bounds__(128) fast_gu_oracle(const half* XG,const half* XU,const uint32_t* TG,const uint32_t* TU,const int* uids,const int* ucount,const int* members,const half* sg,const half* su,const half* sd,half* xd,int K,int N,int maxm,int slots,int E,float limit) {gateup_body<2>(XG,XU,TG,TU,uids,ucount,members,sg,su,sd,xd,K,N,maxm,slots,E,limit);}

extern "C" __global__ void __launch_bounds__(128) fast_down_oracle(const half* XD,const uint32_t* TD,const int* uids,const int* ucount,const int* members,const half* sv,float* y,int K,int N,int maxm,int slots,int E) {down_body<4>(XD,TD,uids,ucount,members,sv,y,K,N,maxm,slots,E);}
