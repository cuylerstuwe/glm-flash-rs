// exl3.cu's grouped_kernel -- the decode / verify / MTP windows' EXL3
// routed-expert GEMV -- with prefetched loads and exl3.cu's arithmetic.
//
// Prefetch the next trellis loads while decoding the current step, avoiding
// dependent metadata reads at the start of each work item. Preserve the
// arithmetic order and grouped-kernel layout.
//
// What is kept, per output element (so the bits are exl3.cu's for every row count and every setting below):
//   - the work item and the grid: program (u, n block, (mat * SK + split) * MT + member tile), 16 member rows of
//     distinct expert u times W_q of matrix mat over K split `split` for NT column tiles; W = 4 warps;
//   - warp w runs k tiles [split * KT/SK + w * PW, + PW) in ascending order, one mma.m16n8k16 (f16 -> f32) per
//     (k tile, column tile, n8 half) into accumulators that start at +0.0, on the same decoded B fragments
//     (decode_tile, verbatim) and the same A fragments (load_pair's values: X[row][k + 2t ...], 0 for a dead row);
//   - the warps' partial sums through shared memory, added in warp order (s = red[0]; s += red[1]; ...), stored to
//     Z[mat][split][row][n] for live member rows only -- member_tiles<NT, 4, 1>'s code, verbatim;
//   - the early exits: u >= ucount, uids[u] >= E (the shared expert), an empty member tile.
//   NT (column tiles a program) places work only, as in exl3.cu (every accumulator is its own chain).
//
// What changes (data movement only):
//   - the prologue: ucount, uids[u], the member codes and the tile's first code are read in ONE round trip (they
//     do not depend on each other), and the first PD k steps of each warp's trellis words are issued before the
//     member rows reach shared memory, so a CTA streams from its first microsecond;
//   - a PD-deep register ring a warp: while k step s is decoded and multiplied, steps s + 1 .. s + PD are in flight
//     (PD x NT x 128 B a warp: NT 8, PD 2 = 2 KB a warp, 24 KB an SM at 3 CTAs), and so are
//     the A fragments of those steps (4 x 4 B a lane a step, L1 / L2 hits);
//   - the load instruction (LD):
//       LD_NC   ld.global.nc.L1::no_allocate.v4: lane l loads bytes 16 l .. 16 l + 15 of each 512 B
//               of the step (NT / 4 loads a lane), i.e. words 4 (l % 8) .. + 3 of tile 4 v + l / 8; the warp stores
//               them to a 1-KB staging area and reads word `lane` of each tile back (decode_tile's layout), two
//               __syncwarp a step, conflict-free (16-B stores of a contiguous 512 B; 4-B reads at stride 128 B);
//       LD_W32  32-bit ld.global.nc (grouped_kernel's own instruction: word `lane` of each tile), PD steps ahead in
//               registers, no staging;
//       LD_CPA  16-byte cp.async.cg into a warp-private ring of PD + 1 steps (PD commit groups in flight), read as
//               LD_NC reads its staging area.
//     The staging area / ring lives in the warp's own slice of `red`, which the warp writes only after its last k
//     step, so shared memory (and so the 3 CTAs an SM at NT 8) is grouped_kernel's.
//
// PDL (optional, the launch attribute; GLM53_TF_DEC_EXPERT_LOADS_PDL=1): the prologue -- the grouping reads and the
// first PD k steps of trellis words -- runs before griddepcontrol.wait, overlapping the previous kernel's tail; X / Xd
// are read and Z written only after it. Safe because the grouping (ucount / uids / members) is written before
// exl3_mm.routed starts and the launch right before this one (rot_in for gate/up, gateup_epilogue for down) is a plain
// launch, so every kernel before it has completed when this one starts; the trellis is constant. The grouping is read
// with ld.global.cg (L2). Without the attribute griddepcontrol.wait returns at once. Nothing triggers dependents (the
// kernels after this one are plain launches).
//
// PROBE (timing only; wrong results): 1 = no trellis decode (the raw words go to the mma), 2 = no mma (decode, then a
// dependent xor), 3 = neither (the load path alone: words and A fragments xor-ed into the accumulators).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace alaska_expert_loads {

constexpr int W = 4;                                 // warps a CTA == exl3_mm's warps (they define the K ranges)
enum { LD_NC = 0, LD_W32 = 1, LD_CPA = 2 };

// ---- exl3.cu, verbatim -------------------------------------------------------------------------------------------
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

// ---- loads -------------------------------------------------------------------------------------------------------
// asm volatile: issued exactly where they stand relative to the (volatile) mma, so the ring depth is what the code
// says. The trellis is read once a window (L1::no_allocate); X / Xd rows are shared by the CTAs of an expert (L1 ok).
__device__ __forceinline__ uint4 ldg_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldg_w(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
// load_pair's value (exl3.cu: ok ? *(const uint32_t*)x : 0) with the load pinned in place
__device__ __forceinline__ uint32_t ldg_pair(const half* x, bool ok) {
    uint32_t v = 0u;
    if (ok) asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(x));
    return v;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

template <int NT, int PD, int LD>
struct Ring {
    static constexpr int NV = NT / 4;                            // 16-byte loads a lane a step (512 B a warp each)
    uint4 v[LD == LD_NC ? PD : 1][NV];                           // LD_NC: the steps in flight
    uint32_t w[LD == LD_W32 ? PD : 1][NT];                       // LD_W32
    uint32_t a[PD][4];                                           // the A fragments of the steps in flight
};

// Program (u, n block, (mat * SK + split) * MT + member tile): grouped_kernel's work item (member_tiles<NT, W, 1>).
constexpr int NT = 8, PD = 1, LD = LD_NC, PROBE = 0;
extern "C" __global__ void __launch_bounds__(128, 3) exl3_load_nc_8_1(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, int E) {
    static_assert(NT % 4 == 0 && PD >= 1 && PD <= 4, "NT a multiple of 4, PD 1-4");
    static_assert(LD != LD_CPA || PD + 1 <= 8, "the cp.async ring fits the warp's slice of red");
    const int u = blockIdx.x;
    const int MT = (maxm + 15) / 16;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;

    // One round trip for the whole prologue (grouped_kernel: ucount, then uids, then members, then the weights).
    // uids[u] / members[u] are in bounds for every u < gridDim.x (the host sizes the grid by uids / members).
    const int cnt = __ldcg(ucount);
    const int e = __ldcg(uids + u);
    const int m = mtile * 16 + threadIdx.x;
    const int code = threadIdx.x < 16 && m < maxm ? __ldcg(members + u * maxm + m) : -1;
    const int first = __ldcg(members + u * maxm + mtile * 16);     // mtile * 16 < maxm always (MT = ceil(maxm / 16))
    if (u >= cnt || e >= E || first < 0) return;                    // grouped_kernel's three exits, the same CTAs

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];

    const int KT = K >> 4, NTILES = N >> 4;
    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const size_t kstep = (size_t)NTILES * 32;                      // words from one k tile to the next
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + (size_t)blockIdx.y * NT) * 32;
    uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);   // this warp's staging area / ring

    Ring<NT, PD, LD> ring;
    auto issue_w = [&](int d, int s) {                             // k step s (of this warp) into ring slot d
        const uint32_t* src = tile + (size_t)s * kstep;
        if constexpr (LD == LD_NC) {
#pragma unroll
            for (int v = 0; v < Ring<NT, PD, LD>::NV; ++v) ring.v[d][v] = ldg_v4(src + 128 * v + 4 * lane);
        } else if constexpr (LD == LD_W32) {
#pragma unroll
            for (int i = 0; i < NT; ++i) ring.w[d][i] = ldg_w(src + i * 32 + lane);
        } else {
            uint32_t* dst = stage + (s % (PD + 1)) * (NT * 32);
#pragma unroll
            for (int v = 0; v < Ring<NT, PD, LD>::NV; ++v) cp_async16(dst + 128 * v + 4 * lane, src + 128 * v + 4 * lane);
        }
    };
#pragma unroll
    for (int d = 0; d < PD; ++d) {                                  // PD <= per_warp (host)
        issue_w(d, d);
        if constexpr (LD == LD_CPA) cp_commit();
    }

    if (threadIdx.x < 16) rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    pdl_wait();                                                     // X / Xd (and Z) belong to the kernels before
    __syncthreads();

    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
    auto issue_a = [&](int d, int s) {
        const int k = (kt0 + s) * 16;
        ring.a[d][0] = ldg_pair(x0 + k, r0 >= 0);
        ring.a[d][1] = ldg_pair(x1 + k, r1 >= 0);
        ring.a[d][2] = ldg_pair(x0 + k + 8, r0 >= 0);
        ring.a[d][3] = ldg_pair(x1 + k + 8, r1 >= 0);
    };
#pragma unroll
    for (int d = 0; d < PD; ++d) issue_a(d, d);

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    for (int kb = 0; kb < per_warp; kb += PD) {                    // per_warp % PD == 0 (host)
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int s = kb + d;
            const bool more = s + PD < per_warp;
            uint32_t words[NT];
            uint32_t a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) a[j] = ring.a[d][j];
            if constexpr (LD == LD_NC) {
                __syncwarp();                                     // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < Ring<NT, PD, LD>::NV; ++v)
                    *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = ring.v[d][v];
                if (more) issue_w(d, s + PD);                     // the slot's registers are free again
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
            } else if constexpr (LD == LD_W32) {
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = ring.w[d][i];
                if (more) issue_w(d, s + PD);
            } else {
                cp_wait<PD - 1>();                                // step s's group has landed (this lane's copies)
                __syncwarp();                                     // ... every lane's; step s - 1's slot is consumed
                if (more) issue_w(d, s + PD);                     // into slot (s + PD) % (PD + 1) == (s - 1)'s
                cp_commit();                                      // (an empty group past the end: counts stay uniform)
                const uint32_t* slot = stage + (s % (PD + 1)) * (NT * 32);
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = slot[i * 32 + lane];
            }
            if (more) issue_a(d, s + PD);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t b0[2], b1[2];
                if (PROBE == 1 || PROBE == 3) {
                    b0[0] = words[i]; b0[1] = words[i] ^ 1u; b1[0] = words[i] ^ 2u; b1[1] = words[i] ^ 3u;
                } else {
                    decode_tile(words[i], lane, b0, b1);
                }
                if (PROBE == 2 || PROBE == 3) {
                    acc[i][0][0] = __uint_as_float(__float_as_uint(acc[i][0][0]) ^ b0[0] ^ b0[1] ^ a[0] ^ a[2]);
                    acc[i][1][0] = __uint_as_float(__float_as_uint(acc[i][1][0]) ^ b1[0] ^ b1[1] ^ a[1] ^ a[3]);
                } else {
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
    if constexpr (LD == LD_CPA) cp_wait<0>();
    __syncwarp();                                                 // the staging area / ring becomes red[warp]

    // exl3.cu member_tiles<NT, W, 1>: the warps' partial sums through shared memory, added in warp order
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        Z[(((size_t)mat * SK + split) * P + r) * N + blockIdx.y * NT * 16 + col] = s;
    }
}

} // namespace alaska_expert_loads
