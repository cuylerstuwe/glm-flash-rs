// No-Python C ABI for the l2pf.cu kernels.
// Launch ordering, table ownership and the retained 8 MiB policy belong to Rust.
#include <cuda_runtime.h>
#include <stdint.h>

namespace tfl2pf {
cudaError_t launch_segments(const int64_t*, int, int, int, uint32_t, bool, cudaStream_t);
cudaError_t launch_experts(const int32_t*, const int32_t*, int, const int64_t*, int,
                           int64_t, int64_t, int64_t, int, int, uint32_t, bool, cudaStream_t);
}
extern "C" uint32_t alaska_l2pf_abi() { return 1; }
extern "C" int alaska_l2pf_properties(int* sms, int* l2) {
    if (!sms || !l2) return int(cudaErrorInvalidValue);
    int device = 0;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return int(error);
    error = cudaDeviceGetAttribute(sms, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess) return int(error);
    return int(cudaDeviceGetAttribute(l2, cudaDevAttrL2CacheSize, device));
}
extern "C" int alaska_l2pf_capturing(void* stream, int* active) {
    if (!stream || !active) return int(cudaErrorInvalidValue);
    cudaStreamCaptureStatus status;
    auto error = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status);
    if (error != cudaSuccess) return int(error);
    if (status == cudaStreamCaptureStatusInvalidated) return int(cudaErrorStreamCaptureInvalidated);
    *active = status == cudaStreamCaptureStatusActive;
    return int(cudaSuccess);
}
extern "C" int alaska_l2pf_segments(uint64_t table, int count, int grid, void* stream) {
    if (!table || count <= 0 || grid <= 0 || !stream) return int(cudaErrorInvalidValue);
    return int(tfl2pf::launch_segments(reinterpret_cast<const int64_t*>(table), count,
                                      grid, 128, 32 * 1024, true,
                                      static_cast<cudaStream_t>(stream)));
}
extern "C" int alaska_l2pf_experts(uint64_t ids, uint64_t count, int maxu, uint64_t bases,
                                   int64_t stride, int64_t bytes, int64_t experts,
                                   int grid, void* stream) {
    if (!ids || !count || !bases || maxu <= 0 || experts <= 0 || stride <= 0 ||
        bytes <= 0 || bytes > stride || bytes > 64 * 1024 || bytes % 16 ||
        stride % 16 || grid <= 0 || !stream) return int(cudaErrorInvalidValue);
    return int(tfl2pf::launch_experts(reinterpret_cast<const int32_t*>(ids),
        reinterpret_cast<const int32_t*>(count), maxu, reinterpret_cast<const int64_t*>(bases),
        2, stride, bytes, experts, grid, 128, 32 * 1024, true, static_cast<cudaStream_t>(stream)));
}
