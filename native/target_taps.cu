// Canonical sequential FP32 four-stream mean, stored as BF16 in a bounded ring.
// Only the five requested layer outputs are observed; target state is read-only.
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" __global__ void target_taps(const __nv_bfloat16* x, __nv_bfloat16* ring,
 const int* position, int first, int rows, int cap, int layer) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=rows*4096)return;
 int r=i/4096,c=i%4096; const __nv_bfloat16* p=x+(r+first)*16384+c;
 float v=__bfloat162float(p[0]);
 v=__fadd_rn(v,__bfloat162float(p[4096]));
 v=__fadd_rn(v,__bfloat162float(p[8192]));
 v=__fadd_rn(v,__bfloat162float(p[12288]));
 ring[(((*position+first+r)%cap)*5+layer)*4096+c]=__float2bfloat16_rn(v*.25f);
}
