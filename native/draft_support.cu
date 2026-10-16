// Rust DFlash2 plumbing. Canonical projections, norm, convolution and attention
// remain pinned kernels. These helpers replace host tensor indexing/pointwise ops.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
extern "C" __global__ void draft_rotary(const int64_t* pos,float* co,float* si,int rows) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*64)return;
 int d=i%64,r=i/64;
 float exponent=float(d)*2.f/128.f;
 float inv=1.f/powf(10000.f,exponent);
 float phase=float(*pos+r)*inv;
 co[i]=cosf(phase);si[i]=sinf(phase);
}
extern "C" __global__ void draft_store(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const int64_t* pos,int rows) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*rows*128)return;
 int h=i/(rows*128),r=(i/128)%rows,c=i%128;
 int out=(h*4096+int((*pos+r)%4096))*128+c;
 kc[out]=k[i];vc[out]=v[i];
}
extern "C" __global__ void draft_advance(int64_t* pos,int rows){if(threadIdx.x==0&&blockIdx.x==0)*pos+=rows;}
extern "C" __global__ void draft_embed(const int* ids,const __nv_bfloat16* w,__nv_bfloat16* x,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*4096)x[i]=w[int64_t(ids[i/4096])*4096+i%4096];
}
extern "C" __global__ void draft_rank_sum(const float* all,__nv_bfloat16* x,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float sum=all[i];for(int rank=1;rank<4;rank++)sum=__fadd_rn(sum,all[rank*count+i]);
 x[i]=__float2bfloat16_rn(sum);
}
