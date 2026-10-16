// CUDA transcription of the keyed GPU sampler, Copyright 2026 Jay Leaton.
// Apache-2.0; see LICENSES/Apache-2.0.txt and LICENSES/Native-Components-NOTICE.txt.
// Modified from Triton to CUDA C++ with a native pointer-based interface.
// Log/exp use Arm Optimized Routines (MIT; see LICENSES/Arm-MIT.txt).
// No fast math.
// Keep explicit fma only at the original fused operations. Compile --fmad=false.
#include <stdint.h>
#include <math.h>
#include "sampling_tables.cuh"
__device__ double bits(uint64_t x) { return __longlong_as_double((long long)x); }
__device__ uint64_t raw(double x) { return (uint64_t)__double_as_longlong(x); }
__device__ double c(int i) { return bits(constants[i]); }
__device__ double ln_exact(double x) {
    uint64_t ix=raw(x);
    if ((ix << 1)==0) return -INFINITY;
    if (ix==0x3ff0000000000000ULL) return 0.0;
    if (ix-0x3fee000000000000ULL <= 0x308ffffffffffULL) {
        double r=x-c(26), d1=__fma_rn(r,c(25),r), r2=r*r;
        double d4=__fma_rn(c(15),r,c(14)), rhi=__fma_rn(-r,c(25),d1), r3=r*r2;
        d4=__fma_rn(c(16),r2,d4);
        double d3=__fma_rn(c(12),r,c(11));
        d4=__fma_rn(c(17),r3,d4);
        double hh=rhi*rhi; d3=__fma_rn(c(13),r2,d3);
        double hi=__fma_rn(hh,c(7),r), d2=__fma_rn(c(9),r,c(8));
        d3=__fma_rn(d4,r3,d3);
        double rlo=r-rhi, rs=r+rhi; d2=__fma_rn(c(10),r2,d2);
        double b0rlo=c(7)*rlo, lo=r-hi;
        lo=__fma_rn(hh,c(7),lo); d2=__fma_rn(d3,r3,d2);
        lo=__fma_rn(b0rlo,rs,lo); lo=__fma_rn(d2,r3,lo);
        return hi+lo;
    }
    uint64_t tmp=ix-0x3fe6000000000000ULL;
    int i=(tmp>>45)&127; int64_t k=((int64_t)tmp)>>52;
    double kd=(double)(int)k, invc=bits(log_table[2*i]), logc=bits(log_table[2*i+1]);
    double z=bits(ix-(tmp&0xfff0000000000000ULL)), w=__fma_rn(c(0),kd,logc);
    double r=__fma_rn(invc,z,c(27)), hi=r+w;
    double p34=__fma_rn(c(6),r,c(5)), p12=__fma_rn(c(4),r,c(3));
    double r2=r*r, t=w-hi, r3=r*r2; t=t+r;
    double p=__fma_rn(p34,r2,p12); t=__fma_rn(c(1),kd,t);
    t=__fma_rn(c(2),r2,t); t=__fma_rn(r3,p,t);
    return t+hi;
}
__device__ double exp_exact(double x) {
    uint64_t ix=raw(x); int abstop=(ix>>52)&0x7ff;
    if(abstop<0x3c9) return 1.0+x;
    if(abstop>=0x409) {
        if(ix==0xfff0000000000000ULL) return 0.0;
        if(abstop==0x7ff) return 1.0+x;
        return ix>>63 ? 0.0 : INFINITY;
    }
    double z=x*c(18); int64_t ki=(int64_t)z;
    double f=z-(double)ki; ki+=(f>=0.5)-(f<=-0.5);
    double kd=(double)ki, d0=__fma_rn(c(19),kd,x), r=__fma_rn(c(20),kd,d0);
    int idx=2*(ki&127); double tail=bits(exp_table[idx]);
    uint64_t sbits=exp_table[idx+1]+((uint64_t)ki<<45);
    double p23=__fma_rn(c(22),r,c(21)), p45=__fma_rn(c(24),r,c(23));
    double r2=r*r, d1=r+tail, r4=r2*r2;
    d1=__fma_rn(p23,r2,d1); double tmp=__fma_rn(r4,p45,d1);
    if(abstop!=0x408) { double s=bits(sbits);return __fma_rn(tmp,s,s); }
    if(!(ki&0x80000000LL)) {double sp=bits(sbits-0x3f10000000000000ULL);return __fma_rn(tmp,sp,sp)*c(28);}
    double sn=bits(sbits+0x3fe0000000000000ULL), st=tmp*sn, yn=sn+st;
    if(yn<1.0) {
        double hi=yn+1.0, a=sn-yn, b=1.0-hi, lo=a+st;
        a=b+yn;lo=a+lo;yn=(lo+hi)-1.0;
        if(yn==0.0) yn=0.0;
    }
    return yn*c(29);
}
__device__ uint64_t mix64(uint64_t x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
__device__ double uniform64(uint64_t seed,uint64_t pos,uint32_t id) {
    uint64_t x=mix64(seed+0x9e3779b97f4a7c15ULL);
    x=mix64(x^(pos*0xd1b54a32d192ed03ULL));x=mix64(x^id);
    return (double)(x>>11)*c(30)+c(31);
}
extern "C" __global__ void sampling_math(const double *x, double *y, uint64_t n, uint64_t seed) {
    uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {y[3*i]=ln_exact(x[i]);y[3*i+1]=exp_exact(-x[i]);y[3*i+2]=uniform64(seed,i,(uint32_t)(i%154880));}
}
// One block per independent row, fixed 256-candidate stride. Input is already
// collected from the vocabulary shards. Candidate collection is qualified later.
extern "C" __global__ void sampling_choose(const float *values,const uint32_t *ids,
 const uint64_t *ints,const double *params,uint32_t *out) {
    __shared__ double scaled[256],prob[256],scores[256];
    __shared__ uint32_t ordered[256];
    __shared__ int keep;
    int row=blockIdx.x,t=threadIdx.x,n=(int)ints[row*4],k=(int)ints[row*4+1];
    double temp=params[row*2],top=params[row*2+1];
    int at=row*256+t,r=0;
    if(t<n) {
        for(int j=0;j<n;j++) {
            float v=values[row*256+j];uint32_t id=ids[row*256+j];
            r+=(v>values[at] || (v==values[at] && id<ids[at]));
        }
        ordered[r]=ids[at];scaled[r]=(double)values[at]/(temp>1e-6?temp:1e-6);
        double u=uniform64(ints[row*4+2],ints[row*4+3],ids[at]);
        scores[r]=scaled[r]-ln_exact(-ln_exact(u));
    }
    __syncthreads();
    if(t<k && temp>0.0 && top>0.0 && top<1.0) prob[t]=exp_exact(scaled[t]-scaled[0]);
    __syncthreads();
    if(!t) {
        keep=temp>0.0?k:1;
        if(temp>0.0 && top>0.0 && top<1.0) {
            double sum=-0.0;
            if(k<8) {for(int j=0;j<k;j++)sum+=prob[j];}
            else {
                double a[8];for(int j=0;j<8;j++)a[j]=prob[j];
                for(int i=8;i<k-k%8;i+=8)for(int j=0;j<8;j++)a[j]+=prob[i+j];
                sum=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7]));
                for(int j=k-k%8;j<k;j++)sum+=prob[j];
            }
            double cs=0.0;keep=1;for(int j=0;j<k;j++){cs+=prob[j]/sum;keep+=cs<top;}
            if(keep>k)keep=k;
        }
        int best=0;
        if(temp>0.0)for(int j=1;j<keep;j++)if(scores[j]>scores[best])best=j;
        out[2*row]=ordered[best];out[2*row+1]=keep;
    }
}
