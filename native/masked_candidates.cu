// Grammar-aware finite BF16 candidate collection.
// One mask bit per global token, 1 = allowed.
#include <stdint.h>
__device__ unsigned masked_key(uint16_t b, unsigned greedy) {
    // Greedy compares zero numerically; stochastic local top-k retains ATen's
    // signed-zero radix ordering. Payloads retain the original logit bits.
    if (greedy && !(b & 0x7fff)) b = 0;
    return b & 0x8000 ? unsigned(b ^ 0xffff) : unsigned(b ^ 0x8000);
}
__device__ bool allowed(const uint32_t* mask, unsigned id) {
    return (mask[id >> 5] >> (id & 31)) & 1;
}
extern "C" __global__ void masked_candidates(const uint16_t* x,
 const uint32_t* mask, unsigned k, unsigned base, unsigned greedy, uint32_t* out) {
    __shared__ unsigned hist[256], high, threshold, remaining, eq_seen, count, limit;
    __shared__ unsigned eq_warp[8], take_warp[8], invalid, total;
    unsigned t=threadIdx.x, lane=t&31, warp=t>>5;
    hist[t]=0;
    if(!t){invalid=0;count=0;eq_seen=0;total=0;high=0;threshold=0;remaining=0;}
    for(unsigned j=t;j<2*k;j+=256)out[j]=0;
    __syncthreads();
    for(unsigned i=t;i<38720;i+=256){
        uint16_t b=x[i];
        if((b&0x7f80)==0x7f80)atomicOr(&invalid,1);
        if(allowed(mask,base+i))atomicAdd(&hist[masked_key(b,greedy)>>8],1);
    }
    __syncthreads();
    if(!t){
        for(unsigned j=0;j<256;j++)total+=hist[j];
        limit=total<k?total:k;
        unsigned n=limit;
        if(n)for(int j=255;j>=0;j--){if(n>hist[j])n-=hist[j];else{high=j;remaining=n;break;}}
    }
    __syncthreads();
    // A shard with no allowed token is legitimate. The global unpacker requires
    // at least one candidate across ranks before a result can be consumed.
    if(!limit){if(!t){out[2*k]=invalid;out[2*k+1]=0;}return;}
    hist[t]=0;__syncthreads();
    for(unsigned i=t;i<38720;i+=256){
        unsigned v=masked_key(x[i],greedy);
        if(allowed(mask,base+i)&&(v>>8)==high)atomicAdd(&hist[v&255],1);
    }
    __syncthreads();
    if(!t){unsigned n=remaining;for(int j=255;j>=0;j--){if(n>hist[j])n-=hist[j];else{threshold=(high<<8)|j;remaining=n;break;}}}
    __syncthreads();
    for(unsigned start=0;start<38720;start+=256){
        unsigned i=start+t;uint16_t b=i<38720?x[i]:0;unsigned v=masked_key(b,greedy);
        bool permit=i<38720&&allowed(mask,base+i);
        unsigned eq=__ballot_sync(0xffffffff,permit&&v==threshold);
        if(!lane)eq_warp[warp]=__popc(eq);__syncthreads();
        unsigned before=eq_seen+__popc(eq&((1u<<lane)-1));for(unsigned w=0;w<warp;w++)before+=eq_warp[w];
        bool take=permit&&(v>threshold||(v==threshold&&before<remaining));
        unsigned ballot=__ballot_sync(0xffffffff,take);
        if(!lane)take_warp[warp]=__popc(ballot);__syncthreads();
        unsigned at=count+__popc(ballot&((1u<<lane)-1));for(unsigned w=0;w<warp;w++)at+=take_warp[w];
        if(take&&at<limit){out[2*at]=uint32_t(b)<<16;out[2*at+1]=base+i;}
        __syncthreads();
        if(!t)for(unsigned w=0;w<8;w++){eq_seen+=eq_warp[w];count+=take_warp[w];}
        __syncthreads();
    }
    if(!t){out[2*k]=invalid|(count!=limit?2u:0u);out[2*k+1]=count;}
}
// Pack finite candidates, then pad the sampler width with zero-mass -inf rows.
// Frozen grammar sampling sums the full kept top-k length, including its zeros;
// reducing only the finite count could change rounded nucleus boundaries.
// Bad payloads have a safe one-token placeholder, but status forbids publication.
extern "C" __global__ void masked_candidates_unpack(const uint32_t* in,unsigned k,
 uint32_t* values,uint32_t* ids,uint64_t* ints,unsigned top_k,unsigned* status) {
    __shared__ unsigned counts[4],offsets[4],total,invalid;
    unsigned t=threadIdx.x;
    if(!t){
        total=0;invalid=0;
        for(unsigned rank=0;rank<4;rank++){
            unsigned tail=(rank*(k+1)+k)*2,n=in[tail+1];
            if(in[tail]||n>k)invalid|=1;
            counts[rank]=n>k?0:n;offsets[rank]=total;total+=counts[rank];
        }
        if(!total||total>256)invalid|=4;
    }
    __syncthreads();
    if(t<4*k){
        unsigned rank=t/k,j=t%k,off=(rank*(k+1)+j)*2;
        unsigned value=in[off],id=in[off+1];
        if(j<counts[rank]){
            bool valid=(value&0x7f800000)!=0x7f800000&&!(value&0xffff)&&
                id>=rank*38720&&id<(rank+1)*38720&&(!j||id>in[off-1]);
            if(!valid)atomicOr(&invalid,2);
            unsigned dst=offsets[rank]+j;
            if(dst<256){values[dst]=valid?value:0;ids[dst]=valid?id:dst;}
        }else if(value||id)atomicOr(&invalid,8);
    }
    __syncthreads();
    if(t>=total&&t<4*k){values[t]=0xff800000u;ids[t]=154880u+t;}
    __syncthreads();
    if(!t){
        if(invalid){values[0]=0;ids[0]=0;ints[0]=1;ints[1]=1;}
        else{ints[0]=4*k;ints[1]=top_k&&top_k<4*k?top_k:4*k;}
        *status=invalid;
    }
}
