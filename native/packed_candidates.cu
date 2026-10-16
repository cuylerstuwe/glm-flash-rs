// Row-address-only adaptation of candidates.cu. Keep selection/validation math identical.
// One independent CTA per row; gathered input is rank-major, row-minor.
// Exact finite BF16 -> frozen ATen float32 top-k candidate set.
// Signed zeros are distinct for local radix selection; final draw compares them
// numerically. Equal-key cutoff ties select the first token IDs, as ATen mbtopk.
#include <stdint.h>
__device__ unsigned key(uint16_t b){return b&0x8000 ? unsigned(b^0xffff) : unsigned(b^0x8000);}
extern "C" __global__ void packed_candidates(const uint16_t* x, unsigned k, unsigned base,
                                      uint32_t* out) {
    x += blockIdx.x * 38720; out += blockIdx.x * (k + 1) * 2;
    __shared__ unsigned hist[256], high, threshold, remaining, eq_seen, count;
    __shared__ unsigned eq_warp[8], take_warp[8], invalid;
    unsigned t=threadIdx.x, lane=t&31, warp=t>>5;
    hist[t]=0;if(!t){invalid=0;count=0;eq_seen=0;}__syncthreads();
    for(unsigned i=t;i<38720;i+=256){uint16_t b=x[i];if((b&0x7f80)==0x7f80)atomicOr(&invalid,1);atomicAdd(&hist[key(b)>>8],1);}
    __syncthreads();
    if(!t){unsigned n=k;for(int j=255;j>=0;j--){if(n>hist[j]){n-=hist[j];}else{high=j;remaining=n;break;}}}
    __syncthreads();hist[t]=0;__syncthreads();
    for(unsigned i=t;i<38720;i+=256){unsigned v=key(x[i]);if(v>>8==high)atomicAdd(&hist[v&255],1);}
    __syncthreads();
    if(!t){unsigned n=remaining;for(int j=255;j>=0;j--){if(n>hist[j]){n-=hist[j];}else{threshold=(high<<8)|j;remaining=n;break;}}}
    __syncthreads();
    for(unsigned start=0;start<38720;start+=256){
        unsigned i=start+t;uint16_t b=i<38720?x[i]:0;unsigned v=key(b);
        unsigned eq=__ballot_sync(0xffffffff,i<38720&&v==threshold);
        if(!lane)eq_warp[warp]=__popc(eq);__syncthreads();
        unsigned before=eq_seen+__popc(eq&((1u<<lane)-1));for(unsigned w=0;w<warp;w++)before+=eq_warp[w];
        bool take=i<38720&&(v>threshold||(v==threshold&&before<remaining));
        unsigned ballot=__ballot_sync(0xffffffff,take);
        if(!lane)take_warp[warp]=__popc(ballot);__syncthreads();
        unsigned at=count+__popc(ballot&((1u<<lane)-1));for(unsigned w=0;w<warp;w++)at+=take_warp[w];
        if(take&&at<k){out[2*at]=uint32_t(b)<<16;out[2*at+1]=base+i;}
        __syncthreads();
        if(!t)for(unsigned w=0;w<8;w++){eq_seen+=eq_warp[w];count+=take_warp[w];}
        __syncthreads();
    }
    if(!t){out[2*k]=invalid|(count!=k?2u:0u);out[2*k+1]=count;}
}
// Compact draw input. The owner zeros status on this same stream before launch.
// Invalid collective payloads remain safe to process but cannot be consumed.
extern "C" __global__ void packed_candidates_unpack(const uint32_t* in, unsigned k, unsigned rows,
                                             uint32_t* values, uint32_t* ids,
                                             unsigned* status) {
    unsigned row=blockIdx.x; values += row * 256; ids += row * 256; status += row;
    unsigned t=threadIdx.x;
    if(t<4){unsigned tail=((t*rows+row)*(k+1)+k)*2;if(in[tail]||in[tail+1]!=k)atomicOr(status,1);}
    if(t<4*k){
        unsigned rank=t/k, j=t%k, off=((rank*rows+row)*(k+1)+j)*2;
        unsigned value=in[off], id=in[off+1];
        bool valid=(value&0x7f800000)!=0x7f800000 && !(value&0xffff) &&
                   id>=rank*38720 && id<(rank+1)*38720 && (!j||id>in[off-1]);
        if(!valid)atomicOr(status,2);
        values[t]=valid?value:0;ids[t]=valid?id:t;
    }
}
