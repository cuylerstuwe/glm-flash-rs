// Primitive native ATen boundary. No model layers, weights, image cache or
// orchestration lives here; those remain in Rust. No Python or runtime JIT.
#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

struct Context { c10::cuda::CUDAStream stream=c10::cuda::getStreamFromPool(false,0); };
static void need(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<class F> static int checked(char* error,size_t cap,F&& f) noexcept {
  if(error&&cap)error[0]=0;
  try { f();return 0; }
  catch(const std::exception& e){if(error&&cap){std::strncpy(error,e.what(),cap-1);error[cap-1]=0;}return 1;}
  catch(...){if(error&&cap){std::strncpy(error,"unknown native vision error",cap-1);error[cap-1]=0;}return 1;}
}
static at::ScalarType dtype(int n){switch(n){case 1:return at::kBFloat16;case 2:return at::kFloat;case 3:return at::kLong;default:throw std::runtime_error("vision dtype");}}
static at::Tensor& tensor(void* p){need(p,"null vision tensor");return *static_cast<at::Tensor*>(p);}
extern "C" {
const char* alaska_vision_abi(){return "alaska-vision-primitives-v1";}
int alaska_vision_context_new(void** output,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(output,"null output");*output=new Context;});
}
int alaska_vision_context_free(void* context,char* error,size_t cap) noexcept {
  // Final tensor handles may have gone while their dedicated stream still has
  // queued kernels. Never release native code until this stream has drained.
  // A failed drain preserves the context for the Rust owner's fail-closed leak.
  return checked(error,cap,[&]{need(context,"null context");auto* owner=static_cast<Context*>(context);owner->stream.synchronize();delete owner;});
}
void alaska_vision_tensor_free(void* p) noexcept {delete static_cast<at::Tensor*>(p);}
int alaska_vision_sync(void* context,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(context,"null context");static_cast<Context*>(context)->stream.synchronize();});
}
int alaska_vision_empty(void* context,int kind,int device,const int64_t* dims,size_t ndims,void** output,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(context&&output&&dims&&ndims>0&&ndims<=8,"tensor shape arguments");
    uint64_t n=1;for(size_t i=0;i<ndims;i++){need(dims[i]>0&&uint64_t(dims[i])<=(uint64_t(1)<<30)&&n<=(uint64_t(1)<<30)/uint64_t(dims[i]),"tensor geometry");n*=uint64_t(dims[i]);}
    need(device==-1||device==0,"tensor device");c10::cuda::CUDAStreamGuard guard(static_cast<Context*>(context)->stream);
    *output=new at::Tensor(at::empty(at::IntArrayRef(dims,ndims),at::TensorOptions().dtype(dtype(kind)).device(device==0?at::kCUDA:at::kCPU)));
  });
}
int alaska_vision_upload(void* context,void* value,uint64_t offset,const uint8_t* data,size_t bytes,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(context&&data&&bytes>0&&bytes<=8*1024*1024,"upload bound");auto& t=tensor(value);need(t.is_contiguous(),"upload contiguous tensor");const auto length=uint64_t(t.numel()*t.element_size());need(offset<=length&&bytes<=length-offset,"upload range");
    c10::cuda::CUDAStreamGuard guard(static_cast<Context*>(context)->stream);
    auto source=at::from_blob(const_cast<uint8_t*>(data),{int64_t(bytes)},at::TensorOptions().dtype(at::kByte));
    t.view(at::kByte).reshape({-1}).narrow(0,int64_t(offset),int64_t(bytes)).copy_(source,false);
  });
}
int alaska_vision_download(void* context,void* value,uint8_t* data,size_t bytes,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(context&&data,"download arguments");c10::cuda::CUDAStreamGuard guard(static_cast<Context*>(context)->stream);
    auto cpu=tensor(value).contiguous().to(at::kCPU);need(uint64_t(cpu.numel()*cpu.element_size())==bytes,"download length");std::memcpy(data,cpu.const_data_ptr(),bytes);
  });
}
int alaska_vision_info(void* value,int64_t* dims,size_t cap,size_t* ndims,int* kind,uint64_t* bytes,char* error,size_t error_cap) noexcept {
  return checked(error,error_cap,[&]{need(dims&&ndims&&kind&&bytes,"info arguments");auto& t=tensor(value);need(t.dim()<=int64_t(cap),"shape output cap");*ndims=t.dim();for(int64_t i=0;i<t.dim();i++)dims[i]=t.size(i);
    switch(t.scalar_type()){case at::kBFloat16:*kind=1;break;case at::kFloat:*kind=2;break;case at::kLong:*kind=3;break;default:throw std::runtime_error("unexpected native dtype");}*bytes=t.numel()*t.element_size();
  });
}
int alaska_vision_op(void* context,int op,void* const* inputs,size_t count,const int64_t* dims,size_t ndims,const double* scalars,size_t nscalars,void** output,char* error,size_t cap) noexcept {
  return checked(error,cap,[&]{need(context&&output&&count<=16&&ndims<=8&&nscalars<=2,"operation arguments");need((count==0||inputs)&&(ndims==0||dims)&&(nscalars==0||scalars),"operation pointers");
    c10::cuda::CUDAStreamGuard guard(static_cast<Context*>(context)->stream);at::Tensor out;
    const auto x=[&](size_t i)->at::Tensor&{need(i<count,"missing tensor operand");return tensor(inputs[i]);};
    const auto d=[&](size_t i){need(i<ndims,"missing integer operand");return dims[i];};
    const auto s=[&](size_t i){need(i<nscalars,"missing scalar operand");return scalars[i];};
    switch(op){
      case 1:need(count==2||count==3,"linear operand count");out=at::linear(x(0),x(1),count==3?x(2):at::Tensor());break;
      case 2:out=x(0).reshape(at::IntArrayRef(dims,ndims));break;
      case 3:out=x(0).permute(at::IntArrayRef(dims,ndims));break;
      case 4:out=x(0).transpose(d(0),d(1));break;
      case 5:out=x(0).slice(d(0),d(1),d(2));break;
      case 6:out=x(0).select(d(0),d(1));break;
      case 7:{std::vector<at::Tensor> values;for(size_t i=0;i<count;i++)values.push_back(x(i));out=at::cat(values,d(0));break;}
      case 8:out=x(0).to(dtype(int(d(0))));break;
      case 9:out=x(0)+x(1);break;
      case 10:out=x(0)*x(1);break;
      case 11:out=x(0)+s(0);break;
      case 12:out=x(0)*s(0);break;
      case 13:out=x(0)/s(0);break;
      case 14:out=x(0).pow(s(0));break;
      case 15:out=at::rsqrt(x(0));break;
      case 16:out=x(0).mean(at::IntArrayRef(dims,ndims),true);break;
      case 17:out=x(0).clamp(s(0),s(1));break;
      case 18:out=x(0).clamp_max(s(0));break;
      case 19:out=at::silu(x(0));break;
      case 20:out=x(0).cos();break;
      case 21:out=x(0).sin();break;
      case 22:out=at::scaled_dot_product_attention(x(0),x(1),x(2),std::nullopt,0.,false,s(0),false);break;
      case 23:out=at::layer_norm(x(0),at::IntArrayRef(dims,ndims),x(1),x(2),s(0),true);break;
      case 24:out=at::gelu(x(0),"none");break;
      case 25:out=at::arange(d(0),d(1),d(2),at::TensorOptions().dtype(dtype(int(d(3)))).device(d(4)==0?at::kCUDA:at::kCPU));break;
      case 26:out=at::pow(s(0),x(0));break;
      case 27:out=at::reciprocal(x(0));break;
      case 28:need(d(0)==-1||d(0)==0,"copy device");out=x(0).to(d(0)==0?at::kCUDA:at::kCPU);break;
      case 29:out=x(0).unsqueeze(d(0));break;
      case 30:out=x(0).expand(at::IntArrayRef(dims,ndims));break;
      case 31:out=-x(0);break;
      default:throw std::runtime_error("unknown primitive");
    }
    need(out.defined()&&out.numel()>0&&uint64_t(out.numel())<=(uint64_t(1)<<30),"primitive result bound");*output=new at::Tensor(std::move(out));
  });
}
}
