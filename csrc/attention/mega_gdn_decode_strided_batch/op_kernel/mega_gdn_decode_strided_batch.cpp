// SPDX-License-Identifier: Apache-2.0
#include "mega_gdn_decode_strided_batch_pto_kernel.h"
struct MegaGdnDecodeStridedBatchTilingData {
  int64_t batch_size, num_k_heads, num_v_heads, num_slots, conv_slot_stride, ssm_slot_stride;
};
extern "C" __global__ __aicore__ void mega_gdn_decode_strided_batch(
 GM_ADDR qkv, GM_ADDR z, GM_ADDR b, GM_ADDR a, GM_ADDR convWeight, GM_ADDR convState,
 GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR ssmState, GM_ADDR readStateIndices, GM_ADDR writeStateIndices,
 GM_ADDR normWeight, GM_ADDR convOut, GM_ADDR convStateOut, GM_ADDR ssmStateOut, GM_ADDR out,
 GM_ADDR workspace, GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 REGISTER_TILING_DEFAULT(MegaGdnDecodeStridedBatchTilingData);
 GET_TILING_DATA_WITH_STRUCT(MegaGdnDecodeStridedBatchTilingData,data,tiling);
 (void)workspace;
 const int32_t read=*reinterpret_cast<__gm__ int32_t*>(readStateIndices);
 const int32_t write=*reinterpret_cast<__gm__ int32_t*>(writeStateIndices);
 // Zero is a reserved graph-padding slot. Invalid slots never access state.
 if (data.batch_size==1 && (read<=0 || write<=0 || read>=data.num_slots || write>=data.num_slots)) {
#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
   using namespace mega_gdn_decode_strided_batch_pto;
   TileUbDataND<uint16_t,1,128> zeros;
   TASSIGN(zeros,0);
   TEXPANDS(zeros,static_cast<uint16_t>(0));
   set_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);
   wait_flag(PIPE_V,PIPE_MTE3,EVENT_ID0);
   const int64_t tiles=2*data.num_k_heads+data.num_v_heads;
   for (int64_t i=get_block_idx();i<tiles;i+=get_block_num())
     CopyUbToGm<uint16_t,uint16_t,1,1,1,1,128,1,1,1,128,1,1,128>(
       reinterpret_cast<__gm__ uint16_t*>(convOut)+i*128,0,0,1,128);
   for (int64_t i=get_block_idx();i<data.num_v_heads;i+=get_block_num())
     CopyUbToGm<uint16_t,uint16_t,1,1,1,1,128,1,1,1,128,1,1,128>(
       reinterpret_cast<__gm__ uint16_t*>(out)+i*128,0,0,1,128);
   pipe_barrier(PIPE_ALL);
#endif
   return;
 }
#define RUN(B,F) mega_gdn_decode_strided_batch_pto::Run<B,F>( \
 reinterpret_cast<__gm__ bfloat16_t*>(qkv),reinterpret_cast<__gm__ bfloat16_t*>(z), \
 reinterpret_cast<__gm__ bfloat16_t*>(b),reinterpret_cast<__gm__ bfloat16_t*>(a), \
 reinterpret_cast<__gm__ bfloat16_t*>(convWeight),reinterpret_cast<__gm__ bfloat16_t*>(convState), \
 reinterpret_cast<__gm__ float*>(aLog),reinterpret_cast<__gm__ float*>(dtBias),reinterpret_cast<__gm__ float*>(ssmState), \
 reinterpret_cast<__gm__ int*>(readStateIndices),reinterpret_cast<__gm__ int*>(writeStateIndices), \
 reinterpret_cast<__gm__ bfloat16_t*>(normWeight),reinterpret_cast<__gm__ bfloat16_t*>(convOut), \
 reinterpret_cast<__gm__ bfloat16_t*>(convStateOut),reinterpret_cast<__gm__ float*>(ssmStateOut), \
 reinterpret_cast<__gm__ bfloat16_t*>(out),data.num_k_heads,data.num_v_heads,data.batch_size,data.conv_slot_stride,data.ssm_slot_stride,data.num_slots)
 if constexpr (TILING_KEY_IS(2)) {RUN(true,true);}
 else if constexpr (TILING_KEY_IS(12)) {RUN(true,false);}
 else if constexpr (TILING_KEY_IS(1)) {RUN(false,true);}
 else if constexpr (TILING_KEY_IS(11)) {RUN(false,false);}
#undef RUN
}
#include "lib/matmul_intf.h"
