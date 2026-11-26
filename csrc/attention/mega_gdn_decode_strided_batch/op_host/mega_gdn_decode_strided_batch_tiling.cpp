// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "mega_gdn_decode_strided_batch_tiling.h"
namespace optiling {
namespace {
bool Shape(const gert::Shape& shape, std::initializer_list<int64_t> dims) {
  if (shape.GetDimNum() != dims.size()) return false;
  size_t i = 0;
  for (auto dim : dims) if (shape.GetDim(i++) != dim) return false;
  return true;
}
bool Fits(int64_t slots, int64_t stride, int64_t row, int64_t bytes, int64_t capacity) {
  if (slots < 1 || slots > std::numeric_limits<int32_t>::max() || stride < row) return false;
  const int64_t max_elems = std::numeric_limits<int64_t>::max() / bytes;
  if (slots - 1 > (max_elems - row) / stride) return false;
  return (slots - 1) * stride + row <= capacity;
}
}
static ge::graphStatus Tiling(gert::TilingContext* ctx) {
  const auto* attrs = ctx->GetAttrs();
  if (!attrs) return ge::GRAPH_FAILED;
  const auto* slots = attrs->GetAttrPointer<int64_t>(0);
  const auto* conv_stride = attrs->GetAttrPointer<int64_t>(1);
  const auto* ssm_stride = attrs->GetAttrPointer<int64_t>(2);
  const auto* fla = attrs->GetAttrPointer<bool>(3);
  if (!slots || !conv_stride || !ssm_stride || !fla) return ge::GRAPH_FAILED;
  constexpr std::array<ge::DataType,12> types = {ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,
    ge::DT_BF16,ge::DT_BF16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_INT32,ge::DT_INT32,ge::DT_BF16};
  for (size_t i=0;i<types.size();++i) {
    if (!ctx->GetInputShape(i) || !ctx->GetInputDesc(i) || ctx->GetInputDesc(i)->GetDataType()!=types[i]) return ge::GRAPH_FAILED;
  }
  const auto& q = ctx->GetInputShape(0)->GetStorageShape();
  const auto& z = ctx->GetInputShape(1)->GetStorageShape();
  if (q.GetDimNum()!=2 || (q.GetDim(0)<1 || q.GetDim(0)>32) || z.GetDimNum()!=3 || z.GetDim(0)!=q.GetDim(0) || z.GetDim(2)!=128) return ge::GRAPH_FAILED;
  const int64_t batch=q.GetDim(0), nv=z.GetDim(1), c=q.GetDim(1), qk=c-nv*128;
  if (qk<256 || qk%256) return ge::GRAPH_FAILED;
  const int64_t nk=qk/256;
  if (nk<1 || nk>16 || (nk&(nk-1)) || nv%nk || nv/nk<1 || nv/nk>4) return ge::GRAPH_FAILED;
  const auto& conv=ctx->GetInputShape(5)->GetStorageShape();
  const auto& ssm=ctx->GetInputShape(8)->GetStorageShape();
  if (conv.GetDimNum()!=1 || ssm.GetDimNum()!=1 ||
      !Fits(*slots,*conv_stride,3*c,2,conv.GetDim(0)) || !Fits(*slots,*ssm_stride,nv*128*128,4,ssm.GetDim(0)) ||
      !Shape(ctx->GetInputShape(2)->GetStorageShape(),{batch,nv}) || !Shape(ctx->GetInputShape(3)->GetStorageShape(),{batch,nv}) ||
      !Shape(ctx->GetInputShape(4)->GetStorageShape(),{4,c}) || !Shape(ctx->GetInputShape(6)->GetStorageShape(),{nv}) ||
      !Shape(ctx->GetInputShape(7)->GetStorageShape(),{nv}) || !Shape(ctx->GetInputShape(9)->GetStorageShape(),{batch}) ||
      !Shape(ctx->GetInputShape(10)->GetStorageShape(),{batch}) || !Shape(ctx->GetInputShape(11)->GetStorageShape(),{128})) return ge::GRAPH_FAILED;
  auto platform=platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
  if (platform.GetSocVersion()!=platform_ascendc::SocVersion::ASCEND950) return ge::GRAPH_FAILED;
  uint64_t ub=0; platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
  if (ub<178688) return ge::GRAPH_FAILED;
  MegaGdnDecodeStridedBatchTilingData data;
  data.set_batch_size(batch);data.set_num_k_heads(nk); data.set_num_v_heads(nv);data.set_num_slots(*slots);
  data.set_conv_slot_stride(*conv_stride);data.set_ssm_slot_stride(*ssm_stride);
  data.SaveToBuffer(ctx->GetRawTilingData()->GetData(),ctx->GetRawTilingData()->GetCapacity());
  ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
  ctx->SetTilingKey((batch==1?2:1)+(*fla?0:10));
  ctx->SetBlockDim(std::min<uint32_t>(std::max<int64_t>((c/128)*((batch+1)/2),batch*nv),platform.GetCoreNumAiv()));
  if (ctx->SetScheduleMode(1)!=ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED;
  ctx->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaGdnDecodeStridedBatch).Tiling(Tiling);
}
