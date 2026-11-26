// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(MegaGdnDecodeStridedBatchTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, num_k_heads);
TILING_DATA_FIELD_DEF(int64_t, num_v_heads);
TILING_DATA_FIELD_DEF(int64_t, num_slots);
TILING_DATA_FIELD_DEF(int64_t, conv_slot_stride);
TILING_DATA_FIELD_DEF(int64_t, ssm_slot_stride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MegaGdnDecodeStridedBatch, MegaGdnDecodeStridedBatchTilingData)
}
