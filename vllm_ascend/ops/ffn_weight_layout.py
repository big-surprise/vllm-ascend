# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Load-time FFN layout; computation and prefetch share the same Parameter."""

import torch
import torch_npu
from vllm.logger import init_logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import is_950

ACL_FORMAT_FRACTAL_NZ = 29
NZ_BLOCK_SIZE = 16
FFN_PROJECTION_SUFFIXES = (".mlp.gate_up_proj", ".mlp.down_proj")
logger = init_logger(__name__)


def maybe_prepare_ffn_nz_weight(layer: torch.nn.Module) -> bool:
    """Return True only when this layer uses the explicitly enabled layout.

    Called after checkpoint/TP sharding and before graph capture. The Parameter
    identity, loader metadata and logical [N,K] shape survive; only its storage
    changes to a [K,N] NZ base. Existing prefetch sites use layer.weight and
    therefore fetch exactly the allocation consumed by F.linear. Never call
    contiguous() on the final transposed view in the inference path.
    """
    if not isinstance(layer.prefix, str) or not layer.prefix.endswith(FFN_PROJECTION_SUFFIXES):
        return False
    if getattr(layer, "skip_weight_nz_conversion", False) or getattr(layer, "precast_fp32_weight", False):
        return False
    config = get_ascend_config()
    if getattr(config, "ffn_weight_layout", "default") != "nz_kn":
        return False
    weight = layer.weight
    if weight.device.type != "npu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return False
    if not is_950():
        raise ValueError("ffn_weight_layout=nz_kn has only been validated on Ascend950")
    if any(size % NZ_BLOCK_SIZE for size in weight.shape):
        raise ValueError("ffn_weight_layout=nz_kn requires FFN shard dimensions divisible by 16")
    if getattr(layer, "_ascend_ffn_nz_transposed", False):
        if torch_npu.get_npu_format(weight) != ACL_FORMAT_FRACTAL_NZ or weight.stride() != (1, weight.shape[0]):
            raise RuntimeError("FFN packed weight was replaced; reload the model and recapture graphs")
        return True
    packed = torch_npu.npu_format_cast(weight.detach().t().contiguous(), ACL_FORMAT_FRACTAL_NZ)
    layer.weight.data = packed.t()
    layer._ascend_ffn_nz_transposed = True
    logger.debug("Prepared FFN [K,N] NZ weight: %s shape=%s", layer.prefix, tuple(weight.shape))
    logger.info_once("Enabled Qwen3.5 BF16 FFN [K,N] NZ weights; existing weight-prefetch boundaries retained")
    return True
