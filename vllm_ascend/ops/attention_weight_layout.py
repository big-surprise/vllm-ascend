# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Load-time attention input layouts shared by prefill, decode and prefetch."""

import torch
import torch_npu
from vllm.logger import init_logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import is_950

ACL_FORMAT_FRACTAL_NZ = 29
NZ_BLOCK_SIZE = 16
ATTENTION_INPUT_LAYOUTS = (
    (".linear_attn.in_proj_qkvz", "linear_attention_qkvz_layout"),
    (".self_attn.qkv_proj", "full_attention_qkv_layout"),
)
logger = init_logger(__name__)


def maybe_prepare_attention_nz_weight(layer: torch.nn.Module) -> bool:
    """Replace storage after TP sharding, retaining one canonical Parameter.

    F.linear and existing prefetch sites both consume layer.weight. Prefill
    and decode use the same allocation, without a token gate or ND cache.
    """
    if not isinstance(layer.prefix, str):
        return False
    option = next((key for suffix, key in ATTENTION_INPUT_LAYOUTS if layer.prefix.endswith(suffix)), None)
    if option is None:
        return False
    if getattr(layer, "skip_weight_nz_conversion", False) or getattr(layer, "precast_fp32_weight", False):
        return False
    if getattr(get_ascend_config(), option, "default") != "nz_kn":
        return False
    weight = layer.weight
    if weight.device.type != "npu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return False
    if not is_950():
        raise ValueError(f"{option}=nz_kn has only been validated on Ascend950")
    if any(size % NZ_BLOCK_SIZE for size in weight.shape):
        raise ValueError(f"{option}=nz_kn requires shard dimensions divisible by 16")
    if getattr(layer, "_ascend_attention_nz_transposed", False):
        if torch_npu.get_npu_format(weight) != ACL_FORMAT_FRACTAL_NZ or weight.stride() != (1, weight.shape[0]):
            raise RuntimeError("Attention packed weight was replaced; reload the model and recapture graphs")
        return True
    packed = torch_npu.npu_format_cast(weight.detach().t().contiguous(), ACL_FORMAT_FRACTAL_NZ)
    layer.weight.data = packed.t()
    layer._ascend_attention_nz_transposed = True
    logger.debug("Prepared attention [K,N] NZ weight: %s shape=%s", layer.prefix, tuple(weight.shape))
    logger.info_once("Enabled attention input [K,N] NZ weights for prefill and decode with one resident weight copy")
    return True
