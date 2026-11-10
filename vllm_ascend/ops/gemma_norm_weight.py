# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare Gemma's effective scale outside the inference/capture hot path."""

import torch
from vllm.model_executor.layers.layernorm import GemmaRMSNorm


def gemma_norm_weight(norm: GemmaRMSNorm) -> torch.Tensor:
    effective = getattr(norm, "_ascend_effective_weight", None)
    if effective is not None and not torch.is_grad_enabled():
        return effective
    return 1.0 + norm.weight


@torch.no_grad()
def refresh_gemma_norm_weight(norm: GemmaRMSNorm) -> None:
    """Refresh in place so existing graphs retain the effective-weight address.

    The caller must retire inference readers before changing raw weights.
    Shape/device/dtype changes require rebuilding the model's captured graphs.
    """
    raw = norm.weight
    effective = getattr(norm, "_ascend_effective_weight", None)
    if effective is None:
        norm.register_buffer("_ascend_effective_weight", 1.0 + raw, persistent=False)
    else:
        if effective.shape != raw.shape or effective.dtype != raw.dtype or effective.device != raw.device:
            raise RuntimeError("Gemma norm weight layout changed; rebuild effective buffers and captured graphs.")
        torch.add(raw, 1.0, out=effective)


def refresh_model_gemma_norm_weights(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        # Offloaded CPU weights must keep the original on-demand path.
        if isinstance(module, GemmaRMSNorm) and module.weight.device.type == "npu":
            refresh_gemma_norm_weight(module)
            count += 1
    return count
