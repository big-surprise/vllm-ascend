# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Overlap Qwen3.5 single-token BA projection with QKVZ on Ascend950."""

import torch
import torch_npu
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.linear import AscendMergedColumnParallelLinear, AscendUnquantizedLinearMethod
from vllm_ascend.utils import attention_calculation_stream, npu_stream_switch

_HIDDEN_SIZE = 5120
_BA_REDUCE_BLOCK = 8192
_FORMAT_ND = 2


def gdn_ba_overlap_enabled() -> bool:
    # The config module may import ops while the worker is initializing.
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        config = get_ascend_config()
    except RuntimeError:
        return False
    return config.ascend_fusion_config.overlap_gdn_qkvz_ba and get_current_hardware_profile().supports(
        HardwareCapability.GDN_BA_VECTOR_OVERLAP
    )


def _is_local_unquantized_projection(projection: torch.nn.Module) -> bool:
    # Only bypass Linear.forward when it reduces to a local, bias-free GEMM.
    return (
        type(projection) is AscendMergedColumnParallelLinear
        and type(projection.quant_method) is AscendUnquantizedLinearMethod
        and projection.custom_op is None
        and not projection.gather_output
        and projection.bias is None
        and projection.weight.dtype == torch.bfloat16
    )


def can_overlap_gdn_qkvz_ba(layer: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    return (
        hidden_states.device.type == "npu"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.shape == (1, _HIDDEN_SIZE)
        and gdn_ba_overlap_enabled()
        and layer.tp_size > 0
        and layer.num_spec == 0
        and not layer.disable_tp_for_ba_proj
        and _is_local_unquantized_projection(layer.in_proj_qkvz)
        and _is_local_unquantized_projection(layer.in_proj_ba)
        and layer.in_proj_qkvz.tp_size == layer.tp_size
        and layer.in_proj_ba.tp_size == layer.tp_size
        and (2 * layer.key_dim + 2 * layer.value_dim) % layer.tp_size == 0
        and (2 * layer.num_v_heads) % layer.tp_size == 0
        and layer.in_proj_qkvz.weight.shape
        == ((2 * layer.key_dim + 2 * layer.value_dim) // layer.tp_size, _HIDDEN_SIZE)
        and layer.in_proj_ba.weight.shape == ((2 * layer.num_v_heads) // layer.tp_size, _HIDDEN_SIZE)
    )


def _is_single_decode(prefix: str) -> bool:
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    layer_metadata = metadata.get(prefix) if isinstance(metadata, dict) else None
    return (
        isinstance(layer_metadata, GDNAttentionMetadata)
        and layer_metadata.num_prefills == 0
        and layer_metadata.num_spec_decodes == 0
        and layer_metadata.spec_sequence_masks is None
        and layer_metadata.num_decodes == 1
        and layer_metadata.num_actual_tokens == 1
    )


@triton.jit
def _gdn_ba_vector_gemv(x, weight, output, K: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_K)
    activation = tl.load(x + col, col < K, other=0).to(tl.float32)
    row_weight = tl.load(weight + row * K + col, col < K, other=0).to(tl.float32)
    tl.store(output + row, tl.sum(activation * row_weight, axis=0))


def _gdn_qkvz_ba_projection(
    hidden_states: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor, prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Read the metadata at execution/capture time so it is not baked into
    # the Dynamo trace. Triton reads the two inputs using their ND addresses.
    # The Triton pointer arithmetic below reads ND storage. QKVZ may be NZ
    # because its projection still goes through F.linear.
    if not (
        _is_single_decode(prefix)
        and hidden_states.is_contiguous()
        and ba_weight.is_contiguous()
        and torch_npu.get_npu_format(hidden_states) == _FORMAT_ND
        and torch_npu.get_npu_format(ba_weight) == _FORMAT_ND
    ):
        return (
            torch.nn.functional.linear(hidden_states, qkvz_weight),
            torch.nn.functional.linear(hidden_states, ba_weight),
        )

    main = torch.npu.current_stream()
    side = attention_calculation_stream()
    # Allocate on the consumer stream. The fork orders the reads of the input
    # and the join orders the caller's reads of BA and the reuse of its storage.
    ba_rows = ba_weight.shape[0]
    ba = hidden_states.new_empty((1, ba_rows))
    side.wait_stream(main)
    with npu_stream_switch(side):
        _gdn_ba_vector_gemv[(ba_rows,)](
            hidden_states, ba_weight, ba, _HIDDEN_SIZE, _BA_REDUCE_BLOCK, enable_fp_fusion=False
        )
    qkvz = torch.nn.functional.linear(hidden_states, qkvz_weight)
    main.wait_stream(side)
    return qkvz, ba


def _gdn_qkvz_ba_projection_fake(
    hidden_states: torch.Tensor, qkvz_weight: torch.Tensor, ba_weight: torch.Tensor, prefix: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        hidden_states.new_empty((hidden_states.shape[0], qkvz_weight.shape[0])),
        hidden_states.new_empty((hidden_states.shape[0], ba_weight.shape[0])),
    )


direct_register_custom_op(
    op_name="ascend_gdn_qkvz_ba_projection",
    op_func=_gdn_qkvz_ba_projection,
    fake_impl=_gdn_qkvz_ba_projection_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
