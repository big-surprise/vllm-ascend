# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.forward_context import override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.fla.split_gdn_projection import split_gdn_projection
from vllm_ascend.utils import attention_calculation_stream


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("nz_qkvz", [False, True])
@torch.inference_mode()
def test_gdn_ba_overlap_graph_after_input_change(nz_qkvz, tp_size):
    if not get_current_hardware_profile().supports(HardwareCapability.GDN_BA_VECTOR_OVERLAP):
        pytest.skip("GDN BA Vector overlap is validated on Ascend950")

    torch.manual_seed(41)
    x = torch.randn((1, 5120), device="npu", dtype=torch.bfloat16)
    qkvz_weight = torch.randn((16384 // tp_size, 5120), device="npu", dtype=torch.bfloat16) * 0.02
    ba_weight = torch.randn((96 // tp_size, 5120), device="npu", dtype=torch.bfloat16) * 0.02
    if nz_qkvz:
        qkvz_weight = torch_npu.npu_format_cast(qkvz_weight.t().contiguous(), 29).t()
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
    )
    context = SimpleNamespace(attn_metadata={"gdn": metadata})

    def project(activation, qkvz_weight, ba_weight):
        return torch.ops.vllm.ascend_gdn_qkvz_ba_projection(activation + 0.25, qkvz_weight, ba_weight, "gdn")

    compiled = torch.compile(project, backend="eager", fullgraph=True)
    with override_forward_context(SimpleNamespace(attn_metadata=None)):
        # The compiled op must use decode metadata at execution, rather than
        # specializing the op body to the context used to trace it.
        with patch("vllm_ascend.ops.gdn_ba_overlap.attention_calculation_stream") as side_stream:
            compiled(x, qkvz_weight, ba_weight)
            side_stream.assert_not_called()

    attention_calculation_stream()
    with override_forward_context(context):
        with patch(
            "vllm_ascend.ops.gdn_ba_overlap.attention_calculation_stream",
            wraps=attention_calculation_stream,
        ) as side_stream:
            compiled(x, qkvz_weight, ba_weight)
            side_stream.assert_called_once()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            qkvz, ba = compiled(x, qkvz_weight, ba_weight)
            actual = split_gdn_projection(qkvz, ba, 10240 // tp_size, 6144 // tp_size, 128)

    for scale in (0.25, 0.7, 1.5):
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        expected_qkvz = torch.nn.functional.linear(x + 0.25, qkvz_weight)
        expected_ba = torch.nn.functional.linear(x + 0.25, ba_weight)
        expected = split_gdn_projection(expected_qkvz, expected_ba, 10240 // tp_size, 6144 // tp_size, 128)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
