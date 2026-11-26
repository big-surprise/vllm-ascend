# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops import ffn_weight_layout as layout
from vllm_ascend.ops import mlp_weight_prefetch  # noqa: F401
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod


@pytest.mark.parametrize("tokens", [1, 4, 128, 512])
@torch.no_grad()
def test_ffn_nz_load_and_graph(tokens):
    if not is_950():
        pytest.skip("[K,N] NZ FFN layout is enabled only on Ascend950")
    torch.manual_seed(42)
    x = torch.randn(tokens, 256, device="npu", dtype=torch.bfloat16)
    original = [torch.randn(256, 256, device="npu", dtype=torch.bfloat16),
                torch.randn(256, 128, device="npu", dtype=torch.bfloat16)]
    layers = []
    method = AscendUnquantizedLinearMethod()
    with patch.object(layout, "get_ascend_config", return_value=SimpleNamespace(ffn_weight_layout="nz_kn")):
        for suffix, weight in zip(("gate_up_proj", "down_proj"), original):
            layer = torch.nn.Module()
            layer.prefix = "model.layers.0.mlp." + suffix
            layer.register_parameter("weight", torch.nn.Parameter(weight.clone(), requires_grad=False))
            parameter = layer.weight
            method.process_weights_after_loading(layer)
            pointer = layer.weight.data_ptr()
            method.process_weights_after_loading(layer)
            assert layer.weight is parameter and layer.weight.data_ptr() == pointer
            assert torch_npu.get_npu_format(layer.weight) == 29
            assert layer.weight.stride() == (1, layer.weight.shape[0])
            restored = torch_npu.npu_format_cast(layer.weight.t(), 2).t()
            assert torch.equal(restored, weight)
            layers.append(layer)

    def forward():
        gate = method.apply(layers[0], x)
        activated = torch.ops.vllm.prefetch_swiglu(gate, layers[1].weight, 1.0)
        return method.apply(layers[1], activated)

    for _ in range(3):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = forward()
    for scale in (0.25, 1.0, 2.0):
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.nn.functional.linear(
            torch_npu.npu_swiglu(torch.nn.functional.linear(x, original[0])), original[1]
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
