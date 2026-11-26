# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops import attention_weight_layout as layout
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod


@pytest.mark.parametrize("suffix", ["linear_attn.in_proj_qkvz", "self_attn.qkv_proj"])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 32, 128, 512, 2048])
@torch.no_grad()
def test_attention_single_weight_prefill_decode_and_graph(suffix, tokens):
    if not is_950():
        pytest.skip("[K,N] NZ attention layout is enabled only on Ascend950")
    torch.manual_seed(42)
    x = torch.randn(tokens, 256, device="npu", dtype=torch.bfloat16)
    original = torch.randn(128, 256, device="npu", dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0." + suffix
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), requires_grad=False))
    parameter = layer.weight
    config = SimpleNamespace(linear_attention_qkvz_layout="nz_kn", full_attention_qkv_layout="nz_kn")
    method = AscendUnquantizedLinearMethod()
    with patch.object(layout, "get_ascend_config", return_value=config):
        method.process_weights_after_loading(layer)
        pointer = layer.weight.data_ptr()
        method.process_weights_after_loading(layer)
    assert layer.weight is parameter and layer.weight.data_ptr() == pointer
    assert not method.supports_weight_switch
    assert not list(layer.named_buffers())
    assert list(layer.state_dict()) == ["weight"]
    assert torch_npu.get_npu_format(layer.weight) == 29
    assert layer.weight.stride() == (1, layer.weight.shape[0])
    assert layer.weight.untyped_storage().nbytes() == original.numel() * original.element_size()
    restored = torch_npu.npu_format_cast(layer.weight.t(), 2).t()
    assert torch.equal(restored, original)
    for _ in range(3):
        method.apply(layer, x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = method.apply(layer, x)
    for scale in (0.25, 0.7, 1.5):
        x.copy_(torch.randn_like(x) * scale)
        graph.replay()
        torch.npu.synchronize()
        expected = torch.nn.functional.linear(x, original)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 4, 8, 128, 512, 2048])
@pytest.mark.parametrize("suffix,output_size", [("linear_attn.in_proj_qkvz", 4096), ("self_attn.qkv_proj", 3584)])
@torch.no_grad()
def test_attention_tp4_shapes_bf16_accuracy(suffix, output_size, tokens):
    if not is_950():
        pytest.skip("[K,N] NZ attention layout is enabled only on Ascend950")
    torch.manual_seed(42)
    original = torch.randn(output_size, 5120, device="npu", dtype=torch.bfloat16) * 0.02
    x = torch.randn(tokens, 5120, device="npu", dtype=torch.bfloat16)
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0." + suffix
    layer.register_parameter("weight", torch.nn.Parameter(original.clone(), requires_grad=False))
    config = SimpleNamespace(linear_attention_qkvz_layout="nz_kn", full_attention_qkv_layout="nz_kn")
    method = AscendUnquantizedLinearMethod()
    with patch.object(layout, "get_ascend_config", return_value=config):
        method.process_weights_after_loading(layer)
    actual = method.apply(layer, x)
    expected = torch.nn.functional.linear(x, original)
    assert torch.isfinite(actual).all()
    # Different CANN tilings can round a few BF16 values differently for
    # large M. Check numerical accuracy without asserting bitwise equality.
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative_l2.item() < 1e-4
