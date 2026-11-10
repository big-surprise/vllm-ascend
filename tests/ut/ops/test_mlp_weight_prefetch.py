# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import mlp_weight_prefetch as prefetch
from vllm_ascend.ops.linear import AscendRowParallelLinear


@pytest.mark.parametrize("effective", [False, True])
def test_prefetch_gemma_accepts_raw_or_effective_scale(monkeypatch, effective):
    from unittest.mock import Mock

    monkeypatch.setattr(prefetch, "_start_prefetch", lambda *args: None)
    monkeypatch.setattr(prefetch, "enable_custom_op", lambda: False)
    kernel = Mock(side_effect=lambda x, r, w, eps: (x, None, r))
    monkeypatch.setattr(prefetch.torch_npu, "npu_add_rms_norm", kernel)
    x = torch.randn(1, 8)
    raw = torch.randn(8)
    scale = 1.0 + raw if effective else raw
    prefetch.prefetch_gemma_rms_norm(x, x, scale, torch.empty(8, 8), 1e-6, 0.0, "", effective)
    torch.testing.assert_close(kernel.call_args.args[2], 1.0 + raw, rtol=0, atol=0)


@pytest.fixture
def prefetch_runtime(monkeypatch):
    """Exercise the real custom-op body with CPU tensors and recorded fences."""
    events = []
    requests = []
    compute = SimpleNamespace(wait_stream=lambda stream: events.append("join"))
    stream = SimpleNamespace(wait_stream=lambda current: events.append("fork"))

    def reduce(x):
        events.append("reduce")
        return x + 2

    def issue(weight, dependency, size, offset):
        events.append("prefetch")
        requests.append((weight, dependency, size, offset))

    monkeypatch.setattr(prefetch, "tensor_model_parallel_all_reduce", reduce)
    monkeypatch.setattr(prefetch.torch_npu.npu, "current_stream", lambda: compute)
    monkeypatch.setattr(prefetch, "prefetch_stream", lambda: stream)
    monkeypatch.setattr(prefetch, "npu_stream_switch", lambda unused: nullcontext())
    monkeypatch.setattr(prefetch.torch_npu, "npu_prefetch", issue, raising=False)
    lib = torch.library.Library("vllm", "IMPL", "CPU")
    try:
        lib.impl("all_reduce_with_weight_prefetch", prefetch.all_reduce_with_weight_prefetch)
        yield events, requests
    finally:
        lib._destroy()


@pytest.mark.parametrize("reduce_first", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 4, 499, 500, 512, 16384])
def test_runtime_token_gate_and_fence_order(prefetch_runtime, tokens, reduce_first):
    events, requests = prefetch_runtime
    x = torch.randn(tokens, 8)
    weight = torch.empty(16, 8)
    actual = torch.ops.vllm.all_reduce_with_weight_prefetch(x, weight, 1.0, reduce_first)
    torch.testing.assert_close(actual, x + 2, rtol=0, atol=0)
    if 0 < tokens < 500:
        middle = ["reduce", "prefetch"] if reduce_first else ["prefetch", "reduce"]
        assert events == ["fork", *middle, "join"]
        assert len(requests) == 1
        assert requests[0][0] is weight
        assert requests[0][1] is (None if reduce_first else x)
    else:
        assert events == ["reduce"]
        assert not requests


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_ratio_and_byte_cap(prefetch_runtime, monkeypatch, ratio):
    events, requests = prefetch_runtime
    monkeypatch.setattr(prefetch, "MAX_PREFETCH_WEIGHT_SIZE", 64)
    x, weight = torch.randn(1, 8), torch.empty(4, 8)
    torch.ops.vllm.all_reduce_with_weight_prefetch(x, weight, ratio, False)
    if ratio == 0:
        assert events == ["reduce"]
    else:
        assert requests[0][2] == min(int(weight.numel() * weight.element_size() * ratio), 64)


def row_projection(prefix, **kwargs):
    layer = AscendRowParallelLinear.__new__(AscendRowParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.custom_op = None
    layer.reduce_results = True
    layer.input_is_parallel = True
    layer.tp_size = 4
    for name, value in kwargs.items():
        setattr(layer, name, value)
    return layer


@pytest.fixture
def decoder_layers(monkeypatch):
    gate = SimpleNamespace(weight=torch.empty(16, 8))
    out = row_projection("language_model.model.layers.0.linear_attn.out_proj")
    down = row_projection("language_model.model.layers.0.mlp.down_proj")
    layers = [
        SimpleNamespace(
            linear_attn=SimpleNamespace(out_proj=out), mlp=SimpleNamespace(gate_up_proj=gate, down_proj=down)
        ),
        SimpleNamespace(linear_attn=SimpleNamespace(in_proj_qkv=SimpleNamespace(weight=torch.empty(4, 8)))),
    ]
    method = SimpleNamespace(
        is_qwen_dense=True,
        use_qwen_dense_mlp_prefetch=True,
        mlp=SimpleNamespace(prefetch_ratio={"gate_up": 1.0}),
        attn=SimpleNamespace(enable=True, prefetch_ratio={"qkv": 1.0}),
    )
    monkeypatch.setattr(prefetch, "get_weight_prefetch_method", lambda: method)
    monkeypatch.setattr(prefetch, "_EXTRA_CTX", SimpleNamespace(model_instance=object()))
    monkeypatch.setattr(prefetch, "layers_of_model_instance", lambda model: layers)
    return layers


def test_gate_target_requires_actual_row_issue_path(decoder_layers):
    layer = decoder_layers[0]
    projection = layer.linear_attn.out_proj
    assert prefetch._attention_gate_up_target(projection.prefix)[0] is layer.mlp.gate_up_proj.weight
    # A different implementation with the same attributes must retain fallback.
    layer.linear_attn.out_proj = SimpleNamespace(
        prefix=projection.prefix, custom_op=None, reduce_results=True, tp_size=4, input_is_parallel=True
    )
    assert prefetch._attention_gate_up_target(projection.prefix) is None


@pytest.mark.parametrize("name,value", [("tp_size", 1), ("reduce_results", False), ("input_is_parallel", False)])
def test_unsupported_row_path_keeps_norm_fallback(decoder_layers, name, value):
    projection = decoder_layers[0].linear_attn.out_proj
    setattr(projection, name, value)
    assert prefetch._attention_gate_up_target(projection.prefix) is None


@pytest.mark.parametrize("projection_name", ["in_proj_qkv", "in_proj_qkvz", "qkv_proj"])
def test_next_qkv_supports_split_fused_and_full_attention(decoder_layers, projection_name):
    weight = torch.empty(4, 8)
    attn = SimpleNamespace(**{projection_name: SimpleNamespace(weight=weight)})
    decoder_layers[1] = SimpleNamespace(**{("self_attn" if projection_name == "qkv_proj" else "linear_attn"): attn})
    prefix = decoder_layers[0].mlp.down_proj.prefix
    assert prefetch._next_layer_input_proj_target(prefix)[0] is weight
    assert prefetch._next_layer_input_proj_target(prefix.replace("layers.0", "layers.1")) is None


@pytest.mark.parametrize("supports_issue", [False, True])
def test_norm_fallback_does_not_duplicate_row_prefetch(prefetch_runtime, decoder_layers, monkeypatch, supports_issue):
    _, requests = prefetch_runtime
    projection = decoder_layers[0].linear_attn.out_proj
    projection.input_is_parallel = supports_issue
    monkeypatch.setattr(prefetch, "enable_custom_op", lambda: False)
    monkeypatch.setattr(
        prefetch.torch_npu,
        "npu_add_rms_norm",
        lambda x, residual, weight, eps: (x + residual, None, residual),
        raising=False,
    )
    x = torch.randn(1, 8)
    prefetch.prefetch_gemma_rms_norm(x, x, torch.ones(8), torch.empty(16, 8), 1e-6, 1.0, projection.prefix)
    assert len(requests) == (0 if supports_issue else 1)


@pytest.mark.parametrize("return_bias", [False, True])
def test_row_forward_retains_prefetch_after_large_shape_trace(prefetch_runtime, monkeypatch, return_bias):
    events, requests = prefetch_runtime
    layer = row_projection("model.layers.0.linear_attn.out_proj")
    layer.tp_rank = 0
    layer.skip_bias_add = False
    layer.return_bias = return_bias
    layer.bias = None
    layer.quant_method = SimpleNamespace(apply=lambda module, x, bias: x * 3)
    weight = torch.empty(16, 8)
    monkeypatch.setattr(prefetch, "_all_reduce_prefetch_target", lambda prefix: (weight, 1.0))
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(layer, backend=backend, fullgraph=True, dynamic=True)
    for tokens in [1024, 4, 1]:
        events.clear()
        requests.clear()
        x = torch.randn(tokens, 8)
        actual = compiled(x)
        if return_bias:
            actual, bias = actual
            assert bias is None
        torch.testing.assert_close(actual, x * 3 + 2, rtol=0, atol=0)
        assert len(requests) == (0 if tokens >= 500 else 1)
    assert graphs
    for gm in graphs:
        assert sum("vllm.all_reduce_with_weight_prefetch" in str(node.target) for node in gm.graph.nodes) == 1
    torch._dynamo.reset()


def test_dynamic_export_preserves_decode_prefetch(prefetch_runtime):
    events, requests = prefetch_runtime

    class Reduction(torch.nn.Module):
        def forward(self, x, weight):
            return torch.ops.vllm.all_reduce_with_weight_prefetch(x, weight, 1.0, False) * 3

    weight = torch.empty(16, 8)
    exported = torch.export.export(
        Reduction(),
        (torch.randn(1024, 8), weight),
        dynamic_shapes=({0: torch.export.Dim("tokens", min=1, max=16384)}, None),
    )
    assert sum("vllm.all_reduce_with_weight_prefetch" in str(n.target) for n in exported.graph.nodes) == 1
    for tokens in [1024, 4, 1]:
        events.clear()
        requests.clear()
        x = torch.randn(tokens, 8)
        torch.testing.assert_close(exported.module()(x, weight), (x + 2) * 3, rtol=0, atol=0)
        assert len(requests) == (0 if tokens >= 500 else 1)
