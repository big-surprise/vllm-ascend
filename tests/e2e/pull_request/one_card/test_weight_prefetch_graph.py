# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.forward_context import ForwardContext, override_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_weight_prefetch_context
from vllm_ascend.ops import mlp_weight_prefetch as prefetch
from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext


@pytest.mark.parametrize("reduce_first", [False, True])
@torch.inference_mode()
def test_prefetch_survives_prefill_compile_and_decode_capture(monkeypatch, reduce_first):
    """Real NPU prefetch/ACL graph regression; collective replaced by NPU add."""
    monkeypatch.setattr(prefetch, "tensor_model_parallel_all_reduce", lambda x: x + 1)
    issue = torch_npu.npu_prefetch
    issued = []

    def record_issue(weight, dependency, size, offset):
        issued.append((size, dependency is None))
        return issue(weight, dependency, size, offset)

    monkeypatch.setattr(torch_npu, "npu_prefetch", record_issue)
    weight = torch.randn(256, 64, device="npu", dtype=torch.bfloat16)

    def forward(x, w):
        reduced = torch.ops.vllm.all_reduce_with_weight_prefetch(x, w, 1.0, reduce_first)
        return torch.nn.functional.linear(reduced, w)

    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=True)
    large = torch.randn(1024, 64, device="npu", dtype=torch.bfloat16)
    torch.testing.assert_close(compiled(large, weight), torch.nn.functional.linear(large + 1, weight), rtol=0, atol=0)
    assert not issued
    x = torch.randn(1, 64, device="npu", dtype=torch.bfloat16)
    warmup = torch.npu.Stream()
    warmup.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(warmup):
        for _ in range(3):
            compiled(x, weight)
    torch.npu.current_stream().wait_stream(warmup)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    issued.clear()
    with torch.npu.graph(graph):
        actual = compiled(x, weight)
    assert issued == [(weight.numel() * weight.element_size(), reduce_first)]
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, torch.nn.functional.linear(x + 1, weight), rtol=0, atol=0)
    # Replay runs the captured device work without re-entering the Python op.
    assert len(issued) == 1
    torch._dynamo.reset()


@pytest.mark.parametrize("enabled", [False, True])
@torch.inference_mode()
def test_v2_model_context_prefetch_compile_capture_replay(monkeypatch, enabled):
    """Resolve real model weights through V2 context before compiled decode."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(prefetch, "tensor_model_parallel_all_reduce", lambda x: x + 1)
    method = SimpleNamespace(
        is_qwen_dense=True,
        use_qwen_dense_mlp_prefetch=enabled,
        mlp=SimpleNamespace(prefetch_ratio={"gate_up": 1.0}),
    )
    monkeypatch.setattr(prefetch, "get_weight_prefetch_method", lambda: method)
    issued = []
    issue = torch_npu.npu_prefetch

    def record_issue(weight, dependency, size, offset):
        issued.append(size)
        return issue(weight, dependency, size, offset)

    monkeypatch.setattr(torch_npu, "npu_prefetch", record_issue)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.start_layer = 0
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_up_proj = torch.nn.Linear(64, 256, bias=False, device="npu", dtype=torch.bfloat16)
            layer.self_attn = torch.nn.Module()
            projection = AscendRowParallelLinear.__new__(AscendRowParallelLinear)
            torch.nn.Module.__init__(projection)
            projection.prefix = "model.layers.0.self_attn.o_proj"
            projection.custom_op = None
            projection.tp_size = 4
            projection.reduce_results = True
            projection.input_is_parallel = True
            layer.self_attn.o_proj = projection
            self.model.layers = torch.nn.ModuleList([layer])
            self.decoder = torch.compile(self.decode, backend="eager", fullgraph=True, dynamic=True)
            if enabled:
                self.register_forward_pre_hook(set_weight_prefetch_context)

        def decode(self, x):
            target = prefetch._attention_gate_up_target("model.layers.0.self_attn.o_proj")
            if target is None:
                reduced = x + 1
            else:
                reduced = torch.ops.vllm.all_reduce_with_weight_prefetch(x, target[0], target[1], False)
            return self.model.layers[0].mlp.gate_up_proj(reduced)

        def forward(self, x):
            return self.decoder(x)

    model = Model()
    wrapped = ModelWithContext(model)
    weight = model.model.layers[0].mlp.gate_up_proj.weight
    context = ForwardContext(
        no_compile_layers={}, attn_metadata=None, slot_mapping=None, additional_kwargs={"num_tokens": 1024}
    )
    with override_forward_context(context):
        large = torch.randn(1024, 64, device="npu", dtype=torch.bfloat16)
        torch.testing.assert_close(model(large), torch.nn.functional.linear(large + 1, weight), rtol=0, atol=0)
        assert not issued
        context.additional_kwargs["num_tokens"] = 1
        x = torch.randn(1, 64, device="npu", dtype=torch.bfloat16)
        warmup = torch.npu.Stream()
        warmup.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(warmup):
            for _ in range(3):
                wrapped(x)
        torch.npu.current_stream().wait_stream(warmup)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        issued.clear()
        with torch.npu.graph(graph):
            actual = wrapped(x)
        expected_issues = [weight.numel() * weight.element_size()] if enabled else []
        assert issued == expected_issues
        assert _EXTRA_CTX.model_instance is (model if enabled else None)
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual, torch.nn.functional.linear(x + 1, weight), rtol=0, atol=0)
        assert issued == expected_issues
    torch._dynamo.reset()
