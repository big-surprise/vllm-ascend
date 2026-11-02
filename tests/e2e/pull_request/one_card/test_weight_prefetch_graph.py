# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu
from vllm_ascend.ops import mlp_weight_prefetch as prefetch


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
