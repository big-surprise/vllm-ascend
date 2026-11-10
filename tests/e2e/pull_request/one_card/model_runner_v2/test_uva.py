#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

MODELS = ["Qwen/Qwen3-0.6B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("enforce_eager", [True])
@patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1", "PYTORCH_NPU_ALLOC_CONF": "pinned_mem_register:True"})
def test_qwen3_dense_eager_mode(
    model: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.5,
        logprobs=2,
        prompt_logprobs=2,
        logit_bias={0: -1.0, 1: 0.5},
        min_p=0.01,
        bad_words=["the", " the"],
    )
    with VllmRunner(
        model,
        max_model_len=1024,
        enforce_eager=enforce_eager,
        async_scheduling=True,
    ) as runner:
        runner.model.generate(prompts, sampling_params)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_sampling_uva_reuses_unchanged_fallback(monkeypatch, dtype):
    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import ReadOnlySamplingUvaTensor

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    tensor = ReadOnlySamplingUvaTensor(32, dtype, max_concurrency=2)
    tensor.np.fill(3)
    first = tensor.copy_to_uva()
    torch.npu.synchronize()
    with patch.object(tensor.pool, "copy_to_uva", wraps=tensor.pool.copy_to_uva) as copies:
        for _ in range(4):
            assert tensor.copy_to_uva() is first
        assert copies.call_count == 0
        tensor.np[2:5].fill(7)
        result = tensor.copy_to_uva()
        torch.npu.synchronize()
        assert copies.call_count == 1
        torch.testing.assert_close(result.cpu(), torch.from_numpy(tensor.np.copy()))
        assert tensor.copy_to_uva(3).shape == (3,)
        assert tensor.copy_to_uva().shape == (32,)
        torch.npu.synchronize()


def test_sampling_uva_request_slot_reuse(monkeypatch):
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import ReadOnlySamplingUvaTensor

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    state = SamplingStates(32, 128)
    assert isinstance(state.temperature, ReadOnlySamplingUvaTensor)
    state.add_request(0, SamplingParams(temperature=0.5, top_k=4, top_p=0.9, seed=11))
    state.apply_staged_writes()
    torch.npu.synchronize()
    old_temperature = state.temperature.gpu.clone()
    state.apply_staged_writes()
    state.add_request(0, SamplingParams(temperature=0.8, top_k=8, top_p=0.7, seed=19))
    state.apply_staged_writes()
    torch.npu.synchronize()
    for tensor in (state.temperature, state.top_k, state.top_p, state.min_p, state.seeds):
        torch.testing.assert_close(tensor.gpu.cpu(), tensor.cpu)
    assert old_temperature[0].item() == 0.5


def test_sampling_uva_preserves_two_inflight_reads(monkeypatch):
    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import ReadOnlySamplingUvaTensor

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    tensor = ReadOnlySamplingUvaTensor(32, torch.int32, max_concurrency=2)
    pending = []
    for step in range(24):
        if len(pending) == 2:
            event, actual, expected = pending.pop(0)
            event.synchronize()
            torch.testing.assert_close(actual.cpu(), expected)
        if step % 3 != 1:
            tensor.np.fill(step)
        actual = tensor.copy_to_uva().clone()
        event = torch.npu.Event()
        event.record()
        pending.append((event, actual, tensor.cpu.clone()))
    for event, actual, expected in pending:
        event.synchronize()
        torch.testing.assert_close(actual.cpu(), expected)


def test_sampling_uva_capture_and_stream_change(monkeypatch):
    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import ReadOnlySamplingUvaTensor

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    tensor = ReadOnlySamplingUvaTensor(32, torch.float32)
    tensor.np.fill(2)
    tensor.copy_to_uva()
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with patch.object(tensor.pool, "copy_to_uva", wraps=tensor.pool.copy_to_uva) as copies:
        with torch.npu.graph(graph, stream=stream):
            captured = tensor.copy_to_uva().clone()
        assert copies.call_count == 1
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured.cpu(), tensor.cpu)
        tensor.copy_to_uva()
        assert copies.call_count == 2
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            tensor.copy_to_uva()
        stream.synchronize()
        assert copies.call_count == 3


def test_sampling_uva_sampler_shares_queries(monkeypatch):
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.sample.sampler import Sampler

    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import ReadOnlySamplingUvaTensor

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    req = SimpleNamespace(max_num_reqs=32, vocab_size=128, device=torch.device("npu:0"))
    sampler = Sampler(32, 128, req.device, req)
    tensor = sampler.sampling_states.temperature
    assert isinstance(tensor, ReadOnlySamplingUvaTensor)
    assert tensor._upload_context is sampler._ascend_sampling_upload_context
    sampler.apply_staged_writes()
    torch.npu.synchronize()
    with (
        patch.object(torch.npu, "current_stream", wraps=torch.npu.current_stream) as stream,
        patch.object(torch.npu, "is_current_stream_capturing", wraps=torch.npu.is_current_stream_capturing) as capture,
        patch.object(tensor.pool, "copy_to_uva", wraps=tensor.pool.copy_to_uva) as copies,
    ):
        sampler.apply_staged_writes()
        assert stream.call_count == capture.call_count == 1
        copies.assert_not_called()
    assert tensor._upload_context.get() is None
    tensor.np[0] = 0.75
    sampler.apply_staged_writes()
    torch.npu.synchronize()
    torch.testing.assert_close(tensor.gpu.cpu(), tensor.cpu)

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=capture_stream):
        sampler.apply_staged_writes()
        captured = tensor.gpu.clone()
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(captured.cpu(), tensor.cpu)
    assert tensor._upload_context.get() is None
    tensor.np[0] = 0.25
    sampler.apply_staged_writes()
    torch.npu.synchronize()
    torch.testing.assert_close(tensor.gpu.cpu(), tensor.cpu)
