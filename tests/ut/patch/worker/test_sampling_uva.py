# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.sample import bad_words, logit_bias, logprob, penalties, states

from vllm_ascend.patch.worker.patch_v2.patch_sampling_uva import (
    ReadOnlySamplingUvaTensor,
    _bind_sampling_upload_context,
    _with_sampling_upload_context,
)
from vllm_ascend.patch.worker.patch_v2.patch_uva import UvaBufferWrapper


class FakeFallbackBuffer(UvaBufferWrapper):
    def __init__(self):
        self._use_real_uva = False


class CpuPool:
    def __init__(self, size, dtype, max_concurrency=None):
        self._uva_bufs = [FakeFallbackBuffer(), FakeFallbackBuffer()]
        self.copies = 0

    def copy_to_uva(self, values):
        self.copies += 1
        return torch.from_numpy(np.asarray(values).copy())


@pytest.fixture
def cpu_buffer(monkeypatch):
    monkeypatch.setattr(buffer_utils, "UvaBufferPool", CpuPool)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    stream = object()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: stream)
    return ReadOnlySamplingUvaTensor(8, torch.float32)


def test_unchanged_values_preserve_tensor_and_skip_copy(cpu_buffer):
    cpu_buffer.np.fill(1)
    first = cpu_buffer.copy_to_uva()
    copies = cpu_buffer.pool.copies
    for _ in range(3):
        assert cpu_buffer.copy_to_uva() is first
    assert cpu_buffer.pool.copies == copies


def test_view_and_fill_mutations_are_uploaded(cpu_buffer):
    cpu_buffer.copy_to_uva()
    cpu_buffer.np[2:5].fill(7)
    torch.testing.assert_close(cpu_buffer.copy_to_uva(), torch.from_numpy(cpu_buffer.np.copy()))
    cpu_buffer.cpu.fill_(3)
    torch.testing.assert_close(cpu_buffer.copy_to_uva(), torch.full((8,), 3.0))


def test_length_changes_do_not_reuse_wrong_shape(cpu_buffer):
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.copy_to_uva(2).shape == (2,)
    assert cpu_buffer.copy_to_uva(0).shape == (0,)
    assert cpu_buffer.copy_to_uva().shape == (8,)


def test_signed_zero_and_nan_payload_are_compared_exactly(cpu_buffer):
    cpu_buffer.copy_to_uva()
    copies = cpu_buffer.pool.copies
    cpu_buffer.np[0] = -0.0
    assert torch.signbit(cpu_buffer.copy_to_uva()[0])
    assert cpu_buffer.pool.copies == copies + 1
    bits = cpu_buffer.np.view(np.uint32)
    bits[1] = 0x7FC00001
    cpu_buffer.copy_to_uva()
    copies = cpu_buffer.pool.copies
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.pool.copies == copies
    bits[1] = 0x7FC00002
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.pool.copies == copies + 1


def test_stream_change_enqueues_a_copy(monkeypatch, cpu_buffer):
    first = cpu_buffer.copy_to_uva()
    monkeypatch.setattr(torch.npu, "current_stream", lambda: "other-stream")
    assert cpu_buffer.copy_to_uva() is not first


def test_capture_keeps_copy_and_invalidates_cache(monkeypatch, cpu_buffer):
    cpu_buffer.copy_to_uva()
    copies = cpu_buffer.pool.copies
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    cpu_buffer.copy_to_uva()
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.pool.copies == copies + 2
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.pool.copies == copies + 3


def test_replaced_device_view_is_not_treated_as_current(cpu_buffer):
    cpu_buffer.np.fill(4)
    cpu_buffer.copy_to_uva()
    cpu_buffer.gpu = torch.zeros(8)
    torch.testing.assert_close(cpu_buffer.copy_to_uva(), torch.full((8,), 4.0))


def test_real_uva_preserves_original_behavior(monkeypatch):
    class RealUvaPool(CpuPool):
        def __init__(self, *args):
            super().__init__(*args)
            for buf in self._uva_bufs:
                buf._use_real_uva = True

    monkeypatch.setattr(buffer_utils, "UvaBufferPool", RealUvaPool)
    tensor = ReadOnlySamplingUvaTensor(8, torch.int32)
    copies = tensor.pool.copies
    tensor.copy_to_uva()
    tensor.copy_to_uva()
    assert tensor.pool.copies == copies + 2


def test_patch_is_scoped_to_sampling_modules():
    for module in (states, penalties, logit_bias, bad_words, logprob):
        assert module.UvaBackedTensor is ReadOnlySamplingUvaTensor
    assert buffer_utils.UvaBackedTensor is not ReadOnlySamplingUvaTensor


def make_sampler(tensor):
    sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(tensor=tensor),
        penalties_state=SimpleNamespace(),
        logit_bias_state=SimpleNamespace(),
        bad_words_state=SimpleNamespace(),
        logprob_token_ids_state=SimpleNamespace(),
    )
    _bind_sampling_upload_context(sampler)
    return sampler


def test_sampler_queries_context_once_per_call(monkeypatch, cpu_buffer):
    sampler = make_sampler(cpu_buffer)
    stream = Mock(return_value="stream-a")
    capture = Mock(return_value=False)
    monkeypatch.setattr(torch.npu, "current_stream", stream)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", capture)

    @_with_sampling_upload_context
    def upload(self):
        for _ in range(15):
            cpu_buffer.copy_to_uva()

    upload(sampler)
    assert stream.call_count == capture.call_count == 1
    copies = cpu_buffer.pool.copies
    upload(sampler)
    assert stream.call_count == capture.call_count == 2
    assert cpu_buffer.pool.copies == copies
    stream.return_value = "stream-b"
    upload(sampler)
    assert cpu_buffer.pool.copies == copies + 1
    assert cpu_buffer._upload_context.get() is None
    cpu_buffer.copy_to_uva()
    assert stream.call_count == capture.call_count == 4


def test_sampler_context_is_cleared_on_exception(monkeypatch, cpu_buffer):
    sampler = make_sampler(cpu_buffer)

    @_with_sampling_upload_context
    def upload(self):
        cpu_buffer.copy_to_uva()
        raise ValueError("upload failed")

    with pytest.raises(ValueError, match="upload failed"):
        upload(sampler)
    assert cpu_buffer._upload_context.get() is None
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    copies = cpu_buffer.pool.copies
    cpu_buffer.copy_to_uva()
    assert cpu_buffer.pool.copies == copies + 1


def test_nested_sampler_calls_restore_outer_context(monkeypatch, cpu_buffer):
    sampler = make_sampler(cpu_buffer)
    stream = Mock(side_effect=["outer", "inner"])
    monkeypatch.setattr(torch.npu, "current_stream", stream)

    @_with_sampling_upload_context
    def inner(self):
        assert cpu_buffer._upload_context.get() == (False, "inner")

    @_with_sampling_upload_context
    def outer(self):
        assert cpu_buffer._upload_context.get() == (False, "outer")
        inner(self)
        assert cpu_buffer._upload_context.get() == (False, "outer")

    outer(sampler)
    assert cpu_buffer._upload_context.get() is None


def test_sampler_context_is_thread_local(cpu_buffer):
    sampler = make_sampler(cpu_buffer)
    context = sampler._ascend_sampling_upload_context
    barrier = Barrier(2)

    def observe(value):
        token = context.set((False, value))
        try:
            barrier.wait(timeout=5)
            assert cpu_buffer._upload_context.get() == (False, value)
        finally:
            context.reset(token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(observe, value) for value in ("first", "second")]
        for task in tasks:
            task.result()
    assert context.get() is None


def test_capture_context_keeps_every_copy(monkeypatch, cpu_buffer):
    sampler = make_sampler(cpu_buffer)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    stream = Mock()
    monkeypatch.setattr(torch.npu, "current_stream", stream)

    @_with_sampling_upload_context
    def upload(self):
        cpu_buffer.copy_to_uva()
        cpu_buffer.copy_to_uva()

    copies = cpu_buffer.pool.copies
    upload(sampler)
    assert cpu_buffer.pool.copies == copies + 2
    stream.assert_not_called()
    assert cpu_buffer._sampling_snapshot is None
    assert cpu_buffer._upload_context.get() is None


def test_sampler_without_fallback_skips_context_queries(monkeypatch, cpu_buffer):
    cpu_buffer._reuse_unchanged = False
    sampler = make_sampler(cpu_buffer)
    assert sampler._ascend_sampling_upload_context is None
    stream = Mock()
    capture = Mock()
    monkeypatch.setattr(torch.npu, "current_stream", stream)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", capture)

    @_with_sampling_upload_context
    def upload(self):
        return cpu_buffer.copy_to_uva()

    torch.testing.assert_close(upload(sampler), cpu_buffer.cpu)
    stream.assert_not_called()
    capture.assert_not_called()
