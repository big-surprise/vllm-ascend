# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from contextvars import ContextVar
from functools import wraps

import torch
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.sample import bad_words, logit_bias, logprob, penalties, states
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.patch.worker.patch_v2.patch_uva import UvaBufferWrapper


class ReadOnlySamplingUvaTensor(UvaBackedTensor):
    """Reuse unchanged CPU-owned sampling parameters on the current NPU stream.

    Device consumers must only read these tensors. Changed values continue to
    use the upstream pool, including its in-flight slot retirement contract.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        super().__init__(size, dtype, max_concurrency)
        self._reuse_unchanged = all(
            isinstance(buf, UvaBufferWrapper) and not buf._use_real_uva for buf in self.pool._uva_bufs
        )
        self._sampling_snapshot: tuple[tuple[int, ...], str, bytes] | None = None
        self._sampling_stream = None
        self._sampling_tensor = None
        self._upload_context: ContextVar | None = None

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        if not self._reuse_unchanged:
            return super().copy_to_uva(n)

        context = self._upload_context.get() if self._upload_context is not None else None
        if context is None:
            capturing = torch.npu.is_current_stream_capturing()
            stream = None if capturing else torch.npu.current_stream()
        else:
            capturing, stream = context

        # Capture must retain its copy nodes. A captured copy is not evidence
        # that the destination is already current outside the captured graph.
        if capturing:
            self._sampling_snapshot = None
            self._sampling_stream = None
            self._sampling_tensor = None
            return super().copy_to_uva(n)

        values = self.np if n is None else self.np[:n]
        # Byte comparison also detects writes through NumPy views/fill and
        # distinguishes signed zero and NaN payloads without device reads.
        snapshot = (values.shape, values.dtype.str, values.tobytes())
        if (
            snapshot != self._sampling_snapshot
            or stream != self._sampling_stream
            or self.gpu is not self._sampling_tensor
        ):
            tensor = super().copy_to_uva(n)
            self._sampling_snapshot = snapshot
            self._sampling_stream = stream
            self._sampling_tensor = tensor
        return self.gpu


def _bind_sampling_upload_context(sampler):
    tensors = []
    for state in (
        sampler.sampling_states,
        sampler.penalties_state,
        sampler.logit_bias_state,
        sampler.bad_words_state,
        sampler.logprob_token_ids_state,
    ):
        for tensor in vars(state).values():
            if isinstance(tensor, ReadOnlySamplingUvaTensor) and tensor._reuse_unchanged:
                tensors.append(tensor)
    # Per-instance ContextVar avoids sharing state between samplers or threads.
    # Real UVA needs no explicit-copy cache and no new stream queries.
    context = ContextVar("ascend_sampling_upload", default=None) if tensors else None
    sampler._ascend_sampling_upload_context = context
    for tensor in tensors:
        tensor._upload_context = context


def _with_sampling_upload_context_init(initialize):
    @wraps(initialize)
    def initialize_with_context(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        _bind_sampling_upload_context(self)

    return initialize_with_context


def _with_sampling_upload_context(apply_staged_writes):
    @wraps(apply_staged_writes)
    def apply_with_context(self):
        context = self._ascend_sampling_upload_context
        if context is None:
            return apply_staged_writes(self)
        # These synchronous state uploads do not switch streams or start graph
        # capture. Query once for the call, not once for each metadata tensor.
        capturing = torch.npu.is_current_stream_capturing()
        stream = None if capturing else torch.npu.current_stream()
        token = context.set((capturing, stream))
        try:
            return apply_staged_writes(self)
        finally:
            context.reset(token)

    return apply_with_context


# These modules use UvaBackedTensor only for CPU-owned, device-read-only
# sampling parameters. Request state, block tables and staged device writes
# keep the upstream buffer implementation.
states.UvaBackedTensor = ReadOnlySamplingUvaTensor
penalties.UvaBackedTensor = ReadOnlySamplingUvaTensor
logit_bias.UvaBackedTensor = ReadOnlySamplingUvaTensor
bad_words.UvaBackedTensor = ReadOnlySamplingUvaTensor
logprob.UvaBackedTensor = ReadOnlySamplingUvaTensor

Sampler.__init__ = _with_sampling_upload_context_init(Sampler.__init__)
Sampler.apply_staged_writes = _with_sampling_upload_context(Sampler.apply_staged_writes)
