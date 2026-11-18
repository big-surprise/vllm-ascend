#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Out-of-place HCCL all-reduce fast path, built on an existing ``ProcessGroupHCCL``.

Motivation
----------
``DeviceCommunicatorBase.all_reduce`` runs ``torch.distributed.all_reduce`` in
place and returns the input tensor. That is invisible in eager mode, but wrong
wherever the shape of the call matters:

* **Graph capture.** The producer's output buffer is a static input of the
  captured region. Reducing in place rewrites it on every replay, so a second
  replay starts from the already-reduced value instead of the freshly computed
  one.
* **torch.compile.** vLLM wraps this collective in ``torch.ops.vllm.all_reduce``,
  whose docstring states the op is always out of place and whose fake
  implementation returns ``empty_like(tensor)``. An in-place implementation
  makes the declared contract disagree with the behaviour, which compilation
  may exploit.
* **Buffer reuse.** An in-place result aliases the producer, so callers that
  still need the pre-reduction values (residual, norm input) must clone.

What this module does
---------------------
Keeps the process group as owner of both the communicator and its dedicated
collective stream, borrows both, and calls ``HcclAllReduce`` with distinct
send/receive buffers. The receive buffer is a fresh allocation, so the op never
touches its argument.

Why borrow instead of building a communicator
---------------------------------------------
``pyhccl.PyHcclCommunicator`` also reduces out of place, but it creates a *new*
HCCL communicator via ``HcclCommInitRootInfo`` for weight transfer. Borrowing
the group's handle instead means no extra device memory, no extra rank-init
handshake, and collective ordering stays inside the same HCCL context as every
other process-group collective.

Cost model
----------
Intended for latency-bound messages (e.g. a 36-layer model with hidden=2048 in
BF16 issues 72 all-reduces of 4 KiB per forward). It removes the
``torch.distributed`` dispatch layer from the host path and, more importantly,
makes the collective capturable. It is *not* a bandwidth optimisation: for
large payloads the compression-based paths remain the better choice.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from vllm.logger import logger

from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    HCCLLibrary,
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataTypeEnum,
    hcclRedOpTypeEnum,
)

# HcclDataType covers these; model all-reduce only needs SUM over floating-point
# activations. Anything else falls back to the process group.
_SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})

# ACL base formats that carry no physical padding: NCHW(0) and ND(2). Internal
# formats such as NZ(29) pad every row up to 16 elements, so the logical
# ``numel`` no longer matches the number of elements HCCL would move and the
# collective would read/write past the logical tensor.
_BASE_ACL_FORMATS = (0, 2)


def _as_device(device: Any) -> torch.device:
    """``NPUCommunicator`` keeps ``self.device`` as an int; normalise both forms."""
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("npu", device)
    return torch.device(device)


def _format_supported(tensor: torch.Tensor) -> bool:
    """Reject layouts whose logical element count differs from the physical one."""
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        return False
    getter = getattr(torch_npu, "get_npu_format", None)
    if getter is None:
        # Older torch_npu: contiguous tensors are ND unless converted explicitly.
        return True
    try:
        return getter(tensor) in _BASE_ACL_FORMATS
    except Exception:  # noqa: BLE001 - unknown format means "do not risk it"
        return False


class OutplaceHcclAllReduce:
    """Borrowed-communicator out-of-place all-reduce.

    Construct with :meth:`try_create`, which returns ``None`` rather than raising:
    enabling the fast path must never take down start-up on a torch_npu build
    that does not expose the (private) accessors used here.
    """

    def __init__(
        self,
        lib: Any,
        comm: Any,
        stream: torch.npu.Stream | None,
        device: torch.device,
        world_size: int,
        global_rank: int,
    ) -> None:
        self._lib = lib
        # Borrowed from the process group. Never passed to hcclCommDestroy.
        self._comm = comm
        self._stream = stream
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank

    @property
    def available(self) -> bool:
        return self._comm is not None

    @classmethod
    def try_create(
        cls,
        group: dist.ProcessGroup,
        device: Any,
        global_rank: int,
    ) -> OutplaceHcclAllReduce | None:
        device = _as_device(device)

        try:
            backend = group._get_backend(device)
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            logger.warning_once("Out-of-place NPU all-reduce disabled: cannot query the HCCL backend (%s).", exc)
            return None

        for name in ("get_hccl_comm_name", "get_hccl_comm"):
            if not callable(getattr(backend, name, None)):
                logger.warning_once(
                    "Out-of-place NPU all-reduce disabled: this torch_npu backend does not expose %s().",
                    name,
                )
                return None

        # get_coll_stream_id(device) is the explicit collective-stream accessor;
        # _get_stream_id(p2p=False) predates it and is kept for older builds.
        coll_stream_id = getattr(backend, "get_coll_stream_id", None)
        if not callable(coll_stream_id) and not callable(getattr(backend, "_get_stream_id", None)):
            logger.warning_once(
                "Out-of-place NPU all-reduce disabled: this torch_npu backend exposes no collective stream id."
            )
            return None

        try:
            lib = HCCLLibrary()
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("Out-of-place NPU all-reduce disabled: cannot load HCCL (%s).", exc)
            return None

        try:
            with torch.npu.device(device):
                # Go through the process group first: ``get_hccl_comm`` alone may
                # initialise the communicator with default HCCL options instead of
                # the ones configured for this group. torch_npu's ``rankid`` is
                # converted to a local device with ``rankid % device_count``;
                # using a subgroup-local rank can therefore select another NPU.
                try:
                    backend.get_hccl_comm_name(global_rank, init_comm=True)
                except TypeError:
                    backend.get_hccl_comm_name(global_rank)
                comm = backend.get_hccl_comm(global_rank)
                if callable(coll_stream_id):
                    stream_id = int(coll_stream_id(device))
                else:
                    stream_id = int(backend._get_stream_id())
                if not comm or stream_id < 0:
                    logger.warning_once(
                        "Out-of-place NPU all-reduce disabled: communicator/stream unavailable "
                        "(comm=%s, stream_id=%s).",
                        comm,
                        stream_id,
                    )
                    return None

                stream: torch.npu.Stream | None = None
                try:
                    current = torch.npu.current_stream(device)
                    stream = torch.npu.Stream(
                        stream_id=stream_id,
                        device_index=device.index,
                        device_type=current.device_type,
                    )
                except Exception as exc:  # noqa: BLE001
                    # The stream is an optimisation (overlap + graph-safe hand-off).
                    # Without it we still reduce out of place, on the caller's stream.
                    logger.warning_once(
                        "Out-of-place NPU all-reduce: cannot wrap the collective stream (%s); "
                        "falling back to the caller's stream.",
                        exc,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("Out-of-place NPU all-reduce disabled: backend borrow failed (%s).", exc)
            return None

        world_size = dist.get_world_size(group)
        logger.info_once(
            "Out-of-place NPU all-reduce fast path enabled (world_size=%d, dedicated stream=%s).",
            world_size,
            stream is not None,
        )
        return cls(
            lib=lib,
            comm=comm,
            stream=stream,
            device=device,
            world_size=world_size,
            global_rank=global_rank,
        )

    def supported(self, tensor: torch.Tensor) -> bool:
        """Whether this tensor can take the borrowed-communicator path."""
        return (
            self._comm is not None
            and tensor.device == self.device
            and tensor.dtype in _SUPPORTED_DTYPES
            and _format_supported(tensor)
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce-sum ``tensor`` across the group, returning a fresh tensor.

        Callers that cannot guarantee :meth:`supported` should check it first and
        fall back; raising here is reserved for programming errors.
        """
        if self._comm is None:
            raise RuntimeError("OutplaceHcclAllReduce is already destroyed")
        if not self.supported(tensor):
            raise ValueError(
                "Out-of-place HCCL all-reduce requires a contiguous ND/NCHW FP16/BF16/FP32 tensor "
                f"on {self.device}, got dtype={tensor.dtype} device={tensor.device}"
            )

        out = torch.empty_like(tensor)
        if tensor.numel() == 0:
            return out

        with torch.npu.device(self.device):
            current = torch.npu.current_stream(self.device)
            comm_stream = self._stream if self._stream is not None else current
            cross_stream = comm_stream != current
            if cross_stream:
                comm_stream.wait_stream(current)
            # Keep both buffers alive on the collective stream: the caching
            # allocator may otherwise hand a freed block to the producer while
            # HCCL is still reading it.
            tensor.record_stream(comm_stream)
            out.record_stream(comm_stream)

            # Read ``npu_stream`` on every call, after the event/allocator work.
            # Its torch_npu getter drains the host task queue before returning the
            # raw ACL stream; caching the pointer would let this direct HCCL call
            # overtake queued producers or stream-wait events.
            self._lib.hcclAllReduce(
                buffer_type(tensor.data_ptr()),
                buffer_type(out.data_ptr()),
                tensor.numel(),
                hcclDataTypeEnum.from_torch(tensor.dtype),
                hcclRedOpTypeEnum.hcclSum,
                hcclComm_t(self._comm),
                aclrtStream_t(comm_stream.npu_stream),
            )
            if cross_stream:
                current.wait_stream(comm_stream)
        return out

    def destroy(self) -> None:
        """Drop the borrowed handles. The process group destroys them."""
        if self._comm is None:
            return
        with torch.npu.device(self.device):
            # Direct HCCL calls are not WorkHCCL objects, so nothing tracks them
            # when the owning process group tears the communicator down.
            try:
                if self._stream is not None:
                    self._stream.synchronize()
                else:
                    torch.npu.synchronize(self.device)
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
        self._comm = None
        self._stream = None
