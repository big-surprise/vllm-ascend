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

"""Out-of-place NPU all-reduce fast path on top of an existing process group.

Why this exists
---------------
``DeviceCommunicatorBase.all_reduce`` calls ``torch.distributed.all_reduce``
which reduces *in place* and returns the input tensor. Nothing notices in
eager mode, but two call sites care about the difference:

1. **NPU graph capture.** Inside a captured region the producer's output
   buffer is a *static* input of the graph. An in-place reduction rewrites
   that buffer on every replay, so the second replay starts from values that
   were already reduced instead of freshly computed ones. An out-of-place
   reduction (fresh receive buffer per call) keeps the producer's buffer
   read-only for the collective.
2. **Host-side dispatch cost.** For latency-bound messages (small hidden
   size, many layers), the ``torch.distributed`` dispatch layer around the
   collective is a measurable share of the per-call overhead. Issuing
   ``HcclAllReduce`` directly removes that layer.

Design
------
The process group (``ProcessGroupHCCL``) owns both the HCCL communicator and
its dedicated collective stream. This module *borrows* both through the
backend's Python accessors and never creates a second communicator (unlike
``pyhccl.PyHcclCommunicator``, which runs ``HcclCommInitRootInfo`` for weight
transfer). Staying in the same HCCL context keeps collective ordering
consistent with every other collective issued by the group, at zero extra
device memory and zero extra init handshake.

Safety nets
-----------
* Construction is best-effort: :meth:`HcclOutOfPlaceAllReduce.build` returns
  ``None`` on any failure (older torch_npu without the accessors, HCCL
  library missing, ...), so opting in can never break start-up.
* Eligibility is checked per call (:meth:`accepts`): only contiguous
  FP16/BF16/FP32 tensors in a base ACL format (ND/NCHW) take the fast path;
  everything else transparently falls back to the default in-place path.
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

# Model all-reduce is always a SUM over floating-point activations.
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ACL base formats without physical padding: NCHW(0) and ND(2). Internal
# formats such as NZ(29) pad each row, so the logical ``numel`` no longer
# matches the physical element count HCCL would move.
_BASE_FORMAT_IDS = frozenset({0, 2})


def _normalize_device(device: Any) -> torch.device:
    """Accept both ``torch.device`` and the plain int that
    ``NPUCommunicator`` stores in ``self.device``."""
    if isinstance(device, torch.device):
        return device
    return torch.device("npu", int(device))


def _tensor_in_base_format(tensor: torch.Tensor) -> bool:
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        return False
    get_format = getattr(torch_npu, "get_npu_format", None)
    if get_format is None:
        # Ancient torch_npu: contiguous tensors are ND unless converted.
        return True
    try:
        return get_format(tensor) in _BASE_FORMAT_IDS
    except Exception:  # noqa: BLE001 - when unsure, stay on the default path
        return False


class HcclOutOfPlaceAllReduce:
    """Direct ``HcclAllReduce`` with a fresh receive buffer per call.

    The communicator handle and collective stream are borrowed from the
    process group; this object never destroys them.
    """

    def __init__(
        self,
        hccl_lib: HCCLLibrary,
        comm_handle: int,
        coll_stream: torch.npu.Stream,
        device: torch.device,
    ) -> None:
        self._lib = hccl_lib
        self._comm = comm_handle
        self._coll_stream = coll_stream
        self.device = device

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        group: dist.ProcessGroup,
        device: Any,
        rank_in_group: int,
    ) -> "HcclOutOfPlaceAllReduce | None":
        """Create the fast path, or return ``None`` when unavailable."""
        device = _normalize_device(device)

        try:
            backend = group._get_backend(device)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once(
                "allreduce_outplace stays disabled: cannot get the HCCL backend (%s).", exc
            )
            return None

        required = ("get_hccl_comm_name", "get_hccl_comm")
        missing = [n for n in required if not callable(getattr(backend, n, None))]
        get_stream_id = getattr(backend, "get_coll_stream_id", None)
        if not callable(get_stream_id):
            get_stream_id = getattr(backend, "_get_stream_id", None)
        if missing or not callable(get_stream_id):
            logger.warning_once(
                "allreduce_outplace stays disabled: this torch_npu build lacks "
                "ProcessGroupHCCL accessors (missing: %s, stream accessor: %s).",
                missing or "none",
                "yes" if callable(get_stream_id) else "no",
            )
            return None

        try:
            hccl_lib = HCCLLibrary()
        except Exception as exc:  # noqa: BLE001
            logger.warning_once(
                "allreduce_outplace stays disabled: cannot load libhccl (%s).", exc
            )
            return None

        try:
            with torch.npu.device(device):
                # ``get_hccl_comm_name`` first: calling ``get_hccl_comm``
                # directly may lazily init the communicator with default
                # options instead of the ones configured for this group.
                try:
                    backend.get_hccl_comm_name(rank_in_group, init_comm=True)
                except TypeError:
                    # Older torch_npu without the init_comm keyword.
                    backend.get_hccl_comm_name(rank_in_group)
                comm_handle = backend.get_hccl_comm(rank_in_group)

                try:
                    stream_id = int(get_stream_id(device))
                except TypeError:
                    # ``_get_stream_id`` takes no device argument.
                    stream_id = int(get_stream_id())
                if not comm_handle or stream_id < 0:
                    raise RuntimeError(
                        f"empty HCCL handle ({comm_handle}) or stream id ({stream_id})"
                    )
                coll_stream = torch.npu.Stream(
                    stream_id=stream_id,
                    device_index=device.index,
                    device_type=torch.npu.current_stream(device).device_type,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning_once(
                "allreduce_outplace stays disabled: borrowing from the process group failed (%s).", exc
            )
            return None

        logger.info_once(
            "Out-of-place NPU all-reduce fast path is ON "
            "(borrowed ProcessGroupHCCL communicator, dedicated collective stream)."
        )
        return cls(hccl_lib, comm_handle, coll_stream, device)

    # ------------------------------------------------------------------ #
    # eligibility
    # ------------------------------------------------------------------ #
    def accepts(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` may take the fast path."""
        return (
            self._comm is not None
            and tensor.device == self.device
            and tensor.dtype in _FAST_DTYPES
            and _tensor_in_base_format(tensor)
        )

    # ------------------------------------------------------------------ #
    # collective
    # ------------------------------------------------------------------ #
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce-sum ``tensor`` across the group into a *new* tensor."""
        if self._comm is None:
            raise RuntimeError("HcclOutOfPlaceAllReduce already shut down")

        out = torch.empty_like(tensor)
        if tensor.numel() == 0:
            return out

        with torch.npu.device(self.device):
            compute_stream = torch.npu.current_stream(self.device)
            if compute_stream != self._coll_stream:
                # Order the collective after everything already queued by
                # the caller, and the caller after the collective.
                self._coll_stream.wait_stream(compute_stream)
            # Pin both buffers against the caching allocator until the
            # collective stream is done with them.
            tensor.record_stream(self._coll_stream)
            out.record_stream(self._coll_stream)

            # Read ``npu_stream`` at call time, after the event/allocator
            # work above: the torch_npu getter drains the host task queue
            # before handing out the raw ACL stream, so a cached pointer
            # could let this call overtake queued producers.
            self._lib.hcclAllReduce(
                buffer_type(tensor.data_ptr()),
                buffer_type(out.data_ptr()),
                tensor.numel(),
                hcclDataTypeEnum.from_torch(tensor.dtype),
                hcclRedOpTypeEnum.hcclSum,
                hcclComm_t(self._comm),
                aclrtStream_t(self._coll_stream.npu_stream),
            )
            if compute_stream != self._coll_stream:
                compute_stream.wait_stream(self._coll_stream)
        return out

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Drop the borrowed handles (the process group owns their lifetime).

        Direct HCCL calls are not ``WorkHCCL`` objects, so nothing else
        tracks them: drain the collective stream before the owning process
        group tears the communicator down.
        """
        if self._comm is None:
            return
        try:
            with torch.npu.device(self.device):
                self._coll_stream.synchronize()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        self._comm = None
        self._coll_stream = None  # type: ignore[assignment]
