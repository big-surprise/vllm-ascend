#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase


class _NpuAll2AllManager:
    """No-op all2all_manager for NPU. Used by vLLM main's fault-tolerance
    check (data_parallel_size > 1 and is_moe); NPU does not register a real
    one because it uses mc2 / all_gather for MoE communication.
    """

    @property
    def support_fault_tolerance(self) -> bool:
        return False

    def query_fault(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")

    def query_active_mask(self) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool, device="cpu")


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            use_all2all=use_all2all,
        )
        self.device = torch.npu.current_device()
        self.ca_comm = None
        # vLLM #53576 reads this CUDA-only communicator during graph capture.
        # Keep the shared coordinator protocol available without enabling the
        # FlashInfer PCIe IPC backend on NPU.
        self.fi_pcie_ipc_ar_comm = None
        self.all2all_manager = _NpuAll2AllManager()

        # Out-of-place all-reduce fast path, opt-in via
        # ``--additional-config '{"allreduce_outplace": true}'``.
        # Borrows this group's communicator instead of creating one, so the
        # collective stays in the same HCCL context as every other group
        # collective and honours the no-mutation contract of the all-reduce op.
        self.outplace_ar = None
        if self.world_size > 1 and _outplace_allreduce_enabled():
            from vllm_ascend.distributed.device_communicators.hccl_outplace import (
                OutplaceHcclAllReduce,
            )

            self.outplace_ar = OutplaceHcclAllReduce.try_create(
                group=self.device_group,
                device=self.device,
                # Rank *within* the group: HCCL comm lookup takes the group-local
                # rank, while ``self.rank`` is the global rank for a non-stateless
                # group (they differ whenever the group does not span the world).
                rank_in_group=self.rank_in_group,
            )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Per-call fallback: the fast path only covers contiguous base-format
        # FP16/BF16/FP32 tensors, so unusual dtypes/layouts keep the default
        # behaviour instead of failing the request.
        if self.outplace_ar is not None and self.outplace_ar.supported(input_):
            return self.outplace_ar.all_reduce(input_)
        return super().all_reduce(input_)

    def destroy(self) -> None:
        if self.outplace_ar is not None:
            self.outplace_ar.destroy()
            self.outplace_ar = None
        super().destroy()


def _outplace_allreduce_enabled() -> bool:
    """Read the switch defensively: the config may not be initialised yet."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().allreduce_outplace)
    except Exception as exc:  # noqa: BLE001 - config unavailable means "off"
        logger.warning_once("Out-of-place NPU all-reduce stays disabled: %s.", exc)
        return False

