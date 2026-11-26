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
from vllm.logger import logger


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

        # Opt-in out-of-place all-reduce fast path
        # (--additional-config '{"allreduce_outplace": true}', off by default).
        # The default path reduces in place; the fast path issues a direct
        # HcclAllReduce into a fresh receive buffer on the process group's
        # borrowed communicator/collective stream. See hccl_outplace_ar.py.
        self.oop_all_reduce = None
        if self.world_size > 1 and _allreduce_outplace_requested():
            from vllm_ascend.distributed.device_communicators.hccl_outplace_ar import (
                HcclOutOfPlaceAllReduce,
            )

            # HCCL communicator lookup is indexed by the group-local rank,
            # not the global rank (they differ when the group is not world).
            self.oop_all_reduce = HcclOutOfPlaceAllReduce.build(
                group=self.device_group,
                device=self.device,
                rank_in_group=self.rank_in_group,
            )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Per-call eligibility keeps unusual dtypes/layouts on the default
        # in-place path instead of failing the request.
        if self.oop_all_reduce is not None and self.oop_all_reduce.accepts(input_):
            return self.oop_all_reduce.all_reduce(input_)
        return super().all_reduce(input_)

    def destroy(self) -> None:
        if self.oop_all_reduce is not None:
            self.oop_all_reduce.shutdown()
            self.oop_all_reduce = None
        super().destroy()


def _allreduce_outplace_requested() -> bool:
    """Defensive read: the Ascend config may not be initialised yet when the
    communicator is constructed, and "config unavailable" must mean "off"."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().allreduce_outplace)
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(
            "allreduce_outplace stays disabled: Ascend config unreadable (%s).", exc
        )
        return False
