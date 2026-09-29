#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fuse a RowParallelLinear's GEMM with its following TP all-reduce.

`RowParallelLinear.forward` emits two custom ops back to back:

    output_parallel = torch.ops.vllm.unquantized_gemm(x, weight, bias)
    output          = torch.ops.vllm.all_reduce(output_parallel, group_name)

CANN provides `aclnnMatmulAllReduce` (exposed as
`torch_npu.npu_mm_all_reduce_base`) which pipelines the collective against the
GEMM along M, so this pass rewrites that pair into a single fused call.

Only worth doing once M is large enough to tile: measured on Ascend950PR, TP2,
bf16, the fused op is *slower* than the unfused pair below M~1024 and only pays
off above it (unfused/fused, comm_mode unset):

    M      o_proj [3072,5120]   down_proj [8704,5120]
     512         0.64x                 0.78x
    1024         1.04x                 1.14x
    2048         1.12x                 1.37x
    4096         1.15x                 1.44x
    8192         1.22x                 1.57x
   16384         1.23x                 1.58x

Both measured shapes cross 1.0x at the same point, so a single configurable M
threshold is used instead of branching on shape. The threshold is enforced at
compile-range granularity via `is_applicable_for_range()`, which keeps the fused
op out of small-batch (decode) graphs entirely instead of branching at runtime.
Ascend config initialization adds the required endpoint immediately below the
configured threshold to `CompilationConfig.compile_ranges_endpoints`.

`aclnnMatmulAllReduce` rejects FRACTAL_NZ weights (`error code is 16`), so the
pass stands down whenever weights may be converted to NZ.
"""

from __future__ import annotations

import torch
import torch_npu
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.distributed import get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import WeightLayoutPolicy, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_matmul_allreduce_group


def _get_hcomm_info(group) -> str:
    """Resolve the HCCL communicator name for a GroupCoordinator's device group.

    Deliberately not cached: a class-wide cache can hand back a handle from a
    different domain or from an already destroyed process group. The backend
    caches its own handle anyway.
    """
    device_group = group.device_group
    rank = torch.distributed.get_rank(device_group)
    global_rank = torch.distributed.get_global_rank(device_group, rank)
    return device_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)


class MatmulAllReducePattern(BasePattern):
    """Matches `unquantized_gemm(x, weight, None)` feeding `all_reduce(...)`.

    Only the bias-free form is matched. In TP>1 `RowParallelLinear` passes
    `bias_ = None` on every rank but rank 0, so matching the bias form too would
    make rank 0's graph differ from the others; Qwen3.5's o_proj / down_proj are
    `bias=False` anyway, so every rank takes the None path here.
    """

    def __init__(self, vllm_config: VllmConfig, group_name: str, hcomm_info: str):
        super().__init__(vllm_config)
        self.group_name = group_name
        self.hcomm_info = hcomm_info

    def get_inputs(self) -> list[torch.Tensor]:
        # Shapes only carry meta information for the pattern matcher; the real
        # ones come from the traced graph.
        x = torch.randn(4, 64, device="npu", dtype=self.dtype)
        weight = torch.randn(128, 64, device="npu", dtype=self.dtype)
        return [x, weight]

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gemm = torch.ops.vllm.unquantized_gemm(x, weight, None)
            return torch.ops.vllm.all_reduce(gemm, self.group_name)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # unquantized_gemm is F.linear, i.e. x @ weight.T, so the fused op
            # takes weight.t(). comm_mode is left unset on purpose: CANN picks
            # it, and an explicit "ccu" measured slower than unfused at every M
            # on Ascend950PR TP2.
            return torch_npu.npu_mm_all_reduce_base(
                x,
                weight.t(),
                self.hcomm_info,
                reduce_op="sum",
            )

        return replacement


class MatmulAllReduceFusionPass(VllmInductorPass):
    """Rewrites RowParallelLinear GEMM + TP all-reduce into one fused NPU op."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="matmul_allreduce_fusion_pass"
        )
        self.enabled = False
        from vllm_ascend.ascend_config import get_ascend_config

        ascend_config = get_ascend_config()
        self.min_tokens = ascend_config.ascend_compilation_config.matmul_allreduce_min_tokens
        logger.info("MatmulAllReduceFusionPass: constructor reached")

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            logger.info("MatmulAllReduce fusion not enabled: TP size is 1")
            return

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            logger.info("MatmulAllReduce fusion not enabled: unsupported dtype %s", dtype)
            return

        # aclnnMatmulAllReduce rejects FRACTAL_NZ weights, so skip the fusion
        # whenever this config would hand it NZ. 310P forces NZ outright; on the
        # other profiles weight_nz_mode == 2 converts bf16/fp16 as well (mode 1
        # leaves them in ND, which is the default).
        if get_current_hardware_profile().weight_layout_policy is WeightLayoutPolicy.FORCE_NZ:
            logger.info("MatmulAllReduce fusion not enabled: hardware forces NZ weight layout")
            return

        if ascend_config.weight_nz_mode == 2:
            logger.info("MatmulAllReduce fusion not enabled: weight_nz_mode=2 converts %s to NZ", dtype)
            return

        # On A5 the fused op must run on its own AICPU MC2 domain, NOT on the
        # plain TP group: that domain is built with the full HCCL_BUFFSIZE and is
        # kept away from decode AllReduce, because AICPU MC2 initialization
        # mutates communicator state. parallel_state builds it under the same
        # fuse_matmul_allreduce switch, so it exists whenever we get here.
        comm_group = tp_group
        if get_ascend_device_type() == AscendDeviceType.A5:
            comm_group = get_matmul_allreduce_group()
            assert comm_group.ranks == tp_group.ranks, "MC2 and TP rank ordering must match"

        # The pattern still keys off the TP group's name, since that is what
        # RowParallelLinear's all_reduce carries in the graph; only the fused
        # op's communicator switches to the MC2 domain.
        MatmulAllReducePattern(
            vllm_config,
            group_name=tp_group.unique_name,
            hcomm_info=_get_hcomm_info(comm_group),
        ).register(self.pattern_match_passes, register_with_backend=False)
        self.enabled = True
        logger.info("MatmulAllReduceFusionPass: pattern registered, enabled")

    def __call__(self, graph: torch.fx.GraphModule) -> None:  # type: ignore[override]
        self.begin()
        logger.info("MAR __call__ entered")
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.info("MAR __call__ matched=%d", self.matched_count)
        # Logged at info level rather than debug: this pass is opt-in and its
        # match count is the only cheap way to confirm from a normal serve log
        # that the fusion actually applied.
        if self.matched_count:
            logger.info("Fused %s matmul_allreduce patterns", self.matched_count)
        else:
            logger.debug("Fused no matmul_allreduce patterns")
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Admit the pass only for ranges entirely at or above the threshold.

        `compile_range` is inclusive of both ends, so gating on `start` means a
        range straddling the threshold (e.g. [1, 16384]) is rejected rather than
        fused for the small sizes it also covers.
        """
        verdict = self.enabled and compile_range.start >= self.min_tokens
        logger.info(
            "MAR is_applicable_for_range(%s) enabled=%s min_tokens=%s -> %s",
            compile_range,
            self.enabled,
            self.min_tokens,
            verdict,
        )
        return verdict
