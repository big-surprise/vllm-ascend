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

``RowParallelLinear.forward`` emits two custom ops back to back::

    output_parallel = torch.ops.vllm.unquantized_gemm(x, weight, bias)
    output          = torch.ops.vllm.all_reduce(output_parallel, group_name)

CANN provides ``aclnnMatmulAllReduce`` (exposed as
``torch_npu.npu_mm_all_reduce_base``) which pipelines the collective against
the GEMM along M, so this pass rewrites that pair into a single fused call.

Only worth doing once M is large enough to tile: measured on Ascend950PR, TP2,
bf16, the fused op is *slower* than the unfused pair below M~1024 and only pays
off above it (unfused/fused, comm_mode unset)::

    M      o_proj [3072,5120]   down_proj [8704,5120]
     512         0.64x                 0.78x
    1024         1.04x                 1.14x
    2048         1.12x                 1.37x
    4096         1.15x                 1.44x
    8192         1.22x                 1.57x
   16384         1.23x                 1.58x

Both measured shapes cross 1.0x at the same point, so a single configurable M
threshold is used instead of branching on shape. The threshold is enforced at
compile-range granularity via ``is_applicable_for_range()``, which keeps the
fused op out of small-batch (decode) graphs entirely instead of branching at
runtime. ``AscendConfig.derive_and_validate`` adds the required endpoint
immediately below the configured threshold to
``CompilationConfig.compile_ranges_endpoints``.

``aclnnMatmulAllReduce`` rejects FRACTAL_NZ weights (``error code is 16``), so
the pass stands down whenever weights may be converted to NZ.

Bias handling
-------------
With TP>1 ``RowParallelLinear`` passes ``bias_ = None`` on every rank but rank
0. Matching only the bias-free form would leave rank 0 on the plain TP
``all_reduce`` while every other rank issued the fused op on the MC2 domain --
two different collectives in the same position, which deadlocks.

Both forms are therefore matched. ``aclnnMatmulAllReduce`` applies ``bias``
before the reduction, which would multiply it by the TP size on rank 0, so the
biased form is rewritten to ``fused(x, weight.t()) + bias`` instead. That is
exactly what vLLM computes today (rank 0 adds the bias before its own
all-reduce) and keeps every rank on the same collective.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from vllm.compilation.passes.inductor_pass import enable_fake_mode
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import WeightLayoutPolicy, get_current_hardware_profile
from vllm_ascend.distributed.parallel_state import get_matmul_allreduce_group

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

_GEMM_OP = torch.ops.vllm.unquantized_gemm.default
_ALL_REDUCE_OP = torch.ops.vllm.all_reduce.default


def _arg(node: fx.Node, index: int, name: str, default=None):
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def _get_hcomm_info(group) -> str:
    """Resolve the HCCL communicator name for a GroupCoordinator's device group.

    Deliberately not cached at class level: a process-wide cache can hand back
    a handle from a different domain or from an already destroyed process
    group. The backend caches its own handle anyway.
    """
    backend = group.device_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(group.rank_in_group)


class _MatmulAllReducePattern(BasePattern):
    """Shared plumbing for the bias-free and biased ``gemm -> all_reduce`` forms."""

    def __init__(self, vllm_config: VllmConfig, group_name: str, hcom_name: str):
        super().__init__(vllm_config)
        self.group_name = group_name
        self.hcom_name = hcom_name

    def _empty(self, *shape: int) -> torch.Tensor:
        dtype = self.dtype if self.dtype in _SUPPORTED_DTYPES else torch.bfloat16
        device = self.vllm_config.device_config.device if self.vllm_config.device_config else "cpu"
        return torch.empty(shape, device=device, dtype=dtype)

    def get_inputs(self) -> list[torch.Tensor]:
        # Shapes only carry meta information for the pattern matcher; the real
        # ones come from the traced graph.
        return [self._empty(8, 512), self._empty(256, 512)]

    def _get_extra_check(self) -> Callable[[Match], bool]:
        """Reject matches this pattern cannot safely rewrite.

        The group name is pinned by the pattern closure already, but the graph
        node's ``group_name`` argument is re-checked because ``all_reduce`` is
        a generic op that ``vocab_parallel_embedding`` also emits.
        """

        def supported(match: Match) -> bool:
            gemm = next((node for node in match.nodes if node.target is _GEMM_OP), None)
            all_reduce = next((node for node in match.nodes if node.target is _ALL_REDUCE_OP), None)
            if gemm is None or all_reduce is None:
                return False
            if _arg(all_reduce, 1, "group_name") != self.group_name:
                return False

            x, weight = gemm.args[0], gemm.args[1]
            if not isinstance(x, fx.Node) or not isinstance(weight, fx.Node):
                return False
            x_val, weight_val = x.meta.get("val"), weight.meta.get("val")
            if x_val is None or weight_val is None:
                return False
            if x_val.dim() != 2 or weight_val.dim() != 2:
                return False
            if x_val.dtype not in _SUPPORTED_DTYPES or x_val.dtype != weight_val.dtype:
                return False
            return True

        return supported

    @enable_fake_mode
    def register(self, pm_pass: PatternMatcherPass) -> None:
        """Register into this pass instance only.

        ``nge.register_replacement`` is process-global and would apply the
        fusion to every compiled graph, bypassing the compile-range threshold
        that ``is_applicable_for_range`` enforces. The MC2 communicator handle
        is per-process too, so a global registration could leak one rank's
        handle into another rank's graph.
        """
        pm.register_replacement(
            self.get_pattern(),
            self.get_replacement(),
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=self._get_extra_check(),
        )


class MatmulAllReducePattern(_MatmulAllReducePattern):
    """Match ``gemm(x, weight, None) -> all_reduce``."""

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gemm = torch.ops.vllm.unquantized_gemm.default(x, weight, None)
            return torch.ops.vllm.all_reduce.default(gemm, self.group_name)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # unquantized_gemm is F.linear, i.e. x @ weight.T, so the fused op
            # takes weight.t(). comm_mode is left to the device adaptor.
            return DeviceOperator.npu_mm_all_reduce_base(x, weight.t(), self.hcom_name, reduce_op="sum")

        return replacement


class BiasMatmulAllReducePattern(_MatmulAllReducePattern):
    """Match ``gemm(x, weight, bias) -> all_reduce``.

    Only rank 0 of a row-parallel layer carries a bias, so this form is fused
    to ``fused(...) + bias`` (see the module docstring) rather than passing
    ``bias`` into ``aclnnMatmulAllReduce``, which would add it before the
    reduction and scale it by the TP size.
    """

    def _get_extra_check(self) -> Callable[[Match], bool]:
        base_check = super()._get_extra_check()

        def supported(match: Match) -> bool:
            if not base_check(match):
                return False
            gemm = next((node for node in match.nodes if node.target is _GEMM_OP), None)
            if gemm is None:
                return False
            bias = _arg(gemm, 2, "bias")
            if not isinstance(bias, fx.Node):
                return False
            bias_val = bias.meta.get("val")
            gemm_val = gemm.meta.get("val")
            return (
                bias_val is not None
                and bias_val.dim() == 1
                and gemm_val is not None
                and bias_val.dtype == gemm_val.dtype
                and bias_val.shape[0] == gemm_val.shape[-1]
            )

        return supported

    def get_inputs(self) -> list[torch.Tensor]:
        return [self._empty(8, 512), self._empty(256, 512), self._empty(256)]

    def get_pattern(self):
        def pattern(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            gemm = torch.ops.vllm.unquantized_gemm.default(x, weight, bias)
            return torch.ops.vllm.all_reduce.default(gemm, self.group_name)

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            fused = DeviceOperator.npu_mm_all_reduce_base(x, weight.t(), self.hcom_name, reduce_op="sum")
            return fused + bias

        return replacement


class MatmulAllReduceFusionPass(VllmInductorPass):
    """Rewrites RowParallelLinear GEMM + TP all-reduce into one fused NPU op."""

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.pattern_match_passes = PatternMatcherPass(pass_name="matmul_allreduce_fusion_pass")
        self.enabled = False
        self.matched_count = 0

        from vllm_ascend.ascend_config import get_ascend_config

        ascend_config = get_ascend_config()
        self.min_tokens = ascend_config.ascend_compilation_config.matmul_allreduce_min_tokens

        tp_group = get_tp_group()
        if tp_group.world_size <= 1:
            logger.info("MatmulAllReduce fusion not enabled: TP size is 1")
            return

        dtype = config.model_config.dtype
        if dtype not in _SUPPORTED_DTYPES:
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
        # plain TP group: that domain is built with the full HCCL_BUFFSIZE and
        # is kept away from decode AllReduce, because AICPU MC2 initialization
        # mutates communicator state. parallel_state builds it under the same
        # fuse_matmul_allreduce switch, so it exists whenever we get here.
        comm_group = tp_group
        if get_ascend_device_type() == AscendDeviceType.A5:
            comm_group = get_matmul_allreduce_group()
            assert comm_group.ranks == tp_group.ranks, "MC2 and TP rank ordering must match"

        # The pattern still keys off the TP group's name, since that is what
        # RowParallelLinear's all_reduce carries in the graph; only the fused
        # op's communicator switches to the MC2 domain.
        group_name = tp_group.unique_name
        hcom_name = _get_hcomm_info(comm_group)

        MatmulAllReducePattern(config, group_name, hcom_name).register(self.pattern_match_passes)
        BiasMatmulAllReducePattern(config, group_name, hcom_name).register(self.pattern_match_passes)
        self.enabled = True

    def __call__(self, graph: fx.Graph) -> None:
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        # Logged at info level rather than debug: this pass is opt-in and its
        # match count is the only cheap way to confirm from a normal serve log
        # that the fusion actually applied.
        if self.matched_count:
            logger.info("Fused %d matmul-allreduce patterns", self.matched_count)
        else:
            logger.debug("Fused no matmul-allreduce patterns")
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """Admit the pass only for ranges entirely at or above the threshold.

        ``compile_range`` is inclusive of both ends, so gating on ``start``
        means a range straddling the threshold (e.g. [1, 16384]) is rejected
        rather than fused for the small sizes it also covers.
        """
        if not self.enabled:
            return False
        applicable = compile_range.start >= self.min_tokens
        if not applicable:
            logger.debug(
                "MatmulAllReduce fusion skipped for range %s (min_tokens=%d)",
                compile_range,
                self.min_tokens,
            )
        return applicable
