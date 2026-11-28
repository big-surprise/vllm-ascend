#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
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
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # DecodePriorityScheduler: maximum successful pure-decode batches while
    # prefill waits. Default 4; integer >= 0; 0 disables deliberate deferral.
    # Read once at scheduler initialization. Not sensitive.
    "VLLM_ASCEND_DECODE_BURST_STEPS": lambda: int(os.getenv("VLLM_ASCEND_DECODE_BURST_STEPS", "4")),
    # DecodePriorityScheduler: stop deliberately deferring prefill after this
    # many milliseconds since first observation / last scheduled progress.
    # Default 20; finite float >= 0; 0 disables deliberate deferral. This is
    # not a TTFT deadline and cannot override KV/dependency constraints.
    # Read once at scheduler initialization. Not sensitive.
    "VLLM_ASCEND_PREFILL_MAX_DELAY_MS": lambda: float(os.getenv("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "20")),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Enable the CANN ops-transformer flash_attn dual-operator attention
    # backend (cann_ops_transformer.flash_attn_metadata + flash_attn) for
    # non-MLA, non-sparse full-attention layers. The operator ABI is only
    # available on the Ascend 950PR/950DT family (A5 profile) with a
    # cann_ops_transformer build that ships the two APIs. Default: 0
    # (disabled); valid values: 0 or 1. Read once at platform backend
    # selection. Not sensitive.
    "VLLM_ASCEND_ENABLE_FLASH_ATTN": lambda: _strict_binary_env("VLLM_ASCEND_ENABLE_FLASH_ATTN"),
    # Record the arguments handed to the CANN flash_attn operator -- cu_seqlens_q,
    # cu_seqlens_kv, seqused_q, seqused_kv, mask_mode, max_seqlen_q,
    # max_seqlen_kv -- plus the step's state, token/request counts and
    # block-table shape. Emitted from the metadata builder's per-step host
    # entry, so it also covers steps executed by ACL-graph replay (decode),
    # which the forward pass cannot report. One line per distinct step shape:
    # 0 disables, N > 0 emits at most N shapes, N < 0 emits every step. Debug
    # aid; not sensitive.
    "VLLM_ASCEND_FLASH_ATTN_DEBUG": lambda: int(os.getenv("VLLM_ASCEND_FLASH_ATTN_DEBUG", "0")),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
