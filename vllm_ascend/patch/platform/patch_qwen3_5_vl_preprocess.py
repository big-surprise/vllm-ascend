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
# mypy: ignore-errors
"""Platform-layer patch: disable CPU rescale/normalize for Qwen3.5-VL.

The per-pixel rescale + normalize float ops are offloaded to the NPU worker
patch (``patch/worker/patch_qwen3_5_vl_preprocess.py``). Here we only turn
them off in the HF multimodal processor so the CPU preprocess path no longer
runs them, which otherwise serialized prefill entry.
"""
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

_ORIG_CALL_HF = Qwen3VLMultiModalProcessor._call_hf_processor


def _patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    model_type = getattr(self.info.get_hf_config(), "model_type", "")
    is_qwen35 = isinstance(model_type, str) and model_type.startswith("qwen3_5")
    if not is_qwen35:
        return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)

    # Disable CPU-side rescale/normalize; re-applied (fused) on the NPU worker.
    mm_kwargs = dict(mm_kwargs)
    mm_kwargs.setdefault("do_rescale", False)
    mm_kwargs.setdefault("do_normalize", False)
    return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)


Qwen3VLMultiModalProcessor._call_hf_processor = _patched_call_hf_processor
