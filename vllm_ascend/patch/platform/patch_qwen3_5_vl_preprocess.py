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
"""Platform-layer: Qwen3.5-VL preprocess offload (resize + patchify -> NPU).

CPU front-end keeps only image decode; resize, rescale/normalize AND patchify
are offloaded to the NPU worker. CPU emits flat RAW uint8 images plus the
original (H, W) per image so the worker can reshape, resize (bicubic+antialias,
matching HF), patchify and normalize on device:
  pixel_values   = 1-D concat of per-image flat (C*H*W) uint8
  image_grid_thw = (num_img, 3), grid_t=1 (target grid from smart_resize)
  image_hw       = (num_img, 2), original (H, W) per image
"""
import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
from vllm.multimodal.inputs import MultiModalFieldConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from transformers.image_processing_utils import BatchFeature


def _is_qwen3_5(info):
    mt = getattr(info.get_hf_config(), "model_type", "")
    return isinstance(mt, str) and mt.startswith("qwen3_5")


# ---- offload _preprocess: emit flat RAW uint8 + (H,W), skip resize/patchify -
_ORIG_PRE = Qwen2VLImageProcessor._preprocess


def _preprocess_offload(
    self, images, do_resize, size, resample, do_rescale, rescale_factor,
    do_normalize, image_mean, image_std, patch_size, temporal_patch_size,
    merge_size, disable_grouping, return_tensors, **kwargs,
):
    if not getattr(self, "_offload_patchify", False):
        return _ORIG_PRE(
            self, images, do_resize, size, resample, do_rescale, rescale_factor,
            do_normalize, image_mean, image_std, patch_size, temporal_patch_size,
            merge_size, disable_grouping, return_tensors, **kwargs,
        )
    # Skip resize/patchify on CPU. Emit flat RAW uint8 + original (H,W) so the
    # NPU worker can reshape back, resize (bicubic+antialias), patchify, norm.
    flats, grids, hws = [], [], []
    for img in images:  # (C, H, W) uint8, original size
        c, h, w = img.shape
        rh, rw = smart_resize(
            h, w, factor=patch_size * merge_size,
            min_pixels=size.shortest_edge, max_pixels=size.longest_edge,
        )
        grids.append([1, rh // patch_size, rw // patch_size])  # target grid
        hws.append([h, w])                                     # original dims
        flats.append(img.reshape(-1).contiguous())
    pixel_values = torch.cat(flats, dim=0)
    image_grid_thw = torch.tensor(grids, dtype=torch.long)
    image_hw = torch.tensor(hws, dtype=torch.long)
    return BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
              "image_hw": image_hw},
        tensor_type=return_tensors,
    )


# ---- field config: pixel_values split by C*H*W per image, carry image_hw ----
_ORIG_MM_FIELDS = Qwen3VLMultiModalProcessor._get_mm_fields_config


def _patched_get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
    base = dict(_ORIG_MM_FIELDS(self, hf_inputs, hf_processor_mm_kwargs))
    if _is_qwen3_5(self.info):
        vc = self.info.get_hf_config().vision_config
        c = vc.in_channels
        # pixel_values is flat RAW image bytes -> split by C*H*W.
        hw = hf_inputs.get("image_hw", torch.empty((0, 2), dtype=torch.long))
        sizes = hw.prod(-1) * c
        base["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", sizes)
        base["image_hw"] = MultiModalFieldConfig.batched("image", keep_on_cpu=True)
    return base


# ---- enable offload + disable CPU rescale/normalize -------------------------
_ORIG_CALL_HF = Qwen3VLMultiModalProcessor._call_hf_processor


def _patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    if not _is_qwen3_5(self.info):
        return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)
    try:
        self.info.get_image_processor()._offload_patchify = True
    except Exception:
        pass
    mm_kwargs = dict(mm_kwargs)
    mm_kwargs.setdefault("do_rescale", False)
    mm_kwargs.setdefault("do_normalize", False)
    return _ORIG_CALL_HF(self, prompt, mm_data, mm_kwargs, tok_kwargs)


Qwen2VLImageProcessor._preprocess = _preprocess_offload
Qwen3VLMultiModalProcessor._get_mm_fields_config = _patched_get_mm_fields_config
Qwen3VLMultiModalProcessor._call_hf_processor = _patched_call_hf_processor
