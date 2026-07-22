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
"""Worker-layer patch for Qwen3.5-VL preprocessing offload (NPU side).

Model forward runs in the worker process, so the on-device re-apply of the
fused rescale+normalize is a worker patch. The matching CPU-side disable of
rescale/normalize is a platform patch (front-end process); see
``patch/platform/patch_qwen3_5_vl_preprocess.py``.

Ref: gitee omniinfer PR !1515, repackaged as vllm-ascend patches.
"""
from functools import lru_cache

import torch
from torchvision.transforms.v2 import functional as tv_functional

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model


@lru_cache(maxsize=4)
def _fuse_mean_std_and_rescale_factor(
    do_normalize, image_mean, image_std, do_rescale, rescale_factor, device
):
    if do_rescale and do_normalize:
        # Fold rescale into normalize so we run a single fused op.
        image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
        image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
        do_rescale = False
    return image_mean, image_std, do_rescale


def _rescale_and_normalize(
    images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
):
    image_mean, image_std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    if do_normalize:
        origin_dtype = images.dtype
        images = tv_functional.normalize(
            images.to(torch.float32), image_mean, image_std
        ).to(origin_dtype)
    elif do_rescale:
        images = images * rescale_factor
    return images


def _ensure_img_pp_cfg(self):
    if getattr(self, "_img_pp_ready", False):
        return
    vision_config = self.config.vision_config
    self.channel = vision_config.in_channels
    self.patch_size = vision_config.patch_size
    self.temporal_patch_size = vision_config.temporal_patch_size
    image_processor = (
        MULTIMODAL_REGISTRY.create_processor(self.model_config)
        .info.get_hf_processor()
        .image_processor
    )
    self.do_rescale = True
    self.do_normalize = True
    self.rescale_factor = image_processor.rescale_factor
    self.image_mean = tuple(image_processor.image_mean)
    self.image_std = tuple(image_processor.image_std)
    self._img_pp_ready = True


def _ascend_process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
    self._ensure_img_pp_cfg()
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        pixel_values = pixel_values.reshape(
            -1, self.channel, self.patch_size, self.patch_size
        )
        pixel_values = _rescale_and_normalize(
            pixel_values, self.do_rescale, self.rescale_factor,
            self.do_normalize, self.image_mean, self.image_std,
        )
        pixel_values = pixel_values.reshape(
            -1,
            self.channel * self.temporal_patch_size
            * self.patch_size * self.patch_size,
        )
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return image_embeds.split(sizes)


def _ascend_process_video_input(self, video_input) -> tuple[torch.Tensor, ...]:
    self._ensure_img_pp_cfg()
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
        pixel_values_videos = pixel_values_videos.reshape(
            -1, self.channel, self.patch_size, self.patch_size
        )
        pixel_values_videos = _rescale_and_normalize(
            pixel_values_videos, self.do_rescale, self.rescale_factor,
            self.do_normalize, self.image_mean, self.image_std,
        )
        pixel_values_videos = pixel_values_videos.reshape(
            -1,
            self.channel * self.temporal_patch_size
            * self.patch_size * self.patch_size,
        )
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values_videos, grid_thw.tolist(),
                rope_type="rope_3d",
            )
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return video_embeds.split(sizes)


Qwen3_5ForConditionalGeneration._ensure_img_pp_cfg = _ensure_img_pp_cfg
Qwen3_5ForConditionalGeneration._process_image_input = _ascend_process_image_input
Qwen3_5ForConditionalGeneration._process_video_input = _ascend_process_video_input
