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
"""Worker-layer: Qwen3.5-VL Step-2 offload (NPU side).

Image path receives the flat uint8 resized-image contract (platform patch) and
does rescale+normalize + patchify on device. Video keeps Step-1 behavior
(CPU still patchifies video; here only rescale+normalize on device).
ViT/prefill/decode timing lives in patch_qwen3_5_timing.py (separate).
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
        image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
        image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
        do_rescale = False
    return image_mean, image_std, do_rescale


def _rescale_and_normalize(images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
    image_mean, image_std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize, image_mean=image_mean, image_std=image_std,
        do_rescale=do_rescale, rescale_factor=rescale_factor, device=images.device,
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
    vc = self.config.vision_config
    self.channel = vc.in_channels
    self.patch_size = vc.patch_size
    self.temporal_patch_size = vc.temporal_patch_size
    self.merge_size = vc.spatial_merge_size
    ip = (
        MULTIMODAL_REGISTRY.create_processor(self.model_config)
        .info.get_hf_processor().image_processor
    )
    self.do_rescale = True
    self.do_normalize = True
    self.rescale_factor = ip.rescale_factor
    self.image_mean = tuple(ip.image_mean)
    self.image_std = tuple(ip.image_std)
    # Batched-normalize fast path: when mean/std are uniform across channels the
    # fused rescale+normalize is a scalar affine y = x*inv_std_r - mean_over_std,
    # which commutes with patchify's rearrange+temporal-expand. So we can apply
    # it ONCE to the concatenated patches instead of per-image (fewer kernels).
    m, s = self.image_mean, self.image_std
    self._uniform_norm = (len(set(m)) == 1 and len(set(s)) == 1)
    if self._uniform_norm:
        mean0, std0 = float(m[0]), float(s[0])
        # x is raw pixel in [0,255]; HF does (x*rescale - mean)/std.
        self._norm_scale = self.rescale_factor / std0
        self._norm_bias = -mean0 / std0
    self._img_pp_ready = True


def _patchify_on_device(self, img):
    # img: (C, rh, rw) -> (gh*gw, C*tps*ps*ps), matches HF layout
    c, rh, rw = img.shape
    ps, ms, tps = self.patch_size, self.merge_size, self.temporal_patch_size
    gh, gw = rh // ps, rw // ps
    x = img.reshape(c, gh // ms, ms, ps, gw // ms, ms, ps)
    x = x.permute(1, 4, 2, 5, 0, 3, 6)
    x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, tps, -1, -1)
    return x.reshape(gh * gw, c * tps * ps * ps)


def _ascend_parse_image_input(self, **kwargs):
    pixel_values = kwargs.pop("pixel_values", None)
    image_embeds = kwargs.pop("image_embeds", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    image_hw = kwargs.pop("image_hw", None)  # Step-3: original (H,W) per image
    if pixel_values is None and image_embeds is None:
        return None
    if pixel_values is not None:
        return {"type": "pixel_values", "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw, "image_hw": image_hw}
    return {"type": "image_embeds", "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw}


def _ascend_process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
    self._ensure_img_pp_cfg()
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        # Step-3: flat RAW uint8 -> reshape (C,H,W) via image_hw -> NPU resize
        # (bicubic+antialias, matches HF) -> rescale+normalize -> patchify.
        import torch.nn.functional as _NF
        flat = image_input["pixel_values"].to(self.visual.device)
        image_hw = image_input["image_hw"]
        c, ps = self.channel, self.patch_size
        hw_list = image_hw.tolist() if hasattr(image_hw, "tolist") else list(image_hw)
        sizes = [c * h * w for (h, w) in hw_list]
        chunks = flat.split(sizes)
        patched = []
        for chunk, (h, w), thw in zip(chunks, hw_list, grid_thw.tolist()):
            _, gh, gw = thw
            rh, rw = gh * ps, gw * ps
            # Per-image (variable size) must stay in the loop: reshape+resize
            # +clip+round only. Keep float32; skip per-image normalize/casts.
            img = chunk.reshape(1, c, h, w).float()
            img = _NF.interpolate(img, size=[rh, rw], mode="bicubic",
                                  align_corners=False, antialias=True)
            img = img.clamp(0, 255).round().squeeze(0)  # (C,rh,rw) f32, raw [0,255]
            patched.append(self._patchify_on_device(img))
        pixel_values = torch.cat(patched, dim=0)
        # Batched rescale+normalize on the full concat (1 op instead of 11x).
        if self._uniform_norm:
            pixel_values = (pixel_values * self._norm_scale + self._norm_bias
                            ).to(self.visual.dtype)
        else:
            pixel_values = _rescale_and_normalize(
                pixel_values.to(self.visual.dtype), self.do_rescale,
                self.rescale_factor, self.do_normalize,
                self.image_mean, self.image_std,
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
        pv = video_input["pixel_values_videos"].type(self.visual.dtype)
        pv = pv.reshape(-1, self.channel, self.patch_size, self.patch_size)
        pv = _rescale_and_normalize(
            pv, self.do_rescale, self.rescale_factor,
            self.do_normalize, self.image_mean, self.image_std,
        )
        pv = pv.reshape(
            -1, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pv, grid_thw.tolist(), rope_type="rope_3d"
            )
        video_embeds = self.visual(pv, grid_thw=grid_thw)
    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return video_embeds.split(sizes)


Qwen3_5ForConditionalGeneration._ensure_img_pp_cfg = _ensure_img_pp_cfg
Qwen3_5ForConditionalGeneration._patchify_on_device = _patchify_on_device
Qwen3_5ForConditionalGeneration._parse_and_validate_image_input = _ascend_parse_image_input
Qwen3_5ForConditionalGeneration._process_image_input = _ascend_process_image_input
Qwen3_5ForConditionalGeneration._process_video_input = _ascend_process_video_input
