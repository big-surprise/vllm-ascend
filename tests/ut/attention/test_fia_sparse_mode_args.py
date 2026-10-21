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

import inspect
import os
from unittest.mock import patch

import pytest

from tests.ut.base import TestBase
from vllm_ascend.attention import attention_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


def _make_impl(sliding_window=None):
    """Build an impl without running the full __init__ (requires NPU + config)."""
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl.sliding_window = sliding_window
    return impl


class TestFiaSparseModeSelection(TestBase):
    """Sparse mode choice for npu_fused_infer_attention_score."""

    def test_causal_prefill_only_defaults_to_right_down_causal(self):
        assert _make_impl()._resolve_fia_sparse_mode(causal=True, is_prefill_only=True) == 3

    @patch.dict(os.environ, {"VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE": "2"})
    def test_causal_prefill_only_honours_left_up_causal_override(self):
        assert _make_impl()._resolve_fia_sparse_mode(causal=True, is_prefill_only=True) == 2

    @patch.dict(os.environ, {"VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE": "2"})
    def test_mixed_batch_stays_on_right_down_causal(self):
        """decode entries have q_len < kv_len; leftUpCausal would drop history."""
        assert _make_impl()._resolve_fia_sparse_mode(causal=True, is_prefill_only=False) == 3

    @patch.dict(os.environ, {"VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE": "2"})
    def test_decode_only_stays_on_right_down_causal(self):
        assert _make_impl()._resolve_fia_sparse_mode(causal=True, is_prefill_only=False) == 3

    def test_non_causal_uses_default_mask_mode(self):
        assert _make_impl()._resolve_fia_sparse_mode(causal=False, is_prefill_only=True) == 0

    @patch.dict(os.environ, {"VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE": "2"})
    def test_sliding_window_wins_over_causal_override(self):
        assert _make_impl(sliding_window=128)._resolve_fia_sparse_mode(causal=True, is_prefill_only=True) == 4

    @patch.dict(os.environ, {"VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE": "5"})
    def test_out_of_range_mode_is_rejected(self):
        with pytest.raises(ValueError, match="VLLM_ASCEND_FIA_PREFILL_SPARSE_MODE"):
            _make_impl()._resolve_fia_sparse_mode(causal=True, is_prefill_only=True)


class TestFiaAttenMaskAlwaysForwarded(TestBase):
    """atten_mask must reach the operator on every sparse mode but 0.

    aclnnFusedInferAttentionScore rejects an empty atten_mask with
    "when sparse_mode is 3 (not 0), atten_mask cannot be empty" (EZ0026),
    so the backend must never filter it out, even though the operator only
    builds the mask itself for modes 2/3/4 and never reads the buffer.
    """

    def test_no_mask_helper_exists_any_more(self):
        """Guard against reintroducing a filter that can null the mask."""
        assert not hasattr(AscendAttentionBackendImpl, "_resolve_fia_atten_mask")

    def test_every_fia_call_site_forwards_attn_metadata_mask(self):
        """Every FIA call passes attn_metadata.attn_mask through unchanged."""
        source = inspect.getsource(attention_v1)
        assert "_resolve_fia_atten_mask" not in source
        forwarded = source.count("atten_mask=attn_metadata.attn_mask")
        forwarded += source.count("attn_mask = attn_metadata.attn_mask")
        # 9 FIA call sites across AscendAttentionBackendImpl / AscendC8 impl.
        assert forwarded >= 9
