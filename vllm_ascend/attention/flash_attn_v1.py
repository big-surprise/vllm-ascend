#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Non-quantized flash_attn attention backend (CANN ops-transformer interface).

Implements the torch API declared in ops-transformer's
``attention/flash_attn/docs/torchapi_flash_attn.md``: a metadata operator
(``cann_ops_transformer.ops.flash_attn_metadata``) that pre-plans the
load-balancing task split, followed by the main operator
(``cann_ops_transformer.ops.flash_attn``). The ABI ships only for the
Ascend 950PR/950DT family (the A5 hardware profile), so the backend is
selected explicitly via ``--additional-config '{"enable_flash_attn": true}'`` and validated
in ``platform.get_attn_backend_cls``.

Scope: shared-KV GQA/MQA full attention (e.g. the full-attention layers of
Qwen3.5) with the standard paged KV cache. Query runs in the varlen TND
layout against the PA_BBND paged cache layout
``(num_blocks, block_size, num_kv_heads, head_size)`` -- exactly the layout
``AscendAttentionBackend`` allocates, so ``reshape_and_cache`` and the KV
cache shape are inherited unchanged. MLA, sparse attention, sliding window,
alibi and quantized KV caches are out of scope; the constructor rejects
them instead of silently falling back.
"""

from importlib import import_module
from importlib.util import find_spec

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_dcp, enable_pcp
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

# Mask modes from the operator doc: 0 computes the full score matrix
# (correct for single-row queries, where the KV range is bounded by
# seqused_kv); 3 applies the causal mask.
FLASH_ATTN_MASK_MODE_NO_MASK = 0
FLASH_ATTN_MASK_MODE_CAUSAL = 3

FLASH_ATTN_LAYOUT_TND = "TND"
FLASH_ATTN_LAYOUT_PA_BBND = "PA_BBND"

# The operator doc restricts D to 64/72/128/256 for the shared-KV
# non-quantized interface (QK and V head dims must match here).
FLASH_ATTN_SUPPORTED_HEAD_DIMS = (64, 72, 128, 256)


def _copy_flash_lengths(
    query_starts: torch.Tensor,
    kv_lengths: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
) -> None:
    """Make the two length arrays agree with the padded request count.

    The runner reserves one extra query boundary for a graph-padding request,
    but its KV length buffer has only max_num_reqs entries. That last request
    reads the harmless block-zero row and still needs a valid KV length.
    """
    cu_seqlens_q.copy_(query_starts[: num_reqs + 1].clamp(min=0, max=num_tokens).cummax(dim=0).values)
    seqused_kv.fill_(1)
    available = min(num_reqs, kv_lengths.numel())
    seqused_kv[:available].copy_(kv_lengths[:available].clamp(min=1))
    # A dummy query can span several padded rows. Give it at least as many
    # readable KV positions as query rows so causal alignment remains valid.
    torch.maximum(seqused_kv, cu_seqlens_q[1:] - cu_seqlens_q[:-1], out=seqused_kv)


def get_flash_attn_ops() -> tuple | None:
    """Find the two flash_attn APIs in the active CANN operator package.

    Mirrors ``attention.utils.get_flashmla_ops``: an absent package or API
    permits falling back to the FIA backend before KV cache allocation,
    while import failures inside an installed package remain errors.
    """
    if find_spec("cann_ops_transformer") is None:
        return None
    ops = import_module("cann_ops_transformer.ops")
    attention_op = getattr(ops, "flash_attn", None)
    metadata_op = getattr(ops, "flash_attn_metadata", None)
    if callable(attention_op) and callable(metadata_op):
        return attention_op, metadata_op
    return None


class AscendFlashAttnMetadataBuilder(AscendAttentionMetadataBuilder):
    """Start one Flash Attention plan on the metadata stream per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        # Stable addresses for graph consumers, keyed by the captured shape.
        self._async_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # The runner may append a dummy request to cover graph-padding rows.
        # Flash Attention's TND query and cu_seqlens_q must include those rows,
        # while reshape_and_cache still writes only num_actual_tokens real K/V.
        query_starts_cpu = common_attn_metadata.query_start_loc_cpu[: metadata.num_reqs + 1]
        metadata.flash_attn_num_tokens = int(query_starts_cpu[-1])
        if getattr(self.vllm_config.compilation_config, "cudagraph_mode", None) == CUDAGraphMode.FULL:
            # Replay can redistribute a fixed graph bucket among requests.
            # A plan captured with the dummy run's exact max would be too small
            # for a later step. The bucket's token count is a stable bound.
            metadata.flash_attn_max_seqlen_q = metadata.flash_attn_num_tokens
        else:
            metadata.flash_attn_max_seqlen_q = (
                int((query_starts_cpu[1:] - query_starts_cpu[:-1]).max()) if metadata.num_reqs else 0
            )
        if (
            not self._device_metadata_enabled
            or metadata.flash_attn_num_tokens <= 0
            or metadata.num_prefills > 0
        ):
            return metadata

        layer = self.vllm_config.compilation_config.static_forward_context[self.layer_names[0]]
        impl = layer.impl
        num_heads, num_kv_heads, head_size = impl.num_heads, impl.num_kv_heads, impl.head_size
        num_reqs = metadata.num_reqs
        num_tokens = metadata.flash_attn_num_tokens
        max_seqlen_q = metadata.flash_attn_max_seqlen_q
        mask_mode = FLASH_ATTN_MASK_MODE_NO_MASK if max_seqlen_q == 1 else FLASH_ATTN_MASK_MODE_CAUSAL
        shared_key = (
            id(metadata.query_start_loc_gpu),
            id(metadata.seq_lens_gpu),
            num_reqs,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_size,
            max_seqlen_q,
            mask_mode,
        )
        shared = common_attn_metadata.flash_attn_shared_plans
        buffers = shared.get(shared_key)
        if buffers is None:
            buffer_key = (
                metadata.query_start_loc_gpu.data_ptr(),
                metadata.seq_lens_gpu.data_ptr(),
                num_reqs,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_size,
                max_seqlen_q,
                mask_mode,
            )
            buffers = self._async_buffers.get(buffer_key)
            metadata_op = impl._flash_attn_metadata_op
            query_starts = metadata.query_start_loc_gpu
            kv_lengths = metadata.seq_lens_gpu

            def update_lengths(cu_seqlens_q: torch.Tensor, seqused_kv: torch.Tensor) -> None:
                _copy_flash_lengths(query_starts, kv_lengths, num_reqs, num_tokens, cu_seqlens_q, seqused_kv)

            def make_plan(cu_seqlens_q: torch.Tensor, seqused_kv: torch.Tensor) -> torch.Tensor:
                return metadata_op(
                    num_heads,
                    num_kv_heads,
                    head_size,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=None,
                    seqused_q=None,
                    seqused_kv=seqused_kv,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=-1,
                    mask_mode=mask_mode,
                    win_left=-1,
                    win_right=-1,
                    layout_q=FLASH_ATTN_LAYOUT_TND,
                    layout_kv=FLASH_ATTN_LAYOUT_PA_BBND,
                    layout_out=FLASH_ATTN_LAYOUT_TND,
                )

            if buffers is None:
                cu_seqlens_q = torch.empty((num_reqs + 1,), dtype=query_starts.dtype, device=query_starts.device)
                seqused_kv = torch.empty((num_reqs,), dtype=kv_lengths.dtype, device=kv_lengths.device)
                # The CANN wrapper owns the metadata output shape. Seed the
                # stable buffer once per captured shape, before graph capture.
                update_lengths(cu_seqlens_q, seqused_kv)
                plan = make_plan(cu_seqlens_q, seqused_kv)
                buffers = (cu_seqlens_q, seqused_kv, plan)
                self._async_buffers[buffer_key] = buffers

            cu_seqlens_q, seqused_kv, plan = buffers

            def prepare() -> None:
                update_lengths(cu_seqlens_q, seqused_kv)
                plan.copy_(make_plan(cu_seqlens_q, seqused_kv))

            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(DeviceMetadataStage.ATTENTION, prepare, id(plan)),
            )
            shared[shared_key] = buffers

        metadata.flash_attn_async_cu, metadata.flash_attn_async_kv, metadata.flash_attn_async_plan = buffers
        metadata.flash_attn_async_group_id = id(buffers[2])
        return metadata


class AscendFlashAttnBackend(AscendAttentionBackend):
    """Backend for the non-quantized CANN flash_attn dual-operator interface."""

    @staticmethod
    def get_impl_cls() -> type["AscendFlashAttnImpl"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("flash_attn backend does not support PCP/DCP yet.")
        return AscendFlashAttnImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendFlashAttnMetadataBuilder


class AscendFlashAttnImpl(AscendAttentionBackendImpl):
    """Full-attention impl driving flash_attn_metadata + flash_attn.

    One flash_attn call per step for the whole batch (decode and prefill
    alike); a step whose every query is one row long drops the mask
    (NO_MASK), which is equivalent there and picks a cheaper kernel. This
    mirrors the QFA design in attention_c8_mxfp.py.

    Graph capture: cu_seqlens_q / seqused_kv are derived on device from the
    runner's persistent buffers (query_start_loc_gpu / seq_lens_gpu), so
    every replay re-executes the derivation and sees the current step's
    lengths. The per-step metadata plan is shared by every full-attention
    layer: its inputs (head counts, lengths, mask mode, layouts) are not
    layer-specific, and the main operator treats ``metadata`` as read-only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sliding_window is not None:
            raise ValueError(
                "AscendFlashAttnImpl does not support sliding window attention. "
                "Please disable sliding window or use the default FIA backend."
            )
        if self.alibi_slopes is not None:
            raise ValueError(
                "AscendFlashAttnImpl does not support ALiBi. Please use the default FIA backend."
            )
        if self.head_size not in FLASH_ATTN_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"AscendFlashAttnImpl requires head_size in {FLASH_ATTN_SUPPORTED_HEAD_DIMS}, "
                f"got {self.head_size}."
            )
        if self.use_bnsd_kv_cache:
            raise ValueError(
                "AscendFlashAttnImpl requires the default PA_BBND paged KV cache layout. "
                "Please unset VLLM_KV_CACHE_LAYOUT (LBHNC/HND) or use the default FIA backend."
            )
        ops = get_flash_attn_ops()
        if ops is None:
            raise ValueError(
                "AscendFlashAttnImpl requires cann_ops_transformer with the "
                "flash_attn and flash_attn_metadata APIs (Ascend 950PR/950DT "
                "build). Please install a compatible cann_ops_transformer or "
                "disable additional_config.enable_flash_attn."
            )
        # Cache the resolved operators once. forward_impl and the metadata
        # derivation run inside the captured region, so re-resolving via
        # find_spec/import_module on every replay is both wasteful and can
        # re-trigger import side effects each step.
        self._flash_attn_op, self._flash_attn_metadata_op = ops

    def _flash_attn_step_cache(self, attn_metadata: AscendMetadata) -> dict:
        # The model runner can build separate AscendMetadata objects for
        # different KV-cache groups in the same forward. Keep the plan in the
        # forward context so those groups share one device-side preparation.
        if is_forward_context_available():
            return get_forward_context().additional_kwargs.setdefault("_ascend_flash_attn_step_cache", {})
        cache = getattr(attn_metadata, "flash_attn_metadata_cache", None)
        if cache is None:
            cache = {}
            attn_metadata.flash_attn_metadata_cache = cache
        return cache

    def _flash_attn_step_lengths(
        self, attn_metadata: AscendMetadata, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this step's (cu_seqlens_q, seqused_kv), derived once.

        Derived in-graph by the first full-attention layer; every other layer
        consumes the recorded tensors, and each replay re-executes that one
        derivation against the refreshed buffers.
        """
        cache = self._flash_attn_step_cache(attn_metadata)
        # Different cache groups may have different length sources. The
        # builder passes the same persistent tensor objects to groups that
        # can share this work. Object identity also works with FakeTensor
        # during tracing, unlike data_ptr(). The forward context bounds the
        # cache lifetime to this one step.
        lengths_key = (
            "lengths",
            id(attn_metadata.query_start_loc_gpu),
            id(attn_metadata.seq_lens_gpu),
            attn_metadata.num_reqs,
            num_tokens,
        )
        lengths = cache.get(lengths_key)
        if lengths is None:
            # Query boundaries may include one graph-padding request beyond
            # the KV length buffer. Keep all three operator inputs aligned
            # with the request count, including that dummy request.
            num_reqs = attn_metadata.num_reqs
            # Clamp sanitizes the tail within the valid prefix: unused
            # query_start_loc slots carry -1 (FIA padding convention) and may
            # hold stale entries from larger earlier steps; cummax restores
            # monotonicity, turning the tail into zero-length requests.
            # Missing/zero KV lengths use the dummy block-zero row; the
            # shared helper also covers multi-row dummy queries.
            cu_seqlens_q = torch.empty(
                (num_reqs + 1,), dtype=attn_metadata.query_start_loc_gpu.dtype,
                device=attn_metadata.query_start_loc_gpu.device,
            )
            seqused_kv = torch.empty(
                (num_reqs,), dtype=attn_metadata.seq_lens_gpu.dtype, device=attn_metadata.seq_lens_gpu.device
            )
            _copy_flash_lengths(
                attn_metadata.query_start_loc_gpu,
                attn_metadata.seq_lens_gpu,
                num_reqs,
                num_tokens,
                cu_seqlens_q,
                seqused_kv,
            )
            lengths = (cu_seqlens_q, seqused_kv)
            cache[lengths_key] = lengths
        return lengths

    def _flash_attn_int8_mask(self, attn_metadata: AscendMetadata) -> torch.Tensor | None:
        """flash_attn's attn_mask is INT8; the shared builder already hands
        out an int8 causal mask, so only convert other sources."""
        if attn_metadata.attn_mask is None:
            return None
        if attn_metadata.attn_mask.dtype == torch.int8:
            return attn_metadata.attn_mask
        return attn_metadata.attn_mask.to(torch.int8)

    def _get_flash_attn_metadata(
        self,
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        max_seqlen_q: int,
        mask_mode: int,
    ):
        """Return the flash_attn metadata plan, derived once a step.

        The plan's inputs are all non-layer-specific (head counts, lengths,
        mask mode, layouts), so one plan serves every full-attention layer of
        a step. Sharing is safe under graph capture for the same reason as
        _flash_attn_step_lengths: the deriving call runs inside the captured
        region, ahead of every flash_attn call that reads its output, and the
        main operator treats ``metadata`` as a read-only input.
        """
        cache = self._flash_attn_step_cache(attn_metadata)
        plan_key = (
            "plan",
            id(cu_seqlens_q),
            id(seqused_kv),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            max_seqlen_q,
            mask_mode,
        )
        metadata = cache.get(plan_key)
        if metadata is None:
            # TND + PA: pass cu_seqlens_q only; the KV side is addressed via
            # block_table + seqused_kv. batch_size must NOT be passed with a
            # TND layout_q.
            metadata = self._flash_attn_metadata_op(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=-1,
                mask_mode=mask_mode,
                win_left=-1,
                win_right=-1,
                layout_q=FLASH_ATTN_LAYOUT_TND,
                layout_kv=FLASH_ATTN_LAYOUT_PA_BBND,
                layout_out=FLASH_ATTN_LAYOUT_TND,
            )
            cache[plan_key] = metadata
        return metadata

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """One flash_attn call per step for the whole batch.

        reshape_and_cache (run by forward() ahead of this hook) has already
        written this step's K/V into the paged cache, so PrefillNoCache steps
        are included: the operator reads the cache through block_table and
        never sees the current-step K/V tensors directly.
        """
        if not attn_metadata.causal:
            raise NotImplementedError("flash_attn backend does not support non-causal attention yet.")

        num_tokens = getattr(attn_metadata, "flash_attn_num_tokens", attn_metadata.num_actual_tokens)
        if num_tokens <= 0:
            return output
        # T as the operator sees it includes any graph-padding query rows.
        # The TND constraint cu_seqlens_q[-1] == T, view/output row counts
        # and length clamp below all share this source.
        query = query[:num_tokens]

        qsl_gpu = attn_metadata.query_start_loc_gpu
        seq_lens_gpu = attn_metadata.seq_lens_gpu
        if qsl_gpu is None or seq_lens_gpu is None:
            raise RuntimeError(
                "flash_attn backend requires the GPU-side length sources "
                "(query_start_loc_gpu / seq_lens_gpu) on AscendMetadata."
            )
        # The builder includes padding requests and uses the graph bucket as
        # a replay-safe upper bound in FULL mode. The fallback serves tests
        # and metadata built outside this builder.
        max_seqlen_q = getattr(attn_metadata, "flash_attn_max_seqlen_q", None)
        if max_seqlen_q is None:
            max_seqlen_q = attn_metadata.max_query_len if attn_metadata.max_query_len is not None else 1

        # NO_MASK is safe only when the bound is one row per request.
        mask_mode = FLASH_ATTN_MASK_MODE_NO_MASK if max_seqlen_q == 1 else FLASH_ATTN_MASK_MODE_CAUSAL

        fa_metadata = getattr(attn_metadata, "flash_attn_async_plan", None)
        if fa_metadata is None:
            cu_seqlens_q, seqused_kv = self._flash_attn_step_lengths(attn_metadata, num_tokens)
            fa_metadata = self._get_flash_attn_metadata(
                attn_metadata,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                max_seqlen_q=max_seqlen_q,
                mask_mode=mask_mode,
            )
        else:
            wait_for_device_metadata(DeviceMetadataStage.ATTENTION, attn_metadata.flash_attn_async_group_id)
            cu_seqlens_q = attn_metadata.flash_attn_async_cu
            seqused_kv = attn_metadata.flash_attn_async_kv

        attention_op = self._flash_attn_op
        # block_tables is a persistent buffer with max_num_reqs rows; slice it
        # to the exact request count so its row count B matches cu_seqlens_q's
        # length B + 1 (the flash_attn op treats block_table's row count as B
        # and does not validate the alignment, so a stale tail would feed bogus
        # block ids for the padding rows).
        block_table = attn_metadata.block_tables[: attn_metadata.num_reqs]
        # The doc forbids attn_mask under NO_MASK and requires the fixed
        # 2048x2048 lower-triangular int8 mask under CAUSAL, so the two
        # travel together.
        result = attention_op(
            query,
            self.key_cache,
            self.value_cache,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=self.sinks,
            metadata=fa_metadata,
            softmax_scale=self.scale,
            mask_mode=mask_mode,
            attn_mask=(
                None if mask_mode == FLASH_ATTN_MASK_MODE_NO_MASK else self._flash_attn_int8_mask(attn_metadata)
            ),
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=-1,
            layout_q=FLASH_ATTN_LAYOUT_TND,
            layout_kv=FLASH_ATTN_LAYOUT_PA_BBND,
            layout_out=FLASH_ATTN_LAYOUT_TND,
            return_softmax_lse=False,
        )
        # return_softmax_lse=False yields an empty LSE tensor in the
        # cann_ops flavor; tolerate both tuple and single-tensor returns.
        attn_output = result[0] if isinstance(result, tuple) else result
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
