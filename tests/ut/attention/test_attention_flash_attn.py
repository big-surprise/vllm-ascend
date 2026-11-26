from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.flash_attn_v1 import (
    AscendFlashAttnBackend,
    AscendFlashAttnImpl,
    get_flash_attn_ops,
)


class TestFlashAttnOpsDiscovery(TestBase):
    """An absent package or API yields None (FIA fallback); a complete
    package yields the (flash_attn, flash_attn_metadata) pair."""

    def test_missing_package_returns_none(self):
        with patch("vllm_ascend.attention.flash_attn_v1.find_spec", return_value=None):
            self.assertIsNone(get_flash_attn_ops())

    def test_missing_apis_return_none(self):
        ops_module = MagicMock(spec=[])  # no flash_attn / flash_attn_metadata
        with (
            patch("vllm_ascend.attention.flash_attn_v1.find_spec", return_value=MagicMock()),
            patch.dict("sys.modules", {"cann_ops_transformer.ops": ops_module}),
        ):
            self.assertIsNone(get_flash_attn_ops())

    def test_complete_package_returns_the_pair(self):
        flash_attn = MagicMock()
        flash_attn_metadata = MagicMock()
        ops_module = MagicMock(flash_attn=flash_attn, flash_attn_metadata=flash_attn_metadata)
        with (
            patch("vllm_ascend.attention.flash_attn_v1.find_spec", return_value=MagicMock()),
            patch.dict("sys.modules", {"cann_ops_transformer.ops": ops_module}),
        ):
            self.assertEqual(get_flash_attn_ops(), (flash_attn, flash_attn_metadata))


class TestFlashAttnBackend(TestBase):
    def test_impl_and_builder_classes(self):
        with (
            patch("vllm_ascend.attention.flash_attn_v1.enable_pcp", return_value=False),
            patch("vllm_ascend.attention.flash_attn_v1.enable_dcp", return_value=False),
        ):
            self.assertIs(AscendFlashAttnBackend.get_impl_cls(), AscendFlashAttnImpl)

    def test_pcp_and_dcp_are_rejected(self):
        for pcp, dcp in ((True, False), (False, True)):
            with (
                self.subTest(pcp=pcp, dcp=dcp),
                patch("vllm_ascend.attention.flash_attn_v1.enable_pcp", return_value=pcp),
                patch("vllm_ascend.attention.flash_attn_v1.enable_dcp", return_value=dcp),
                pytest.raises(NotImplementedError, match="PCP/DCP"),
            ):
                AscendFlashAttnBackend.get_impl_cls()

    def test_kv_cache_layout_is_the_generic_paged_layout(self):
        # PA_BBND per-tensor view: (num_blocks, block_size, num_kv_heads, head_size).
        self.assertEqual(
            AscendFlashAttnBackend.get_kv_cache_shape(8, 128, 2, 128),
            (2, 8, 128, 2, 128),
        )
        self.assertEqual(AscendFlashAttnBackend.get_supported_kernel_block_sizes(), [128])


def _bare_impl(*, num_heads=8, num_kv_heads=2, head_size=128):
    """Build an impl without running __init__ (no vllm/torch_npu needed).

    P5 caches the resolved operators on the instance in __init__, so tests
    that bypass __init__ must install the two op attributes by hand.
    """
    impl = object.__new__(AscendFlashAttnImpl)
    impl.num_heads = num_heads
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_size
    impl.scale = 1.0 / (head_size**0.5)
    impl.sinks = None
    impl.key_cache = torch.zeros(4, 128, num_kv_heads, head_size)
    impl.value_cache = torch.zeros(4, 128, num_kv_heads, head_size)
    impl._flash_attn_op = MagicMock()
    impl._flash_attn_metadata_op = MagicMock()
    return impl


class TestFlashAttnStepLengths(TestBase):
    """cu_seqlens_q / seqused_kv are derived once a step and shared by
    every full-attention layer; the sanitize clamp turns padded tails into
    zero-length requests."""

    def setUp(self):
        self.impl = _bare_impl()

    def test_derivation_matches_the_tnd_contract(self):
        # num_reqs slices the persistent buffers down to the valid prefix, so
        # cu_seqlens_q has length num_reqs + 1 and seqused_kv length num_reqs.
        attn_metadata = SimpleNamespace(
            num_reqs=2,
            query_start_loc_gpu=torch.tensor([0, 3, 5, -1, -1], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20, 0, 0], dtype=torch.int32),
        )
        cu_seqlens_q, seqused_kv = self.impl._flash_attn_step_lengths(attn_metadata, 5)
        self.assertTrue(torch.equal(cu_seqlens_q, torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(seqused_kv, torch.tensor([10, 20], dtype=torch.int32)))

    def test_stale_tail_within_prefix_is_sanitized(self):
        # A stale/negative slot *inside* the valid prefix is clamped and
        # cummax-restored to monotonic; the padding beyond num_reqs is dropped.
        attn_metadata = SimpleNamespace(
            num_reqs=2,
            query_start_loc_gpu=torch.tensor([0, 3, 5, -1, -1], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 0], dtype=torch.int32),
        )
        cu_seqlens_q, seqused_kv = self.impl._flash_attn_step_lengths(attn_metadata, 5)
        self.assertTrue(torch.equal(cu_seqlens_q, torch.tensor([0, 3, 5], dtype=torch.int32)))
        self.assertTrue(torch.equal(seqused_kv, torch.tensor([10, 1], dtype=torch.int32)))

    def test_one_derivation_serves_every_layer(self):
        attn_metadata = SimpleNamespace(
            num_reqs=2,
            query_start_loc_gpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([7, 9], dtype=torch.int32),
        )
        first = self.impl._flash_attn_step_lengths(attn_metadata, 2)
        second = self.impl._flash_attn_step_lengths(attn_metadata, 2)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])


class TestFlashAttnStepScalars(TestBase):
    """max_seqlen_q / mask_mode are derived once and shared."""

    def setUp(self):
        self.impl = _bare_impl()

    def test_decode_falls_back_to_single_row_and_no_mask(self):
        scalars = self.impl._flash_attn_step_scalars(SimpleNamespace(max_query_len=None))
        self.assertEqual(scalars, (1, 0))

    def test_prefill_keeps_causal_mask(self):
        scalars = self.impl._flash_attn_step_scalars(SimpleNamespace(max_query_len=7))
        self.assertEqual(scalars, (7, 3))

    def test_scalars_are_cached_by_identity(self):
        md = SimpleNamespace(max_query_len=7)
        self.assertIs(self.impl._flash_attn_step_scalars(md), self.impl._flash_attn_step_scalars(md))


class TestFlashAttnMetadataPlan(TestBase):
    """One metadata plan a step, capture included.

    The plan's inputs are head counts, head dim, the two length tensors,
    mask mode, window and layouts -- nothing layer-specific -- and the main
    operator declares ``metadata`` a read-only input. So the step's
    full-attention layers share one plan instead of each paying an
    AICore-to-AICPU round trip.
    """

    LAYERS_PER_STEP = 16  # Qwen3.5-27B full-attention layer count

    def setUp(self):
        self.impl = _bare_impl()

    def _plans(self, calls):
        """Issue one step's worth of plans; return (plans, operator calls)."""
        attn_metadata = SimpleNamespace()
        invocations = []

        def metadata_op(*args, **kwargs):
            invocations.append((args, kwargs))
            return MagicMock()

        self.impl._flash_attn_metadata_op = metadata_op
        defaults = {
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "seqused_kv": torch.tensor([10, 20], dtype=torch.int32),
            "max_seqlen_q": 1,
            "mask_mode": 0,
        }
        plans = [self.impl._get_flash_attn_metadata(attn_metadata, **{**defaults, **call}) for call in calls]
        return plans, invocations

    def test_one_plan_serves_every_layer(self):
        plans, invocations = self._plans([{}] * self.LAYERS_PER_STEP)
        self.assertEqual(len(invocations), 1)
        for plan in plans[1:]:
            self.assertIs(plan, plans[0])

    def test_a_different_non_tensor_input_gets_its_own_plan(self):
        for override in ({"mask_mode": 3}, {"max_seqlen_q": 2}):
            with self.subTest(**override):
                plans, invocations = self._plans([{}, override])
                self.assertEqual(len(invocations), 2)
                self.assertIsNot(plans[0], plans[1])

    def test_the_step_mask_mode_reaches_the_operator(self):
        _, invocations = self._plans([{"mask_mode": 3}])
        self.assertEqual(invocations[0][1]["mask_mode"], 3)

    def test_layouts_and_head_geometry_reach_the_operator(self):
        _, invocations = self._plans([{}])
        args, kwargs = invocations[0]
        self.assertEqual(args, (8, 2, 128))
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_kv"], "PA_BBND")
        self.assertEqual(kwargs["layout_out"], "TND")
        self.assertEqual(kwargs["win_left"], -1)
        self.assertEqual(kwargs["win_right"], -1)


class TestFlashAttnCrossLayerReuse(TestBase):
    """The real object model: every layer owns a distinct impl instance, but
    the builder produces ONE AscendMetadata per step shared by all layers.

    Each per-step derivation -- lengths, scalars, int8 mask, block-table slice
    and the metadata plan -- must be computed once by the first full-attention
    layer and consumed by identity by all the others. This is what makes the
    FlashAttnMetadata op appear once per step instead of once per layer.
    """

    NUM_FULL_ATTN_LAYERS = 16  # Qwen3.5-27B: 64 layers, 16 full-attention

    def setUp(self):
        self.metadata_invocations = []
        self.attention_kwargs = []

        def metadata_op(*args, **kwargs):
            self.metadata_invocations.append((args, kwargs))
            return MagicMock(name="fa_plan")

        def attention_op(*args, **kwargs):
            self.attention_kwargs.append(kwargs)
            return torch.ones(5, 8, 128), torch.empty(1)

        self.metadata_op = metadata_op
        self.attention_op = attention_op

    def _new_layer_impl(self):
        impl = _bare_impl()
        impl._flash_attn_op = self.attention_op
        impl._flash_attn_metadata_op = self.metadata_op
        return impl

    def _shared_step_metadata(self, *, max_query_len=3, mask_is_int8=True):
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
        mask = mask.to(torch.int8) if mask_is_int8 else mask.to(torch.bool)
        return SimpleNamespace(
            causal=True,
            num_actual_tokens=5,
            num_reqs=2,
            query_start_loc_gpu=torch.tensor([0, 3, 5], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20], dtype=torch.int32),
            max_query_len=max_query_len,
            block_tables=torch.zeros(4, 1, dtype=torch.int32),
            attn_mask=mask,
            flash_attn_metadata_cache={},
        )

    def _run_all_layers(self, md):
        for _ in range(self.NUM_FULL_ATTN_LAYERS):
            impl = self._new_layer_impl()
            output = torch.zeros(5, 8, 128)
            impl.forward_impl(torch.randn(5, 8, 128), None, None, None, md, output)

    def test_all_layers_share_one_metadata_op_call(self):
        md = self._shared_step_metadata()
        self._run_all_layers(md)
        self.assertEqual(len(self.metadata_invocations), 1)

    def test_all_layers_consume_the_same_derived_tensors(self):
        md = self._shared_step_metadata()
        self._run_all_layers(md)
        self.assertEqual(len(self.attention_kwargs), self.NUM_FULL_ATTN_LAYERS)
        first = self.attention_kwargs[0]
        for key in ("metadata", "cu_seqlens_q", "seqused_kv", "block_table", "attn_mask"):
            for kwargs in self.attention_kwargs[1:]:
                self.assertIs(kwargs[key], first[key], f"{key} not shared across layers")

    def test_non_int8_mask_is_converted_once_and_shared(self):
        md = self._shared_step_metadata(mask_is_int8=False)
        self._run_all_layers(md)
        first_mask = self.attention_kwargs[0]["attn_mask"]
        self.assertEqual(first_mask.dtype, torch.int8)
        self.assertIsNot(first_mask, md.attn_mask)  # a conversion happened
        for kwargs in self.attention_kwargs[1:]:
            self.assertIs(kwargs["attn_mask"], first_mask)

    def test_block_table_is_sliced_once_to_num_reqs(self):
        md = self._shared_step_metadata()
        self._run_all_layers(md)
        block_table = self.attention_kwargs[0]["block_table"]
        self.assertEqual(block_table.shape[0], md.num_reqs)
        self.assertIs(block_table, md.flash_attn_metadata_cache["block_table"])

    def test_decode_step_drops_mask_everywhere(self):
        md = self._shared_step_metadata(max_query_len=None)
        self._run_all_layers(md)
        self.assertEqual(self.metadata_invocations[0][1]["mask_mode"], 0)
        for kwargs in self.attention_kwargs:
            self.assertIsNone(kwargs["attn_mask"])


class TestFlashAttnForward(TestBase):
    """forward_impl drives one flash_attn call over the paged cache."""

    NUM_HEADS = 8
    NUM_KV_HEADS = 2
    HEAD_SIZE = 128
    NUM_TOKENS = 5

    def setUp(self):
        self.impl = _bare_impl()

    def _forward(self, *, max_query_len, attn_mask=None, num_reqs=2, num_block_rows=2):
        attn_metadata = SimpleNamespace(
            causal=True,
            num_actual_tokens=self.NUM_TOKENS,
            num_reqs=num_reqs,
            query_start_loc_gpu=torch.tensor([0, 3, 5], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20], dtype=torch.int32),
            max_query_len=max_query_len,
            block_tables=torch.zeros(num_block_rows, 1, dtype=torch.int32),
            attn_mask=attn_mask,
        )
        query = torch.randn(8, self.NUM_HEADS, self.HEAD_SIZE)  # padded past num_actual_tokens
        output = torch.zeros(8, self.NUM_HEADS, self.HEAD_SIZE)
        seen = {}

        def attention_op(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return torch.ones(self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_SIZE), torch.empty(1)

        self.impl._flash_attn_op = attention_op
        result = self.impl.forward_impl(query, None, None, None, attn_metadata, output)
        return result, seen

    def test_single_row_queries_drop_the_mask(self):
        result, seen = self._forward(max_query_len=1)
        kwargs = seen["kwargs"]
        self.assertEqual(kwargs["mask_mode"], 0)
        self.assertIsNone(kwargs["attn_mask"])
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_kv"], "PA_BBND")
        self.assertEqual(kwargs["layout_out"], "TND")
        self.assertEqual(kwargs["softmax_scale"], self.impl.scale)
        self.assertIs(kwargs["sinks"], None)
        # Output carries the operator's rows for the actual tokens.
        self.assertTrue(torch.equal(result[: self.NUM_TOKENS], torch.ones_like(result[: self.NUM_TOKENS])))

    def test_prefill_step_keeps_the_causal_mask(self):
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8)
        _, seen = self._forward(max_query_len=3, attn_mask=mask)
        kwargs = seen["kwargs"]
        self.assertEqual(kwargs["mask_mode"], 3)
        self.assertIs(kwargs["attn_mask"], mask)
        self.assertEqual(kwargs["max_seqlen_q"], 3)

    def test_decode_step_falls_back_to_single_row(self):
        # max_query_len is None on decode; max_seqlen_q must become 1 (not the
        # batch token total) so the step still hits the NO_MASK fast path.
        _, seen = self._forward(max_query_len=None)
        self.assertEqual(seen["kwargs"]["max_seqlen_q"], 1)
        self.assertEqual(seen["kwargs"]["mask_mode"], 0)

    def test_block_table_is_sliced_to_num_reqs(self):
        # block_tables is a persistent max_num_reqs-row buffer; only num_reqs
        # rows may reach the op so B stays aligned with cu_seqlens_q (B + 1).
        _, seen = self._forward(max_query_len=1, num_reqs=2, num_block_rows=4)
        block_table = seen["kwargs"]["block_table"]
        self.assertEqual(block_table.shape[0], 2)
        cu_seqlens_q = seen["kwargs"]["cu_seqlens_q"]
        self.assertEqual(cu_seqlens_q.shape[0], 3)  # num_reqs + 1

    def test_non_int8_masks_are_converted(self):
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.bool)
        _, seen = self._forward(max_query_len=3, attn_mask=mask)
        self.assertEqual(seen["kwargs"]["attn_mask"].dtype, torch.int8)

    def test_attention_sinks_are_forwarded(self):
        self.impl.sinks = torch.zeros(self.NUM_HEADS)
        _, seen = self._forward(max_query_len=1)
        self.assertIs(seen["kwargs"]["sinks"], self.impl.sinks)

    def test_non_causal_is_rejected(self):
        attn_metadata = SimpleNamespace(causal=False, num_actual_tokens=self.NUM_TOKENS)
        with pytest.raises(NotImplementedError):
            self.impl.forward_impl(torch.randn(5, 8, 128), None, None, None, attn_metadata, torch.zeros(5, 8, 128))


class TestFlashAttnBackendValidation(TestBase):
    """platform._validate_flash_attn_backend gating: env opt-in plus the
    A5 (Ascend 950) capability plus a complete cann_ops_transformer build."""

    def _validate(self, key=(False, False)):
        from vllm_ascend.platform import _validate_flash_attn_backend

        return _validate_flash_attn_backend(key, MagicMock())

    def test_env_disabled_skips_validation(self):
        with patch("vllm_ascend.platform.envs") as mock_envs:
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = False
            self.assertFalse(self._validate())

    def test_mla_or_sparse_keys_are_rejected(self):
        with patch("vllm_ascend.platform.envs") as mock_envs:
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = True
            for key in ((True, False), (True, True)):
                with self.subTest(key=key), pytest.raises(ValueError, match="MLA and SFA"):
                    self._validate(key)

    def test_non_950_hardware_is_rejected(self):
        with (
            patch("vllm_ascend.platform.envs") as mock_envs,
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
        ):
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = True
            mock_profile.return_value.supports.return_value = False
            with pytest.raises(ValueError, match="950PR/950DT"):
                self._validate()

    def test_missing_package_is_rejected(self):
        with (
            patch("vllm_ascend.platform.envs") as mock_envs,
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
            patch("vllm_ascend.platform.util.find_spec", return_value=None),
        ):
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = True
            mock_profile.return_value.supports.return_value = True
            with pytest.raises(ValueError, match="cann_ops_transformer is not installed"):
                self._validate()

    def test_missing_apis_are_rejected(self):
        with (
            patch("vllm_ascend.platform.envs") as mock_envs,
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
            patch("vllm_ascend.platform.util.find_spec", return_value=MagicMock()),
            patch("vllm_ascend.platform.import_module", return_value=MagicMock(spec=[])),
        ):
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = True
            mock_profile.return_value.supports.return_value = True
            with pytest.raises(ValueError, match="flash_attn_metadata"):
                self._validate()

    def test_complete_setup_selects_the_backend(self):
        ops_module = MagicMock()
        ops_module.flash_attn = MagicMock()
        ops_module.flash_attn_metadata = MagicMock()
        with (
            patch("vllm_ascend.platform.envs") as mock_envs,
            patch("vllm_ascend.platform.get_current_hardware_profile") as mock_profile,
            patch("vllm_ascend.platform.util.find_spec", return_value=MagicMock()),
            patch("vllm_ascend.platform.import_module", return_value=ops_module),
        ):
            mock_envs.VLLM_ASCEND_ENABLE_FLASH_ATTN = True
            mock_profile.return_value.supports.return_value = True
            self.assertTrue(self._validate())
