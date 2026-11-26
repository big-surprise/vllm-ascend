from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.base import TestBase
from vllm_ascend.attention import flash_attn_v1
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


class TestFlashAttnStepLengths(TestBase):
    """cu_seqlens_q / seqused_kv are derived once a step and shared by
    every full-attention layer; the sanitize clamp turns padded tails into
    zero-length requests."""

    def setUp(self):
        self.impl = object.__new__(AscendFlashAttnImpl)

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


class TestFlashAttnMetadataPlan(TestBase):
    """One metadata plan a step, capture included.

    The plan's inputs are head counts, head dim, the two length tensors,
    mask mode, window and layouts -- nothing layer-specific -- and the main
    operator declares ``metadata`` a read-only input. So the step's
    full-attention layers share one plan instead of each paying an
    AICore-to-AICPU round trip.
    """

    LAYERS_PER_STEP = 48  # Qwen3.5-27B full-attention layer count ballpark

    def setUp(self):
        self.impl = object.__new__(AscendFlashAttnImpl)
        self.impl.num_heads = 8
        self.impl.num_kv_heads = 2
        self.impl.head_size = 128

    def _plans(self, calls):
        """Issue one step's worth of plans; return (plans, operator calls)."""
        attn_metadata = SimpleNamespace()
        invocations = []

        def metadata_op(*args, **kwargs):
            invocations.append((args, kwargs))
            return MagicMock()

        defaults = {
            "cu_seqlens_q": torch.tensor([0, 1, 2], dtype=torch.int32),
            "seqused_kv": torch.tensor([10, 20], dtype=torch.int32),
            "max_seqlen_q": 1,
            "mask_mode": 0,
        }
        ops_module = MagicMock(flash_attn=MagicMock(), flash_attn_metadata=metadata_op)
        with (
            patch("vllm_ascend.attention.flash_attn_v1.find_spec", return_value=MagicMock()),
            patch.dict("sys.modules", {"cann_ops_transformer.ops": ops_module}),
        ):
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


class TestFlashAttnForward(TestBase):
    """forward_impl drives one flash_attn call over the paged cache."""

    NUM_HEADS = 8
    NUM_KV_HEADS = 2
    HEAD_SIZE = 128
    NUM_TOKENS = 5

    def setUp(self):
        self.impl = object.__new__(AscendFlashAttnImpl)
        self.impl.num_heads = self.NUM_HEADS
        self.impl.num_kv_heads = self.NUM_KV_HEADS
        self.impl.head_size = self.HEAD_SIZE
        self.impl.scale = 1.0 / (self.HEAD_SIZE**0.5)
        self.impl.sinks = None
        self.impl.key_cache = torch.zeros(4, 128, self.NUM_KV_HEADS, self.HEAD_SIZE)
        self.impl.value_cache = torch.zeros(4, 128, self.NUM_KV_HEADS, self.HEAD_SIZE)

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

        # The ops are resolved once in __init__ and cached on the impl, so the
        # test injects them there instead of faking the import machinery.
        self.impl._flash_attn_op = attention_op
        self.impl._flash_attn_metadata_op = MagicMock(return_value=MagicMock())
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


class TestFlashAttnArgsDump(TestBase):
    """VLLM_ASCEND_FLASH_ATTN_DEBUG dumps the operator operands once a step.

    The first full-attention layer of a step emits the record and marks the
    metadata; the remaining layers skip it, so a 48-layer model pays one
    device sync a step at most. Values are materialized only when the step
    runs eagerly -- a capture logs shapes instead.
    """

    NUM_HEADS = 8
    NUM_KV_HEADS = 2
    HEAD_SIZE = 128
    NUM_TOKENS = 5

    def setUp(self):
        self.impl = object.__new__(AscendFlashAttnImpl)
        self.impl.num_heads = self.NUM_HEADS
        self.impl.num_kv_heads = self.NUM_KV_HEADS
        self.impl.head_size = self.HEAD_SIZE
        self.impl.scale = 1.0 / (self.HEAD_SIZE**0.5)
        self.impl.sinks = None
        self.impl.key_cache = torch.zeros(4, 128, self.NUM_KV_HEADS, self.HEAD_SIZE)
        self.impl.value_cache = torch.zeros(4, 128, self.NUM_KV_HEADS, self.HEAD_SIZE)
        self.impl._flash_attn_op = MagicMock(
            return_value=(torch.zeros(self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_SIZE), torch.empty(1))
        )
        self.impl._flash_attn_metadata_op = MagicMock(return_value=MagicMock())

    def tearDown(self):
        # The record counter and the memoized limit are module-level state.
        flash_attn_v1._flash_attn_debug_records = 0
        flash_attn_v1._flash_attn_debug_limit = None

    def _metadata(self, *, max_query_len=1, attn_mask=None):
        return SimpleNamespace(
            causal=True,
            num_actual_tokens=self.NUM_TOKENS,
            num_reqs=2,
            query_start_loc_gpu=torch.tensor([0, 3, 5], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([10, 20], dtype=torch.int32),
            max_query_len=max_query_len,
            block_tables=torch.zeros(2, 2, dtype=torch.int32),
            attn_mask=attn_mask,
        )

    def _run(self, attn_metadata, layers=1):
        """Drive `layers` decode layers over one step's metadata."""
        query = torch.zeros(self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_SIZE)
        output = torch.zeros_like(query)
        with patch.object(flash_attn_v1, "logger") as mock_logger:
            for _ in range(layers):
                self.impl.forward_impl(query, None, None, None, attn_metadata, output)
        return mock_logger

    @staticmethod
    def _record(mock_logger):
        """The single emitted record, flattened to text for assertions."""
        assert mock_logger.info.call_count == 1, mock_logger.info.call_args_list
        return " ".join(str(arg) for arg in mock_logger.info.call_args[0])

    def test_disabled_by_default(self):
        mock_logger = self._run(self._metadata())
        mock_logger.info.assert_not_called()

    def test_one_record_a_step_shared_by_every_layer(self):
        flash_attn_v1._flash_attn_debug_limit = 3  # enabled, 3 steps
        record = self._record(self._run(self._metadata(), layers=48))
        self.assertIn("[FLASH_ATTN_DEBUG]", record)
        self.assertIn("mode=eager", record)
        self.assertIn("T=5", record)
        self.assertIn("B=2", record)
        self.assertIn("max_seqlen_q=1", record)
        self.assertIn("max_seqlen_kv=-1", record)
        self.assertIn("mask_mode=0", record)
        # The two length tensors carry their values, the unset ones read None.
        self.assertIn("values=[0, 3, 5]", record)
        self.assertIn("values=[10, 20]", record)
        self.assertIn("cu_seqlens_kv=None", record)
        self.assertIn("seqused_q=None", record)

    def test_prefill_step_reports_the_causal_geometry(self):
        flash_attn_v1._flash_attn_debug_limit = 3
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8)
        record = self._record(self._run(self._metadata(max_query_len=3, attn_mask=mask)))
        self.assertIn("max_seqlen_q=3", record)
        self.assertIn("mask_mode=3", record)

    def test_capture_logs_shapes_only(self):
        # A device sync is illegal inside a graph capture, so values are
        # skipped there rather than faulting the capture.
        flash_attn_v1._flash_attn_debug_limit = 3
        with patch.object(flash_attn_v1, "_EXTRA_CTX", SimpleNamespace(capturing=True)):
            record = self._record(self._run(self._metadata()))
        self.assertIn("mode=capture", record)
        self.assertIn("values=skipped(capture)", record)
        self.assertIn("shape=(3,)", record)
        self.assertNotIn("[0, 3, 5]", record)

    def test_cap_stops_the_dump(self):
        # Each step brings fresh metadata, so only the cap limits the records.
        flash_attn_v1._flash_attn_debug_limit = 1
        query = torch.zeros(self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_SIZE)
        output = torch.zeros_like(query)
        with patch.object(flash_attn_v1, "logger") as mock_logger:
            for _ in range(3):
                self.impl.forward_impl(query, None, None, None, self._metadata(), output)
        self.assertEqual(mock_logger.info.call_count, 1)

    def test_negative_cap_never_stops(self):
        flash_attn_v1._flash_attn_debug_limit = -1
        query = torch.zeros(self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_SIZE)
        output = torch.zeros_like(query)
        with patch.object(flash_attn_v1, "logger") as mock_logger:
            for _ in range(3):
                self.impl.forward_impl(query, None, None, None, self._metadata(), output)
        self.assertEqual(mock_logger.info.call_count, 3)
