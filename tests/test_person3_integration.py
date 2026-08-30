"""End-to-end tests for Person 3's optimized Transformer assembly."""

from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock

import torch

from bench_shape14_blockwise import QueryTiledSelfAttention, SeparateQKVSDPAAttention
from bench_shape14_streaming import (
    Shape14MemoryEstimate,
    validate_memory_budget,
)
from person1_triton_attention import PackedQKVSDPAAttention
from person3_dispatch import (
    AttentionDispatchKey,
    DispatchMeasurement,
    historical_t4_measurements,
    make_dispatch_key,
    select_attention_plan,
)
from run_sweep import OFFICIAL_CASES, parse_result, shape14_memory_summary
from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    UserOptimizedTransformerBlock,
    copy_model_weights,
    estimate_reference_working_set_bytes,
    validate_reference_memory_budget,
)


def assert_or_close(
    testcase: unittest.TestCase,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 0.001,
    rtol: float = 0.01,
) -> None:
    testcase.assertEqual(reference.shape, candidate.shape)
    testcase.assertEqual(reference.dtype, candidate.dtype)
    ref = reference.detach().float()
    opt = candidate.detach().float()
    error = (opt - ref).abs()
    passed = torch.isfinite(ref) & torch.isfinite(opt)
    passed &= (error <= atol) | (error <= rtol * ref.abs())
    if not bool(passed.all().item()):
        testcase.fail(
            f"output mismatch: max_abs={error.max().item():.6g}, "
            f"failed={int((~passed).sum().item())}/{passed.numel()}"
        )


class Person3IntegrationTests(unittest.TestCase):
    def test_dispatch_key_distinguishes_padding_and_runtime(self) -> None:
        cpu = torch.device("cpu")
        no_padding = make_dispatch_key(
            batch_size=2,
            seq_len=7,
            d_model=32,
            num_heads=4,
            dtype=torch.float32,
            causal=True,
            valid_token_mask=torch.ones(2, 7, dtype=torch.bool),
            device=cpu,
        )
        padded = make_dispatch_key(
            batch_size=2,
            seq_len=7,
            d_model=32,
            num_heads=4,
            dtype=torch.float32,
            causal=True,
            valid_token_mask=torch.tensor(
                [[True] * 7, [True, True, True, True, False, False, False]]
            ),
            device=cpu,
        )
        self.assertIsInstance(no_padding, AttentionDispatchKey)
        self.assertEqual(no_padding.padding, "none")
        self.assertEqual(padded.padding, "padded")
        self.assertNotEqual(no_padding, padded)

    def test_dispatch_uses_only_exact_measured_candidates(self) -> None:
        key = make_dispatch_key(
            batch_size=2,
            seq_len=7,
            d_model=32,
            num_heads=4,
            dtype=torch.float32,
            causal=False,
            valid_token_mask=None,
            device=torch.device("cpu"),
        )
        passing = DispatchMeasurement(
            key=key,
            suffix_layers=1,
            correctness_passed=True,
            process_speedups=(1.10, 1.08, 1.09),
            baseline_p90_ms=(10.0, 10.0, 10.0),
            optimized_p90_ms=(9.0, 9.5, 9.2),
        )
        failing = DispatchMeasurement(
            key=key,
            suffix_layers=2,
            correctness_passed=False,
            process_speedups=(1.50, 1.50, 1.50),
            baseline_p90_ms=(10.0, 10.0, 10.0),
            optimized_p90_ms=(1.0, 1.0, 1.0),
        )
        plan = select_attention_plan(
            key,
            num_layers=2,
            measurements=(passing, failing),
        )
        self.assertEqual(plan.label, "packed-sdpa-suffix:1")
        self.assertIn("exact measured candidate", plan.reason)

        unknown = select_attention_plan(
            key,
            num_layers=2,
            measurements=(),
        )
        self.assertEqual(unknown.label, "native")

    def test_dispatch_manual_override_is_explicit(self) -> None:
        key = make_dispatch_key(
            batch_size=1,
            seq_len=8,
            d_model=32,
            num_heads=4,
            dtype=torch.float32,
            causal=True,
            valid_token_mask=None,
            device=torch.device("cpu"),
        )
        plan = select_attention_plan(
            key,
            num_layers=2,
            manual_suffix_layers=1,
        )
        self.assertEqual(plan.label, "packed-sdpa-suffix:1")
        self.assertIn("manual experiment", plan.reason)

    def test_historical_t4_measurements_are_exact_passing_entries(self) -> None:
        measurements = historical_t4_measurements()
        self.assertEqual(len(measurements), 3)
        self.assertEqual(
            {
                (measurement.key.batch_size, measurement.key.d_model)
                for measurement in measurements
            },
            {(8, 512), (4, 128), (16, 128)},
        )
        for measurement in measurements:
            self.assertTrue(measurement.passes_gate)
            plan = select_attention_plan(
                measurement.key,
                num_layers=4,
                measurements=measurements,
            )
            self.assertEqual(plan.label, "packed-sdpa-suffix:1")

    def test_shape14_memory_guard_rejects_unsafe_block(self) -> None:
        unsafe = Shape14MemoryEstimate(
            batch_block=32,
            seq_len=100_000,
            d_model=1024,
            dtype=torch.float16,
            working_set_bytes=100,
            free_bytes=100,
        )
        with self.assertRaises(MemoryError):
            validate_memory_budget(unsafe, free_fraction=0.01)

    def test_full_batch_reference_memory_guard_rejects_unsafe_shape(self) -> None:
        config = TransformerConfig(10_000, 128, 128, 4, 128, 4, True)
        estimate = estimate_reference_working_set_bytes(config, torch.float16)
        with self.assertRaises(MemoryError):
            validate_reference_memory_budget(
                config,
                torch.float16,
                free_bytes=estimate - 1,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for streaming smoke")
    def test_shape14_streaming_small_smoke(self) -> None:
        from bench_shape14_streaming import run_streaming_shape14

        result = run_streaming_shape14(
            logical_batch=2,
            batch_block=1,
            seq_len=128,
            d_model=32,
            heads=4,
            ffn_dim=32,
            layers=1,
            dtype=torch.float16,
            device=torch.device("cuda"),
            warmup=0,
            repeats=1,
        )
        self.assertEqual(result["blocks"], 2)
        self.assertEqual(result["backend"], "packed-sdpa-suffix:1")
        self.assertGreater(result["peak_gpu_gib"], 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for streaming correctness")
    def test_shape14_streaming_reduced_exact_correctness(self) -> None:
        from bench_shape14_streaming import build_shape14_models

        torch.manual_seed(120)
        baseline, optimized = build_shape14_models(
            batch_block=1,
            seq_len=128,
            d_model=64,
            heads=1,
            ffn_dim=64,
            layers=1,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
        x = torch.randn(1, 128, 64, device="cuda", dtype=torch.float16)
        mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
        optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=0)
        with torch.inference_mode():
            reference = baseline(x, mask)
            candidate = optimized(x, mask)
        assert_or_close(self, reference, candidate, atol=0.001, rtol=0.01)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_SHAPE14_100K") == "1",
        "set RUN_SHAPE14_100K=1 with CUDA to run the 100k smoke test",
    )
    def test_shape14_100k_finite_no_oom_smoke(self) -> None:
        from bench_shape14_streaming import run_streaming_shape14

        result = run_streaming_shape14(
            logical_batch=1,
            batch_block=1,
            seq_len=100_000,
            d_model=64,
            heads=1,
            ffn_dim=64,
            layers=1,
            dtype=torch.float16,
            device=torch.device("cuda"),
            warmup=0,
            repeats=1,
        )
        self.assertEqual(result["blocks"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(result["median_ms"])))

    def test_query_tiled_reference_matches_baseline_attention(self) -> None:
        torch.manual_seed(99)
        baseline = BaselineSelfAttention(32, 4).eval()
        tiled = QueryTiledSelfAttention(32, 4, query_block=3).eval()
        tiled.load_state_dict(baseline.state_dict())
        x = torch.randn(2, 7, 32)
        masks = (
            None,
            torch.tensor(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, False, False],
                ]
            ),
        )

        for causal in (False, True):
            for mask in masks:
                with torch.inference_mode():
                    expected = baseline(x, mask, causal)
                    actual = tiled(x, mask, causal)
                self.assertTrue(torch.equal(expected, actual))

    def test_separate_qkv_sdpa_preserves_baseline_parameters(self) -> None:
        torch.manual_seed(100)
        baseline = BaselineSelfAttention(32, 4).eval()
        sdpa = SeparateQKVSDPAAttention(32, 4).eval()
        sdpa.load_state_dict(baseline.state_dict())
        self.assertTrue(torch.equal(sdpa.q_proj.weight, baseline.q_proj.weight))
        self.assertTrue(torch.equal(sdpa.k_proj.bias, baseline.k_proj.bias))
        self.assertTrue(torch.equal(sdpa.v_proj.weight, baseline.v_proj.weight))
        self.assertTrue(torch.equal(sdpa.out_proj.bias, baseline.out_proj.bias))

    def test_official_shape_table_matches_appendix(self) -> None:
        self.assertEqual(len(OFFICIAL_CASES), 14)
        self.assertEqual(
            (
                OFFICIAL_CASES[0].case_id,
                OFFICIAL_CASES[0].batch,
                OFFICIAL_CASES[0].d_model,
                OFFICIAL_CASES[0].heads,
                OFFICIAL_CASES[0].seq_len,
                OFFICIAL_CASES[0].layers,
                OFFICIAL_CASES[0].ffn_dim,
                OFFICIAL_CASES[0].causal,
            ),
            (1, 64, 128, 4, 128, 4, 128, True),
        )
        self.assertEqual(
            (
                OFFICIAL_CASES[-1].case_id,
                OFFICIAL_CASES[-1].batch,
                OFFICIAL_CASES[-1].d_model,
                OFFICIAL_CASES[-1].heads,
                OFFICIAL_CASES[-1].seq_len,
                OFFICIAL_CASES[-1].layers,
                OFFICIAL_CASES[-1].ffn_dim,
                OFFICIAL_CASES[-1].causal,
            ),
            (14, 32, 1024, 16, 100000, 2, 1024, True),
        )
        memory = shape14_memory_summary(OFFICIAL_CASES[-1], "float16")
        self.assertIn("input=6.10 GiB", memory)
        self.assertIn("input+output=12.21 GiB", memory)
        self.assertIn("explicit scores=9536.74 GiB", memory)

    @staticmethod
    def make_models(
        causal: bool = False,
    ) -> tuple[BaselineTransformer, UserOptimizedTransformer]:
        config = TransformerConfig(2, 7, 32, 4, 64, 2, causal)
        torch.manual_seed(101)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)
        return baseline, optimized

    def test_public_forward_signature_and_integrated_layers(self) -> None:
        baseline, optimized = self.make_models()
        self.assertEqual(
            inspect.signature(baseline.forward),
            inspect.signature(optimized.forward),
        )
        self.assertTrue(
            all(
                type(layer) is UserOptimizedTransformerBlock
                for layer in optimized.layers
            )
        )
        self.assertTrue(
            all(
                isinstance(layer.attention, BaselineSelfAttention)
                for layer in optimized.layers
            )
        )
        self.assertEqual(optimized.attention_backend, "baseline")

    def test_packed_sdpa_suffix_uses_only_final_layers(self) -> None:
        config = TransformerConfig(2, 7, 32, 4, 64, 3, False)
        optimized = UserOptimizedTransformer(
            config, packed_sdpa_suffix_layers=1
        )

        self.assertTrue(
            all(
                isinstance(layer.attention, BaselineSelfAttention)
                for layer in optimized.layers[:-1]
            )
        )
        self.assertIsInstance(
            optimized.layers[-1].attention, PackedQKVSDPAAttention
        )
        self.assertEqual(optimized.attention_backend, "packed-sdpa-suffix:1")

        with self.assertRaisesRegex(ValueError, "between 0 and num_layers"):
            UserOptimizedTransformer(config, packed_sdpa_suffix_layers=4)

    def test_packed_sdpa_suffix_weight_transfer_uses_qkv_row_order(self) -> None:
        config = TransformerConfig(2, 7, 32, 4, 64, 2, False)
        torch.manual_seed(102)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(
            config, packed_sdpa_suffix_layers=1
        ).eval()
        copy_model_weights(baseline, optimized)

        source = baseline.layers[-1].attention
        packed = optimized.layers[-1].attention
        expected_weight = torch.cat(
            [source.q_proj.weight, source.k_proj.weight, source.v_proj.weight]
        )
        expected_bias = torch.cat(
            [source.q_proj.bias, source.k_proj.bias, source.v_proj.bias]
        )
        self.assertTrue(torch.equal(packed.qkv_proj.weight, expected_weight))
        self.assertTrue(torch.equal(packed.qkv_proj.bias, expected_bias))
        self.assertTrue(
            torch.equal(packed.out_proj.weight, source.out_proj.weight)
        )

    def test_packed_sdpa_suffix_cpu_end_to_end(self) -> None:
        torch.manual_seed(104)
        x = torch.randn(2, 7, 32)
        masks = (
            None,
            torch.ones(2, 7, dtype=torch.bool),
            torch.tensor(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, False, False],
                ]
            ),
        )

        for causal in (False, True):
            config = TransformerConfig(2, 7, 32, 4, 64, 2, causal)
            baseline = BaselineTransformer(config).eval()
            optimized = UserOptimizedTransformer(
                config, packed_sdpa_suffix_layers=1
            ).eval()
            copy_model_weights(baseline, optimized)
            for mask in masks:
                with torch.inference_mode():
                    reference = baseline(x, mask)
                    candidate = optimized(x, mask)
                assert_or_close(self, reference, candidate)
                if mask is not None:
                    self.assertTrue(bool((candidate[~mask] == 0).all().item()))

    def test_weight_transfer_preserves_attention_and_refreshes_ffn_state(self) -> None:
        baseline, optimized = self.make_models()

        for source, target in zip(baseline.layers, optimized.layers):
            self.assertTrue(
                torch.equal(
                    target.attention.q_proj.weight,
                    source.attention.q_proj.weight,
                )
            )
            self.assertTrue(
                torch.equal(
                    target.attention.out_proj.weight,
                    source.attention.out_proj.weight,
                )
            )
            self.assertTrue(torch.equal(target.norm1.weight, source.norm1.weight))
            self.assertTrue(torch.equal(target.norm2.bias, source.norm2.bias))
            self.assertTrue(torch.equal(target.ffn_in.weight, source.ffn_in.weight))
            self.assertTrue(torch.equal(target.ffn_out.bias, source.ffn_out.bias))
            self.assertTrue(
                torch.equal(
                    target._ffn_out_weight_nt,
                    target.ffn_out.weight.detach().t().contiguous(),
                )
            )

        self.assertTrue(
            torch.equal(optimized.final_norm.weight, baseline.final_norm.weight)
        )
        optimized_keys = set(optimized.state_dict())
        self.assertIn("layers.0.attention.q_proj.weight", optimized_keys)
        self.assertNotIn("layers.0.attention.qkv_proj.weight", optimized_keys)

    def test_weight_transfer_rejects_mismatched_configuration(self) -> None:
        baseline, _optimized = self.make_models()
        mismatched = UserOptimizedTransformer(
            TransformerConfig(1, 8, 32, 4, 64, 2, False)
        )
        with self.assertRaisesRegex(ValueError, "configurations must match"):
            copy_model_weights(baseline, mismatched)

    def test_cpu_end_to_end_masks_and_causality(self) -> None:
        masks = (
            None,
            torch.ones(2, 7, dtype=torch.bool),
            torch.tensor(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, False, False],
                ]
            ),
        )
        torch.manual_seed(103)
        x = torch.randn(2, 7, 32)

        for causal in (False, True):
            baseline, optimized = self.make_models(causal)
            for mask in masks:
                with torch.inference_mode():
                    reference = baseline(x, mask)
                    candidate = optimized(x, mask)
                    repeated = optimized(x, mask)
                assert_or_close(self, reference, candidate)
                self.assertTrue(torch.equal(candidate, repeated))
                self.assertNotEqual(candidate.data_ptr(), repeated.data_ptr())
                if mask is not None:
                    self.assertTrue(bool((candidate[~mask] == 0).all().item()))

    def test_model_preparation_counts_layers_and_cpu_falls_back(self) -> None:
        _baseline, optimized = self.make_models()
        x = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.bool)

        self.assertEqual(optimized.prepare_for_inference(x, mask), 0)
        self.assertTrue(
            all(
                "unsupported" in (layer.fast_ffn_error or "")
                for layer in optimized.layers
            )
        )

        with mock.patch.object(
            UserOptimizedTransformerBlock,
            "prepare_fast_ffn",
            return_value=True,
        ):
            self.assertEqual(
                optimized.prepare_for_inference(x, mask),
                len(optimized.layers),
            )

        def enable_fast_ffn(layer, _x, _mask):
            layer._fast_ffn_enabled = True
            return True

        with mock.patch.object(
            UserOptimizedTransformerBlock,
            "prepare_fast_ffn",
            autospec=True,
            side_effect=enable_fast_ffn,
        ) as prepare:
            self.assertEqual(
                optimized.prepare_for_inference(
                    x, mask, fast_ffn_suffix_layers=1
                ),
                1,
            )
            self.assertEqual(prepare.call_count, 1)
        self.assertFalse(optimized.layers[0]._fast_ffn_enabled)
        self.assertTrue(optimized.layers[-1]._fast_ffn_enabled)

        self.assertEqual(
            optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=0),
            0,
        )
        self.assertTrue(
            all(not layer._fast_ffn_enabled for layer in optimized.layers)
        )
        with self.assertRaisesRegex(ValueError, "between 0 and num_layers"):
            optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=3)

    def test_all_valid_mask_bypass_is_identity_and_version_guarded(self) -> None:
        baseline, optimized = self.make_models()
        torch.manual_seed(107)
        x = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.bool)

        self.assertEqual(
            optimized.prepare_for_inference(
                x, mask, fast_ffn_suffix_layers=0
            ),
            0,
        )
        self.assertEqual(optimized.mask_dispatch, "all-valid-bypass")
        with mock.patch.object(
            optimized.layers[0].attention,
            "forward",
            wraps=optimized.layers[0].attention.forward,
        ) as attention:
            with torch.inference_mode():
                reference = baseline(x, mask)
                candidate = optimized(x, mask)
            self.assertIsNone(attention.call_args.args[1])
        self.assertTrue(torch.equal(reference, candidate))

        cloned_mask = mask.clone()
        with mock.patch.object(
            optimized.layers[0].attention,
            "forward",
            wraps=optimized.layers[0].attention.forward,
        ) as attention:
            with torch.inference_mode():
                cloned_candidate = optimized(x, cloned_mask)
            self.assertIs(attention.call_args.args[1], cloned_mask)
        self.assertTrue(torch.equal(reference, cloned_candidate))

        mask[1, -1] = False
        self.assertEqual(optimized.mask_dispatch, "masked")
        with torch.inference_mode():
            mutated_reference = baseline(x, mask)
            mutated_candidate = optimized(x, mask)
        self.assertTrue(torch.equal(mutated_reference, mutated_candidate))
        self.assertTrue(bool((mutated_candidate[~mask] == 0).all().item()))

    def test_padded_mask_is_not_bypassed(self) -> None:
        _baseline, optimized = self.make_models()
        x = torch.randn(2, 7, 32)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, False, False],
            ]
        )
        optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=0)
        self.assertEqual(optimized.mask_dispatch, "masked")
        with mock.patch.object(
            optimized.layers[0].attention,
            "forward",
            wraps=optimized.layers[0].attention.forward,
        ) as attention:
            with torch.inference_mode():
                optimized(x, mask)
            self.assertIs(attention.call_args.args[1], mask)

    def test_sweep_parser_reads_strict_result_and_timings(self) -> None:
        output = """
attention_backend=packed-sdpa-suffix:1
summary: PASS | max_abs=0.0001 | max_rel=0.02 | failed=0/1024
baseline : median=4.2000 ms | mean=4.2500 ms | p90=4.5000 ms | min=4.0000 ms
optimized: median=2.1000 ms | mean=2.1500 ms | p90=2.3000 ms | min=2.0000 ms
speedup  : 2.000x based on median latency
"""
        result = parse_result(output)
        self.assertTrue(result.correct)
        self.assertEqual(result.backend, "packed-sdpa-suffix:1")
        self.assertEqual(result.baseline_median_ms, 4.2)
        self.assertEqual(result.baseline_p90_ms, 4.5)
        self.assertEqual(result.optimized_median_ms, 2.1)
        self.assertEqual(result.optimized_p90_ms, 2.3)
        self.assertEqual(result.speedup, 2.0)


if __name__ == "__main__":
    unittest.main()
