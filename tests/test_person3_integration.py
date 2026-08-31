"""End-to-end tests for Person 3's optimized Transformer assembly."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch_transformer_benchmark as benchmark_module

from bench_shape14_blockwise import (
    QueryTiledSelfAttention,
    SeparateQKVSDPAAttention,
    SeparateQKVTritonAttention,
)
from bench_shape14_streaming import Shape14MemoryEstimate, validate_memory_budget
from person3_dispatch import (
    AttentionDispatchKey,
    DispatchMeasurement,
    historical_t4_measurements,
    make_dispatch_key,
    select_attention_plan,
)
from run_sweep import (
    OFFICIAL_CASES,
    Candidate,
    CandidateEvaluation,
    RunResult,
    build_command,
    canonical_backend_label,
    failed_summary,
    parse_result,
    select_relative_winner,
    shape14_memory_summary,
    suite_summary_lines,
)
from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    PackedQKVSelfAttention,
    TransformerConfig,
    UserOptimizedTransformer,
    UserOptimizedTransformerBlock,
    copy_model_weights,
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
        no_padding = make_dispatch_key(
            batch_size=2,
            seq_len=7,
            d_model=32,
            num_heads=4,
            dtype=torch.float32,
            causal=True,
            valid_token_mask=torch.ones(2, 7, dtype=torch.bool),
            device=torch.device("cpu"),
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
            device=torch.device("cpu"),
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
            key, num_layers=2, measurements=(passing, failing)
        )
        self.assertEqual(plan.label, "packed-sdpa-suffix:1")
        self.assertEqual(
            select_attention_plan(key, num_layers=2).label, "native"
        )

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
            key, num_layers=2, manual_suffix_layers=1
        )
        self.assertEqual(plan.label, "packed-sdpa-suffix:1")
        self.assertIn("manual experiment", plan.reason)

    def test_historical_t4_measurements_are_exact_passing_entries(self) -> None:
        measurements = historical_t4_measurements()
        self.assertEqual(len(measurements), 5)
        self.assertEqual(
            {
                (
                    measurement.key.batch_size,
                    measurement.key.seq_len,
                    measurement.key.d_model,
                    measurement.key.num_heads,
                )
                for measurement in measurements
            },
            {
                (8, 512, 512, 8),
                (1, 128, 128, 4),
                (4, 128, 128, 4),
                (16, 128, 128, 4),
                (64, 32, 128, 4),
            },
        )
        for measurement in measurements:
            self.assertTrue(measurement.passes_gate)
            plan = select_attention_plan(
                measurement.key,
                num_layers=4,
                measurements=measurements,
            )
            self.assertEqual(plan.label, "packed-sdpa-suffix:1")

    def test_unvalidated_official_candidates_remain_native(self) -> None:
        measurements = historical_t4_measurements()
        for batch_size, seq_len, d_model in ((64, 128, 32),):
            key = AttentionDispatchKey(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=4,
                dtype="torch.float16",
                causal=True,
                padding="none",
                device_type="cuda",
                device_index=0,
                compute_capability=(7, 5),
                torch_version="2.11.0",
            )
            plan = select_attention_plan(
                key,
                num_layers=4,
                measurements=measurements,
            )
            self.assertEqual(plan.label, "native")

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

    def test_query_tiled_reference_matches_baseline_attention(self) -> None:
        torch.manual_seed(99)
        baseline = BaselineSelfAttention(32, 4).eval()
        tiled = QueryTiledSelfAttention(32, 4, query_block=3).eval()
        wider_tiled = QueryTiledSelfAttention(32, 4, query_block=5).eval()
        tiled.load_state_dict(baseline.state_dict())
        wider_tiled.load_state_dict(baseline.state_dict())
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
                    wider_actual = wider_tiled(x, mask, causal)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(expected, wider_actual))

    def test_separate_qkv_sdpa_preserves_baseline_parameters(self) -> None:
        torch.manual_seed(100)
        baseline = BaselineSelfAttention(32, 4).eval()
        sdpa = SeparateQKVSDPAAttention(32, 4).eval()
        sdpa.load_state_dict(baseline.state_dict())
        self.assertTrue(torch.equal(sdpa.q_proj.weight, baseline.q_proj.weight))
        self.assertTrue(torch.equal(sdpa.k_proj.bias, baseline.k_proj.bias))
        self.assertTrue(torch.equal(sdpa.v_proj.weight, baseline.v_proj.weight))
        self.assertTrue(torch.equal(sdpa.out_proj.bias, baseline.out_proj.bias))

        triton_attention = SeparateQKVTritonAttention(32, 4).eval()
        triton_attention.load_state_dict(baseline.state_dict())
        self.assertTrue(
            torch.equal(triton_attention.q_proj.weight, baseline.q_proj.weight)
        )

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

    def test_exact_official_dispatch_uses_only_final_packed_layer(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        optimized = UserOptimizedTransformer(config)

        self.assertTrue(optimized._packed_candidate)
        self.assertTrue(
            all(
                isinstance(layer.attention, BaselineSelfAttention)
                for layer in optimized.layers[:-1]
            )
        )
        self.assertIsInstance(
            optimized.layers[-1].attention, PackedQKVSelfAttention
        )

        nearby = UserOptimizedTransformer(
            TransformerConfig(2, 128, 128, 4, 128, 4, True)
        )
        self.assertFalse(nearby._packed_candidate)
        self.assertTrue(
            all(
                isinstance(layer.attention, BaselineSelfAttention)
                for layer in nearby.layers
            )
        )

    def test_packed_weight_transfer_uses_qkv_row_order(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        torch.manual_seed(102)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
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

    def test_packed_core_cpu_strict_comparison(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        torch.manual_seed(104)
        baseline = BaselineTransformer(config).eval()
        optimized = UserOptimizedTransformer(config).eval()
        copy_model_weights(baseline, optimized)
        x = torch.randn(1, 128, 128)
        mask = torch.ones(1, 128, dtype=torch.bool)

        with (
            mock.patch.object(
                optimized, "_runtime_supports_packed", return_value=True
            ),
            torch.inference_mode(),
        ):
            reference = baseline(x, mask)
            candidate = optimized(x, mask)
        assert_or_close(self, reference, candidate)

    def test_weight_transfer_preserves_native_parameters(self) -> None:
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
            torch.equal(optimized.final_norm.weight, baseline.final_norm.weight)
        )
        self.assertEqual(
            set(optimized.state_dict()),
            set(baseline.state_dict()),
        )

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

    def test_all_valid_mask_cache_is_identity_and_version_guarded(self) -> None:
        baseline, optimized = self.make_models()
        torch.manual_seed(107)
        x = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.bool)

        with torch.inference_mode():
            reference = baseline(x, mask)
            candidate = optimized(x, mask)
        self.assertTrue(torch.equal(reference, candidate))
        self.assertIs(optimized._cached_mask, mask)
        self.assertTrue(optimized._cached_mask_is_all_valid)

        cloned_mask = mask.clone()
        with torch.inference_mode():
            cloned_candidate = optimized(x, cloned_mask)
        self.assertTrue(torch.equal(reference, cloned_candidate))
        self.assertIs(optimized._cached_mask, cloned_mask)

        cloned_mask[1, -1] = False
        with torch.inference_mode():
            mutated_reference = baseline(x, cloned_mask)
            mutated_candidate = optimized(x, cloned_mask)
        self.assertTrue(torch.equal(mutated_reference, mutated_candidate))
        self.assertFalse(optimized._cached_mask_is_all_valid)
        self.assertTrue(bool((mutated_candidate[~cloned_mask] == 0).all().item()))

    def test_inference_tensor_masks_are_not_cached(self) -> None:
        _baseline, optimized = self.make_models()
        x = torch.randn(2, 7, 32)
        with torch.inference_mode():
            mask = torch.ones(2, 7, dtype=torch.bool)
            optimized(x, mask)
        self.assertIsNone(optimized._cached_mask)
        self.assertIsNone(optimized._cached_mask_version)

    def test_padded_mask_is_not_bypassed(self) -> None:
        baseline, optimized = self.make_models()
        x = torch.randn(2, 7, 32)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, False, False],
            ]
        )
        with mock.patch.object(
            optimized.layers[0].attention,
            "forward",
            wraps=optimized.layers[0].attention.forward,
        ) as attention:
            with torch.inference_mode():
                reference = baseline(x, mask)
                candidate = optimized(x, mask)
            self.assertIs(attention.call_args.args[1], mask)
        self.assertTrue(torch.equal(reference, candidate))
        self.assertFalse(optimized._cached_mask_is_all_valid)

    def test_invalid_mask_metadata_never_uses_bypass(self) -> None:
        _baseline, optimized = self.make_models()
        x = torch.randn(2, 7, 32)
        self.assertFalse(
            optimized._mask_is_all_valid(torch.ones(2, 6, dtype=torch.bool), x)
        )
        self.assertFalse(optimized._mask_is_all_valid(torch.ones(2, 7), x))

    def test_runtime_gate_rejects_cpu_and_training(self) -> None:
        config = TransformerConfig(1, 128, 128, 4, 128, 4, True)
        optimized = UserOptimizedTransformer(config).eval()
        x = torch.randn(1, 128, 128)
        self.assertFalse(optimized._runtime_supports_packed(x))
        optimized.train()
        self.assertFalse(optimized._runtime_supports_packed(x))

    def test_canonical_template_sections_have_not_drifted(self) -> None:
        source = Path(benchmark_module.__file__).read_text()
        self.assertNotIn("import person", source)
        self.assertNotIn("import triton", source)
        prefix = source[: source.index("class PackedQKVSelfAttention")]
        tail = source[source.index("def resolve_device") :].rstrip() + "\n"
        self.assertEqual(
            hashlib.sha256(prefix.encode()).hexdigest(),
            "3e25293daeb8826cdbbf64fece4a4ce51a634d093473f018f47e2530d52fc89c",
        )
        self.assertEqual(
            hashlib.sha256(tail.encode()).hexdigest(),
            "e8fd8365ade5d3ab1bafc70303d4fe5b7886a0de8258aff86aa803d188ea83d0",
        )

    def test_sweep_parser_reads_strict_result_and_timings(self) -> None:
        output = """
summary: PASS | max_abs=0.0001 | max_rel=0.02 | failed=0/1024
baseline : median=4.2000 ms | mean=4.2500 ms | p90=4.5000 ms | min=4.0000 ms
optimized: median=2.1000 ms | mean=2.1500 ms | p90=2.3000 ms | min=2.0000 ms
speedup  : 2.000x based on median latency
"""
        result = parse_result(output)
        self.assertTrue(result.correct)
        self.assertEqual(result.baseline_median_ms, 4.2)
        self.assertEqual(result.baseline_p90_ms, 4.5)
        self.assertEqual(result.optimized_median_ms, 2.1)
        self.assertEqual(result.optimized_p90_ms, 2.3)
        self.assertEqual(result.speedup, 2.0)

    def test_calibration_gate_requires_correct_repeatable_results(self) -> None:
        passing = RunResult(True, 4.0, 4.5, 3.0, 3.4, 4.0 / 3.0)
        accepted = CandidateEvaluation(Candidate(1, 0), (passing,) * 3)
        self.assertTrue(accepted.accepted)
        self.assertAlmostEqual(accepted.median_speedup, 4.0 / 3.0)

        slow = RunResult(True, 4.0, 4.5, 3.95, 4.4, 4.0 / 3.95)
        self.assertFalse(
            CandidateEvaluation(Candidate(0, 0), (slow,) * 3).accepted
        )
        bad_p90 = RunResult(True, 4.0, 4.5, 3.0, 4.6, 4.0 / 3.0)
        self.assertFalse(
            CandidateEvaluation(Candidate(1, 1), (bad_p90,) * 3).accepted
        )
        rejected = CandidateEvaluation(
            Candidate(2, 0), (), "strict correctness failed (1/1024)"
        )
        self.assertFalse(rejected.accepted)

        faster = RunResult(True, 4.0, 4.5, 2.8, 3.2, 4.0 / 2.8)
        improved = CandidateEvaluation(Candidate(1, 1), (faster,) * 3)
        self.assertTrue(improved.improves_on(accepted))
        self.assertFalse(accepted.improves_on(improved))

        mixed = CandidateEvaluation(
            Candidate(1, 1), (faster, passing, faster)
        )
        self.assertFalse(mixed.improves_on(accepted))

        self.assertEqual(
            select_relative_winner([accepted, improved]), improved
        )
        self.assertEqual(
            select_relative_winner([accepted, mixed]), accepted
        )
        self.assertIsNone(select_relative_winner([rejected]))

    def test_runner_builds_canonical_strict_command(self) -> None:
        output = (
            "summary: FAIL | max_abs=0.003 | max_rel=2 | "
            "failed=3/344064\n"
        )
        self.assertEqual(failed_summary(output), "3/344064")
        self.assertIsNone(failed_summary("summary: PASS | failed=0/344064\n"))

        command = build_command(
            OFFICIAL_CASES[1],
            "cuda",
            "float16",
            20,
            100,
            3,
            0,
            None,
        )
        self.assertEqual(command[command.index("--atol") + 1], "0.001")
        self.assertEqual(command[command.index("--rtol") + 1], "0.01")
        self.assertEqual(
            command[command.index("--accuracy-trials") + 1], "20"
        )
        self.assertIn("--causal", command)
        self.assertNotIn("--dispatch-mode", command)
        self.assertNotIn("--packed-sdpa-suffix-layers", command)
        self.assertNotIn("--fast-ffn-suffix-layers", command)

    def test_runner_reports_canonical_policy_and_suite_summary(self) -> None:
        with (
            mock.patch("run_sweep.torch.cuda.is_available", return_value=True),
            mock.patch("run_sweep.torch.cuda.current_device", return_value=0),
            mock.patch(
                 "run_sweep.torch.cuda.get_device_capability", return_value=(7, 5)
            ),
            mock.patch("run_sweep.torch.__version__", "2.11.0+cu128"),
        ):
            self.assertEqual(
                canonical_backend_label(
                    OFFICIAL_CASES[1], "cuda", "float16"
                ),
                "packed-sdpa-suffix:1",
            )
            self.assertEqual(
                canonical_backend_label(
                    OFFICIAL_CASES[0], "cuda", "float16"
                ),
                "native",
            )

        native = RunResult(True, 4.0, 4.1, 3.2, 3.3, 1.25, "native")
        packed = RunResult(
            True, 4.0, 4.1, 2.0, 2.1, 2.0, "packed-sdpa-suffix:1"
        )
        report = "\n".join(
            suite_summary_lines(
                [
                    (OFFICIAL_CASES[0], (native,) * 3),
                    (OFFICIAL_CASES[1], (packed,) * 3),
                ]
            )
        )
        self.assertIn("cases=2 | processes=6", report)
        self.assertIn("speedup range=1.250x-2.000x", report)
        self.assertIn("best=ID2 2.000x", report)
        self.assertIn("packed-sdpa-suffix:1 IDs=2", report)


if __name__ == "__main__":
    unittest.main()
