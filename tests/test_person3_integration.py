"""End-to-end tests for Person 3's optimized Transformer assembly."""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from bench_shape14_blockwise import (
    QueryTiledSelfAttention,
    SeparateQKVSDPAAttention,
    SeparateQKVTritonAttention,
)
from person1_triton_attention import PackedQKVSDPAAttention
from run_sweep import (
    OFFICIAL_CASES,
    Candidate,
    CandidateEvaluation,
    RunResult,
    build_command,
    failed_summary,
    parse_result,
    select_relative_winner,
    shape14_memory_summary,
)
from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
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

    def test_calibration_parses_failure_and_builds_strict_command(self) -> None:
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
            1,
        )
        self.assertEqual(command[command.index("--atol") + 1], "0.001")
        self.assertEqual(command[command.index("--rtol") + 1], "0.01")
        self.assertEqual(
            command[command.index("--accuracy-trials") + 1], "20"
        )
        self.assertIn("--causal", command)


if __name__ == "__main__":
    unittest.main()
