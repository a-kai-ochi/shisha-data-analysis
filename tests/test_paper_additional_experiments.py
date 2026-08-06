#!/usr/bin/env python3
"""Tests for additional paper experiment helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_additional_experiments import (  # noqa: E402
    ThresholdSetting,
    apply_threshold_setting,
    apply_weight_setting,
    combine_candidate_pool,
    jaccard_at_k,
    lagos_supplementary_analysis,
    manual_validation_source,
    rank_correlations,
    sensitivity_analysis,
    threshold_analysis,
    tier_feature_summary,
)


class PaperAdditionalExperimentsTests(unittest.TestCase):
    def make_ranking_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pair_key": "A||B",
                    "flavor_a": "A",
                    "flavor_b": "B",
                    "rank_overall": 1,
                    "ranking_tier": "Tier1",
                    "normalized_support": 1.0,
                    "adjusted_lift": 0.8,
                    "normalized_centrality_mean": 0.6,
                    "normalized_smoothed_positive_ratio": 0.9,
                    "normalized_smoothed_negative_ratio": 0.1,
                    "normalized_smoothed_role_ratio": 0.7,
                    "pair_count": 5,
                    "document_cooccurrence_count": 5,
                    "same_sentence_evidence_document_count": 3,
                    "same_sentence_positive_count": 2,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 1,
                    "smoothed_positive_ratio": 0.4,
                    "smoothed_negative_ratio": 0.0,
                    "smoothed_role_ratio": 0.2,
                    "document_level_positive_count": 2,
                    "document_level_negative_count": 0,
                    "document_level_role_count": 1,
                    "document_level_positive_ratio": 0.4,
                    "document_level_negative_ratio": 0.0,
                    "document_level_role_ratio": 0.2,
                    "excluded_as_product_name_pair": False,
                    "is_parent_child_pair": False,
                    "support": 0.08,
                    "lift": 2.0,
                    "centrality_mean": 0.3,
                    "template_evidence_count": 0,
                    "evidence_rows_after_dedup": 3,
                    "evidence_rows_before_dedup": 4,
                    "evidence_duplicates_removed": 1,
                    "explicit_mix_count": 1,
                    "overall_score_v2": 0.50,
                },
                {
                    "pair_key": "A||C",
                    "flavor_a": "A",
                    "flavor_b": "C",
                    "rank_overall": 2,
                    "ranking_tier": "Tier1",
                    "normalized_support": 0.8,
                    "adjusted_lift": 0.7,
                    "normalized_centrality_mean": 0.5,
                    "normalized_smoothed_positive_ratio": 0.7,
                    "normalized_smoothed_negative_ratio": 0.0,
                    "normalized_smoothed_role_ratio": 0.5,
                    "pair_count": 4,
                    "document_cooccurrence_count": 4,
                    "same_sentence_evidence_document_count": 2,
                    "same_sentence_positive_count": 1,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 1,
                    "smoothed_positive_ratio": 0.3,
                    "smoothed_negative_ratio": 0.0,
                    "smoothed_role_ratio": 0.2,
                    "document_level_positive_count": 1,
                    "document_level_negative_count": 0,
                    "document_level_role_count": 1,
                    "document_level_positive_ratio": 0.25,
                    "document_level_negative_ratio": 0.0,
                    "document_level_role_ratio": 0.25,
                    "excluded_as_product_name_pair": False,
                    "is_parent_child_pair": False,
                    "support": 0.07,
                    "lift": 1.8,
                    "centrality_mean": 0.25,
                    "template_evidence_count": 0,
                    "evidence_rows_after_dedup": 2,
                    "evidence_rows_before_dedup": 2,
                    "evidence_duplicates_removed": 0,
                    "explicit_mix_count": 0,
                    "overall_score_v2": 0.40,
                },
                {
                    "pair_key": "B||C",
                    "flavor_a": "B",
                    "flavor_b": "C",
                    "rank_overall": 3,
                    "ranking_tier": "Tier2",
                    "normalized_support": 0.4,
                    "adjusted_lift": 0.3,
                    "normalized_centrality_mean": 0.3,
                    "normalized_smoothed_positive_ratio": 0.0,
                    "normalized_smoothed_negative_ratio": 0.0,
                    "normalized_smoothed_role_ratio": 0.0,
                    "pair_count": 2,
                    "document_cooccurrence_count": 2,
                    "same_sentence_evidence_document_count": 1,
                    "same_sentence_positive_count": 0,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 0,
                    "smoothed_positive_ratio": 0.1,
                    "smoothed_negative_ratio": 0.0,
                    "smoothed_role_ratio": 0.0,
                    "document_level_positive_count": 1,
                    "document_level_negative_count": 0,
                    "document_level_role_count": 0,
                    "document_level_positive_ratio": 0.5,
                    "document_level_negative_ratio": 0.0,
                    "document_level_role_ratio": 0.0,
                    "excluded_as_product_name_pair": False,
                    "is_parent_child_pair": False,
                    "support": 0.03,
                    "lift": 1.1,
                    "centrality_mean": 0.10,
                    "template_evidence_count": 1,
                    "evidence_rows_after_dedup": 1,
                    "evidence_rows_before_dedup": 2,
                    "evidence_duplicates_removed": 1,
                    "explicit_mix_count": 0,
                    "overall_score_v2": 0.10,
                },
            ]
        )

    def make_excluded_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pair_key": "D||E",
                    "flavor_a": "D",
                    "flavor_b": "E",
                    "rank_overall": 4,
                    "ranking_tier": "Excluded",
                    "normalized_support": 0.2,
                    "adjusted_lift": 0.2,
                    "normalized_centrality_mean": 0.2,
                    "normalized_smoothed_positive_ratio": 0.0,
                    "normalized_smoothed_negative_ratio": 0.0,
                    "normalized_smoothed_role_ratio": 0.0,
                    "pair_count": 2,
                    "document_cooccurrence_count": 2,
                    "same_sentence_evidence_document_count": 0,
                    "same_sentence_positive_count": 0,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 0,
                    "smoothed_positive_ratio": 0.0,
                    "smoothed_negative_ratio": 0.0,
                    "smoothed_role_ratio": 0.0,
                    "document_level_positive_count": 1,
                    "document_level_negative_count": 0,
                    "document_level_role_count": 0,
                    "document_level_positive_ratio": 0.5,
                    "document_level_negative_ratio": 0.0,
                    "document_level_role_ratio": 0.0,
                    "excluded_as_product_name_pair": False,
                    "is_parent_child_pair": False,
                    "support": 0.02,
                    "lift": 1.0,
                    "centrality_mean": 0.05,
                    "template_evidence_count": 0,
                    "evidence_rows_after_dedup": 1,
                    "evidence_rows_before_dedup": 1,
                    "evidence_duplicates_removed": 0,
                    "explicit_mix_count": 0,
                    "overall_score_v2": 0.05,
                }
            ]
        )

    def make_manual_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"pair_key": "A||B", "reviewer1_recommendation_validity": "valid"},
                {"pair_key": "A||C", "reviewer1_recommendation_validity": "partially_valid"},
                {"pair_key": "B||C", "reviewer1_recommendation_validity": "partially_valid"},
            ]
        )

    def make_lagos_unique_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"mix_pair_key": "A||B", "lagos_article_count": 2, "lagos_row_count": 2, "lagos_article_urls": "u1|u2", "source_text": "t1"},
                {"mix_pair_key": "B||C", "lagos_article_count": 1, "lagos_row_count": 1, "lagos_article_urls": "u3", "source_text": "t2"},
                {"mix_pair_key": "X||Y", "lagos_article_count": 2, "lagos_row_count": 3, "lagos_article_urls": "u4|u5", "source_text": "t3"},
            ]
        )

    def test_combine_candidate_pool(self) -> None:
        combined = combine_candidate_pool(self.make_ranking_df(), self.make_excluded_df())
        self.assertEqual(len(combined), 4)

    def test_manual_validation_source_prefers_reviewer1(self) -> None:
        self.assertEqual(manual_validation_source(self.make_manual_df()), "reviewer1_recommendation_validity")

    def test_jaccard_at_k(self) -> None:
        baseline = self.make_ranking_df()
        variant = baseline.copy()
        variant["variant_rank"] = [2, 1, 3]
        result = jaccard_at_k(baseline, variant, "rank_overall", "variant_rank", 2)
        self.assertEqual(result["common_pair_count"], 2)
        self.assertAlmostEqual(result["jaccard"], 1.0)

    def test_rank_correlations(self) -> None:
        baseline = self.make_ranking_df()
        variant = baseline.copy()
        variant["variant_rank"] = [2, 1, 3]
        result = rank_correlations(baseline, variant, "rank_overall", "variant_rank")
        self.assertEqual(result["common_candidate_count"], 3)
        self.assertIsNotNone(result["spearman_rank_correlation"])
        self.assertIsNotNone(result["kendall_rank_correlation"])

    def test_apply_weight_setting_keeps_candidate_count(self) -> None:
        ranking = self.make_ranking_df()
        variant = apply_weight_setting(
            ranking,
            {
                "normalized_support": 0.30,
                "adjusted_lift": 0.25,
                "normalized_centrality_mean": 0.15,
                "normalized_smoothed_positive_ratio": 0.15,
                "normalized_smoothed_role_ratio": 0.10,
                "normalized_smoothed_negative_ratio": -0.05,
            },
            "baseline",
        )
        self.assertEqual(len(variant), len(ranking))
        self.assertIn("variant_rank", variant.columns)

    def test_apply_threshold_setting_document_only_can_restore_no_same_sentence_pair(self) -> None:
        full_pool = combine_candidate_pool(self.make_ranking_df(), self.make_excluded_df())
        variant = apply_threshold_setting(
            full_pool,
            ThresholdSetting(
                name="document_only_pair3_doc2",
                context_mode="document_only",
                tier1_pair_min=3,
                tier1_doc_min=2,
            ),
        )
        self.assertIn("D||E", variant["pair_key"].tolist())

    def test_sensitivity_analysis_outputs_expected_sections(self) -> None:
        summary_df, corr_df, overlap_df, transition_df, manual_df, _report = sensitivity_analysis(
            self.make_ranking_df(),
            self.make_manual_df(),
        )
        self.assertIn("baseline", summary_df["setting"].tolist())
        self.assertTrue((overlap_df["k"].isin([10, 20, 50])).all())
        self.assertGreater(len(transition_df), 0)
        self.assertGreater(len(manual_df), 0)
        self.assertIn("spearman_rank_correlation", corr_df.columns)

    def test_lagos_supplementary_analysis_counts_tier_matches(self) -> None:
        ranking = self.make_ranking_df()
        lagos_unique = self.make_lagos_unique_df()
        lagos_common = pd.DataFrame(
            [
                {"existing_rank": 1, "mix_pair_key": "A||B", "tier": "Tier1", "LAGOS出現記事数": 2, "LAGOS出現行数": 2},
                {"existing_rank": 3, "mix_pair_key": "B||C", "tier": "Tier2", "LAGOS出現記事数": 1, "LAGOS出現行数": 1},
            ]
        )
        lagos_only = pd.DataFrame(
            [{"既存ランキングに存在しない理由の候補": "共起なし"}, {"既存ランキングに存在しない理由の候補": "共起なし"}]
        )
        agreement = pd.DataFrame(
            [
                {"k": 10, "common_pair_count": 2, "precision_at_k": 0.2, "recall_at_k": 0.3, "jaccard_at_k": 0.1},
                {"k": 20, "common_pair_count": 2, "precision_at_k": 0.1, "recall_at_k": 0.3, "jaccard_at_k": 0.09},
                {"k": 50, "common_pair_count": 2, "precision_at_k": 0.04, "recall_at_k": 0.3, "jaccard_at_k": 0.03},
            ]
        )
        tier_df, rank_df, _report, _draft = lagos_supplementary_analysis(
            ranking,
            lagos_unique,
            lagos_common,
            lagos_only,
            agreement,
        )
        self.assertEqual(int(tier_df.loc[tier_df["scope"] == "Tier1", "lagos_common_count"].iloc[0]), 1)
        self.assertEqual(int(tier_df.loc[tier_df["scope"] == "Tier2", "lagos_common_count"].iloc[0]), 1)
        self.assertGreater(len(rank_df), 0)

    def test_tier_feature_summary_outputs_distribution_rows(self) -> None:
        comp_df, dist_df, lagos_df, _report, _draft = tier_feature_summary(
            self.make_ranking_df(),
            self.make_lagos_unique_df(),
        )
        self.assertGreater(len(comp_df), 0)
        self.assertGreater(len(dist_df), 0)
        self.assertEqual(set(lagos_df["tier"]), {"Tier1", "Tier2"})

    def test_threshold_analysis_reproduces_baseline_setting_row(self) -> None:
        full_pool = combine_candidate_pool(self.make_ranking_df(), self.make_excluded_df())
        result_df, stability_df, manual_df, lagos_df, _report, _draft = threshold_analysis(
            full_pool,
            self.make_ranking_df(),
            self.make_manual_df(),
            self.make_lagos_unique_df(),
        )
        baseline_rows = result_df[
            (result_df["context_mode"] == "same_sentence_required")
            & (result_df["tier1_pair_min"] == 3)
            & (result_df["tier1_doc_min"] == 2)
        ]
        self.assertEqual(len(baseline_rows), 1)
        self.assertTrue((stability_df["k"].isin([10, 20, 50])).all())
        self.assertGreater(len(manual_df), 0)
        self.assertGreater(len(lagos_df), 0)


if __name__ == "__main__":
    unittest.main()
