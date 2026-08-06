#!/usr/bin/env python3
"""Tests for Shisha LAGOS external validation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from shisha_lagos_external_validation import (  # noqa: E402
    BASELINE_TOP_KS,
    FlavorDictionary,
    build_baseline_records,
    build_common_pairs_with_existing,
    build_dictionary_candidate_audit,
    build_dictionary_candidate_manual_review,
    build_dictionary_update_impact_simulation,
    build_existing_topk_not_in_lagos,
    build_extraction_summary,
    build_lagos_only_pairs,
    build_pair_repetition_audit,
    build_topk_pair_audit,
    build_unique_pairs,
    compute_at_k_metrics,
    extract_mix_pairs,
    ordered_pair_key,
)


class ShishaLagosExternalValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.flavor_dict = FlavorDictionary(
            canonical_flavors=[
                "レモン",
                "ライム",
                "ミント",
                "バニラ",
                "グレープ",
                "グレープミント",
                "メロン",
            ],
            pattern_to_canonical={
                "グレープミント": "グレープミント",
                "グレープ": "グレープ",
                "レモン": "レモン",
                "ライム": "ライム",
                "ミント": "ミント",
                "バニラ": "バニラ",
                "メロン": "メロン",
            },
            sorted_patterns=["グレープミント", "グレープ", "レモン", "ライム", "ミント", "バニラ", "メロン"],
        )

    def make_articles(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        records = []
        for idx, row in enumerate(rows, start=1):
            records.append(
                {
                    "article_id": row.get("article_id", f"A{idx}"),
                    "article_url": row.get("article_url", f"https://example.com/{idx}"),
                    "article_title": row.get("article_title", f"Title {idx}"),
                    "brand": row.get("brand", "Al Fakher"),
                    "target_flavor": row.get("target_flavor", ""),
                    "recommended_mix_heading": row.get("recommended_mix_heading", "おすすめミックス"),
                }
            )
        return pd.DataFrame(records)

    def make_tables(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def make_ranking(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "rank_overall": 1,
                    "flavor_a": "ミント",
                    "flavor_b": "レモン",
                    "pair_key": "ミント||レモン",
                    "pair_count": 6,
                    "support": 0.07,
                    "lift": 1.2,
                    "adjusted_lift": 0.03,
                    "centrality_mean": 0.28,
                    "smoothed_positive_ratio": 0.05,
                    "smoothed_negative_ratio": 0.00,
                    "smoothed_role_ratio": 0.01,
                    "overall_score_v2": 0.42,
                    "ranking_tier": "Tier1",
                },
                {
                    "rank_overall": 2,
                    "flavor_a": "バニラ",
                    "flavor_b": "ミント",
                    "pair_key": "バニラ||ミント",
                    "pair_count": 2,
                    "support": 0.02,
                    "lift": 1.4,
                    "adjusted_lift": 0.01,
                    "centrality_mean": 0.12,
                    "smoothed_positive_ratio": 0.02,
                    "smoothed_negative_ratio": 0.00,
                    "smoothed_role_ratio": 0.00,
                    "overall_score_v2": 0.19,
                    "ranking_tier": "Tier2",
                },
                {
                    "rank_overall": 3,
                    "flavor_a": "グレープ",
                    "flavor_b": "レモン",
                    "pair_key": "グレープ||レモン",
                    "pair_count": 3,
                    "support": 0.03,
                    "lift": 1.1,
                    "adjusted_lift": 0.01,
                    "centrality_mean": 0.10,
                    "smoothed_positive_ratio": 0.00,
                    "smoothed_negative_ratio": 0.00,
                    "smoothed_role_ratio": 0.00,
                    "overall_score_v2": 0.15,
                    "ranking_tier": "Tier2",
                },
                {
                    "rank_overall": 10,
                    "flavor_a": "メロン",
                    "flavor_b": "レモン",
                    "pair_key": "メロン||レモン",
                    "pair_count": 2,
                    "support": 0.01,
                    "lift": 1.1,
                    "adjusted_lift": 0.01,
                    "centrality_mean": 0.08,
                    "smoothed_positive_ratio": 0.01,
                    "smoothed_negative_ratio": 0.00,
                    "smoothed_role_ratio": 0.00,
                    "overall_score_v2": 0.10,
                    "ranking_tier": "Tier2",
                },
            ]
        )

    def test_extracts_pairs_from_recommended_table_only(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント", "article_title": "ミント記事"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 1, "row_index": 1, "cell_index": 1, "section_heading": "重さ", "cell_text": "とても重い", "is_recommended_mix_section": False},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 1, "row_index": 1, "cell_index": 2, "section_heading": "重さ", "cell_text": "レモン/ライム", "is_recommended_mix_section": False},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 1, "cell_index": 2, "section_heading": "おすすめミックス", "cell_text": "スイーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン/ライム", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 2, "cell_index": 2, "section_heading": "おすすめミックス", "cell_text": "バニラ", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        valid = extracted[extracted["is_valid_pair"]]
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid["recommended_flavor_normalized"].tolist(), ["レモン", "ライム", "バニラ"])
        self.assertTrue((valid["table_index"] == 2).all())

    def test_multiple_candidates_and_unregistered_candidate_are_recorded(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン/パイン", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        self.assertEqual(len(extracted), 2)
        unresolved = extracted[extracted["extraction_status"] == "unresolved"].iloc[0]
        self.assertEqual(unresolved["recommended_flavor_raw"], "パイン")
        self.assertEqual(unresolved["exclusion_reason"], "recommended_flavor_unregistered")

    def test_self_pair_is_excluded(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "ミント", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        self.assertEqual(extracted.iloc[0]["extraction_status"], "excluded")
        self.assertEqual(extracted.iloc[0]["exclusion_reason"], "self_pair")

    def test_orderless_pair_key_is_normalized(self) -> None:
        self.assertEqual(ordered_pair_key("ミント", "レモン"), "ミント||レモン")
        self.assertEqual(ordered_pair_key("レモン", "ミント"), "ミント||レモン")

    def test_duplicate_rows_are_flagged(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 2, "section_heading": "おすすめミックス", "cell_text": "レモン", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        self.assertEqual(int(extracted["is_duplicate_pair"].sum()), 1)

    def test_empty_cell_and_none_marker_are_kept_as_audit_rows(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "スイーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 2, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "×", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        self.assertIn("empty_cell", extracted["exclusion_reason"].tolist())
        self.assertIn("explicit_none_marker", extracted["exclusion_reason"].tolist())

    def test_missing_target_flavor_is_recorded(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": ""}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        self.assertEqual(extracted.iloc[0]["exclusion_reason"], "missing_target_flavor")

    def test_dictionary_candidate_audit(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン/パイン", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        audit_df = build_dictionary_candidate_audit(extracted, self.flavor_dict)
        self.assertEqual(audit_df.iloc[0]["raw表記"], "パイン")

    def test_build_unique_pairs_and_comparison_metrics(self) -> None:
        extracted = pd.DataFrame(
            [
                {
                    "article_id": "A1",
                    "article_url": "https://example.com/1",
                    "target_flavor_raw": "ミント",
                    "recommended_flavor_raw": "レモン",
                    "target_flavor_normalized": "ミント",
                    "recommended_flavor_normalized": "レモン",
                    "directed_pair_key": "ミント -> レモン",
                    "mix_pair_key": "ミント||レモン",
                    "source_text": "レモン",
                    "extraction_status": "ok",
                    "is_valid_pair": True,
                },
                {
                    "article_id": "A2",
                    "article_url": "https://example.com/2",
                    "target_flavor_raw": "ミント",
                    "recommended_flavor_raw": "バニラ",
                    "target_flavor_normalized": "ミント",
                    "recommended_flavor_normalized": "バニラ",
                    "directed_pair_key": "ミント -> バニラ",
                    "mix_pair_key": "バニラ||ミント",
                    "source_text": "バニラ",
                    "extraction_status": "ok",
                    "is_valid_pair": True,
                },
            ]
        )
        unique_pairs = build_unique_pairs(extracted)
        ranking_df = self.make_ranking()
        agreement = compute_at_k_metrics(ranking_df, set(unique_pairs["mix_pair_key"].tolist()), [1, 2, 3])
        self.assertEqual(int(agreement.loc[agreement["k"] == 1, "common_pair_count"].iloc[0]), 1)
        self.assertAlmostEqual(float(agreement.loc[agreement["k"] == 2, "precision_at_k"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(agreement.loc[agreement["k"] == 3, "recall_at_k"].iloc[0]), 1.0)

        common_df = build_common_pairs_with_existing(unique_pairs, ranking_df)
        self.assertEqual(len(common_df), 2)
        top_missing_df = build_existing_topk_not_in_lagos(ranking_df, set(unique_pairs["mix_pair_key"].tolist()), top_k=3)
        self.assertEqual(len(top_missing_df), 1)

    def test_build_lagos_only_pairs_reasons(self) -> None:
        unique_pairs = pd.DataFrame(
            [
                {
                    "mix_pair_key": "グレープ||ライム",
                    "flavor_a": "グレープ",
                    "flavor_b": "ライム",
                    "lagos_article_count": 1,
                    "lagos_row_count": 1,
                    "lagos_article_urls": "https://example.com/1",
                    "target_flavor": "グレープ",
                    "recommended_flavor": "ライム",
                    "directed_pair": "グレープ -> ライム",
                    "source_text": "ライム",
                }
            ]
        )
        ranking_df = self.make_ranking()
        pair_features = pd.DataFrame(
            [
                {
                    "flavor_a": "グレープ",
                    "flavor_b": "ライム",
                    "pair_key": "グレープ||ライム",
                    "pair_count": 1,
                    "excluded_as_product_name_pair": False,
                    "is_parent_child_pair": False,
                }
            ]
        )
        excluded_df = pd.DataFrame(columns=["pair_key", "excluded_reason"])
        lagos_only_df = build_lagos_only_pairs(unique_pairs, ranking_df, pair_features, excluded_df)
        self.assertEqual(lagos_only_df.iloc[0]["既存ランキングに存在しない理由の候補"], "pair_count閾値未満")

    def test_extraction_summary_counts(self) -> None:
        articles = self.make_articles([{"article_id": "A1", "target_flavor": "ミント"}])
        tables = self.make_tables(
            [
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 1, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "フルーツ系", "is_recommended_mix_section": True},
                {"article_id": "A1", "article_url": "https://example.com/1", "table_index": 3, "row_index": 2, "cell_index": 1, "section_heading": "おすすめミックス", "cell_text": "レモン", "is_recommended_mix_section": True},
            ]
        )
        extracted = extract_mix_pairs(articles, tables, self.flavor_dict)
        summary = build_extraction_summary(articles, tables, extracted)
        self.assertEqual(int(summary.iloc[0]["article_count"]), 1)
        self.assertEqual(int(summary.iloc[0]["valid_row_count"]), 1)

    def test_state_counts_sum_to_extracted_rows(self) -> None:
        extracted = pd.DataFrame(
            [
                {"extraction_status": "ok", "is_valid_pair": True},
                {"extraction_status": "ok", "is_valid_pair": True},
                {"extraction_status": "unresolved", "is_valid_pair": False},
                {"extraction_status": "excluded", "is_valid_pair": False},
            ]
        )
        total = len(extracted)
        valid = int(extracted["is_valid_pair"].sum())
        unresolved = int(extracted["extraction_status"].eq("unresolved").sum())
        excluded = int(extracted["extraction_status"].eq("excluded").sum())
        self.assertEqual(valid + unresolved + excluded, total)

    def test_pair_repetition_audit_counts(self) -> None:
        extracted = pd.DataFrame(
            [
                {
                    "article_id": "A1",
                    "article_url": "https://example.com/1",
                    "table_index": 3,
                    "row_index": 2,
                    "directed_pair_key": "ミント -> レモン",
                    "mix_pair_key": "ミント||レモン",
                    "is_valid_pair": True,
                },
                {
                    "article_id": "A2",
                    "article_url": "https://example.com/2",
                    "table_index": 3,
                    "row_index": 2,
                    "directed_pair_key": "レモン -> ミント",
                    "mix_pair_key": "ミント||レモン",
                    "is_valid_pair": True,
                },
                {
                    "article_id": "A2",
                    "article_url": "https://example.com/2",
                    "table_index": 3,
                    "row_index": 2,
                    "directed_pair_key": "バニラ -> ミント",
                    "mix_pair_key": "バニラ||ミント",
                    "is_valid_pair": True,
                },
            ]
        )
        audit_df, summary_df = build_pair_repetition_audit(extracted)
        self.assertEqual(int(summary_df.loc[summary_df["metric"] == "repeated_mix_pair_key_count", "value"].iloc[0]), 1)
        self.assertEqual(int(summary_df.loc[summary_df["metric"] == "multi_article_mix_pair_count", "value"].iloc[0]), 1)
        self.assertEqual(int(audit_df.loc[audit_df["mix_pair_key"] == "ミント||レモン", "article_count"].iloc[0]), 2)

    def test_dictionary_manual_review_counts_target_and_recommended_separately(self) -> None:
        extracted = pd.DataFrame(
            [
                {
                    "article_id": "A1",
                    "article_url": "https://example.com/1",
                    "target_flavor_raw": "スイカ",
                    "target_flavor_normalized": "",
                    "recommended_flavor_raw": "レモン",
                    "recommended_flavor_normalized": "レモン",
                    "source_text": "レモン",
                },
                {
                    "article_id": "A2",
                    "article_url": "https://example.com/2",
                    "target_flavor_raw": "ミント",
                    "target_flavor_normalized": "ミント",
                    "recommended_flavor_raw": "スイカ",
                    "recommended_flavor_normalized": "",
                    "source_text": "スイカ",
                },
            ]
        )
        master_df = pd.DataFrame({"フレーバー名": ["デクラウド　スイカ"], "ブランド": ["不明"]})
        review_df = build_dictionary_candidate_manual_review(extracted, self.flavor_dict, master_df)
        row = review_df.loc[review_df["raw表記"] == "スイカ"].iloc[0]
        self.assertEqual(int(row["target側出現行数"]), 1)
        self.assertEqual(int(row["recommended側出現行数"]), 1)

    def test_dictionary_update_impact_simulation_produces_separate_rows(self) -> None:
        extracted = pd.DataFrame(
            [
                {
                    "article_id": "A1",
                    "article_url": "https://example.com/1",
                    "target_flavor_raw": "スイカ",
                    "target_flavor_normalized": "",
                    "recommended_flavor_raw": "レモン",
                    "recommended_flavor_normalized": "レモン",
                    "directed_pair_key": "",
                    "mix_pair_key": "",
                    "source_text": "レモン",
                    "extraction_status": "unresolved",
                    "exclusion_reason": "target_flavor_unregistered",
                    "is_valid_pair": False,
                    "is_self_pair": False,
                    "is_duplicate_pair": False,
                }
            ]
        )
        ranking_df = pd.DataFrame(
            [
                {
                    "rank_overall": 1,
                    "flavor_a": "デクラウド　スイカ",
                    "flavor_b": "レモン",
                    "pair_key": "デクラウド　スイカ||レモン",
                    "pair_count": 3,
                    "support": 0.03,
                    "lift": 1.1,
                    "adjusted_lift": 0.01,
                    "centrality_mean": 0.10,
                    "smoothed_positive_ratio": 0.00,
                    "smoothed_negative_ratio": 0.00,
                    "smoothed_role_ratio": 0.00,
                    "overall_score_v2": 0.15,
                    "ranking_tier": "Tier2",
                }
            ]
        )
        simulation_df = build_dictionary_update_impact_simulation(extracted, ranking_df)
        self.assertEqual(simulation_df["scenario"].tolist(), ["baseline", "conservative", "extended"])
        baseline_valid = int(simulation_df.loc[simulation_df["scenario"] == "baseline", "lagos_valid_row_count"].iloc[0])
        conservative_valid = int(simulation_df.loc[simulation_df["scenario"] == "conservative", "lagos_valid_row_count"].iloc[0])
        self.assertLess(baseline_valid, conservative_valid)

    def test_topk_pair_audit_row_count(self) -> None:
        unique_pairs = pd.DataFrame(
            [
                {
                    "mix_pair_key": "ミント||レモン",
                    "flavor_a": "ミント",
                    "flavor_b": "レモン",
                    "lagos_article_count": 1,
                    "lagos_row_count": 1,
                    "lagos_article_urls": "https://example.com/1",
                    "target_flavor": "ミント",
                    "recommended_flavor": "レモン",
                    "directed_pair": "ミント -> レモン",
                    "source_text": "レモン",
                }
            ]
        )
        ranking_df = self.make_ranking()
        audit_df = build_topk_pair_audit(ranking_df, unique_pairs, [1, 2, 3])
        self.assertEqual(len(audit_df), 6)
        self.assertTrue(audit_df["K"].isin([1, 2, 3]).all())

    def test_baseline_records_are_reproducible(self) -> None:
        extracted = pd.DataFrame(
            [
                {
                    "article_id": "A1",
                    "article_url": "https://example.com/1",
                    "target_flavor_raw": "ミント",
                    "recommended_flavor_raw": "レモン",
                    "target_flavor_normalized": "ミント",
                    "recommended_flavor_normalized": "レモン",
                    "directed_pair_key": "ミント -> レモン",
                    "mix_pair_key": "ミント||レモン",
                    "source_text": "レモン",
                    "extraction_status": "ok",
                    "is_valid_pair": True,
                }
            ]
        )
        unique_pairs = build_unique_pairs(extracted)
        ranking_df = self.make_ranking()
        agreement_df = compute_at_k_metrics(ranking_df, set(unique_pairs["mix_pair_key"].tolist()), BASELINE_TOP_KS)
        baseline_df = build_baseline_records(
            extracted_df=extracted,
            unique_pairs_df=unique_pairs,
            ranking_df=ranking_df,
            agreement_df=agreement_df,
            ranking_path="outputs/extended_analysis_v2/pair_ranking_tier2.csv",
            ranking_columns=["rank_overall", "pair_key"],
            dictionary_hash="dummyhash",
            git_commit_hash="dummycommit",
            executed_at="2026-08-02T12:00:00",
        )
        self.assertEqual(
            baseline_df.loc[baseline_df["metric"] == "valid_row_count", "value"].iloc[0],
            1,
        )
        self.assertEqual(
            baseline_df.loc[baseline_df["metric"] == "executed_at", "value"].iloc[0],
            "2026-08-02T12:00:00",
        )


if __name__ == "__main__":
    unittest.main()
