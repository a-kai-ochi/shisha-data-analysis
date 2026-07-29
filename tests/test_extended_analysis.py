#!/usr/bin/env python3
"""Tests for extended normalized flavor-pair analysis."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extended_analysis_utils import (  # noqa: E402
    add_normalized_features,
    add_normalized_features_v2,
    build_cooccurrence_graph,
    build_manual_validation_tier1_dataframe,
    build_pair_ranking,
    build_pair_ranking_v2,
    compute_manual_validation_agreement,
    compute_manual_validation_disagreements,
    compute_manual_validation_outputs,
    compute_centrality_dataframe,
    compute_manual_validation_summary,
    compute_pair_statistics,
    deduplicate_evidence_rows,
    detect_parent_child_pair,
    extract_pair_expression_features_v2,
    find_category_matches,
    load_expression_dictionary,
    load_template_sentence_patterns,
    merge_pair_features_v2,
    manual_validation_has_labels,
    output_paths,
    representative_context_rows_for_pair,
    role_terms_for_sentence,
    split_ranking_tiers_v2,
    spearman_rank_correlation,
    write_manual_validation_summary_markdown,
    write_manual_validation_guideline,
    write_manual_validation_summary_markdown_v2,
    analyze_sentence_categories,
)


class ExtendedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.expression_dictionary = load_expression_dictionary(
            root / "config" / "taste_expression_dictionary.json"
        )
        cls.template_patterns = load_template_sentence_patterns(
            root / "config" / "template_sentence_patterns.json"
        )
        cls.sorted_patterns = [
            "グレープミント",
            "カルダモンミルク",
            "グレープ",
            "ミント",
            "バニラ",
            "マンゴー",
            "メロン",
        ]
        cls.pattern_to_canonical = {
            "グレープミント": "グレープミント",
            "カルダモンミルク": "カルダモンミルク",
            "グレープ": "グレープ",
            "ミント": "ミント",
            "バニラ": "バニラ",
            "マンゴー": "マンゴー",
            "メロン": "メロン",
        }

    def make_docs(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        base_rows = []
        for idx, row in enumerate(rows, 1):
            base_rows.append(
                {
                    "document_id": f"R{idx:04d}",
                    "review_title": row.get("review_title", ""),
                    "review_date": "",
                    "review_url": row.get("review_url", f"https://example.com/{idx}"),
                    "review_summary": row.get("review_summary", ""),
                    "review_body": row.get("review_body", ""),
                    "normalized_flavors": row.get("normalized_flavors", []),
                    "title_flavors": row.get("title_flavors", []),
                    "summary_flavors": row.get("summary_flavors", []),
                    "flavor_count": len(row.get("normalized_flavors", [])),
                    "has_mix_keyword": row.get("has_mix_keyword", False),
                    "mix_keywords": row.get("mix_keywords", ""),
                }
            )
        return pd.DataFrame(base_rows)

    def make_pair_df(self, pairs: list[dict[str, object]]) -> pd.DataFrame:
        rows = []
        for pair in pairs:
            rows.append(
                {
                    "flavor_a": pair["flavor_a"],
                    "flavor_b": pair["flavor_b"],
                    "pair_key": f"{pair['flavor_a']}||{pair['flavor_b']}",
                    "pair_count": pair.get("pair_count", 1),
                    "support": pair.get("support", 0.1),
                    "lift": pair.get("lift", 1.0),
                    "centrality_mean": pair.get("centrality_mean", 0.1),
                    "centrality_max": pair.get("centrality_max", 0.1),
                    "centrality_geometric_mean": pair.get("centrality_geometric_mean", 0.1),
                }
            )
        return pd.DataFrame(rows)

    def test_support_lift_on_dummy_data(self) -> None:
        docs = pd.DataFrame(
            {
                "normalized_flavors": [
                    ["A", "B"],
                    ["A", "B", "C"],
                    ["A", "C"],
                ]
            }
        )
        pair_df, _, _ = compute_pair_statistics(docs)
        ab_row = pair_df[pair_df["pair_key"] == "A||B"].iloc[0]
        self.assertEqual(int(ab_row["pair_count"]), 2)
        self.assertTrue(math.isclose(float(ab_row["support"]), 2 / 3))
        self.assertTrue(math.isclose(float(ab_row["lift"]), 1.0))

    def test_existing_support_lift_match_current_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        support_df = pd.read_csv(root / "poster_analysis" / "support_top_pairs.csv")
        lift_df = pd.read_csv(root / "poster_analysis" / "lift_top_pairs.csv")
        self.assertEqual(support_df.iloc[0]["pair"], "グレープ × ミント")
        self.assertEqual(int(support_df.iloc[0]["n"]), 7)
        self.assertEqual(lift_df.iloc[0]["pair"], "カルダモンミルク × モヒート")
        self.assertAlmostEqual(float(lift_df.iloc[0]["lift"]), 18.15, places=2)

    def test_betweenness_can_be_computed(self) -> None:
        pair_df = pd.DataFrame(
            [
                {"flavor_a": "A", "flavor_b": "B", "pair_count": 3, "pair_key": "A||B"},
                {"flavor_a": "B", "flavor_b": "C", "pair_count": 2, "pair_key": "B||C"},
            ]
        )
        graph = build_cooccurrence_graph(pair_df)
        centrality_df = compute_centrality_dataframe(graph)
        self.assertIn("weighted_betweenness_centrality", centrality_df.columns)
        b_row = centrality_df[centrality_df["flavor"] == "B"].iloc[0]
        self.assertGreater(float(b_row["weighted_betweenness_centrality"]), 0.0)

    def test_zero_division_is_avoided(self) -> None:
        pair_df = pd.DataFrame(columns=["flavor_a", "flavor_b", "pair_count", "pair_key"])
        graph = build_cooccurrence_graph(pair_df)
        centrality_df = compute_centrality_dataframe(graph)
        self.assertEqual(len(centrality_df), 0)

    def test_dictionary_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dictionary = load_expression_dictionary(root / "config" / "taste_expression_dictionary.json")
        self.assertIn("taste", dictionary)
        self.assertIn("evaluation", dictionary)
        self.assertIn("negations", dictionary)

    def test_negation_is_not_counted_as_positive(self) -> None:
        negations = ["ない", "なく", "ません", "ではない", "じゃない"]
        matched, negated = find_category_matches("この組み合わせは美味しくない", ["美味しい"], negations)
        self.assertEqual(matched, [])
        self.assertEqual(negated, ["美味しい"])

    def test_normalized_values_are_within_unit_interval(self) -> None:
        df = pd.DataFrame(
            {
                "support": [0.01, 0.05, 0.10],
                "pair_count": [1, 3, 5],
                "lift": [1.0, 4.0, 20.0],
                "centrality_mean": [0.0, 0.2, 0.8],
                "positive_document_ratio": [0.0, 0.5, 1.0],
                "taste_role_explanation_ratio": [0.0, 0.3, 0.9],
            }
        )
        normalized_df = add_normalized_features(df)
        for column in [
            "normalized_support",
            "normalized_lift",
            "normalized_centrality_mean",
            "normalized_positive_document_ratio",
            "normalized_taste_role_explanation_ratio",
        ]:
            self.assertTrue(((normalized_df[column] >= 0.0) & (normalized_df[column] <= 1.0)).all())

    def test_lift_outlier_clip_works(self) -> None:
        df = pd.DataFrame(
            {
                "support": [0.01, 0.02, 0.03],
                "pair_count": [2, 2, 2],
                "lift": [1.0, 2.0, 1000.0],
                "centrality_mean": [0.1, 0.2, 0.3],
                "positive_document_ratio": [0.1, 0.2, 0.3],
                "taste_role_explanation_ratio": [0.1, 0.2, 0.3],
            }
        )
        normalized_df = add_normalized_features(df)
        self.assertLess(float(normalized_df["lift_clipped"].max()), 1000.0)

    def test_overall_score_matches_formula(self) -> None:
        df = pd.DataFrame(
            {
                "support": [0.01],
                "pair_count": [2],
                "lift": [3.0],
                "centrality_mean": [0.5],
                "positive_document_ratio": [0.2],
                "taste_role_explanation_ratio": [0.1],
                "flavor_a": ["A"],
                "flavor_b": ["B"],
                "pair_key": ["A||B"],
                "evidence_document_count": [1],
                "negative_document_ratio": [0.0],
                "centrality_max": [0.5],
                "centrality_geometric_mean": [0.5],
            }
        )
        normalized_df = add_normalized_features(df)
        ranked_df = build_pair_ranking(normalized_df)
        row = ranked_df.iloc[0]
        expected = (
            0.30 * row["normalized_support"]
            + 0.25 * row["normalized_lift"]
            + 0.20 * row["normalized_centrality_mean"]
            + 0.15 * row["normalized_positive_document_ratio"]
            + 0.10 * row["normalized_taste_role_explanation_ratio"]
        )
        self.assertAlmostEqual(float(row["overall_score"]), float(expected), places=10)

    def test_tiebreak_is_deterministic(self) -> None:
        df = pd.DataFrame(
            {
                "support": [0.1, 0.1],
                "pair_count": [3, 3],
                "lift": [2.0, 2.0],
                "centrality_mean": [0.2, 0.2],
                "positive_document_ratio": [0.1, 0.1],
                "taste_role_explanation_ratio": [0.1, 0.1],
                "flavor_a": ["A", "A"],
                "flavor_b": ["B", "C"],
                "pair_key": ["A||B", "A||C"],
                "evidence_document_count": [1, 1],
                "negative_document_ratio": [0.0, 0.0],
                "centrality_max": [0.2, 0.2],
                "centrality_geometric_mean": [0.2, 0.2],
            }
        )
        ranked_df = build_pair_ranking(add_normalized_features(df))
        self.assertEqual(ranked_df.iloc[0]["pair_key"], "A||B")

    def test_same_input_same_result(self) -> None:
        df = pd.DataFrame(
            {
                "support": [0.1, 0.2],
                "pair_count": [2, 4],
                "lift": [2.0, 3.0],
                "centrality_mean": [0.2, 0.4],
                "positive_document_ratio": [0.1, 0.3],
                "taste_role_explanation_ratio": [0.0, 0.2],
                "flavor_a": ["A", "B"],
                "flavor_b": ["C", "D"],
                "pair_key": ["A||C", "B||D"],
                "evidence_document_count": [1, 1],
                "negative_document_ratio": [0.0, 0.0],
                "centrality_max": [0.2, 0.4],
                "centrality_geometric_mean": [0.2, 0.4],
            }
        )
        ranked_1 = build_pair_ranking(add_normalized_features(df.copy()))
        ranked_2 = build_pair_ranking(add_normalized_features(df.copy()))
        self.assertEqual(ranked_1["pair_key"].tolist(), ranked_2["pair_key"].tolist())
        self.assertEqual(ranked_1["rank_overall"].tolist(), ranked_2["rank_overall"].tolist())

    def test_manual_validation_without_labels_does_not_crash(self) -> None:
        df = pd.DataFrame(
            {
                "pair_key": ["A||B"],
                "rank_overall": [1],
                "rank_support": [1],
                "rank_lift": [1],
                "rank_support_lift": [1],
                "mix_relation_label": [""],
                "evaluation_label": [""],
                "taste_role_label": [""],
                "recommendation_validity": [""],
            }
        )
        self.assertFalse(manual_validation_has_labels(df))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = output_paths(output_dir)
            write_manual_validation_summary_markdown(pd.DataFrame(), paths.manual_validation_summary_md, has_labels=False)
            content = paths.manual_validation_summary_md.read_text(encoding="utf-8")
            self.assertIn("未評価のため集計できない", content)

    def test_manual_validation_summary_computation(self) -> None:
        df = pd.DataFrame(
            {
                "rank_overall": [1, 2],
                "rank_support": [1, 2],
                "rank_lift": [2, 1],
                "rank_support_lift": [1, 2],
                "mix_relation_label": ["explicit_mix", "co_mention_only"],
                "evaluation_label": ["positive", "neutral"],
                "taste_role_label": ["explained", "not_explained"],
                "recommendation_validity": ["valid", "partially_valid"],
            }
        )
        summary_df = compute_manual_validation_summary(df, [1, 2])
        self.assertFalse(summary_df.empty)
        explicit_top1 = summary_df[
            (summary_df["ranking_name"] == "overall")
            & (summary_df["k"] == 1)
            & (summary_df["metric"] == "explicit_mix_rate")
        ]["value"].iloc[0]
        self.assertAlmostEqual(float(explicit_top1), 1.0)

    def test_manual_validation_tier1_contains_all_17_candidates(self) -> None:
        ranking_df = pd.DataFrame(
            [
                {
                    "rank_overall": idx,
                    "pair_key": f"A{idx}||B{idx}",
                    "flavor_a": f"A{idx}",
                    "flavor_b": f"B{idx}",
                    "pair_count": 3,
                    "same_sentence_evidence_document_count": 2,
                    "support": 0.05,
                    "lift": 2.0,
                    "adjusted_lift": 0.04,
                    "centrality_mean": 0.1,
                    "smoothed_positive_ratio": 0.1,
                    "smoothed_negative_ratio": 0.0,
                    "smoothed_role_ratio": 0.05,
                    "overall_score_v2": 0.2,
                    "ranking_tier": "Tier1",
                }
                for idx in range(1, 18)
            ]
        )
        evidence_df = pd.DataFrame(
            [
                {
                    "pair_key": f"A{idx}||B{idx}",
                    "document_id": f"R{idx:04d}",
                    "sentence": f"A{idx}とB{idx}をミックスする。",
                    "is_same_sentence_pair": True,
                    "has_explicit_mix_expression": True,
                    "is_template_sentence": False,
                    "has_positive_expression": False,
                    "has_negative_expression": False,
                    "has_taste_role_explanation": False,
                }
                for idx in range(1, 18)
            ]
        )
        manual_df = build_manual_validation_tier1_dataframe(ranking_df, evidence_df)
        self.assertEqual(len(manual_df), 17)
        self.assertEqual(manual_df["rank"].tolist(), list(range(1, 18)))
        self.assertEqual(len(set(manual_df["rank"].tolist())), 17)

    def test_representative_context_prefers_same_sentence_and_non_template(self) -> None:
        evidence_df = pd.DataFrame(
            [
                {
                    "pair_key": "A||B",
                    "document_id": "R0001",
                    "sentence": "AとBをミックスすると美味しい。",
                    "is_same_sentence_pair": True,
                    "has_explicit_mix_expression": True,
                    "is_template_sentence": False,
                    "has_positive_expression": True,
                    "has_negative_expression": False,
                    "has_taste_role_explanation": False,
                },
                {
                    "pair_key": "A||B",
                    "document_id": "R0002",
                    "sentence": "AとBのおすすめミックス",
                    "is_same_sentence_pair": True,
                    "has_explicit_mix_expression": True,
                    "is_template_sentence": True,
                    "has_positive_expression": False,
                    "has_negative_expression": False,
                    "has_taste_role_explanation": False,
                },
                {
                    "pair_key": "A||B",
                    "document_id": "R0003",
                    "sentence": "AとBを組み合わせると甘さが加わる。",
                    "is_same_sentence_pair": True,
                    "has_explicit_mix_expression": True,
                    "is_template_sentence": False,
                    "has_positive_expression": False,
                    "has_negative_expression": False,
                    "has_taste_role_explanation": True,
                },
            ]
        )
        contexts = representative_context_rows_for_pair(evidence_df, "A||B")
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[0]["document_id"], "R0001")
        self.assertEqual(contexts[1]["document_id"], "R0003")
        self.assertNotIn("おすすめミックス", " ".join(item["sentence"] for item in contexts))

    def test_manual_validation_outputs_ignore_unfilled_labels(self) -> None:
        df = pd.DataFrame(
            {
                "rank": [1, 2],
                "pair_key": ["A||B", "C||D"],
                "flavor_a": ["A", "C"],
                "flavor_b": ["B", "D"],
                "smoothed_positive_ratio": [0.2, 0.0],
                "smoothed_negative_ratio": [0.0, 0.1],
                "smoothed_role_ratio": [0.1, 0.0],
                "overall_score_v2": [0.3, 0.2],
                "mix_relation_label": ["", ""],
                "evaluation_label": ["", ""],
                "taste_role_label": ["", ""],
                "recommendation_validity": ["", ""],
                "semantic_overlap_label": ["", ""],
            }
        )
        summary_df, crosstab_df, agreement_df, disagreements_df, primary_source = compute_manual_validation_outputs(
            df,
            k_values=[5, 10, 17],
        )
        self.assertTrue(summary_df.empty)
        self.assertTrue(crosstab_df.empty)
        self.assertIsNone(primary_source)
        self.assertTrue(disagreements_df.empty)
        self.assertIn("status", agreement_df.columns)

    def test_manual_validation_outputs_support_single_reviewer(self) -> None:
        df = pd.DataFrame(
            {
                "rank": [1, 2],
                "pair_key": ["A||B", "C||D"],
                "flavor_a": ["A", "C"],
                "flavor_b": ["B", "D"],
                "smoothed_positive_ratio": [0.8, 0.0],
                "smoothed_negative_ratio": [0.0, 0.6],
                "smoothed_role_ratio": [0.5, 0.0],
                "overall_score_v2": [0.9, 0.2],
                "mix_relation_label": ["explicit_mix", "co_mention_only"],
                "evaluation_label": ["positive", "negative"],
                "taste_role_label": ["explained", "not_explained"],
                "recommendation_validity": ["valid", "invalid"],
                "semantic_overlap_label": ["distinct", "duplicate"],
            }
        )
        summary_df, crosstab_df, _agreement_df, _disagreements_df, primary_source = compute_manual_validation_outputs(
            df,
            k_values=[5, 10, 17],
        )
        self.assertEqual(primary_source, "base")
        candidate_count = summary_df[
            (summary_df["section"] == "scope_metric")
            & (summary_df["scope"] == "all")
            & (summary_df["metric"] == "candidate_count")
        ]["value"].iloc[0]
        explicit_rate = summary_df[
            (summary_df["section"] == "scope_metric")
            & (summary_df["scope"] == "all")
            & (summary_df["metric"] == "explicit_mix_rate")
        ]["value"].iloc[0]
        self.assertEqual(int(candidate_count), 2)
        self.assertAlmostEqual(float(explicit_rate), 0.5)
        self.assertFalse(crosstab_df.empty)
        self.assertTrue({"top_5", "top_10", "top_17"}.issubset(set(summary_df["scope"])))

    def test_manual_validation_agreement_and_disagreements_for_two_reviewers(self) -> None:
        df = pd.DataFrame(
            {
                "rank": [1, 2],
                "pair_key": ["A||B", "C||D"],
                "flavor_a": ["A", "C"],
                "flavor_b": ["B", "D"],
                "context_1": ["ctx1", "ctx2"],
                "context_2": ["", ""],
                "context_3": ["", ""],
                "reviewer1_mix_relation_label": ["explicit_mix", "likely_mix"],
                "reviewer2_mix_relation_label": ["explicit_mix", "co_mention_only"],
                "reviewer1_evaluation_label": ["positive", "negative"],
                "reviewer2_evaluation_label": ["positive", "negative"],
                "reviewer1_taste_role_label": ["explained", "not_explained"],
                "reviewer2_taste_role_label": ["explained", "not_explained"],
                "reviewer1_recommendation_validity": ["valid", "invalid"],
                "reviewer2_recommendation_validity": ["valid", "unclear"],
                "reviewer1_semantic_overlap_label": ["distinct", "similar"],
                "reviewer2_semantic_overlap_label": ["distinct", "duplicate"],
                "reviewer1_comment": ["ok", "r1"],
                "reviewer2_comment": ["ok", "r2"],
            }
        )
        agreement_df = compute_manual_validation_agreement(df)
        disagreements_df = compute_manual_validation_disagreements(df)
        mix_row = agreement_df[agreement_df["field"] == "mix_relation_label"].iloc[0]
        self.assertEqual(int(mix_row["comparable_count"]), 2)
        self.assertAlmostEqual(float(mix_row["simple_agreement"]), 0.5)
        self.assertEqual(len(disagreements_df), 1)
        self.assertIn("mix_relation_label", disagreements_df.iloc[0]["disagreement_fields"])

    def test_manual_validation_summary_markdown_v2_handles_single_reviewer(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "section": "scope_metric",
                    "scope": "all",
                    "metric": "candidate_count",
                    "label": "",
                    "value": 17,
                    "n_labeled": 17,
                }
            ]
        )
        agreement_df = pd.DataFrame(
            [{"field": "", "comparable_count": 0, "simple_agreement": math.nan, "cohen_kappa": math.nan, "status": "評価者間一致は未計算"}]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = output_paths(output_dir)
            write_manual_validation_summary_markdown_v2(
                summary_df,
                agreement_df,
                paths.manual_validation_summary_md,
                primary_source="base",
            )
            content = paths.manual_validation_summary_md.read_text(encoding="utf-8")
            self.assertIn("primary_label_source", content)
            self.assertIn("評価者間一致は未計算", content)

    def test_manual_validation_guideline_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "guideline.md"
            write_manual_validation_guideline(path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("mix_relation_label", content)
            self.assertIn("semantic_overlap_label", content)

    def test_spearman_helper(self) -> None:
        corr = spearman_rank_correlation([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(float(corr), 1.0)

    def test_product_name_only_grape_mint_is_excluded(self) -> None:
        docs_df = self.make_docs(
            [
                {
                    "review_title": "MALAKI – Grape Mint(グレープミント)",
                    "review_body": "MALAKI – Grape Mint(グレープミント)の特徴\nMALAKI – Grape Mint(グレープミント)のおすすめミックス",
                    "normalized_flavors": ["グレープ", "ミント"],
                    "title_flavors": ["グレープ", "ミント"],
                }
            ]
        )
        pair_df = self.make_pair_df([{"flavor_a": "グレープ", "flavor_b": "ミント"}])
        features_df, _evidence_df, excluded_product_df, _excluded_parent_df = extract_pair_expression_features_v2(
            docs_df,
            pair_df,
            self.sorted_patterns,
            self.pattern_to_canonical,
            self.expression_dictionary,
            self.template_patterns,
        )
        self.assertTrue(bool(features_df.iloc[0]["excluded_as_product_name_pair"]))
        self.assertEqual(len(excluded_product_df), 1)

    def test_explicit_mix_sentence_keeps_product_name_pair(self) -> None:
        docs_df = self.make_docs(
            [
                {
                    "review_title": "MALAKI – Grape Mint(グレープミント)",
                    "review_body": "グレープとミントをミックスすると爽やかです。",
                    "normalized_flavors": ["グレープ", "ミント"],
                    "title_flavors": ["グレープ", "ミント"],
                }
            ]
        )
        pair_df = self.make_pair_df([{"flavor_a": "グレープ", "flavor_b": "ミント"}])
        features_df, evidence_df, excluded_product_df, _excluded_parent_df = extract_pair_expression_features_v2(
            docs_df,
            pair_df,
            self.sorted_patterns,
            self.pattern_to_canonical,
            self.expression_dictionary,
            self.template_patterns,
        )
        self.assertFalse(bool(features_df.iloc[0]["excluded_as_product_name_pair"]))
        self.assertEqual(len(excluded_product_df), 0)
        self.assertTrue(bool(evidence_df.iloc[0]["has_explicit_mix_expression"]))

    def test_parent_child_pair_is_detected(self) -> None:
        detected, _reason = detect_parent_child_pair("グレープ", "グレープミント")
        self.assertTrue(detected)

    def test_separate_sentences_do_not_add_same_sentence_positive(self) -> None:
        docs_df = self.make_docs(
            [
                {
                    "review_title": "separate",
                    "review_body": "グレープは美味しい。ミントは爽やか。",
                    "normalized_flavors": ["グレープ", "ミント"],
                    "title_flavors": [],
                }
            ]
        )
        pair_df = self.make_pair_df([{"flavor_a": "グレープ", "flavor_b": "ミント"}])
        features_df, _evidence_df, _excluded_product_df, _excluded_parent_df = extract_pair_expression_features_v2(
            docs_df,
            pair_df,
            self.sorted_patterns,
            self.pattern_to_canonical,
            self.expression_dictionary,
            self.template_patterns,
        )
        row = features_df.iloc[0]
        self.assertEqual(int(row["same_sentence_positive_count"]), 0)
        self.assertEqual(int(row["same_sentence_evidence_document_count"]), 0)

    def test_template_heading_does_not_count_as_positive_or_role(self) -> None:
        docs_df = self.make_docs(
            [
                {
                    "review_title": "MALAKI – Grape Mint(グレープミント)",
                    "review_body": "グレープとミントのおすすめミックス",
                    "normalized_flavors": ["グレープ", "ミント"],
                    "title_flavors": ["グレープ", "ミント"],
                }
            ]
        )
        pair_df = self.make_pair_df([{"flavor_a": "グレープ", "flavor_b": "ミント"}])
        features_df, evidence_df, _excluded_product_df, _excluded_parent_df = extract_pair_expression_features_v2(
            docs_df,
            pair_df,
            self.sorted_patterns,
            self.pattern_to_canonical,
            self.expression_dictionary,
            self.template_patterns,
        )
        self.assertEqual(int(features_df.iloc[0]["same_sentence_positive_count"]), 0)
        self.assertEqual(int(features_df.iloc[0]["same_sentence_role_count"]), 0)
        self.assertTrue(bool(evidence_df.iloc[0]["is_template_sentence"]))

    def test_mix_word_alone_is_not_role(self) -> None:
        has_role, _rule, _action, _effect = role_terms_for_sentence(
            "グレープとミントのミックスです。",
            self.expression_dictionary,
        )
        self.assertFalse(has_role)

    def test_sweetness_addition_is_role(self) -> None:
        has_role, rule, action, effect = role_terms_for_sentence(
            "バニラが甘さを加えるのでミントと合う。",
            self.expression_dictionary,
        )
        self.assertTrue(has_role)
        self.assertIn(rule, {"action", "action+effect", "effect"})
        self.assertTrue(action or effect)

    def test_negative_strong_expression_detected(self) -> None:
        info = analyze_sentence_categories("この組み合わせは強すぎる。", self.expression_dictionary)
        self.assertTrue(info["has_negative_expression"])

    def test_negative_kirai_expression_detected(self) -> None:
        info = analyze_sentence_categories("マンゴーが苦手な方は吸えないかもしれません。", self.expression_dictionary)
        self.assertTrue(info["has_negative_expression"])

    def test_negative_awanai_expression_detected(self) -> None:
        info = analyze_sentence_categories("グレープとミントは合わない。", self.expression_dictionary)
        self.assertTrue(info["has_negative_expression"])

    def test_negated_taste_and_positive_experience_both_kept(self) -> None:
        info = analyze_sentence_categories("甘くないので吸いやすい。", self.expression_dictionary)
        self.assertIn("taste:sweetness", info["matched_negated_categories"])
        self.assertIn("experience:smoothness", info["matched_categories"])

    def test_evidence_dedup_removes_complete_duplicates(self) -> None:
        evidence_df = pd.DataFrame(
            [
                {
                    "document_id": "R0001",
                    "flavor_a": "A",
                    "flavor_b": "B",
                    "sentence": "AとBをミックスする。",
                    "matched_categories": "evaluation:positive",
                    "matched_terms": "おすすめ",
                    "has_positive_expression": True,
                },
                {
                    "document_id": "R0001",
                    "flavor_a": "A",
                    "flavor_b": "B",
                    "sentence": "AとBをミックスする。",
                    "matched_categories": "evaluation:positive",
                    "matched_terms": "おすすめ",
                    "has_positive_expression": True,
                },
            ]
        )
        deduped = deduplicate_evidence_rows(evidence_df)
        self.assertEqual(len(deduped), 1)

    def test_pair_count_two_ratio_one_does_not_dominate_v2(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "flavor_a": "A",
                    "flavor_b": "B",
                    "pair_key": "A||B",
                    "pair_count": 2,
                    "support": 0.02,
                    "lift": 8.0,
                    "centrality_mean": 0.2,
                    "same_sentence_evidence_document_count": 2,
                    "same_sentence_positive_count": 2,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 2,
                },
                {
                    "flavor_a": "C",
                    "flavor_b": "D",
                    "pair_key": "C||D",
                    "pair_count": 5,
                    "support": 0.08,
                    "lift": 3.0,
                    "centrality_mean": 0.15,
                    "same_sentence_evidence_document_count": 3,
                    "same_sentence_positive_count": 2,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 1,
                },
            ]
        )
        for col in [
            "centrality_max",
            "centrality_geometric_mean",
            "document_cooccurrence_count",
            "same_sentence_cooccurrence_count",
            "explicit_mix_count",
            "document_level_positive_count",
            "document_level_negative_count",
            "document_level_role_count",
            "document_level_positive_ratio",
            "document_level_negative_ratio",
            "document_level_role_ratio",
            "same_sentence_positive_ratio",
            "same_sentence_negative_ratio",
            "same_sentence_role_ratio",
            "template_evidence_count",
            "positive_evidence_count",
            "negative_evidence_count",
            "role_evidence_count",
            "is_product_name_derived",
            "has_explicit_mix_expression",
            "excluded_as_product_name_pair",
            "is_parent_child_pair",
            "parent_child_reason",
            "evidence_rows_before_dedup",
            "evidence_rows_after_dedup",
            "evidence_duplicates_removed",
        ]:
            if col not in pair_df.columns:
                pair_df[col] = 0
        pair_df["document_cooccurrence_count"] = pair_df["pair_count"]
        normalized = add_normalized_features_v2(pair_df)
        ranked = build_pair_ranking_v2(normalized)
        self.assertEqual(ranked.iloc[0]["pair_key"], "C||D")

    def test_same_input_same_result_v2(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "flavor_a": "A",
                    "flavor_b": "B",
                    "pair_key": "A||B",
                    "pair_count": 3,
                    "support": 0.05,
                    "lift": 2.0,
                    "centrality_mean": 0.1,
                    "same_sentence_evidence_document_count": 2,
                    "same_sentence_positive_count": 1,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 1,
                    "document_cooccurrence_count": 3,
                    "same_sentence_cooccurrence_count": 2,
                    "explicit_mix_count": 1,
                },
                {
                    "flavor_a": "C",
                    "flavor_b": "D",
                    "pair_key": "C||D",
                    "pair_count": 4,
                    "support": 0.06,
                    "lift": 2.5,
                    "centrality_mean": 0.2,
                    "same_sentence_evidence_document_count": 2,
                    "same_sentence_positive_count": 1,
                    "same_sentence_negative_count": 0,
                    "same_sentence_role_count": 1,
                    "document_cooccurrence_count": 4,
                    "same_sentence_cooccurrence_count": 2,
                    "explicit_mix_count": 1,
                },
            ]
        )
        for col in [
            "centrality_max",
            "centrality_geometric_mean",
            "document_level_positive_count",
            "document_level_negative_count",
            "document_level_role_count",
            "document_level_positive_ratio",
            "document_level_negative_ratio",
            "document_level_role_ratio",
            "same_sentence_positive_ratio",
            "same_sentence_negative_ratio",
            "same_sentence_role_ratio",
            "template_evidence_count",
            "positive_evidence_count",
            "negative_evidence_count",
            "role_evidence_count",
            "is_product_name_derived",
            "has_explicit_mix_expression",
            "excluded_as_product_name_pair",
            "is_parent_child_pair",
            "parent_child_reason",
            "evidence_rows_before_dedup",
            "evidence_rows_after_dedup",
            "evidence_duplicates_removed",
        ]:
            if col not in pair_df.columns:
                pair_df[col] = 0
        ranked_1 = build_pair_ranking_v2(add_normalized_features_v2(pair_df.copy()))
        ranked_2 = build_pair_ranking_v2(add_normalized_features_v2(pair_df.copy()))
        self.assertEqual(ranked_1["pair_key"].tolist(), ranked_2["pair_key"].tolist())
        self.assertEqual(ranked_1["rank_overall"].tolist(), ranked_2["rank_overall"].tolist())


if __name__ == "__main__":
    unittest.main()
