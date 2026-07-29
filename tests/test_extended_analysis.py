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
    build_cooccurrence_graph,
    build_pair_ranking,
    compute_centrality_dataframe,
    compute_manual_validation_summary,
    compute_pair_statistics,
    find_category_matches,
    load_expression_dictionary,
    manual_validation_has_labels,
    output_paths,
    spearman_rank_correlation,
    write_manual_validation_summary_markdown,
)


class ExtendedAnalysisTests(unittest.TestCase):
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

    def test_spearman_helper(self) -> None:
        corr = spearman_rank_correlation([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(float(corr), 1.0)


if __name__ == "__main__":
    unittest.main()
