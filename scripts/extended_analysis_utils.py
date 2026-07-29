#!/usr/bin/env python3
"""Utilities for extended flavor-pair analysis on normalized flavor cooccurrence."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


MIX_KEYWORDS = [
    "ミックス",
    "組み合わせ",
    "ブレンド",
    "合わせ",
    "入れ",
    "加え",
    "メイン",
    "アクセント",
    "配合",
]

VERIFIED_CROSS_LANGUAGE_ALIASES = {
    "ライチ": ["LYCHEE"],
    "マンゴー": ["MANGO"],
    "チョコレート": ["CHOCOLATE"],
    "アールグレイ": ["EARL GREY"],
    "グレープフルーツ": ["GRAPEFRUIT"],
    "グレープ": ["GRAPE"],
    "コーラ": ["COLA"],
    "キウイ": ["KIWI"],
}

DEFAULT_RANKING_SETTINGS = {
    "SettingA_balanced": {
        "normalized_support": 0.30,
        "normalized_lift": 0.25,
        "normalized_centrality_mean": 0.20,
        "normalized_positive_document_ratio": 0.15,
        "normalized_taste_role_explanation_ratio": 0.10,
    },
    "SettingB_cooccurrence": {
        "normalized_support": 0.40,
        "normalized_lift": 0.35,
        "normalized_centrality_mean": 0.10,
        "normalized_positive_document_ratio": 0.10,
        "normalized_taste_role_explanation_ratio": 0.05,
    },
    "SettingC_context": {
        "normalized_support": 0.20,
        "normalized_lift": 0.15,
        "normalized_centrality_mean": 0.15,
        "normalized_positive_document_ratio": 0.30,
        "normalized_taste_role_explanation_ratio": 0.20,
    },
}

STANDARD_OVERALL_COLUMNS = [
    "normalized_support",
    "normalized_lift",
    "normalized_centrality_mean",
    "normalized_positive_document_ratio",
    "normalized_taste_role_explanation_ratio",
]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    description: str
    min_flavors: int
    max_flavors: int | None
    require_mix_keyword: bool


LIMITED_2_5 = ConditionSpec(
    name="limited_2_5",
    description="抽出フレーバー数が2〜5種類のレビュー",
    min_flavors=2,
    max_flavors=5,
    require_mix_keyword=False,
)


@dataclass(frozen=True)
class OutputPaths:
    flavor_centrality_csv: Path
    flavor_centrality_top20_md: Path
    pair_expression_features_csv: Path
    pair_expression_evidence_csv: Path
    pair_ranking_csv: Path
    manual_validation_candidates_csv: Path
    ranking_sensitivity_csv: Path
    ranking_sensitivity_md: Path
    extended_summary_md: Path
    figure_centrality_png: Path
    figure_score_breakdown_png: Path
    figure_support_lift_scatter_png: Path
    figure_support_overall_scatter_png: Path
    figure_lift_overall_scatter_png: Path
    figure_ranking_comparison_png: Path
    figure_manual_validity_png: Path
    manual_validation_summary_csv: Path
    manual_validation_summary_md: Path


def output_paths(output_dir: Path) -> OutputPaths:
    return OutputPaths(
        flavor_centrality_csv=output_dir / "flavor_centrality.csv",
        flavor_centrality_top20_md=output_dir / "flavor_centrality_top20.md",
        pair_expression_features_csv=output_dir / "pair_expression_features.csv",
        pair_expression_evidence_csv=output_dir / "pair_expression_evidence.csv",
        pair_ranking_csv=output_dir / "pair_ranking.csv",
        manual_validation_candidates_csv=output_dir / "manual_validation_candidates.csv",
        ranking_sensitivity_csv=output_dir / "ranking_sensitivity.csv",
        ranking_sensitivity_md=output_dir / "ranking_sensitivity.md",
        extended_summary_md=output_dir / "extended_analysis_summary.md",
        figure_centrality_png=output_dir / "figure_centrality_top20.png",
        figure_score_breakdown_png=output_dir / "figure_overall_score_breakdown.png",
        figure_support_lift_scatter_png=output_dir / "figure_support_lift_scatter.png",
        figure_support_overall_scatter_png=output_dir / "figure_support_overall_scatter.png",
        figure_lift_overall_scatter_png=output_dir / "figure_lift_overall_scatter.png",
        figure_ranking_comparison_png=output_dir / "figure_ranking_comparison.png",
        figure_manual_validity_png=output_dir / "figure_manual_validity.png",
        manual_validation_summary_csv=output_dir / "manual_validation_summary.csv",
        manual_validation_summary_md=output_dir / "manual_validation_summary.md",
    )


def clean_flavor_entry(raw_name: str, raw_brand: str) -> tuple[str, str]:
    name = str(raw_name).strip()
    sep_match = re.match(r"^(.+?)\s*[-–]\s*([A-Za-zァ-ヿ].+)", name)
    if sep_match:
        flavor_part = sep_match.group(1).strip()
        brand_part = sep_match.group(2).strip()
        brand_match = re.match(r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", brand_part)
        brand_short = brand_match.group(1) if brand_match else str(raw_brand)[:10]
        return flavor_part, brand_short

    brand_match = re.match(
        r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", str(raw_brand).strip()
    )
    brand_short = brand_match.group(1) if brand_match else "不明"
    return name, brand_short


def build_canonical_and_patterns(flavor_name: str) -> tuple[str, list[str]]:
    name = flavor_name.strip()
    ja_in_paren = re.search(r"\(([ァ-ヿ][^\)]+)\)", name)
    en_part = re.match(r"^([A-Za-z][A-Za-z0-9 .&'\-/]+?)(?:\(|$)", name)

    if ja_in_paren:
        ja_str = ja_in_paren.group(1).strip()
        en_str = en_part.group(1).strip() if en_part else ""
        patterns = [ja_str]
        if en_str and len(en_str) >= 3:
            patterns.append(en_str)
        return ja_str, patterns

    if re.match(r"^[ァ-ヿぁ-んー]", name):
        return name, [name]
    if re.match(r"^[A-Za-z]", name):
        return name.upper(), [name]
    return name, [name]


def build_flavor_dictionary(
    master_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    flavor_dict: dict[str, dict[str, Any]] = {}
    pattern_to_canonical: dict[str, str] = {}
    english_to_japanese_candidates: dict[str, set[str]] = {}

    for _, row in master_df.iterrows():
        flavor_clean, _brand_short = clean_flavor_entry(row["フレーバー名"], row["ブランド"])
        if not flavor_clean or flavor_clean.strip() in ("nan", ""):
            continue
        name = flavor_clean.strip()
        ja_in_paren = re.search(r"\(([ァ-ヿ][^\)]+)\)", name)
        en_part = re.match(r"^([A-Za-z][A-Za-z0-9 .&'\-/]+?)(?:\(|$)", name)
        if ja_in_paren and en_part:
            english_key = en_part.group(1).strip().upper()
            english_to_japanese_candidates.setdefault(english_key, set()).add(
                ja_in_paren.group(1).strip()
            )

    english_to_japanese = {
        english_key: next(iter(japanese_values))
        for english_key, japanese_values in english_to_japanese_candidates.items()
        if len(japanese_values) == 1
    }

    for _, row in master_df.iterrows():
        flavor_clean, brand_short = clean_flavor_entry(row["フレーバー名"], row["ブランド"])
        if not flavor_clean or flavor_clean.strip() in ("nan", ""):
            continue

        canonical, patterns = build_canonical_and_patterns(flavor_clean)
        if canonical in english_to_japanese:
            canonical = english_to_japanese[canonical]
            if canonical not in patterns:
                patterns = [canonical] + patterns

        for alias in VERIFIED_CROSS_LANGUAGE_ALIASES.get(canonical, []):
            if alias not in patterns:
                patterns.append(alias)

        patterns = [pat for pat in patterns if len(pat) >= 3]
        if not patterns:
            continue

        if canonical not in flavor_dict:
            flavor_dict[canonical] = {
                "patterns": set(),
                "brand": brand_short,
                "raw_names": [],
            }

        flavor_dict[canonical]["patterns"].update(patterns)
        flavor_dict[canonical]["raw_names"].append(flavor_clean)

        for pat in patterns:
            pattern_to_canonical.setdefault(pat, canonical)

    sorted_patterns = sorted(pattern_to_canonical.keys(), key=len, reverse=True)
    return flavor_dict, pattern_to_canonical, sorted_patterns


def extract_flavors(
    text: str,
    sorted_patterns: list[str],
    pattern_to_canonical: dict[str, str],
) -> list[str]:
    if not isinstance(text, str):
        return []

    found: set[str] = set()
    masked: set[int] = set()
    text_upper = text.upper()

    for pat in sorted_patterns:
        pat_upper = pat.upper()
        start = 0
        while True:
            idx = text_upper.find(pat_upper, start)
            if idx == -1:
                break
            end = idx + len(pat)
            if any(pos in masked for pos in range(idx, end)):
                start = idx + 1
                continue

            before_ok = True
            after_ok = True
            if idx > 0:
                prev_char = text[idx - 1]
                if (
                    unicodedata.category(prev_char) in ("Lo", "Ll", "Lu", "Nd")
                    and unicodedata.category(text[idx]) in ("Lo",)
                ):
                    before_ok = False
            if end < len(text):
                next_char = text[end]
                if (
                    unicodedata.category(next_char) in ("Lo",)
                    and unicodedata.category(text[end - 1]) in ("Lo",)
                ):
                    after_ok = False

            if before_ok and after_ok:
                found.add(pattern_to_canonical[pat])
                masked.update(range(idx, end))
            start = idx + 1

    return sorted(found)


def detect_mix_keywords(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    return [keyword for keyword in MIX_KEYWORDS if keyword in text]


def make_document_id(index: int) -> str:
    return f"R{index + 1:04d}"


def load_documents(
    reviews_csv: Path,
    master_csv: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, str], list[str]]:
    reviews_df = pd.read_csv(reviews_csv)
    master_df = pd.read_csv(master_csv)
    flavor_dict, pattern_to_canonical, sorted_patterns = build_flavor_dictionary(master_df)

    docs: list[dict[str, Any]] = []
    for idx, row in reviews_df.iterrows():
        body = row.get("レビュー本文", "")
        extracted = extract_flavors(body, sorted_patterns, pattern_to_canonical)
        docs.append(
            {
                "document_id": make_document_id(idx),
                "review_title": row.get("レビュータイトル", ""),
                "review_date": row.get("更新日", ""),
                "review_url": row.get("レビューURL", ""),
                "review_summary": row.get("概要", ""),
                "review_body": body,
                "normalized_flavors": extracted,
                "flavor_count": len(extracted),
                "has_mix_keyword": bool(detect_mix_keywords(body)),
                "mix_keywords": "|".join(detect_mix_keywords(body)),
            }
        )

    docs_df = pd.DataFrame(docs)
    return docs_df, flavor_dict, pattern_to_canonical, sorted_patterns


def apply_condition(docs_df: pd.DataFrame, condition: ConditionSpec) -> pd.DataFrame:
    df = docs_df.copy()
    df = df[df["flavor_count"] >= condition.min_flavors]
    if condition.max_flavors is not None:
        df = df[df["flavor_count"] <= condition.max_flavors]
    if condition.require_mix_keyword:
        df = df[df["has_mix_keyword"]]
    return df.reset_index(drop=True)


def compute_pair_statistics(docs_df: pd.DataFrame) -> tuple[pd.DataFrame, Counter, Counter]:
    flavor_frequency: Counter = Counter()
    pair_counts: Counter = Counter()

    for flavors in docs_df["normalized_flavors"]:
        unique_flavors = sorted(set(flavors))
        for flavor in unique_flavors:
            flavor_frequency[flavor] += 1
        for pair in combinations(unique_flavors, 2):
            pair_counts[pair] += 1

    total_docs = len(docs_df)
    rows = []
    for (flavor_a, flavor_b), pair_count in sorted(pair_counts.items()):
        freq_a = flavor_frequency[flavor_a]
        freq_b = flavor_frequency[flavor_b]
        support = pair_count / total_docs if total_docs else 0.0
        lift = 0.0
        if total_docs and freq_a and freq_b:
            lift = pair_count * total_docs / (freq_a * freq_b)
        rows.append(
            {
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "pair_key": f"{flavor_a}||{flavor_b}",
                "pair_count": pair_count,
                "frequency_a": freq_a,
                "frequency_b": freq_b,
                "support": support,
                "lift": lift,
            }
        )

    pair_df = pd.DataFrame(rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            by=["pair_count", "lift", "flavor_a", "flavor_b"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return pair_df, flavor_frequency, pair_counts


def build_cooccurrence_graph(pair_df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in pair_df.itertuples(index=False):
        pair_count = int(row.pair_count)
        if pair_count <= 0:
            continue
        distance = 1.0 / pair_count
        graph.add_edge(
            row.flavor_a,
            row.flavor_b,
            weight=pair_count,
            distance=distance,
        )
    return graph


def compute_centrality_dataframe(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "flavor",
                "degree",
                "weighted_degree",
                "betweenness_centrality",
                "weighted_betweenness_centrality",
                "cooccurrence_pair_count",
                "total_cooccurrence_count",
            ]
        )

    unweighted_bc = nx.betweenness_centrality(graph, normalized=True, weight=None)
    weighted_bc = nx.betweenness_centrality(graph, normalized=True, weight="distance")

    rows = []
    for flavor in sorted(graph.nodes()):
        degree = graph.degree(flavor)
        weighted_degree = graph.degree(flavor, weight="weight")
        rows.append(
            {
                "flavor": flavor,
                "degree": degree,
                "weighted_degree": float(weighted_degree),
                "betweenness_centrality": float(unweighted_bc.get(flavor, 0.0)),
                "weighted_betweenness_centrality": float(weighted_bc.get(flavor, 0.0)),
                "cooccurrence_pair_count": degree,
                "total_cooccurrence_count": float(weighted_degree),
            }
        )

    centrality_df = pd.DataFrame(rows).sort_values(
        by=[
            "weighted_betweenness_centrality",
            "betweenness_centrality",
            "total_cooccurrence_count",
            "flavor",
        ],
        ascending=[False, False, False, True],
    )
    centrality_df = centrality_df.reset_index(drop=True)
    return centrality_df


def add_pair_centrality_features(pair_df: pd.DataFrame, centrality_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    centrality_map = centrality_df.set_index("flavor")["weighted_betweenness_centrality"].to_dict()
    result = pair_df.copy()
    result["centrality_a"] = result["flavor_a"].map(centrality_map).fillna(0.0)
    result["centrality_b"] = result["flavor_b"].map(centrality_map).fillna(0.0)
    result["centrality_mean"] = (result["centrality_a"] + result["centrality_b"]) / 2.0
    result["centrality_max"] = result[["centrality_a", "centrality_b"]].max(axis=1)
    result["centrality_geometric_mean"] = np.sqrt(
        result["centrality_a"].clip(lower=0) * result["centrality_b"].clip(lower=0)
    )
    return result


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces = re.split(r"(?<=[。！？!?])|\n+", normalized)
    sentences = [piece.strip() for piece in pieces if piece and piece.strip()]
    return sentences


def load_expression_dictionary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def is_negated(text: str, start: int, end: int, negations: list[str]) -> bool:
    window = text[max(0, start - 4) : min(len(text), end + 8)]
    suffix = text[end : min(len(text), end + 8)]
    prefix = text[max(0, start - 4) : start]
    for neg in negations:
        if neg in suffix:
            return True
        if neg in window and any(marker in prefix for marker in ["あまり", "それほど"]):
            return True
    return False


def find_category_matches(
    text: str,
    terms: list[str],
    negations: list[str],
) -> tuple[list[str], list[str]]:
    matched_terms: list[str] = []
    negated_terms: list[str] = []
    for term in terms:
        matched_any = False
        for match in re.finditer(re.escape(term), text):
            if is_negated(text, match.start(), match.end(), negations):
                negated_terms.append(term)
            else:
                matched_terms.append(term)
            matched_any = True
        if matched_any:
            continue

        # "美味しい" -> "美味しくない" のようなイ形容詞の否定活用に最低限対応する
        if term.endswith("い") and len(term) >= 2:
            stem = re.escape(term[:-1])
            neg_pattern = "|".join(re.escape(neg) for neg in negations)
            pattern = rf"{stem}く(?:{neg_pattern})"
            if re.search(pattern, text):
                negated_terms.append(term)

    return matched_terms, negated_terms


def extract_pair_expression_features(
    docs_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    sorted_patterns: list[str],
    pattern_to_canonical: dict[str, str],
    expression_dictionary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negations = expression_dictionary["negations"]
    pair_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_lookup = set(pair_df["pair_key"].tolist())

    for row in docs_df.itertuples(index=False):
        flavors = sorted(set(row.normalized_flavors))
        for flavor_a, flavor_b in combinations(flavors, 2):
            pair_key = f"{flavor_a}||{flavor_b}"
            if pair_key in pair_lookup:
                pair_documents[pair_key].append(row._asdict())

    feature_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    category_columns = []
    for group_name in ("taste", "experience", "evaluation"):
        for category_name in expression_dictionary[group_name]:
            category_columns.append((group_name, category_name, f"{group_name}_{category_name}_count"))

    for pair_row in pair_df.itertuples(index=False):
        pair_key = pair_row.pair_key
        doc_rows = pair_documents.get(pair_key, [])
        flavor_a = pair_row.flavor_a
        flavor_b = pair_row.flavor_b
        category_counter: Counter = Counter()
        positive_doc_count = 0
        negative_doc_count = 0
        role_doc_count = 0
        same_sentence_doc_count = 0
        evidence_doc_count = 0

        for doc in doc_rows:
            sentences = split_sentences(doc["review_body"])
            pair_sentences: list[str] = []
            fallback_sentences: list[str] = []
            for sentence in sentences:
                sentence_flavors = extract_flavors(sentence, sorted_patterns, pattern_to_canonical)
                contains_a = flavor_a in sentence_flavors
                contains_b = flavor_b in sentence_flavors
                if contains_a and contains_b:
                    pair_sentences.append(sentence)
                elif contains_a or contains_b:
                    fallback_sentences.append(sentence)

            if pair_sentences:
                analysis_sentences = pair_sentences
                is_same_sentence_pair = True
                same_sentence_doc_count += 1
            else:
                analysis_sentences = fallback_sentences if fallback_sentences else sentences[:1]
                is_same_sentence_pair = False

            if not analysis_sentences:
                continue

            evidence_doc_count += 1
            doc_positive = False
            doc_negative = False
            doc_role = False

            for sentence in analysis_sentences:
                matched_categories: list[str] = []
                matched_terms: list[str] = []
                has_positive_expression = False
                has_negative_expression = False
                has_role = False

                for group_name, category_name, column_name in category_columns:
                    terms = expression_dictionary[group_name][category_name]
                    valid_terms, negated_terms = find_category_matches(sentence, terms, negations)
                    if not valid_terms:
                        continue
                    category_counter[column_name] += len(valid_terms)
                    matched_categories.append(f"{group_name}:{category_name}")
                    matched_terms.extend(valid_terms)
                    if group_name == "evaluation" and category_name == "positive":
                        has_positive_expression = True
                    if group_name == "evaluation" and category_name == "negative":
                        has_negative_expression = True
                    if negated_terms and group_name == "evaluation" and category_name == "positive":
                        category_counter["evaluation_positive_negated_count"] += len(negated_terms)

                role_terms, _ = find_category_matches(
                    sentence,
                    expression_dictionary["taste_role"],
                    negations,
                )
                if role_terms:
                    matched_categories.append("taste_role")
                    matched_terms.extend(role_terms)
                    has_role = True

                if has_positive_expression:
                    doc_positive = True
                if has_negative_expression:
                    doc_negative = True
                if has_role:
                    doc_role = True

                evidence_rows.append(
                    {
                        "document_id": doc["document_id"],
                        "flavor_a": flavor_a,
                        "flavor_b": flavor_b,
                        "pair_key": pair_key,
                        "sentence": sentence,
                        "matched_categories": "|".join(sorted(set(matched_categories))),
                        "matched_terms": "|".join(sorted(set(matched_terms))),
                        "is_same_sentence_pair": is_same_sentence_pair,
                        "has_positive_expression": has_positive_expression,
                        "has_negative_expression": has_negative_expression,
                        "has_taste_role_explanation": has_role,
                        "review_title": doc["review_title"],
                        "review_url": doc["review_url"],
                    }
                )

            if doc_positive:
                positive_doc_count += 1
            if doc_negative:
                negative_doc_count += 1
            if doc_role:
                role_doc_count += 1

        feature_row = {
            "flavor_a": flavor_a,
            "flavor_b": flavor_b,
            "pair_key": pair_key,
            "pair_count": int(pair_row.pair_count),
            "same_sentence_count": same_sentence_doc_count,
            "same_document_count": len(doc_rows),
            "evidence_document_count": evidence_doc_count,
            "positive_expression_count": positive_doc_count,
            "negative_expression_count": negative_doc_count,
            "positive_document_ratio": positive_doc_count / evidence_doc_count if evidence_doc_count else 0.0,
            "negative_document_ratio": negative_doc_count / evidence_doc_count if evidence_doc_count else 0.0,
            "taste_role_explanation_count": role_doc_count,
            "taste_role_explanation_ratio": role_doc_count / evidence_doc_count if evidence_doc_count else 0.0,
        }
        for _, _, column_name in category_columns:
            feature_row[column_name] = int(category_counter.get(column_name, 0))
        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows)
    evidence_df = pd.DataFrame(evidence_rows)
    if features_df.empty:
        features_df = pd.DataFrame(
            columns=[
                "flavor_a",
                "flavor_b",
                "pair_key",
                "pair_count",
                "same_sentence_count",
                "same_document_count",
                "evidence_document_count",
                "positive_expression_count",
                "negative_expression_count",
                "positive_document_ratio",
                "negative_document_ratio",
                "taste_role_explanation_count",
                "taste_role_explanation_ratio",
            ]
        )
    return features_df, evidence_df


def merge_pair_features(
    pair_df: pd.DataFrame,
    centrality_pair_df: pd.DataFrame,
    expression_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = centrality_pair_df.merge(
        expression_df,
        on=["flavor_a", "flavor_b", "pair_key", "pair_count"],
        how="left",
    )
    fill_zero_cols = [
        column
        for column in merged.columns
        if column.endswith("_count")
        or column.endswith("_ratio")
        or column.startswith("centrality_")
    ]
    for column in fill_zero_cols:
        merged[column] = merged[column].fillna(0.0)
    return merged


def clip_series_upper(series: pd.Series, quantile: float) -> pd.Series:
    if series.empty:
        return series
    upper = series.quantile(quantile)
    return series.clip(upper=upper)


def minmax_normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    minimum = float(series.min())
    maximum = float(series.max())
    if math.isclose(maximum, minimum):
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    return (series - minimum) / (maximum - minimum)


def add_normalized_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    df = pair_df.copy()
    df["support_log1p"] = np.log1p(df["support"])
    df["pair_count_log1p"] = np.log1p(df["pair_count"])
    df["lift_clipped"] = clip_series_upper(df["lift"], 0.99)
    df["normalized_support"] = minmax_normalize(df["support_log1p"])
    df["normalized_pair_count"] = minmax_normalize(df["pair_count_log1p"])
    df["normalized_lift"] = minmax_normalize(df["lift_clipped"])
    df["normalized_centrality_mean"] = minmax_normalize(df["centrality_mean"])
    df["normalized_positive_document_ratio"] = minmax_normalize(df["positive_document_ratio"])
    df["normalized_taste_role_explanation_ratio"] = minmax_normalize(
        df["taste_role_explanation_ratio"]
    )
    return df


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    for column, weight in weights.items():
        score += df[column].fillna(0.0) * float(weight)
    return score


def stable_rank(
    df: pd.DataFrame,
    score_column: str,
    rank_name: str,
) -> pd.DataFrame:
    ranked = df.sort_values(
        by=[score_column, "pair_count", "lift", "flavor_a", "flavor_b"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    ranked[rank_name] = np.arange(1, len(ranked) + 1)
    return ranked


def build_pair_ranking(pair_df: pd.DataFrame) -> pd.DataFrame:
    ranked = pair_df.copy()
    ranked["support_lift_score"] = (
        0.5 * ranked["normalized_support"] + 0.5 * ranked["normalized_lift"]
    )
    ranked["overall_score"] = (
        0.30 * ranked["normalized_support"]
        + 0.25 * ranked["normalized_lift"]
        + 0.20 * ranked["normalized_centrality_mean"]
        + 0.15 * ranked["normalized_positive_document_ratio"]
        + 0.10 * ranked["normalized_taste_role_explanation_ratio"]
    )

    ranked = stable_rank(ranked, "overall_score", "rank_overall")
    ranked = stable_rank(ranked, "support", "rank_support")
    ranked = stable_rank(ranked, "lift", "rank_lift")
    ranked = stable_rank(ranked, "support_lift_score", "rank_support_lift")

    keep_columns = [
        "rank_overall",
        "rank_support",
        "rank_lift",
        "rank_support_lift",
        "flavor_a",
        "flavor_b",
        "pair_key",
        "support",
        "lift",
        "centrality_mean",
        "centrality_max",
        "centrality_geometric_mean",
        "positive_document_ratio",
        "negative_document_ratio",
        "taste_role_explanation_ratio",
        "normalized_support",
        "normalized_lift",
        "normalized_centrality_mean",
        "normalized_positive_document_ratio",
        "normalized_taste_role_explanation_ratio",
        "support_log1p",
        "lift_clipped",
        "support_lift_score",
        "overall_score",
        "pair_count",
        "evidence_document_count",
    ]
    for column in keep_columns:
        if column not in ranked.columns:
            ranked[column] = 0.0
    ranked = ranked[keep_columns + [c for c in ranked.columns if c not in keep_columns]]
    return ranked


def top_pairs_for_ranking(pair_ranking_df: pd.DataFrame, ranking_name: str, top_k: int) -> pd.DataFrame:
    rank_column = {
        "overall": "rank_overall",
        "support": "rank_support",
        "lift": "rank_lift",
        "support_lift": "rank_support_lift",
    }[ranking_name]
    top_df = pair_ranking_df.nsmallest(top_k, rank_column).copy()
    top_df["ranking_source"] = ranking_name
    top_df["ranking_rank"] = top_df[rank_column]
    return top_df


def build_manual_validation_candidates(
    pair_ranking_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    sources = []
    for ranking_name in ("overall", "support", "lift", "support_lift"):
        sources.append(top_pairs_for_ranking(pair_ranking_df, ranking_name, top_k))
    source_df = pd.concat(sources, ignore_index=True)

    aggregated = (
        source_df.groupby("pair_key")
        .agg(
            {
                "flavor_a": "first",
                "flavor_b": "first",
                "rank_overall": "first",
                "rank_support": "first",
                "rank_lift": "first",
                "rank_support_lift": "first",
                "support": "first",
                "lift": "first",
                "overall_score": "first",
                "support_lift_score": "first",
                "pair_count": "first",
                "evidence_document_count": "first",
                "ranking_source": lambda s: "|".join(sorted(set(s))),
            }
        )
        .reset_index()
    )

    context_map = (
        evidence_df.sort_values(by=["is_same_sentence_pair", "document_id"], ascending=[False, True])
        .groupby("pair_key")["sentence"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for row in aggregated.itertuples(index=False):
        contexts = []
        seen = set()
        for sentence in context_map.get(row.pair_key, []):
            if sentence not in seen:
                seen.add(sentence)
                contexts.append(sentence)
            if len(contexts) >= 3:
                break
        rows.append(
            {
                **row._asdict(),
                "representative_context_1": contexts[0] if len(contexts) >= 1 else "",
                "representative_context_2": contexts[1] if len(contexts) >= 2 else "",
                "representative_context_3": contexts[2] if len(contexts) >= 3 else "",
                "mix_relation_label": "",
                "evaluation_label": "",
                "taste_role_label": "",
                "recommendation_validity": "",
                "reviewer_comment": "",
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["rank_overall", "pair_count", "lift", "flavor_a", "flavor_b"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def spearman_rank_correlation(series_a: list[int], series_b: list[int]) -> float | None:
    if len(series_a) < 2 or len(series_b) < 2:
        return None
    arr_a = np.array(series_a, dtype=float)
    arr_b = np.array(series_b, dtype=float)
    if arr_a.std() == 0 or arr_b.std() == 0:
        return None
    rank_a = pd.Series(arr_a).rank(method="average").to_numpy()
    rank_b = pd.Series(arr_b).rank(method="average").to_numpy()
    corr = np.corrcoef(rank_a, rank_b)[0, 1]
    return float(corr)


def compute_sensitivity(pair_ranking_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_df = pair_ranking_df[["pair_key", "flavor_a", "flavor_b", "pair_count", "lift"]].copy()
    for setting_name, weights in DEFAULT_RANKING_SETTINGS.items():
        setting_df[setting_name] = weighted_score(pair_ranking_df, weights)

    ranking_frames = []
    top_lists: dict[str, list[str]] = {}
    rank_maps: dict[str, dict[str, int]] = {}
    for setting_name in DEFAULT_RANKING_SETTINGS:
        temp_df = pair_ranking_df[["pair_key", "flavor_a", "flavor_b", "pair_count", "lift"]].copy()
        temp_df["score"] = setting_df[setting_name]
        temp_df = stable_rank(temp_df, "score", f"rank_{setting_name}")
        temp_df["setting"] = setting_name
        ranking_frames.append(temp_df)
        top_df = temp_df.nsmallest(top_k, f"rank_{setting_name}")
        top_lists[setting_name] = top_df["pair_key"].tolist()
        rank_maps[setting_name] = dict(
            zip(top_df["pair_key"].tolist(), top_df[f"rank_{setting_name}"].tolist())
        )

    comparison_rows = []
    setting_names = list(DEFAULT_RANKING_SETTINGS.keys())
    for idx, left_name in enumerate(setting_names):
        for right_name in setting_names[idx + 1 :]:
            left_set = set(top_lists[left_name])
            right_set = set(top_lists[right_name])
            intersection = sorted(left_set & right_set)
            union = left_set | right_set
            jaccard = len(intersection) / len(union) if union else 0.0
            spearman = None
            if intersection:
                spearman = spearman_rank_correlation(
                    [rank_maps[left_name][pair_key] for pair_key in intersection],
                    [rank_maps[right_name][pair_key] for pair_key in intersection],
                )
            comparison_rows.append(
                {
                    "setting_a": left_name,
                    "setting_b": right_name,
                    "top_k": top_k,
                    "common_candidate_count": len(intersection),
                    "jaccard_topk": jaccard,
                    "spearman_rank_correlation": spearman,
                    "common_pairs": "|".join(intersection),
                }
            )

    detailed_df = pd.concat(ranking_frames, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_rows)
    return detailed_df, comparison_df


def write_centrality_top20_markdown(centrality_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 媒介中心性上位20フレーバー",
        "",
        "| 順位 | フレーバー | weighted_betweenness | betweenness | weighted_degree |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    top_df = centrality_df.head(20)
    for rank, row in enumerate(top_df.itertuples(index=False), 1):
        lines.append(
            f"| {rank} | {row.flavor} | {row.weighted_betweenness_centrality:.6f} | "
            f"{row.betweenness_centrality:.6f} | {row.weighted_degree:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sensitivity_markdown(comparison_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# ランキング感度分析",
        "",
        "初期重み設定3種（バランス型・共起重視・文脈重視）の上位20件比較。",
        "",
        "| Setting A | Setting B | Common | Jaccard | Spearman |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in comparison_df.itertuples(index=False):
        spearman = "" if row.spearman_rank_correlation is None else f"{row.spearman_rank_correlation:.4f}"
        lines.append(
            f"| {row.setting_a} | {row.setting_b} | {row.common_candidate_count} | "
            f"{row.jaccard_topk:.4f} | {spearman} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_extended_summary(
    path: Path,
    docs_df: pd.DataFrame,
    conditioned_df: pd.DataFrame,
    pair_ranking_df: pd.DataFrame,
) -> None:
    lines = [
        "# Extended Analysis Summary",
        "",
        f"- total_reviews: {len(docs_df)}",
        f"- condition_reviews: {len(conditioned_df)}",
        f"- unique_flavors_condition: {len(set(fl for flavors in conditioned_df['normalized_flavors'] for fl in flavors))}",
        f"- pair_count: {len(pair_ranking_df)}",
        "- overall_score weights are initial settings, not experimentally fixed values.",
        "- analysis granularity is normalized flavor pairs, not brand-level pairs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    top_k: int = 20,
) -> None:
    plot_df = df.head(top_k).copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=240)
    ax.barh(plot_df[y], plot_df[x], color="#4472C4")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_score_breakdown_plot(pair_ranking_df: pd.DataFrame, path: Path, top_k: int = 20) -> None:
    plot_df = pair_ranking_df.nsmallest(top_k, "rank_overall").copy()
    plot_df = plot_df.sort_values("rank_overall", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=240)
    components = [
        ("normalized_support", "#4C78A8"),
        ("normalized_lift", "#F58518"),
        ("normalized_centrality_mean", "#54A24B"),
        ("normalized_positive_document_ratio", "#E45756"),
        ("normalized_taste_role_explanation_ratio", "#72B7B2"),
    ]
    left = np.zeros(len(plot_df))
    for column, color in components:
        values = plot_df[column].to_numpy()
        ax.barh(plot_df["pair_key"], values, left=left, color=color, label=column)
        left += values
    ax.set_title("総合ランキング上位20ペアのスコア内訳")
    ax.set_xlabel("正規化スコア")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=240)
    ax.scatter(df[x], df[y], alpha=0.7, color="#4C78A8", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def create_ranking_comparison_plot(pair_ranking_df: pd.DataFrame, path: Path, top_k: int = 20) -> None:
    rankings = [
        ("rank_overall", "overall"),
        ("rank_support", "support"),
        ("rank_lift", "lift"),
        ("rank_support_lift", "support_lift"),
    ]
    candidates = set()
    for rank_column, _ in rankings:
        candidates.update(pair_ranking_df.nsmallest(top_k, rank_column)["pair_key"].tolist())
    plot_df = pair_ranking_df[pair_ranking_df["pair_key"].isin(candidates)].copy()
    plot_df = plot_df.sort_values("rank_overall").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=240)
    y_positions = np.arange(len(plot_df))
    for offset, (rank_column, label) in enumerate(rankings):
        ax.scatter(
            plot_df[rank_column],
            y_positions + offset * 0.05,
            label=label,
            s=30,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["pair_key"])
    ax.invert_yaxis()
    ax.set_xlabel("順位")
    ax.set_title("ランキング方式ごとの上位候補比較")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def manual_validation_has_labels(df: pd.DataFrame) -> bool:
    label_columns = [
        "mix_relation_label",
        "evaluation_label",
        "taste_role_label",
        "recommendation_validity",
    ]
    for column in label_columns:
        if column in df.columns and df[column].fillna("").astype(str).str.strip().ne("").any():
            return True
    return False


def create_manual_validity_plot(summary_df: pd.DataFrame, path: Path) -> None:
    if summary_df.empty:
        return
    plot_df = summary_df[summary_df["metric"] == "recommendation_valid_rate"].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=240)
    width = 0.2
    k_values = sorted(plot_df["k"].unique())
    ranking_names = sorted(plot_df["ranking_name"].unique())
    x_positions = np.arange(len(ranking_names))
    for idx, k_value in enumerate(k_values):
        values = []
        for ranking_name in ranking_names:
            row = plot_df[(plot_df["ranking_name"] == ranking_name) & (plot_df["k"] == k_value)]
            values.append(float(row["value"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x_positions + idx * width, values, width=width, label=f"Top {k_value}")
    ax.set_xticks(x_positions + width)
    ax.set_xticklabels(ranking_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("妥当率")
    ax.set_title("ランキング方式別の推薦妥当率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def compute_manual_validation_summary(
    manual_df: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    rows = []
    ranking_columns = {
        "overall": "rank_overall",
        "support": "rank_support",
        "lift": "rank_lift",
        "support_lift": "rank_support_lift",
    }
    for ranking_name, rank_column in ranking_columns.items():
        ranked_df = manual_df[manual_df[rank_column].notna()].copy()
        ranked_df = ranked_df.sort_values(rank_column)
        for k_value in k_values:
            top_df = ranked_df.head(k_value)
            if top_df.empty:
                continue
            rows.extend(
                [
                    {
                        "ranking_name": ranking_name,
                        "k": k_value,
                        "metric": "explicit_mix_rate",
                        "value": (top_df["mix_relation_label"] == "explicit_mix").mean(),
                    },
                    {
                        "ranking_name": ranking_name,
                        "k": k_value,
                        "metric": "positive_evaluation_rate",
                        "value": (top_df["evaluation_label"] == "positive").mean(),
                    },
                    {
                        "ranking_name": ranking_name,
                        "k": k_value,
                        "metric": "taste_role_explained_rate",
                        "value": (top_df["taste_role_label"] == "explained").mean(),
                    },
                    {
                        "ranking_name": ranking_name,
                        "k": k_value,
                        "metric": "recommendation_valid_rate",
                        "value": (top_df["recommendation_validity"] == "valid").mean(),
                    },
                    {
                        "ranking_name": ranking_name,
                        "k": k_value,
                        "metric": "unclear_rate",
                        "value": (
                            (top_df["mix_relation_label"] == "unclear")
                            | (top_df["evaluation_label"] == "unclear")
                            | (top_df["taste_role_label"] == "unclear")
                            | (top_df["recommendation_validity"] == "unclear")
                        ).mean(),
                    },
                ]
            )
    return pd.DataFrame(rows)


def write_manual_validation_summary_markdown(summary_df: pd.DataFrame, path: Path, has_labels: bool) -> None:
    if not has_labels or summary_df.empty:
        path.write_text("# Manual Validation Summary\n\n未評価のため集計できない。\n", encoding="utf-8")
        return
    lines = [
        "# Manual Validation Summary",
        "",
        "| Ranking | K | Metric | Value |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.ranking_name} | {row.k} | {row.metric} | {float(row.value):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
