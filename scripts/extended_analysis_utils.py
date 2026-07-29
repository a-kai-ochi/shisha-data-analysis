#!/usr/bin/env python3
"""Utilities for extended flavor-pair analysis on normalized flavor cooccurrence."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
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

EXPLICIT_MIX_TERMS = [
    "ミックス",
    "混ぜる",
    "混ぜ",
    "組み合わせる",
    "組み合わせ",
    "加える",
    "加え",
    "足す",
    "足し",
    "入れる",
    "入れ",
]

GENERIC_ROLE_TERMS = {"ミックス", "組み合わせ", "相性", "おすすめ"}
JAPANESE_BOUNDARY_CHARS = set("とのにへがをやもでなだかはばまでよりねよぞさし")

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

MANUAL_VALIDATION_BASE_FIELDS = [
    "mix_relation_label",
    "evaluation_label",
    "taste_role_label",
    "recommendation_validity",
    "semantic_overlap_label",
    "comment",
]

MANUAL_VALIDATION_REVIEWER_PREFIXES = ["reviewer1", "reviewer2"]

MANUAL_VALIDATION_GUIDELINE = """# Manual Validation Guideline

## mix_relation_label

- `explicit_mix`
  実際に2つのフレーバーを混ぜる、加える、組み合わせる記述が明確にある
- `likely_mix`
  配合は明示されていないが、同じミックスとして説明されている可能性が高い
- `co_mention_only`
  同じ文にあるだけで、ミックス関係はない
- `unclear`
  文脈だけでは判定できない

## evaluation_label

- `positive`
  ペアまたはミックス全体への肯定的評価がある
- `neutral`
  配合説明のみで評価はない
- `negative`
  ペアまたはミックス全体への否定的評価がある
- `mixed`
  肯定と否定の両方がある
- `unclear`
  評価対象が単体かペアか分からない

## taste_role_label

- `explained`
  少なくとも一方のフレーバーが、甘さ、清涼感、香り、コク、アクセントなどをどのように与えるか説明されている
- `not_explained`
  組合せの記述のみで役割説明はない
- `unclear`
  役割説明か判断できない

## recommendation_validity

- `valid`
  明示的または可能性の高いミックスで、肯定評価または役割説明がある
- `partially_valid`
  ミックス関係は確認できるが、評価・役割説明が弱い
- `invalid`
  共起のみ、商品名由来、意味的重複、または否定的な候補
- `unclear`
  根拠不足で判断できない

## semantic_overlap_label

- `distinct`
  別のフレーバーとして扱える
- `similar`
  近い意味だが別候補として残しうる
- `duplicate`
  実質的に同じ意味で、候補としての重複が大きい
- `unclear`
  文脈だけでは判断できない

## Reviewer Columns

- 1名評価の場合は、基本列 `mix_relation_label` から `reviewer_comment` までを入力する
- 2名評価の場合は、`reviewer1_*` と `reviewer2_*` の列を使用する
- 未評価の項目は空欄のままにする
"""


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
    pair_ranking_tier1_csv: Path
    pair_ranking_tier2_csv: Path
    pair_ranking_excluded_csv: Path
    excluded_product_name_pairs_csv: Path
    excluded_parent_child_pairs_csv: Path
    manual_validation_candidates_csv: Path
    manual_validation_tier1_csv: Path
    manual_validation_guideline_md: Path
    ranking_sensitivity_csv: Path
    ranking_sensitivity_md: Path
    ranking_before_after_comparison_csv: Path
    ranking_before_after_comparison_md: Path
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
    manual_validation_crosstab_csv: Path
    manual_validation_disagreements_csv: Path


def output_paths(output_dir: Path) -> OutputPaths:
    return OutputPaths(
        flavor_centrality_csv=output_dir / "flavor_centrality.csv",
        flavor_centrality_top20_md=output_dir / "flavor_centrality_top20.md",
        pair_expression_features_csv=output_dir / "pair_expression_features.csv",
        pair_expression_evidence_csv=output_dir / "pair_expression_evidence.csv",
        pair_ranking_csv=output_dir / "pair_ranking.csv",
        pair_ranking_tier1_csv=output_dir / "pair_ranking_tier1.csv",
        pair_ranking_tier2_csv=output_dir / "pair_ranking_tier2.csv",
        pair_ranking_excluded_csv=output_dir / "pair_ranking_excluded.csv",
        excluded_product_name_pairs_csv=output_dir / "excluded_product_name_pairs.csv",
        excluded_parent_child_pairs_csv=output_dir / "excluded_parent_child_pairs.csv",
        manual_validation_candidates_csv=output_dir / "manual_validation_candidates.csv",
        manual_validation_tier1_csv=output_dir / "manual_validation_tier1.csv",
        manual_validation_guideline_md=output_dir / "manual_validation_guideline.md",
        ranking_sensitivity_csv=output_dir / "ranking_sensitivity.csv",
        ranking_sensitivity_md=output_dir / "ranking_sensitivity.md",
        ranking_before_after_comparison_csv=output_dir / "ranking_before_after_comparison.csv",
        ranking_before_after_comparison_md=output_dir / "ranking_before_after_comparison.md",
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
        manual_validation_crosstab_csv=output_dir / "manual_validation_crosstab.csv",
        manual_validation_disagreements_csv=output_dir / "manual_validation_disagreements.csv",
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
                    and prev_char not in JAPANESE_BOUNDARY_CHARS
                ):
                    before_ok = False
            if end < len(text):
                next_char = text[end]
                if (
                    unicodedata.category(next_char) in ("Lo",)
                    and unicodedata.category(text[end - 1]) in ("Lo",)
                    and next_char not in JAPANESE_BOUNDARY_CHARS
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
        title = row.get("レビュータイトル", "")
        summary = row.get("概要", "")
        body = row.get("レビュー本文", "")
        extracted = extract_flavors(body, sorted_patterns, pattern_to_canonical)
        title_flavors = extract_flavors(str(title), sorted_patterns, pattern_to_canonical)
        summary_flavors = extract_flavors(str(summary), sorted_patterns, pattern_to_canonical)
        docs.append(
            {
                "document_id": make_document_id(idx),
                "review_title": title,
                "review_date": row.get("更新日", ""),
                "review_url": row.get("レビューURL", ""),
                "review_summary": summary,
                "review_body": body,
                "normalized_flavors": extracted,
                "title_flavors": sorted(set(title_flavors)),
                "summary_flavors": sorted(set(summary_flavors)),
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


def load_template_sentence_patterns(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def normalize_compare_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\-_/\|・･\.\(\)（）\[\]【】]+", "", normalized)


def compile_template_patterns(patterns: dict[str, list[str]]) -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for regex in patterns.get("regex", []):
        compiled.append((regex, re.compile(regex)))
    for term in patterns.get("contains", []):
        compiled.append((term, re.compile(re.escape(term))))
    return compiled


def detect_template_sentence(
    sentence: str,
    compiled_patterns: list[tuple[str, re.Pattern[str]]],
) -> tuple[bool, str]:
    if not isinstance(sentence, str):
        return False, ""
    for pattern_text, pattern in compiled_patterns:
        if pattern.search(sentence):
            return True, pattern_text
    return False, ""


def detect_explicit_mix_expression(sentence: str, flavor_a: str, flavor_b: str) -> tuple[bool, str]:
    if not isinstance(sentence, str):
        return False, ""
    if flavor_a not in sentence or flavor_b not in sentence:
        return False, ""

    escaped_a = re.escape(flavor_a)
    escaped_b = re.escape(flavor_b)
    separator_patterns = [
        rf"{escaped_a}\s*(?:×|✕|x|X|\+|＋|/|／)\s*{escaped_b}",
        rf"{escaped_b}\s*(?:×|✕|x|X|\+|＋|/|／)\s*{escaped_a}",
    ]
    for pattern in separator_patterns:
        match = re.search(pattern, sentence)
        if match:
            return True, match.group(0)

    phrase_patterns = [
        rf"{escaped_a}.{{0,12}}{escaped_b}.{{0,12}}(?:ミックス|混ぜる|混ぜた|組み合わせる|組み合わせた|組み合わせ|加える|加えた|足す|足した)",
        rf"{escaped_b}.{{0,12}}{escaped_a}.{{0,12}}(?:ミックス|混ぜる|混ぜた|組み合わせる|組み合わせた|組み合わせ|加える|加えた|足す|足した)",
        rf"{escaped_a}と{escaped_b}.{{0,12}}(?:ミックス|混ぜる|混ぜた|組み合わせる|組み合わせた|組み合わせ)",
        rf"{escaped_b}と{escaped_a}.{{0,12}}(?:ミックス|混ぜる|混ぜた|組み合わせる|組み合わせた|組み合わせ)",
        rf"{escaped_a}に{escaped_b}を(?:加える|足す|混ぜる)",
        rf"{escaped_b}に{escaped_a}を(?:加える|足す|混ぜる)",
    ]
    for pattern in phrase_patterns:
        match = re.search(pattern, sentence)
        if match:
            return True, match.group(0)
    return False, ""


def detect_parent_child_pair(
    flavor_a: str,
    flavor_b: str,
) -> tuple[bool, str]:
    key_a = normalize_compare_text(flavor_a)
    key_b = normalize_compare_text(flavor_b)
    if not key_a or not key_b or key_a == key_b:
        return False, ""
    if key_a in key_b:
        return True, f"{flavor_a} is contained in {flavor_b}"
    if key_b in key_a:
        return True, f"{flavor_b} is contained in {flavor_a}"
    return False, ""


def is_negated(text: str, start: int, end: int, negations: list[str]) -> bool:
    window = text[max(0, start - 4) : min(len(text), end + 8)]
    suffix = text[end : min(len(text), end + 8)]
    prefix = text[max(0, start - 4) : start]
    if any(pattern in suffix for pattern in ["しない", "できない", "とは思わない"]):
        return True
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
                continue

        if term.endswith("る") and len(term) >= 2:
            stem = re.escape(term[:-1])
            verb_patterns = [
                rf"{stem}(?:ら)?ない",
                rf"{stem}(?:ら)?なく",
                rf"{stem}(?:ら)?なかった",
                rf"{stem}(?:ら)?ません",
                rf"{stem}とは思わない",
                rf"{stem}しない",
                rf"{stem}できない",
            ]
            for pattern in verb_patterns:
                if re.search(pattern, text):
                    negated_terms.append(term)
                    break

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


def merge_pipe_values(values: list[str]) -> str:
    unique = sorted({value for value in values if value})
    return "|".join(unique)


def role_terms_for_sentence(
    sentence: str,
    expression_dictionary: dict[str, Any],
) -> tuple[bool, str, str, str]:
    action_terms = expression_dictionary.get("taste_role_action_terms", [])
    effect_terms = expression_dictionary.get("taste_role_effect_terms", [])

    action = next((term for term in action_terms if term in sentence), "")
    effect = next((term for term in effect_terms if term in sentence), "")
    if action and effect:
        return True, "action+effect", action, effect
    if action:
        return True, "action", action, ""
    if effect:
        return True, "effect", "", effect
    return False, "", "", ""


def deduplicate_evidence_rows(evidence_df: pd.DataFrame) -> pd.DataFrame:
    if evidence_df.empty:
        return evidence_df.copy()

    group_keys = [
        "document_id",
        "flavor_a",
        "flavor_b",
        "sentence",
        "matched_categories",
        "matched_terms",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in evidence_df.groupby(group_keys, dropna=False, sort=False):
        first = group.iloc[0].to_dict()
        for column in [
            "matched_negated_categories",
            "matched_negated_terms",
            "template_pattern",
            "role_detection_rule",
            "role_action_term",
            "role_effect_term",
            "explicit_mix_text",
        ]:
            if column in group.columns:
                first[column] = merge_pipe_values(group[column].fillna("").tolist())
        for column in [
            "is_same_sentence_pair",
            "has_explicit_mix_expression",
            "is_template_sentence",
            "has_positive_expression",
            "has_negative_expression",
            "has_taste_role_explanation",
            "is_product_name_derived",
        ]:
            if column in group.columns:
                first[column] = bool(group[column].fillna(False).any())
        rows.append(first)
    return pd.DataFrame(rows)


def analyze_sentence_categories(
    sentence: str,
    expression_dictionary: dict[str, Any],
) -> dict[str, Any]:
    negations = expression_dictionary["negations"]
    matched_categories: list[str] = []
    matched_terms: list[str] = []
    matched_negated_categories: list[str] = []
    matched_negated_terms: list[str] = []
    has_positive = False
    has_negative = False

    for group_name in ("taste", "experience", "evaluation"):
        for category_name, terms in expression_dictionary[group_name].items():
            if group_name == "evaluation" and category_name == "negative":
                valid_terms = [term for term in terms if term in sentence]
                negated_terms = []
            else:
                valid_terms, negated_terms = find_category_matches(sentence, terms, negations)
            if valid_terms:
                matched_categories.append(f"{group_name}:{category_name}")
                matched_terms.extend(valid_terms)
            if negated_terms:
                matched_negated_categories.append(f"{group_name}:{category_name}")
                matched_negated_terms.extend(negated_terms)
            if group_name == "evaluation" and category_name == "positive" and valid_terms:
                has_positive = True
            if group_name == "evaluation" and category_name == "negative" and valid_terms:
                has_negative = True
            if group_name == "evaluation" and category_name == "positive" and negated_terms:
                has_negative = True

    return {
        "matched_categories": merge_pipe_values(matched_categories),
        "matched_terms": merge_pipe_values(matched_terms),
        "matched_negated_categories": merge_pipe_values(matched_negated_categories),
        "matched_negated_terms": merge_pipe_values(matched_negated_terms),
        "has_positive_expression": has_positive,
        "has_negative_expression": has_negative,
    }


def aggregate_pair_flags(
    docs_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    sorted_patterns: list[str],
    pattern_to_canonical: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag_rows: list[dict[str, Any]] = []
    excluded_product_rows: list[dict[str, Any]] = []
    title_flavor_map = {
        row.document_id: sorted(set(row.title_flavors) | set(row.summary_flavors))
        for row in docs_df.itertuples(index=False)
    }
    title_map = {
        row.document_id: row.review_title
        for row in docs_df.itertuples(index=False)
    }

    for row in pair_df.itertuples(index=False):
        is_parent_child_pair, parent_child_reason = detect_parent_child_pair(
            row.flavor_a,
            row.flavor_b,
        )
        product_name_doc_ids: list[str] = []
        explicit_doc_ids: list[str] = []
        for doc in docs_df.itertuples(index=False):
            flavors = sorted(set(doc.normalized_flavors))
            if row.flavor_a not in flavors or row.flavor_b not in flavors:
                continue
            title_flavors = title_flavor_map.get(doc.document_id, [])
            if row.flavor_a in title_flavors and row.flavor_b in title_flavors:
                product_name_doc_ids.append(doc.document_id)
            for sentence in split_sentences(doc.review_body):
                sentence_flavors = extract_flavors(sentence, sorted_patterns, pattern_to_canonical)
                if row.flavor_a in sentence_flavors and row.flavor_b in sentence_flavors:
                    has_explicit, _ = detect_explicit_mix_expression(sentence, row.flavor_a, row.flavor_b)
                    if has_explicit:
                        explicit_doc_ids.append(doc.document_id)
                        break

        is_product_name_derived = bool(product_name_doc_ids)
        has_explicit_mix_expression = bool(explicit_doc_ids)
        excluded_as_product_name_pair = is_product_name_derived and not has_explicit_mix_expression
        if excluded_as_product_name_pair:
            for document_id in sorted(set(product_name_doc_ids)):
                excluded_product_rows.append(
                    {
                        "document_id": document_id,
                        "product_name": title_map.get(document_id, ""),
                        "flavor_a": row.flavor_a,
                        "flavor_b": row.flavor_b,
                        "matched_text": title_map.get(document_id, ""),
                        "reason": "product_name_derived_without_explicit_mix",
                    }
                )

        flag_rows.append(
            {
                "pair_key": row.pair_key,
                "flavor_a": row.flavor_a,
                "flavor_b": row.flavor_b,
                "is_product_name_derived": is_product_name_derived,
                "has_explicit_mix_expression": has_explicit_mix_expression,
                "excluded_as_product_name_pair": excluded_as_product_name_pair,
                "is_parent_child_pair": is_parent_child_pair,
                "parent_child_reason": parent_child_reason,
            }
        )

    return pd.DataFrame(flag_rows), pd.DataFrame(excluded_product_rows)


def extract_pair_expression_features_v2(
    docs_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    sorted_patterns: list[str],
    pattern_to_canonical: dict[str, str],
    expression_dictionary: dict[str, Any],
    template_patterns: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compiled_templates = compile_template_patterns(template_patterns)
    pair_lookup = set(pair_df["pair_key"].tolist())
    pair_documents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in docs_df.itertuples(index=False):
        flavors = sorted(set(row.normalized_flavors))
        for flavor_a, flavor_b in combinations(flavors, 2):
            pair_key = f"{flavor_a}||{flavor_b}"
            if pair_key in pair_lookup:
                pair_documents[pair_key].append(row._asdict())

    pair_flag_df, excluded_product_df = aggregate_pair_flags(
        docs_df,
        pair_df,
        sorted_patterns,
        pattern_to_canonical,
    )
    feature_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    excluded_parent_child_rows: list[dict[str, Any]] = []
    flag_map = pair_flag_df.set_index("pair_key").to_dict("index") if not pair_flag_df.empty else {}

    for pair_row in pair_df.itertuples(index=False):
        pair_key = pair_row.pair_key
        flags = flag_map.get(pair_key, {})
        doc_rows = pair_documents.get(pair_key, [])
        flavor_a = pair_row.flavor_a
        flavor_b = pair_row.flavor_b
        document_positive_count = 0
        document_negative_count = 0
        document_role_count = 0
        same_sentence_positive_count = 0
        same_sentence_negative_count = 0
        same_sentence_role_count = 0
        same_sentence_cooccurrence_count = 0
        same_sentence_evidence_document_count = 0
        explicit_mix_count = 0
        template_evidence_count = 0
        positive_evidence_count = 0
        negative_evidence_count = 0
        role_evidence_count = 0

        if flags.get("is_parent_child_pair"):
            excluded_parent_child_rows.append(
                {
                    "flavor_a": flavor_a,
                    "flavor_b": flavor_b,
                    "pair_key": pair_key,
                    "reason": flags.get("parent_child_reason", ""),
                }
            )

        for doc in doc_rows:
            sentences = split_sentences(doc["review_body"])
            same_sentence_rows: list[dict[str, Any]] = []
            fallback_rows: list[dict[str, Any]] = []

            for sentence in sentences:
                sentence_flavors = set(
                    extract_flavors(sentence, sorted_patterns, pattern_to_canonical)
                )
                contains_a = flavor_a in sentence_flavors
                contains_b = flavor_b in sentence_flavors
                has_explicit_mix, explicit_mix_text = detect_explicit_mix_expression(
                    sentence,
                    flavor_a,
                    flavor_b,
                )
                is_template_sentence, template_pattern = detect_template_sentence(
                    sentence,
                    compiled_templates,
                )
                category_info = analyze_sentence_categories(sentence, expression_dictionary)
                has_role, role_rule, role_action_term, role_effect_term = role_terms_for_sentence(
                    sentence,
                    expression_dictionary,
                )

                row_dict = {
                    "document_id": doc["document_id"],
                    "review_title": doc["review_title"],
                    "review_url": doc["review_url"],
                    "product_name": doc["review_title"],
                    "flavor_a": flavor_a,
                    "flavor_b": flavor_b,
                    "pair_key": pair_key,
                    "sentence": sentence,
                    "matched_categories": category_info["matched_categories"],
                    "matched_terms": category_info["matched_terms"],
                    "matched_negated_categories": category_info["matched_negated_categories"],
                    "matched_negated_terms": category_info["matched_negated_terms"],
                    "is_same_sentence_pair": contains_a and contains_b,
                    "has_explicit_mix_expression": has_explicit_mix,
                    "explicit_mix_text": explicit_mix_text,
                    "is_template_sentence": is_template_sentence,
                    "template_pattern": template_pattern,
                    "has_positive_expression": category_info["has_positive_expression"],
                    "has_negative_expression": category_info["has_negative_expression"],
                    "has_taste_role_explanation": has_role,
                    "role_detection_rule": role_rule,
                    "role_action_term": role_action_term,
                    "role_effect_term": role_effect_term,
                    "is_product_name_derived": bool(flags.get("is_product_name_derived", False)),
                    "is_parent_child_pair": bool(flags.get("is_parent_child_pair", False)),
                    "parent_child_reason": flags.get("parent_child_reason", ""),
                }

                if contains_a and contains_b:
                    same_sentence_rows.append(row_dict)
                elif contains_a or contains_b:
                    fallback_rows.append(row_dict)

            if same_sentence_rows:
                same_sentence_cooccurrence_count += 1
                eligible_same_sentence = [
                    row
                    for row in same_sentence_rows
                    if not row["is_template_sentence"]
                ]
                if eligible_same_sentence:
                    same_sentence_evidence_document_count += 1
                    doc_positive = any(row["has_positive_expression"] for row in eligible_same_sentence)
                    doc_negative = any(row["has_negative_expression"] for row in eligible_same_sentence)
                    doc_role = any(row["has_taste_role_explanation"] for row in eligible_same_sentence)
                    if doc_positive:
                        same_sentence_positive_count += 1
                    if doc_negative:
                        same_sentence_negative_count += 1
                    if doc_role:
                        same_sentence_role_count += 1
                    if any(row["has_explicit_mix_expression"] for row in eligible_same_sentence):
                        explicit_mix_count += 1
                    for row in eligible_same_sentence:
                        positive_evidence_count += int(row["has_positive_expression"])
                        negative_evidence_count += int(row["has_negative_expression"])
                        role_evidence_count += int(row["has_taste_role_explanation"])
                template_evidence_count += sum(int(row["is_template_sentence"]) for row in same_sentence_rows)

                doc_level_rows = eligible_same_sentence if eligible_same_sentence else []
                if doc_level_rows:
                    document_positive_count += int(any(row["has_positive_expression"] for row in doc_level_rows))
                    document_negative_count += int(any(row["has_negative_expression"] for row in doc_level_rows))
                    document_role_count += int(any(row["has_taste_role_explanation"] for row in doc_level_rows))

                evidence_rows.extend(same_sentence_rows)
            else:
                non_template_fallback = [
                    row for row in fallback_rows if not row["is_template_sentence"]
                ]
                if non_template_fallback:
                    document_positive_count += int(
                        any(row["has_positive_expression"] for row in non_template_fallback)
                    )
                    document_negative_count += int(
                        any(row["has_negative_expression"] for row in non_template_fallback)
                    )
                    document_role_count += int(
                        any(row["has_taste_role_explanation"] for row in non_template_fallback)
                    )
                template_evidence_count += sum(int(row["is_template_sentence"]) for row in fallback_rows)
                evidence_rows.extend(fallback_rows)

        document_cooccurrence_count = int(pair_row.pair_count)
        feature_rows.append(
            {
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "pair_key": pair_key,
                "pair_count": document_cooccurrence_count,
                "document_cooccurrence_count": document_cooccurrence_count,
                "same_sentence_cooccurrence_count": same_sentence_cooccurrence_count,
                "same_sentence_evidence_document_count": same_sentence_evidence_document_count,
                "explicit_mix_count": explicit_mix_count,
                "document_level_positive_count": document_positive_count,
                "document_level_negative_count": document_negative_count,
                "document_level_role_count": document_role_count,
                "same_sentence_positive_count": same_sentence_positive_count,
                "same_sentence_negative_count": same_sentence_negative_count,
                "same_sentence_role_count": same_sentence_role_count,
                "document_level_positive_ratio": (
                    document_positive_count / document_cooccurrence_count if document_cooccurrence_count else 0.0
                ),
                "document_level_negative_ratio": (
                    document_negative_count / document_cooccurrence_count if document_cooccurrence_count else 0.0
                ),
                "document_level_role_ratio": (
                    document_role_count / document_cooccurrence_count if document_cooccurrence_count else 0.0
                ),
                "same_sentence_positive_ratio": (
                    same_sentence_positive_count / same_sentence_evidence_document_count
                    if same_sentence_evidence_document_count
                    else 0.0
                ),
                "same_sentence_negative_ratio": (
                    same_sentence_negative_count / same_sentence_evidence_document_count
                    if same_sentence_evidence_document_count
                    else 0.0
                ),
                "same_sentence_role_ratio": (
                    same_sentence_role_count / same_sentence_evidence_document_count
                    if same_sentence_evidence_document_count
                    else 0.0
                ),
                "template_evidence_count": template_evidence_count,
                "positive_evidence_count": positive_evidence_count,
                "negative_evidence_count": negative_evidence_count,
                "role_evidence_count": role_evidence_count,
                "is_product_name_derived": bool(flags.get("is_product_name_derived", False)),
                "has_explicit_mix_expression": bool(flags.get("has_explicit_mix_expression", False)),
                "excluded_as_product_name_pair": bool(flags.get("excluded_as_product_name_pair", False)),
                "is_parent_child_pair": bool(flags.get("is_parent_child_pair", False)),
                "parent_child_reason": flags.get("parent_child_reason", ""),
            }
        )

    features_df = pd.DataFrame(feature_rows)
    evidence_before_dedup = pd.DataFrame(evidence_rows)
    before_count = len(evidence_before_dedup)
    evidence_df = deduplicate_evidence_rows(evidence_before_dedup)
    after_count = len(evidence_df)
    if not features_df.empty:
        features_df["evidence_rows_before_dedup"] = before_count
        features_df["evidence_rows_after_dedup"] = after_count
        features_df["evidence_duplicates_removed"] = before_count - after_count
    if excluded_product_df.empty:
        excluded_product_df = pd.DataFrame(
            columns=["document_id", "product_name", "flavor_a", "flavor_b", "matched_text", "reason"]
        )
    excluded_parent_child_df = pd.DataFrame(excluded_parent_child_rows).drop_duplicates().reset_index(drop=True)
    if excluded_parent_child_df.empty:
        excluded_parent_child_df = pd.DataFrame(
            columns=["flavor_a", "flavor_b", "pair_key", "reason"]
        )
    return (
        features_df,
        evidence_df,
        excluded_product_df,
        excluded_parent_child_df,
    )


def merge_pair_features_v2(
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
    for column in [
        "is_product_name_derived",
        "has_explicit_mix_expression",
        "excluded_as_product_name_pair",
        "is_parent_child_pair",
    ]:
        if column in merged.columns:
            merged[column] = merged[column].fillna(False)
    if "parent_child_reason" in merged.columns:
        merged["parent_child_reason"] = merged["parent_child_reason"].fillna("")
    return merged


def add_normalized_features_v2(
    pair_df: pd.DataFrame,
    alpha: float = 3.0,
    min_pair_count_for_context: int = 3,
    min_same_sentence_docs_for_context: int = 2,
) -> pd.DataFrame:
    df = pair_df.copy()
    df["support_log1p"] = np.log1p(df["support"])
    df["pair_count_log1p"] = np.log1p(df["pair_count"])
    df["lift_clipped"] = clip_series_upper(df["lift"], 0.99)
    df["normalized_support"] = minmax_normalize(df["support_log1p"])
    df["normalized_pair_count"] = minmax_normalize(df["pair_count_log1p"])
    df["normalized_lift"] = minmax_normalize(df["lift_clipped"])
    df["normalized_centrality_mean"] = minmax_normalize(df["centrality_mean"])

    total_same_sentence_docs = float(df["same_sentence_evidence_document_count"].sum())
    global_positive_rate = (
        float(df["same_sentence_positive_count"].sum()) / total_same_sentence_docs
        if total_same_sentence_docs
        else 0.0
    )
    global_negative_rate = (
        float(df["same_sentence_negative_count"].sum()) / total_same_sentence_docs
        if total_same_sentence_docs
        else 0.0
    )
    global_role_rate = (
        float(df["same_sentence_role_count"].sum()) / total_same_sentence_docs
        if total_same_sentence_docs
        else 0.0
    )

    df["smoothed_positive_ratio"] = (
        df["same_sentence_positive_count"] + alpha * global_positive_rate
    ) / (df["same_sentence_evidence_document_count"] + alpha)
    df["smoothed_negative_ratio"] = (
        df["same_sentence_negative_count"] + alpha * global_negative_rate
    ) / (df["same_sentence_evidence_document_count"] + alpha)
    df["smoothed_role_ratio"] = (
        df["same_sentence_role_count"] + alpha * global_role_rate
    ) / (df["same_sentence_evidence_document_count"] + alpha)

    df["context_score_eligible"] = (
        (df["pair_count"] >= min_pair_count_for_context)
        & (df["same_sentence_evidence_document_count"] >= min_same_sentence_docs_for_context)
    )
    df["smoothed_positive_ratio_for_score"] = np.where(
        df["context_score_eligible"],
        df["smoothed_positive_ratio"],
        0.0,
    )
    df["smoothed_negative_ratio_for_score"] = np.where(
        df["context_score_eligible"],
        df["smoothed_negative_ratio"],
        0.0,
    )
    df["smoothed_role_ratio_for_score"] = np.where(
        df["context_score_eligible"],
        df["smoothed_role_ratio"],
        0.0,
    )

    df["normalized_smoothed_positive_ratio"] = minmax_normalize(
        df["smoothed_positive_ratio_for_score"]
    )
    df["normalized_smoothed_negative_ratio"] = minmax_normalize(
        df["smoothed_negative_ratio_for_score"]
    )
    df["normalized_smoothed_role_ratio"] = minmax_normalize(
        df["smoothed_role_ratio_for_score"]
    )
    df["confidence_factor"] = np.minimum(1.0, df["pair_count"] / 5.0)
    df["adjusted_lift"] = df["normalized_lift"] * df["confidence_factor"]
    return df


def build_pair_ranking_v2(pair_df: pd.DataFrame) -> pd.DataFrame:
    ranked = pair_df.copy()
    ranked["support_lift_score"] = 0.5 * ranked["normalized_support"] + 0.5 * ranked["adjusted_lift"]
    ranked["overall_score_v2"] = (
        0.30 * ranked["normalized_support"]
        + 0.25 * ranked["adjusted_lift"]
        + 0.15 * ranked["normalized_centrality_mean"]
        + 0.15 * ranked["normalized_smoothed_positive_ratio"]
        + 0.10 * ranked["normalized_smoothed_role_ratio"]
        - 0.05 * ranked["normalized_smoothed_negative_ratio"]
    )

    ranked = stable_rank(ranked, "overall_score_v2", "rank_overall")
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
        "pair_count",
        "document_cooccurrence_count",
        "same_sentence_cooccurrence_count",
        "same_sentence_evidence_document_count",
        "explicit_mix_count",
        "support",
        "lift",
        "normalized_lift",
        "adjusted_lift",
        "confidence_factor",
        "centrality_mean",
        "smoothed_positive_ratio",
        "smoothed_negative_ratio",
        "smoothed_role_ratio",
        "overall_score_v2",
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
        "context_score_eligible",
    ]
    keep_columns += [
        "normalized_support",
        "normalized_pair_count",
        "normalized_centrality_mean",
        "normalized_smoothed_positive_ratio",
        "normalized_smoothed_negative_ratio",
        "normalized_smoothed_role_ratio",
        "support_log1p",
        "pair_count_log1p",
        "lift_clipped",
        "evidence_rows_before_dedup",
        "evidence_rows_after_dedup",
        "evidence_duplicates_removed",
    ]
    for column in keep_columns:
        if column not in ranked.columns:
            ranked[column] = 0.0
    return ranked[keep_columns + [c for c in ranked.columns if c not in keep_columns]]


def split_ranking_tiers_v2(pair_ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pair_ranking_df.copy()
    base_excluded = (
        df["excluded_as_product_name_pair"]
        | df["is_parent_child_pair"]
        | (df["flavor_a"] == df["flavor_b"])
        | (df["same_sentence_evidence_document_count"] == 0)
    )
    tier1_mask = (
        ~base_excluded
        & (df["pair_count"] >= 3)
        & (df["same_sentence_evidence_document_count"] >= 2)
    )
    tier2_mask = (
        ~base_excluded
        & (df["pair_count"] >= 2)
    )

    excluded_reason = np.select(
        [
            df["excluded_as_product_name_pair"],
            df["is_parent_child_pair"],
            df["flavor_a"] == df["flavor_b"],
            df["same_sentence_evidence_document_count"] == 0,
        ],
        [
            "product_name_derived",
            "parent_child_pair",
            "self_loop",
            "no_same_sentence_context",
        ],
        default="other",
    )
    df["ranking_tier"] = np.select(
        [tier1_mask, tier2_mask],
        ["Tier1", "Tier2"],
        default="Excluded",
    )
    df["excluded_reason"] = excluded_reason

    tier1_df = df[tier1_mask].copy()
    tier2_df = df[tier2_mask].copy()
    excluded_df = df[~tier2_mask].copy()
    return tier1_df, tier2_df, excluded_df


def representative_context_rows_for_pair(
    evidence_df: pd.DataFrame,
    pair_key: str,
    max_contexts: int = 3,
) -> list[dict[str, str]]:
    pair_evidence = evidence_df[evidence_df["pair_key"] == pair_key].copy()
    if pair_evidence.empty:
        return []
    eligible = pair_evidence[
        pair_evidence["is_same_sentence_pair"] | pair_evidence["has_explicit_mix_expression"]
    ].copy()
    if eligible.empty:
        return []
    eligible = eligible[~eligible["is_template_sentence"]].copy()
    if eligible.empty:
        return []
    eligible["expression_count"] = (
        eligible["has_positive_expression"].astype(int)
        + eligible["has_negative_expression"].astype(int)
        + eligible["has_taste_role_explanation"].astype(int)
    )
    eligible["has_any_expression"] = eligible["expression_count"] > 0
    eligible = eligible.sort_values(
        by=[
            "is_same_sentence_pair",
            "has_explicit_mix_expression",
            "has_any_expression",
            "expression_count",
            "document_id",
            "sentence",
        ],
        ascending=[False, False, False, False, True, True],
    )

    contexts: list[dict[str, str]] = []
    seen_sentences: set[str] = set()
    seen_documents: set[str] = set()
    for row in eligible.itertuples(index=False):
        sentence = str(row.sentence)
        document_id = str(row.document_id)
        if sentence in seen_sentences or document_id in seen_documents:
            continue
        seen_sentences.add(sentence)
        seen_documents.add(document_id)
        contexts.append({"sentence": sentence, "document_id": document_id})
        if len(contexts) >= max_contexts:
            break

    if len(contexts) < max_contexts:
        for row in eligible.itertuples(index=False):
            sentence = str(row.sentence)
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            contexts.append({"sentence": sentence, "document_id": str(row.document_id)})
            if len(contexts) >= max_contexts:
                break
    return contexts


def representative_contexts_for_pair(
    evidence_df: pd.DataFrame,
    pair_key: str,
) -> list[str]:
    return [item["sentence"] for item in representative_context_rows_for_pair(evidence_df, pair_key)]


def build_manual_validation_candidates_v2(
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
                "pair_count": "first",
                "same_sentence_evidence_document_count": "first",
                "explicit_mix_count": "first",
                "support": "first",
                "adjusted_lift": "first",
                "centrality_mean": "first",
                "smoothed_positive_ratio": "first",
                "smoothed_negative_ratio": "first",
                "smoothed_role_ratio": "first",
                "template_evidence_count": "first",
                "positive_evidence_count": "first",
                "negative_evidence_count": "first",
                "role_evidence_count": "first",
                "ranking_tier": "first",
                "is_product_name_derived": "first",
                "is_parent_child_pair": "first",
                "ranking_source": lambda s: "|".join(sorted(set(s))),
            }
        )
        .reset_index()
    )

    rows = []
    for row in aggregated.itertuples(index=False):
        contexts = representative_context_rows_for_pair(evidence_df, row.pair_key)
        rows.append(
            {
                **row._asdict(),
                "representative_context_1": contexts[0]["sentence"] if len(contexts) >= 1 else "",
                "representative_context_2": contexts[1]["sentence"] if len(contexts) >= 2 else "",
                "representative_context_3": contexts[2]["sentence"] if len(contexts) >= 3 else "",
                "mix_relation_label": "",
                "evaluation_label": "",
                "taste_role_label": "",
                "recommendation_validity": "",
                "reviewer_comment": "",
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["rank_overall", "pair_count", "adjusted_lift", "flavor_a", "flavor_b"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def build_manual_validation_tier1_dataframe(
    pair_ranking_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    tier1_df = pair_ranking_df.copy()
    if "ranking_tier" in tier1_df.columns:
        tier1_df = tier1_df[tier1_df["ranking_tier"] == "Tier1"].copy()
    tier1_df = tier1_df.sort_values(
        by=["rank_overall", "pair_count", "adjusted_lift", "flavor_a", "flavor_b"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for row in tier1_df.itertuples(index=False):
        contexts = representative_context_rows_for_pair(evidence_df, row.pair_key, max_contexts=3)
        record: dict[str, Any] = {
            "rank": int(row.rank_overall),
            "pair_key": row.pair_key,
            "flavor_a": row.flavor_a,
            "flavor_b": row.flavor_b,
            "pair_count": int(row.pair_count),
            "same_sentence_evidence_document_count": int(row.same_sentence_evidence_document_count),
            "support": float(row.support),
            "lift": float(row.lift),
            "adjusted_lift": float(row.adjusted_lift),
            "centrality_mean": float(row.centrality_mean),
            "smoothed_positive_ratio": float(row.smoothed_positive_ratio),
            "smoothed_negative_ratio": float(row.smoothed_negative_ratio),
            "smoothed_role_ratio": float(row.smoothed_role_ratio),
            "overall_score_v2": float(row.overall_score_v2),
            "context_1": contexts[0]["sentence"] if len(contexts) >= 1 else "",
            "context_2": contexts[1]["sentence"] if len(contexts) >= 2 else "",
            "context_3": contexts[2]["sentence"] if len(contexts) >= 3 else "",
            "context_1_document_id": contexts[0]["document_id"] if len(contexts) >= 1 else "",
            "context_2_document_id": contexts[1]["document_id"] if len(contexts) >= 2 else "",
            "context_3_document_id": contexts[2]["document_id"] if len(contexts) >= 3 else "",
            "mix_relation_label": "",
            "evaluation_label": "",
            "taste_role_label": "",
            "recommendation_validity": "",
            "semantic_overlap_label": "",
            "reviewer_comment": "",
        }
        for prefix in MANUAL_VALIDATION_REVIEWER_PREFIXES:
            record[f"{prefix}_mix_relation_label"] = ""
            record[f"{prefix}_evaluation_label"] = ""
            record[f"{prefix}_taste_role_label"] = ""
            record[f"{prefix}_recommendation_validity"] = ""
            record[f"{prefix}_semantic_overlap_label"] = ""
            record[f"{prefix}_comment"] = ""
        rows.append(record)

    columns = [
        "rank",
        "pair_key",
        "flavor_a",
        "flavor_b",
        "pair_count",
        "same_sentence_evidence_document_count",
        "support",
        "lift",
        "adjusted_lift",
        "centrality_mean",
        "smoothed_positive_ratio",
        "smoothed_negative_ratio",
        "smoothed_role_ratio",
        "overall_score_v2",
        "context_1",
        "context_2",
        "context_3",
        "context_1_document_id",
        "context_2_document_id",
        "context_3_document_id",
        "mix_relation_label",
        "evaluation_label",
        "taste_role_label",
        "recommendation_validity",
        "semantic_overlap_label",
        "reviewer_comment",
        "reviewer1_mix_relation_label",
        "reviewer1_evaluation_label",
        "reviewer1_taste_role_label",
        "reviewer1_recommendation_validity",
        "reviewer1_semantic_overlap_label",
        "reviewer1_comment",
        "reviewer2_mix_relation_label",
        "reviewer2_evaluation_label",
        "reviewer2_taste_role_label",
        "reviewer2_recommendation_validity",
        "reviewer2_semantic_overlap_label",
        "reviewer2_comment",
    ]
    return pd.DataFrame(rows, columns=columns)


def write_manual_validation_guideline(path: Path) -> None:
    path.write_text(MANUAL_VALIDATION_GUIDELINE + "\n", encoding="utf-8")


def compute_sensitivity_v2(pair_ranking_df: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_weights = {
        "SettingA_balanced": {
            "normalized_support": 0.30,
            "adjusted_lift": 0.25,
            "normalized_centrality_mean": 0.15,
            "normalized_smoothed_positive_ratio": 0.15,
            "normalized_smoothed_role_ratio": 0.10,
            "normalized_smoothed_negative_ratio": -0.05,
        },
        "SettingB_cooccurrence": {
            "normalized_support": 0.40,
            "adjusted_lift": 0.30,
            "normalized_centrality_mean": 0.10,
            "normalized_smoothed_positive_ratio": 0.10,
            "normalized_smoothed_role_ratio": 0.05,
            "normalized_smoothed_negative_ratio": -0.05,
        },
        "SettingC_context": {
            "normalized_support": 0.20,
            "adjusted_lift": 0.15,
            "normalized_centrality_mean": 0.15,
            "normalized_smoothed_positive_ratio": 0.25,
            "normalized_smoothed_role_ratio": 0.20,
            "normalized_smoothed_negative_ratio": -0.05,
        },
    }

    ranking_frames = []
    top_lists: dict[str, list[str]] = {}
    rank_maps: dict[str, dict[str, int]] = {}
    setting_rows = []
    for setting_name, weights in setting_weights.items():
        temp_df = pair_ranking_df.copy()
        temp_df["score"] = weighted_score(temp_df, weights)
        temp_df = stable_rank(temp_df, "score", f"rank_{setting_name}")
        temp_df["setting"] = setting_name
        ranking_frames.append(temp_df)
        top_df = temp_df.nsmallest(top_k, f"rank_{setting_name}")
        top_lists[setting_name] = top_df["pair_key"].tolist()
        rank_maps[setting_name] = dict(
            zip(top_df["pair_key"].tolist(), top_df[f"rank_{setting_name}"].tolist())
        )
        setting_rows.append(
            {
                "row_type": "setting",
                "setting": setting_name,
                "top_k": top_k,
                "product_name_pair_count_topk": int(top_df["excluded_as_product_name_pair"].sum()),
                "parent_child_pair_count_topk": int(top_df["is_parent_child_pair"].sum()),
                "pair_count_2_topk": int((top_df["pair_count"] == 2).sum()),
                "same_sentence_evidence_0_topk": int(
                    (top_df["same_sentence_evidence_document_count"] == 0).sum()
                ),
            }
        )

    comparison_rows = []
    setting_names = list(setting_weights.keys())
    for idx, left_name in enumerate(setting_names):
        for right_name in setting_names[idx + 1 :]:
            left_set = set(top_lists[left_name])
            right_set = set(top_lists[right_name])
            intersection = sorted(left_set & right_set)
            union = left_set | right_set
            comparison_rows.append(
                {
                    "row_type": "comparison",
                    "setting_a": left_name,
                    "setting_b": right_name,
                    "top_k": top_k,
                    "common_candidate_count": len(intersection),
                    "jaccard_topk": len(intersection) / len(union) if union else 0.0,
                    "spearman_rank_correlation": spearman_rank_correlation(
                        [rank_maps[left_name][pair_key] for pair_key in intersection],
                        [rank_maps[right_name][pair_key] for pair_key in intersection],
                    )
                    if len(intersection) >= 2
                    else None,
                    "common_pairs": "|".join(intersection),
                }
            )

    detail_df = pd.concat(ranking_frames, ignore_index=True)
    summary_df = pd.concat(
        [pd.DataFrame(comparison_rows), pd.DataFrame(setting_rows)],
        ignore_index=True,
        sort=False,
    )
    return detail_df, summary_df


def write_sensitivity_markdown_v2(summary_df: pd.DataFrame, path: Path) -> None:
    comparison_df = summary_df[summary_df["row_type"] == "comparison"].copy()
    setting_df = summary_df[summary_df["row_type"] == "setting"].copy()
    lines = [
        "# ランキング感度分析 v2",
        "",
        "## Settingごとの上位20件監査",
        "",
        "| Setting | pair_count=2 | same_sentence=0 | product_name | parent_child |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in setting_df.itertuples(index=False):
        lines.append(
            f"| {row.setting} | {int(row.pair_count_2_topk)} | "
            f"{int(row.same_sentence_evidence_0_topk)} | {int(row.product_name_pair_count_topk)} | "
            f"{int(row.parent_child_pair_count_topk)} |"
        )
    lines.extend(
        [
            "",
            "## Setting間比較",
            "",
            "| Setting A | Setting B | Common | Jaccard | Spearman |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_df.itertuples(index=False):
        spearman = "" if row.spearman_rank_correlation is None else f"{row.spearman_rank_correlation:.4f}"
        lines.append(
            f"| {row.setting_a} | {row.setting_b} | {int(row.common_candidate_count)} | "
            f"{row.jaccard_topk:.4f} | {spearman} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_before_after_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    top_k: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_top = before_df.nsmallest(top_k, "rank_overall")[["pair_key", "rank_overall"]].rename(
        columns={"rank_overall": "before_rank"}
    )
    after_top = after_df.nsmallest(top_k, "rank_overall")[["pair_key", "rank_overall"]].rename(
        columns={"rank_overall": "after_rank"}
    )
    merged = before_top.merge(after_top, on="pair_key", how="outer")
    merged["status"] = np.select(
        [
            merged["before_rank"].notna() & merged["after_rank"].notna(),
            merged["before_rank"].notna(),
        ],
        ["common", "removed"],
        default="new",
    )
    common = merged[merged["status"] == "common"].copy()
    common_keys = common["pair_key"].tolist()
    summary = {
        "top_k": top_k,
        "common_candidate_count": int((merged["status"] == "common").sum()),
        "removed_candidate_count": int((merged["status"] == "removed").sum()),
        "new_candidate_count": int((merged["status"] == "new").sum()),
        "jaccard_topk": (
            int((merged["status"] == "common").sum())
            / int(len(set(before_top["pair_key"]) | set(after_top["pair_key"])))
            if len(set(before_top["pair_key"]) | set(after_top["pair_key"])) > 0
            else 0.0
        ),
        "spearman_rank_correlation": spearman_rank_correlation(
            common["before_rank"].astype(int).tolist(),
            common["after_rank"].astype(int).tolist(),
        )
        if len(common_keys) >= 2
        else None,
    }
    return merged.sort_values(["status", "before_rank", "after_rank"], na_position="last"), summary


def write_before_after_comparison_markdown(
    comparison_df: pd.DataFrame,
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Ranking Before/After Comparison",
        "",
        f"- top_k: {summary['top_k']}",
        f"- common_candidate_count: {summary['common_candidate_count']}",
        f"- removed_candidate_count: {summary['removed_candidate_count']}",
        f"- new_candidate_count: {summary['new_candidate_count']}",
        f"- jaccard_topk: {summary['jaccard_topk']:.4f}",
        f"- spearman_rank_correlation: {summary['spearman_rank_correlation'] if summary['spearman_rank_correlation'] is not None else 'NA'}",
        "",
        "| pair_key | before_rank | after_rank | status |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in comparison_df.itertuples(index=False):
        before_rank = "" if pd.isna(row.before_rank) else int(row.before_rank)
        after_rank = "" if pd.isna(row.after_rank) else int(row.after_rank)
        lines.append(f"| {row.pair_key} | {before_rank} | {after_rank} | {row.status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_extended_summary_v2(
    path: Path,
    docs_df: pd.DataFrame,
    conditioned_df: pd.DataFrame,
    tier1_df: pd.DataFrame,
    tier2_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    excluded_product_df: pd.DataFrame,
    excluded_parent_child_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> None:
    lines = [
        "# Extended Analysis Summary v2",
        "",
        f"- total_reviews: {len(docs_df)}",
        f"- condition_reviews: {len(conditioned_df)}",
        f"- tier1_candidate_count: {len(tier1_df)}",
        f"- tier2_candidate_count: {len(tier2_df)}",
        f"- excluded_candidate_count: {len(excluded_df)}",
        f"- excluded_product_name_pair_rows: {len(excluded_product_df)}",
        f"- excluded_parent_child_pair_rows: {len(excluded_parent_child_df)}",
        f"- template_evidence_rows: {int(evidence_df.get('is_template_sentence', pd.Series(dtype=bool)).sum()) if not evidence_df.empty else 0}",
        f"- negative_evidence_rows: {int(evidence_df.get('has_negative_expression', pd.Series(dtype=bool)).sum()) if not evidence_df.empty else 0}",
        "- document-level cooccurrence is retained for Support/Lift.",
        "- context scoring uses same-sentence evidence and explicit mix context only.",
        "- product-name-derived pairs and parent-child pairs are excluded from standard ranking.",
        "- smoothed context ratios and confidence-adjusted lift are used in v2 ranking.",
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
    if "normalized_smoothed_positive_ratio" in plot_df.columns:
        components = [
            ("normalized_support", "#4C78A8"),
            ("adjusted_lift", "#F58518"),
            ("normalized_centrality_mean", "#54A24B"),
            ("normalized_smoothed_positive_ratio", "#E45756"),
            ("normalized_smoothed_role_ratio", "#72B7B2"),
            ("normalized_smoothed_negative_ratio", "#B279A2"),
        ]
    else:
        components = [
            ("normalized_support", "#4C78A8"),
            ("normalized_lift", "#F58518"),
            ("normalized_centrality_mean", "#54A24B"),
            ("normalized_positive_document_ratio", "#E45756"),
            ("normalized_taste_role_explanation_ratio", "#72B7B2"),
        ]
    left = np.zeros(len(plot_df))
    for column, color in components:
        if column not in plot_df.columns:
            continue
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


MANUAL_VALIDATION_BASE_COLUMNS = {
    "mix_relation_label": "mix_relation_label",
    "evaluation_label": "evaluation_label",
    "taste_role_label": "taste_role_label",
    "recommendation_validity": "recommendation_validity",
    "semantic_overlap_label": "semantic_overlap_label",
    "comment": "reviewer_comment",
}


def _manual_label_columns_for_source(source: str) -> dict[str, str]:
    if source == "base":
        return MANUAL_VALIDATION_BASE_COLUMNS
    return {
        "mix_relation_label": f"{source}_mix_relation_label",
        "evaluation_label": f"{source}_evaluation_label",
        "taste_role_label": f"{source}_taste_role_label",
        "recommendation_validity": f"{source}_recommendation_validity",
        "semantic_overlap_label": f"{source}_semantic_overlap_label",
        "comment": f"{source}_comment",
    }


def _series_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column].fillna("").astype(str).str.strip()


def _has_any_labels_for_source(df: pd.DataFrame, source: str) -> bool:
    columns = _manual_label_columns_for_source(source)
    label_keys = [
        "mix_relation_label",
        "evaluation_label",
        "taste_role_label",
        "recommendation_validity",
        "semantic_overlap_label",
    ]
    for key in label_keys:
        if _series_text(df, columns[key]).ne("").any():
            return True
    return False


def manual_validation_has_labels(df: pd.DataFrame) -> bool:
    if _has_any_labels_for_source(df, "base"):
        return True
    return any(_has_any_labels_for_source(df, prefix) for prefix in MANUAL_VALIDATION_REVIEWER_PREFIXES)


def manual_validation_primary_source(df: pd.DataFrame) -> str | None:
    if _has_any_labels_for_source(df, "base"):
        return "base"
    for prefix in MANUAL_VALIDATION_REVIEWER_PREFIXES:
        if _has_any_labels_for_source(df, prefix):
            return prefix
    return None


def _scope_label(k_value: int | None) -> str:
    return "all" if k_value is None else f"top_{k_value}"


def _scope_dataframe(df: pd.DataFrame, k_value: int | None) -> pd.DataFrame:
    sorted_df = df.sort_values("rank").reset_index(drop=True)
    if k_value is None:
        return sorted_df
    return sorted_df.head(k_value).copy()


def _rate_from_labels(values: pd.Series, accepted: set[str]) -> tuple[float, int]:
    normalized = values.fillna("").astype(str).str.strip()
    eligible = normalized.ne("")
    count = int(eligible.sum())
    if count == 0:
        return math.nan, 0
    return float(normalized[eligible].isin(accepted).mean()), count


def _unclear_rate(scope_df: pd.DataFrame, columns: dict[str, str]) -> tuple[float, int]:
    label_columns = [
        columns["mix_relation_label"],
        columns["evaluation_label"],
        columns["taste_role_label"],
        columns["recommendation_validity"],
        columns["semantic_overlap_label"],
    ]
    normalized = pd.DataFrame({col: _series_text(scope_df, col) for col in label_columns})
    eligible = normalized.ne("").any(axis=1)
    count = int(eligible.sum())
    if count == 0:
        return math.nan, 0
    unclear_mask = normalized.eq("unclear").any(axis=1)
    return float(unclear_mask[eligible].mean()), count


def _summary_metric_row(
    scope: str,
    metric: str,
    value: float | int | None,
    labeled_count: int,
    section: str = "scope_metric",
    label: str = "",
) -> dict[str, Any]:
    return {
        "section": section,
        "scope": scope,
        "metric": metric,
        "label": label,
        "value": value,
        "n_labeled": labeled_count,
    }


def _spearman_if_possible(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return math.nan
    x_valid = x[valid]
    y_valid = y[valid]
    if x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return math.nan
    return spearman_rank_correlation(x_valid.tolist(), y_valid.tolist())


def _build_binary_feature_relation(
    scope_df: pd.DataFrame,
    feature_col: str,
    label_values: pd.Series,
    positive_labels: set[str],
    analysis_name: str,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    crosstabs: list[dict[str, Any]] = []
    normalized_labels = label_values.fillna("").astype(str).str.strip()
    eligible = normalized_labels.ne("")
    if not eligible.any():
        return rows, crosstabs

    binary = normalized_labels.isin(positive_labels)
    feature = pd.to_numeric(scope_df[feature_col], errors="coerce")
    eligible_feature = eligible & feature.notna()
    if not eligible_feature.any():
        return rows, crosstabs

    subset = pd.DataFrame(
        {
            "feature": feature[eligible_feature],
            "manual_binary": binary[eligible_feature],
        }
    )
    for label_name, label_mask in [("manual_no", ~subset["manual_binary"]), ("manual_yes", subset["manual_binary"])]:
        label_df = subset[label_mask]
        if label_df.empty:
            continue
        rows.append(
            _summary_metric_row(
                scope=scope,
                metric=f"{analysis_name}_feature_mean",
                value=float(label_df["feature"].mean()),
                labeled_count=int(len(label_df)),
                section="feature_relation",
                label=label_name,
            )
        )
        rows.append(
            _summary_metric_row(
                scope=scope,
                metric=f"{analysis_name}_feature_median",
                value=float(label_df["feature"].median()),
                labeled_count=int(len(label_df)),
                section="feature_relation",
                label=label_name,
            )
        )

    rows.append(
        _summary_metric_row(
            scope=scope,
            metric=f"{analysis_name}_spearman",
            value=_spearman_if_possible(subset["feature"], subset["manual_binary"].astype(float)),
            labeled_count=int(len(subset)),
            section="feature_relation",
            label="manual_binary",
        )
    )

    subset["auto_positive"] = subset["feature"] > 0
    for manual_label, auto_label in product([False, True], [False, True]):
        count = int(((subset["manual_binary"] == manual_label) & (subset["auto_positive"] == auto_label)).sum())
        crosstabs.append(
            {
                "analysis": analysis_name,
                "scope": scope,
                "row_label": "manual_yes" if manual_label else "manual_no",
                "column_label": "auto_positive" if auto_label else "auto_zero",
                "count": count,
            }
        )
    return rows, crosstabs


def _score_bin_labels(values: pd.Series) -> pd.Series:
    valid = pd.to_numeric(values, errors="coerce")
    if valid.notna().sum() < 3:
        return pd.Series([""] * len(values), index=values.index, dtype="object")
    ranks = valid.rank(method="average", pct=True)
    bins = pd.Series(index=values.index, dtype="object")
    bins[ranks <= (1 / 3)] = "low"
    bins[(ranks > (1 / 3)) & (ranks <= (2 / 3))] = "mid"
    bins[ranks > (2 / 3)] = "high"
    bins[valid.isna()] = ""
    return bins.fillna("")


def _build_validity_feature_relation(
    scope_df: pd.DataFrame,
    label_values: pd.Series,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    crosstabs: list[dict[str, Any]] = []
    normalized = label_values.fillna("").astype(str).str.strip()
    feature = pd.to_numeric(scope_df["overall_score_v2"], errors="coerce")
    eligible = normalized.ne("") & feature.notna()
    if not eligible.any():
        return rows, crosstabs

    subset = pd.DataFrame(
        {
            "validity": normalized[eligible],
            "overall_score_v2": feature[eligible],
        }
    )
    for label_name in sorted(subset["validity"].unique()):
        label_df = subset[subset["validity"] == label_name]
        rows.append(
            _summary_metric_row(
                scope=scope,
                metric="recommendation_validity_score_mean",
                value=float(label_df["overall_score_v2"].mean()),
                labeled_count=int(len(label_df)),
                section="feature_relation",
                label=label_name,
            )
        )
        rows.append(
            _summary_metric_row(
                scope=scope,
                metric="recommendation_validity_score_median",
                value=float(label_df["overall_score_v2"].median()),
                labeled_count=int(len(label_df)),
                section="feature_relation",
                label=label_name,
            )
        )

    ordinal = subset["validity"].map({"invalid": 0, "partially_valid": 1, "valid": 2})
    rows.append(
        _summary_metric_row(
            scope=scope,
            metric="recommendation_validity_score_spearman",
            value=_spearman_if_possible(subset["overall_score_v2"], ordinal.astype(float)),
            labeled_count=int(ordinal.notna().sum()),
            section="feature_relation",
            label="ordinal_validity",
        )
    )

    score_bins = _score_bin_labels(subset["overall_score_v2"])
    for validity_label in sorted(subset["validity"].unique()):
        for score_bin in ["low", "mid", "high"]:
            count = int(((subset["validity"] == validity_label) & (score_bins == score_bin)).sum())
            crosstabs.append(
                {
                    "analysis": "recommendation_validity_vs_overall_score",
                    "scope": scope,
                    "row_label": validity_label,
                    "column_label": score_bin,
                    "count": count,
                }
            )
    return rows, crosstabs


def compute_manual_validation_agreement(manual_df: pd.DataFrame) -> pd.DataFrame:
    if not (_has_any_labels_for_source(manual_df, "reviewer1") and _has_any_labels_for_source(manual_df, "reviewer2")):
        return pd.DataFrame(
            [
                {
                    "field": "",
                    "comparable_count": 0,
                    "simple_agreement": math.nan,
                    "cohen_kappa": math.nan,
                    "status": "評価者間一致は未計算",
                }
            ]
        )

    rows = []
    for field in [
        "mix_relation_label",
        "evaluation_label",
        "taste_role_label",
        "recommendation_validity",
        "semantic_overlap_label",
    ]:
        col1 = _manual_label_columns_for_source("reviewer1")[field]
        col2 = _manual_label_columns_for_source("reviewer2")[field]
        values1 = _series_text(manual_df, col1)
        values2 = _series_text(manual_df, col2)
        comparable = values1.ne("") & values2.ne("")
        count = int(comparable.sum())
        if count == 0:
            rows.append(
                {
                    "field": field,
                    "comparable_count": 0,
                    "simple_agreement": math.nan,
                    "cohen_kappa": math.nan,
                    "status": "評価者間一致は未計算",
                }
            )
            continue
        comparable1 = values1[comparable].tolist()
        comparable2 = values2[comparable].tolist()
        simple_agreement = float(np.mean([a == b for a, b in zip(comparable1, comparable2)]))
        categories = sorted(set(comparable1) | set(comparable2))
        p1 = {label: comparable1.count(label) / count for label in categories}
        p2 = {label: comparable2.count(label) / count for label in categories}
        expected = sum(p1[label] * p2[label] for label in categories)
        kappa = math.nan if math.isclose(expected, 1.0) else (simple_agreement - expected) / (1 - expected)
        rows.append(
            {
                "field": field,
                "comparable_count": count,
                "simple_agreement": simple_agreement,
                "cohen_kappa": kappa,
                "status": "computed",
            }
        )
    return pd.DataFrame(rows)


def compute_manual_validation_disagreements(manual_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "pair_key",
        "flavor_a",
        "flavor_b",
        "context_1",
        "context_2",
        "context_3",
        "reviewer1_mix_relation_label",
        "reviewer2_mix_relation_label",
        "reviewer1_evaluation_label",
        "reviewer2_evaluation_label",
        "reviewer1_taste_role_label",
        "reviewer2_taste_role_label",
        "reviewer1_recommendation_validity",
        "reviewer2_recommendation_validity",
        "reviewer1_semantic_overlap_label",
        "reviewer2_semantic_overlap_label",
        "reviewer1_comment",
        "reviewer2_comment",
        "disagreement_fields",
    ]
    if not (_has_any_labels_for_source(manual_df, "reviewer1") and _has_any_labels_for_source(manual_df, "reviewer2")):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    field_names = [
        "mix_relation_label",
        "evaluation_label",
        "taste_role_label",
        "recommendation_validity",
        "semantic_overlap_label",
    ]
    for row in manual_df.itertuples(index=False):
        differing_fields: list[str] = []
        for field in field_names:
            col1 = f"reviewer1_{field}"
            col2 = f"reviewer2_{field}"
            value1 = str(getattr(row, col1, "") or "").strip()
            value2 = str(getattr(row, col2, "") or "").strip()
            if value1 and value2 and value1 != value2:
                differing_fields.append(field)
        if not differing_fields:
            continue
        rows.append(
            {
                "rank": getattr(row, "rank", ""),
                "pair_key": getattr(row, "pair_key", ""),
                "flavor_a": getattr(row, "flavor_a", ""),
                "flavor_b": getattr(row, "flavor_b", ""),
                "context_1": getattr(row, "context_1", ""),
                "context_2": getattr(row, "context_2", ""),
                "context_3": getattr(row, "context_3", ""),
                "reviewer1_mix_relation_label": getattr(row, "reviewer1_mix_relation_label", ""),
                "reviewer2_mix_relation_label": getattr(row, "reviewer2_mix_relation_label", ""),
                "reviewer1_evaluation_label": getattr(row, "reviewer1_evaluation_label", ""),
                "reviewer2_evaluation_label": getattr(row, "reviewer2_evaluation_label", ""),
                "reviewer1_taste_role_label": getattr(row, "reviewer1_taste_role_label", ""),
                "reviewer2_taste_role_label": getattr(row, "reviewer2_taste_role_label", ""),
                "reviewer1_recommendation_validity": getattr(row, "reviewer1_recommendation_validity", ""),
                "reviewer2_recommendation_validity": getattr(row, "reviewer2_recommendation_validity", ""),
                "reviewer1_semantic_overlap_label": getattr(row, "reviewer1_semantic_overlap_label", ""),
                "reviewer2_semantic_overlap_label": getattr(row, "reviewer2_semantic_overlap_label", ""),
                "reviewer1_comment": getattr(row, "reviewer1_comment", ""),
                "reviewer2_comment": getattr(row, "reviewer2_comment", ""),
                "disagreement_fields": "|".join(differing_fields),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_manual_validation_outputs(
    manual_df: pd.DataFrame,
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    primary_source = manual_validation_primary_source(manual_df)
    summary_rows: list[dict[str, Any]] = []
    crosstab_rows: list[dict[str, Any]] = []
    if primary_source is None:
        return (
            pd.DataFrame(columns=["section", "scope", "metric", "label", "value", "n_labeled"]),
            pd.DataFrame(columns=["analysis", "scope", "row_label", "column_label", "count"]),
            compute_manual_validation_agreement(manual_df),
            compute_manual_validation_disagreements(manual_df),
            None,
        )

    columns = _manual_label_columns_for_source(primary_source)
    scopes: list[tuple[str, pd.DataFrame]] = [("all", _scope_dataframe(manual_df, None))]
    scopes.extend((_scope_label(k_value), _scope_dataframe(manual_df, k_value)) for k_value in k_values)

    for scope, scope_df in scopes:
        summary_rows.append(
            _summary_metric_row(
                scope=scope,
                metric="candidate_count",
                value=int(len(scope_df)),
                labeled_count=int(len(scope_df)),
            )
        )

        explicit_rate, n_mix = _rate_from_labels(_series_text(scope_df, columns["mix_relation_label"]), {"explicit_mix"})
        explicit_likely_rate, _ = _rate_from_labels(
            _series_text(scope_df, columns["mix_relation_label"]),
            {"explicit_mix", "likely_mix"},
        )
        positive_rate, n_eval = _rate_from_labels(
            _series_text(scope_df, columns["evaluation_label"]),
            {"positive", "mixed"},
        )
        negative_rate, _ = _rate_from_labels(
            _series_text(scope_df, columns["evaluation_label"]),
            {"negative", "mixed"},
        )
        role_rate, n_role = _rate_from_labels(
            _series_text(scope_df, columns["taste_role_label"]),
            {"explained"},
        )
        valid_rate, n_valid = _rate_from_labels(
            _series_text(scope_df, columns["recommendation_validity"]),
            {"valid"},
        )
        valid_partial_rate, _ = _rate_from_labels(
            _series_text(scope_df, columns["recommendation_validity"]),
            {"valid", "partially_valid"},
        )
        overlap_rate, n_overlap = _rate_from_labels(
            _series_text(scope_df, columns["semantic_overlap_label"]),
            {"similar", "duplicate"},
        )
        unclear_rate, n_unclear = _unclear_rate(scope_df, columns)

        summary_rows.extend(
            [
                _summary_metric_row(scope, "explicit_mix_rate", explicit_rate, n_mix),
                _summary_metric_row(scope, "explicit_or_likely_mix_rate", explicit_likely_rate, n_mix),
                _summary_metric_row(scope, "positive_rate", positive_rate, n_eval),
                _summary_metric_row(scope, "negative_rate", negative_rate, n_eval),
                _summary_metric_row(scope, "role_explained_rate", role_rate, n_role),
                _summary_metric_row(scope, "recommendation_valid_rate", valid_rate, n_valid),
                _summary_metric_row(scope, "valid_or_partially_valid_rate", valid_partial_rate, n_valid),
                _summary_metric_row(scope, "semantic_overlap_similar_or_duplicate_rate", overlap_rate, n_overlap),
                _summary_metric_row(scope, "unclear_rate", unclear_rate, n_unclear),
            ]
        )

        relation_rows, relation_crosstabs = _build_binary_feature_relation(
            scope_df,
            "smoothed_positive_ratio",
            _series_text(scope_df, columns["evaluation_label"]),
            {"positive", "mixed"},
            "smoothed_positive_ratio_vs_manual_positive",
            scope,
        )
        summary_rows.extend(relation_rows)
        crosstab_rows.extend(relation_crosstabs)

        relation_rows, relation_crosstabs = _build_binary_feature_relation(
            scope_df,
            "smoothed_negative_ratio",
            _series_text(scope_df, columns["evaluation_label"]),
            {"negative", "mixed"},
            "smoothed_negative_ratio_vs_manual_negative",
            scope,
        )
        summary_rows.extend(relation_rows)
        crosstab_rows.extend(relation_crosstabs)

        relation_rows, relation_crosstabs = _build_binary_feature_relation(
            scope_df,
            "smoothed_role_ratio",
            _series_text(scope_df, columns["taste_role_label"]),
            {"explained"},
            "smoothed_role_ratio_vs_manual_role",
            scope,
        )
        summary_rows.extend(relation_rows)
        crosstab_rows.extend(relation_crosstabs)

        validity_rows, validity_crosstabs = _build_validity_feature_relation(
            scope_df,
            _series_text(scope_df, columns["recommendation_validity"]),
            scope,
        )
        summary_rows.extend(validity_rows)
        crosstab_rows.extend(validity_crosstabs)

    agreement_df = compute_manual_validation_agreement(manual_df)
    disagreements_df = compute_manual_validation_disagreements(manual_df)
    summary_df = pd.DataFrame(summary_rows)
    crosstab_df = pd.DataFrame(crosstab_rows)
    return summary_df, crosstab_df, agreement_df, disagreements_df, primary_source


def write_manual_validation_summary_markdown_v2(
    summary_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    path: Path,
    primary_source: str | None,
) -> None:
    if primary_source is None or summary_df.empty:
        path.write_text("# Manual Validation Summary\n\n未評価のため集計できない。\n", encoding="utf-8")
        return

    lines = [
        "# Manual Validation Summary",
        "",
        f"- primary_label_source: `{primary_source}`",
        "",
        "## Scope Metrics",
        "",
        "| Scope | Metric | Value | Labeled N |",
        "| --- | --- | ---: | ---: |",
    ]
    scope_df = summary_df[summary_df["section"] == "scope_metric"].copy()
    for row in scope_df.itertuples(index=False):
        value = "" if pd.isna(row.value) else (
            f"{float(row.value):.4f}" if isinstance(row.value, (float, np.floating)) else str(row.value)
        )
        lines.append(f"| {row.scope} | {row.metric} | {value} | {row.n_labeled} |")

    feature_df = summary_df[summary_df["section"] == "feature_relation"].copy()
    if not feature_df.empty:
        lines.extend(
            [
                "",
                "## Feature Relations",
                "",
                "| Scope | Metric | Label | Value | N |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in feature_df.itertuples(index=False):
            value = "" if pd.isna(row.value) else f"{float(row.value):.4f}"
            lines.append(f"| {row.scope} | {row.metric} | {row.label} | {value} | {row.n_labeled} |")

    lines.extend(["", "## Inter-Rater Agreement", ""])
    if agreement_df.empty or (
        "status" in agreement_df.columns
        and agreement_df["status"].eq("評価者間一致は未計算").all()
    ):
        lines.append("評価者間一致は未計算。")
    else:
        lines.extend(
            [
                "| Field | Comparable N | Simple Agreement | Cohen's Kappa |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in agreement_df.itertuples(index=False):
            simple = "" if pd.isna(row.simple_agreement) else f"{float(row.simple_agreement):.4f}"
            kappa = "" if pd.isna(row.cohen_kappa) else f"{float(row.cohen_kappa):.4f}"
            lines.append(f"| {row.field} | {row.comparable_count} | {simple} | {kappa} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
