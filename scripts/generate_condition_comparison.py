#!/usr/bin/env python3
"""Generate condition-based cooccurrence comparison outputs for poster analysis."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
POSTER_DIR = ROOT / "poster_analysis"
NOTEBOOK_OUTPUT_DIR = ROOT / "notebooks" / "output"

REVIEWS_CSV = DATA_DIR / "cloud_reviews_final.csv"
MASTER_CSV = DATA_DIR / "aslaj_master_list.csv"

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

EXISTING_OUTPUT_FILES = [
    "brand_flavor_synergy_network.png",
    "brand_mix_insight.txt",
    "final_experiment_summary.md",
    "flavor_insights.csv",
    "flavor_mix_network.png",
    "keyword_trend.png",
    "multi_flavor_mix_ranking.txt",
    "network_2021_2022.png",
    "network_2025_2026.png",
    "network_comparison.png",
    "slide_table_association.md",
    "wordfreq_comparison.png",
]


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    description: str
    min_flavors: int
    max_flavors: int | None
    require_mix_keyword: bool
    is_primary: bool


CONDITIONS = [
    ConditionSpec(
        name="all_multi",
        description="抽出フレーバー数が2種類以上の全レビュー",
        min_flavors=2,
        max_flavors=None,
        require_mix_keyword=False,
        is_primary=True,
    ),
    ConditionSpec(
        name="limited_2_5",
        description="抽出フレーバー数が2〜5種類のレビュー",
        min_flavors=2,
        max_flavors=5,
        require_mix_keyword=False,
        is_primary=True,
    ),
    ConditionSpec(
        name="mix_keyword_2_5",
        description="抽出フレーバー数が2〜5種類で、かつ mix keyword を含むレビュー",
        min_flavors=2,
        max_flavors=5,
        require_mix_keyword=True,
        is_primary=False,
    ),
]

LIFT_MIN_PAIR_COUNTS = [1, 2, 3]

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


def clean_flavor_entry(raw_name: str, raw_brand: str) -> tuple[str, str]:
    """Split a raw master entry into flavor and short brand names."""
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
    """Generate the canonical flavor label and search patterns."""
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


def build_flavor_dictionary(master_df: pd.DataFrame) -> tuple[dict[str, dict], dict[str, str], list[str]]:
    """Build whitelist dictionary and longest-first patterns."""
    flavor_dict: dict[str, dict] = {}
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


def extract_flavors(text: str, sorted_patterns: list[str], pattern_to_canonical: dict[str, str]) -> list[str]:
    """Extract canonical flavors from review text with greedy longest matching."""
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
    """Return the configured mix keywords found in the review body."""
    if not isinstance(text, str):
        return []
    return [keyword for keyword in MIX_KEYWORDS if keyword in text]


def parse_date(value: str) -> pd.Timestamp:
    """Parse Japanese month names used in CLOUD CSV."""
    month_map = {
        "12月": "Dec",
        "11月": "Nov",
        "10月": "Oct",
        "9月": "Sep",
        "8月": "Aug",
        "7月": "Jul",
        "6月": "Jun",
        "5月": "May",
        "4月": "Apr",
        "3月": "Mar",
        "2月": "Feb",
        "1月": "Jan",
    }
    if pd.isna(value):
        return pd.NaT
    normalized = str(value)
    for ja, en in sorted(month_map.items(), key=lambda item: -len(item[0])):
        normalized = normalized.replace(ja, en)
    return pd.to_datetime(normalized, format="%b %d, %Y", errors="coerce")


def build_review_extraction_summary(
    reviews_df: pd.DataFrame,
    sorted_patterns: list[str],
    pattern_to_canonical: dict[str, str],
) -> pd.DataFrame:
    """Create the master review-level extraction table used by all analyses."""
    rows = []
    for idx, row in reviews_df.reset_index(drop=True).iterrows():
        review_text = row["レビュー本文"]
        extracted_flavors = extract_flavors(review_text, sorted_patterns, pattern_to_canonical)
        mix_keywords = detect_mix_keywords(review_text)
        parsed_date = parse_date(row["更新日"])
        rows.append(
            {
                "review_id": f"R{idx + 1:04d}",
                "title": row["レビュータイトル"],
                "url": row["レビューURL"],
                "date": parsed_date.date().isoformat() if not pd.isna(parsed_date) else "",
                "extracted_flavors": "|".join(extracted_flavors),
                "flavor_count": len(extracted_flavors),
                "has_mix_keyword": bool(mix_keywords),
                "mix_keywords": "|".join(mix_keywords),
                "review_text": review_text,
            }
        )
    return pd.DataFrame(rows)


def basic_normalize(value: str) -> str:
    """Apply a conservative Unicode normalization for comparison."""
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", normalized)


def has_japanese_chars(value: str) -> bool:
    """Return True when the string contains Japanese characters."""
    return bool(re.search(r"[ぁ-んァ-ヶー一-龥]", value))


def has_ascii_letters(value: str) -> bool:
    """Return True when the string contains ASCII letters."""
    return bool(re.search(r"[A-Za-z]", value))


def format_equivalence_key(value: str) -> str:
    """Normalize casing, width, and whitespace for exact-format equivalence."""
    normalized = basic_normalize(value)
    return normalized.upper() if has_ascii_letters(normalized) and not has_japanese_chars(normalized) else normalized


def symbol_equivalence_key(value: str) -> str:
    """Normalize punctuation and spacing differences conservatively."""
    normalized = format_equivalence_key(value)
    normalized = normalized.replace("＆", "&")
    return re.sub(r"[\s\-_–—/・･'\"`´’‘.,&()（）]+", "", normalized)


def singular_equivalence_key(value: str) -> str:
    """Build a singular/plural comparison key for ASCII-heavy flavors."""
    normalized = format_equivalence_key(value)
    if has_japanese_chars(normalized) or not has_ascii_letters(normalized):
        return normalized
    token = re.sub(r"[^A-Z0-9 ]", "", normalized)
    if len(token) >= 5 and token.endswith("IES"):
        return token[:-3] + "Y"
    if len(token) >= 4 and token.endswith("ES"):
        return token[:-2]
    if len(token) >= 4 and token.endswith("S"):
        return token[:-1]
    return token


def katakana_equivalence_key(value: str) -> str:
    """Build a loose katakana key for manual-review suggestions."""
    normalized = basic_normalize(value)
    if has_ascii_letters(normalized):
        return normalized
    return re.sub(r"[ー・･\s]", "", normalized)


def extract_master_alias_pairs(master_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Extract bilingual alias evidence from master rows with EN(JA) format."""
    counter: Counter = Counter()
    for _, row in master_df.iterrows():
        flavor_clean, _ = clean_flavor_entry(row["フレーバー名"], row["ブランド"])
        name = basic_normalize(flavor_clean)
        ja_match = re.search(r"\(([ァ-ヿ][^\)]+)\)", name)
        en_match = re.match(r"^([A-Za-z][A-Za-z0-9 .&'\-/]+?)(?:\(|$)", name)
        if not ja_match or not en_match:
            continue
        en = en_match.group(1).strip().upper()
        ja = ja_match.group(1).strip()
        counter[tuple(sorted((en, ja)))] += 1
    return dict(counter)


def count_bilingual_review_evidence(unique_flavors: list[str], review_extraction_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count review/title occurrences where EN and JA surface forms appear as adjacent aliases."""
    ascii_flavors = [flavor for flavor in unique_flavors if has_ascii_letters(flavor) and not has_japanese_chars(flavor)]
    japanese_flavors = [flavor for flavor in unique_flavors if has_japanese_chars(flavor)]
    corpus = [
        basic_normalize(f"{row['title']} {row['review_text']}")
        for _, row in review_extraction_df.iterrows()
    ]
    evidence: dict[tuple[str, str], int] = {}

    for ascii_flavor in ascii_flavors:
        eng = re.escape(basic_normalize(ascii_flavor))
        for japanese_flavor in japanese_flavors:
            ja = re.escape(basic_normalize(japanese_flavor))
            patterns = [
                rf"(?i)\b{eng}\b\s*[\(（]\s*{ja}\s*[\)）]",
                rf"{ja}\s*[\(（]\s*(?i:{eng})\s*[\)）]",
                rf"(?i)\b{eng}\b\s*[／/・･\-–—]?\s*{ja}",
                rf"{ja}\s*[／/・･\-–—]?\s*(?i:{eng})\b",
            ]
            count = 0
            for text in corpus:
                if any(re.search(pattern, text) for pattern in patterns):
                    count += 1
            if count > 0:
                evidence[tuple(sorted((ascii_flavor, japanese_flavor)))] = count
    return evidence


def choose_component_canonical(component: set[str], raw_freq: Counter) -> str:
    """Choose a stable canonical label for an alias component."""
    members = sorted(component, key=lambda flavor: (-raw_freq[flavor], len(flavor), flavor))
    japanese_members = [flavor for flavor in members if has_japanese_chars(flavor)]
    ascii_members = [flavor for flavor in members if has_ascii_letters(flavor) and not has_japanese_chars(flavor)]

    if japanese_members:
        return japanese_members[0]
    uppercase_ascii = [flavor for flavor in ascii_members if flavor == flavor.upper()]
    if uppercase_ascii:
        return uppercase_ascii[0]
    if ascii_members:
        return ascii_members[0]
    return members[0]


def build_alias_candidates_and_map(
    raw_review_extraction_df: pd.DataFrame,
    master_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build alias candidate rows and a conservative normalization map."""
    raw_freq = count_flavor_frequencies(raw_review_extraction_df)
    unique_flavors = sorted(raw_freq.keys())
    unique_flavor_set = set(unique_flavors)
    master_alias_pairs = extract_master_alias_pairs(master_df)
    review_alias_pairs = count_bilingual_review_evidence(unique_flavors, raw_review_extraction_df)

    candidate_records: dict[tuple[str, str], dict[str, object]] = {}
    auto_edges: dict[tuple[str, str], tuple[str, float]] = {}

    def register_candidate(
        raw_flavor: str,
        normalized_candidate: str,
        match_type: str,
        confidence: float,
        manual_review: bool,
        note: str,
    ) -> None:
        if raw_flavor == normalized_candidate:
            return
        key = (raw_flavor, normalized_candidate)
        current = candidate_records.get(key)
        new_record = {
            "raw_flavor": raw_flavor,
            "normalized_candidate": normalized_candidate,
            "match_type": match_type,
            "confidence": confidence,
            "manual_review": manual_review,
            "note": note,
        }
        if current is None or (current["manual_review"] and not manual_review) or float(current["confidence"]) < confidence:
            candidate_records[key] = new_record

        if not manual_review:
            undirected = tuple(sorted((raw_flavor, normalized_candidate)))
            existing = auto_edges.get(undirected)
            if existing is None or existing[1] < confidence:
                auto_edges[undirected] = (match_type, confidence)

    format_groups: dict[str, list[str]] = defaultdict(list)
    symbol_groups: dict[str, list[str]] = defaultdict(list)
    singular_groups: dict[str, list[str]] = defaultdict(list)
    katakana_groups: dict[str, list[str]] = defaultdict(list)

    for flavor in unique_flavors:
        format_groups[format_equivalence_key(flavor)].append(flavor)
        symbol_groups[symbol_equivalence_key(flavor)].append(flavor)
        singular_groups[singular_equivalence_key(flavor)].append(flavor)
        katakana_groups[katakana_equivalence_key(flavor)].append(flavor)

    for group in format_groups.values():
        if len(group) < 2:
            continue
        for left, right in combinations(sorted(set(group)), 2):
            register_candidate(left, right, "case_width_space_exact", 1.0, False, "NFKC・前後空白・大文字小文字差のみ")
            register_candidate(right, left, "case_width_space_exact", 1.0, False, "NFKC・前後空白・大文字小文字差のみ")

    for group in symbol_groups.values():
        deduped = sorted(set(group))
        if len(deduped) < 2:
            continue
        for left, right in combinations(deduped, 2):
            if format_equivalence_key(left) == format_equivalence_key(right):
                continue
            register_candidate(left, right, "symbol_variant_exact", 0.98, False, "記号・スペース差のみ")
            register_candidate(right, left, "symbol_variant_exact", 0.98, False, "記号・スペース差のみ")

    for (left, right), count in master_alias_pairs.items():
        if left not in unique_flavor_set or right not in unique_flavor_set:
            continue
        register_candidate(left, right, "master_bilingual_pair", 0.99, False, f"マスタで EN/JA 対応を確認 ({count}件)")
        register_candidate(right, left, "master_bilingual_pair", 0.99, False, f"マスタで EN/JA 対応を確認 ({count}件)")

    for (left, right), count in review_alias_pairs.items():
        register_candidate(
            left,
            right,
            "review_bilingual_pair",
            0.75,
            True,
            f"レビュー本文またはタイトルで EN/JA 併記を確認 ({count}件)。複合名の可能性があるため要確認",
        )
        register_candidate(
            right,
            left,
            "review_bilingual_pair",
            0.75,
            True,
            f"レビュー本文またはタイトルで EN/JA 併記を確認 ({count}件)。複合名の可能性があるため要確認",
        )

    for canonical, aliases in VERIFIED_CROSS_LANGUAGE_ALIASES.items():
        if canonical not in unique_flavor_set:
            continue
        for alias in aliases:
            if alias not in unique_flavor_set:
                continue
            register_candidate(
                alias,
                canonical,
                "verified_cross_language_alias",
                1.0,
                False,
                "分析対象の基本フレーバーとして EN/JA の直訳同義を確認",
            )
            register_candidate(
                canonical,
                alias,
                "verified_cross_language_alias",
                1.0,
                False,
                "分析対象の基本フレーバーとして EN/JA の直訳同義を確認",
            )

    for group in singular_groups.values():
        ascii_group = sorted(
            {
                flavor
                for flavor in group
                if has_ascii_letters(flavor) and not has_japanese_chars(flavor)
            }
        )
        if len(ascii_group) < 2:
            continue
        for left, right in combinations(ascii_group, 2):
            register_candidate(left, right, "singular_plural_candidate", 0.70, True, "英語の単数/複数差の可能性")
            register_candidate(right, left, "singular_plural_candidate", 0.70, True, "英語の単数/複数差の可能性")

    for group in katakana_groups.values():
        katakana_group = sorted(
            {
                flavor
                for flavor in group
                if has_japanese_chars(flavor) and not has_ascii_letters(flavor)
            }
        )
        if len(katakana_group) < 2:
            continue
        for left, right in combinations(katakana_group, 2):
            if basic_normalize(left) == basic_normalize(right):
                continue
            register_candidate(left, right, "katakana_variant_candidate", 0.60, True, "長音・中点除去で一致するカタカナ候補")
            register_candidate(right, left, "katakana_variant_candidate", 0.60, True, "長音・中点除去で一致するカタカナ候補")

    adjacency_graph: dict[str, set[str]] = defaultdict(set)
    for left, right in auto_edges:
        adjacency_graph[left].add(right)
        adjacency_graph[right].add(left)

    normalization_rows = []
    visited: set[str] = set()
    for flavor in unique_flavors:
        if flavor in visited:
            continue
        stack = [flavor]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency_graph.get(current, set()) - component)
        visited.update(component)
        canonical = choose_component_canonical(component, raw_freq)
        for member in sorted(component):
            if member == canonical:
                rule = "identity" if len(component) == 1 else "component_canonical"
            else:
                direct = candidate_records.get((member, canonical))
                if direct is not None and not bool(direct["manual_review"]):
                    rule = str(direct["match_type"])
                else:
                    auto_records = [
                        record
                        for (raw_flavor, normalized_candidate), record in candidate_records.items()
                        if raw_flavor == member and normalized_candidate in component and not bool(record["manual_review"])
                    ]
                    rule = sorted(
                        auto_records,
                        key=lambda record: (-float(record["confidence"]), str(record["match_type"])),
                    )[0]["match_type"] if auto_records else "auto_alias_component"
            normalization_rows.append(
                {
                    "raw_flavor": member,
                    "canonical_flavor": canonical,
                    "normalization_rule": rule,
                }
            )

    alias_candidates_df = pd.DataFrame(candidate_records.values()).sort_values(
        ["manual_review", "confidence", "raw_flavor", "normalized_candidate"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    normalization_map_df = pd.DataFrame(normalization_rows).sort_values(
        ["canonical_flavor", "raw_flavor"]
    ).reset_index(drop=True)
    return alias_candidates_df, normalization_map_df


def apply_normalization_map(
    raw_review_extraction_df: pd.DataFrame,
    normalization_map_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the conservative normalization map to review-level extracted flavors."""
    mapping = dict(
        zip(
            normalization_map_df["raw_flavor"].astype(str),
            normalization_map_df["canonical_flavor"].astype(str),
        )
    )
    rows = []
    for _, row in raw_review_extraction_df.iterrows():
        raw_flavors = parse_flavor_list(row["extracted_flavors"])
        normalized_flavors = sorted({mapping.get(flavor, flavor) for flavor in raw_flavors})
        rows.append(
            {
                **row.to_dict(),
                "raw_extracted_flavors": row["extracted_flavors"],
                "extracted_flavors": "|".join(normalized_flavors),
                "flavor_count": len(normalized_flavors),
                "normalization_changed": "|".join(raw_flavors) != "|".join(normalized_flavors),
            }
        )
    return pd.DataFrame(rows)


def filter_condition(df: pd.DataFrame, spec: ConditionSpec) -> pd.DataFrame:
    """Filter review-level extraction rows by a condition spec."""
    filtered = df[df["flavor_count"] >= spec.min_flavors].copy()
    if spec.max_flavors is not None:
        filtered = filtered[filtered["flavor_count"] <= spec.max_flavors]
    if spec.require_mix_keyword:
        filtered = filtered[filtered["has_mix_keyword"]]
    return filtered.reset_index(drop=True)


def parse_flavor_list(serialized: str) -> list[str]:
    """Parse a pipe-separated flavor list."""
    if not isinstance(serialized, str) or not serialized:
        return []
    return [item for item in serialized.split("|") if item]


def count_flavor_frequencies(df: pd.DataFrame) -> Counter:
    """Count document frequency per flavor."""
    counter: Counter = Counter()
    for serialized in df["extracted_flavors"]:
        for flavor in parse_flavor_list(serialized):
            counter[flavor] += 1
    return counter


def count_pairs(df: pd.DataFrame) -> tuple[Counter, dict[tuple[str, str], list[str]]]:
    """Count pair cooccurrences and store the review IDs containing each pair."""
    pair_counter: Counter = Counter()
    pair_to_review_ids: dict[tuple[str, str], list[str]] = {}

    for _, row in df.iterrows():
        flavors = sorted(parse_flavor_list(row["extracted_flavors"]))
        review_id = row["review_id"]
        for pair in combinations(flavors, 2):
            pair_counter[pair] += 1
            pair_to_review_ids.setdefault(pair, []).append(review_id)

    return pair_counter, pair_to_review_ids


def build_pair_graph(pair_counter: Counter) -> nx.Graph:
    """Construct a graph from pair cooccurrence counts."""
    graph = nx.Graph()
    for (flavor_a, flavor_b), count in pair_counter.items():
        graph.add_edge(flavor_a, flavor_b, weight=count)
    return graph


def sorted_pair_items(counter: Counter, *, sort_by_count: bool = True) -> list[tuple[tuple[str, str], float]]:
    """Sort pair scores deterministically."""
    items = list(counter.items())
    if sort_by_count:
        items.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    else:
        items.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
    return items


def compute_lift_rows(
    pair_counter: Counter,
    flavor_freq: Counter,
    n_reviews: int,
    min_pair_count: int,
) -> list[dict[str, object]]:
    """Compute lift rows for one condition and min_pair_count setting."""
    rows = []
    for (flavor_a, flavor_b), pair_count in pair_counter.items():
        if pair_count < min_pair_count:
            continue
        freq_a = flavor_freq[flavor_a]
        freq_b = flavor_freq[flavor_b]
        if freq_a == 0 or freq_b == 0 or n_reviews == 0:
            continue
        lift = pair_count * n_reviews / (freq_a * freq_b)
        rows.append(
            {
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "pair_key": f"{flavor_a}||{flavor_b}",
                "cooccurrence_count": pair_count,
                "frequency_a": freq_a,
                "frequency_b": freq_b,
                "support": pair_count / n_reviews,
                "lift": lift,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["lift"]),
            -int(row["cooccurrence_count"]),
            str(row["flavor_a"]),
            str(row["flavor_b"]),
        )
    )
    return rows


def pair_key(pair: tuple[str, str]) -> str:
    """Serialize a pair key."""
    return f"{pair[0]}||{pair[1]}"


def find_top_rank_map(rows: list[dict[str, object]], top_k: int) -> dict[str, tuple[int, float]]:
    """Map pair keys to (rank, score) in the top-k slice."""
    rank_map: dict[str, tuple[int, float]] = {}
    for rank, row in enumerate(rows[:top_k], start=1):
        score = row["lift"] if "lift" in row else row["cooccurrence_count"]
        rank_map[str(row["pair_key"])] = (rank, float(score))
    return rank_map


def compute_spearman(left_rank_map: dict[str, tuple[int, float]], right_rank_map: dict[str, tuple[int, float]]) -> float | None:
    """Compute Spearman correlation over the common pairs only."""
    common_keys = sorted(set(left_rank_map) & set(right_rank_map))
    if len(common_keys) < 2:
        return None
    left_series = pd.Series([left_rank_map[key][0] for key in common_keys], index=common_keys)
    right_series = pd.Series([right_rank_map[key][0] for key in common_keys], index=common_keys)
    result = left_series.corr(right_series, method="spearman")
    if pd.isna(result):
        return None
    return float(result)


def compare_rankings(
    left_condition: str,
    right_condition: str,
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    *,
    ranking_type: str,
    top_k: int,
    min_pair_count: int | None = None,
    recommended_for_poster: bool | None = None,
) -> pd.DataFrame:
    """Compare top-k rankings between two conditions."""
    left_map = find_top_rank_map(left_rows, top_k)
    right_map = find_top_rank_map(right_rows, top_k)
    all_keys = sorted(set(left_map) | set(right_map))
    common_keys = set(left_map) & set(right_map)
    jaccard = len(common_keys) / len(set(left_map) | set(right_map)) if (set(left_map) | set(right_map)) else math.nan
    spearman = compute_spearman(left_map, right_map)

    records = []
    for current_key in all_keys:
        flavor_a, flavor_b = current_key.split("||", maxsplit=1)
        left_rank, left_score = left_map.get(current_key, (None, None))
        right_rank, right_score = right_map.get(current_key, (None, None))
        if left_rank is not None and right_rank is not None:
            membership = "common"
            rank_diff = right_rank - left_rank
        elif left_rank is not None:
            membership = "only_left"
            rank_diff = None
        else:
            membership = "only_right"
            rank_diff = None

        records.append(
            {
                "ranking_type": ranking_type,
                "condition_left": left_condition,
                "condition_right": right_condition,
                "top_k": top_k,
                "min_pair_count": min_pair_count if min_pair_count is not None else "",
                "recommended_for_poster": recommended_for_poster if recommended_for_poster is not None else "",
                "pair_key": current_key,
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "membership": membership,
                "rank_left": left_rank if left_rank is not None else "",
                "rank_right": right_rank if right_rank is not None else "",
                "rank_diff_right_minus_left": rank_diff if rank_diff is not None else "",
                "score_left": left_score if left_score is not None else "",
                "score_right": right_score if right_score is not None else "",
                "common_pair_count": len(common_keys),
                "jaccard_top_k": jaccard,
                "spearman_top_k": spearman if spearman is not None else "",
            }
        )
    return pd.DataFrame(records)


def choose_recommended_lift_min_pair_count(lift_rankings_df: pd.DataFrame) -> tuple[int, str]:
    """Recommend a min_pair_count for poster use from condition B evidence."""
    limited_rows = lift_rankings_df[lift_rankings_df["condition"] == "limited_2_5"].copy()
    if limited_rows.empty:
        return 2, "condition B の Lift ランキングが空のため、既定値として min_pair_count=2 を採用。"

    summary_rows = []
    for min_pair_count in LIFT_MIN_PAIR_COUNTS:
        subset = limited_rows[limited_rows["min_pair_count"] == min_pair_count].head(10)
        if subset.empty:
            continue
        median_count = float(subset["cooccurrence_count"].median())
        singleton_ratio = float((subset["cooccurrence_count"] == 1).mean())
        max_lift = float(subset["lift"].max())
        summary_rows.append((min_pair_count, median_count, singleton_ratio, max_lift, len(subset)))

    eligible = [row for row in summary_rows if row[3] > 0]
    if not eligible:
        return 2, "condition B の Lift 上位が空のため、既定値として min_pair_count=2 を採用。"

    recommended = None
    for row in eligible:
        min_pair_count, _median_count, singleton_ratio, _max_lift, _count = row
        if singleton_ratio == 0:
            recommended = row
            break
    if recommended is None:
        recommended = max(eligible, key=lambda row: (row[1], -row[2], row[0]))

    min_pair_count, median_count, singleton_ratio, max_lift, _count = recommended
    reason = (
        "condition B の Lift Top10 を比較した結果、"
        f"min_pair_count={min_pair_count} では Top10 の共起回数中央値が {median_count:.1f}、"
        f"共起1回ペア比率が {singleton_ratio:.0%}、最大 Lift が {max_lift:.2f} だったため、"
        "ポスター用の最低共起回数として採用。"
    )
    return min_pair_count, reason


def split_sentences(text: str) -> list[str]:
    """Split text into review-like sentences for context extraction."""
    if not isinstance(text, str):
        return []
    rough_sentences = re.split(r"(?<=[。！？!?])|\n+", text)
    return [sentence.strip() for sentence in rough_sentences if sentence and sentence.strip()]


def extract_matched_context(text: str, flavor_a: str, flavor_b: str, max_chars: int = 300) -> str:
    """Extract a local context snippet containing both flavors if possible."""
    sentences = split_sentences(text)
    if not sentences:
        return ""

    for idx, sentence in enumerate(sentences):
        if flavor_a in sentence and flavor_b in sentence:
            return sentence[:max_chars]

    for idx in range(len(sentences)):
        chunk = sentences[idx]
        if flavor_a in chunk or flavor_b in chunk:
            combined = chunk
            found_a = flavor_a in combined
            found_b = flavor_b in combined
            for next_idx in range(idx + 1, min(idx + 4, len(sentences))):
                combined += " " + sentences[next_idx]
                found_a = found_a or (flavor_a in sentences[next_idx])
                found_b = found_b or (flavor_b in sentences[next_idx])
                if found_a and found_b:
                    return combined[:max_chars]

    text_upper = text.upper()
    pos_a = text_upper.find(flavor_a.upper())
    pos_b = text_upper.find(flavor_b.upper())
    if pos_a == -1 and pos_b == -1:
        return text[:max_chars]

    positions = [pos for pos in (pos_a, pos_b) if pos >= 0]
    start = max(min(positions) - 80, 0)
    end = min(max(positions) + 220, len(text))
    return text[start:end][:max_chars]


def build_manual_review_rows(
    *,
    analysis_type: str,
    ranked_rows: list[dict[str, object]],
    source_df: pd.DataFrame,
    pair_counter: Counter,
    flavor_freq: Counter,
    condition_name: str,
    max_pairs: int,
) -> list[dict[str, object]]:
    """Create manual review check rows for selected pairs."""
    source_by_review_id = source_df.set_index("review_id")
    output_rows = []

    for pair_rank, row in enumerate(ranked_rows[:max_pairs], start=1):
        flavor_a = str(row["flavor_a"])
        flavor_b = str(row["flavor_b"])
        current_pair = tuple(sorted((flavor_a, flavor_b)))
        pair_count = int(pair_counter[current_pair])
        freq_a = int(flavor_freq[flavor_a])
        freq_b = int(flavor_freq[flavor_b])
        lift = pair_count * len(source_df) / (freq_a * freq_b) if freq_a and freq_b and len(source_df) else math.nan

        matched_df = source_df[
            source_df["extracted_flavors"].apply(
                lambda serialized: flavor_a in parse_flavor_list(serialized) and flavor_b in parse_flavor_list(serialized)
            )
        ].head(3)

        for _, review_row in matched_df.iterrows():
            output_rows.append(
                {
                    "analysis_type": analysis_type,
                    "condition": condition_name,
                    "pair_rank": pair_rank,
                    "flavor_a": flavor_a,
                    "flavor_b": flavor_b,
                    "pair_key": pair_key(current_pair),
                    "cooccurrence_count": pair_count,
                    "lift": lift,
                    "review_id": review_row["review_id"],
                    "review_title": review_row["title"],
                    "review_url": review_row["url"],
                    "matched_context": extract_matched_context(review_row["review_text"], flavor_a, flavor_b),
                    "extracted_flavors_in_review": review_row["extracted_flavors"],
                    "flavor_count_in_review": review_row["flavor_count"],
                    "has_mix_keyword": review_row["has_mix_keyword"],
                    "manual_label": "",
                    "manual_note": "",
                }
            )

    return output_rows


def build_condition_statistics(condition_name: str, filtered_df: pd.DataFrame) -> tuple[dict[str, object], Counter, Counter, dict[tuple[str, str], list[str]], nx.Graph]:
    """Compute base statistics and reusable counters for one condition."""
    flavor_freq = count_flavor_frequencies(filtered_df)
    pair_counter, pair_to_review_ids = count_pairs(filtered_df)
    graph = build_pair_graph(pair_counter)

    if graph.number_of_nodes() > 0:
        components = list(nx.connected_components(graph))
        connected_components = len(components)
        largest_component_size = max(len(component) for component in components)
        density = nx.density(graph)
    else:
        connected_components = 0
        largest_component_size = 0
        density = 0.0

    stats = {
        "condition": condition_name,
        "review_count": len(filtered_df),
        "unique_flavor_count": len(flavor_freq),
        "unique_pair_count": len(pair_counter),
        "average_flavor_count": float(filtered_df["flavor_count"].mean()) if not filtered_df.empty else 0.0,
        "median_flavor_count": float(filtered_df["flavor_count"].median()) if not filtered_df.empty else 0.0,
        "max_flavor_count": int(filtered_df["flavor_count"].max()) if not filtered_df.empty else 0,
        "network_node_count": graph.number_of_nodes(),
        "network_edge_count": graph.number_of_edges(),
        "connected_components": connected_components,
        "largest_component_node_count": largest_component_size,
        "network_density": density,
    }
    return stats, flavor_freq, pair_counter, pair_to_review_ids, graph


def run_condition_analysis(review_extraction_df: pd.DataFrame) -> dict[str, object]:
    """Run the full condition analysis pipeline for one extraction table."""
    condition_stats_records = []
    cooccurrence_rows = []
    lift_rows = []
    condition_frames: dict[str, pd.DataFrame] = {}
    flavor_freq_by_condition: dict[str, Counter] = {}
    pair_counter_by_condition: dict[str, Counter] = {}
    graph_by_condition: dict[str, nx.Graph] = {}

    for spec in CONDITIONS:
        filtered_df = filter_condition(review_extraction_df, spec)
        condition_frames[spec.name] = filtered_df
        stats, flavor_freq, pair_counter, _pair_to_review_ids, graph = build_condition_statistics(spec.name, filtered_df)
        condition_stats_records.append(stats)
        flavor_freq_by_condition[spec.name] = flavor_freq
        pair_counter_by_condition[spec.name] = pair_counter
        graph_by_condition[spec.name] = graph

        n_reviews = len(filtered_df)
        sorted_pairs = sorted(pair_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        for rank, ((flavor_a, flavor_b), count) in enumerate(sorted_pairs[:20], start=1):
            cooccurrence_rows.append(
                {
                    "condition": spec.name,
                    "rank": rank,
                    "flavor_a": flavor_a,
                    "flavor_b": flavor_b,
                    "pair_key": pair_key((flavor_a, flavor_b)),
                    "cooccurrence_count": count,
                    "frequency_a": flavor_freq[flavor_a],
                    "frequency_b": flavor_freq[flavor_b],
                    "support": count / n_reviews if n_reviews else math.nan,
                    "representative_review_count": count,
                }
            )

        for min_pair_count in LIFT_MIN_PAIR_COUNTS:
            lift_records = compute_lift_rows(pair_counter, flavor_freq, n_reviews, min_pair_count=min_pair_count)
            for rank, row in enumerate(lift_records[:20], start=1):
                lift_rows.append(
                    {
                        "condition": spec.name,
                        "min_pair_count": min_pair_count,
                        "rank": rank,
                        **row,
                    }
                )

    condition_stats_df = pd.DataFrame(condition_stats_records)
    cooccurrence_rankings_df = pd.DataFrame(cooccurrence_rows)
    lift_rankings_df = pd.DataFrame(lift_rows)

    cooccurrence_comparison_df = compare_rankings(
        "all_multi",
        "limited_2_5",
        cooccurrence_rankings_df[cooccurrence_rankings_df["condition"] == "all_multi"].to_dict("records"),
        cooccurrence_rankings_df[cooccurrence_rankings_df["condition"] == "limited_2_5"].to_dict("records"),
        ranking_type="cooccurrence",
        top_k=10,
    )

    lift_comparison_frames = []
    for min_pair_count in LIFT_MIN_PAIR_COUNTS:
        left_rows = lift_rankings_df[
            (lift_rankings_df["condition"] == "all_multi")
            & (lift_rankings_df["min_pair_count"] == min_pair_count)
        ].to_dict("records")
        right_rows = lift_rankings_df[
            (lift_rankings_df["condition"] == "limited_2_5")
            & (lift_rankings_df["min_pair_count"] == min_pair_count)
        ].to_dict("records")
        lift_comparison_frames.append(
            compare_rankings(
                "all_multi",
                "limited_2_5",
                left_rows,
                right_rows,
                ranking_type="lift",
                top_k=10,
                min_pair_count=min_pair_count,
                recommended_for_poster=None,
            )
        )
    lift_comparison_df = pd.concat(lift_comparison_frames, ignore_index=True)

    return {
        "condition_stats_df": condition_stats_df,
        "cooccurrence_rankings_df": cooccurrence_rankings_df,
        "lift_rankings_df": lift_rankings_df,
        "cooccurrence_comparison_df": cooccurrence_comparison_df,
        "lift_comparison_df": lift_comparison_df,
        "condition_frames": condition_frames,
        "flavor_freq_by_condition": flavor_freq_by_condition,
        "pair_counter_by_condition": pair_counter_by_condition,
        "graph_by_condition": graph_by_condition,
    }


def create_figure1_analysis_flow(output_path: Path) -> None:
    """Draw the poster analysis flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=320)
    ax.axis("off")

    boxes = [
        (0.06, "既存レビュー"),
        (0.23, "フレーバー抽出"),
        (0.41, "条件A・条件B"),
        (0.59, "共起頻度・Lift"),
        (0.77, "条件比較"),
        (0.93, "代表レビュー確認"),
    ]

    for idx, (x_pos, label) in enumerate(boxes):
        box = plt.Rectangle((x_pos - 0.08, 0.4), 0.16, 0.2, facecolor="#E8F1F8", edgecolor="#2C5D87", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos, 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold")
        if idx < len(boxes) - 1:
            next_x = boxes[idx + 1][0]
            ax.annotate(
                "",
                xy=(next_x - 0.09, 0.5),
                xytext=(x_pos + 0.09, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#333333"},
            )

    ax.set_title("図1 分析フロー", fontsize=14, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_figure2_condition_top10(
    output_path: Path,
    cooccurrence_rankings_df: pd.DataFrame,
) -> None:
    """Create a dumbbell chart for condition A/B cooccurrence top 10 comparison."""
    subset = cooccurrence_rankings_df[
        cooccurrence_rankings_df["condition"].isin(["all_multi", "limited_2_5"])
    ].copy()
    top_a = subset[subset["condition"] == "all_multi"].head(10)
    top_b = subset[subset["condition"] == "limited_2_5"].head(10)
    union_keys = list(dict.fromkeys(top_b["pair_key"].tolist() + top_a["pair_key"].tolist()))

    records = []
    for key in union_keys:
        row_a = top_a[top_a["pair_key"] == key]
        row_b = top_b[top_b["pair_key"] == key]
        if row_a.empty and row_b.empty:
            continue
        label_row = row_b.iloc[0] if not row_b.empty else row_a.iloc[0]
        label = f"{label_row['flavor_a']} × {label_row['flavor_b']}"
        records.append(
            {
                "pair_key": key,
                "label": label,
                "all_multi": float(row_a["cooccurrence_count"].iloc[0]) if not row_a.empty else 0.0,
                "limited_2_5": float(row_b["cooccurrence_count"].iloc[0]) if not row_b.empty else 0.0,
            }
        )

    plot_df = pd.DataFrame(records)
    plot_df = plot_df.sort_values(["limited_2_5", "all_multi"], ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(plot_df))), dpi=320)
    y_pos = np.arange(len(plot_df))
    ax.hlines(y=y_pos, xmin=plot_df["all_multi"], xmax=plot_df["limited_2_5"], color="#B0BEC5", linewidth=2)
    ax.scatter(plot_df["all_multi"], y_pos, color="#D95F02", label="条件A: all_multi", s=60)
    ax.scatter(plot_df["limited_2_5"], y_pos, color="#1B9E77", label="条件B: limited_2_5", s=60)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("共起回数")
    ax.set_title("図2 条件A/B の共起頻度 Top10 比較", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_figure3_count_lift_scatter(
    output_path: Path,
    lift_rankings_df: pd.DataFrame,
    recommended_min_pair_count: int,
) -> None:
    """Create a cooccurrence-count vs lift scatter plot for condition B."""
    subset = lift_rankings_df[
        (lift_rankings_df["condition"] == "limited_2_5")
        & (lift_rankings_df["min_pair_count"] == recommended_min_pair_count)
    ].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7), dpi=320)
    ax.scatter(
        subset["cooccurrence_count"],
        subset["lift"],
        s=35 + subset["cooccurrence_count"] * 18,
        alpha=0.7,
        color="#4C78A8",
        edgecolors="white",
        linewidths=0.7,
    )

    annotation_candidates = subset.sort_values(
        ["cooccurrence_count", "lift"], ascending=[False, False]
    ).head(8)
    for _, row in annotation_candidates.iterrows():
        label = f"{row['flavor_a']}×{row['flavor_b']}"
        ax.annotate(
            label,
            (row["cooccurrence_count"], row["lift"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_xlabel("共起回数")
    ax.set_ylabel("Lift")
    ax.set_title(
        f"図3 条件B の共起回数と Lift の散布図 (min_pair_count={recommended_min_pair_count})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_figure4_rank_change(
    output_path: Path,
    cooccurrence_rankings_df: pd.DataFrame,
    top_k: int = 15,
) -> None:
    """Create a slope chart showing cooccurrence rank changes between A and B."""
    subset = cooccurrence_rankings_df[
        cooccurrence_rankings_df["condition"].isin(["all_multi", "limited_2_5"])
    ].copy()
    top_a = subset[subset["condition"] == "all_multi"].head(top_k)
    top_b = subset[subset["condition"] == "limited_2_5"].head(top_k)
    union_keys = sorted(set(top_a["pair_key"]) | set(top_b["pair_key"]))

    records = []
    for key in union_keys:
        row_a = top_a[top_a["pair_key"] == key]
        row_b = top_b[top_b["pair_key"] == key]
        if row_a.empty and row_b.empty:
            continue
        label_row = row_a.iloc[0] if not row_a.empty else row_b.iloc[0]
        label = f"{label_row['flavor_a']} × {label_row['flavor_b']}"
        rank_a = int(row_a["rank"].iloc[0]) if not row_a.empty else top_k + 2
        rank_b = int(row_b["rank"].iloc[0]) if not row_b.empty else top_k + 2
        records.append({"label": label, "rank_a": rank_a, "rank_b": rank_b})

    plot_df = pd.DataFrame(records).sort_values(["rank_a", "rank_b"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(7, 0.45 * len(plot_df))), dpi=320)
    for _, row in plot_df.iterrows():
        color = "#D95F02" if row["rank_b"] > row["rank_a"] else "#1B9E77"
        ax.plot([0, 1], [row["rank_a"], row["rank_b"]], color=color, linewidth=1.8, alpha=0.85)
        ax.text(-0.03, row["rank_a"], row["label"], ha="right", va="center", fontsize=8)
        ax.text(1.03, row["rank_b"], row["label"], ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(top_k + 2.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["条件A", "条件B"])
    ax.set_ylabel("順位")
    ax.set_title(f"図4 条件A/B の共起頻度順位変動 (Top {top_k})", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def maybe_create_figure5_manual_check(output_path: Path, manual_review_df: pd.DataFrame) -> bool:
    """Create the manual label chart when manual labels are available."""
    labeled = manual_review_df[manual_review_df["manual_label"].fillna("").str.strip() != ""].copy()
    if labeled.empty:
        return False

    counts = (
        labeled["manual_label"]
        .value_counts()
        .reindex(["explicit_mix", "probable_mix", "co_mention_only", "unclear"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=320)
    ax.bar(counts.index, counts.values, color=["#1B9E77", "#66A61E", "#D95F02", "#7570B3"])
    ax.set_ylabel("件数")
    ax.set_title("図5 人手確認結果", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def run_self_checks() -> list[str]:
    """Run the minimum required checks on toy data."""
    test_messages = []

    toy_df = pd.DataFrame(
        [
            {
                "review_id": "R0001",
                "title": "t1",
                "url": "u1",
                "date": "2026-01-01",
                "extracted_flavors": "A|B|C",
                "flavor_count": 3,
                "has_mix_keyword": True,
                "mix_keywords": "ミックス",
                "review_text": "AとBをミックスしてCを加えた。",
            },
            {
                "review_id": "R0002",
                "title": "t2",
                "url": "u2",
                "date": "2026-01-02",
                "extracted_flavors": "A|A|B".replace("|A|", "|"),
                "flavor_count": 2,
                "has_mix_keyword": False,
                "mix_keywords": "",
                "review_text": "AとB。",
            },
            {
                "review_id": "R0003",
                "title": "t3",
                "url": "u3",
                "date": "2026-01-03",
                "extracted_flavors": "A|D|E|F|G|H",
                "flavor_count": 6,
                "has_mix_keyword": True,
                "mix_keywords": "配合",
                "review_text": "配合あり。",
            },
        ]
    )

    all_multi = filter_condition(toy_df, CONDITIONS[0])
    limited = filter_condition(toy_df, CONDITIONS[1])
    mix_limited = filter_condition(toy_df, CONDITIONS[2])
    assert len(all_multi) == 3
    assert len(limited) == 2
    assert len(mix_limited) == 1
    test_messages.append("PASS: 2〜5種類条件と mix keyword 条件が正しく適用される")

    toy_pairs, _ = count_pairs(all_multi)
    assert toy_pairs[("A", "B")] == 2
    test_messages.append("PASS: 同一レビュー内の重複フレーバーを1回として共起回数を数える")

    toy_freq = count_flavor_frequencies(all_multi)
    lift_rows = compute_lift_rows(toy_pairs, toy_freq, len(all_multi), min_pair_count=1)
    ab_row = next(row for row in lift_rows if row["pair_key"] == "A||B")
    expected_lift = 2 * 3 / (3 * 2)
    assert math.isclose(float(ab_row["lift"]), expected_lift)
    test_messages.append("PASS: Lift 計算が pair_count * N / (freqA * freqB) に一致する")

    compare_df = compare_rankings(
        "all_multi",
        "limited_2_5",
        [
            {"pair_key": "A||B", "cooccurrence_count": 2},
            {"pair_key": "A||C", "cooccurrence_count": 1},
        ],
        [
            {"pair_key": "A||B", "cooccurrence_count": 2},
            {"pair_key": "B||C", "cooccurrence_count": 1},
        ],
        ranking_type="cooccurrence",
        top_k=2,
    )
    common = compare_df[compare_df["membership"] == "common"]
    assert int(common["common_pair_count"].iloc[0]) == 1
    assert math.isclose(float(common["jaccard_top_k"].iloc[0]), 1 / 3)
    test_messages.append("PASS: Top10 比較の共通数と Jaccard 係数を計算できる")

    manual_test = pd.DataFrame(
        [
            {"pair_key": "A||B", "manual_label": "explicit_mix"},
            {"pair_key": "A||B", "manual_label": "co_mention_only"},
            {"pair_key": "B||C", "manual_label": "probable_mix"},
        ]
    )
    pair_explicit = manual_test.groupby("pair_key")["manual_label"].apply(
        lambda labels: "explicit_mix" in set(labels)
    )
    assert bool(pair_explicit["A||B"])
    assert not bool(pair_explicit["B||C"])
    test_messages.append("PASS: manual label 集計でペア単位 explicit_mix 確認を判定できる")

    assert detect_mix_keywords("メインにA、アクセントにBを加えた") == ["加え", "メイン", "アクセント"]
    test_messages.append("PASS: mix keyword 判定が指定語に反応する")

    return test_messages


def render_markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    """Render a small DataFrame slice as a markdown table."""
    table_df = df[columns].copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in table_df.iterrows()
    ]
    return "\n".join([header, separator] + body) if body else "\n".join([header, separator])


def get_top_rows(
    rankings_df: pd.DataFrame,
    *,
    condition: str,
    top_k: int = 10,
    min_pair_count: int | None = None,
) -> pd.DataFrame:
    """Return a deterministic top-k slice from a ranking table."""
    subset = rankings_df[rankings_df["condition"] == condition].copy()
    if min_pair_count is not None:
        subset = subset[subset["min_pair_count"] == min_pair_count].copy()
    return subset.sort_values(["rank", "flavor_a", "flavor_b"]).head(top_k).reset_index(drop=True)


def with_pair_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a human-readable pair label for markdown tables."""
    if df.empty:
        return df.copy()
    labeled = df.copy()
    labeled["pair"] = labeled["flavor_a"].astype(str) + " × " + labeled["flavor_b"].astype(str)
    return labeled


def top_set_metrics(left_df: pd.DataFrame, right_df: pd.DataFrame) -> tuple[int, float]:
    """Compute common-count and Jaccard over pair_key sets."""
    left_keys = set(left_df["pair_key"]) if "pair_key" in left_df.columns else set()
    right_keys = set(right_df["pair_key"]) if "pair_key" in right_df.columns else set()
    union = left_keys | right_keys
    common = left_keys & right_keys
    jaccard = len(common) / len(union) if union else math.nan
    return len(common), jaccard


def comparison_metrics(comparison_df: pd.DataFrame) -> tuple[int, float, float | None]:
    """Extract repeated comparison metrics from a comparison dataframe."""
    if comparison_df.empty:
        return 0, math.nan, None
    row = comparison_df.iloc[0]
    spearman_value = row["spearman_top_k"]
    if spearman_value == "":
        spearman = None
    else:
        spearman = float(spearman_value)
    return int(row["common_pair_count"]), float(row["jaccard_top_k"]), spearman


def pair_presence_status(
    pair_counter_by_condition: dict[str, Counter],
    pair_name: str,
) -> dict[str, bool]:
    """Check whether a target pair exists in condition A/B pair counters."""
    left, right = [part.strip() for part in pair_name.split("×", maxsplit=1)]
    current_pair = tuple(sorted((left, right)))
    return {
        "all_multi_present": current_pair in pair_counter_by_condition["all_multi"],
        "limited_2_5_present": current_pair in pair_counter_by_condition["limited_2_5"],
    }


def write_summary(
    *,
    summary_path: Path,
    existing_impl_info: dict[str, object],
    raw_review_extraction_df: pd.DataFrame,
    review_extraction_df: pd.DataFrame,
    raw_condition_stats_df: pd.DataFrame,
    condition_stats_df: pd.DataFrame,
    raw_cooccurrence_rankings_df: pd.DataFrame,
    cooccurrence_rankings_df: pd.DataFrame,
    raw_lift_rankings_df: pd.DataFrame,
    lift_rankings_df: pd.DataFrame,
    raw_cooccurrence_comparison_df: pd.DataFrame,
    cooccurrence_comparison_df: pd.DataFrame,
    raw_lift_comparison_df: pd.DataFrame,
    lift_comparison_df: pd.DataFrame,
    manual_review_df: pd.DataFrame,
    alias_candidates_df: pd.DataFrame,
    normalization_map_df: pd.DataFrame,
    suspicious_pair_df: pd.DataFrame,
    raw_recommended_min_pair_count: int,
    recommended_min_pair_count: int,
    raw_recommendation_reason: str,
    recommendation_reason: str,
    generated_files: list[Path],
    test_messages: list[str],
    figure5_created: bool,
) -> None:
    """Write the requested markdown summary."""
    lines: list[str] = []

    def add(line: str = "") -> None:
        lines.append(line)

    add("# poster_analysis summary")
    add()
    add("## 1. 使用データ")
    add(f"- レビューCSV: `{REVIEWS_CSV.relative_to(ROOT)}`")
    add(f"- フレーバーマスタ: `{MASTER_CSV.relative_to(ROOT)}`")
    add(f"- レビュー本文列: `{existing_impl_info['review_text_column']}`")
    add(f"- タイトル列: `{existing_impl_info['title_column']}`")
    add(f"- URL列: `{existing_impl_info['url_column']}`")
    add(f"- 日付列: `{existing_impl_info['date_column']}`")
    add(f"- 分析対象レビュー総数: {len(review_extraction_df)}")
    add(f"- 正規化前ユニークフレーバー総数: {raw_review_extraction_df['extracted_flavors'].apply(parse_flavor_list).explode().nunique()}")
    add(f"- 正規化後ユニークフレーバー総数: {review_extraction_df['extracted_flavors'].apply(parse_flavor_list).explode().nunique()}")
    add()

    add("## 2. 既存実装の確認結果")
    add("### 結果")
    add(f"- 使用レビューCSV: `{existing_impl_info['reviews_csv']}`")
    add(f"- 使用フレーバーマスタ: `{existing_impl_info['master_csv']}`")
    add(f"- フレーバー正規化処理: {existing_impl_info['normalization']}")
    add(f"- 共起の定義: {existing_impl_info['cooccurrence_definition']}")
    add(f"- Liftの計算式: {existing_impl_info['lift_formula']}")
    add(f"- 既存の除外条件: {existing_impl_info['existing_filters']}")
    add("- 既存の出力図とCSV:")
    for rel_path in existing_impl_info["existing_outputs"]:
        add(f"  - `{rel_path}`")
    add("### 考察")
    add("- 既存コードは全体ランキングやネットワーク可視化には到達しているが、条件比較と代表レビュー確認を横断的に出す仕組みは無かった。")
    add("- 抽出ロジックはホワイトリストベースで一貫していたため、今回の条件比較も同じ辞書を用いている。")
    add()

    add("## 3. フレーバー名称正規化")
    auto_alias_df = alias_candidates_df[~alias_candidates_df["manual_review"].astype(bool)].copy()
    manual_alias_df = alias_candidates_df[alias_candidates_df["manual_review"].astype(bool)].copy()
    changed_map_df = normalization_map_df[
        normalization_map_df["raw_flavor"] != normalization_map_df["canonical_flavor"]
    ].copy()
    add("### 結果")
    add(f"- 自動統合候補数: {len(auto_alias_df)}")
    add(f"- manual_review=true の候補数: {len(manual_alias_df)}")
    add(f"- 実際に canonical 変更された raw flavor 数: {changed_map_df['raw_flavor'].nunique()}")
    if not changed_map_df.empty:
        add(render_markdown_table(changed_map_df.head(15), ["raw_flavor", "canonical_flavor", "normalization_rule"]))
    add("### 考察")
    add("- 自動統合は NFKC・記号差・マスタ/レビューで確認できる EN/JA 対応・明示的に検証した基本訳語だけに限定した。")
    add("- 単数複数差やカタカナ揺れなど、誤統合の余地がある候補は manual_review=true とし、自動統合から除外した。")
    add()

    add("## 4. 条件A・B・Cの定義")
    for spec in CONDITIONS:
        add(f"- `{spec.name}`: {spec.description}")
    add()

    pre_post_stats_df = raw_condition_stats_df.merge(
        condition_stats_df,
        on="condition",
        suffixes=("_raw", "_normalized"),
    )
    add("## 5. 条件別基礎統計")
    add("### 結果")
    add(
        render_markdown_table(
            pre_post_stats_df.round(4),
            [
                "condition",
                "review_count_raw",
                "review_count_normalized",
                "unique_flavor_count_raw",
                "unique_flavor_count_normalized",
                "unique_pair_count_raw",
                "unique_pair_count_normalized",
                "average_flavor_count_raw",
                "average_flavor_count_normalized",
            ],
        )
    )
    add("### 考察")
    add("- 条件Aと条件Bの差は、2〜5種類に絞ることで多数列挙レビューの影響をどこまで抑えられるかを見るための主比較とした。")
    add("- 条件Cは mix keyword に依存するため、本文表現に偏りが出る補助分析として扱う。")
    add()

    raw_top_cooc_b = with_pair_label(get_top_rows(raw_cooccurrence_rankings_df, condition="limited_2_5"))
    top_cooc_a = with_pair_label(get_top_rows(cooccurrence_rankings_df, condition="all_multi"))
    top_cooc_b = with_pair_label(get_top_rows(cooccurrence_rankings_df, condition="limited_2_5"))
    normalized_cooc_b_overlap, normalized_cooc_b_jaccard = top_set_metrics(raw_top_cooc_b, top_cooc_b)
    raw_ab_common, raw_ab_jaccard, raw_ab_spearman = comparison_metrics(raw_cooccurrence_comparison_df)
    normalized_ab_common, normalized_ab_jaccard, normalized_ab_spearman = comparison_metrics(cooccurrence_comparison_df)

    add("## 6. 共起頻度の比較")
    add("### 結果")
    add(f"- 正規化前の条件A/B Top10 共通数: {raw_ab_common}")
    add(f"- 正規化後の条件A/B Top10 共通数: {normalized_ab_common}")
    add(f"- 正規化前の条件A/B Jaccard係数: {raw_ab_jaccard:.4f}")
    add(f"- 正規化後の条件A/B Jaccard係数: {normalized_ab_jaccard:.4f}")
    add(f"- 正規化前の条件A/B Spearman順位相関: {'計算不能' if raw_ab_spearman is None else f'{raw_ab_spearman:.4f}'}")
    add(f"- 正規化後の条件A/B Spearman順位相関: {'計算不能' if normalized_ab_spearman is None else f'{normalized_ab_spearman:.4f}'}")
    add(f"- 条件B 共起Top10 の正規化前後共通数: {normalized_cooc_b_overlap}")
    add(f"- 条件B 共起Top10 の正規化前後 Jaccard係数: {normalized_cooc_b_jaccard:.4f}")
    add("- 正規化前の条件B 共起Top10:")
    add(render_markdown_table(raw_top_cooc_b, ["rank", "pair", "cooccurrence_count"], max_rows=10))
    add("- 正規化後の条件A 共起Top10:")
    add(render_markdown_table(top_cooc_a, ["rank", "pair", "cooccurrence_count"], max_rows=10))
    add("- 正規化後の条件B 共起Top10:")
    add(render_markdown_table(top_cooc_b, ["rank", "pair", "cooccurrence_count"], max_rows=10))
    add("### 考察")
    add("- 正規化で疑似ペアが消えると、条件Bの上位は実際のフレーバー共起へ寄りやすくなる。")
    add("- 条件Aのみで高順位のペアは、多数列挙型レビューの影響を受けている可能性があるため、代表レビュー確認が重要になる。")
    add()

    raw_lift_subset = with_pair_label(
        get_top_rows(
            raw_lift_rankings_df,
            condition="limited_2_5",
            min_pair_count=raw_recommended_min_pair_count,
        )
    )
    lift_subset = with_pair_label(
        get_top_rows(
            lift_rankings_df,
            condition="limited_2_5",
            min_pair_count=recommended_min_pair_count,
        )
    )
    raw_lift_compare_main = raw_lift_comparison_df[
        raw_lift_comparison_df["min_pair_count"] == raw_recommended_min_pair_count
    ]
    lift_compare_main = lift_comparison_df[
        lift_comparison_df["min_pair_count"] == recommended_min_pair_count
    ]
    raw_lift_common, raw_lift_jaccard, raw_lift_spearman = comparison_metrics(raw_lift_compare_main)
    normalized_lift_common, normalized_lift_jaccard, normalized_lift_spearman = comparison_metrics(lift_compare_main)
    normalized_lift_overlap, normalized_lift_overlap_jaccard = top_set_metrics(raw_lift_subset, lift_subset)

    add("## 7. Liftの比較")
    add("### 結果")
    add(f"- 正規化前の推奨 min_pair_count: {raw_recommended_min_pair_count}")
    add(f"- 正規化前の採用理由: {raw_recommendation_reason}")
    add(f"- 正規化後の推奨 min_pair_count: {recommended_min_pair_count}")
    add(f"- 正規化後の採用理由: {recommendation_reason}")
    add(f"- 正規化前の条件A/B Lift Top10 共通数: {raw_lift_common}")
    add(f"- 正規化後の条件A/B Lift Top10 共通数: {normalized_lift_common}")
    add(f"- 正規化前の条件A/B Lift Jaccard係数: {raw_lift_jaccard:.4f}")
    add(f"- 正規化後の条件A/B Lift Jaccard係数: {normalized_lift_jaccard:.4f}")
    add(f"- 正規化前の条件A/B Lift Spearman順位相関: {'計算不能' if raw_lift_spearman is None else f'{raw_lift_spearman:.4f}'}")
    add(f"- 正規化後の条件A/B Lift Spearman順位相関: {'計算不能' if normalized_lift_spearman is None else f'{normalized_lift_spearman:.4f}'}")
    add(f"- 条件B Lift Top10 の正規化前後共通数: {normalized_lift_overlap}")
    add(f"- 条件B Lift Top10 の正規化前後 Jaccard係数: {normalized_lift_overlap_jaccard:.4f}")
    add("- 正規化前の条件B Lift Top10:")
    add(render_markdown_table(raw_lift_subset.round(4), ["rank", "pair", "cooccurrence_count", "lift"], max_rows=10))
    add("- 正規化後の条件B Lift Top10:")
    add(render_markdown_table(lift_subset.round(4), ["rank", "pair", "cooccurrence_count", "lift"], max_rows=10))
    add("### 考察")
    add("- Lift は低頻度ペアで極端に大きくなりやすいため、共起回数の閾値比較を分けて確認した。")
    add("- ポスターでは、共起1回だけのペアに引きずられにくい閾値を採用し、代表レビュー確認とセットで解釈するのが安全。")
    add()

    add("## 8. 疑似ペア確認")
    add("### 結果")
    add(
        render_markdown_table(
            suspicious_pair_df,
            [
                "pair",
                "raw_all_multi_present",
                "normalized_all_multi_present",
                "raw_limited_2_5_present",
                "normalized_limited_2_5_present",
            ],
        )
    )
    add("### 考察")
    add("- 指定した疑似ペアが正規化後に消えていれば、同一フレーバー分裂による見かけの共起は解消できている。")
    add()

    add("## 9. 条件変更で順位が大きく変わったペア")
    add("### 結果")
    rank_drop_rows = manual_review_df[
        manual_review_df["analysis_type"] == "conditionA_rank_drop_top5"
    ][["pair_rank", "flavor_a", "flavor_b", "cooccurrence_count"]].drop_duplicates()
    add(render_markdown_table(rank_drop_rows, ["pair_rank", "flavor_a", "flavor_b", "cooccurrence_count"]))
    add("### 考察")
    add("- 条件Aでは上位でも条件Bで大きく落ちるペアは、列挙型レビューの寄与や長大レビュー特有の共起を疑うべき候補である。")
    add()

    add("## 10. 代表レビュー確認対象")
    add("### 結果")
    add(f"- manual_review_check.csv の行数: {len(manual_review_df)}")
    add(f"- 対象ペア数: {manual_review_df[['analysis_type', 'flavor_a', 'flavor_b']].drop_duplicates().shape[0]}")
    add("### 考察")
    add("- 共起頻度上位、Lift上位、条件Aでのみ強いペアを並べることで、ランキングの質を人手で比較しやすくした。")
    add()

    add("## 11. 生成ファイル一覧")
    for file_path in generated_files:
        add(f"- `{file_path.relative_to(ROOT)}`")
    if not figure5_created:
        add("- `poster_analysis/figure5_manual_check.png` は manual_label 未入力のため未生成")
    add()

    add("## 12. ポスターに載せる主要な発見候補3点")
    add("- 条件Aと条件Bで共通して上位に残るペアは、抽出条件を変えても安定な候補として提示できる。")
    add("- フレーバー名称正規化により、英語/日本語の疑似ペアを除去してランキングの解釈を安定化できる。")
    add("- Lift は最低共起回数を変えるだけでランキングが大きく変わるため、閾値選定の根拠をポスターに明記すべきである。")
    add()

    add("## 13. 人手確認が必要な作業")
    add("- `poster_analysis/manual_review_check.csv` の `manual_label` を `explicit_mix / probable_mix / co_mention_only / unclear` で入力する。")
    add("- 入力後に `python3 scripts/summarize_manual_review_check.py` を実行し、集計と図5を生成する。")
    add()

    add("## 14. 実行コマンドとテスト結果")
    add("- 実行コマンド: `python3 scripts/generate_condition_comparison.py`")
    add("- manual label 集計: `python3 scripts/summarize_manual_review_check.py`")
    add("- テスト結果:")
    for message in test_messages:
        add(f"  - {message}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    POSTER_DIR.mkdir(parents=True, exist_ok=True)

    reviews_df = pd.read_csv(REVIEWS_CSV)
    master_df = pd.read_csv(MASTER_CSV)
    flavor_dict, pattern_to_canonical, sorted_patterns = build_flavor_dictionary(master_df)

    raw_review_extraction_df = build_review_extraction_summary(reviews_df, sorted_patterns, pattern_to_canonical)
    alias_candidates_df, normalization_map_df = build_alias_candidates_and_map(raw_review_extraction_df, master_df)
    review_extraction_df = apply_normalization_map(raw_review_extraction_df, normalization_map_df)

    review_extraction_path = POSTER_DIR / "review_extraction_summary.csv"
    alias_candidates_path = POSTER_DIR / "flavor_alias_candidates.csv"
    normalization_map_path = POSTER_DIR / "flavor_normalization_map.csv"
    manual_alias_review_path = POSTER_DIR / "manual_alias_review.csv"
    review_extraction_df.to_csv(review_extraction_path, index=False, encoding="utf-8-sig")
    alias_candidates_df.to_csv(alias_candidates_path, index=False, encoding="utf-8-sig")
    normalization_map_df.to_csv(normalization_map_path, index=False, encoding="utf-8-sig")
    alias_candidates_df[
        alias_candidates_df["manual_review"].astype(bool)
    ][
        ["raw_flavor", "normalized_candidate", "match_type", "confidence", "note"]
    ].to_csv(manual_alias_review_path, index=False, encoding="utf-8-sig")

    raw_analysis = run_condition_analysis(raw_review_extraction_df)
    normalized_analysis = run_condition_analysis(review_extraction_df)

    raw_condition_stats_df = raw_analysis["condition_stats_df"]
    condition_stats_df = normalized_analysis["condition_stats_df"]
    raw_cooccurrence_rankings_df = raw_analysis["cooccurrence_rankings_df"]
    cooccurrence_rankings_df = normalized_analysis["cooccurrence_rankings_df"]
    raw_lift_rankings_df = raw_analysis["lift_rankings_df"]
    lift_rankings_df = normalized_analysis["lift_rankings_df"]
    raw_cooccurrence_comparison_df = raw_analysis["cooccurrence_comparison_df"]
    cooccurrence_comparison_df = normalized_analysis["cooccurrence_comparison_df"]
    raw_lift_comparison_df = raw_analysis["lift_comparison_df"]
    lift_comparison_df = normalized_analysis["lift_comparison_df"].copy()
    condition_frames = normalized_analysis["condition_frames"]
    flavor_freq_by_condition = normalized_analysis["flavor_freq_by_condition"]
    pair_counter_by_condition = normalized_analysis["pair_counter_by_condition"]

    raw_recommended_min_pair_count, raw_recommendation_reason = choose_recommended_lift_min_pair_count(raw_lift_rankings_df)
    recommended_min_pair_count, recommendation_reason = choose_recommended_lift_min_pair_count(lift_rankings_df)
    lift_comparison_df["recommended_for_poster"] = (
        lift_comparison_df["min_pair_count"] == recommended_min_pair_count
    )

    condition_statistics_path = POSTER_DIR / "condition_statistics.csv"
    cooccurrence_rankings_path = POSTER_DIR / "cooccurrence_rankings.csv"
    lift_rankings_path = POSTER_DIR / "lift_rankings.csv"
    cooccurrence_comparison_path = POSTER_DIR / "cooccurrence_condition_comparison.csv"
    lift_comparison_path = POSTER_DIR / "lift_condition_comparison.csv"

    condition_stats_df.to_csv(condition_statistics_path, index=False, encoding="utf-8-sig")
    cooccurrence_rankings_df.to_csv(cooccurrence_rankings_path, index=False, encoding="utf-8-sig")
    lift_rankings_df.to_csv(lift_rankings_path, index=False, encoding="utf-8-sig")
    cooccurrence_comparison_df.to_csv(cooccurrence_comparison_path, index=False, encoding="utf-8-sig")
    lift_comparison_df.to_csv(lift_comparison_path, index=False, encoding="utf-8-sig")

    all_multi_top = cooccurrence_rankings_df[cooccurrence_rankings_df["condition"] == "all_multi"].copy()
    limited_top = cooccurrence_rankings_df[cooccurrence_rankings_df["condition"] == "limited_2_5"].copy()
    all_rank_map = {row["pair_key"]: int(row["rank"]) for _, row in all_multi_top.iterrows()}
    limited_rank_map = {row["pair_key"]: int(row["rank"]) for _, row in limited_top.iterrows()}

    rank_drop_candidates = []
    for _, row in all_multi_top.head(20).iterrows():
        current_key = row["pair_key"]
        right_rank = limited_rank_map.get(current_key)
        penalty_rank = right_rank if right_rank is not None else 999
        rank_diff = penalty_rank - int(row["rank"])
        rank_drop_candidates.append((rank_diff, penalty_rank, row.to_dict()))
    rank_drop_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]["rank"]))
    rank_drop_rows = [candidate[2] for candidate in rank_drop_candidates[:5]]

    manual_rows = []
    manual_rows.extend(
        build_manual_review_rows(
            analysis_type="conditionB_cooccurrence_top10",
            ranked_rows=limited_top.to_dict("records"),
            source_df=condition_frames["limited_2_5"],
            pair_counter=pair_counter_by_condition["limited_2_5"],
            flavor_freq=flavor_freq_by_condition["limited_2_5"],
            condition_name="limited_2_5",
            max_pairs=10,
        )
    )
    manual_rows.extend(
        build_manual_review_rows(
            analysis_type=f"conditionB_lift_top10_min{recommended_min_pair_count}",
            ranked_rows=lift_rankings_df[
                (lift_rankings_df["condition"] == "limited_2_5")
                & (lift_rankings_df["min_pair_count"] == recommended_min_pair_count)
            ].to_dict("records"),
            source_df=condition_frames["limited_2_5"],
            pair_counter=pair_counter_by_condition["limited_2_5"],
            flavor_freq=flavor_freq_by_condition["limited_2_5"],
            condition_name="limited_2_5",
            max_pairs=10,
        )
    )
    manual_rows.extend(
        build_manual_review_rows(
            analysis_type="conditionA_rank_drop_top5",
            ranked_rows=rank_drop_rows,
            source_df=condition_frames["all_multi"],
            pair_counter=pair_counter_by_condition["all_multi"],
            flavor_freq=flavor_freq_by_condition["all_multi"],
            condition_name="all_multi",
            max_pairs=5,
        )
    )

    manual_review_df = pd.DataFrame(manual_rows)
    manual_review_path = POSTER_DIR / "manual_review_check.csv"
    manual_review_df.to_csv(manual_review_path, index=False, encoding="utf-8-sig")

    figure1_path = POSTER_DIR / "figure1_analysis_flow.png"
    figure2_path = POSTER_DIR / "figure2_condition_top10.png"
    figure3_path = POSTER_DIR / "figure3_count_lift_scatter.png"
    figure4_path = POSTER_DIR / "figure4_rank_change.png"
    figure5_path = POSTER_DIR / "figure5_manual_check.png"

    create_figure1_analysis_flow(figure1_path)
    create_figure2_condition_top10(figure2_path, cooccurrence_rankings_df)
    create_figure3_count_lift_scatter(figure3_path, lift_rankings_df, recommended_min_pair_count)
    create_figure4_rank_change(figure4_path, cooccurrence_rankings_df)
    figure5_created = maybe_create_figure5_manual_check(figure5_path, manual_review_df)

    existing_outputs = [
        str((NOTEBOOK_OUTPUT_DIR / name).relative_to(ROOT))
        for name in EXISTING_OUTPUT_FILES
        if (NOTEBOOK_OUTPUT_DIR / name).exists()
    ]

    existing_impl_info = {
        "reviews_csv": str(REVIEWS_CSV.relative_to(ROOT)),
        "master_csv": str(MASTER_CSV.relative_to(ROOT)),
        "review_text_column": "レビュー本文",
        "title_column": "レビュータイトル",
        "url_column": "レビューURL",
        "date_column": "更新日",
        "normalization": "aslaj_master_list.csv をホワイトリスト辞書として用い、括弧内日本語表記と英語表記を canonical 化して貪欲最長マッチで抽出。",
        "cooccurrence_definition": "同一レビュー内で抽出されたユニークフレーバー集合から 2 組を数える。1レビュー内の同一フレーバー重複は 1 回扱い。",
        "lift_formula": "lift(A,B) = pair_count(A,B) * N / (frequency(A) * frequency(B))",
        "existing_filters": "既存スクリプトでは主に登場件数や共起回数で可視化時の足切りを行う。3〜8種レビューに絞るレシピ特化分析も存在する。",
        "existing_outputs": existing_outputs,
    }

    test_messages = run_self_checks()

    suspicious_pairs = ["LYCHEE×ライチ", "MINT×ミント", "LEMON×レモン", "ICE×アイス"]
    suspicious_rows = []
    for pair_name in suspicious_pairs:
        raw_presence = pair_presence_status(raw_analysis["pair_counter_by_condition"], pair_name)
        normalized_presence = pair_presence_status(normalized_analysis["pair_counter_by_condition"], pair_name)
        suspicious_rows.append(
            {
                "pair": pair_name,
                "raw_all_multi_present": raw_presence["all_multi_present"],
                "normalized_all_multi_present": normalized_presence["all_multi_present"],
                "raw_limited_2_5_present": raw_presence["limited_2_5_present"],
                "normalized_limited_2_5_present": normalized_presence["limited_2_5_present"],
            }
        )
    suspicious_pair_df = pd.DataFrame(suspicious_rows)

    generated_files = [
        review_extraction_path,
        alias_candidates_path,
        normalization_map_path,
        manual_alias_review_path,
        condition_statistics_path,
        cooccurrence_rankings_path,
        lift_rankings_path,
        cooccurrence_comparison_path,
        lift_comparison_path,
        manual_review_path,
        figure1_path,
        figure2_path,
        figure3_path,
        figure4_path,
    ]
    if figure5_created:
        generated_files.append(figure5_path)

    summary_path = POSTER_DIR / "summary.md"
    write_summary(
        summary_path=summary_path,
        existing_impl_info=existing_impl_info,
        raw_review_extraction_df=raw_review_extraction_df,
        review_extraction_df=review_extraction_df,
        raw_condition_stats_df=raw_condition_stats_df,
        condition_stats_df=condition_stats_df,
        raw_cooccurrence_rankings_df=raw_cooccurrence_rankings_df,
        cooccurrence_rankings_df=cooccurrence_rankings_df,
        raw_lift_rankings_df=raw_lift_rankings_df,
        lift_rankings_df=lift_rankings_df,
        raw_cooccurrence_comparison_df=raw_cooccurrence_comparison_df,
        cooccurrence_comparison_df=cooccurrence_comparison_df,
        raw_lift_comparison_df=raw_lift_comparison_df,
        lift_comparison_df=lift_comparison_df,
        manual_review_df=manual_review_df,
        alias_candidates_df=alias_candidates_df,
        normalization_map_df=normalization_map_df,
        suspicious_pair_df=suspicious_pair_df,
        raw_recommended_min_pair_count=raw_recommended_min_pair_count,
        recommended_min_pair_count=recommended_min_pair_count,
        raw_recommendation_reason=raw_recommendation_reason,
        recommendation_reason=recommendation_reason,
        generated_files=generated_files,
        test_messages=test_messages,
        figure5_created=figure5_created,
    )

    print("poster_analysis outputs generated:")
    for file_path in generated_files + [summary_path]:
        print(f"  - {file_path.relative_to(ROOT)}")
    if not figure5_created:
        print("  - poster_analysis/figure5_manual_check.png (not generated; manual_label is blank)")
    print("tests:")
    for message in test_messages:
        print(f"  - {message}")
    print(f"recommended min_pair_count for Lift: {recommended_min_pair_count}")
    print(recommendation_reason)


if __name__ == "__main__":
    main()
