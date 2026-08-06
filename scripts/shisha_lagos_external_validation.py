#!/usr/bin/env python3
"""Utilities for extracting and comparing Shisha LAGOS recommended mix pairs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from extended_analysis_utils import detect_parent_child_pair, normalize_compare_text
from generate_condition_comparison import build_flavor_dictionary, extract_flavors

SOURCE_SITE = "shisha_lagos"
SOURCE_TYPE = "editorial_review"
NONE_MARKERS = {"×", "x", "X", "なし", "無し", "-", "―", "ー"}
BASELINE_TOP_KS = [10, 20, 50]

CONSERVATIVE_OVERRIDE_MAP = {
    "すいか": "デクラウド　スイカ",
    "スイカ": "デクラウド　スイカ",
    "ゆず": "デクラウド　ゆず",
    "コーヒー": "デクラウド　コーヒー",
    "ミルクティ": "デクラウド　ミルクティー",
    "ミルクティー": "デクラウド　ミルクティー",
}

EXTENDED_OVERRIDE_MAP = {
    **CONSERVATIVE_OVERRIDE_MAP,
    "チャイ": "スパイスドチャイ",
    "パイン": "デクラウド　パイナップル",
    "洋ナシ": "デクラウド　ペアー",
}

CANDIDATE_MANUAL_RULES = {
    "すいか": {
        "normalization_candidate": "デクラウド　スイカ",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "low: ひらがな表記ゆれ",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "カタカナの「スイカ」と同義の表記ゆれとして扱える可能性が高い。",
    },
    "ゆず": {
        "normalization_candidate": "デクラウド　ゆず",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "low: 2文字語のため既存抽出条件で漏れている可能性",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタ内に近い表記があり、短語フィルタの影響で未解決になっている可能性が高い。",
    },
    "りんご": {
        "normalization_candidate": "",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "medium: アップル系候補が複数あり単純対応不可",
        "dictionary_add_candidate": "見送り",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタではレッドアップル、グリーンアップル、ツーアップルなど複数候補に分岐する。",
    },
    "グレナデン": {
        "normalization_candidate": "",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "high: ザクロ系の意味対応か独立語か要確認",
        "dictionary_add_candidate": "見送り",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタに同表記がなく、ザクロ系への意味対応を仮定する必要があるため保留。",
    },
    "コーヒー": {
        "normalization_candidate": "デクラウド　コーヒー",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "medium: コーヒーミルク等との区別が必要",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタに近い単体コーヒー表記があるが、ミルク入り候補との混同余地がある。",
    },
    "スイカ": {
        "normalization_candidate": "デクラウド　スイカ",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "low: 一般的な味名",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタに近いスイカ表記が存在し、記事タイトル由来ターゲット語としても自然。",
    },
    "チャイ": {
        "normalization_candidate": "スパイスドチャイ",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "medium: 修飾語付き商品の省略形の可能性",
        "dictionary_add_candidate": "拡張候補",
        "requires_manual_review": "yes",
        "judgment_reason": "チャイ系味名として自然だが、既存マスタでは修飾付きの別商品名が近い。",
    },
    "パイン": {
        "normalization_candidate": "デクラウド　パイナップル",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "medium: 省略形からパイナップルへ寄せる判断が必要",
        "dictionary_add_candidate": "拡張候補",
        "requires_manual_review": "yes",
        "judgment_reason": "一般にはパイナップルの省略形と解釈できるが、正式正規化には人手確認が必要。",
    },
    "ミルクティ": {
        "normalization_candidate": "デクラウド　ミルクティー",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "low: 長音有無の表記ゆれ",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "長音欠落の表記ゆれとして扱える可能性が高い。",
    },
    "ミルクティー": {
        "normalization_candidate": "デクラウド　ミルクティー",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "low: 一般的な味名",
        "dictionary_add_candidate": "保守的候補",
        "requires_manual_review": "yes",
        "judgment_reason": "既存マスタに近い同名表記があり、比較的保守的に追加候補とみなせる。",
    },
    "洋ナシ": {
        "normalization_candidate": "デクラウド　ペアー",
        "general_flavor_name": "yes",
        "brand_category_or_description_risk": "medium: 翻訳対応が必要",
        "dictionary_add_candidate": "拡張候補",
        "requires_manual_review": "yes",
        "judgment_reason": "ペアー/ラフランス系との意味対応が考えられるが、完全一致ではない。",
    },
}


@dataclass
class FlavorDictionary:
    canonical_flavors: list[str]
    pattern_to_canonical: dict[str, str]
    sorted_patterns: list[str]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_flavor_dictionary(master_csv: Path) -> FlavorDictionary:
    master_df = read_csv(master_csv)
    _flavor_dict, pattern_to_canonical, sorted_patterns = build_flavor_dictionary(master_df)
    canonical_flavors = sorted(set(pattern_to_canonical.values()))
    return FlavorDictionary(
        canonical_flavors=canonical_flavors,
        pattern_to_canonical=pattern_to_canonical,
        sorted_patterns=sorted_patterns,
    )


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        value = "" if pd.isna(value) else str(value)
    text = value.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_single_flavor(raw_text: str, flavor_dict: FlavorDictionary) -> str:
    candidate = clean_text(raw_text)
    if not candidate:
        return ""
    candidate = re.sub(r"[（(].*?[)）]", "", candidate).strip()
    matches = extract_flavors(candidate, flavor_dict.sorted_patterns, flavor_dict.pattern_to_canonical)
    if not matches:
        return ""
    if len(matches) == 1:
        return matches[0]

    exact_key = normalize_compare_text(candidate)
    for match in matches:
        if normalize_compare_text(match) == exact_key:
            return match
    return matches[0]


def ordered_pair_key(flavor_a: str, flavor_b: str) -> str:
    members = [clean_text(flavor_a), clean_text(flavor_b)]
    if not all(members):
        return ""
    return "||".join(sorted(members))


def directed_pair_key(target_flavor: str, recommended_flavor: str) -> str:
    if not clean_text(target_flavor) or not clean_text(recommended_flavor):
        return ""
    return f"{clean_text(target_flavor)} -> {clean_text(recommended_flavor)}"


def split_recommended_cell(cell_text: str) -> list[str]:
    text = clean_text(cell_text)
    if not text:
        return []
    parts = re.split(r"[\s/／|｜、,，]+", text)
    tokens: list[str] = []
    for part in parts:
        token = clean_text(part).strip("・･;；:：")
        if not token:
            continue
        tokens.append(token)
    return tokens


def infer_normalization_candidate(raw_text: str, canonical_flavors: list[str]) -> str:
    raw_key = normalize_compare_text(raw_text)
    if not raw_key:
        return ""
    candidates: list[tuple[int, int, str]] = []
    for canonical in canonical_flavors:
        canonical_key = normalize_compare_text(canonical)
        if not canonical_key:
            continue
        if raw_key in canonical_key or canonical_key in raw_key:
            candidates.append((abs(len(canonical_key) - len(raw_key)), len(canonical_key), canonical))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][2]


def list_similar_master_forms(raw_text: str, master_df: pd.DataFrame) -> list[str]:
    raw_key = normalize_compare_text(raw_text)
    if not raw_key:
        return []
    similar_forms: list[str] = []
    for value in master_df["フレーバー名"].fillna("").astype(str).tolist():
        candidate = clean_text(value)
        if not candidate:
            continue
        candidate_key = normalize_compare_text(candidate)
        if raw_key in candidate_key or candidate_key in raw_key:
            similar_forms.append(candidate)
    return sorted(dict.fromkeys(similar_forms))


def build_override_lookup(scenario: str) -> dict[str, str]:
    if scenario == "conservative":
        return dict(CONSERVATIVE_OVERRIDE_MAP)
    if scenario == "extended":
        return dict(EXTENDED_OVERRIDE_MAP)
    return {}


def build_table_map(tables_df: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    table_map: dict[tuple[str, int], pd.DataFrame] = {}
    for (article_id, table_index), group in tables_df.groupby(["article_id", "table_index"], sort=False):
        table_map[(str(article_id), int(table_index))] = group.sort_values(["row_index", "cell_index"]).reset_index(drop=True)
    return table_map


def extract_mix_pairs(
    articles_df: pd.DataFrame,
    tables_df: pd.DataFrame,
    flavor_dict: FlavorDictionary,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recommended_tables = tables_df[tables_df["is_recommended_mix_section"].fillna(False)].copy()
    table_map = build_table_map(recommended_tables)

    for article in articles_df.itertuples(index=False):
        article_id = str(article.article_id)
        article_tables = [
            group
            for (group_article_id, _table_index), group in table_map.items()
            if group_article_id == article_id
        ]

        target_raw = clean_text(article.target_flavor)
        target_norm = normalize_single_flavor(target_raw, flavor_dict)
        if not article_tables:
            rows.append(
                {
                    "source_site": SOURCE_SITE,
                    "source_type": SOURCE_TYPE,
                    "article_id": article_id,
                    "article_url": article.article_url,
                    "article_title": article.article_title,
                    "target_brand": clean_text(article.brand),
                    "target_flavor_raw": target_raw,
                    "target_flavor_normalized": target_norm,
                    "recommended_flavor_raw": "",
                    "recommended_flavor_normalized": "",
                    "directed_pair_key": "",
                    "mix_pair_key": "",
                    "table_index": "",
                    "row_index": "",
                    "cell_index": "",
                    "header_cell_text": "",
                    "source_heading": clean_text(article.recommended_mix_heading),
                    "source_text": "",
                    "extraction_status": "excluded",
                    "exclusion_reason": "missing_recommended_mix_table",
                }
            )
            continue

        for table_df in article_tables:
            table_index = int(table_df["table_index"].iloc[0])
            source_heading = clean_text(table_df["section_heading"].iloc[0])
            row_indices = sorted(int(value) for value in table_df["row_index"].dropna().unique().tolist())
            if not row_indices:
                rows.append(
                    {
                        "source_site": SOURCE_SITE,
                        "source_type": SOURCE_TYPE,
                        "article_id": article_id,
                        "article_url": article.article_url,
                        "article_title": article.article_title,
                        "target_brand": clean_text(article.brand),
                        "target_flavor_raw": target_raw,
                        "target_flavor_normalized": target_norm,
                        "recommended_flavor_raw": "",
                        "recommended_flavor_normalized": "",
                        "directed_pair_key": "",
                        "mix_pair_key": "",
                        "table_index": table_index,
                        "row_index": "",
                        "cell_index": "",
                        "header_cell_text": "",
                        "source_heading": source_heading,
                        "source_text": "",
                        "extraction_status": "excluded",
                        "exclusion_reason": "empty_table_rows",
                    }
                )
                continue

            header_row_index = min(row_indices)
            header_map = (
                table_df[table_df["row_index"] == header_row_index]
                .sort_values("cell_index")
                .set_index("cell_index")["cell_text"]
                .to_dict()
            )
            data_rows = [row_index for row_index in row_indices if row_index != header_row_index]
            if not data_rows:
                rows.append(
                    {
                        "source_site": SOURCE_SITE,
                        "source_type": SOURCE_TYPE,
                        "article_id": article_id,
                        "article_url": article.article_url,
                        "article_title": article.article_title,
                        "target_brand": clean_text(article.brand),
                        "target_flavor_raw": target_raw,
                        "target_flavor_normalized": target_norm,
                        "recommended_flavor_raw": "",
                        "recommended_flavor_normalized": "",
                        "directed_pair_key": "",
                        "mix_pair_key": "",
                        "table_index": table_index,
                        "row_index": "",
                        "cell_index": "",
                        "header_cell_text": "",
                        "source_heading": source_heading,
                        "source_text": "",
                        "extraction_status": "excluded",
                        "exclusion_reason": "missing_data_row",
                    }
                )
                continue

            for row_index in data_rows:
                row_df = table_df[table_df["row_index"] == row_index].sort_values("cell_index")
                for cell in row_df.itertuples(index=False):
                    source_text = clean_text(cell.cell_text)
                    header_cell_text = clean_text(header_map.get(cell.cell_index, ""))
                    common = {
                        "source_site": SOURCE_SITE,
                        "source_type": SOURCE_TYPE,
                        "article_id": article_id,
                        "article_url": article.article_url,
                        "article_title": article.article_title,
                        "target_brand": clean_text(article.brand),
                        "target_flavor_raw": target_raw,
                        "target_flavor_normalized": target_norm,
                        "table_index": table_index,
                        "row_index": int(cell.row_index),
                        "cell_index": int(cell.cell_index),
                        "header_cell_text": header_cell_text,
                        "source_heading": source_heading,
                        "source_text": source_text,
                    }

                    if not source_text:
                        rows.append(
                            {
                                **common,
                                "recommended_flavor_raw": "",
                                "recommended_flavor_normalized": "",
                                "directed_pair_key": "",
                                "mix_pair_key": "",
                                "extraction_status": "excluded",
                                "exclusion_reason": "empty_cell",
                            }
                        )
                        continue

                    if source_text in NONE_MARKERS:
                        rows.append(
                            {
                                **common,
                                "recommended_flavor_raw": "",
                                "recommended_flavor_normalized": "",
                                "directed_pair_key": "",
                                "mix_pair_key": "",
                                "extraction_status": "excluded",
                                "exclusion_reason": "explicit_none_marker",
                            }
                        )
                        continue

                    raw_candidates = split_recommended_cell(source_text)
                    if not raw_candidates:
                        rows.append(
                            {
                                **common,
                                "recommended_flavor_raw": "",
                                "recommended_flavor_normalized": "",
                                "directed_pair_key": "",
                                "mix_pair_key": "",
                                "extraction_status": "excluded",
                                "exclusion_reason": "unparsed_cell",
                            }
                        )
                        continue

                    for raw_candidate in raw_candidates:
                        recommended_norm = normalize_single_flavor(raw_candidate, flavor_dict)
                        pair_key = ordered_pair_key(target_norm, recommended_norm)
                        directed_key = directed_pair_key(target_norm, recommended_norm)
                        extraction_status = "ok"
                        exclusion_reason = ""

                        if not target_raw:
                            extraction_status = "excluded"
                            exclusion_reason = "missing_target_flavor"
                        elif not target_norm:
                            extraction_status = "unresolved"
                            exclusion_reason = "target_flavor_unregistered"
                        elif not recommended_norm:
                            extraction_status = "unresolved"
                            exclusion_reason = "recommended_flavor_unregistered"
                        else:
                            left = target_norm or target_raw
                            right = recommended_norm or raw_candidate
                            if normalize_compare_text(left) == normalize_compare_text(right):
                                extraction_status = "excluded"
                                exclusion_reason = "self_pair"
                            else:
                                is_parent_child, parent_child_reason = detect_parent_child_pair(left, right)
                                if is_parent_child:
                                    extraction_status = "excluded"
                                    exclusion_reason = f"parent_child_pair:{parent_child_reason}"

                        if extraction_status != "ok":
                            pair_key = ""
                            directed_key = ""

                        rows.append(
                            {
                                **common,
                                "recommended_flavor_raw": raw_candidate,
                                "recommended_flavor_normalized": recommended_norm,
                                "directed_pair_key": directed_key,
                                "mix_pair_key": pair_key,
                                "extraction_status": extraction_status,
                                "exclusion_reason": exclusion_reason,
                            }
                        )

    extracted_df = pd.DataFrame(rows)
    if extracted_df.empty:
        return pd.DataFrame(
            columns=[
                "source_site",
                "source_type",
                "article_id",
                "article_url",
                "article_title",
                "target_brand",
                "target_flavor_raw",
                "target_flavor_normalized",
                "recommended_flavor_raw",
                "recommended_flavor_normalized",
                "directed_pair_key",
                "mix_pair_key",
                "table_index",
                "row_index",
                "cell_index",
                "header_cell_text",
                "source_heading",
                "source_text",
                "extraction_status",
                "exclusion_reason",
            ]
        )

    extracted_df["is_valid_pair"] = extracted_df["extraction_status"].eq("ok")
    extracted_df["is_self_pair"] = extracted_df["exclusion_reason"].eq("self_pair")
    duplicate_key = extracted_df["article_id"].astype(str) + "||" + extracted_df["mix_pair_key"].fillna("")
    extracted_df["is_duplicate_pair"] = duplicate_key.duplicated(keep="first") & extracted_df["mix_pair_key"].fillna("").ne("")
    return extracted_df


def build_dictionary_candidate_audit(
    extracted_df: pd.DataFrame,
    flavor_dict: FlavorDictionary,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in extracted_df.itertuples(index=False):
        if row.target_flavor_raw and not row.target_flavor_normalized:
            candidate_records[(str(row.target_flavor_raw), "target")].append(row._asdict())
        if row.recommended_flavor_raw and not row.recommended_flavor_normalized:
            candidate_records[(str(row.recommended_flavor_raw), "recommended")].append(row._asdict())

    for (raw_text, side), records in sorted(candidate_records.items()):
        article_urls = sorted({str(record["article_url"]) for record in records if record.get("article_url")})
        context_examples = []
        for record in records[:5]:
            context = str(record.get("source_text", "")).strip()
            if context and context not in context_examples:
                context_examples.append(context)
        normalization_candidate = infer_normalization_candidate(raw_text, flavor_dict.canonical_flavors)
        reason = "existing_master_unmatched"
        auto_add_allowed = "no"
        if normalization_candidate:
            reason = f"possible_alias_or_partial_match:{normalization_candidate}"
        rows.append(
            {
                "raw表記": raw_text,
                "出現記事数": len({str(record["article_id"]) for record in records}),
                "出現行数": len(records),
                "出現記事URL": " | ".join(article_urls),
                "target側かrecommended側か": side,
                "文脈": " || ".join(context_examples),
                "正規化候補": normalization_candidate,
                "自動追加可否": auto_add_allowed,
                "判断理由": reason,
            }
        )
    return pd.DataFrame(rows)


def build_unique_pairs(extracted_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = extracted_df[extracted_df["is_valid_pair"]].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "mix_pair_key",
                "flavor_a",
                "flavor_b",
                "lagos_article_count",
                "lagos_row_count",
                "lagos_article_urls",
                "target_flavor",
                "recommended_flavor",
                "directed_pair",
                "source_text",
            ]
        )

    rows: list[dict[str, Any]] = []
    for pair_key, group in valid_df.groupby("mix_pair_key", sort=True):
        flavors = str(pair_key).split("||")
        flavor_a = flavors[0] if len(flavors) > 0 else ""
        flavor_b = flavors[1] if len(flavors) > 1 else ""
        rows.append(
            {
                "mix_pair_key": pair_key,
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "lagos_article_count": int(group["article_id"].nunique()),
                "lagos_row_count": int(len(group)),
                "lagos_article_urls": " | ".join(sorted(group["article_url"].astype(str).unique().tolist())),
                "target_flavor": " | ".join(sorted(group["target_flavor_raw"].astype(str).unique().tolist())),
                "recommended_flavor": " | ".join(sorted(group["recommended_flavor_raw"].astype(str).unique().tolist())),
                "directed_pair": " | ".join(sorted(group["directed_pair_key"].astype(str).unique().tolist())),
                "source_text": " || ".join(sorted(group["source_text"].astype(str).unique().tolist())),
            }
        )
    unique_df = pd.DataFrame(rows)
    return unique_df.sort_values(["lagos_article_count", "lagos_row_count", "mix_pair_key"], ascending=[False, False, True]).reset_index(drop=True)


def build_extraction_summary(
    articles_df: pd.DataFrame,
    tables_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
) -> pd.DataFrame:
    recommended_tables = tables_df[tables_df["is_recommended_mix_section"].fillna(False)].copy()
    recommended_table_articles = int(recommended_tables["article_id"].nunique()) if not recommended_tables.empty else 0
    summary = {
        "article_count": len(articles_df),
        "articles_with_recommended_tables": recommended_table_articles,
        "recommended_table_count": int(recommended_tables[["article_id", "table_index"]].drop_duplicates().shape[0]) if not recommended_tables.empty else 0,
        "extracted_row_count": len(extracted_df),
        "valid_row_count": int(extracted_df["is_valid_pair"].sum()) if "is_valid_pair" in extracted_df.columns else 0,
        "unique_pair_count": int(extracted_df.loc[extracted_df["is_valid_pair"], "mix_pair_key"].nunique()) if "is_valid_pair" in extracted_df.columns else 0,
        "target_flavor_missing_count": int(extracted_df["target_flavor_raw"].fillna("").eq("").sum()),
        "target_flavor_unregistered_count": int(extracted_df["target_flavor_raw"].fillna("").ne("").sum() - extracted_df["target_flavor_normalized"].fillna("").ne("").sum()),
        "recommended_flavor_missing_count": int(extracted_df["recommended_flavor_raw"].fillna("").eq("").sum()),
        "recommended_flavor_unregistered_count": int(
            extracted_df["recommended_flavor_raw"].fillna("").ne("").sum()
            - extracted_df["recommended_flavor_normalized"].fillna("").ne("").sum()
        ),
        "self_pair_count": int(extracted_df["is_self_pair"].sum()) if "is_self_pair" in extracted_df.columns else 0,
        "duplicate_pair_count": int(extracted_df["is_duplicate_pair"].sum()) if "is_duplicate_pair" in extracted_df.columns else 0,
        "unresolved_row_count": int(extracted_df["extraction_status"].eq("unresolved").sum()),
        "excluded_row_count": int(extracted_df["extraction_status"].eq("excluded").sum()),
        "article_url_missing_count": int(extracted_df["article_url"].fillna("").eq("").sum()),
        "source_text_missing_count": int(extracted_df["source_text"].fillna("").eq("").sum()),
        "mix_pair_key_blank_count": int(extracted_df["mix_pair_key"].fillna("").eq("").sum()),
        "pair_key_order_normalized": bool(
            extracted_df.loc[extracted_df["mix_pair_key"].fillna("").ne(""), "mix_pair_key"]
            .eq(
                extracted_df.loc[extracted_df["mix_pair_key"].fillna("").ne(""), ["target_flavor_normalized", "recommended_flavor_normalized"]]
                .apply(lambda row: ordered_pair_key(row["target_flavor_normalized"], row["recommended_flavor_normalized"]), axis=1)
            )
            .all()
        ),
    }
    return pd.DataFrame([summary])


def build_pair_repetition_audit(extracted_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_df = extracted_df[extracted_df["is_valid_pair"]].copy()
    if valid_df.empty:
        audit_columns = [
            "mix_pair_key",
            "flavor_a",
            "flavor_b",
            "valid_row_count",
            "unique_directed_pair_count",
            "directed_pair_values",
            "article_count",
            "article_ids",
            "article_urls",
            "same_article_repeat_count",
            "max_rows_in_single_article",
            "row_index_values",
            "table_row_keys",
            "is_multi_article_pair",
            "is_repeated_mix_pair",
        ]
        summary_columns = ["metric", "value", "definition"]
        return pd.DataFrame(columns=audit_columns), pd.DataFrame(columns=summary_columns)

    full_row_duplicate_count = int(valid_df.duplicated().sum())
    same_table_row_duplicate_count = int(
        valid_df.duplicated(subset=["article_id", "table_index", "row_index", "directed_pair_key"]).sum()
    )

    pair_article_counts = valid_df.groupby(["mix_pair_key", "article_id"]).size().rename("rows_in_article").reset_index()
    mix_pair_counts = valid_df.groupby("mix_pair_key").size().rename("valid_row_count").reset_index()
    directed_counts = valid_df.groupby("directed_pair_key").size().rename("directed_pair_row_count").reset_index()
    mix_pair_article_counts = valid_df.groupby("mix_pair_key")["article_id"].nunique().rename("article_count").reset_index()

    rows: list[dict[str, Any]] = []
    for pair_key, group in valid_df.groupby("mix_pair_key", sort=True):
        flavors = str(pair_key).split("||")
        per_article = pair_article_counts[pair_article_counts["mix_pair_key"] == pair_key]
        rows.append(
            {
                "mix_pair_key": pair_key,
                "flavor_a": flavors[0] if len(flavors) > 0 else "",
                "flavor_b": flavors[1] if len(flavors) > 1 else "",
                "valid_row_count": int(len(group)),
                "unique_directed_pair_count": int(group["directed_pair_key"].nunique()),
                "directed_pair_values": " | ".join(sorted(group["directed_pair_key"].astype(str).unique().tolist())),
                "article_count": int(group["article_id"].nunique()),
                "article_ids": " | ".join(sorted(group["article_id"].astype(str).unique().tolist())),
                "article_urls": " | ".join(sorted(group["article_url"].astype(str).unique().tolist())),
                "same_article_repeat_count": int((per_article["rows_in_article"] - 1).clip(lower=0).sum()),
                "max_rows_in_single_article": int(per_article["rows_in_article"].max()),
                "row_index_values": " | ".join(sorted({f"{int(r.table_index)}:{int(r.row_index)}" for r in group.itertuples(index=False)})),
                "table_row_keys": " | ".join(
                    sorted({f"{r.article_id}::{int(r.table_index)}::{int(r.row_index)}" for r in group.itertuples(index=False)})
                ),
                "is_multi_article_pair": bool(group["article_id"].nunique() > 1),
                "is_repeated_mix_pair": bool(len(group) > 1),
            }
        )

    audit_df = pd.DataFrame(rows).sort_values(
        ["valid_row_count", "article_count", "mix_pair_key"], ascending=[False, False, True]
    ).reset_index(drop=True)

    repeated_directed = directed_counts[directed_counts["directed_pair_row_count"] > 1]
    repeated_mix_pairs = mix_pair_counts[mix_pair_counts["valid_row_count"] > 1]
    multi_article_pairs = mix_pair_article_counts[mix_pair_article_counts["article_count"] > 1]
    repeated_within_article_pairs = (
        pair_article_counts.groupby("mix_pair_key")["rows_in_article"].max().gt(1).sum()
    )

    summary_rows = [
        {
            "metric": "valid_row_count",
            "value": int(len(valid_df)),
            "definition": "有効抽出行の総数。",
        },
        {
            "metric": "unique_mix_pair_count",
            "value": int(valid_df["mix_pair_key"].nunique()),
            "definition": "順序なし pair key 単位で集約したユニークペア数。",
        },
        {
            "metric": "surplus_valid_rows_over_unique_pairs",
            "value": int(len(valid_df) - valid_df["mix_pair_key"].nunique()),
            "definition": "有効行数とユニークペア数の差分。複数出現分の総和。",
        },
        {
            "metric": "fully_identical_valid_rows",
            "value": full_row_duplicate_count,
            "definition": "有効行どうしで全列が完全一致する重複行数。",
        },
        {
            "metric": "same_article_table_row_duplicate_rows",
            "value": same_table_row_duplicate_count,
            "definition": "同一記事・同一表・同一行で同じ directed_pair_key が重複した行数。",
        },
        {
            "metric": "repeated_directed_pair_key_count",
            "value": int(len(repeated_directed)),
            "definition": "同一 directed_pair_key が2回以上出現した key 数。",
        },
        {
            "metric": "repeated_mix_pair_key_count",
            "value": int(len(repeated_mix_pairs)),
            "definition": "同一 mix_pair_key が2回以上出現した key 数。",
        },
        {
            "metric": "multi_article_mix_pair_count",
            "value": int(len(multi_article_pairs)),
            "definition": "複数記事に出現した mix_pair_key 数。",
        },
        {
            "metric": "within_article_repeated_mix_pair_count",
            "value": int(repeated_within_article_pairs),
            "definition": "同一記事内で2回以上出現した mix_pair_key 数。",
        },
        {
            "metric": "max_mix_pair_row_count",
            "value": int(mix_pair_counts["valid_row_count"].max()),
            "definition": "1つの mix_pair_key が持つ最大出現行数。",
        },
        {
            "metric": "max_mix_pair_article_count",
            "value": int(mix_pair_article_counts["article_count"].max()),
            "definition": "1つの mix_pair_key が出現した最大記事数。",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    return audit_df, summary_df


def build_dictionary_candidate_manual_review(
    extracted_df: pd.DataFrame,
    flavor_dict: FlavorDictionary,
    master_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    unresolved_df = extracted_df[
        (extracted_df["target_flavor_raw"].fillna("").ne("") & extracted_df["target_flavor_normalized"].fillna("").eq(""))
        | (
            extracted_df["recommended_flavor_raw"].fillna("").ne("")
            & extracted_df["recommended_flavor_normalized"].fillna("").eq("")
        )
    ].copy()

    candidate_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unresolved_df.itertuples(index=False):
        if row.target_flavor_raw and not row.target_flavor_normalized:
            candidate_records[str(row.target_flavor_raw)].append({"side": "target", **row._asdict()})
        if row.recommended_flavor_raw and not row.recommended_flavor_normalized:
            candidate_records[str(row.recommended_flavor_raw)].append({"side": "recommended", **row._asdict()})

    for raw_text, records in sorted(candidate_records.items()):
        similar_forms = list_similar_master_forms(raw_text, master_df)
        rule = CANDIDATE_MANUAL_RULES.get(raw_text, {})
        normalization_candidate = clean_text(rule.get("normalization_candidate") or infer_normalization_candidate(raw_text, flavor_dict.canonical_flavors))
        target_count = sum(1 for record in records if record["side"] == "target")
        recommended_count = sum(1 for record in records if record["side"] == "recommended")
        context_examples: list[str] = []
        for record in records:
            context = clean_text(record.get("source_text", ""))
            if context and context not in context_examples:
                context_examples.append(context)
        article_urls = sorted({clean_text(record.get("article_url", "")) for record in records if clean_text(record.get("article_url", ""))})
        candidate_rows.append(
            {
                "raw表記": raw_text,
                "正規化候補": normalization_candidate,
                "target側出現行数": target_count,
                "recommended側出現行数": recommended_count,
                "合計出現行数": len(records),
                "出現記事数": len({clean_text(record.get("article_id", "")) for record in records}),
                "出現記事URL": " | ".join(article_urls),
                "source_text": " || ".join(context_examples),
                "既存辞書に類似表記があるか": "yes" if similar_forms else "no",
                "類似表記": " | ".join(similar_forms),
                "完全な新規語か": "no" if similar_forms else "yes",
                "一般的なフレーバー名か": rule.get("general_flavor_name", "unclear"),
                "ブランド名・カテゴリ名・説明語の可能性": rule.get("brand_category_or_description_risk", "unclear"),
                "辞書追加候補": rule.get("dictionary_add_candidate", "見送り"),
                "要人手確認": rule.get("requires_manual_review", "yes"),
                "判断理由": rule.get(
                    "judgment_reason",
                    "既存辞書との対応関係が自明でないため、人手確認が必要。",
                ),
            }
        )
    return pd.DataFrame(candidate_rows)


def apply_override_normalization(raw_text: str, current_normalized: str, override_map: dict[str, str]) -> str:
    normalized = clean_text(current_normalized)
    if normalized:
        return normalized
    raw = clean_text(raw_text)
    if not raw:
        return ""
    return clean_text(override_map.get(raw, ""))


def simulate_extracted_pairs_with_override(extracted_df: pd.DataFrame, override_map: dict[str, str]) -> pd.DataFrame:
    simulated = extracted_df.copy()
    for idx, row in simulated.iterrows():
        target_norm = apply_override_normalization(
            row.get("target_flavor_raw", ""),
            row.get("target_flavor_normalized", ""),
            override_map,
        )
        recommended_norm = apply_override_normalization(
            row.get("recommended_flavor_raw", ""),
            row.get("recommended_flavor_normalized", ""),
            override_map,
        )

        simulated.at[idx, "target_flavor_normalized"] = target_norm
        simulated.at[idx, "recommended_flavor_normalized"] = recommended_norm

        extraction_status = clean_text(row.get("extraction_status", ""))
        exclusion_reason = clean_text(row.get("exclusion_reason", ""))
        pair_key = ""
        directed_key = ""

        if clean_text(row.get("target_flavor_raw", "")) and target_norm and recommended_norm:
            left = target_norm
            right = recommended_norm
            if normalize_compare_text(left) == normalize_compare_text(right):
                extraction_status = "excluded"
                exclusion_reason = "self_pair"
            else:
                is_parent_child, parent_child_reason = detect_parent_child_pair(left, right)
                if is_parent_child:
                    extraction_status = "excluded"
                    exclusion_reason = f"parent_child_pair:{parent_child_reason}"
                else:
                    extraction_status = "ok"
                    exclusion_reason = ""
                    pair_key = ordered_pair_key(left, right)
                    directed_key = directed_pair_key(left, right)
        elif clean_text(row.get("target_flavor_raw", "")) and not target_norm:
            extraction_status = "unresolved"
            exclusion_reason = "target_flavor_unregistered"
        elif clean_text(row.get("recommended_flavor_raw", "")) and not recommended_norm:
            extraction_status = "unresolved"
            exclusion_reason = "recommended_flavor_unregistered"

        simulated.at[idx, "extraction_status"] = extraction_status
        simulated.at[idx, "exclusion_reason"] = exclusion_reason
        simulated.at[idx, "mix_pair_key"] = pair_key
        simulated.at[idx, "directed_pair_key"] = directed_key

    simulated["is_valid_pair"] = simulated["extraction_status"].eq("ok")
    simulated["is_self_pair"] = simulated["exclusion_reason"].eq("self_pair")
    duplicate_key = simulated["article_id"].astype(str) + "||" + simulated["mix_pair_key"].fillna("")
    simulated["is_duplicate_pair"] = duplicate_key.duplicated(keep="first") & simulated["mix_pair_key"].fillna("").ne("")
    return simulated


def prepare_existing_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
    prepared = ranking_df.copy()
    prepared = prepared.sort_values(["rank_overall", "pair_count", "flavor_a", "flavor_b"]).reset_index(drop=True)
    return prepared


def compute_at_k_metrics(
    ranking_df: pd.DataFrame,
    lagos_pairs: set[str],
    ks: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in ks:
        top_df = ranking_df.nsmallest(min(k, len(ranking_df)), "rank_overall").copy()
        top_pairs = set(top_df["pair_key"].astype(str).tolist())
        common = top_pairs & lagos_pairs
        denominator_precision = k if k > 0 else 1
        denominator_recall = len(lagos_pairs) if lagos_pairs else 1
        union = top_pairs | lagos_pairs
        rows.append(
            {
                "k": k,
                "top_k_ranking_size": len(top_df),
                "common_pair_count": len(common),
                "precision_at_k": len(common) / denominator_precision,
                "recall_at_k": len(common) / denominator_recall if lagos_pairs else 0.0,
                "jaccard_at_k": len(common) / len(union) if union else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_common_pairs_with_existing(
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    ranking_lookup = ranking_df.set_index("pair_key")
    rows: list[dict[str, Any]] = []
    for row in unique_pairs_df.itertuples(index=False):
        if row.mix_pair_key not in ranking_lookup.index:
            continue
        ranking = ranking_lookup.loc[row.mix_pair_key]
        if isinstance(ranking, pd.DataFrame):
            ranking = ranking.iloc[0]
        rows.append(
            {
                "existing_rank": int(ranking["rank_overall"]),
                "flavor_a": ranking["flavor_a"],
                "flavor_b": ranking["flavor_b"],
                "pair_count": int(ranking["pair_count"]),
                "support": float(ranking["support"]),
                "lift": float(ranking["lift"]),
                "adjusted_lift": float(ranking["adjusted_lift"]),
                "centrality_mean": float(ranking["centrality_mean"]),
                "pos_ratio": float(ranking["smoothed_positive_ratio"]),
                "neg_ratio": float(ranking["smoothed_negative_ratio"]),
                "role_ratio": float(ranking["smoothed_role_ratio"]),
                "overall_score_v2": float(ranking["overall_score_v2"]),
                "tier": ranking["ranking_tier"],
                "LAGOS出現記事数": int(row.lagos_article_count),
                "LAGOS出現行数": int(row.lagos_row_count),
                "LAGOS記事URL": row.lagos_article_urls,
                "target_flavor": row.target_flavor,
                "recommended_flavor": row.recommended_flavor,
                "source_text": row.source_text,
                "directed_pair": row.directed_pair,
                "mix_pair_key": row.mix_pair_key,
            }
        )
    common_df = pd.DataFrame(rows)
    if common_df.empty:
        return pd.DataFrame(
            columns=[
                "existing_rank",
                "flavor_a",
                "flavor_b",
                "pair_count",
                "support",
                "lift",
                "adjusted_lift",
                "centrality_mean",
                "pos_ratio",
                "neg_ratio",
                "role_ratio",
                "overall_score_v2",
                "tier",
                "LAGOS出現記事数",
                "LAGOS出現行数",
                "LAGOS記事URL",
                "target_flavor",
                "recommended_flavor",
                "source_text",
                "directed_pair",
                "mix_pair_key",
            ]
        )
    return common_df.sort_values(["existing_rank", "mix_pair_key"]).reset_index(drop=True)


def build_lagos_only_pairs(
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    pair_features_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
) -> pd.DataFrame:
    ranking_pairs = set(ranking_df["pair_key"].astype(str).tolist())
    pair_feature_lookup = pair_features_df.set_index("pair_key") if not pair_features_df.empty else pd.DataFrame()
    excluded_lookup = excluded_df.set_index("pair_key") if not excluded_df.empty else pd.DataFrame()
    existing_flavors: set[str] = set(pair_features_df["flavor_a"].astype(str).tolist()) | set(
        pair_features_df["flavor_b"].astype(str).tolist()
    )

    rows: list[dict[str, Any]] = []
    for row in unique_pairs_df.itertuples(index=False):
        if row.mix_pair_key in ranking_pairs:
            continue
        flavor_status = "registered"
        reason = "不明"
        if row.flavor_a not in existing_flavors or row.flavor_b not in existing_flavors:
            missing = []
            if row.flavor_a not in existing_flavors:
                missing.append(row.flavor_a)
            if row.flavor_b not in existing_flavors:
                missing.append(row.flavor_b)
            flavor_status = "片方または両方が既存コーパス未出現"
            reason = f"既存コーパスで出現なし:{'|'.join(missing)}"
        elif not pair_feature_lookup.empty and row.mix_pair_key in pair_feature_lookup.index:
            feature_row = pair_feature_lookup.loc[row.mix_pair_key]
            if isinstance(feature_row, pd.DataFrame):
                feature_row = feature_row.iloc[0]
            if bool(feature_row.get("excluded_as_product_name_pair", False)):
                reason = "除外ルール該当:product_name_pair"
            elif bool(feature_row.get("is_parent_child_pair", False)):
                reason = "除外ルール該当:parent_child_pair"
            elif row.mix_pair_key in getattr(excluded_lookup, "index", []):
                excluded_row = excluded_lookup.loc[row.mix_pair_key]
                if isinstance(excluded_row, pd.DataFrame):
                    excluded_row = excluded_row.iloc[0]
                reason = f"ランキング対象外:{excluded_row.get('excluded_reason', 'unknown')}"
            elif int(feature_row.get("pair_count", 0)) < 2:
                reason = "pair_count閾値未満"
            else:
                reason = "ランキング対象外"
        else:
            reason = "共起なし"

        rows.append(
            {
                "flavor_a": row.flavor_a,
                "flavor_b": row.flavor_b,
                "LAGOS出現記事数": int(row.lagos_article_count),
                "LAGOS出現行数": int(row.lagos_row_count),
                "記事URL": row.lagos_article_urls,
                "target_flavor": row.target_flavor,
                "recommended_flavor": row.recommended_flavor,
                "source_text": row.source_text,
                "辞書登録状況": flavor_status,
                "既存ランキングに存在しない理由の候補": reason,
                "directed_pair": row.directed_pair,
                "mix_pair_key": row.mix_pair_key,
            }
        )
    lagos_only_df = pd.DataFrame(rows)
    if lagos_only_df.empty:
        return pd.DataFrame(
            columns=[
                "flavor_a",
                "flavor_b",
                "LAGOS出現記事数",
                "LAGOS出現行数",
                "記事URL",
                "target_flavor",
                "recommended_flavor",
                "source_text",
                "辞書登録状況",
                "既存ランキングに存在しない理由の候補",
                "directed_pair",
                "mix_pair_key",
            ]
        )
    return lagos_only_df.sort_values(["LAGOS出現記事数", "LAGOS出現行数", "mix_pair_key"], ascending=[False, False, True]).reset_index(drop=True)


def build_existing_topk_not_in_lagos(
    ranking_df: pd.DataFrame,
    lagos_pairs: set[str],
    top_k: int,
) -> pd.DataFrame:
    top_df = ranking_df.nsmallest(min(top_k, len(ranking_df)), "rank_overall").copy()
    missing_df = top_df[~top_df["pair_key"].isin(lagos_pairs)].copy()
    columns = ["rank_overall", "flavor_a", "flavor_b", "pair_count", "support", "lift", "overall_score_v2", "ranking_tier", "pair_key"]
    if missing_df.empty:
        return pd.DataFrame(columns=columns)
    return missing_df[columns].reset_index(drop=True)


def build_external_validation_statistics(
    extracted_df: pd.DataFrame,
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    lagos_pairs = set(unique_pairs_df["mix_pair_key"].astype(str).tolist())
    existing_pairs = set(ranking_df["pair_key"].astype(str).tolist())
    top10_common = int(agreement_df.loc[agreement_df["k"] == 10, "common_pair_count"].iloc[0]) if (agreement_df["k"] == 10).any() else 0
    top20_common = int(agreement_df.loc[agreement_df["k"] == 20, "common_pair_count"].iloc[0]) if (agreement_df["k"] == 20).any() else 0
    top50_common = int(agreement_df.loc[agreement_df["k"] == 50, "common_pair_count"].iloc[0]) if (agreement_df["k"] == 50).any() else 0
    return pd.DataFrame(
        [
            {
                "lagos_extracted_row_count": int(len(extracted_df)),
                "lagos_valid_row_count": int(extracted_df["is_valid_pair"].sum()) if "is_valid_pair" in extracted_df.columns else 0,
                "lagos_unique_pair_count": int(len(lagos_pairs)),
                "existing_ranking_pair_count": int(len(existing_pairs)),
                "common_pair_count": int(len(lagos_pairs & existing_pairs)),
                "lagos_only_pair_count": int(len(lagos_pairs - existing_pairs)),
                "existing_only_pair_count": int(len(existing_pairs - lagos_pairs)),
                "top10_common_count": top10_common,
                "top20_common_count": top20_common,
                "top50_common_count": top50_common,
            }
        ]
    )


def build_ranking_metadata_summary(
    ranking_df: pd.DataFrame,
    pair_features_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    ranking_path: str,
) -> pd.DataFrame:
    tier_values = sorted(ranking_df["ranking_tier"].dropna().astype(str).unique().tolist()) if "ranking_tier" in ranking_df.columns else []
    pair_features_pairs = int(pair_features_df["pair_key"].nunique()) if "pair_key" in pair_features_df.columns else 0
    excluded_pairs = int(excluded_df["pair_key"].nunique()) if "pair_key" in excluded_df.columns else 0
    total_pairs_before_ranking = pair_features_pairs
    rows = [
        {"metric": "ranking_file", "value": ranking_path, "definition": "今回の Top-K 比較に使用したランキングCSV。"},
        {"metric": "ranking_row_count", "value": int(len(ranking_df)), "definition": "比較対象ランキングの総行数。"},
        {"metric": "included_tiers", "value": " | ".join(tier_values), "definition": "ranking_tier 列に含まれる tier 値。"},
        {
            "metric": "rank_overall_min",
            "value": int(ranking_df["rank_overall"].min()) if not ranking_df.empty else "",
            "definition": "rank_overall の最小値。",
        },
        {
            "metric": "rank_overall_max",
            "value": int(ranking_df["rank_overall"].max()) if not ranking_df.empty else "",
            "definition": "rank_overall の最大値。",
        },
        {
            "metric": "pair_count_min",
            "value": int(ranking_df["pair_count"].min()) if not ranking_df.empty else "",
            "definition": "ランキングに残った候補の pair_count 最小値。",
        },
        {
            "metric": "pair_count_max",
            "value": int(ranking_df["pair_count"].max()) if not ranking_df.empty else "",
            "definition": "ランキングに残った候補の pair_count 最大値。",
        },
        {
            "metric": "same_sentence_evidence_document_count_min",
            "value": int(ranking_df["same_sentence_evidence_document_count"].min()) if not ranking_df.empty else "",
            "definition": "same_sentence_evidence_document_count の最小値。",
        },
        {
            "metric": "same_sentence_evidence_document_count_max",
            "value": int(ranking_df["same_sentence_evidence_document_count"].max()) if not ranking_df.empty else "",
            "definition": "same_sentence_evidence_document_count の最大値。",
        },
        {
            "metric": "all_context_score_eligible",
            "value": bool(ranking_df["context_score_eligible"].all()) if "context_score_eligible" in ranking_df.columns and not ranking_df.empty else False,
            "definition": "比較対象ランキングが文脈加点の適用条件を満たした候補のみか。",
        },
        {
            "metric": "excluded_pairs_removed",
            "value": excluded_pairs,
            "definition": "pair_expression_features から除外され pair_ranking_tier2 へ入っていないペア数。",
        },
        {
            "metric": "pair_expression_feature_pair_count",
            "value": pair_features_pairs,
            "definition": "特徴量計算後の全 pair_key 数。",
        },
        {
            "metric": "ranking_is_non_excluded_pool",
            "value": bool(pair_features_pairs == len(ranking_df) + excluded_pairs) if pair_features_pairs else False,
            "definition": "pair_ranking_tier2 が除外後の非除外候補全体になっているか。",
        },
        {
            "metric": "contains_tier1_rows",
            "value": bool("Tier1" in tier_values),
            "definition": "Tier1 候補を含むか。",
        },
        {
            "metric": "contains_tier2_rows",
            "value": bool("Tier2" in tier_values),
            "definition": "Tier2 候補を含むか。",
        },
        {
            "metric": "is_tier2_only",
            "value": bool(tier_values == ["Tier2"]),
            "definition": "Tier2 のみで構成されるか。",
        },
        {
            "metric": "usable_for_top50_comparison",
            "value": bool(len(ranking_df) >= 50),
            "definition": "Top50 比較に必要な件数を満たすか。",
        },
        {
            "metric": "better_full_ranking_candidate",
            "value": "outputs/extended_analysis_v2/pair_expression_features.csv",
            "definition": "より広い母集団に近い候補。ただし順位列がないため現状のTop-K比較には直接使っていない。",
        },
        {
            "metric": "pair_ranking_csv_note",
            "value": "outputs/extended_analysis_v2/pair_ranking.csv は 17 行の Tier1 のみ",
            "definition": "従来の pair_ranking.csv は Top50 比較に不足する。",
        },
    ]
    return pd.DataFrame(rows)


def build_topk_pair_audit(
    ranking_df: pd.DataFrame,
    unique_pairs_df: pd.DataFrame,
    ks: list[int],
) -> pd.DataFrame:
    lagos_lookup = unique_pairs_df.set_index("mix_pair_key") if not unique_pairs_df.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for k in ks:
        top_df = ranking_df.nsmallest(min(k, len(ranking_df)), "rank_overall").copy()
        for row in top_df.itertuples(index=False):
            present = row.pair_key in getattr(lagos_lookup, "index", [])
            lagos_row = lagos_lookup.loc[row.pair_key] if present else None
            if isinstance(lagos_row, pd.DataFrame):
                lagos_row = lagos_row.iloc[0]
            rows.append(
                {
                    "K": k,
                    "existing_rank": int(row.rank_overall),
                    "flavor_a": row.flavor_a,
                    "flavor_b": row.flavor_b,
                    "mix_pair_key": row.pair_key,
                    "present_in_lagos": bool(present),
                    "lagos_article_count": int(lagos_row["lagos_article_count"]) if present else 0,
                    "lagos_row_count": int(lagos_row["lagos_row_count"]) if present else 0,
                    "lagos_article_urls": lagos_row["lagos_article_urls"] if present else "",
                    "lagos_source_text": lagos_row["source_text"] if present else "",
                    "dictionary_status": "matched_valid_lagos_pair" if present else "not_present_in_lagos_baseline",
                    "comparison_status": "common_pair" if present else "ranking_only",
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(
    extracted_df: pd.DataFrame,
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    ks: list[int],
) -> dict[str, Any]:
    lagos_pairs = set(unique_pairs_df["mix_pair_key"].astype(str).tolist()) if not unique_pairs_df.empty else set()
    existing_pairs = set(ranking_df["pair_key"].astype(str).tolist()) if not ranking_df.empty else set()
    agreement_df = compute_at_k_metrics(ranking_df, lagos_pairs, ks)
    row: dict[str, Any] = {
        "lagos_extracted_row_count": int(len(extracted_df)),
        "lagos_valid_row_count": int(extracted_df["is_valid_pair"].sum()) if "is_valid_pair" in extracted_df.columns else 0,
        "lagos_unresolved_row_count": int(extracted_df["extraction_status"].eq("unresolved").sum()) if "extraction_status" in extracted_df.columns else 0,
        "lagos_excluded_row_count": int(extracted_df["extraction_status"].eq("excluded").sum()) if "extraction_status" in extracted_df.columns else 0,
        "lagos_unique_pair_count": int(len(lagos_pairs)),
        "common_pair_count": int(len(lagos_pairs & existing_pairs)),
        "lagos_only_pair_count": int(len(lagos_pairs - existing_pairs)),
        "existing_only_pair_count": int(len(existing_pairs - lagos_pairs)),
    }
    metric_lookup = agreement_df.set_index("k").to_dict("index") if not agreement_df.empty else {}
    for k in ks:
        metric = metric_lookup.get(k, {})
        row[f"top{k}_common_count"] = int(metric.get("common_pair_count", 0))
        row[f"precision_at_{k}"] = float(metric.get("precision_at_k", 0.0))
        row[f"recall_at_{k}"] = float(metric.get("recall_at_k", 0.0))
        row[f"jaccard_at_{k}"] = float(metric.get("jaccard_at_k", 0.0))
    return row


def build_dictionary_update_impact_simulation(
    extracted_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in ["baseline", "conservative", "extended"]:
        override_map = build_override_lookup(scenario)
        simulated_df = extracted_df if scenario == "baseline" else simulate_extracted_pairs_with_override(extracted_df, override_map)
        unique_pairs_df = build_unique_pairs(simulated_df)
        metrics = summarize_metrics(simulated_df, unique_pairs_df, ranking_df, BASELINE_TOP_KS)
        metrics.update(
            {
                "scenario": scenario,
                "override_mapping_count": len(override_map),
                "override_raws": " | ".join(sorted(override_map.keys())),
            }
        )
        rows.append(metrics)
    columns = [
        "scenario",
        "override_mapping_count",
        "override_raws",
        "lagos_extracted_row_count",
        "lagos_valid_row_count",
        "lagos_unresolved_row_count",
        "lagos_excluded_row_count",
        "lagos_unique_pair_count",
        "common_pair_count",
        "lagos_only_pair_count",
        "existing_only_pair_count",
        "top10_common_count",
        "top20_common_count",
        "top50_common_count",
        "precision_at_10",
        "precision_at_20",
        "precision_at_50",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "jaccard_at_10",
        "jaccard_at_20",
        "jaccard_at_50",
    ]
    return pd.DataFrame(rows)[columns]


def build_dictionary_update_impact_report(simulation_df: pd.DataFrame) -> str:
    baseline = simulation_df[simulation_df["scenario"] == "baseline"].iloc[0]
    lines = [
        "# Shisha LAGOS Dictionary Update Impact Simulation",
        "",
        "このファイルは辞書更新前の暫定シミュレーションであり、正式結果ではない。",
        "",
    ]
    for row in simulation_df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.scenario}",
                "",
                f"- override_mapping_count: {row.override_mapping_count}",
                f"- override_raws: {row.override_raws}",
                f"- valid_rows: {row.lagos_valid_row_count}",
                f"- unresolved_rows: {row.lagos_unresolved_row_count}",
                f"- unique_pairs: {row.lagos_unique_pair_count}",
                f"- common_pairs: {row.common_pair_count}",
                f"- Top10/20/50 common: {row.top10_common_count}/{row.top20_common_count}/{row.top50_common_count}",
                f"- Precision@10/20/50: {row.precision_at_10:.3f}/{row.precision_at_20:.3f}/{row.precision_at_50:.3f}",
                f"- Recall@10/20/50: {row.recall_at_10:.3f}/{row.recall_at_20:.3f}/{row.recall_at_50:.3f}",
                f"- Jaccard@10/20/50: {row.jaccard_at_10:.3f}/{row.jaccard_at_20:.3f}/{row.jaccard_at_50:.3f}",
                "",
            ]
        )
        if row.scenario != "baseline":
            delta_valid = int(row.lagos_valid_row_count - baseline["lagos_valid_row_count"])
            delta_common = int(row.common_pair_count - baseline["common_pair_count"])
            lines.append(f"- baseline差分: valid_rows {delta_valid:+d}, common_pairs {delta_common:+d}")
            lines.append("")
    return "\n".join(lines) + "\n"


def build_baseline_records(
    extracted_df: pd.DataFrame,
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    ranking_path: str,
    ranking_columns: list[str],
    dictionary_hash: str,
    git_commit_hash: str,
    executed_at: str | None = None,
) -> pd.DataFrame:
    if executed_at is None:
        executed_at = datetime.now().isoformat(timespec="seconds")
    metrics = summarize_metrics(extracted_df, unique_pairs_df, ranking_df, BASELINE_TOP_KS)
    records = [
        ("article_count", 13, "LAGOS 対象記事数。"),
        ("extracted_row_count", metrics["lagos_extracted_row_count"], "抽出総行数。"),
        ("valid_row_count", metrics["lagos_valid_row_count"], "有効行数。"),
        ("unresolved_row_count", metrics["lagos_unresolved_row_count"], "未解決行数。"),
        ("excluded_row_count", metrics["lagos_excluded_row_count"], "除外行数。"),
        ("unique_pair_count", metrics["lagos_unique_pair_count"], "ユニークペア数。"),
        ("common_pair_count", metrics["common_pair_count"], "既存ランキングとの共通ペア数。"),
        ("lagos_only_pair_count", metrics["lagos_only_pair_count"], "LAGOS 固有ペア数。"),
        ("top10_common_count", metrics["top10_common_count"], "Top10 一致数。"),
        ("top20_common_count", metrics["top20_common_count"], "Top20 一致数。"),
        ("top50_common_count", metrics["top50_common_count"], "Top50 一致数。"),
        ("precision_at_10", metrics["precision_at_10"], "Precision@10。"),
        ("precision_at_20", metrics["precision_at_20"], "Precision@20。"),
        ("precision_at_50", metrics["precision_at_50"], "Precision@50。"),
        ("recall_at_10", metrics["recall_at_10"], "Recall@10。"),
        ("recall_at_20", metrics["recall_at_20"], "Recall@20。"),
        ("recall_at_50", metrics["recall_at_50"], "Recall@50。"),
        ("jaccard_at_10", metrics["jaccard_at_10"], "Jaccard@10。"),
        ("jaccard_at_20", metrics["jaccard_at_20"], "Jaccard@20。"),
        ("jaccard_at_50", metrics["jaccard_at_50"], "Jaccard@50。"),
        ("ranking_file", ranking_path, "比較に用いたランキングファイル。"),
        ("ranking_columns", ", ".join(ranking_columns), "比較で参照した主要列。"),
        ("dictionary_hash", dictionary_hash, "辞書ファイルの sha256。"),
        ("executed_at", executed_at, "baseline 作成日時。"),
        ("git_commit_hash", git_commit_hash, "Git commit hash。"),
    ]
    return pd.DataFrame(records, columns=["metric", "value", "definition"])


def build_baseline_markdown(baseline_df: pd.DataFrame) -> str:
    lines = [
        "# Baseline: Shisha LAGOS External Validation",
        "",
    ]
    for row in baseline_df.itertuples(index=False):
        lines.append(f"- {row.metric}: {row.value}")
    return "\n".join(lines) + "\n"


def build_external_validation_report(
    extracted_df: pd.DataFrame,
    unique_pairs_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    common_df: pd.DataFrame,
    lagos_only_df: pd.DataFrame,
) -> str:
    lagos_pairs = set(unique_pairs_df["mix_pair_key"].astype(str).tolist())
    existing_pairs = set(ranking_df["pair_key"].astype(str).tolist())
    lines = [
        "# Shisha LAGOS External Validation Report",
        "",
        "## Summary",
        "",
        f"- LAGOS抽出行数: {len(extracted_df)}",
        f"- LAGOS有効行数: {int(extracted_df['is_valid_pair'].sum()) if 'is_valid_pair' in extracted_df.columns else 0}",
        f"- LAGOSユニークペア数: {len(lagos_pairs)}",
        f"- 既存ランキングとの一致ペア数: {len(lagos_pairs & existing_pairs)}",
        f"- LAGOSのみのペア数: {len(lagos_pairs - existing_pairs)}",
        f"- 既存ランキングのみにあるペア数: {len(existing_pairs - lagos_pairs)}",
        "",
        "## Agreement@K",
        "",
    ]
    for row in agreement_df.itertuples(index=False):
        lines.append(
            f"- K={row.k}: common={row.common_pair_count}, "
            f"precision={row.precision_at_k:.4f}, recall={row.recall_at_k:.4f}, jaccard={row.jaccard_at_k:.4f}"
        )
    lines.extend(["", "## Common Pairs", ""])
    if common_df.empty:
        lines.append("- なし")
    else:
        for row in common_df.head(10).itertuples(index=False):
            lines.append(
                f"- rank {row.existing_rank}: {row.flavor_a}||{row.flavor_b} "
                f"(LAGOS記事数={row.LAGOS出現記事数}, row数={row.LAGOS出現行数})"
            )
    lines.extend(["", "## LAGOS-only Pairs", ""])
    if lagos_only_df.empty:
        lines.append("- なし")
    else:
        for row in lagos_only_df.head(10).itertuples(index=False):
            lines.append(
                f"- {row.flavor_a}||{row.flavor_b}: {row.既存ランキングに存在しない理由の候補}"
            )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "- 一致率は推薦精度ではなく、編集記事型の小規模補助データソースとの一致・被覆率として扱う。",
            "- LAGOSは13記事であり、既存のユーザー投稿型レビューとは source_type が異なる。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_paper_validation_draft(
    stats_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
) -> str:
    stats = stats_df.iloc[0]
    metric_lookup = agreement_df.set_index("k").to_dict("index")
    top10 = metric_lookup.get(10, {})
    top20 = metric_lookup.get(20, {})
    top50 = metric_lookup.get(50, {})
    lines = [
        "# Paper Draft: External Validation with Shisha LAGOS",
        "",
        "## 日本語案",
        "",
        (
            f"補助的な外部比較として，Shisha Cafe & Bar LAGOS の編集記事型データに明示されたおすすめミックス表を構造化し，"
            f"既存レビューから得たランキング候補との一致を調べた。LAGOS 側は 13 記事の小規模データであり，"
            "ユーザー投稿型レビューとは性質が異なるため，推薦精度ではなく，独立した編集記事型ソースとの外部的一致として扱う。"
        ),
        (
            f"LAGOS から得られた有効ユニークペア数は {int(stats['lagos_unique_pair_count'])} 組であり，"
            f"既存ランキング上位10件との一致数は {int(stats['top10_common_count'])}，"
            f"上位20件との一致数は {int(stats['top20_common_count'])}，"
            f"上位50件との一致数は {int(stats['top50_common_count'])} であった。"
        ),
        (
            f"一致率・被覆率としてみると，Precision@10={top10.get('precision_at_k', 0.0):.3f}，"
            f"Precision@20={top20.get('precision_at_k', 0.0):.3f}，Precision@50={top50.get('precision_at_k', 0.0):.3f} であり，"
            f"Recall@10={top10.get('recall_at_k', 0.0):.3f}，Recall@20={top20.get('recall_at_k', 0.0):.3f}，"
            f"Recall@50={top50.get('recall_at_k', 0.0):.3f} であった。"
        ),
        "この結果はあくまで小規模な補助的比較であり，LAGOS に存在しない候補を不適切とみなすものではない。",
        "",
        "## English Draft",
        "",
        (
            "As an auxiliary external comparison, we structured the recommended-mix tables explicitly stated in the "
            "editorial articles from Shisha Cafe & Bar LAGOS and compared them with the ranked candidates obtained "
            "from the existing review corpus. Because the LAGOS corpus consists of only 13 editorial articles and "
            "differs in source type from user-review-like data, we treat the results as external agreement rather than recommendation accuracy."
        ),
        (
            f"The number of valid unique LAGOS pairs was {int(stats['lagos_unique_pair_count'])}, and the overlap with the existing ranking was "
            f"{int(stats['top10_common_count'])} pairs in the top 10, {int(stats['top20_common_count'])} pairs in the top 20, "
            f"and {int(stats['top50_common_count'])} pairs in the top 50."
        ),
        (
            f"In terms of agreement/coverage, Precision@10={top10.get('precision_at_k', 0.0):.3f}, "
            f"Precision@20={top20.get('precision_at_k', 0.0):.3f}, Precision@50={top50.get('precision_at_k', 0.0):.3f}, "
            f"while Recall@10={top10.get('recall_at_k', 0.0):.3f}, Recall@20={top20.get('recall_at_k', 0.0):.3f}, "
            f"and Recall@50={top50.get('recall_at_k', 0.0):.3f}."
        ),
        "These values should not be interpreted as recommendation accuracy or as definitive ground-truth validation.",
    ]
    return "\n".join(lines) + "\n"
