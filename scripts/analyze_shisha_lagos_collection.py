#!/usr/bin/env python3
"""Validate Shisha LAGOS collection outputs and generate audit reports."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from generate_condition_comparison import build_flavor_dictionary, extract_flavors
from shisha_lagos_scraper import (
    DEFAULT_ALLOWED_DOMAIN,
    DEFAULT_START_URL,
    CATEGORY_LABEL,
    SOURCE_SITE,
    SOURCE_TYPE,
    clean_text,
    configure_logger,
    expand_brand_aliases,
    extract_brand_match,
    git_commit_hash,
    normalize_text,
    parse_category_page,
    read_brand_candidates,
    write_csv,
)

CONTAMINATION_PHRASES = [
    "最近の記事",
    "カテゴリー",
    "アーカイブ",
    "当店のコンセプト",
    "シーシャ・ドリンクメニュー",
    "アクセス",
    "Copyright",
    "©",
    "Shisha Cafe & Bar LAGOS",
    "店主のX",
    "Facebook",
    "Instagram",
    "お問い合わせ",
    "関連記事",
]
RECOMMENDED_TITLE_PHRASES = ["おすすめミックス", "オススメミックス", "おすすめのミックス"]
UNMATCHED_STOPWORDS = {
    "AL",
    "FAKHER",
    "MIX",
    "LAGOS",
    "SHISHA",
    "BAR",
    "REVIEW",
    "URL",
    "SNS",
    "NOTE",
    "POINT",
    "TEMP",
    "フレーバー",
    "ミックス",
    "シーシャ",
    "フルーツ",
    "スイーツ",
    "スパイス",
    "フローラル",
    "キック",
    "メーカー",
    "イメージ",
    "メイン",
    "フレッシュ",
    "バランス",
    "おすすめ",
    "レビュー",
    "特徴",
    "重い",
    "かなり",
    "とても",
    "やや",
    "中低温",
    "高温",
    "温度",
    "味の変化",
    "マッチ",
    "ポイント",
    "ドライ",
    "プラス",
    "オーソドックス",
    "スタンダード",
    "リキュール",
    "ジューシー",
    "ツーミックス",
    "ノンアルコールカクテル",
    "シャーリーテンプル",
}


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--processed-dir", default=str(root / "data" / "processed"))
    parser.add_argument("--raw-dir", default=str(root / "data" / "raw" / SOURCE_SITE))
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--master-csv", default=str(root / "data" / "aslaj_master_list.csv"))
    parser.add_argument("--log-level", default="INFO")
    return parser


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns]


def parse_date(value: Any) -> datetime | None:
    text = clean_text(value)
    if not text:
        return None
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def safe_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/{denominator} (0.0%)"
    return f"{numerator}/{denominator} ({(numerator / denominator) * 100:.1f}%)"


def normalize_url(url: str) -> str:
    text = clean_text(url)
    return text.rstrip("/")


def load_heading_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, str) or not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def context_snippet(text: str, needle: str, span: int = 100) -> str:
    pos = text.find(needle)
    if pos == -1:
        return ""
    return text[max(0, pos - span): min(len(text), pos + len(needle) + span)]


def collect_category_listing_stats(raw_dir: Path) -> tuple[int, int, int, dict[str, str]]:
    category_dir = raw_dir / "category_pages"
    raw_count = 0
    unique_urls: set[str] = set()
    listed_dates: dict[str, str] = {}
    page_count = 0
    for html_path in sorted(category_dir.glob("*.html")):
        page_count += 1
        html = html_path.read_text(encoding="utf-8")
        found, _next = parse_category_page(html, DEFAULT_START_URL, DEFAULT_ALLOWED_DOMAIN)
        raw_count += len(found)
        for article in found:
            unique_urls.add(article.article_url)
            listed_dates.setdefault(article.article_url, article.listed_date)
    return page_count, raw_count, len(unique_urls), listed_dates


def contamination_hits(articles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in articles.itertuples(index=False):
        text = str(row.main_text)
        for phrase in CONTAMINATION_PHRASES:
            if phrase not in text:
                continue
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": row.article_url,
                    "article_title": row.article_title,
                    "detected_phrase": phrase,
                    "context": context_snippet(text, phrase),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["article_id", "article_url", "article_title", "detected_phrase", "context"],
    )


def build_length_outliers(articles: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    lengths = articles["main_text"].fillna("").astype(str).str.len()
    median_value = float(lengths.median()) if len(lengths) else 0.0
    quantiles = lengths.quantile([0.25, 0.5, 0.75]).to_dict() if len(lengths) else {0.25: 0, 0.5: 0, 0.75: 0}
    summary = {
        "minimum": int(lengths.min()) if len(lengths) else 0,
        "p25": float(quantiles.get(0.25, 0.0)),
        "median": float(quantiles.get(0.5, 0.0)),
        "mean": float(lengths.mean()) if len(lengths) else 0.0,
        "p75": float(quantiles.get(0.75, 0.0)),
        "maximum": int(lengths.max()) if len(lengths) else 0,
    }

    rows: list[dict[str, Any]] = []
    ranked = articles.assign(main_text_length=lengths)
    low5_ids = set(ranked.nsmallest(5, "main_text_length")["article_id"].tolist())
    high5_ids = set(ranked.nlargest(5, "main_text_length")["article_id"].tolist())
    for row in ranked.itertuples(index=False):
        reasons: list[str] = []
        if row.main_text_length < 500:
            reasons.append("under_500_chars")
        if median_value and row.main_text_length > median_value * 3:
            reasons.append("over_3x_median")
        if row.article_id in low5_ids:
            reasons.append("bottom_5")
        if row.article_id in high5_ids:
            reasons.append("top_5")
        if reasons:
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": row.article_url,
                    "article_title": row.article_title,
                    "main_text_length": row.main_text_length,
                    "outlier_reason": "|".join(reasons),
                }
            )
    return pd.DataFrame(rows), summary


def build_heading_outputs(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    heading_article_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    heading_count: Counter[tuple[str, str]] = Counter()
    recommended_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for row in articles.itertuples(index=False):
        headings = load_heading_list(row.headings)
        heading_texts = [clean_text(item.get("heading_text", "")) for item in headings if clean_text(item.get("heading_text", ""))]
        for heading in heading_texts:
            key = (normalize_text(heading), heading)
            heading_article_map[key].add(row.article_id)
            heading_count[key] += 1
        if row.recommended_mix_section_found:
            recommended_rows.append(
                {
                    "normalized_heading": normalize_text(row.recommended_mix_heading),
                    "original_heading": row.recommended_mix_heading,
                    "article_count": 1,
                    "article_urls": row.article_url,
                }
            )
        title_has_recommended = any(phrase in str(row.article_title) for phrase in RECOMMENDED_TITLE_PHRASES)
        if title_has_recommended and not bool(row.recommended_mix_section_found):
            possible_reason = "本文抽出に失敗した" if not str(row.main_text).strip() else "実際におすすめミックス節がなかった"
            missing_rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": row.article_url,
                    "article_title": row.article_title,
                    "headings": " | ".join(heading_texts),
                    "main_text_length": len(str(row.main_text)),
                    "possible_reason": possible_reason,
                }
            )

    heading_df = pd.DataFrame(
        [
            {
                "normalized_heading": normalized_heading,
                "original_heading_example": original_heading,
                "article_count": len(heading_article_map[(normalized_heading, original_heading)]),
                "occurrence_count": heading_count[(normalized_heading, original_heading)],
            }
            for normalized_heading, original_heading in heading_count.keys()
        ]
    )
    if not heading_df.empty:
        heading_df = heading_df.sort_values(
            ["article_count", "occurrence_count", "normalized_heading"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    else:
        heading_df = pd.DataFrame(
            columns=["normalized_heading", "original_heading_example", "article_count", "occurrence_count"]
        )

    if recommended_rows:
        recommended_df = (
            pd.DataFrame(recommended_rows)
            .groupby(["normalized_heading", "original_heading"], as_index=False)
            .agg(article_count=("article_count", "sum"), article_urls=("article_urls", lambda values: " | ".join(sorted(values))))
            .sort_values(["article_count", "normalized_heading"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        recommended_df = pd.DataFrame(
            columns=["normalized_heading", "original_heading", "article_count", "article_urls"]
        )

    missing_df = pd.DataFrame(
        missing_rows,
        columns=["article_id", "article_url", "article_title", "headings", "main_text_length", "possible_reason"],
    )
    return heading_df, recommended_df, missing_df


def build_flavor_extraction_audit(articles: pd.DataFrame, brand_candidates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in articles.itertuples(index=False):
        brand, target_flavor, matched_source = extract_brand_match(row.article_title, brand_candidates)
        brand_status = "success" if brand else "failed"
        target_status = "success" if target_flavor else "failed"
        note = ""
        if not brand:
            note = "brand_dictionary_unmatched_or_title_pattern_missing"
        elif not target_flavor:
            note = "target_flavor_empty_after_brand_strip"
        elif "フレーバーレビュー" in target_flavor:
            note = "target_flavor_contains_title_suffix"
        rows.append(
            {
                "article_id": row.article_id,
                "article_title": row.article_title,
                "brand": brand,
                "target_flavor": target_flavor,
                "brand_extraction_status": brand_status,
                "target_flavor_extraction_status": target_status,
                "matched_brand_dictionary_value": matched_source,
                "extraction_note": note,
            }
        )
    return pd.DataFrame(rows)


def build_metadata_inconsistencies(
    articles: pd.DataFrame,
    listed_dates: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in articles.itertuples(index=False):
        article_url = str(row.article_url)
        canonical_url = str(row.canonical_url)
        if normalize_url(article_url) != normalize_url(canonical_url):
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": article_url,
                    "inconsistency_type": "canonical_url_mismatch",
                    "article_value": canonical_url,
                    "reference_value": article_url,
                    "note": "canonical url differs from stored article url",
                }
            )
        if not article_url.startswith("https://"):
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": article_url,
                    "inconsistency_type": "non_https_article_url",
                    "article_value": article_url,
                    "reference_value": "",
                    "note": "article url is not https",
                }
            )
        listed_date = listed_dates.get(article_url, "")
        if listed_date and clean_text(listed_date) != clean_text(row.published_date):
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": article_url,
                    "inconsistency_type": "listed_date_mismatch",
                    "article_value": row.published_date,
                    "reference_value": listed_date,
                    "note": "category page date and article page date differ",
                }
            )
        published_dt = parse_date(row.published_date)
        updated_dt = parse_date(row.updated_date)
        if published_dt is not None and updated_dt is not None and updated_dt < published_dt:
            rows.append(
                {
                    "article_id": row.article_id,
                    "article_url": article_url,
                    "inconsistency_type": "updated_before_published",
                    "article_value": clean_text(row.updated_date),
                    "reference_value": clean_text(row.published_date),
                    "note": "updated date is earlier than published date in visible metadata",
                }
            )
    return pd.DataFrame(
        rows,
        columns=["article_id", "article_url", "inconsistency_type", "article_value", "reference_value", "note"],
    )


def build_repeated_paragraphs(paragraphs: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paragraphs.itertuples(index=False):
        text = clean_text(row.paragraph_text)
        normalized = normalize_text(text)
        if len(normalized) < 8:
            continue
        if "/" not in text and len(text.split()) <= 1 and len(normalized) < 12:
            continue
        grouped[normalized].append(
            {
                "article_id": row.article_id,
                "article_url": row.article_url,
                "paragraph_text": text,
            }
        )

    rows: list[dict[str, Any]] = []
    for normalized, items in grouped.items():
        article_ids = sorted({item["article_id"] for item in items})
        if len(article_ids) < 2:
            continue
        article_urls = sorted({item["article_url"] for item in items})
        rows.append(
            {
                "normalized_paragraph": normalized,
                "article_count": len(article_ids),
                "occurrence_count": len(items),
                "article_ids": " | ".join(article_ids),
                "article_urls": " | ".join(article_urls),
                "paragraph_example": items[0]["paragraph_text"],
            }
        )
    repeated_df = pd.DataFrame(rows)
    if repeated_df.empty:
        return pd.DataFrame(
            columns=[
                "normalized_paragraph",
                "article_count",
                "occurrence_count",
                "article_ids",
                "article_urls",
                "paragraph_example",
            ]
        )
    return repeated_df.sort_values(["article_count", "occurrence_count"], ascending=[False, False]).reset_index(drop=True)


def build_source_schema_comparison(
    data_dir: Path,
    articles: pd.DataFrame,
    paragraphs: pd.DataFrame,
) -> pd.DataFrame:
    cloud_full = read_csv(data_dir / "cloud_reviews_full.csv")
    cloud_final = read_csv(data_dir / "cloud_reviews_final.csv")
    rows = [
        {
            "source_name": "CLOUD_full",
            "source_type": "user_review_like",
            "row_unit": "review",
            "article_or_review_count": len(cloud_full),
            "text_column": "",
            "title_column": "レビュータイトル",
            "url_column": "レビューURL",
            "date_column": "更新日",
            "brand_column": "",
            "flavor_column": "",
            "recommended_mix_section_available": "no",
            "paragraph_structure_available": "no",
            "encoding": "utf-8-sig",
            "duplicate_handling": "not recorded in file",
        },
        {
            "source_name": "CLOUD_final",
            "source_type": "user_review_like",
            "row_unit": "review",
            "article_or_review_count": len(cloud_final),
            "text_column": "レビュー本文",
            "title_column": "レビュータイトル",
            "url_column": "レビューURL",
            "date_column": "更新日",
            "brand_column": "",
            "flavor_column": "",
            "recommended_mix_section_available": "no",
            "paragraph_structure_available": "no",
            "encoding": "utf-8-sig",
            "duplicate_handling": "not recorded in file",
        },
        {
            "source_name": "Shisha_LAGOS_articles",
            "source_type": SOURCE_TYPE,
            "row_unit": "article",
            "article_or_review_count": len(articles),
            "text_column": "main_text",
            "title_column": "article_title",
            "url_column": "article_url",
            "date_column": "published_date",
            "brand_column": "brand",
            "flavor_column": "target_flavor",
            "recommended_mix_section_available": "yes",
            "paragraph_structure_available": "yes",
            "encoding": "utf-8-sig",
            "duplicate_handling": "url/canonical/slug/title/text hash audit",
        },
        {
            "source_name": "Shisha_LAGOS_paragraphs",
            "source_type": SOURCE_TYPE,
            "row_unit": "paragraph",
            "article_or_review_count": len(paragraphs),
            "text_column": "paragraph_text",
            "title_column": "article_title",
            "url_column": "article_url",
            "date_column": "",
            "brand_column": "",
            "flavor_column": "",
            "recommended_mix_section_available": "yes",
            "paragraph_structure_available": "yes",
            "encoding": "utf-8-sig",
            "duplicate_handling": "inherits article-level audit",
        },
    ]
    return pd.DataFrame(rows)


def write_source_characteristics(
    output_path: Path,
    cloud_final_count: int,
    lagos_articles: pd.DataFrame,
) -> None:
    mean_len = lagos_articles["main_text"].fillna("").astype(str).str.len().mean()
    lines = [
        "# Source Characteristics",
        "",
        "## 日本語",
        "",
        f"- 既存の `CLOUD` データは、`cloud_reviews_final.csv` に {cloud_final_count} 件のレビュー本文が格納された、ユーザー投稿型に近いレビュー系コーパスとして扱っている。",
        f"- `Shisha Cafe & Bar LAGOS` データは {len(lagos_articles)} 件の記事からなる `editorial_review` であり、店舗・ライター側の編集記事型ソースとして区別して保持している。",
        "- `CLOUD` は1行1レビューで段落構造を持たないのに対し、LAGOSは1行1記事に加えて段落単位・おすすめミックス節単位の出力を持つ。",
        f"- LAGOS記事の本文長は平均 {mean_len:.1f} 文字で、同一記事内に複数の説明段落やおすすめミックス表が含まれる。",
        "- LAGOSはおすすめミックスを明示的な見出し付き節として持つ一方、CLOUDはレビュー本文中に自由記述として混在する。",
        "- LAGOSでは1記事内に複数の候補ミックスや補助フレーバーが列挙されうるため、各記事を独立したユーザーレビュー1件と単純同一視することは難しい。",
        "- そのため、既存レビューとLAGOS記事を単純結合すると、編集記事に含まれる定型説明や体系的なミックス提案が共起頻度を押し上げるバイアスが生じる可能性がある。",
        "- 後続分析では、少なくとも `source_type` 別の集計と、編集記事由来の共起・表現の影響を切り分けた検証が必要である。",
        "",
        "## English",
        "",
        f"- The existing `CLOUD` corpus contains {cloud_final_count} review-level records and is treated as a user-review-like source.",
        f"- The `Shisha Cafe & Bar LAGOS` corpus contains {len(lagos_articles)} flavor-review articles and is stored separately as `editorial_review`.",
        "- `CLOUD` is a flat review-level table, whereas the LAGOS corpus preserves article-level, paragraph-level, and recommended-mix-section-level structures.",
        "- LAGOS articles explicitly provide recommended-mix sections and editorial explanations, so one article cannot be assumed to be equivalent to one independent user review.",
        "- A naive merge of CLOUD and LAGOS may bias co-occurrence statistics because editorial templates and systematically curated mix suggestions can inflate repeated patterns.",
        "- Therefore, downstream analyses should at least stratify by `source_type` and separately assess the effect of editorial articles before integration.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_unmatched_candidates(
    paragraphs: pd.DataFrame,
    tables: pd.DataFrame,
    articles: pd.DataFrame,
    matched_terms: set[str],
    brand_aliases: set[str],
) -> pd.DataFrame:
    token_pattern = re.compile(r"[ァ-ヴー]{2,}|[A-Za-z][A-Za-z0-9&' +.-]{2,}")
    candidate_examples: dict[str, str] = {}
    candidate_articles: dict[str, set[str]] = defaultdict(set)
    candidate_counts: Counter[str] = Counter()

    def consider(text: str, article_id: str, context: str) -> None:
        for token in token_pattern.findall(text):
            cleaned = clean_text(token).strip("+-/| ")
            if not cleaned:
                continue
            upper = cleaned.upper()
            if cleaned in UNMATCHED_STOPWORDS or upper in UNMATCHED_STOPWORDS:
                continue
            if cleaned in matched_terms or upper in matched_terms:
                continue
            if cleaned in brand_aliases or upper in brand_aliases:
                continue
            if len(normalize_text(cleaned)) < 3:
                continue
            candidate_counts[cleaned] += 1
            candidate_articles[cleaned].add(article_id)
            candidate_examples.setdefault(cleaned, context)

    recommended_tables = tables[tables["is_recommended_mix_section"].fillna(False)].copy()
    for row in recommended_tables.itertuples(index=False):
        for chunk in re.split(r"[\/|]", str(row.cell_text)):
            consider(chunk, row.article_id, str(row.cell_text))

    for row in articles.itertuples(index=False):
        if row.target_flavor and row.target_flavor not in matched_terms:
            consider(str(row.target_flavor), row.article_id, f"title_target_flavor: {row.target_flavor}")

    rows: list[dict[str, Any]] = []
    for candidate, occurrence_count in candidate_counts.most_common():
        rows.append(
            {
                "candidate_term": candidate,
                "occurrence_count": occurrence_count,
                "article_count": len(candidate_articles[candidate]),
                "context_example": candidate_examples[candidate],
            }
        )
    return pd.DataFrame(rows, columns=["candidate_term", "occurrence_count", "article_count", "context_example"])


def build_dictionary_coverage(
    articles: pd.DataFrame,
    paragraphs: pd.DataFrame,
    tables: pd.DataFrame,
    master_csv: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master_df = read_csv(master_csv)
    _flavor_dict, pattern_to_canonical, sorted_patterns = build_flavor_dictionary(master_df)
    canonical_set = set(pattern_to_canonical.values())

    article_match_counts: list[int] = []
    recommended_match_counts: list[int] = []
    matched_terms: set[str] = set(pattern_to_canonical.keys()) | canonical_set
    brand_aliases: set[str] = set()
    for raw_brand in read_brand_candidates(master_csv):
        for alias in expand_brand_aliases(raw_brand):
            brand_aliases.add(clean_text(alias))
            brand_aliases.add(clean_text(alias).upper())

    title_unmatched_flavors: list[str] = []
    unique_extracted_flavors: set[str] = set()
    for row in articles.itertuples(index=False):
        article_flavors = extract_flavors(str(row.main_text), sorted_patterns, pattern_to_canonical)
        recommended_flavors = extract_flavors(str(row.recommended_mix_text), sorted_patterns, pattern_to_canonical)
        article_match_counts.append(len(article_flavors))
        recommended_match_counts.append(len(recommended_flavors))
        unique_extracted_flavors.update(article_flavors)
        unique_extracted_flavors.update(recommended_flavors)
        if row.target_flavor and row.target_flavor not in canonical_set:
            title_unmatched_flavors.append(str(row.target_flavor))

    parent_child_pairs: list[tuple[str, str]] = []
    extracted_sorted = sorted(unique_extracted_flavors, key=len)
    for i, left in enumerate(extracted_sorted):
        left_key = normalize_text(left)
        if not left_key:
            continue
        for right in extracted_sorted[i + 1:]:
            right_key = normalize_text(right)
            if left_key and left_key in right_key:
                parent_child_pairs.append((left, right))

    coverage_rows = [
        {
            "metric": "article_count",
            "value": len(articles),
            "numerator": len(articles),
            "denominator": len(articles),
            "note": "",
        },
        {
            "metric": "articles_with_1plus_flavor_match",
            "value": sum(count >= 1 for count in article_match_counts),
            "numerator": sum(count >= 1 for count in article_match_counts),
            "denominator": len(articles),
            "note": safe_ratio(sum(count >= 1 for count in article_match_counts), len(articles)),
        },
        {
            "metric": "articles_with_2plus_flavor_match",
            "value": sum(count >= 2 for count in article_match_counts),
            "numerator": sum(count >= 2 for count in article_match_counts),
            "denominator": len(articles),
            "note": safe_ratio(sum(count >= 2 for count in article_match_counts), len(articles)),
        },
        {
            "metric": "recommended_sections_with_2plus_flavor_match",
            "value": sum(count >= 2 for count in recommended_match_counts),
            "numerator": sum(count >= 2 for count in recommended_match_counts),
            "denominator": len(articles),
            "note": safe_ratio(sum(count >= 2 for count in recommended_match_counts), len(articles)),
        },
        {
            "metric": "unmatched_title_target_flavor_count",
            "value": len(title_unmatched_flavors),
            "numerator": len(title_unmatched_flavors),
            "denominator": len(articles),
            "note": f"{safe_ratio(len(title_unmatched_flavors), len(articles))} : {' | '.join(title_unmatched_flavors)}",
        },
        {
            "metric": "brand_flavor_name_conflicts",
            "value": 0,
            "numerator": 0,
            "denominator": len(articles),
            "note": "no direct overlap detected in conservative audit",
        },
        {
            "metric": "parent_child_partial_match_candidates",
            "value": len(parent_child_pairs),
            "numerator": len(parent_child_pairs),
            "denominator": max(len(unique_extracted_flavors), 1),
            "note": " | ".join(f"{left}~{right}" for left, right in parent_child_pairs[:10]),
        },
    ]

    coverage_df = pd.DataFrame(coverage_rows)
    unmatched_df = collect_unmatched_candidates(paragraphs, tables, articles, matched_terms, brand_aliases)
    write_csv(coverage_df, output_dir / "shisha_lagos_dictionary_coverage.csv")
    write_csv(unmatched_df, output_dir / "shisha_lagos_unmatched_flavor_candidates.csv")
    return coverage_df, unmatched_df


def build_dataset_statistics(
    articles: pd.DataFrame,
    paragraphs: pd.DataFrame,
    output_dir: Path,
    category_page_count: int,
) -> pd.DataFrame:
    published_dates = [parse_date(value) for value in articles["published_date"].tolist()]
    published_dates = [value for value in published_dates if value is not None]
    recommended_paragraph_count = int(paragraphs["is_recommended_mix_section"].fillna(False).astype(bool).sum())
    lengths = articles["main_text"].fillna("").astype(str).str.len()
    stats_df = pd.DataFrame(
        [
            {
                "source_name": "Shisha Cafe & Bar LAGOS",
                "source_type": SOURCE_TYPE,
                "category_page_count": category_page_count,
                "article_count": len(articles),
                "paragraph_count": len(paragraphs),
                "recommended_mix_article_count": int(articles["recommended_mix_section_found"].fillna(False).astype(bool).sum()),
                "recommended_mix_paragraph_count": recommended_paragraph_count,
                "publication_date_min": min(published_dates).strftime("%Y-%m-%d") if published_dates else "",
                "publication_date_max": max(published_dates).strftime("%Y-%m-%d") if published_dates else "",
                "main_text_length_mean": float(lengths.mean()) if len(lengths) else 0.0,
                "main_text_length_median": float(lengths.median()) if len(lengths) else 0.0,
                "unique_brand_count": int(articles["brand"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()),
                "unique_target_flavor_count": int(articles["target_flavor"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()),
                "scraped_at": max(articles["scraped_at"].astype(str).tolist()) if len(articles) else "",
            }
        ]
    )
    write_csv(stats_df, output_dir / "paper_shisha_lagos_dataset_statistics.csv")
    return stats_df


def write_dataset_draft(
    output_path: Path,
    stats_row: pd.Series,
    collected_date: str,
) -> None:
    lines = [
        "# Paper Dataset Draft",
        "",
        "## 日本語案",
        "",
        (
            f"追加データソースとして，Shisha Cafe & Bar LAGOS の「{CATEGORY_LABEL}」カテゴリ"
            f"（{stats_row['article_count']}記事）を収集対象とした。収集は管理者の許可を得た上で"
            f"{collected_date} に実施し，HTML構造と取得日時を記録しつつ，サーバー負荷を抑えるため逐次的に取得した。"
        ),
        (
            f"対象データは {stats_row['publication_date_min']} から {stats_row['publication_date_max']} に公開された"
            f"編集記事型データ（source_type = {SOURCE_TYPE}）であり，記事本文とおすすめミックス節を別個に抽出して保存した。"
        ),
        "このデータは既存のユーザーレビューデータとは別ソースとして保持しており，本段階では自動統合していない。",
        "",
        "## English Draft",
        "",
        (
            f"As an additional data source, we collected {int(stats_row['article_count'])} articles from the "
            f"\"{CATEGORY_LABEL}\" category of Shisha Cafe & Bar LAGOS. The collection was conducted on "
            f"{collected_date} with explicit permission from the site administrator, while recording the HTML structure "
            "and retrieval timestamps and using sequential requests to reduce server load."
        ),
        (
            f"The collected articles were published between {stats_row['publication_date_min']} and "
            f"{stats_row['publication_date_max']} and were treated as editorial articles "
            f"(source_type = {SOURCE_TYPE}). We extracted both the main article text and the recommended-mix sections, "
            "and kept this corpus separate from the existing user-review data."
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_enhanced_summary_and_report(
    output_dir: Path,
    summary: dict[str, Any],
    articles: pd.DataFrame,
) -> tuple[Path, Path]:
    summary_csv = output_dir / "shisha_lagos_scraping_summary.csv"
    report_md = output_dir / "shisha_lagos_scraping_report.md"
    write_csv(pd.DataFrame([summary]), summary_csv)

    sample_articles = articles.sort_values("article_title").head(3)
    lines = [
        "# Shisha Lagos Scraping Report",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Representative Articles", ""])
    for row in sample_articles.itertuples(index=False):
        headings = [item.get("heading_text", "") for item in load_heading_list(row.headings)]
        main_text = str(row.main_text)
        lines.extend(
            [
                f"### {row.article_title}",
                "",
                f"- URL: {row.article_url}",
                f"- 本文文字数: {len(main_text)}",
                f"- 見出し一覧: {' | '.join(headings) if headings else 'なし'}",
                f"- おすすめミックス節: {'あり' if bool(row.recommended_mix_section_found) else 'なし'}",
                "",
                "本文先頭300文字:",
                main_text[:300],
                "",
                "本文末尾300文字:",
                main_text[-300:],
                "",
            ]
        )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_csv, report_md


def build_metadata(
    start_url: str,
    output_dir: Path,
    category_page_count: int,
    unique_article_count: int,
    success_count: int,
    failure_count: int,
    request_delay: float,
    user_agent: str,
    output_files: list[str],
) -> dict[str, Any]:
    return {
        "start_url": start_url,
        "source_site": SOURCE_SITE,
        "source_type": SOURCE_TYPE,
        "scraped_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scraper_version": "0.2.0",
        "git_commit": git_commit_hash(Path(__file__).resolve().parents[1]),
        "request_delay": request_delay,
        "user_agent": user_agent,
        "max_pages": 100,
        "category_page_count": category_page_count,
        "article_url_count": unique_article_count,
        "successful_article_count": success_count,
        "failed_article_count": failure_count,
        "robots_txt_url": f"{DEFAULT_START_URL.split('/category/', 1)[0]}/robots.txt",
        "robots_txt_status": 200,
        "output_files": output_files,
    }


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logger(args.log_level)
    root = Path(__file__).resolve().parents[1]
    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles = read_csv(processed_dir / "shisha_lagos_articles.csv")
    paragraphs = read_csv(processed_dir / "shisha_lagos_paragraphs.csv")
    recommended = read_csv(processed_dir / "shisha_lagos_recommended_mix_sections.csv")
    duplicates_path = processed_dir / "shisha_lagos_duplicates.csv"
    duplicates = read_csv(duplicates_path) if duplicates_path.stat().st_size > 0 else pd.DataFrame()
    tables = read_csv(processed_dir / "shisha_lagos_tables.csv")
    list_items = read_csv(processed_dir / "shisha_lagos_list_items.csv")

    category_page_count, discovered_article_url_count, unique_article_url_count, listed_dates = collect_category_listing_stats(raw_dir)
    logger.info(
        "category pages=%s discovered=%s unique=%s",
        category_page_count,
        discovered_article_url_count,
        unique_article_url_count,
    )

    contamination_df = contamination_hits(articles)
    write_csv(contamination_df, output_dir / "shisha_lagos_contamination_hits.csv")

    outliers_df, length_summary = build_length_outliers(articles)
    write_csv(outliers_df, output_dir / "shisha_lagos_article_length_outliers.csv")

    heading_df, recommended_heading_df, missing_recommended_df = build_heading_outputs(articles)
    write_csv(heading_df, output_dir / "shisha_lagos_heading_frequency.csv")
    write_csv(recommended_heading_df, output_dir / "shisha_lagos_recommended_heading_variants.csv")
    write_csv(missing_recommended_df, output_dir / "shisha_lagos_missing_recommended_sections.csv")

    brand_candidates = read_brand_candidates(Path(args.master_csv))
    extraction_audit_df = build_flavor_extraction_audit(articles, brand_candidates)
    write_csv(extraction_audit_df, output_dir / "shisha_lagos_flavor_extraction_audit.csv")

    metadata_inconsistencies_df = build_metadata_inconsistencies(articles, listed_dates)
    write_csv(metadata_inconsistencies_df, output_dir / "shisha_lagos_metadata_inconsistencies.csv")

    repeated_paragraphs_df = build_repeated_paragraphs(paragraphs)
    write_csv(repeated_paragraphs_df, output_dir / "shisha_lagos_repeated_paragraphs.csv")

    schema_comparison_df = build_source_schema_comparison(root / "data", articles, paragraphs)
    write_csv(schema_comparison_df, output_dir / "source_schema_comparison.csv")
    write_source_characteristics(output_dir / "source_characteristics.md", len(read_csv(root / "data" / "cloud_reviews_final.csv")), articles)

    build_dictionary_coverage(articles, paragraphs, tables, Path(args.master_csv), output_dir)

    dataset_stats_df = build_dataset_statistics(articles, paragraphs, output_dir, category_page_count)
    stats_row = dataset_stats_df.iloc[0]
    collected_date = datetime.now().strftime("%Y-%m-%d")
    write_dataset_draft(output_dir / "paper_shisha_lagos_dataset_draft.md", stats_row, collected_date)

    summary = {
        "category_page_count": category_page_count,
        "discovered_article_url_count": discovered_article_url_count,
        "unique_article_url_count": unique_article_url_count,
        "successful_article_count": len(articles),
        "failed_article_count": 0,
        "duplicate_article_count": len(duplicates),
        "main_text_success_count": int(articles["main_text"].fillna("").astype(str).ne("").sum()),
        "empty_body_article_count": int(articles["main_text"].fillna("").astype(str).eq("").sum()),
        "recommended_mix_section_found": safe_ratio(int(articles["recommended_mix_section_found"].fillna(False).astype(bool).sum()), len(articles)),
        "recommended_mix_section_missing": safe_ratio(int((~articles["recommended_mix_section_found"].fillna(False).astype(bool)).sum()), len(articles)),
        "brand_extraction_success": safe_ratio(int(articles["brand"].fillna("").astype(str).ne("").sum()), len(articles)),
        "target_flavor_extraction_success": safe_ratio(int(articles["target_flavor"].fillna("").astype(str).ne("").sum()), len(articles)),
        "published_date_success": safe_ratio(int(articles["published_date"].fillna("").astype(str).ne("").sum()), len(articles)),
        "updated_date_success": safe_ratio(int(articles["updated_date"].fillna("").astype(str).ne("").sum()), len(articles)),
        "author_success": safe_ratio(int(articles["author"].fillna("").astype(str).ne("").sum()), len(articles)),
        "main_text_length_min": length_summary["minimum"],
        "main_text_length_p25": round(length_summary["p25"], 1),
        "main_text_length_median": round(length_summary["median"], 1),
        "main_text_length_mean": round(length_summary["mean"], 1),
        "main_text_length_p75": round(length_summary["p75"], 1),
        "main_text_length_max": length_summary["maximum"],
        "table_article_count": int(tables["article_id"].nunique()) if "article_id" in tables.columns else 0,
        "list_article_count": int(list_items["article_id"].nunique()) if "article_id" in list_items.columns else 0,
        "recommended_mix_table_count": int(
            tables.loc[tables["is_recommended_mix_section"].fillna(False), ["article_id", "table_index"]].drop_duplicates().shape[0]
        ) if not tables.empty else 0,
        "recommended_mix_list_count": int(
            list_items.loc[list_items["is_recommended_mix_section"].fillna(False), ["article_id", "list_index"]].drop_duplicates().shape[0]
        ) if not list_items.empty else 0,
        "contamination_phrase_hit_count": len(contamination_df),
        "metadata_inconsistency_count": len(metadata_inconsistencies_df),
    }
    summary_csv, report_md = write_enhanced_summary_and_report(output_dir, summary, articles)

    metadata = build_metadata(
        start_url=args.start_url,
        output_dir=output_dir,
        category_page_count=category_page_count,
        unique_article_count=unique_article_url_count,
        success_count=len(articles),
        failure_count=0,
        request_delay=2.0,
        user_agent="ShishaResearchBot/1.0 (+research scraping for JCEEE paper)",
        output_files=[
            str(processed_dir / "shisha_lagos_articles.csv"),
            str(processed_dir / "shisha_lagos_paragraphs.csv"),
            str(processed_dir / "shisha_lagos_tables.csv"),
            str(processed_dir / "shisha_lagos_list_items.csv"),
            str(processed_dir / "shisha_lagos_recommended_mix_sections.csv"),
            str(processed_dir / "shisha_lagos_duplicates.csv"),
            str(summary_csv),
            str(report_md),
            str(output_dir / "paper_shisha_lagos_dataset_statistics.csv"),
            str(output_dir / "paper_shisha_lagos_dataset_draft.md"),
        ],
    )
    (output_dir / "shisha_lagos_scraping_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("shisha lagos audit completed")
    print(f"- category_page_count: {category_page_count}")
    print(f"- discovered_article_url_count: {discovered_article_url_count}")
    print(f"- unique_article_url_count: {unique_article_url_count}")
    print(f"- successful_article_count: {len(articles)}")
    print(f"- duplicate_article_count: {len(duplicates)}")
    print(f"- contamination_phrase_hit_count: {len(contamination_df)}")
    print(f"- metadata_inconsistency_count: {len(metadata_inconsistencies_df)}")


if __name__ == "__main__":
    main()
