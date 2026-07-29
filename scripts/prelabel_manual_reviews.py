#!/usr/bin/env python3
"""Apply rule-based provisional labels to manual review rows."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster_analysis"
INPUT_CSV = POSTER_DIR / "manual_review_check.csv"
OUTPUT_CSV = POSTER_DIR / "manual_review_check_prelabelled.csv"
SUMMARY_CSV = POSTER_DIR / "manual_review_prelabel_summary.csv"
SUMMARY_MD = POSTER_DIR / "summary.md"

SUMMARY_START = "<!-- manual_review_prelabel:start -->"
SUMMARY_END = "<!-- manual_review_prelabel:end -->"

EXPLICIT_KEYWORDS = [
    "ミックス",
    "mix",
    "m i x",
    "ブレンド",
    "組み合わせ",
    "レシピ",
    "配合",
    "混ぜ",
    "合わせ",
    "加え",
    "足し",
    "使用",
    "半々",
    "flavor mix",
    "mixed with",
    "blend",
    "recipe",
    "combine",
]

LIST_CONTEXT_KEYWORDS = [
    "ランキング",
    "top",
    "一覧",
    "まとめ",
    "種類",
    "ラインナップ",
    "商品紹介",
    "ブランド紹介",
    "フレーバー紹介",
    "おすすめ",
    "関連商品",
    "メニュー",
    "特徴",
    "レビューをご紹介",
    "購入はこちら",
    "目次",
]

PRODUCT_CONTEXT_KEYWORDS = [
    "の特徴",
    "レビュー",
    "とは",
    "ブランド",
    "フレーバー",
    "商品",
]

SESSION_KEYWORDS = [
    "吸っ",
    "作っ",
    "今回",
    "こちら",
    "ボウル",
    "注文",
    "盛り",
]

RATIO_PATTERNS = [
    r"\b\d{1,3}\s?%",
    r"\b\d+\s?g\b",
    r"\b\d+\s?グラム\b",
    r"\b\d+\s?:\s?\d+\b",
    r"半々",
    r"比率",
    r"配合",
]


def normalize_text(value: object) -> str:
    """Normalize Unicode width/case/whitespace for rule matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    normalized = normalized.replace("　", " ")
    return re.sub(r"\s+", " ", normalized)


def split_segments(title: str, context: str) -> list[str]:
    """Split title/context into review-local segments for conservative matching."""
    prepared = "\n".join(part for part in [title, context] if part)
    prepared = prepared.replace("・", "\n・")
    raw_segments = re.split(r"(?:[。！？!?]\s*|\n+)", prepared)
    return [segment.strip() for segment in raw_segments if segment and segment.strip()]


def is_ascii_term(term: str) -> bool:
    """Return True for ASCII-heavy labels."""
    return bool(re.fullmatch(r"[a-z0-9 .&'\-/]+", term))


def contains_term(text: str, term: str) -> bool:
    """Check whether a normalized term appears in normalized text."""
    if not text or not term:
        return False
    if is_ascii_term(term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


def has_keyword(text: str, keywords: list[str]) -> bool:
    """Return True if any normalized keyword appears in the text."""
    return any(keyword in text for keyword in keywords)


def has_ratio_pattern(text: str) -> bool:
    """Return True when ratio-like notation appears in the text."""
    return any(re.search(pattern, text) for pattern in RATIO_PATTERNS)


def has_separator_between(text: str, left: str, right: str) -> bool:
    """Check whether two flavors appear with a mix-like separator nearby."""
    separators = r"(?:\s|[、,/\+\-&＆＋]|と|や|に|を|へ|×|x){0,12}"
    patterns = [
        rf"{re.escape(left)}{separators}{re.escape(right)}",
        rf"{re.escape(right)}{separators}{re.escape(left)}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def only_compound_name(text: str, flavor_a: str, flavor_b: str) -> bool:
    """Detect concatenated product names like 'グレープミント' without mix context."""
    compact = re.sub(r"\s+", "", text)
    joined_variants = [
        flavor_a + flavor_b,
        flavor_b + flavor_a,
        f"{flavor_a} {flavor_b}",
        f"{flavor_b} {flavor_a}",
    ]
    return any(variant.replace(" ", "") in compact for variant in joined_variants)


def classify_row(row: pd.Series) -> dict[str, object]:
    """Assign a provisional label to one manual-review row."""
    title = normalize_text(row.get("review_title", ""))
    context = normalize_text(row.get("matched_context", ""))
    flavor_a = normalize_text(row.get("flavor_a", ""))
    flavor_b = normalize_text(row.get("flavor_b", ""))
    combined = "\n".join(part for part in [title, context] if part).strip()
    segments = split_segments(title, context)

    if not context:
        return {
            "auto_label": "unclear",
            "auto_confidence": "low",
            "auto_reason": "matched_context が空で、タイトル以外の本文文脈を確認できない",
            "matched_rule": "title_only",
            "needs_manual_review": True,
        }

    combined_has_a = contains_term(combined, flavor_a)
    combined_has_b = contains_term(combined, flavor_b)
    if not combined_has_a or not combined_has_b:
        return {
            "auto_label": "unclear",
            "auto_confidence": "low",
            "auto_reason": "対象2フレーバーの両方が文脈内で確認できない",
            "matched_rule": "missing_target_flavor",
            "needs_manual_review": True,
        }

    both_segments = [
        segment
        for segment in segments
        if contains_term(segment, flavor_a) and contains_term(segment, flavor_b)
    ]

    if len(context) < 18 and not both_segments:
        return {
            "auto_label": "unclear",
            "auto_confidence": "low",
            "auto_reason": "本文文脈が短く、対象ペアの関係を判定できない",
            "matched_rule": "context_too_short",
            "needs_manual_review": True,
        }

    strong_explicit_segments = []
    probable_segments = []
    product_like_segments = []
    list_like_segments = []

    for segment in both_segments:
        explicit_hit = has_keyword(segment, EXPLICIT_KEYWORDS)
        ratio_hit = has_ratio_pattern(segment)
        separator_hit = has_separator_between(segment, flavor_a, flavor_b)
        session_hit = has_keyword(segment, SESSION_KEYWORDS)
        product_hit = has_keyword(segment, PRODUCT_CONTEXT_KEYWORDS)
        list_hit = has_keyword(segment, LIST_CONTEXT_KEYWORDS)
        compound_only = only_compound_name(segment, flavor_a, flavor_b) and not separator_hit
        chain_mix_hit = any(symbol in segment for symbol in ["×", "+", "＋", "/"])

        if list_hit:
            list_like_segments.append(segment)
        if product_hit or compound_only:
            product_like_segments.append(segment)
        if explicit_hit and (separator_hit or chain_mix_hit) and not compound_only:
            strong_explicit_segments.append(segment)
            continue
        if (product_hit or compound_only) and not ratio_hit:
            continue
        if ratio_hit and not compound_only:
            probable_segments.append(segment)
            continue
        if session_hit and separator_hit and not compound_only:
            probable_segments.append(segment)

    if strong_explicit_segments:
        segment = strong_explicit_segments[0]
        confidence = "high"
        needs_review = False
        if has_keyword(segment, LIST_CONTEXT_KEYWORDS):
            confidence = "medium"
            needs_review = True
        return {
            "auto_label": "explicit_mix",
            "auto_confidence": confidence,
            "auto_reason": "本文中に明示的なミックス表現があり、対象2フレーバーが同一文脈に並記されている",
            "matched_rule": "explicit_keyword_with_both_flavors",
            "needs_manual_review": needs_review,
        }

    if probable_segments:
        return {
            "auto_label": "probable_mix",
            "auto_confidence": "medium",
            "auto_reason": "対象2フレーバーが同一文脈にあり、配合や同一セッションを示す記述がある",
            "matched_rule": "recipe_ratio_pattern",
            "needs_manual_review": True,
        }

    if has_keyword(combined, LIST_CONTEXT_KEYWORDS):
        return {
            "auto_label": "co_mention_only",
            "auto_confidence": "high",
            "auto_reason": "ランキング・一覧・紹介文脈での列挙が中心で、同一ミックスの根拠が弱い",
            "matched_rule": "product_list_context",
            "needs_manual_review": False,
        }

    if both_segments and (product_like_segments or list_like_segments):
        return {
            "auto_label": "co_mention_only",
            "auto_confidence": "high",
            "auto_reason": "商品紹介・特徴説明・一覧文脈での同時言及に留まり、同一ミックスの根拠が弱い",
            "matched_rule": "product_list_context",
            "needs_manual_review": False,
        }

    if not both_segments:
        if title and not context:
            return {
                "auto_label": "unclear",
                "auto_confidence": "low",
                "auto_reason": "タイトルには対象ペアがあるが、本文文脈が不足している",
                "matched_rule": "title_only",
                "needs_manual_review": True,
            }
        if only_compound_name(combined, flavor_a, flavor_b):
            return {
                "auto_label": "unclear",
                "auto_confidence": "low",
                "auto_reason": "複合商品名の可能性があり、ミックス表記か判別できない",
                "matched_rule": "product_name_ambiguity",
                "needs_manual_review": True,
            }
        return {
            "auto_label": "co_mention_only",
            "auto_confidence": "medium",
            "auto_reason": "対象2フレーバーは同一レビュー内に出現するが、同一文脈では結び付いていない",
            "matched_rule": "separate_context_mentions",
            "needs_manual_review": False,
        }

    return {
        "auto_label": "unclear",
        "auto_confidence": "low",
        "auto_reason": "判定ルールが十分な根拠を持たず、人手確認が必要",
        "matched_rule": "conflicting_rules",
        "needs_manual_review": True,
    }


def summarize_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build aggregate summaries for labels, confidence, and rules."""
    total = len(df)
    label_counts = df["auto_label"].value_counts().reindex(
        ["explicit_mix", "probable_mix", "co_mention_only", "unclear"],
        fill_value=0,
    )
    needs_counts = (
        df.groupby("auto_label")["needs_manual_review"]
        .sum()
        .reindex(["explicit_mix", "probable_mix", "co_mention_only", "unclear"], fill_value=0)
    )
    summary_df = pd.DataFrame(
        {
            "auto_label": label_counts.index,
            "count": label_counts.values,
            "ratio": [count / total if total else 0.0 for count in label_counts.values],
            "needs_manual_review_count": [int(needs_counts[label]) for label in label_counts.index],
        }
    )
    confidence_counts = df["auto_confidence"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    rule_counts = df["matched_rule"].value_counts()
    return summary_df, confidence_counts, rule_counts


def upsert_summary_section(
    summary_path: Path,
    *,
    total_rows: int,
    summary_df: pd.DataFrame,
    confidence_counts: pd.Series,
    rule_counts: pd.Series,
    needs_manual_review_count: int,
) -> None:
    """Append or replace the provisional-label summary section in summary.md."""
    if summary_path.exists():
        original = summary_path.read_text(encoding="utf-8")
    else:
        original = "# poster_analysis summary\n"

    lines = [
        SUMMARY_START,
        "## 15. 仮ラベル付け集計",
        f"- 総件数: {total_rows}",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['auto_label']}: {int(row['count'])}件 ({float(row['ratio']):.1%})"
        )
    lines.append(f"- needs_manual_review 件数: {needs_manual_review_count}")
    lines.append("- confidence 別件数:")
    for label, count in confidence_counts.items():
        lines.append(f"  - {label}: {int(count)}")
    lines.append("- ルール別件数:")
    for label, count in rule_counts.items():
        lines.append(f"  - {label}: {int(count)}")
    lines.append(SUMMARY_END)
    block = "\n".join(lines)

    pattern = re.compile(
        rf"{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}",
        flags=re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(block, original)
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    summary_path.write_text(updated, encoding="utf-8")


def run_self_tests() -> list[str]:
    """Run the requested minimum rule tests."""
    test_messages = []

    def classify(flavor_a: str, flavor_b: str, title: str, context: str, manual_label: str = "") -> dict[str, object]:
        row = pd.Series(
            {
                "flavor_a": flavor_a,
                "flavor_b": flavor_b,
                "review_title": title,
                "matched_context": context,
                "manual_label": manual_label,
            }
        )
        return classify_row(row)

    result = classify("グレープ", "ミント", "", "グレープとミントを半々でミックスしました")
    assert result["auto_label"] == "explicit_mix"
    test_messages.append("PASS: 明示的なミックス表現は explicit_mix になる")

    result = classify("グレープ", "ミント", "", "おすすめフレーバーランキング。グレープ、ミント、レモンを紹介")
    assert result["auto_label"] == "co_mention_only"
    test_messages.append("PASS: ランキング文脈は co_mention_only になる")

    result = classify("グレープ", "ミント", "", "グレープ 50%、ミント 50%")
    assert result["auto_label"] in {"probable_mix", "explicit_mix"}
    test_messages.append("PASS: 比率のみの記述は probable_mix 以上として扱う")

    # 商品名だけでは明示的ミックスとは見なさない。
    result = classify("グレープ", "ミント", "MALAKI Grape Mint の特徴", "MALAKI Grape Mint の特徴")
    assert result["auto_label"] in {"unclear", "co_mention_only"}
    assert result["auto_label"] != "explicit_mix"
    test_messages.append("PASS: 商品名らしい Grape Mint は explicit_mix にしない")

    result = classify("グレープ", "ミント", "グレープ×ミント", "")
    assert result["auto_label"] == "unclear"
    test_messages.append("PASS: タイトルだけの pair 表記は unclear にする")

    result = classify("グレープ", "ミント", "", "グレープにミントを少し足したミックス")
    assert result["auto_label"] == "explicit_mix"
    test_messages.append("PASS: 足したミックス表現は explicit_mix になる")

    result = classify("グレープ", "ミント", "", "グレープを使ったおすすめレシピです")
    assert result["auto_label"] == "unclear"
    test_messages.append("PASS: 対象ペアの一方しかない場合は unclear にする")

    df = pd.DataFrame(
        [
            {
                "flavor_a": "グレープ",
                "flavor_b": "ミント",
                "review_title": "",
                "matched_context": "グレープとミントを半々でミックスしました",
                "manual_label": "explicit_mix",
            }
        ]
    )
    original_label = df.loc[0, "manual_label"]
    _ = classify_row(df.loc[0])
    assert df.loc[0, "manual_label"] == original_label
    test_messages.append("PASS: 既存の manual_label は上書きしない")

    return test_messages


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    tests = run_self_tests()

    output_rows = []
    for _, row in df.iterrows():
        classified = classify_row(row)
        output_rows.append(
            {
                **row.to_dict(),
                **classified,
                "reviewer_label": "",
                "reviewer_note": "",
            }
        )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    summary_df, confidence_counts, rule_counts = summarize_labels(output_df)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    needs_manual_review_count = int(output_df["needs_manual_review"].sum())
    upsert_summary_section(
        SUMMARY_MD,
        total_rows=len(output_df),
        summary_df=summary_df,
        confidence_counts=confidence_counts,
        rule_counts=rule_counts,
        needs_manual_review_count=needs_manual_review_count,
    )

    print("manual_review prelabel summary")
    print(f"- 使用したレビュー本文列: matched_context")
    print(f"- 補助列: review_title")
    print(f"- 入力件数: {len(output_df)}")
    print("- auto_label別件数:")
    for _, row in summary_df.iterrows():
        print(f"  - {row['auto_label']}: {int(row['count'])} ({float(row['ratio']):.1%})")
    print(f"- needs_manual_review件数: {needs_manual_review_count}")
    print("- confidence別件数:")
    for label, count in confidence_counts.items():
        print(f"  - {label}: {int(count)}")
    print("- ルール別件数:")
    for label, count in rule_counts.items():
        print(f"  - {label}: {int(count)}")
    print("- 出力ファイル:")
    print(f"  - {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"  - {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"  - {SUMMARY_MD.relative_to(ROOT)}")
    print("- テスト結果:")
    for message in tests:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
