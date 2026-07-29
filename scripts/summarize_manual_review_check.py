#!/usr/bin/env python3
"""Summarize final manual-review labels for poster analysis."""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster_analysis"
PRELABELLED_CSV = POSTER_DIR / "manual_review_check_prelabelled.csv"
FINAL_CSV = POSTER_DIR / "manual_review_check_final.csv"
SUMMARY_CSV = POSTER_DIR / "manual_review_final_summary.csv"
FIGURE5_PATH = POSTER_DIR / "figure5_manual_check.png"
SUMMARY_MD = POSTER_DIR / "summary.md"

FINAL_SUMMARY_START = "<!-- manual_review_final:start -->"
FINAL_SUMMARY_END = "<!-- manual_review_final:end -->"

LABEL_ORDER = ["explicit_mix", "probable_mix", "co_mention_only", "unclear", "unresolved"]
PLOT_LABEL_ORDER = ["explicit_mix", "probable_mix", "co_mention_only", "unclear", "unresolved"]


def build_pair_key(df: pd.DataFrame) -> pd.Series:
    """Create pair keys from columns when pair_key is absent."""
    return df["flavor_a"].astype(str) + "||" + df["flavor_b"].astype(str)


def normalize_label(value: object) -> str:
    """Normalize label cells for final adoption."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def resolve_final_label(row: pd.Series) -> tuple[str, str]:
    """Choose the final label from reviewer_label/auto_label/unresolved rules."""
    reviewer_label = normalize_label(row.get("reviewer_label", ""))
    auto_label = normalize_label(row.get("auto_label", ""))
    needs_manual_review = bool(row.get("needs_manual_review", False))

    if reviewer_label:
        return reviewer_label, "reviewer_label"
    if not needs_manual_review and auto_label:
        return auto_label, "auto_label"
    return "unresolved", "unresolved"


def create_figure5(counts: pd.Series) -> None:
    """Create final manual check chart including unresolved rows."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=320)
    colors = ["#1B9E77", "#66A61E", "#D95F02", "#7570B3", "#BDBDBD"]
    ax.bar(
        counts.index,
        counts.values,
        color=colors[: len(counts)],
    )
    ax.set_ylabel("件数")
    ax.set_title("図5 人手確認結果（最終採用ラベル）", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE5_PATH, bbox_inches="tight")
    plt.close(fig)


def build_summary_df(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Aggregate final labels, label sources, and resolution state."""
    total = len(final_df)
    counts = final_df["final_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    source_counts = final_df["final_label_source"].value_counts().reindex(
        ["reviewer_label", "auto_label", "unresolved"],
        fill_value=0,
    )
    resolution_counts = final_df["is_resolved"].value_counts().reindex([True, False], fill_value=0)

    summary_df = pd.DataFrame(
        {
            "final_label": counts.index,
            "count": counts.values,
            "ratio": [count / total if total else 0.0 for count in counts.values],
        }
    )
    return summary_df, source_counts, resolution_counts


def upsert_summary_section(
    *,
    summary_path: Path,
    total_rows: int,
    summary_df: pd.DataFrame,
    source_counts: pd.Series,
    resolution_counts: pd.Series,
    pair_level_explicit: pd.Series,
) -> None:
    """Append or replace the final-label summary section in summary.md."""
    if summary_path.exists():
        original = summary_path.read_text(encoding="utf-8")
    else:
        original = "# poster_analysis summary\n"

    lines = [
        FINAL_SUMMARY_START,
        "## 16. 最終ラベル集計",
        f"- 総件数: {total_rows}",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['final_label']}: {int(row['count'])}件 ({float(row['ratio']):.1%})"
        )
    lines.append(f"- 解決済み件数: {int(resolution_counts[True])}")
    lines.append(f"- unresolved 件数: {int(resolution_counts[False])}")
    lines.append("- 採用元別件数:")
    for label, count in source_counts.items():
        lines.append(f"  - {label}: {int(count)}")
    lines.append(f"- 少なくとも1件が explicit_mix の pair 数: {int(pair_level_explicit.sum())}")
    lines.append(f"- 少なくとも1件が explicit_mix の pair 割合: {pair_level_explicit.mean():.1%}")
    lines.append(FINAL_SUMMARY_END)
    block = "\n".join(lines)

    pattern = re.compile(
        rf"{re.escape(FINAL_SUMMARY_START)}.*?{re.escape(FINAL_SUMMARY_END)}",
        flags=re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(block, original)
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    summary_path.write_text(updated, encoding="utf-8")


def run_self_checks() -> list[str]:
    """Run small checks for final aggregation logic."""
    df = pd.DataFrame(
        [
            {
                "pair_key": "A||B",
                "auto_label": "co_mention_only",
                "needs_manual_review": False,
                "reviewer_label": "",
            },
            {
                "pair_key": "B||C",
                "auto_label": "explicit_mix",
                "needs_manual_review": True,
                "reviewer_label": "",
            },
            {
                "pair_key": "C||D",
                "auto_label": "unclear",
                "needs_manual_review": True,
                "reviewer_label": "explicit_mix",
            },
            {
                "pair_key": "C||D",
                "auto_label": "co_mention_only",
                "needs_manual_review": False,
                "reviewer_label": "co_mention_only",
            },
        ]
    )
    messages = []

    resolved_rows = [resolve_final_label(row) for _, row in df.iterrows()]
    assert resolved_rows[0] == ("co_mention_only", "auto_label")
    messages.append("PASS: reviewer_label が空で needs_manual_review=false のとき auto_label を採用する")

    assert resolved_rows[1] == ("unresolved", "unresolved")
    messages.append("PASS: needs_manual_review=true かつ reviewer_label 空欄は unresolved になる")

    assert resolved_rows[2] == ("explicit_mix", "reviewer_label")
    messages.append("PASS: reviewer_label があれば最優先で採用する")

    final_df = df.copy()
    final_df[["final_label", "final_label_source"]] = pd.DataFrame(resolved_rows, index=final_df.index)
    counts = final_df["final_label"].value_counts()
    assert counts["co_mention_only"] == 2
    assert counts["explicit_mix"] == 1
    assert counts["unresolved"] == 1
    messages.append("PASS: final_label 別件数を集計できる")

    pair_level = (
        final_df.groupby("pair_key")["final_label"]
        .apply(lambda labels: "explicit_mix" in set(labels))
        .rename("explicit_mix_confirmed")
    )
    assert bool(pair_level["C||D"])
    assert not bool(pair_level["A||B"])
    messages.append("PASS: pair 単位 explicit_mix 確認済み判定を作れる")

    explicit_ratio = (final_df["final_label"] == "explicit_mix").mean()
    assert math.isclose(explicit_ratio, 0.25)
    messages.append("PASS: final_label 比率を計算できる")

    return messages


def main() -> None:
    if not PRELABELLED_CSV.exists():
        raise FileNotFoundError(f"{PRELABELLED_CSV} が見つかりません")

    df = pd.read_csv(PRELABELLED_CSV)
    if "pair_key" not in df.columns:
        df["pair_key"] = build_pair_key(df)

    resolved = [resolve_final_label(row) for _, row in df.iterrows()]
    final_df = df.copy()
    final_df[["final_label", "final_label_source"]] = pd.DataFrame(resolved, index=final_df.index)
    final_df["is_resolved"] = final_df["final_label"] != "unresolved"
    final_df["final_label_reason"] = final_df.apply(
        lambda row: (
            "reviewer_label を採用"
            if row["final_label_source"] == "reviewer_label"
            else "needs_manual_review=false のため auto_label を採用"
            if row["final_label_source"] == "auto_label"
            else "needs_manual_review=true かつ reviewer_label 未入力のため unresolved"
        ),
        axis=1,
    )
    final_df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    summary_df, source_counts, resolution_counts = build_summary_df(final_df)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    counts = final_df["final_label"].value_counts().reindex(PLOT_LABEL_ORDER, fill_value=0)
    create_figure5(counts)

    pair_level_explicit = (
        final_df.groupby("pair_key")["final_label"]
        .apply(lambda labels: "explicit_mix" in set(labels))
        .rename("explicit_mix_confirmed")
    )
    upsert_summary_section(
        summary_path=SUMMARY_MD,
        total_rows=len(final_df),
        summary_df=summary_df,
        source_counts=source_counts,
        resolution_counts=resolution_counts,
        pair_level_explicit=pair_level_explicit,
    )

    test_messages = run_self_checks()

    print("manual_review_check final summary")
    print(f"- 入力ファイル: {PRELABELLED_CSV.relative_to(ROOT)}")
    print(f"- 総件数: {len(final_df)}")
    print(f"- 解決済み件数: {int(resolution_counts[True])}")
    print(f"- unresolved件数: {int(resolution_counts[False])}")
    print("- final_label別件数:")
    for _, row in summary_df.iterrows():
        print(f"  - {row['final_label']}: {int(row['count'])} ({float(row['ratio']):.2%})")
    print("- 採用元別件数:")
    for label, count in source_counts.items():
        print(f"  - {label}: {int(count)}")
    print(f"- 少なくとも1件がexplicit_mixのペア数: {int(pair_level_explicit.sum())}")
    print(f"- 少なくとも1件がexplicit_mixのペア割合: {pair_level_explicit.mean():.2%}")
    print("- 出力ファイル:")
    print(f"  - {FINAL_CSV.relative_to(ROOT)}")
    print(f"  - {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"  - {FIGURE5_PATH.relative_to(ROOT)}")
    print(f"  - {SUMMARY_MD.relative_to(ROOT)}")
    print("- テスト結果:")
    for message in test_messages:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
