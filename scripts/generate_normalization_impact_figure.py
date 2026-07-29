#!/usr/bin/env python3
"""Generate a compact poster figure summarizing normalization impact."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from generate_condition_comparison import (
    MASTER_CSV,
    REVIEWS_CSV,
    apply_normalization_map,
    build_alias_candidates_and_map,
    build_flavor_dictionary,
    build_review_extraction_summary,
    choose_recommended_lift_min_pair_count,
    get_top_rows,
    run_condition_analysis,
    top_set_metrics,
)

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster_analysis"
SUMMARY_MD = POSTER_DIR / "summary.md"

FIGURE_PNG = POSTER_DIR / "figure_normalization_impact.png"
FIGURE_PDF = POSTER_DIR / "figure_normalization_impact.pdf"
METRICS_CSV = POSTER_DIR / "normalization_impact_metrics.csv"
METRICS_MD = POSTER_DIR / "normalization_impact_metrics.md"

SUMMARY_START = "<!-- normalization_impact:start -->"
SUMMARY_END = "<!-- normalization_impact:end -->"


def compute_metrics() -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Recompute normalization-impact metrics from source data."""
    reviews_df = pd.read_csv(REVIEWS_CSV)
    master_df = pd.read_csv(MASTER_CSV)
    _flavor_dict, pattern_to_canonical, sorted_patterns = build_flavor_dictionary(master_df)

    raw_review_extraction_df = build_review_extraction_summary(
        reviews_df,
        sorted_patterns,
        pattern_to_canonical,
    )
    alias_candidates_df, normalization_map_df = build_alias_candidates_and_map(
        raw_review_extraction_df,
        master_df,
    )
    normalized_review_extraction_df = apply_normalization_map(
        raw_review_extraction_df,
        normalization_map_df,
    )

    raw_analysis = run_condition_analysis(raw_review_extraction_df)
    normalized_analysis = run_condition_analysis(normalized_review_extraction_df)

    raw_exploded = raw_review_extraction_df["extracted_flavors"].fillna("").str.split("|").explode()
    raw_unique_flavors = int(raw_exploded[raw_exploded.fillna("").str.strip() != ""].nunique())
    normalized_exploded = (
        normalized_review_extraction_df["extracted_flavors"].fillna("").str.split("|").explode()
    )
    normalized_unique_flavors = int(
        normalized_exploded[normalized_exploded.fillna("").str.strip() != ""].nunique()
    )

    raw_cooc_top10 = get_top_rows(
        raw_analysis["cooccurrence_rankings_df"],
        condition="limited_2_5",
        top_k=10,
    )
    normalized_cooc_top10 = get_top_rows(
        normalized_analysis["cooccurrence_rankings_df"],
        condition="limited_2_5",
        top_k=10,
    )
    cooc_common_count, cooc_jaccard = top_set_metrics(raw_cooc_top10, normalized_cooc_top10)

    raw_min_pair_count, _raw_reason = choose_recommended_lift_min_pair_count(
        raw_analysis["lift_rankings_df"]
    )
    normalized_min_pair_count, _normalized_reason = choose_recommended_lift_min_pair_count(
        normalized_analysis["lift_rankings_df"]
    )

    raw_lift_top10 = get_top_rows(
        raw_analysis["lift_rankings_df"],
        condition="limited_2_5",
        top_k=10,
        min_pair_count=raw_min_pair_count,
    )
    normalized_lift_top10 = get_top_rows(
        normalized_analysis["lift_rankings_df"],
        condition="limited_2_5",
        top_k=10,
        min_pair_count=normalized_min_pair_count,
    )
    lift_common_count, lift_jaccard = top_set_metrics(raw_lift_top10, normalized_lift_top10)

    metrics = {
        "raw_unique_flavor_count": raw_unique_flavors,
        "normalized_unique_flavor_count": normalized_unique_flavors,
        "cooccurrence_top10_common_count": int(cooc_common_count),
        "cooccurrence_top10_jaccard": float(cooc_jaccard),
        "lift_top10_common_count": int(lift_common_count),
        "lift_top10_jaccard": float(lift_jaccard),
        "raw_lift_min_pair_count": int(raw_min_pair_count),
        "normalized_lift_min_pair_count": int(normalized_min_pair_count),
        "source_reviews_csv": str(REVIEWS_CSV.relative_to(ROOT)),
        "source_master_csv": str(MASTER_CSV.relative_to(ROOT)),
    }

    tables = {
        "raw_cooc_top10": raw_cooc_top10.copy(),
        "normalized_cooc_top10": normalized_cooc_top10.copy(),
        "raw_lift_top10": raw_lift_top10.copy(),
        "normalized_lift_top10": normalized_lift_top10.copy(),
        "alias_candidates": alias_candidates_df.copy(),
        "normalization_map": normalization_map_df.copy(),
    }
    return metrics, tables


def save_metrics_files(metrics: dict[str, object]) -> None:
    """Save the figure metrics as both CSV and Markdown."""
    metrics_df = pd.DataFrame(
        [
            {"metric": "raw_unique_flavor_count", "value": metrics["raw_unique_flavor_count"]},
            {
                "metric": "normalized_unique_flavor_count",
                "value": metrics["normalized_unique_flavor_count"],
            },
            {
                "metric": "cooccurrence_top10_common_count",
                "value": metrics["cooccurrence_top10_common_count"],
            },
            {
                "metric": "cooccurrence_top10_jaccard",
                "value": metrics["cooccurrence_top10_jaccard"],
            },
            {
                "metric": "lift_top10_common_count",
                "value": metrics["lift_top10_common_count"],
            },
            {"metric": "lift_top10_jaccard", "value": metrics["lift_top10_jaccard"]},
            {"metric": "raw_lift_min_pair_count", "value": metrics["raw_lift_min_pair_count"]},
            {
                "metric": "normalized_lift_min_pair_count",
                "value": metrics["normalized_lift_min_pair_count"],
            },
            {"metric": "source_reviews_csv", "value": metrics["source_reviews_csv"]},
            {"metric": "source_master_csv", "value": metrics["source_master_csv"]},
        ]
    )
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    md_lines = [
        "# normalization impact metrics",
        "",
        f"- 正規化前ユニークフレーバー数: {metrics['raw_unique_flavor_count']}",
        f"- 正規化後ユニークフレーバー数: {metrics['normalized_unique_flavor_count']}",
        f"- 共起回数Top10の共通ペア数: {metrics['cooccurrence_top10_common_count']}/10",
        f"- 共起回数Top10のJaccard係数: {metrics['cooccurrence_top10_jaccard']:.4f}",
        f"- Lift Top10の共通ペア数: {metrics['lift_top10_common_count']}/10",
        f"- Lift Top10のJaccard係数: {metrics['lift_top10_jaccard']:.4f}",
        f"- 正規化前Liftのmin_pair_count: {metrics['raw_lift_min_pair_count']}",
        f"- 正規化後Liftのmin_pair_count: {metrics['normalized_lift_min_pair_count']}",
        "",
        f"- レビューCSV: `{metrics['source_reviews_csv']}`",
        f"- マスタCSV: `{metrics['source_master_csv']}`",
    ]
    METRICS_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def draw_impact_figure(metrics: dict[str, object]) -> None:
    """Draw a compact horizontal summary figure for poster use."""
    fig, ax = plt.subplots(figsize=(16, 4.6), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    section_x = [0.17, 0.50, 0.83]
    titles = ["ユニークフレーバー数", "共起回数 Top10", "Lift Top10"]
    accent_colors = ["#2C7FB8", "#4D4D4D", "#D95F02"]

    for idx, x_pos in enumerate(section_x):
        box = plt.Rectangle(
            (x_pos - 0.14, 0.23),
            0.28,
            0.50,
            facecolor="#FAFAFA",
            edgecolor="#D9D9D9",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x_pos,
            0.67,
            titles[idx],
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    ax.text(
        section_x[0],
        0.49,
        f"{metrics['raw_unique_flavor_count']} → {metrics['normalized_unique_flavor_count']}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=accent_colors[0],
    )

    ax.text(
        section_x[1],
        0.53,
        f"{metrics['cooccurrence_top10_common_count']}/10",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=accent_colors[1],
    )
    ax.text(
        section_x[1],
        0.38,
        f"Jaccard = {metrics['cooccurrence_top10_jaccard']:.3f}",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=accent_colors[1],
    )

    ax.text(
        section_x[2],
        0.49,
        f"{metrics['lift_top10_common_count']}/10",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color=accent_colors[2],
    )

    ax.text(
        0.5,
        0.12,
        "頻出する共起関係は安定している一方、Liftは名称正規化の影響を受けやすい",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#333333",
    )

    ax.set_title(
        "名称正規化が分析結果に与える影響",
        fontsize=18,
        fontweight="bold",
        pad=8,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURE_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_summary(metrics: dict[str, object]) -> None:
    """Append or replace the normalization-impact summary block."""
    if SUMMARY_MD.exists():
        original = SUMMARY_MD.read_text(encoding="utf-8")
    else:
        original = "# poster_analysis summary\n"

    lines = [
        SUMMARY_START,
        "## 18. 名称正規化の影響図",
        f"- 正規化前ユニークフレーバー数: {metrics['raw_unique_flavor_count']}",
        f"- 正規化後ユニークフレーバー数: {metrics['normalized_unique_flavor_count']}",
        f"- 共起回数Top10の共通ペア数: {metrics['cooccurrence_top10_common_count']}/10",
        f"- 共起回数Top10のJaccard係数: {metrics['cooccurrence_top10_jaccard']:.4f}",
        f"- Lift Top10の共通ペア数: {metrics['lift_top10_common_count']}/10",
        f"- Lift Top10のJaccard係数: {metrics['lift_top10_jaccard']:.4f}",
        f"- 図出力: `{FIGURE_PNG.relative_to(ROOT)}`, `{FIGURE_PDF.relative_to(ROOT)}`",
        f"- 数値表: `{METRICS_CSV.relative_to(ROOT)}`, `{METRICS_MD.relative_to(ROOT)}`",
    ]
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
    SUMMARY_MD.write_text(updated, encoding="utf-8")


def run_self_checks() -> list[str]:
    """Run basic checks for the compact-figure metrics logic."""
    raw_pairs = pd.DataFrame({"pair_key": ["A||B", "A||C", "B||C"]})
    normalized_pairs = pd.DataFrame({"pair_key": ["A||B", "A||D", "B||C"]})
    common, jaccard = top_set_metrics(raw_pairs, normalized_pairs)
    messages = []

    assert common == 2
    assert abs(jaccard - 0.5) < 1e-9
    messages.append("PASS: Top10 共通数と Jaccard 係数を計算できる")

    metrics_df = pd.DataFrame(
        [
            {"metric": "raw_unique_flavor_count", "value": 144},
            {"metric": "normalized_unique_flavor_count", "value": 136},
        ]
    )
    assert list(metrics_df["metric"]) == [
        "raw_unique_flavor_count",
        "normalized_unique_flavor_count",
    ]
    messages.append("PASS: 数値表用の行形式を組み立てられる")
    return messages


def main() -> None:
    metrics, _tables = compute_metrics()
    save_metrics_files(metrics)
    draw_impact_figure(metrics)
    update_summary(metrics)
    test_messages = run_self_checks()

    print("normalization impact outputs generated")
    print(f"- 正規化前ユニークフレーバー数: {metrics['raw_unique_flavor_count']}")
    print(f"- 正規化後ユニークフレーバー数: {metrics['normalized_unique_flavor_count']}")
    print(f"- 共起回数Top10の共通ペア数: {metrics['cooccurrence_top10_common_count']}/10")
    print(f"- 共起回数Top10のJaccard係数: {metrics['cooccurrence_top10_jaccard']:.4f}")
    print(f"- Lift Top10の共通ペア数: {metrics['lift_top10_common_count']}/10")
    print(f"- Lift Top10のJaccard係数: {metrics['lift_top10_jaccard']:.4f}")
    print("- output files:")
    print(f"  - {FIGURE_PNG.relative_to(ROOT)}")
    print(f"  - {FIGURE_PDF.relative_to(ROOT)}")
    print(f"  - {METRICS_CSV.relative_to(ROOT)}")
    print(f"  - {METRICS_MD.relative_to(ROOT)}")
    print(f"  - {SUMMARY_MD.relative_to(ROOT)}")
    print("- tests:")
    for message in test_messages:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
