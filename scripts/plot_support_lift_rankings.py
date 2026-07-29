#!/usr/bin/env python3
"""Create a poster-ready Support vs Lift ranking figure for Condition B."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster_analysis"
CONDITION_STATS_CSV = POSTER_DIR / "condition_statistics.csv"
COOCCURRENCE_CSV = POSTER_DIR / "cooccurrence_rankings.csv"
LIFT_CSV = POSTER_DIR / "lift_rankings.csv"
SOURCE_SUMMARY_MD = POSTER_DIR / "summary.md"

FIGURE_PNG = POSTER_DIR / "figure_support_lift_rankings.png"
FIGURE_PDF = POSTER_DIR / "figure_support_lift_rankings.pdf"
SUPPORT_TOP_CSV = POSTER_DIR / "support_top_pairs.csv"
LIFT_TOP_CSV = POSTER_DIR / "lift_top_pairs.csv"

CONDITION_NAME = "limited_2_5"
TOP_K = 5
RECOMMENDED_MIN_PAIR_COUNT = 2


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the normalized poster-analysis outputs."""
    condition_stats_df = pd.read_csv(CONDITION_STATS_CSV)
    cooccurrence_df = pd.read_csv(COOCCURRENCE_CSV)
    lift_df = pd.read_csv(LIFT_CSV)
    return condition_stats_df, cooccurrence_df, lift_df


def format_pair(row: pd.Series) -> str:
    """Format one pair name consistently as A × B."""
    return f"{row['flavor_a']} × {row['flavor_b']}"


def select_support_top_pairs(cooccurrence_df: pd.DataFrame) -> pd.DataFrame:
    """Select Support top pairs under the requested ranking rules."""
    subset = cooccurrence_df[cooccurrence_df["condition"] == CONDITION_NAME].copy()
    subset = subset.sort_values(
        ["support", "cooccurrence_count", "flavor_a", "flavor_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    subset["pair"] = subset.apply(format_pair, axis=1)
    subset["rank"] = range(1, len(subset) + 1)
    return subset.head(TOP_K).copy()


def select_lift_top_pairs(lift_df: pd.DataFrame) -> pd.DataFrame:
    """Select Lift top pairs with the existing recommended min_pair_count."""
    subset = lift_df[
        (lift_df["condition"] == CONDITION_NAME)
        & (lift_df["min_pair_count"] == RECOMMENDED_MIN_PAIR_COUNT)
    ].copy()
    subset = subset.sort_values(
        ["lift", "cooccurrence_count", "flavor_a", "flavor_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    subset["pair"] = subset.apply(format_pair, axis=1)
    subset["rank"] = range(1, len(subset) + 1)
    return subset.head(TOP_K).copy()


def export_rankings(
    support_top_df: pd.DataFrame,
    lift_top_df: pd.DataFrame,
) -> None:
    """Write compact CSVs for the plotted rankings."""
    support_top_df[
        ["rank", "pair", "flavor_a", "flavor_b", "support", "cooccurrence_count", "pair_key"]
    ].rename(columns={"cooccurrence_count": "n"}).to_csv(
        SUPPORT_TOP_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    lift_top_df[
        ["rank", "pair", "flavor_a", "flavor_b", "lift", "cooccurrence_count", "pair_key", "min_pair_count"]
    ].rename(columns={"cooccurrence_count": "n"}).to_csv(
        LIFT_TOP_CSV,
        index=False,
        encoding="utf-8-sig",
    )


def add_bar_labels(
    ax: plt.Axes,
    values: list[float],
    rows_df: pd.DataFrame,
    *,
    metric_name: str,
    decimals: int,
) -> None:
    """Add direct labels at the ends of the bars."""
    max_value = max(values) if values else 1.0
    x_offset = max_value * 0.015
    for idx, (_, row) in enumerate(rows_df.iterrows()):
        value = float(row[metric_name.lower()]) if metric_name.lower() in row else float(row[metric_name])
        ax.text(
            value + x_offset,
            idx,
            f"{metric_name} = {value:.{decimals}f}   n = {int(row['cooccurrence_count'])}",
            va="center",
            ha="left",
            fontsize=10,
            color="#222222",
            fontweight="bold",
        )


def build_axis_labels(rows_df: pd.DataFrame, *, metric_name: str) -> list[str]:
    """Build direct y-axis labels with rank and pair name."""
    labels = []
    for _, row in rows_df.iterrows():
        labels.append(f"{int(row['rank'])}. {row['pair']}")
    return labels


def draw_figure(
    *,
    support_top_df: pd.DataFrame,
    lift_top_df: pd.DataFrame,
    review_count: int,
) -> None:
    """Render the horizontal two-column poster figure."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7.2),
        dpi=320,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.patch.set_facecolor("white")
    support_color = "#2C7FB8"
    lift_color = "#D95F02"

    plot_specs = [
        (
            axes[0],
            support_top_df,
            "support",
            "Support上位：頻出するフレーバーペア",
            support_color,
            3,
            "レビュー内で頻繁に一緒に登場する関係",
        ),
        (
            axes[1],
            lift_top_df,
            "lift",
            "Lift上位：特徴的なフレーバーペア",
            lift_color,
            2,
            "単独での出現頻度に対して、特徴的に一緒に登場する関係",
        ),
    ]

    for ax, rows_df, metric_col, title, color, decimals, caption in plot_specs:
        values = rows_df[metric_col].astype(float).tolist()
        y_positions = list(range(len(rows_df)))
        ax.barh(y_positions, values, color=color, alpha=0.88, height=0.66)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(build_axis_labels(rows_df, metric_name=metric_col.upper()), fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(
            0.0,
            1.08,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            0.0,
            1.015,
            caption,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#444444",
        )
        add_bar_labels(
            ax,
            values,
            rows_df,
            metric_name=metric_col,
            decimals=decimals,
        )
        max_value = max(values) if values else 1.0
        ax.set_xlim(0, max_value * 1.52)
        ax.set_xlabel(metric_col.capitalize(), fontsize=11, fontweight="bold")

    fig.suptitle(
        "Support と Lift によるフレーバーペア比較（Condition B）",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.06,
        "Supportは定番候補、Liftは特徴的な候補の抽出に利用できる可能性がある。",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
    fig.text(
        0.5,
        0.025,
        (
            "共起は同一レビュー内で一緒に言及された関係であり、実際のミックスを直接示すものではない。"
            f"  Condition B: 2〜5フレーバー, n={review_count}, Liftは min_pair_count={RECOMMENDED_MIN_PAIR_COUNT}"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.1, 0.99, 0.9), w_pad=2.5)
    fig.savefig(FIGURE_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_self_checks() -> list[str]:
    """Run minimum ranking and tie-break checks."""
    messages = []

    support_toy = pd.DataFrame(
        [
            {
                "condition": CONDITION_NAME,
                "flavor_a": "A",
                "flavor_b": "B",
                "pair_key": "A||B",
                "cooccurrence_count": 3,
                "support": 0.3,
            },
            {
                "condition": CONDITION_NAME,
                "flavor_a": "A",
                "flavor_b": "C",
                "pair_key": "A||C",
                "cooccurrence_count": 3,
                "support": 0.3,
            },
            {
                "condition": CONDITION_NAME,
                "flavor_a": "A",
                "flavor_b": "A2",
                "pair_key": "A||A2",
                "cooccurrence_count": 3,
                "support": 0.3,
            },
        ]
    )
    support_ranked = select_support_top_pairs(support_toy)
    assert support_ranked["pair_key"].tolist() == ["A||A2", "A||B", "A||C"]
    messages.append("PASS: Support の同値時はフレーバー名辞書順で並ぶ")

    lift_toy = pd.DataFrame(
        [
            {
                "condition": CONDITION_NAME,
                "min_pair_count": RECOMMENDED_MIN_PAIR_COUNT,
                "flavor_a": "A",
                "flavor_b": "B",
                "pair_key": "A||B",
                "cooccurrence_count": 2,
                "lift": 5.0,
            },
            {
                "condition": CONDITION_NAME,
                "min_pair_count": RECOMMENDED_MIN_PAIR_COUNT,
                "flavor_a": "A",
                "flavor_b": "C",
                "pair_key": "A||C",
                "cooccurrence_count": 3,
                "lift": 5.0,
            },
        ]
    )
    lift_ranked = select_lift_top_pairs(lift_toy)
    assert lift_ranked["pair_key"].tolist() == ["A||C", "A||B"]
    messages.append("PASS: Lift の同値時は共起回数降順で並ぶ")

    assert all("×" in pair for pair in support_ranked["pair"].tolist())
    messages.append("PASS: ペア表記を A × B で統一できる")
    return messages


def main() -> None:
    condition_stats_df, cooccurrence_df, lift_df = load_inputs()
    review_count = int(
        condition_stats_df.loc[condition_stats_df["condition"] == CONDITION_NAME, "review_count"].iloc[0]
    )

    support_top_df = select_support_top_pairs(cooccurrence_df)
    lift_top_df = select_lift_top_pairs(lift_df)

    export_rankings(support_top_df, lift_top_df)
    draw_figure(
        support_top_df=support_top_df,
        lift_top_df=lift_top_df,
        review_count=review_count,
    )

    test_messages = run_self_checks()

    print("support/lift ranking outputs generated")
    print(f"- condition: {CONDITION_NAME}")
    print(f"- review_count: {review_count}")
    print(f"- lift min_pair_count: {RECOMMENDED_MIN_PAIR_COUNT}")
    print("- support top 5:")
    for _, row in support_top_df.iterrows():
        print(f"  - {int(row['rank'])}. {row['pair']} | support={float(row['support']):.3f} | n={int(row['cooccurrence_count'])}")
    print("- lift top 5:")
    for _, row in lift_top_df.iterrows():
        print(f"  - {int(row['rank'])}. {row['pair']} | lift={float(row['lift']):.2f} | n={int(row['cooccurrence_count'])}")
    print("- output files:")
    print(f"  - {FIGURE_PNG.relative_to(ROOT)}")
    print(f"  - {FIGURE_PDF.relative_to(ROOT)}")
    print(f"  - {SUPPORT_TOP_CSV.relative_to(ROOT)}")
    print(f"  - {LIFT_TOP_CSV.relative_to(ROOT)}")
    print("- tests:")
    for message in test_messages:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
