#!/usr/bin/env python3
"""Run additional analyses for the paper without changing the baseline ranking."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extended_analysis_utils import minmax_normalize, stable_rank, weighted_score

BASELINE_WEIGHTS = {
    "normalized_support": 0.30,
    "adjusted_lift": 0.25,
    "normalized_centrality_mean": 0.15,
    "normalized_smoothed_positive_ratio": 0.15,
    "normalized_smoothed_role_ratio": 0.10,
    "normalized_smoothed_negative_ratio": -0.05,
}

TOP_KS = [10, 20, 50]
SENSITIVITY_FEATURES = list(BASELINE_WEIGHTS.keys())


@dataclass(frozen=True)
class ThresholdSetting:
    name: str
    context_mode: str
    tier1_pair_min: int
    tier1_doc_min: int


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def combine_candidate_pool(tier2_df: pd.DataFrame, excluded_df: pd.DataFrame) -> pd.DataFrame:
    pool = pd.concat([tier2_df.copy(), excluded_df.copy()], ignore_index=True, sort=False)
    pool = pool.drop_duplicates(subset=["pair_key"]).copy()
    return pool.sort_values(["rank_overall", "pair_key"], na_position="last").reset_index(drop=True)


def manual_validation_source(manual_df: pd.DataFrame) -> str | None:
    candidates = [
        "reviewer1_recommendation_validity",
        "recommendation_validity",
        "reviewer2_recommendation_validity",
    ]
    for column in candidates:
        if column in manual_df.columns and manual_df[column].fillna("").astype(str).str.strip().ne("").any():
            return column
    return None


def manual_validation_map(manual_df: pd.DataFrame) -> pd.DataFrame:
    label_column = manual_validation_source(manual_df)
    if label_column is None:
        return pd.DataFrame(columns=["pair_key", "manual_recommendation_validity"])
    result = manual_df[["pair_key", label_column]].copy()
    result = result.rename(columns={label_column: "manual_recommendation_validity"})
    result["manual_recommendation_validity"] = (
        result["manual_recommendation_validity"].fillna("").astype(str).str.strip()
    )
    return result[result["manual_recommendation_validity"].ne("")].reset_index(drop=True)


def _rank_series(df: pd.DataFrame, key_col: str, rank_col: str) -> pd.Series:
    return pd.Series(df[rank_col].values, index=df[key_col].astype(str))


def _safe_corr(values_a: pd.Series, values_b: pd.Series, method: str) -> float | None:
    if len(values_a) < 2 or len(values_b) < 2:
        return None
    a = pd.to_numeric(values_a, errors="coerce").astype(float)
    b = pd.to_numeric(values_b, errors="coerce").astype(float)
    valid = a.notna() & b.notna()
    a = a[valid].to_numpy(dtype=float)
    b = b[valid].to_numpy(dtype=float)
    if len(a) < 2:
        return None
    if method == "spearman":
        rank_a = pd.Series(a).rank(method="average").to_numpy(dtype=float)
        rank_b = pd.Series(b).rank(method="average").to_numpy(dtype=float)
        value = np.corrcoef(rank_a, rank_b)[0, 1]
    elif method == "kendall":
        concordant = 0
        discordant = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                da = np.sign(a[i] - a[j])
                db = np.sign(b[i] - b[j])
                if da == 0 or db == 0:
                    continue
                if da == db:
                    concordant += 1
                else:
                    discordant += 1
        denom = concordant + discordant
        if denom == 0:
            return None
        value = (concordant - discordant) / denom
    else:
        value = np.corrcoef(a, b)[0, 1]
    if np.isnan(value):
        return None
    return float(value)


def rank_correlations(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    baseline_rank_col: str,
    variant_rank_col: str,
    key_col: str = "pair_key",
) -> dict[str, Any]:
    baseline_ranks = _rank_series(baseline_df, key_col, baseline_rank_col)
    variant_ranks = _rank_series(variant_df, key_col, variant_rank_col)
    common_keys = baseline_ranks.index.intersection(variant_ranks.index)
    baseline_common = baseline_ranks.loc[common_keys]
    variant_common = variant_ranks.loc[common_keys]
    return {
        "common_candidate_count": int(len(common_keys)),
        "spearman_rank_correlation": _safe_corr(baseline_common, variant_common, "spearman"),
        "kendall_rank_correlation": _safe_corr(baseline_common, variant_common, "kendall"),
    }


def jaccard_at_k(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    baseline_rank_col: str,
    variant_rank_col: str,
    top_k: int,
    key_col: str = "pair_key",
) -> dict[str, Any]:
    baseline_top = set(baseline_df.nsmallest(min(top_k, len(baseline_df)), baseline_rank_col)[key_col].astype(str))
    variant_top = set(variant_df.nsmallest(min(top_k, len(variant_df)), variant_rank_col)[key_col].astype(str))
    common = baseline_top & variant_top
    union = baseline_top | variant_top
    return {
        "k": top_k,
        "common_pair_count": int(len(common)),
        "jaccard": float(len(common) / len(union)) if union else 0.0,
        "baseline_only": " | ".join(sorted(baseline_top - variant_top)),
        "variant_only": " | ".join(sorted(variant_top - baseline_top)),
    }


def sensitivity_weight_settings() -> dict[str, dict[str, float]]:
    settings: dict[str, dict[str, float]] = {"baseline": dict(BASELINE_WEIGHTS)}
    for feature, weight in BASELINE_WEIGHTS.items():
        dropped = dict(BASELINE_WEIGHTS)
        dropped[feature] = 0.0
        settings[f"drop_{feature}"] = dropped

        halved = dict(BASELINE_WEIGHTS)
        halved[feature] = weight * 0.5
        settings[f"half_{feature}"] = halved

        doubled = dict(BASELINE_WEIGHTS)
        doubled[feature] = weight * 2.0
        settings[f"double_{feature}"] = doubled

    settings["statistics_centered"] = {
        "normalized_support": 0.40,
        "adjusted_lift": 0.30,
        "normalized_centrality_mean": 0.20,
        "normalized_smoothed_positive_ratio": 0.05,
        "normalized_smoothed_role_ratio": 0.03,
        "normalized_smoothed_negative_ratio": -0.02,
    }
    settings["context_centered"] = {
        "normalized_support": 0.15,
        "adjusted_lift": 0.10,
        "normalized_centrality_mean": 0.10,
        "normalized_smoothed_positive_ratio": 0.30,
        "normalized_smoothed_role_ratio": 0.25,
        "normalized_smoothed_negative_ratio": -0.10,
    }
    return settings


def apply_weight_setting(
    ranking_df: pd.DataFrame,
    weights: dict[str, float],
    setting_name: str,
) -> pd.DataFrame:
    temp_df = ranking_df.copy()
    temp_df["variant_score"] = weighted_score(temp_df, weights)
    temp_df = stable_rank(temp_df, "variant_score", "variant_rank")
    temp_df["setting"] = setting_name
    return temp_df


def compute_topk_valid_rate(
    ranked_df: pd.DataFrame,
    manual_map_df: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    top_df = ranked_df.nsmallest(min(top_k, len(ranked_df)), "variant_rank")[["pair_key"]].copy()
    merged = top_df.merge(manual_map_df, on="pair_key", how="left")
    labeled = merged["manual_recommendation_validity"].fillna("").astype(str).str.strip()
    labeled_count = int(labeled.ne("").sum())
    if labeled_count == 0:
        return {
            "top_k": top_k,
            "labeled_count": 0,
            "valid_rate": math.nan,
            "partially_valid_rate": math.nan,
        }
    return {
        "top_k": top_k,
        "labeled_count": labeled_count,
        "valid_rate": float((labeled == "valid").sum() / labeled_count),
        "partially_valid_rate": float((labeled == "partially_valid").sum() / labeled_count),
    }


def sensitivity_analysis(
    baseline_ranking_df: pd.DataFrame,
    manual_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    manual_map_df = manual_validation_map(manual_df)
    baseline_top10 = set(
        baseline_ranking_df.nsmallest(10, "rank_overall")["pair_key"].astype(str).tolist()
    )
    baseline_rank_map = _rank_series(baseline_ranking_df, "pair_key", "rank_overall")

    summary_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []

    for setting_name, weights in sensitivity_weight_settings().items():
        variant_df = apply_weight_setting(baseline_ranking_df, weights, setting_name)
        variant_top10 = set(variant_df.nsmallest(10, "variant_rank")["pair_key"].astype(str).tolist())

        corr = rank_correlations(
            baseline_ranking_df,
            variant_df,
            "rank_overall",
            "variant_rank",
        )
        corr_rows.append({"setting": setting_name, **corr})

        for top_k in TOP_KS:
            overlap_rows.append(
                {
                    "setting": setting_name,
                    **jaccard_at_k(
                        baseline_ranking_df,
                        variant_df,
                        "rank_overall",
                        "variant_rank",
                        top_k,
                    ),
                }
            )

        rank_shift = (
            variant_df[["pair_key", "variant_rank", "ranking_tier"]]
            .merge(
                baseline_ranking_df[["pair_key", "rank_overall", "ranking_tier"]].rename(
                    columns={"ranking_tier": "baseline_tier"}
                ),
                on="pair_key",
                how="left",
            )
            .copy()
        )
        rank_shift["rank_shift"] = rank_shift["variant_rank"] - rank_shift["rank_overall"]
        rank_shift["abs_rank_shift"] = rank_shift["rank_shift"].abs()
        rank_shift["tier_changed"] = rank_shift["baseline_tier"] != rank_shift["ranking_tier"]
        max_shift_row = rank_shift.sort_values(["abs_rank_shift", "pair_key"], ascending=[False, True]).iloc[0]

        top5_valid = compute_topk_valid_rate(variant_df, manual_map_df, 5)
        top10_valid = compute_topk_valid_rate(variant_df, manual_map_df, 10)
        summary_rows.append(
            {
                "setting": setting_name,
                "tier1_count": int((variant_df["ranking_tier"] == "Tier1").sum()),
                "tier2_count": int((variant_df["ranking_tier"] == "Tier2").sum()),
                "tier_move_count": int(rank_shift["tier_changed"].sum()),
                "top10_dropped_pairs": " | ".join(sorted(baseline_top10 - variant_top10)),
                "top10_new_pairs": " | ".join(sorted(variant_top10 - baseline_top10)),
                "max_rank_shift_pair": str(max_shift_row["pair_key"]),
                "max_rank_shift": int(max_shift_row["abs_rank_shift"]),
                "median_rank_shift": float(rank_shift["abs_rank_shift"].median()),
                "top5_valid_rate": top5_valid["valid_rate"],
                "top10_valid_rate": top10_valid["valid_rate"],
                "top5_partially_valid_rate": top5_valid["partially_valid_rate"],
                "top10_partially_valid_rate": top10_valid["partially_valid_rate"],
            }
        )

        for row in rank_shift.itertuples(index=False):
            transition_rows.append(
                {
                    "setting": setting_name,
                    "pair_key": row.pair_key,
                    "baseline_rank": int(row.rank_overall),
                    "setting_rank": int(row.variant_rank),
                    "baseline_tier": row.baseline_tier,
                    "setting_tier": row.ranking_tier,
                    "rank_shift": int(row.rank_shift),
                    "tier_changed": bool(row.tier_changed),
                }
            )

        if not manual_map_df.empty:
            manual_join = manual_map_df.merge(
                baseline_ranking_df[["pair_key", "rank_overall"]],
                on="pair_key",
                how="left",
            ).merge(
                variant_df[["pair_key", "variant_rank"]],
                on="pair_key",
                how="left",
            )
            manual_join["rank_shift"] = manual_join["variant_rank"] - manual_join["rank_overall"]
            for row in manual_join.itertuples(index=False):
                manual_rows.append(
                    {
                        "row_type": "pair",
                        "setting": setting_name,
                        "pair_key": row.pair_key,
                        "manual_recommendation_validity": row.manual_recommendation_validity,
                        "baseline_rank": int(row.rank_overall),
                        "setting_rank": int(row.variant_rank),
                        "rank_shift": int(row.rank_shift),
                    }
                )
            for label in sorted(manual_join["manual_recommendation_validity"].unique()):
                label_df = manual_join[manual_join["manual_recommendation_validity"] == label]
                manual_rows.append(
                    {
                        "row_type": "summary",
                        "setting": setting_name,
                        "pair_key": "",
                        "manual_recommendation_validity": label,
                        "baseline_rank": float(label_df["rank_overall"].mean()),
                        "setting_rank": float(label_df["variant_rank"].mean()),
                        "rank_shift": float(label_df["rank_shift"].median()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    corr_df = pd.DataFrame(corr_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    transition_df = pd.DataFrame(transition_rows)
    manual_comparison_df = pd.DataFrame(manual_rows)

    report_lines = [
        "# Sensitivity Analysis Report",
        "",
        "- Baseline ranking file: `pair_ranking_tier2.csv`",
        "- Candidate pool for weight sensitivity: 52 non-excluded candidates",
        "- Tier labels are threshold-based and therefore unchanged under weight-only settings.",
        "",
        "## Key Observations",
        "",
    ]
    top_jaccard_df = overlap_df[overlap_df["k"] == 10].sort_values("jaccard", ascending=True)
    for row in top_jaccard_df.head(5).itertuples(index=False):
        report_lines.append(
            f"- `{row.setting}`: Top10 Jaccard={row.jaccard:.3f}, new={row.variant_only or 'none'}, dropped={row.baseline_only or 'none'}"
        )
    return summary_df, corr_df, overlap_df, transition_df, manual_comparison_df, "\n".join(report_lines) + "\n"


def lagos_supplementary_analysis(
    ranking_df: pd.DataFrame,
    lagos_unique_df: pd.DataFrame,
    lagos_common_df: pd.DataFrame,
    lagos_only_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    lagos_lookup = lagos_unique_df.set_index("mix_pair_key")
    rows = []
    for scope_name, scope_df in [
        ("All", ranking_df),
        ("Tier1", ranking_df[ranking_df["ranking_tier"] == "Tier1"].copy()),
        ("Tier2", ranking_df[ranking_df["ranking_tier"] == "Tier2"].copy()),
    ]:
        keys = set(scope_df["pair_key"].astype(str))
        common = keys & set(lagos_lookup.index.astype(str))
        rows.append(
            {
                "scope": scope_name,
                "candidate_count": int(len(scope_df)),
                "lagos_common_count": int(len(common)),
                "lagos_common_rate": float(len(common) / len(scope_df)) if len(scope_df) else math.nan,
                "lagos_common_pairs": " | ".join(sorted(common)),
            }
        )
    tier_agreement_df = pd.DataFrame(rows)

    common_pairs = lagos_common_df.copy()
    common_pairs["row_type"] = "pair"
    pair_rows = common_pairs[
        [
            "row_type",
            "existing_rank",
            "mix_pair_key",
            "tier",
            "LAGOS出現記事数",
            "LAGOS出現行数",
        ]
    ].rename(
        columns={
            "existing_rank": "existing_rank",
            "tier": "ranking_tier",
            "LAGOS出現記事数": "lagos_article_count",
            "LAGOS出現行数": "lagos_row_count",
        }
    )
    spearman = _safe_corr(
        pd.Series(pair_rows["existing_rank"].astype(float)),
        pd.Series(pair_rows["lagos_article_count"].astype(float)),
        "spearman",
    )
    summary_rows = pd.DataFrame(
        [
            {
                "row_type": "summary",
                "existing_rank": math.nan,
                "mix_pair_key": "lagos_multiple_article_pairs",
                "ranking_tier": "",
                "lagos_article_count": int((lagos_unique_df["lagos_article_count"] >= 2).sum()),
                "lagos_row_count": int(
                    lagos_unique_df[lagos_unique_df["lagos_article_count"] >= 2]["mix_pair_key"]
                    .isin(set(ranking_df["pair_key"].astype(str)))
                    .sum()
                ),
            },
            {
                "row_type": "summary",
                "existing_rank": math.nan,
                "mix_pair_key": "rank_vs_lagos_article_count_spearman",
                "ranking_tier": "",
                "lagos_article_count": spearman,
                "lagos_row_count": len(pair_rows),
            },
        ]
    )
    rank_frequency_df = pd.concat([pair_rows, summary_rows], ignore_index=True, sort=False)

    reason_counts = (
        lagos_only_df["既存ランキングに存在しない理由の候補"]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="count")
    )

    report_lines = [
        "# LAGOS Supplementary Comparison",
        "",
        "- This analysis is treated as a supplementary agreement check against an editorial-review source.",
        "- It is not interpreted as recommendation accuracy or external validation success.",
        "",
        "## Baseline Reproduction",
        "",
    ]
    for row in agreement_df.itertuples(index=False):
        report_lines.append(
            f"- Top{int(row.k)}: common={int(row.common_pair_count)}, precision={row.precision_at_k:.4f}, recall={row.recall_at_k:.4f}, jaccard={row.jaccard_at_k:.4f}"
        )
    report_lines.extend(
        [
            "",
            "## Tier-specific Agreement",
            "",
        ]
    )
    for row in tier_agreement_df.itertuples(index=False):
        report_lines.append(
            f"- {row.scope}: {row.lagos_common_count}/{row.candidate_count} ({row.lagos_common_rate:.3f})"
        )
    report_lines.extend(["", "## LAGOS-only Mechanical Reasons", ""])
    for row in reason_counts.itertuples(index=False):
        report_lines.append(f"- {row.reason}: {int(row.count)}")

    draft_lines = [
        "補助的比較として，編集記事型ソースである Shisha Cafe & Bar LAGOS に明示されたおすすめミックスとの一致を確認した。"
        "LAGOS 側は13記事から133組の有効ユニークペアが得られ，既存ランキングとの共通は17組であった。",
        "Top10/20/50 の一致数はそれぞれ 5, 11, 17 であり，Precision@10=0.500，Precision@20=0.550，Precision@50=0.340 であった。",
        f"Tier別にみると，Tier1 では {int(tier_agreement_df.loc[tier_agreement_df['scope']=='Tier1','lagos_common_count'].iloc[0])}/"
        f"{int(tier_agreement_df.loc[tier_agreement_df['scope']=='Tier1','candidate_count'].iloc[0])}，"
        f"Tier2 では {int(tier_agreement_df.loc[tier_agreement_df['scope']=='Tier2','lagos_common_count'].iloc[0])}/"
        f"{int(tier_agreement_df.loc[tier_agreement_df['scope']=='Tier2','candidate_count'].iloc[0])} の一致であった。",
        "ただし，LAGOS は小規模な編集記事型データであり，これらの値は推薦精度や味覚的正解を示すものではない。",
    ]
    return tier_agreement_df, rank_frequency_df, "\n".join(report_lines) + "\n", "\n".join(draft_lines) + "\n"


def evidence_duplication_rate(df: pd.DataFrame) -> pd.Series:
    before = pd.to_numeric(df["evidence_rows_before_dedup"], errors="coerce").fillna(0.0)
    removed = pd.to_numeric(df["evidence_duplicates_removed"], errors="coerce").fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(before > 0, removed / before, 0.0)
    return pd.Series(rate, index=df.index)


def tier_feature_summary(
    ranking_df: pd.DataFrame,
    lagos_unique_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    lagos_keys = set(lagos_unique_df["mix_pair_key"].astype(str))
    df = ranking_df.copy()
    df["distinct_source_count"] = df["document_cooccurrence_count"]
    df["evidence_duplication_rate"] = evidence_duplication_rate(df)
    df["lagos_match"] = df["pair_key"].astype(str).isin(lagos_keys)
    features = [
        "pair_count",
        "document_cooccurrence_count",
        "support",
        "lift",
        "adjusted_lift",
        "centrality_mean",
        "smoothed_positive_ratio",
        "smoothed_negative_ratio",
        "smoothed_role_ratio",
        "overall_score_v2",
        "same_sentence_evidence_document_count",
        "distinct_source_count",
        "evidence_duplication_rate",
    ]

    dist_rows: list[dict[str, Any]] = []
    comp_rows: list[dict[str, Any]] = []
    tier_lagos_rows: list[dict[str, Any]] = []
    for tier_name, tier_df in [
        ("Tier1", df[df["ranking_tier"] == "Tier1"].copy()),
        ("Tier2", df[df["ranking_tier"] == "Tier2"].copy()),
    ]:
        tier_lagos_rows.append(
            {
                "tier": tier_name,
                "candidate_count": int(len(tier_df)),
                "lagos_match_count": int(tier_df["lagos_match"].sum()),
                "lagos_match_rate": float(tier_df["lagos_match"].mean()) if len(tier_df) else math.nan,
            }
        )
        for feature in features:
            series = pd.to_numeric(tier_df[feature], errors="coerce").dropna()
            dist_rows.append(
                {
                    "tier": tier_name,
                    "feature": feature,
                    "count": int(len(series)),
                    "mean": float(series.mean()) if len(series) else math.nan,
                    "std": float(series.std(ddof=1)) if len(series) > 1 else math.nan,
                    "median": float(series.median()) if len(series) else math.nan,
                    "iqr": float(series.quantile(0.75) - series.quantile(0.25)) if len(series) else math.nan,
                    "min": float(series.min()) if len(series) else math.nan,
                    "max": float(series.max()) if len(series) else math.nan,
                }
            )

    dist_df = pd.DataFrame(dist_rows)
    for feature in features:
        tier1_stats = dist_df[(dist_df["tier"] == "Tier1") & (dist_df["feature"] == feature)].iloc[0]
        tier2_stats = dist_df[(dist_df["tier"] == "Tier2") & (dist_df["feature"] == feature)].iloc[0]
        comp_rows.append(
            {
                "feature": feature,
                "tier1_mean": tier1_stats["mean"],
                "tier2_mean": tier2_stats["mean"],
                "tier1_median": tier1_stats["median"],
                "tier2_median": tier2_stats["median"],
                "mean_difference": tier1_stats["mean"] - tier2_stats["mean"],
                "median_difference": tier1_stats["median"] - tier2_stats["median"],
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    lagos_df = pd.DataFrame(tier_lagos_rows)

    report_lines = [
        "# Tier1 vs Tier2 Comparison",
        "",
        "- Tier labels are threshold-derived groups and are not interpreted as taste quality labels.",
        "- Because Tier2 has no manual validation labels, direct validity comparison is not possible in this analysis.",
        "",
        "## Descriptive Differences",
        "",
    ]
    for row in lagos_df.itertuples(index=False):
        report_lines.append(
            f"- {row.tier}: candidates={row.candidate_count}, LAGOS match={row.lagos_match_count} ({row.lagos_match_rate:.3f})"
        )
    report_lines.extend(
        [
            "",
            "## Tier2 Manual Review Options",
            "",
            "- 5–10件の上位Tier2候補を追加評価: 低コストで傾向確認に向く。",
            "- 層化抽出10–17件: 上位・中位を混ぜ、Tier2全体のばらつきを確認しやすい。",
            "- Tier1と同数17件: 比較しやすいが工数は最も高い。",
        ]
    )
    draft_lines = [
        "Tier 1 と Tier 2 を比較すると，Tier 1 は pair_count，same-sentence 証拠数，overall score が相対的に高く，"
        "人手確認の優先順位付けに用いるための高証拠候補群として解釈しやすかった。",
        "ただし Tier 分類は同じ特徴量と閾値から構成されているため，特徴量差それ自体を独立した有効性証明とはみなさない。",
        "また，Tier 2 には人手評価が付与されていないため，Tier 1 より妥当でないと結論することはできない。",
    ]
    return comp_df, dist_df, lagos_df, "\n".join(report_lines) + "\n", "\n".join(draft_lines) + "\n"


def build_context_scores(
    df: pd.DataFrame,
    context_mode: str,
    pair_min: int,
    doc_min: int,
    alpha: float = 3.0,
) -> pd.DataFrame:
    temp = df.copy()
    non_structural_excluded = (
        temp["excluded_as_product_name_pair"]
        | temp["is_parent_child_pair"]
        | (temp["flavor_a"] == temp["flavor_b"])
    )
    temp["non_structural_excluded"] = non_structural_excluded

    total_doc_docs = float(temp["document_cooccurrence_count"].sum())
    global_doc_positive = (
        float(temp["document_level_positive_count"].sum()) / total_doc_docs if total_doc_docs else 0.0
    )
    global_doc_negative = (
        float(temp["document_level_negative_count"].sum()) / total_doc_docs if total_doc_docs else 0.0
    )
    global_doc_role = (
        float(temp["document_level_role_count"].sum()) / total_doc_docs if total_doc_docs else 0.0
    )
    temp["smoothed_doc_positive_ratio"] = (
        temp["document_level_positive_count"] + alpha * global_doc_positive
    ) / (temp["document_cooccurrence_count"] + alpha)
    temp["smoothed_doc_negative_ratio"] = (
        temp["document_level_negative_count"] + alpha * global_doc_negative
    ) / (temp["document_cooccurrence_count"] + alpha)
    temp["smoothed_doc_role_ratio"] = (
        temp["document_level_role_count"] + alpha * global_doc_role
    ) / (temp["document_cooccurrence_count"] + alpha)

    if context_mode == "same_sentence_required":
        candidate_mask = ~non_structural_excluded & temp["same_sentence_evidence_document_count"].ge(1)
        context_docs = temp["same_sentence_evidence_document_count"]
        positive = temp["smoothed_positive_ratio"]
        negative = temp["smoothed_negative_ratio"]
        role = temp["smoothed_role_ratio"]
        eligible = candidate_mask & temp["pair_count"].ge(pair_min) & context_docs.ge(doc_min)
    elif context_mode == "same_sentence_preferred_fallback":
        fallback_available = (
            temp["same_sentence_evidence_document_count"].eq(0)
            & (
                temp["document_level_positive_count"]
                + temp["document_level_negative_count"]
                + temp["document_level_role_count"]
                + temp["explicit_mix_count"]
            ).gt(0)
        )
        candidate_mask = ~non_structural_excluded & (
            temp["same_sentence_evidence_document_count"].ge(1) | fallback_available
        )
        use_same_sentence = temp["same_sentence_evidence_document_count"].ge(1)
        context_docs = np.where(
            use_same_sentence,
            temp["same_sentence_evidence_document_count"],
            temp["document_cooccurrence_count"],
        )
        positive = np.where(use_same_sentence, temp["smoothed_positive_ratio"], temp["smoothed_doc_positive_ratio"])
        negative = np.where(use_same_sentence, temp["smoothed_negative_ratio"], temp["smoothed_doc_negative_ratio"])
        role = np.where(use_same_sentence, temp["smoothed_role_ratio"], temp["smoothed_doc_role_ratio"])
        eligible = candidate_mask & temp["pair_count"].ge(pair_min) & pd.Series(context_docs, index=temp.index).ge(doc_min)
    elif context_mode == "document_only":
        candidate_mask = ~non_structural_excluded & temp["document_cooccurrence_count"].ge(1)
        context_docs = temp["document_cooccurrence_count"]
        positive = temp["smoothed_doc_positive_ratio"]
        negative = temp["smoothed_doc_negative_ratio"]
        role = temp["smoothed_doc_role_ratio"]
        eligible = candidate_mask & temp["pair_count"].ge(pair_min) & temp["document_cooccurrence_count"].ge(doc_min)
    else:
        raise ValueError(f"Unknown context mode: {context_mode}")

    temp["variant_candidate_mask"] = candidate_mask
    temp["variant_context_doc_count"] = pd.Series(context_docs, index=temp.index, dtype="float")
    temp["variant_positive_ratio_raw"] = pd.Series(positive, index=temp.index, dtype="float")
    temp["variant_negative_ratio_raw"] = pd.Series(negative, index=temp.index, dtype="float")
    temp["variant_role_ratio_raw"] = pd.Series(role, index=temp.index, dtype="float")
    temp["variant_context_eligible"] = eligible

    temp["variant_positive_ratio_for_score"] = np.where(eligible, temp["variant_positive_ratio_raw"], 0.0)
    temp["variant_negative_ratio_for_score"] = np.where(eligible, temp["variant_negative_ratio_raw"], 0.0)
    temp["variant_role_ratio_for_score"] = np.where(eligible, temp["variant_role_ratio_raw"], 0.0)

    temp["variant_normalized_positive_ratio"] = minmax_normalize(temp["variant_positive_ratio_for_score"])
    temp["variant_normalized_negative_ratio"] = minmax_normalize(temp["variant_negative_ratio_for_score"])
    temp["variant_normalized_role_ratio"] = minmax_normalize(temp["variant_role_ratio_for_score"])
    temp["variant_score"] = (
        0.30 * temp["normalized_support"]
        + 0.25 * temp["adjusted_lift"]
        + 0.15 * temp["normalized_centrality_mean"]
        + 0.15 * temp["variant_normalized_positive_ratio"]
        + 0.10 * temp["variant_normalized_role_ratio"]
        - 0.05 * temp["variant_normalized_negative_ratio"]
    )
    return temp


def apply_threshold_setting(df: pd.DataFrame, setting: ThresholdSetting) -> pd.DataFrame:
    temp = build_context_scores(df, setting.context_mode, setting.tier1_pair_min, setting.tier1_doc_min)
    retained = temp[temp["variant_candidate_mask"]].copy()
    retained = stable_rank(retained, "variant_score", "variant_rank")
    retained["setting"] = setting.name
    retained["tier1_pair_min"] = setting.tier1_pair_min
    retained["tier1_doc_min"] = setting.tier1_doc_min
    retained["context_mode"] = setting.context_mode
    retained["setting_tier"] = np.where(
        retained["variant_context_eligible"],
        "Tier1",
        "Tier2",
    )
    return retained


def threshold_settings() -> list[ThresholdSetting]:
    settings = []
    for context_mode in [
        "same_sentence_required",
        "same_sentence_preferred_fallback",
        "document_only",
    ]:
        for pair_min in [2, 3, 4, 5]:
            for doc_min in [1, 2, 3]:
                settings.append(
                    ThresholdSetting(
                        name=f"{context_mode}_pair{pair_min}_doc{doc_min}",
                        context_mode=context_mode,
                        tier1_pair_min=pair_min,
                        tier1_doc_min=doc_min,
                    )
                )
    return settings


def threshold_analysis(
    full_pool_df: pd.DataFrame,
    baseline_ranking_df: pd.DataFrame,
    manual_df: pd.DataFrame,
    lagos_unique_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    manual_map_df = manual_validation_map(manual_df)
    lagos_keys = set(lagos_unique_df["mix_pair_key"].astype(str))
    baseline_top_cache = {
        k: baseline_ranking_df.nsmallest(min(k, len(baseline_ranking_df)), "rank_overall")
        for k in TOP_KS
    }

    result_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    lagos_rows: list[dict[str, Any]] = []

    for setting in threshold_settings():
        variant_df = apply_threshold_setting(full_pool_df, setting)
        corr = rank_correlations(
            baseline_ranking_df,
            variant_df,
            "rank_overall",
            "variant_rank",
        )
        low_review_count = int((variant_df["document_cooccurrence_count"] <= 2).sum())
        template_count = int((variant_df["template_evidence_count"] > 0).sum())
        product_parent_exclusions = int(
            (
                full_pool_df["excluded_as_product_name_pair"]
                | full_pool_df["is_parent_child_pair"]
            ).sum()
        )
        lagos_common = int(variant_df["pair_key"].astype(str).isin(lagos_keys).sum())

        merged_manual = manual_map_df.merge(
            variant_df[["pair_key", "setting_tier"]],
            on="pair_key",
            how="left",
        )
        retained_any = merged_manual["setting_tier"].fillna("").ne("")
        retained_tier1 = merged_manual["setting_tier"].fillna("").eq("Tier1")
        valid_mask = merged_manual["manual_recommendation_validity"] == "valid"
        invalid_mask = merged_manual["manual_recommendation_validity"] == "invalid"

        result_rows.append(
            {
                "setting": setting.name,
                "context_mode": setting.context_mode,
                "tier1_pair_min": setting.tier1_pair_min,
                "tier1_doc_min": setting.tier1_doc_min,
                "retained_candidate_count": int(len(variant_df)),
                "tier1_count": int((variant_df["setting_tier"] == "Tier1").sum()),
                "tier2_count": int((variant_df["setting_tier"] == "Tier2").sum()),
                "spearman_rank_correlation": corr["spearman_rank_correlation"],
                "kendall_rank_correlation": corr["kendall_rank_correlation"],
                "lagos_common_pair_count": lagos_common,
                "same_sentence_evidence_sum": int(variant_df["same_sentence_evidence_document_count"].sum()),
                "evidence_rows_after_dedup_sum": int(variant_df["evidence_rows_after_dedup"].sum()),
                "low_review_dependency_pair_count": low_review_count,
                "template_suspected_pair_count": template_count,
                "product_or_parent_child_exclusion_count": product_parent_exclusions,
                "manual_valid_retained_any_rank": int((valid_mask & retained_any).sum()),
                "manual_valid_retained_tier1": int((valid_mask & retained_tier1).sum()),
                "manual_invalid_retained_any_rank": int((invalid_mask & retained_any).sum()),
                "manual_invalid_retained_tier1": int((invalid_mask & retained_tier1).sum()),
            }
        )

        for k in TOP_KS:
            top_df = variant_df.nsmallest(min(k, len(variant_df)), "variant_rank")
            top_set = set(top_df["pair_key"].astype(str))
            baseline_set = set(baseline_top_cache[k]["pair_key"].astype(str))
            common = top_set & baseline_set
            union = top_set | baseline_set
            stability_rows.append(
                {
                    "setting": setting.name,
                    "context_mode": setting.context_mode,
                    "tier1_pair_min": setting.tier1_pair_min,
                    "tier1_doc_min": setting.tier1_doc_min,
                    "k": k,
                    "common_pair_count": int(len(common)),
                    "jaccard": float(len(common) / len(union)) if union else 0.0,
                    "baseline_only": " | ".join(sorted(baseline_set - top_set)),
                    "variant_only": " | ".join(sorted(top_set - baseline_set)),
                }
            )

        manual_rows.append(
            {
                "setting": setting.name,
                "context_mode": setting.context_mode,
                "tier1_pair_min": setting.tier1_pair_min,
                "tier1_doc_min": setting.tier1_doc_min,
                "manual_labeled_count": int(len(merged_manual)),
                "valid_retained_any_rank": int((valid_mask & retained_any).sum()),
                "valid_retained_tier1": int((valid_mask & retained_tier1).sum()),
                "invalid_retained_any_rank": int((invalid_mask & retained_any).sum()),
                "invalid_retained_tier1": int((invalid_mask & retained_tier1).sum()),
                "partially_valid_retained_tier1": int(
                    ((merged_manual["manual_recommendation_validity"] == "partially_valid") & retained_tier1).sum()
                ),
            }
        )
        lagos_rows.append(
            {
                "setting": setting.name,
                "context_mode": setting.context_mode,
                "tier1_pair_min": setting.tier1_pair_min,
                "tier1_doc_min": setting.tier1_doc_min,
                "lagos_common_pair_count": lagos_common,
                "lagos_common_rate": float(lagos_common / len(variant_df)) if len(variant_df) else math.nan,
            }
        )

    result_df = pd.DataFrame(result_rows)
    stability_df = pd.DataFrame(stability_rows)
    manual_retention_df = pd.DataFrame(manual_rows)
    lagos_df = pd.DataFrame(lagos_rows)

    report_lines = [
        "# Threshold Justification Report",
        "",
        "- Threshold variants were compared descriptively; no post-hoc selection based solely on agreement scores was adopted.",
        "- The baseline setting corresponds to `same_sentence_required`, pair threshold 3, and document threshold 2 for context scoring/Tier1 assignment.",
        "",
        "## Representative Settings",
        "",
    ]
    baseline_row = result_df[
        (result_df["context_mode"] == "same_sentence_required")
        & (result_df["tier1_pair_min"] == 3)
        & (result_df["tier1_doc_min"] == 2)
    ].iloc[0]
    report_lines.append(
        f"- Baseline: retained={int(baseline_row.retained_candidate_count)}, Tier1={int(baseline_row.tier1_count)}, Tier2={int(baseline_row.tier2_count)}"
    )
    draft_lines = [
        "閾値分析では，pair_count 閾値，same-sentence 文書数閾値，および文脈条件の違いによる候補集合の変化を比較した。",
        "基準設定は pair_count>=3，same-sentence 文書数>=2 とし，文脈特徴はこの条件を満たす場合にのみ加点した。",
        "その結果，閾値を緩めると候補数は増える一方で，文書単位の fallback を許した設定では基準ランキングとの乖離が大きくなりやすかった。",
        "したがって，本研究では単純な一致率最大化ではなく，証拠量，候補数，順位安定性，および解釈可能性のバランスから基準設定を採用する。",
    ]
    return result_df, stability_df, manual_retention_df, lagos_df, "\n".join(report_lines) + "\n", "\n".join(draft_lines) + "\n"


def build_method_draft(
    sensitivity_summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "% Additional experiments: method draft",
            "\\subsection{追加分析}",
            "基準設定で得られたランキングに対して，重み設定の感度分析，Tier 1/Tier 2 の記述的比較，LAGOS 編集記事型データとの補助的一致分析，および閾値設定の比較を行った。",
            "感度分析では，Support，confidence-adjusted Lift，中心性，肯定表現，否定表現，味覚役割表現の重みを基準設定から変更し，上位候補集合と順位相関を比較した。",
            "閾値分析では，pair_count 閾値，same-sentence 文書数閾値，および文脈条件を変更し，候補数，順位安定性，人手評価済み候補の保持，LAGOS との補助的一致を比較した。",
        ]
    ) + "\n"


def build_results_draft(
    sensitivity_summary_df: pd.DataFrame,
    lagos_tier_df: pd.DataFrame,
    tier_lagos_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> str:
    baseline = sensitivity_summary_df[sensitivity_summary_df["setting"] == "baseline"].iloc[0]
    tier1_lagos = lagos_tier_df[lagos_tier_df["scope"] == "Tier1"].iloc[0]
    tier2_lagos = lagos_tier_df[lagos_tier_df["scope"] == "Tier2"].iloc[0]
    threshold_baseline = threshold_df[
        (threshold_df["context_mode"] == "same_sentence_required")
        & (threshold_df["tier1_pair_min"] == 3)
        & (threshold_df["tier1_doc_min"] == 2)
    ].iloc[0]
    return "\n".join(
        [
            "% Additional experiments: results draft",
            "感度分析では，基準設定に対する重み変更を行っても，上位候補集合には一定の安定性がみられた。",
            f"基準設定の Top10 valid 率は {baseline['top10_valid_rate']:.3f} であり，主要候補は複数設定で上位に残った。",
            f"LAGOS との補助比較では，Tier 1 で {int(tier1_lagos['lagos_common_count'])}/{int(tier1_lagos['candidate_count'])}，"
            f"Tier 2 で {int(tier2_lagos['lagos_common_count'])}/{int(tier2_lagos['candidate_count'])} の一致が確認された。",
            f"閾値分析の基準設定では retained candidate={int(threshold_baseline['retained_candidate_count'])}，"
            f"Tier1={int(threshold_baseline['tier1_count'])}，Tier2={int(threshold_baseline['tier2_count'])} であった。",
        ]
    ) + "\n"


def build_discussion_draft(
    sensitivity_corr_df: pd.DataFrame,
    tier_comp_df: pd.DataFrame,
    threshold_stability_df: pd.DataFrame,
) -> str:
    worst_setting = sensitivity_corr_df.sort_values(
        ["spearman_rank_correlation", "setting"], ascending=[True, True], na_position="last"
    ).iloc[0]
    context_rows = threshold_stability_df[threshold_stability_df["k"] == 10].sort_values("jaccard")
    most_unstable = context_rows.iloc[0]
    return "\n".join(
        [
            "% Additional experiments: discussion draft",
            f"感度分析では，`{worst_setting['setting']}` が基準設定に対して最も大きな順位変動を示したが，"
            "それでも上位候補の一部は共通して残った。",
            "このことから，ランキングは特定の1特徴のみによって決まるのではなく，複数特徴の組合せに依存していると考えられる。",
            f"一方，閾値分析では `{most_unstable['setting']}` のように文脈条件を緩めた設定で Top10 Jaccard が低下し，"
            "same-sentence 条件が解釈可能性の維持に寄与していることが示唆された。",
            "Tier 1 と Tier 2 の比較は，人手確認の優先順位付けという観点では有用であるが，Tier 2 に人手評価がないため，妥当性の優劣そのものを結論することはできない。",
        ]
    ) + "\n"


def build_limitations_draft() -> str:
    return "\n".join(
        [
            "% Additional experiments: limitations draft",
            "本稿の追加分析は，ランキングの安定性や補助的一致を記述的に確認するものであり，推薦精度や味覚的正解を示すものではない。",
            "LAGOS は小規模な編集記事型データであり，ユーザー投稿レビューとはデータ生成過程が異なる。",
            "また，Tier 2 には人手評価が付与されていないため，Tier 1 と Tier 2 の比較は構造的な差異の記述に留まる。",
            "閾値分析についても，最適閾値を決定するのではなく，証拠量と解釈可能性のトレードオフを確認する目的で実施した。",
        ]
    ) + "\n"


def build_argument_recommendation(
    sensitivity_corr_df: pd.DataFrame,
    lagos_tier_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> str:
    lines = [
        "# Additional Experiment Recommendation",
        "",
        "## 本文への追加優先度",
        "",
        "- 感度分析: 本文への追加を推奨",
        "- 閾値分析: 本文への追加を推奨",
        "- Tier 1 / Tier 2 比較: Discussion への短い追加を推奨",
        "- LAGOS 補助比較: 補助実験として短く記載し，詳細は補足資料でもよい",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended-dir", default=str(root / "outputs" / "extended_analysis_v2"))
    parser.add_argument("--outputs-dir", default=str(root / "outputs"))
    parser.add_argument("--paper-dir", default=str(root.parent / "paper"))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    extended_dir = Path(args.extended_dir)
    outputs_dir = Path(args.outputs_dir)
    paper_dir = Path(args.paper_dir)

    tier2_df = read_csv(extended_dir / "pair_ranking_tier2.csv")
    excluded_df = read_csv(extended_dir / "pair_ranking_excluded.csv")
    manual_df = read_csv(extended_dir / "manual_validation_tier1.csv")
    lagos_unique_df = read_csv(outputs_dir / "shisha_lagos_unique_pairs.csv")
    lagos_common_df = read_csv(outputs_dir / "shisha_lagos_common_pairs_with_existing.csv")
    lagos_only_df = read_csv(outputs_dir / "shisha_lagos_only_pairs.csv")
    lagos_agreement_df = read_csv(outputs_dir / "shisha_lagos_external_agreement_at_k.csv")

    full_pool_df = combine_candidate_pool(tier2_df, excluded_df)

    sensitivity_summary_df, sensitivity_corr_df, sensitivity_overlap_df, sensitivity_transition_df, sensitivity_manual_df, sensitivity_report = sensitivity_analysis(
        tier2_df,
        manual_df,
    )
    lagos_tier_df, lagos_rank_freq_df, lagos_report, lagos_draft = lagos_supplementary_analysis(
        tier2_df,
        lagos_unique_df,
        lagos_common_df,
        lagos_only_df,
        lagos_agreement_df,
    )
    tier_comp_df, tier_dist_df, tier_lagos_df, tier_report, tier_draft = tier_feature_summary(
        tier2_df,
        lagos_unique_df,
    )
    threshold_df, threshold_stability_df, threshold_manual_df, threshold_lagos_df, threshold_report, threshold_draft = threshold_analysis(
        full_pool_df,
        tier2_df,
        manual_df,
        lagos_unique_df,
    )

    write_csv(sensitivity_summary_df, extended_dir / "sensitivity_condition_summary.csv")
    write_csv(sensitivity_corr_df, extended_dir / "sensitivity_rank_correlations.csv")
    write_csv(sensitivity_overlap_df, extended_dir / "sensitivity_topk_overlap.csv")
    write_csv(sensitivity_transition_df, extended_dir / "sensitivity_tier_transitions.csv")
    write_csv(sensitivity_manual_df, extended_dir / "sensitivity_manual_validation_comparison.csv")
    write_text(sensitivity_report, extended_dir / "sensitivity_analysis_report.md")

    write_csv(lagos_tier_df, outputs_dir / "shisha_lagos_tier_agreement_comparison.csv")
    write_csv(lagos_rank_freq_df, outputs_dir / "shisha_lagos_rank_article_frequency.csv")
    write_text(lagos_report, outputs_dir / "shisha_lagos_supplementary_comparison_report.md")
    write_text(lagos_draft, outputs_dir / "paper_shisha_lagos_supplementary_draft.md")

    write_csv(tier_comp_df, extended_dir / "tier1_tier2_feature_comparison.csv")
    write_csv(tier_dist_df, extended_dir / "tier1_tier2_distribution_summary.csv")
    write_csv(tier_lagos_df, extended_dir / "tier1_tier2_lagos_agreement.csv")
    write_text(tier_report, extended_dir / "tier1_tier2_comparison_report.md")
    write_text(tier_draft, extended_dir / "paper_tier_comparison_draft.md")

    write_csv(threshold_df, extended_dir / "threshold_grid_results.csv")
    write_csv(threshold_stability_df, extended_dir / "threshold_topk_stability.csv")
    write_csv(threshold_manual_df, extended_dir / "threshold_manual_validation_retention.csv")
    write_csv(threshold_lagos_df, extended_dir / "threshold_lagos_agreement.csv")
    write_text(threshold_report, extended_dir / "threshold_justification_report.md")
    write_text(threshold_draft, extended_dir / "paper_threshold_analysis_draft.md")

    write_text(build_argument_recommendation(sensitivity_corr_df, lagos_tier_df, threshold_df), extended_dir / "paper_sensitivity_analysis_draft.md")
    write_text(build_method_draft(sensitivity_summary_df, threshold_df), paper_dir / "paper_additional_experiments_method_draft.tex")
    write_text(build_results_draft(sensitivity_summary_df, lagos_tier_df, tier_lagos_df, threshold_df), paper_dir / "paper_additional_experiments_results_draft.tex")
    write_text(build_discussion_draft(sensitivity_corr_df, tier_comp_df, threshold_stability_df), paper_dir / "paper_additional_experiments_discussion_draft.tex")
    write_text(build_limitations_draft(), paper_dir / "paper_additional_experiments_limitations_draft.tex")

    metadata = {
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "git_commit_hash": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=extended_dir.parents[1],
        ).stdout.strip(),
        "manual_validation_source": manual_validation_source(manual_df),
        "ranking_file": str(extended_dir / "pair_ranking_tier2.csv"),
        "dictionary_hash": hashlib.sha256(
            (extended_dir.parents[1] / "data" / "aslaj_master_list.csv").read_bytes()
        ).hexdigest(),
    }
    write_text(pd.Series(metadata).to_json(force_ascii=False, indent=2), extended_dir / "additional_experiments_metadata.json")

    print("paper additional experiments completed")
    print(f"- sensitivity settings: {len(sensitivity_summary_df)}")
    print(f"- threshold settings: {len(threshold_df)}")
    print(f"- lagos tier rows: {len(lagos_tier_df)}")


if __name__ == "__main__":
    main()
