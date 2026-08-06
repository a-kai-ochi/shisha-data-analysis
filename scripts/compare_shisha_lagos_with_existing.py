#!/usr/bin/env python3
"""Compare structured Shisha LAGOS recommended pairs with existing rankings."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

from shisha_lagos_external_validation import (
    BASELINE_TOP_KS,
    build_common_pairs_with_existing,
    build_baseline_markdown,
    build_baseline_records,
    build_dictionary_update_impact_report,
    build_dictionary_update_impact_simulation,
    build_existing_topk_not_in_lagos,
    build_external_validation_report,
    build_external_validation_statistics,
    build_lagos_only_pairs,
    build_paper_validation_draft,
    build_ranking_metadata_summary,
    build_topk_pair_audit,
    compute_at_k_metrics,
    prepare_existing_ranking,
    read_csv,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lagos-pairs",
        default=str(root / "data" / "processed" / "shisha_lagos_recommended_mix_pairs.csv"),
    )
    parser.add_argument(
        "--unique-pairs",
        default=str(root / "outputs" / "shisha_lagos_unique_pairs.csv"),
    )
    parser.add_argument(
        "--ranking",
        default=str(root / "outputs" / "extended_analysis_v2" / "pair_ranking_tier2.csv"),
    )
    parser.add_argument(
        "--pair-features",
        default=str(root / "outputs" / "extended_analysis_v2" / "pair_expression_features.csv"),
    )
    parser.add_argument(
        "--excluded-ranking",
        default=str(root / "outputs" / "extended_analysis_v2" / "pair_ranking_excluded.csv"),
    )
    parser.add_argument(
        "--dictionary-file",
        default=str(root / "data" / "aslaj_master_list.csv"),
    )
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    extracted_df = read_csv(Path(args.lagos_pairs))
    unique_pairs_df = read_csv(Path(args.unique_pairs))
    ranking_df = prepare_existing_ranking(read_csv(Path(args.ranking)))
    pair_features_df = read_csv(Path(args.pair_features))
    excluded_df = read_csv(Path(args.excluded_ranking))

    lagos_pair_set = set(unique_pairs_df["mix_pair_key"].astype(str).tolist())
    agreement_df = compute_at_k_metrics(ranking_df, lagos_pair_set, BASELINE_TOP_KS)
    common_df = build_common_pairs_with_existing(unique_pairs_df, ranking_df)
    lagos_only_df = build_lagos_only_pairs(unique_pairs_df, ranking_df, pair_features_df, excluded_df)
    top50_missing_df = build_existing_topk_not_in_lagos(ranking_df, lagos_pair_set, top_k=50)
    stats_df = build_external_validation_statistics(extracted_df, unique_pairs_df, ranking_df, agreement_df)
    ranking_metadata_df = build_ranking_metadata_summary(
        ranking_df,
        pair_features_df,
        excluded_df,
        args.ranking,
    )
    topk_audit_df = build_topk_pair_audit(ranking_df, unique_pairs_df, BASELINE_TOP_KS)
    simulation_df = build_dictionary_update_impact_simulation(extracted_df, ranking_df)
    simulation_report = build_dictionary_update_impact_report(simulation_df)
    dictionary_hash = hashlib.sha256(Path(args.dictionary_file).read_bytes()).hexdigest()
    git_commit_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    baseline_df = build_baseline_records(
        extracted_df=extracted_df,
        unique_pairs_df=unique_pairs_df,
        ranking_df=ranking_df,
        agreement_df=agreement_df,
        ranking_path=args.ranking,
        ranking_columns=[
            "rank_overall",
            "flavor_a",
            "flavor_b",
            "pair_key",
            "pair_count",
            "support",
            "lift",
            "adjusted_lift",
            "centrality_mean",
            "smoothed_positive_ratio",
            "smoothed_negative_ratio",
            "smoothed_role_ratio",
            "overall_score_v2",
            "ranking_tier",
        ],
        dictionary_hash=dictionary_hash,
        git_commit_hash=git_commit_hash,
    )
    baseline_md = build_baseline_markdown(baseline_df)
    report_text = build_external_validation_report(
        extracted_df=extracted_df,
        unique_pairs_df=unique_pairs_df,
        ranking_df=ranking_df,
        agreement_df=agreement_df,
        common_df=common_df,
        lagos_only_df=lagos_only_df,
    )
    draft_text = build_paper_validation_draft(stats_df, agreement_df)

    write_csv(common_df, output_dir / "shisha_lagos_common_pairs_with_existing.csv")
    write_csv(lagos_only_df, output_dir / "shisha_lagos_only_pairs.csv")
    write_csv(top50_missing_df, output_dir / "existing_top50_not_in_shisha_lagos.csv")
    write_csv(agreement_df, output_dir / "shisha_lagos_external_agreement_at_k.csv")
    write_csv(stats_df, output_dir / "paper_shisha_lagos_external_validation_statistics.csv")
    write_csv(ranking_metadata_df, output_dir / "shisha_lagos_ranking_target_summary.csv")
    write_csv(topk_audit_df, output_dir / "shisha_lagos_existing_topk_pair_audit.csv")
    write_csv(simulation_df, output_dir / "shisha_lagos_dictionary_update_impact_simulation.csv")
    write_csv(baseline_df, output_dir / "paper_shisha_lagos_external_validation_baseline.csv")
    (output_dir / "shisha_lagos_external_validation_report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "paper_shisha_lagos_external_validation_draft.md").write_text(draft_text, encoding="utf-8")
    (output_dir / "shisha_lagos_dictionary_update_impact_report.md").write_text(simulation_report, encoding="utf-8")
    (output_dir / "paper_shisha_lagos_external_validation_baseline.md").write_text(baseline_md, encoding="utf-8")

    print("shisha lagos external comparison completed")
    print(f"- common_pair_count: {len(common_df)}")
    print(f"- lagos_only_pair_count: {len(lagos_only_df)}")
    print(f"- top50_missing_count: {len(top50_missing_df)}")


if __name__ == "__main__":
    main()
