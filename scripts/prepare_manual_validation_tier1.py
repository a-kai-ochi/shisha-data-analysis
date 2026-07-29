#!/usr/bin/env python3
"""Prepare Tier 1 manual validation data and guideline without changing rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extended_analysis_utils import (
    build_manual_validation_tier1_dataframe,
    output_paths,
    write_manual_validation_guideline,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "extended_analysis_v2"),
        help="extended_analysis_v2 の出力ディレクトリ",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    paths = output_paths(output_dir)

    ranking_path = paths.pair_ranking_tier1_csv if paths.pair_ranking_tier1_csv.exists() else paths.pair_ranking_csv
    evidence_path = paths.pair_expression_evidence_csv
    if not ranking_path.exists():
        raise FileNotFoundError(f"Tier 1 ranking CSV not found: {ranking_path}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Pair evidence CSV not found: {evidence_path}")

    ranking_df = pd.read_csv(ranking_path)
    evidence_df = pd.read_csv(evidence_path)
    manual_df = build_manual_validation_tier1_dataframe(ranking_df, evidence_df)
    manual_df.to_csv(paths.manual_validation_tier1_csv, index=False, encoding="utf-8-sig")
    write_manual_validation_guideline(paths.manual_validation_guideline_md)

    context_counts = (
        manual_df[["context_1", "context_2", "context_3"]]
        .fillna("")
        .astype(str)
        .ne("")
        .sum(axis=1)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("manual validation Tier 1 data prepared")
    print(f"- output_dir: {output_dir}")
    print(f"- manual_validation_tier1_csv: {paths.manual_validation_tier1_csv}")
    print(f"- manual_validation_guideline_md: {paths.manual_validation_guideline_md}")
    print(f"- tier1_candidate_count: {len(manual_df)}")
    print(f"- context_count_distribution: {context_counts}")


if __name__ == "__main__":
    main()
