#!/usr/bin/env python3
"""Extract recommended mix pairs from saved Shisha LAGOS tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from shisha_lagos_external_validation import (
    build_dictionary_candidate_audit,
    build_dictionary_candidate_manual_review,
    build_extraction_summary,
    build_pair_repetition_audit,
    build_unique_pairs,
    extract_mix_pairs,
    load_flavor_dictionary,
    read_csv,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", default=str(root / "data" / "processed" / "shisha_lagos_articles.csv"))
    parser.add_argument("--tables", default=str(root / "data" / "processed" / "shisha_lagos_tables.csv"))
    parser.add_argument("--master-csv", default=str(root / "data" / "aslaj_master_list.csv"))
    parser.add_argument("--output-pairs", default=str(root / "data" / "processed" / "shisha_lagos_recommended_mix_pairs.csv"))
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    articles_df = read_csv(Path(args.articles))
    tables_df = read_csv(Path(args.tables))
    master_df = read_csv(Path(args.master_csv))
    flavor_dict = load_flavor_dictionary(Path(args.master_csv))

    extracted_df = extract_mix_pairs(articles_df, tables_df, flavor_dict)
    summary_df = build_extraction_summary(articles_df, tables_df, extracted_df)
    unique_pairs_df = build_unique_pairs(extracted_df)
    candidate_audit_df = build_dictionary_candidate_audit(extracted_df, flavor_dict)
    repetition_audit_df, repetition_summary_df = build_pair_repetition_audit(extracted_df)
    manual_review_df = build_dictionary_candidate_manual_review(extracted_df, flavor_dict, master_df)

    output_dir = Path(args.output_dir)
    write_csv(extracted_df, Path(args.output_pairs))
    write_csv(summary_df, output_dir / "shisha_lagos_mix_extraction_summary.csv")
    write_csv(extracted_df, output_dir / "shisha_lagos_mix_extraction_audit.csv")
    write_csv(candidate_audit_df, output_dir / "shisha_lagos_dictionary_candidate_audit.csv")
    write_csv(manual_review_df, output_dir / "shisha_lagos_dictionary_candidate_manual_review.csv")
    write_csv(unique_pairs_df, output_dir / "shisha_lagos_unique_pairs.csv")
    write_csv(repetition_audit_df, output_dir / "shisha_lagos_pair_repetition_audit.csv")
    write_csv(repetition_summary_df, output_dir / "shisha_lagos_pair_repetition_summary.csv")

    print("shisha lagos recommended mix extraction completed")
    print(f"- extracted_row_count: {len(extracted_df)}")
    print(f"- valid_row_count: {int(extracted_df['is_valid_pair'].sum()) if 'is_valid_pair' in extracted_df.columns else 0}")
    print(f"- unique_pair_count: {int(unique_pairs_df.shape[0])}")
    print(f"- dictionary_candidate_count: {int(candidate_audit_df.shape[0])}")


if __name__ == "__main__":
    main()
