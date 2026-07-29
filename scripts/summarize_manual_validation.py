#!/usr/bin/env python3
"""Summarize manual validation results for extended flavor-pair analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extended_analysis_utils import (
    compute_manual_validation_outputs,
    output_paths,
    write_manual_validation_summary_markdown_v2,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(root / "outputs" / "extended_analysis_v2" / "manual_validation_tier1.csv"),
        help="人手評価候補CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "extended_analysis_v2"),
        help="集計出力ディレクトリ",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)

    if not input_csv.exists():
        message = "未評価のため集計できない。manual_validation_tier1.csv が見つかりません。"
        print(message)
        paths.manual_validation_summary_csv.write_text("", encoding="utf-8")
        pd.DataFrame().to_csv(paths.manual_validation_crosstab_csv, index=False, encoding="utf-8-sig")
        pd.DataFrame().to_csv(paths.manual_validation_disagreements_csv, index=False, encoding="utf-8-sig")
        write_manual_validation_summary_markdown_v2(
            pd.DataFrame(),
            pd.DataFrame(),
            paths.manual_validation_summary_md,
            primary_source=None,
        )
        return

    manual_df = pd.read_csv(input_csv)
    summary_df, crosstab_df, agreement_df, disagreements_df, primary_source = compute_manual_validation_outputs(
        manual_df,
        k_values=[5, 10, 17],
    )
    summary_df.to_csv(paths.manual_validation_summary_csv, index=False, encoding="utf-8-sig")
    crosstab_df.to_csv(paths.manual_validation_crosstab_csv, index=False, encoding="utf-8-sig")
    disagreements_df.to_csv(paths.manual_validation_disagreements_csv, index=False, encoding="utf-8-sig")
    write_manual_validation_summary_markdown_v2(
        summary_df,
        agreement_df,
        paths.manual_validation_summary_md,
        primary_source=primary_source,
    )
    print("manual validation summary completed")
    print(f"- input: {input_csv}")
    print(f"- output_dir: {output_dir}")
    if primary_source is None:
        print("- primary_label_source: none")
    else:
        print(f"- primary_label_source: {primary_source}")


if __name__ == "__main__":
    main()
