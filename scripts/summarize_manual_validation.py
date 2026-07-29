#!/usr/bin/env python3
"""Summarize manual validation results for extended flavor-pair analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from extended_analysis_utils import (
    compute_manual_validation_summary,
    create_manual_validity_plot,
    manual_validation_has_labels,
    output_paths,
    write_manual_validation_summary_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(root / "outputs" / "extended_analysis" / "manual_validation_candidates.csv"),
        help="人手評価候補CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "extended_analysis"),
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
        message = "未評価のため集計できない。manual_validation_candidates.csv が見つかりません。"
        print(message)
        paths.manual_validation_summary_csv.write_text("", encoding="utf-8")
        write_manual_validation_summary_markdown(pd.DataFrame(), paths.manual_validation_summary_md, has_labels=False)
        return

    manual_df = pd.read_csv(input_csv)
    has_labels = manual_validation_has_labels(manual_df)
    if not has_labels:
        message = "未評価のため集計できない。評価ラベル列が未入力です。"
        print(message)
        pd.DataFrame().to_csv(paths.manual_validation_summary_csv, index=False, encoding="utf-8-sig")
        write_manual_validation_summary_markdown(pd.DataFrame(), paths.manual_validation_summary_md, has_labels=False)
        return

    summary_df = compute_manual_validation_summary(manual_df, k_values=[5, 10, 20])
    summary_df.to_csv(paths.manual_validation_summary_csv, index=False, encoding="utf-8-sig")
    write_manual_validation_summary_markdown(summary_df, paths.manual_validation_summary_md, has_labels=True)
    create_manual_validity_plot(summary_df, paths.figure_manual_validity_png)
    print("manual validation summary completed")
    print(f"- input: {input_csv}")
    print(f"- output_dir: {output_dir}")


if __name__ == "__main__":
    main()
