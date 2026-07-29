#!/usr/bin/env python3
"""Run extended normalized flavor-pair analysis without overwriting existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from extended_analysis_utils import (
    LIMITED_2_5,
    add_normalized_features,
    add_pair_centrality_features,
    apply_condition,
    build_cooccurrence_graph,
    build_manual_validation_candidates,
    build_pair_ranking,
    compute_centrality_dataframe,
    compute_manual_validation_summary,
    compute_pair_statistics,
    compute_sensitivity,
    create_bar_plot,
    create_manual_validity_plot,
    create_ranking_comparison_plot,
    create_scatter_plot,
    create_score_breakdown_plot,
    extract_pair_expression_features,
    load_documents,
    load_expression_dictionary,
    manual_validation_has_labels,
    merge_pair_features,
    output_paths,
    write_centrality_top20_markdown,
    write_extended_summary,
    write_manual_validation_summary_markdown,
    write_sensitivity_markdown,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(description=__doc__).parse_args(namespace=argparse.Namespace(
        input=str(root / "data" / "cloud_reviews_final.csv"),
        output_dir=str(root / "outputs" / "extended_analysis"),
        top_k=20,
        dictionary=str(root / "config" / "taste_expression_dictionary.json"),
        min_pair_count=2,
        random_seed=42,
        overwrite=False,
    ))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(root / "data" / "cloud_reviews_final.csv"),
        help="レビューCSV。既定値は data/cloud_reviews_final.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "extended_analysis"),
        help="追加実験の出力ディレクトリ",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="上位候補数。人手確認候補や感度分析で利用",
    )
    parser.add_argument(
        "--dictionary",
        default=str(root / "config" / "taste_expression_dictionary.json"),
        help="評価・味覚・体験表現辞書JSON",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=2,
        help="ランキング対象ペアの最小共起回数",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="将来の乱数処理用。現状は決定的処理のみだが記録用に保持",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存出力ディレクトリを上書きする",
    )
    return parser


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{path} は既に存在します。既存出力を保護するため上書きしません。"
            " 上書きする場合は --overwrite を指定してください。"
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    master_csv = root / "data" / "aslaj_master_list.csv"
    reviews_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    paths = output_paths(output_dir)

    docs_df, _flavor_dict, pattern_to_canonical, sorted_patterns = load_documents(
        reviews_csv=reviews_csv,
        master_csv=master_csv,
    )
    conditioned_df = apply_condition(docs_df, LIMITED_2_5)
    expression_dictionary = load_expression_dictionary(Path(args.dictionary))

    pair_df, _flavor_frequency, _pair_counts = compute_pair_statistics(conditioned_df)
    graph = build_cooccurrence_graph(pair_df)
    centrality_df = compute_centrality_dataframe(graph)
    centrality_df.to_csv(paths.flavor_centrality_csv, index=False, encoding="utf-8-sig")
    write_centrality_top20_markdown(centrality_df, paths.flavor_centrality_top20_md)

    pair_with_centrality_df = add_pair_centrality_features(pair_df, centrality_df)
    filtered_pair_df = pair_with_centrality_df[pair_with_centrality_df["pair_count"] >= args.min_pair_count].copy()

    expression_df, evidence_df = extract_pair_expression_features(
        docs_df=conditioned_df,
        pair_df=filtered_pair_df,
        sorted_patterns=sorted_patterns,
        pattern_to_canonical=pattern_to_canonical,
        expression_dictionary=expression_dictionary,
    )
    expression_df.to_csv(paths.pair_expression_features_csv, index=False, encoding="utf-8-sig")
    evidence_df.to_csv(paths.pair_expression_evidence_csv, index=False, encoding="utf-8-sig")

    merged_df = merge_pair_features(filtered_pair_df, filtered_pair_df, expression_df)
    normalized_df = add_normalized_features(merged_df)
    pair_ranking_df = build_pair_ranking(normalized_df)
    pair_ranking_df.to_csv(paths.pair_ranking_csv, index=False, encoding="utf-8-sig")

    manual_candidates_df = build_manual_validation_candidates(
        pair_ranking_df=pair_ranking_df,
        evidence_df=evidence_df,
        top_k=args.top_k,
    )
    manual_candidates_df.to_csv(
        paths.manual_validation_candidates_csv, index=False, encoding="utf-8-sig"
    )

    sensitivity_detail_df, sensitivity_summary_df = compute_sensitivity(
        pair_ranking_df=pair_ranking_df,
        top_k=args.top_k,
    )
    sensitivity_detail_df.to_csv(
        paths.ranking_sensitivity_csv, index=False, encoding="utf-8-sig"
    )
    write_sensitivity_markdown(sensitivity_summary_df, paths.ranking_sensitivity_md)

    create_bar_plot(
        centrality_df,
        x="weighted_betweenness_centrality",
        y="flavor",
        title="媒介中心性上位20フレーバー",
        xlabel="重み付き媒介中心性",
        ylabel="フレーバー",
        path=paths.figure_centrality_png,
        top_k=20,
    )
    create_score_breakdown_plot(pair_ranking_df, paths.figure_score_breakdown_png, top_k=args.top_k)
    create_scatter_plot(
        pair_ranking_df,
        x="support",
        y="lift",
        title="SupportとLiftの散布図",
        xlabel="Support",
        ylabel="Lift",
        path=paths.figure_support_lift_scatter_png,
    )
    create_scatter_plot(
        pair_ranking_df,
        x="support",
        y="overall_score",
        title="Supportと総合スコアの散布図",
        xlabel="Support",
        ylabel="Overall Score",
        path=paths.figure_support_overall_scatter_png,
    )
    create_scatter_plot(
        pair_ranking_df,
        x="lift",
        y="overall_score",
        title="Liftと総合スコアの散布図",
        xlabel="Lift",
        ylabel="Overall Score",
        path=paths.figure_lift_overall_scatter_png,
    )
    create_ranking_comparison_plot(pair_ranking_df, paths.figure_ranking_comparison_png, top_k=args.top_k)

    summary_df = compute_manual_validation_summary(manual_candidates_df, k_values=[5, 10, 20])
    summary_df.to_csv(paths.manual_validation_summary_csv, index=False, encoding="utf-8-sig")
    has_labels = manual_validation_has_labels(manual_candidates_df)
    write_manual_validation_summary_markdown(
        summary_df,
        paths.manual_validation_summary_md,
        has_labels=has_labels,
    )
    if has_labels:
        create_manual_validity_plot(summary_df, paths.figure_manual_validity_png)

    write_extended_summary(paths.extended_summary_md, docs_df, conditioned_df, pair_ranking_df)

    print("extended analysis completed")
    print(f"- input reviews: {reviews_csv}")
    print(f"- master: {master_csv}")
    print(f"- condition: {LIMITED_2_5.name}")
    print(f"- output_dir: {output_dir}")
    print(f"- pair_count_for_ranking: {len(pair_ranking_df)}")
    print("- overall_score weights are initial settings only")


if __name__ == "__main__":
    main()
