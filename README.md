# shisha-data-analysis

## 研究目的
本ディレクトリでは、シーシャのWebレビュー・投稿データを用いて、レビュー本文中で一緒に言及されるフレーバー関係を分析し、ミックス提案支援に利用できる知識を抽出する。現時点の分析粒度は「銘柄間」ではなく、「正規化されたフレーバー間」に統一している。

## データの出典
- 販売データ: [ASLAJ](https://www.aslaj.com/)
- レビューデータ: [CLOUD](https://cloud-jp.net/)

## 入力データ
主要な入力ファイルは以下の2つである。

- `data/aslaj_master_list.csv`
  - 主な列: `ブランド`, `フレーバー名`, `価格`, `詳細URL`
- `data/cloud_reviews_final.csv`
  - 主な列: `レビュータイトル`, `更新日`, `概要`, `レビューURL`, `レビュー本文`

追加実験で利用する本文列は主に `レビュー本文` であり、数値の評価値列は現時点では含まれていない。

## 既存実験と追加実験
既存実験では、以下を実装している。

- フレーバー名の名称正規化
- レビュー本文からのフレーバー抽出
- 共起ネットワーク構築
- Support / Lift の計算
- 人手確認用データの作成

追加実験では、これに加えて以下を実施する。

- 媒介中心性を用いた接続性評価
- Support / Lift / 媒介中心性 / 文脈特徴を統合した総合ランキング
- 評価表現・味覚表現・体験表現の辞書ベース抽出
- 味の役割説明の抽出
- 人手評価候補の拡張
- ランキング感度分析

## 総合スコア
標準の総合スコアは次式で定義している。

```text
overall_score =
    0.30 * normalized_support
  + 0.25 * normalized_lift
  + 0.20 * normalized_centrality_mean
  + 0.15 * normalized_positive_document_ratio
  + 0.10 * normalized_taste_role_explanation_ratio
```

この重みは研究上確定したものではなく、初期設定である。感度分析では別設定も比較する。

## 評価語抽出
評価語抽出は LLM や外部APIを使わず、`config/taste_expression_dictionary.json` に定義した辞書ベースで実装している。味覚表現、体験・吸い心地表現、肯定/否定評価、味の役割説明を対象とする。

### 否定表現処理の限界
- `ない`
- `なく`
- `ません`
- `ではない`
- `じゃない`

のような否定表現には最低限対応しているが、複雑な文脈依存の否定や皮肉表現までは扱っていない。

## 実行コマンド
追加実験は以下で一括実行できる。

```bash
python3 scripts/run_extended_analysis.py \
  --input data/cloud_reviews_final.csv \
  --output-dir outputs/extended_analysis \
  --top-k 20 \
  --dictionary config/taste_expression_dictionary.json \
  --min-pair-count 2 \
  --random-seed 42
```

既存出力を保護するため、既に出力先ディレクトリが存在する場合は、`--overwrite` を指定しない限り上書きしない。

人手評価集計は以下で実行する。

```bash
python3 scripts/summarize_manual_validation.py \
  --input outputs/extended_analysis/manual_validation_candidates.csv \
  --output-dir outputs/extended_analysis
```

## 出力ファイル
追加実験の主な出力は `outputs/extended_analysis/` に保存する。

- `flavor_centrality.csv`
  - フレーバーごとの次数、重み付き次数、媒介中心性
- `flavor_centrality_top20.md`
  - 媒介中心性上位20件の表
- `pair_expression_features.csv`
  - フレーバーペアごとの文脈特徴量
- `pair_expression_evidence.csv`
  - 抽出根拠となる文
- `pair_ranking.csv`
  - Support / Lift / 総合ランキング
- `manual_validation_candidates.csv`
  - 人手評価用候補CSV
- `manual_validation_summary.csv`
  - 人手評価結果の集計
- `manual_validation_summary.md`
  - 人手評価結果の要約
- `ranking_sensitivity.csv`
  - 重み設定別ランキング
- `ranking_sensitivity.md`
  - 感度分析の要約
- `figure_*.png`
  - 可視化図

## 人手評価の手順
`manual_validation_candidates.csv` に対して、少なくとも以下の列を入力する。

- `mix_relation_label`
  - `explicit_mix`, `co_mention_only`, `unclear`
- `evaluation_label`
  - `positive`, `neutral`, `negative`, `unclear`
- `taste_role_label`
  - `explained`, `not_explained`, `unclear`
- `recommendation_validity`
  - `valid`, `partially_valid`, `invalid`, `unclear`
- `reviewer_comment`

入力後に `scripts/summarize_manual_validation.py` を実行すると、ランキング方式ごとの妥当率などを集計できる。

## 現時点の対象外
- 銘柄粒度の分析
- 店舗の提案品質改善そのものの直接検証
- 作業時間短縮そのものの直接検証

今回の実験では、レビュー内共起と文脈表現から推薦候補の特徴を評価するまでを対象としており、店舗運用上の効果は直接検証していない。

## ディレクトリ構成

```text
├── README.md
├── requirements.txt
├── config/
│   └── taste_expression_dictionary.json
├── data/
│   ├── aslaj_master_list.csv
│   ├── aslaj_test_with_desc.csv
│   └── cloud_reviews_final.csv
├── notebooks/
│   ├── flavor_mix_network.py
│   ├── flavor_insights.py
│   └── output/
├── poster_analysis/
│   ├── *.csv
│   ├── *.png
│   └── *.pdf
├── scripts/
│   ├── generate_condition_comparison.py
│   ├── summarize_manual_review_check.py
│   ├── extended_analysis_utils.py
│   ├── run_extended_analysis.py
│   └── summarize_manual_validation.py
└── tests/
    └── test_extended_analysis.py
```
