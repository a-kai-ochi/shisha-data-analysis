# shisha-data-analysis

## 研究目的
本ディレクトリでは、シーシャのWebレビュー・投稿データを用いて、レビュー本文中で一緒に言及されるフレーバー関係を分析し、ミックス提案支援に利用できる知識を抽出する。現時点の分析粒度は「銘柄間」ではなく、「正規化されたフレーバー間」に統一している。

## データの出典
- 販売データ: [ASLAJ](https://www.aslaj.com/)
- レビューデータ: [CLOUD](https://cloud-jp.net/)
- 追加収集対象: [Shisha Cafe & Bar LAGOS](https://shisha-lagos.com/)

## 入力データ
主要な入力ファイルは以下の2つである。

- `data/aslaj_master_list.csv`
  - 主な列: `ブランド`, `フレーバー名`, `価格`, `詳細URL`
- `data/cloud_reviews_final.csv`
  - 主な列: `レビュータイトル`, `更新日`, `概要`, `レビューURL`, `レビュー本文`

追加実験で利用する本文列は主に `レビュー本文` であり、数値の評価値列は現時点では含まれていない。

## LAGOSデータの位置づけ
LAGOSサイトの「フレーバーレビュー」カテゴリは、既存の `CLOUD` レビューとは性質が異なる可能性がある。そのため、本ディレクトリでは以下のように区別して扱う。

- `CLOUD`
  - 既存のユーザー投稿・レビュー系コーパスとして扱う
- `Shisha Cafe & Bar LAGOS`
  - `editorial_review` として扱う
  - 既存レビューコーパスへ自動統合せず、単独の出力として保存する

論文や分析で利用する際は、取得日、対象件数、`source_type` を明記し、既存レビューと混在させる場合も `source_type` 別に集計する。

## 既存実験と追加実験
既存実験では、以下を実装している。

- フレーバー名の名称正規化
- レビュー本文からのフレーバー抽出
- 共起ネットワーク構築
- Support / Lift の計算
- 人手確認用データの作成

修正版の追加実験では、これに加えて以下を実施する。

- 媒介中心性を用いた接続性評価
- 文書単位の共起に基づく Support / Lift と、同一文文脈に基づく評価特徴の分離
- Support / Lift / 媒介中心性 / 文脈特徴を統合した総合ランキング
- 評価表現・味覚表現・体験表現の辞書ベース抽出
- 単一商品名由来ペア、親子語・部分一致ペア、テンプレート文の除外
- 味の役割説明の抽出
- 人手評価候補の拡張
- ランキング感度分析

## 総合スコア
修正版の標準スコアは次式で定義している。

```text
overall_score_v2 =
    0.30 * normalized_support
  + 0.25 * adjusted_lift
  + 0.15 * normalized_centrality_mean
  + 0.15 * normalized_smoothed_positive_ratio
  + 0.10 * normalized_smoothed_role_ratio
  - 0.05 * normalized_smoothed_negative_ratio
```

正の重みと負のペナルティは研究上確定したものではなく、暫定設定である。感度分析では別設定も比較する。

`adjusted_lift` は、文書単位 Lift に出現回数による信頼度補正をかけた値である。文脈比率には縮約を適用し、`pair_count < 3` または `same_sentence_evidence_document_count < 2` の候補は文脈加点を得ない。

## 評価語抽出
評価語抽出は LLM や外部APIを使わず、`config/taste_expression_dictionary.json` に定義した辞書ベースで実装している。味覚表現、体験・吸い心地表現、肯定/否定評価、味の役割説明を対象とする。

共起は文書単位で計算するが、評価表現・否定表現・役割表現のスコアには、原則として同一文に両フレーバーが現れる文脈だけを利用する。テンプレート文や見出しは `config/template_sentence_patterns.json` で除外し、単一商品名由来のペアや親子語・部分一致ペアは標準ランキングから除外する。

### 否定表現処理の限界
- `ない`
- `なく`
- `ません`
- `ではない`
- `じゃない`

のような否定表現には最低限対応しているが、複雑な文脈依存の否定や皮肉表現までは扱っていない。

## 実行コマンド
修正版の追加実験は以下で一括実行できる。

```bash
python3 scripts/run_extended_analysis.py \
  --input data/cloud_reviews_final.csv \
  --output-dir outputs/extended_analysis_v2 \
  --top-k 20 \
  --dictionary config/taste_expression_dictionary.json \
  --template-patterns config/template_sentence_patterns.json \
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

## LAGOSスクレイピング
Shisha Cafe & Bar LAGOS の収集は、対象カテゴリを `フレーバーレビュー` に限定して実行する。スクレイピングは管理者の許可を得たうえで行い、`robots.txt` の内容も毎回確認・記録する。LAGOSデータは `editorial_review` として扱い、既存のユーザーレビューとは別ソースのまま保存する。

### dry-run
まずはカテゴリ巡回と記事本文のプレビューだけを行う。

```bash
uv run --with requests --with beautifulsoup4 --with pandas \
python scripts/scrape_shisha_lagos.py \
  --start-url "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/" \
  --output-dir data/processed \
  --raw-dir data/raw/shisha_lagos \
  --report-dir outputs \
  --delay 2.0 \
  --max-pages 2 \
  --dry-run
```

dry-run では以下のみを行う。

- カテゴリページの確認
- 記事URLの列挙
- ページネーションの確認
- 本文取得は最大3記事まで
- 恒久的なCSVやraw HTMLは出力しない

### 本収集
dry-run で問題がなければ、以下のように本収集を行う。

```bash
uv run --with requests --with beautifulsoup4 --with pandas \
python scripts/scrape_shisha_lagos.py \
  --start-url "https://shisha-lagos.com/category/%E3%83%95%E3%83%AC%E3%83%BC%E3%83%90%E3%83%BC%E3%83%AC%E3%83%93%E3%83%A5%E3%83%BC/" \
  --output-dir data/processed \
  --raw-dir data/raw/shisha_lagos \
  --report-dir outputs \
  --delay 2.0 \
  --max-pages 100 \
  --resume
```

### 2026-07-29 時点の収集結果

- 収集日: `2026-07-29`
- 対象カテゴリページ数: `2`
- 収集記事数: `13`
- source_type: `editorial_review`
- 記事本文抽出成功: `13/13`
- おすすめミックス節検出: `13/13`
- brand 抽出成功: `13/13`
- target_flavor 抽出成功: `13/13`

既存の `CLOUD` レビューとは性質が異なるため、この13件は既存レビューコーパスへ直接追記せず、別ソースのまま保持する。既存分析へ統合する前に、`source_type` 別の検証とバイアス確認を行う。

### 安全設定

- `User-Agent` は研究目的を明示した値を使用する
- 連絡先は環境変数 `SHISHA_SCRAPER_CONTACT` または CLI 引数で指定する
- デフォルトの待機時間は2秒で、ランダムジッターを加える
- 並列アクセスは行わない
- `429` と `5xx` は retry と exponential backoff で処理する
- `robots.txt` を確認し、矛盾がある場合は自動実行を停止する
- `--resume` 指定時は保存済みHTMLを再利用する

### LAGOS出力ファイル

- `data/processed/shisha_lagos_articles.csv`
- `data/processed/shisha_lagos_paragraphs.csv`
- `data/processed/shisha_lagos_tables.csv`
- `data/processed/shisha_lagos_list_items.csv`
- `data/processed/shisha_lagos_recommended_mix_sections.csv`
- `data/processed/shisha_lagos_duplicates.csv`
- `outputs/shisha_lagos_scraping_summary.csv`
- `outputs/shisha_lagos_scraping_report.md`
- `outputs/shisha_lagos_scraping_metadata.json`
- `outputs/shisha_lagos_article_length_outliers.csv`
- `outputs/shisha_lagos_heading_frequency.csv`
- `outputs/shisha_lagos_recommended_heading_variants.csv`
- `outputs/shisha_lagos_missing_recommended_sections.csv`
- `outputs/shisha_lagos_flavor_extraction_audit.csv`
- `outputs/shisha_lagos_metadata_inconsistencies.csv`
- `outputs/shisha_lagos_repeated_paragraphs.csv`
- `outputs/source_schema_comparison.csv`
- `outputs/source_characteristics.md`
- `outputs/shisha_lagos_dictionary_coverage.csv`
- `outputs/shisha_lagos_unmatched_flavor_candidates.csv`
- `outputs/paper_shisha_lagos_dataset_statistics.csv`
- `outputs/paper_shisha_lagos_dataset_draft.md`

### 生HTML
再現性のため、raw HTML は以下に保存できる。

- `data/raw/shisha_lagos/category_pages/`
- `data/raw/shisha_lagos/articles/`

### 論文利用時の注意

- 記事本文を大量転載しない
- 元URLを保持する
- 取得日と対象件数を明記する
- `editorial_review` であることを明示する
- 記事本文とおすすめミックス節は別保存していることを明記する
- 既存ユーザーレビューと直接同一視しない
- 統合前に `source_type` 別の検証が必要である

## 出力ファイル
修正版追加実験の主な出力は `outputs/extended_analysis_v2/` に保存する。

- `flavor_centrality.csv`
  - フレーバーごとの次数、重み付き次数、媒介中心性
- `flavor_centrality_top20.md`
  - 媒介中心性上位20件の表
- `pair_expression_features.csv`
  - フレーバーペアごとの文書単位・同一文単位の文脈特徴量
- `pair_expression_evidence.csv`
  - 抽出根拠となる文。テンプレート文フラグや role 判定根拠も含む
- `pair_ranking.csv`
  - Tier 1 の標準ランキング
- `pair_ranking_tier1.csv`
  - Tier 1 の強い候補
- `pair_ranking_tier2.csv`
  - Tier 2 の探索的候補
- `pair_ranking_excluded.csv`
  - 除外候補と除外理由
- `excluded_product_name_pairs.csv`
  - 単一商品名由来として除外した監査用CSV
- `excluded_parent_child_pairs.csv`
  - 親子語・部分一致として除外した監査用CSV
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
- `ranking_before_after_comparison.csv`
  - 修正前後ランキング比較
- `ranking_before_after_comparison.md`
  - 修正前後ランキング比較の要約
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

## Tierの考え方
- Tier 1
  - `pair_count >= 3`
  - `same_sentence_evidence_document_count >= 2`
  - 商品名由来ペア・親子語ペアを除外
- Tier 2
  - `pair_count >= 2`
  - 商品名由来ペア・親子語ペアを除外
  - 推薦の確定候補ではなく探索的候補として扱う

## 限界と注意
- 辞書ベース抽出であるため、複雑な言い換え・皮肉・長距離依存は扱えない
- 明示的ミックス表現の網羅性には限界がある
- 複合商品名と実ミックスの完全な識別は困難である
- ランキングは推薦候補の探索支援であり、推薦の有効性を保証しない

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
│   └── template_sentence_patterns.json
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
