# poster_analysis summary

## 1. 使用データ
- レビューCSV: `data/cloud_reviews_final.csv`
- フレーバーマスタ: `data/aslaj_master_list.csv`
- レビュー本文列: `レビュー本文`
- タイトル列: `レビュータイトル`
- URL列: `レビューURL`
- 日付列: `更新日`
- 分析対象レビュー総数: 222
- 正規化前ユニークフレーバー総数: 144
- 正規化後ユニークフレーバー総数: 136

## 2. 既存実装の確認結果
### 結果
- 使用レビューCSV: `data/cloud_reviews_final.csv`
- 使用フレーバーマスタ: `data/aslaj_master_list.csv`
- フレーバー正規化処理: aslaj_master_list.csv をホワイトリスト辞書として用い、括弧内日本語表記と英語表記を canonical 化して貪欲最長マッチで抽出。
- 共起の定義: 同一レビュー内で抽出されたユニークフレーバー集合から 2 組を数える。1レビュー内の同一フレーバー重複は 1 回扱い。
- Liftの計算式: lift(A,B) = pair_count(A,B) * N / (frequency(A) * frequency(B))
- 既存の除外条件: 既存スクリプトでは主に登場件数や共起回数で可視化時の足切りを行う。3〜8種レビューに絞るレシピ特化分析も存在する。
- 既存の出力図とCSV:
  - `notebooks/output/brand_flavor_synergy_network.png`
  - `notebooks/output/brand_mix_insight.txt`
  - `notebooks/output/final_experiment_summary.md`
  - `notebooks/output/flavor_insights.csv`
  - `notebooks/output/flavor_mix_network.png`
  - `notebooks/output/keyword_trend.png`
  - `notebooks/output/multi_flavor_mix_ranking.txt`
  - `notebooks/output/network_2021_2022.png`
  - `notebooks/output/network_2025_2026.png`
  - `notebooks/output/network_comparison.png`
  - `notebooks/output/slide_table_association.md`
  - `notebooks/output/wordfreq_comparison.png`
### 考察
- 既存コードは全体ランキングやネットワーク可視化には到達しているが、条件比較と代表レビュー確認を横断的に出す仕組みは無かった。
- 抽出ロジックはホワイトリストベースで一貫していたため、今回の条件比較も同じ辞書を用いている。

## 3. フレーバー名称正規化
### 結果
- 自動統合候補数: 16
- manual_review=true の候補数: 4
- 実際に canonical 変更された raw flavor 数: 8
| raw_flavor | canonical_flavor | normalization_rule |
| --- | --- | --- |
| EARL GREY | アールグレイ | verified_cross_language_alias |
| KIWI | キウイ | verified_cross_language_alias |
| GRAPE | グレープ | verified_cross_language_alias |
| GRAPEFRUIT | グレープフルーツ | verified_cross_language_alias |
| COLA | コーラ | verified_cross_language_alias |
| CHOCOLATE | チョコレート | verified_cross_language_alias |
| MANGO | マンゴー | verified_cross_language_alias |
| LYCHEE | ライチ | verified_cross_language_alias |
### 考察
- 自動統合は NFKC・記号差・マスタ/レビューで確認できる EN/JA 対応・明示的に検証した基本訳語だけに限定した。
- 単数複数差やカタカナ揺れなど、誤統合の余地がある候補は manual_review=true とし、自動統合から除外した。

## 4. 条件A・B・Cの定義
- `all_multi`: 抽出フレーバー数が2種類以上の全レビュー
- `limited_2_5`: 抽出フレーバー数が2〜5種類のレビュー
- `mix_keyword_2_5`: 抽出フレーバー数が2〜5種類で、かつ mix keyword を含むレビュー

## 5. 条件別基礎統計
### 結果
| condition | review_count_raw | review_count_normalized | unique_flavor_count_raw | unique_flavor_count_normalized | unique_pair_count_raw | unique_pair_count_normalized | average_flavor_count_raw | average_flavor_count_normalized |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_multi | 177 | 175 | 139 | 132 | 1205 | 1136 | 4.6271 | 4.5829 |
| limited_2_5 | 122 | 121 | 100 | 95 | 432 | 417 | 3.3525 | 3.3223 |
| mix_keyword_2_5 | 114 | 113 | 99 | 94 | 416 | 403 | 3.3772 | 3.354 |
### 考察
- 条件Aと条件Bの差は、2〜5種類に絞ることで多数列挙レビューの影響をどこまで抑えられるかを見るための主比較とした。
- 条件Cは mix keyword に依存するため、本文表現に偏りが出る補助分析として扱う。

## 6. 共起頻度の比較
### 結果
- 正規化前の条件A/B Top10 共通数: 3
- 正規化後の条件A/B Top10 共通数: 3
- 正規化前の条件A/B Jaccard係数: 0.1765
- 正規化後の条件A/B Jaccard係数: 0.1765
- 正規化前の条件A/B Spearman順位相関: -1.0000
- 正規化後の条件A/B Spearman順位相関: -0.5000
- 条件B 共起Top10 の正規化前後共通数: 9
- 条件B 共起Top10 の正規化前後 Jaccard係数: 0.8182
- 正規化前の条件B 共起Top10:
| rank | pair | cooccurrence_count |
| --- | --- | --- |
| 1 | GRAPE × ミント | 6 |
| 2 | アールグレイ × バニラ | 6 |
| 3 | ハニー × ミルク | 6 |
| 4 | バニラ × ミルク | 6 |
| 5 | ミント × レモン | 6 |
| 6 | アールグレイ × コニャック | 5 |
| 7 | コニャック × バニラ | 5 |
| 8 | バナナ × ミルク | 5 |
| 9 | ミルク × メロン | 5 |
| 10 | オレンジ × グレープフルーツ | 4 |
- 正規化後の条件A 共起Top10:
| rank | pair | cooccurrence_count |
| --- | --- | --- |
| 1 | ブルーベリー × ミント | 21 |
| 2 | オレンジ × ミント | 20 |
| 3 | ミント × レモン | 18 |
| 4 | オレンジ × ブルーベリー | 17 |
| 5 | ブルーベリー × ベリー | 15 |
| 6 | バニラ × ミント | 14 |
| 7 | オレンジ × レモン | 11 |
| 8 | グレープ × ミント | 11 |
| 9 | バニラ × ミルク | 11 |
| 10 | ベリー × ミント | 11 |
- 正規化後の条件B 共起Top10:
| rank | pair | cooccurrence_count |
| --- | --- | --- |
| 1 | グレープ × ミント | 7 |
| 2 | アールグレイ × バニラ | 6 |
| 3 | ハニー × ミルク | 6 |
| 4 | バニラ × ミルク | 6 |
| 5 | ミント × レモン | 6 |
| 6 | アールグレイ × コニャック | 5 |
| 7 | コニャック × バニラ | 5 |
| 8 | バナナ × ミルク | 5 |
| 9 | ミルク × メロン | 5 |
| 10 | オレンジ × グレープフルーツ | 4 |
### 考察
- 正規化で疑似ペアが消えると、条件Bの上位は実際のフレーバー共起へ寄りやすくなる。
- 条件Aのみで高順位のペアは、多数列挙型レビューの影響を受けている可能性があるため、代表レビュー確認が重要になる。

## 7. Liftの比較
### 結果
- 正規化前の推奨 min_pair_count: 2
- 正規化前の採用理由: condition B の Lift Top10 を比較した結果、min_pair_count=2 では Top10 の共起回数中央値が 2.0、共起1回ペア比率が 0%、最大 Lift が 18.30 だったため、ポスター用の最低共起回数として採用。
- 正規化後の推奨 min_pair_count: 2
- 正規化後の採用理由: condition B の Lift Top10 を比較した結果、min_pair_count=2 では Top10 の共起回数中央値が 2.0、共起1回ペア比率が 0%、最大 Lift が 18.15 だったため、ポスター用の最低共起回数として採用。
- 正規化前の条件A/B Lift Top10 共通数: 1
- 正規化後の条件A/B Lift Top10 共通数: 1
- 正規化前の条件A/B Lift Jaccard係数: 0.0526
- 正規化後の条件A/B Lift Jaccard係数: 0.0526
- 正規化前の条件A/B Lift Spearman順位相関: 計算不能
- 正規化後の条件A/B Lift Spearman順位相関: 計算不能
- 条件B Lift Top10 の正規化前後共通数: 6
- 条件B Lift Top10 の正規化前後 Jaccard係数: 0.4286
- 正規化前の条件B Lift Top10:
| rank | pair | cooccurrence_count | lift |
| --- | --- | --- | --- |
| 1 | カルダモンミルク × モヒート | 3 | 18.3 |
| 2 | LYCHEE × ライチ | 2 | 15.25 |
| 3 | アールグレイ × コニャック | 5 | 13.5556 |
| 4 | MANGO × マンゴー | 2 | 12.2 |
| 5 | カルダモン × チェリー | 2 | 12.2 |
| 6 | GRAPEFRUIT × グレープフルーツ | 2 | 11.619 |
| 7 | GRAPE × グレープミント | 2 | 10.1667 |
| 8 | グアバ × パッションフルーツ | 3 | 9.15 |
| 9 | ブルーベリー × ミントクリーム | 2 | 9.037 |
| 10 | アサイー × ベリー | 2 | 8.7143 |
- 正規化後の条件B Lift Top10:
| rank | pair | cooccurrence_count | lift |
| --- | --- | --- | --- |
| 1 | カルダモンミルク × モヒート | 3 | 18.15 |
| 2 | カルダモン × チェリー | 2 | 12.1 |
| 3 | アールグレイ × コニャック | 5 | 11.0 |
| 4 | グレープ × グレープミント | 2 | 9.3077 |
| 5 | グアバ × パッションフルーツ | 3 | 9.075 |
| 6 | ブルーベリー × ミントクリーム | 2 | 8.963 |
| 7 | アサイー × ベリー | 2 | 8.6429 |
| 8 | カルダモン × シナモン | 2 | 8.6429 |
| 9 | カルダモンミルク × バナナ | 3 | 8.25 |
| 10 | ウォーターメロン/スイカ × メロン | 2 | 8.0667 |
### 考察
- Lift は低頻度ペアで極端に大きくなりやすいため、共起回数の閾値比較を分けて確認した。
- ポスターでは、共起1回だけのペアに引きずられにくい閾値を採用し、代表レビュー確認とセットで解釈するのが安全。

## 8. 疑似ペア確認
### 結果
| pair | raw_all_multi_present | normalized_all_multi_present | raw_limited_2_5_present | normalized_limited_2_5_present |
| --- | --- | --- | --- | --- |
| LYCHEE×ライチ | True | False | True | False |
| MINT×ミント | False | False | False | False |
| LEMON×レモン | False | False | False | False |
| ICE×アイス | False | False | False | False |
### 考察
- 指定した疑似ペアが正規化後に消えていれば、同一フレーバー分裂による見かけの共起は解消できている。

## 9. 条件変更で順位が大きく変わったペア
### 結果
| pair_rank | flavor_a | flavor_b | cooccurrence_count |
| --- | --- | --- | --- |
| 1 | ブルーベリー | ミント | 21 |
| 2 | オレンジ | ブルーベリー | 17 |
| 3 | ブルーベリー | ベリー | 15 |
| 4 | バニラ | ミント | 14 |
| 5 | オレンジ | レモン | 11 |
### 考察
- 条件Aでは上位でも条件Bで大きく落ちるペアは、列挙型レビューの寄与や長大レビュー特有の共起を疑うべき候補である。

## 10. 代表レビュー確認対象
### 結果
- manual_review_check.csv の行数: 69
- 対象ペア数: 25
### 考察
- 共起頻度上位、Lift上位、条件Aでのみ強いペアを並べることで、ランキングの質を人手で比較しやすくした。

## 11. 生成ファイル一覧
- `poster_analysis/review_extraction_summary.csv`
- `poster_analysis/flavor_alias_candidates.csv`
- `poster_analysis/flavor_normalization_map.csv`
- `poster_analysis/manual_alias_review.csv`
- `poster_analysis/condition_statistics.csv`
- `poster_analysis/cooccurrence_rankings.csv`
- `poster_analysis/lift_rankings.csv`
- `poster_analysis/cooccurrence_condition_comparison.csv`
- `poster_analysis/lift_condition_comparison.csv`
- `poster_analysis/manual_review_check.csv`
- `poster_analysis/figure1_analysis_flow.png`
- `poster_analysis/figure2_condition_top10.png`
- `poster_analysis/figure3_count_lift_scatter.png`
- `poster_analysis/figure4_rank_change.png`
- `poster_analysis/figure5_manual_check.png` は manual_label 未入力のため未生成

## 12. ポスターに載せる主要な発見候補3点
- 条件Aと条件Bで共通して上位に残るペアは、抽出条件を変えても安定な候補として提示できる。
- フレーバー名称正規化により、英語/日本語の疑似ペアを除去してランキングの解釈を安定化できる。
- Lift は最低共起回数を変えるだけでランキングが大きく変わるため、閾値選定の根拠をポスターに明記すべきである。

## 13. 人手確認が必要な作業
- `poster_analysis/manual_review_check.csv` の `manual_label` を `explicit_mix / probable_mix / co_mention_only / unclear` で入力する。
- 入力後に `python3 scripts/summarize_manual_review_check.py` を実行し、集計と図5を生成する。

## 14. 実行コマンドとテスト結果
- 実行コマンド: `python3 scripts/generate_condition_comparison.py`
- manual label 集計: `python3 scripts/summarize_manual_review_check.py`
- テスト結果:
  - PASS: 2〜5種類条件と mix keyword 条件が正しく適用される
  - PASS: 同一レビュー内の重複フレーバーを1回として共起回数を数える
  - PASS: Lift 計算が pair_count * N / (freqA * freqB) に一致する
  - PASS: Top10 比較の共通数と Jaccard 係数を計算できる
  - PASS: manual label 集計でペア単位 explicit_mix 確認を判定できる
  - PASS: mix keyword 判定が指定語に反応する

<!-- manual_review_prelabel:start -->
## 15. 仮ラベル付け集計
- 総件数: 69
- explicit_mix: 21件 (30.4%)
- probable_mix: 0件 (0.0%)
- co_mention_only: 34件 (49.3%)
- unclear: 14件 (20.3%)
- needs_manual_review 件数: 23
- confidence 別件数:
  - high: 42
  - medium: 13
  - low: 14
- ルール別件数:
  - product_list_context: 30
  - explicit_keyword_with_both_flavors: 21
  - missing_target_flavor: 10
  - separate_context_mentions: 4
  - conflicting_rules: 4
<!-- manual_review_prelabel:end -->

<!-- manual_review_final:start -->
## 16. 最終ラベル集計
- 総件数: 69
- explicit_mix: 12件 (17.4%)
- probable_mix: 0件 (0.0%)
- co_mention_only: 44件 (63.8%)
- unclear: 13件 (18.8%)
- unresolved: 0件 (0.0%)
- 解決済み件数: 69
- unresolved 件数: 0
- 採用元別件数:
  - reviewer_label: 69
  - auto_label: 0
  - unresolved: 0
- 少なくとも1件が explicit_mix の pair 数: 7
- 少なくとも1件が explicit_mix の pair 割合: 29.2%
<!-- manual_review_final:end -->

<!-- conditionB_network:start -->
## 17. Condition B 共起ネットワーク
- ノード数: 38
- エッジ数: 75
- 使用した閾値: Condition B（抽出フレーバー数 2〜5）、名称正規化後、min_pair_count=2
- コミュニティ数: 5
- 上位5ノード（degree / weighted degree / betweenness centrality）:
  - ミルク: degree=13, weighted_degree=45.0, betweenness=0.4459
  - ミント: degree=13, weighted_degree=38.0, betweenness=0.3994
  - バニラ: degree=11, weighted_degree=34.0, betweenness=0.2571
  - ハニー: degree=7, weighted_degree=22.0, betweenness=0.0282
  - レモン: degree=8, weighted_degree=21.0, betweenness=0.0375
- 既存図との差分:
  - 既存の `notebooks/flavor_mix_network.py` は全レビューを対象にし、出現頻度 `>=3` と `MAX_NODES=55` を併用した旧ネットワークだった。
  - 今回の図は `poster_analysis/review_extraction_summary.csv` の正規化後データから Condition B のみを抽出し、エッジ閾値を `min_pair_count=2` に統一している。
  - 背景を白基調に変更し、ノード色はコミュニティ、ノードサイズは weighted degree、エッジ幅は共起回数に比例させてポスター用に再設計した。
<!-- conditionB_network:end -->

<!-- normalization_impact:start -->
## 18. 名称正規化の影響図
- 正規化前ユニークフレーバー数: 144
- 正規化後ユニークフレーバー数: 136
- 共起回数Top10の共通ペア数: 9/10
- 共起回数Top10のJaccard係数: 0.8182
- Lift Top10の共通ペア数: 6/10
- Lift Top10のJaccard係数: 0.4286
- 図出力: `poster_analysis/figure_normalization_impact.png`, `poster_analysis/figure_normalization_impact.pdf`
- 数値表: `poster_analysis/normalization_impact_metrics.csv`, `poster_analysis/normalization_impact_metrics.md`
<!-- normalization_impact:end -->
