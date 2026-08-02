# Source Characteristics

## 日本語

- 既存の `CLOUD` データは、`cloud_reviews_final.csv` に 222 件のレビュー本文が格納された、ユーザー投稿型に近いレビュー系コーパスとして扱っている。
- `Shisha Cafe & Bar LAGOS` データは 13 件の記事からなる `editorial_review` であり、店舗・ライター側の編集記事型ソースとして区別して保持している。
- `CLOUD` は1行1レビューで段落構造を持たないのに対し、LAGOSは1行1記事に加えて段落単位・おすすめミックス節単位の出力を持つ。
- LAGOS記事の本文長は平均 1540.6 文字で、同一記事内に複数の説明段落やおすすめミックス表が含まれる。
- LAGOSはおすすめミックスを明示的な見出し付き節として持つ一方、CLOUDはレビュー本文中に自由記述として混在する。
- LAGOSでは1記事内に複数の候補ミックスや補助フレーバーが列挙されうるため、各記事を独立したユーザーレビュー1件と単純同一視することは難しい。
- そのため、既存レビューとLAGOS記事を単純結合すると、編集記事に含まれる定型説明や体系的なミックス提案が共起頻度を押し上げるバイアスが生じる可能性がある。
- 後続分析では、少なくとも `source_type` 別の集計と、編集記事由来の共起・表現の影響を切り分けた検証が必要である。

## English

- The existing `CLOUD` corpus contains 222 review-level records and is treated as a user-review-like source.
- The `Shisha Cafe & Bar LAGOS` corpus contains 13 flavor-review articles and is stored separately as `editorial_review`.
- `CLOUD` is a flat review-level table, whereas the LAGOS corpus preserves article-level, paragraph-level, and recommended-mix-section-level structures.
- LAGOS articles explicitly provide recommended-mix sections and editorial explanations, so one article cannot be assumed to be equivalent to one independent user review.
- A naive merge of CLOUD and LAGOS may bias co-occurrence statistics because editorial templates and systematically curated mix suggestions can inflate repeated patterns.
- Therefore, downstream analyses should at least stratify by `source_type` and separately assess the effect of editorial articles before integration.
