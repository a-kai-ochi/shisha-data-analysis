# Paper Dataset Draft

## 日本語案

追加データソースとして，Shisha Cafe & Bar LAGOS の「フレーバーレビュー」カテゴリ（13記事）を収集対象とした。収集は管理者の許可を得た上で2026-07-29 に実施し，HTML構造と取得日時を記録しつつ，サーバー負荷を抑えるため逐次的に取得した。
対象データは 2024-12-27 から 2025-12-10 に公開された編集記事型データ（source_type = editorial_review）であり，記事本文とおすすめミックス節を別個に抽出して保存した。
このデータは既存のユーザーレビューデータとは別ソースとして保持しており，本段階では自動統合していない。

## English Draft

As an additional data source, we collected 13 articles from the "フレーバーレビュー" category of Shisha Cafe & Bar LAGOS. The collection was conducted on 2026-07-29 with explicit permission from the site administrator, while recording the HTML structure and retrieval timestamps and using sequential requests to reduce server load.
The collected articles were published between 2024-12-27 and 2025-12-10 and were treated as editorial articles (source_type = editorial_review). We extracted both the main article text and the recommended-mix sections, and kept this corpus separate from the existing user-review data.
