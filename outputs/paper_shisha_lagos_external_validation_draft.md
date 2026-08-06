# Paper Draft: External Validation with Shisha LAGOS

## 日本語案

補助的な外部比較として，Shisha Cafe & Bar LAGOS の編集記事型データに明示されたおすすめミックス表を構造化し，既存レビューから得たランキング候補との一致を調べた。LAGOS 側は 13 記事の小規模データであり，ユーザー投稿型レビューとは性質が異なるため，推薦精度ではなく，独立した編集記事型ソースとの外部的一致として扱う。
LAGOS から得られた有効ユニークペア数は 133 組であり，既存ランキング上位10件との一致数は 5，上位20件との一致数は 11，上位50件との一致数は 17 であった。
一致率・被覆率としてみると，Precision@10=0.500，Precision@20=0.550，Precision@50=0.340 であり，Recall@10=0.038，Recall@20=0.083，Recall@50=0.128 であった。
この結果はあくまで小規模な補助的比較であり，LAGOS に存在しない候補を不適切とみなすものではない。

## English Draft

As an auxiliary external comparison, we structured the recommended-mix tables explicitly stated in the editorial articles from Shisha Cafe & Bar LAGOS and compared them with the ranked candidates obtained from the existing review corpus. Because the LAGOS corpus consists of only 13 editorial articles and differs in source type from user-review-like data, we treat the results as external agreement rather than recommendation accuracy.
The number of valid unique LAGOS pairs was 133, and the overlap with the existing ranking was 5 pairs in the top 10, 11 pairs in the top 20, and 17 pairs in the top 50.
In terms of agreement/coverage, Precision@10=0.500, Precision@20=0.550, Precision@50=0.340, while Recall@10=0.038, Recall@20=0.083, and Recall@50=0.128.
These values should not be interpreted as recommendation accuracy or as definitive ground-truth validation.
