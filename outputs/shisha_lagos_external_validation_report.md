# Shisha LAGOS External Validation Report

## Summary

- LAGOS抽出行数: 200
- LAGOS有効行数: 151
- LAGOSユニークペア数: 133
- 既存ランキングとの一致ペア数: 17
- LAGOSのみのペア数: 116
- 既存ランキングのみにあるペア数: 35

## Agreement@K

- K=10: common=5, precision=0.5000, recall=0.0376, jaccard=0.0362
- K=20: common=11, precision=0.5500, recall=0.0827, jaccard=0.0775
- K=50: common=17, precision=0.3400, recall=0.1278, jaccard=0.1024

## Common Pairs

- rank 1: グレープフルーツ||レモン (LAGOS記事数=1, row数=1)
- rank 3: ミント||レモン (LAGOS記事数=1, row数=1)
- rank 8: バニラ||ミルク (LAGOS記事数=1, row数=1)
- rank 9: ミント||ライム (LAGOS記事数=1, row数=1)
- rank 10: オレンジ||グレープフルーツ (LAGOS記事数=2, row数=2)
- rank 11: オレンジ||レモン (LAGOS記事数=1, row数=1)
- rank 12: ミルク||メロン (LAGOS記事数=1, row数=1)
- rank 13: ミント||ライチ (LAGOS記事数=1, row数=1)
- rank 16: グアバ||ミント (LAGOS記事数=1, row数=1)
- rank 17: グアバ||グレープフルーツ (LAGOS記事数=1, row数=1)

## LAGOS-only Pairs

- オレンジ||バニラ: 共起なし
- オレンジ||ピーチ: 共起なし
- オレンジ||マンゴー: 共起なし
- オレンジ||メロン: 共起なし
- ココナッツ||ストロベリー: 既存コーパスで出現なし:ストロベリー
- ココナッツ||バニラ: ランキング対象外:no_same_sentence_context
- ココナッツ||ピーチ: 共起なし
- ココナッツ||ブルーベリー: 共起なし
- ココナッツ||マンゴー: 共起なし
- ココナッツ||メロン: 共起なし

## Note

- 一致率は推薦精度ではなく、編集記事型の小規模補助データソースとの一致・被覆率として扱う。
- LAGOSは13記事であり、既存のユーザー投稿型レビューとは source_type が異なる。
