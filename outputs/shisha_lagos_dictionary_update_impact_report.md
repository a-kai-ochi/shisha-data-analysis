# Shisha LAGOS Dictionary Update Impact Simulation

このファイルは辞書更新前の暫定シミュレーションであり、正式結果ではない。

## baseline

- override_mapping_count: 0
- override_raws: 
- valid_rows: 151
- unresolved_rows: 47
- unique_pairs: 133
- common_pairs: 17
- Top10/20/50 common: 5/11/17
- Precision@10/20/50: 0.500/0.550/0.340
- Recall@10/20/50: 0.038/0.083/0.128
- Jaccard@10/20/50: 0.036/0.077/0.102

## conservative

- override_mapping_count: 6
- override_raws: すいか | ゆず | コーヒー | スイカ | ミルクティ | ミルクティー
- valid_rows: 169
- unresolved_rows: 29
- unique_pairs: 149
- common_pairs: 17
- Top10/20/50 common: 5/11/17
- Precision@10/20/50: 0.500/0.550/0.340
- Recall@10/20/50: 0.034/0.074/0.114
- Jaccard@10/20/50: 0.032/0.070/0.093

- baseline差分: valid_rows +18, common_pairs +0

## extended

- override_mapping_count: 9
- override_raws: すいか | ゆず | コーヒー | スイカ | チャイ | パイン | ミルクティ | ミルクティー | 洋ナシ
- valid_rows: 175
- unresolved_rows: 23
- unique_pairs: 155
- common_pairs: 17
- Top10/20/50 common: 5/11/17
- Precision@10/20/50: 0.500/0.550/0.340
- Recall@10/20/50: 0.032/0.071/0.110
- Jaccard@10/20/50: 0.031/0.067/0.090

- baseline差分: valid_rows +24, common_pairs +0

