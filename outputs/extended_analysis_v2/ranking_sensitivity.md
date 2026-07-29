# ランキング感度分析 v2

## Settingごとの上位20件監査

| Setting | pair_count=2 | same_sentence=0 | product_name | parent_child |
| --- | ---: | ---: | ---: | ---: |
| SettingA_balanced | 0 | 0 | 0 | 0 |
| SettingB_cooccurrence | 0 | 0 | 0 | 0 |
| SettingC_context | 1 | 0 | 0 | 0 |

## Setting間比較

| Setting A | Setting B | Common | Jaccard | Spearman |
| --- | --- | ---: | ---: | ---: |
| SettingA_balanced | SettingB_cooccurrence | 19 | 0.9048 | 0.9105 |
| SettingA_balanced | SettingC_context | 19 | 0.9048 | 0.8947 |
| SettingB_cooccurrence | SettingC_context | 18 | 0.8182 | 0.6739 |
