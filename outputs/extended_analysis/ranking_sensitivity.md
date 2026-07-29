# ランキング感度分析

初期重み設定3種（バランス型・共起重視・文脈重視）の上位20件比較。

| Setting A | Setting B | Common | Jaccard | Spearman |
| --- | --- | ---: | ---: | ---: |
| SettingA_balanced | SettingB_cooccurrence | 15 | 0.6000 | 0.8143 |
| SettingA_balanced | SettingC_context | 15 | 0.6000 | -0.1286 |
| SettingB_cooccurrence | SettingC_context | 11 | 0.3793 | -0.0273 |
