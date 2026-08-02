# Manual Validation Summary

- primary_label_source: `reviewer1`

## Scope Metrics

| Scope | Metric | Value | Labeled N |
| --- | --- | ---: | ---: |
| all | candidate_count | 17.0000 | 17 |
| all | explicit_mix_rate | 0.4118 | 17 |
| all | explicit_or_likely_mix_rate | 1.0000 | 17 |
| all | positive_rate | 1.0000 | 17 |
| all | negative_rate | 0.0588 | 17 |
| all | role_explained_rate | 0.7059 | 17 |
| all | recommendation_valid_rate | 0.4706 | 17 |
| all | valid_or_partially_valid_rate | 1.0000 | 17 |
| all | semantic_overlap_similar_or_duplicate_rate | 0.2941 | 17 |
| all | unclear_rate | 0.0000 | 17 |
| top_5 | candidate_count | 5.0000 | 5 |
| top_5 | explicit_mix_rate | 0.8000 | 5 |
| top_5 | explicit_or_likely_mix_rate | 1.0000 | 5 |
| top_5 | positive_rate | 1.0000 | 5 |
| top_5 | negative_rate | 0.0000 | 5 |
| top_5 | role_explained_rate | 1.0000 | 5 |
| top_5 | recommendation_valid_rate | 1.0000 | 5 |
| top_5 | valid_or_partially_valid_rate | 1.0000 | 5 |
| top_5 | semantic_overlap_similar_or_duplicate_rate | 0.2000 | 5 |
| top_5 | unclear_rate | 0.0000 | 5 |
| top_10 | candidate_count | 10.0000 | 10 |
| top_10 | explicit_mix_rate | 0.6000 | 10 |
| top_10 | explicit_or_likely_mix_rate | 1.0000 | 10 |
| top_10 | positive_rate | 1.0000 | 10 |
| top_10 | negative_rate | 0.0000 | 10 |
| top_10 | role_explained_rate | 0.9000 | 10 |
| top_10 | recommendation_valid_rate | 0.7000 | 10 |
| top_10 | valid_or_partially_valid_rate | 1.0000 | 10 |
| top_10 | semantic_overlap_similar_or_duplicate_rate | 0.3000 | 10 |
| top_10 | unclear_rate | 0.0000 | 10 |
| top_17 | candidate_count | 17.0000 | 17 |
| top_17 | explicit_mix_rate | 0.4118 | 17 |
| top_17 | explicit_or_likely_mix_rate | 1.0000 | 17 |
| top_17 | positive_rate | 1.0000 | 17 |
| top_17 | negative_rate | 0.0588 | 17 |
| top_17 | role_explained_rate | 0.7059 | 17 |
| top_17 | recommendation_valid_rate | 0.4706 | 17 |
| top_17 | valid_or_partially_valid_rate | 1.0000 | 17 |
| top_17 | semantic_overlap_similar_or_duplicate_rate | 0.2941 | 17 |
| top_17 | unclear_rate | 0.0000 | 17 |

## Feature Relations

| Scope | Metric | Label | Value | N |
| --- | --- | --- | ---: | ---: |
| all | smoothed_positive_ratio_vs_manual_positive_feature_mean | manual_yes | 0.1376 | 17 |
| all | smoothed_positive_ratio_vs_manual_positive_feature_median | manual_yes | 0.0667 | 17 |
| all | smoothed_positive_ratio_vs_manual_positive_spearman | manual_binary |  | 17 |
| all | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_no | 0.0059 | 16 |
| all | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_no | 0.0067 | 16 |
| all | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_yes | 0.1722 | 1 |
| all | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_yes | 0.1722 | 1 |
| all | smoothed_negative_ratio_vs_manual_negative_spearman | manual_binary | 0.4588 | 17 |
| all | smoothed_role_ratio_vs_manual_role_feature_mean | manual_no | 0.0133 | 5 |
| all | smoothed_role_ratio_vs_manual_role_feature_median | manual_no | 0.0133 | 5 |
| all | smoothed_role_ratio_vs_manual_role_feature_mean | manual_yes | 0.0383 | 12 |
| all | smoothed_role_ratio_vs_manual_role_feature_median | manual_yes | 0.0122 | 12 |
| all | smoothed_role_ratio_vs_manual_role_spearman | manual_binary | -0.2864 | 17 |
| all | recommendation_validity_score_mean | partially_valid | 0.2249 | 9 |
| all | recommendation_validity_score_median | partially_valid | 0.2323 | 9 |
| all | recommendation_validity_score_mean | valid | 0.3474 | 8 |
| all | recommendation_validity_score_median | valid | 0.3613 | 8 |
| all | recommendation_validity_score_spearman | ordinal_validity | 0.5292 | 17 |
| top_5 | smoothed_positive_ratio_vs_manual_positive_feature_mean | manual_yes | 0.1540 | 5 |
| top_5 | smoothed_positive_ratio_vs_manual_positive_feature_median | manual_yes | 0.0556 | 5 |
| top_5 | smoothed_positive_ratio_vs_manual_positive_spearman | manual_binary |  | 5 |
| top_5 | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_no | 0.0047 | 5 |
| top_5 | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_no | 0.0048 | 5 |
| top_5 | smoothed_negative_ratio_vs_manual_negative_spearman | manual_binary |  | 5 |
| top_5 | smoothed_role_ratio_vs_manual_role_feature_mean | manual_yes | 0.0344 | 5 |
| top_5 | smoothed_role_ratio_vs_manual_role_feature_median | manual_yes | 0.0095 | 5 |
| top_5 | smoothed_role_ratio_vs_manual_role_spearman | manual_binary |  | 5 |
| top_5 | recommendation_validity_score_mean | valid | 0.4233 | 5 |
| top_5 | recommendation_validity_score_median | valid | 0.4262 | 5 |
| top_5 | recommendation_validity_score_spearman | ordinal_validity |  | 5 |
| top_10 | smoothed_positive_ratio_vs_manual_positive_feature_mean | manual_yes | 0.1684 | 10 |
| top_10 | smoothed_positive_ratio_vs_manual_positive_feature_median | manual_yes | 0.1667 | 10 |
| top_10 | smoothed_positive_ratio_vs_manual_positive_spearman | manual_binary |  | 10 |
| top_10 | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_no | 0.0055 | 10 |
| top_10 | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_no | 0.0052 | 10 |
| top_10 | smoothed_negative_ratio_vs_manual_negative_spearman | manual_binary |  | 10 |
| top_10 | smoothed_role_ratio_vs_manual_role_feature_mean | manual_no | 0.0133 | 1 |
| top_10 | smoothed_role_ratio_vs_manual_role_feature_median | manual_no | 0.0133 | 1 |
| top_10 | smoothed_role_ratio_vs_manual_role_feature_mean | manual_yes | 0.0246 | 9 |
| top_10 | smoothed_role_ratio_vs_manual_role_feature_median | manual_yes | 0.0111 | 9 |
| top_10 | smoothed_role_ratio_vs_manual_role_spearman | manual_binary | -0.2426 | 10 |
| top_10 | recommendation_validity_score_mean | partially_valid | 0.2678 | 3 |
| top_10 | recommendation_validity_score_median | partially_valid | 0.2632 | 3 |
| top_10 | recommendation_validity_score_mean | valid | 0.3762 | 7 |
| top_10 | recommendation_validity_score_median | valid | 0.4259 | 7 |
| top_10 | recommendation_validity_score_spearman | ordinal_validity | 0.4179 | 10 |
| top_17 | smoothed_positive_ratio_vs_manual_positive_feature_mean | manual_yes | 0.1376 | 17 |
| top_17 | smoothed_positive_ratio_vs_manual_positive_feature_median | manual_yes | 0.0667 | 17 |
| top_17 | smoothed_positive_ratio_vs_manual_positive_spearman | manual_binary |  | 17 |
| top_17 | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_no | 0.0059 | 16 |
| top_17 | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_no | 0.0067 | 16 |
| top_17 | smoothed_negative_ratio_vs_manual_negative_feature_mean | manual_yes | 0.1722 | 1 |
| top_17 | smoothed_negative_ratio_vs_manual_negative_feature_median | manual_yes | 0.1722 | 1 |
| top_17 | smoothed_negative_ratio_vs_manual_negative_spearman | manual_binary | 0.4588 | 17 |
| top_17 | smoothed_role_ratio_vs_manual_role_feature_mean | manual_no | 0.0133 | 5 |
| top_17 | smoothed_role_ratio_vs_manual_role_feature_median | manual_no | 0.0133 | 5 |
| top_17 | smoothed_role_ratio_vs_manual_role_feature_mean | manual_yes | 0.0383 | 12 |
| top_17 | smoothed_role_ratio_vs_manual_role_feature_median | manual_yes | 0.0122 | 12 |
| top_17 | smoothed_role_ratio_vs_manual_role_spearman | manual_binary | -0.2864 | 17 |
| top_17 | recommendation_validity_score_mean | partially_valid | 0.2249 | 9 |
| top_17 | recommendation_validity_score_median | partially_valid | 0.2323 | 9 |
| top_17 | recommendation_validity_score_mean | valid | 0.3474 | 8 |
| top_17 | recommendation_validity_score_median | valid | 0.3613 | 8 |
| top_17 | recommendation_validity_score_spearman | ordinal_validity | 0.5292 | 17 |

## Inter-Rater Agreement

評価者間一致は未計算。
