# Sensitivity Analysis Report

- Baseline ranking file: `pair_ranking_tier2.csv`
- Candidate pool for weight sensitivity: 52 non-excluded candidates
- Tier labels are threshold-based and therefore unchanged under weight-only settings.

## Key Observations

- `drop_normalized_support`: Top10 Jaccard=0.429, new=グアバ||レモン | グレープ||ベリー | バニラ||ミント | ミルク||メロン, dropped=オレンジ||グレープフルーツ | グレープ||ミント | コニャック||バニラ | ミント||ライム
- `drop_normalized_smoothed_positive_ratio`: Top10 Jaccard=0.538, new=オレンジ||レモン | ミルク||メロン | ミント||ライチ, dropped=アイス||ミント | バニラ||ミルク | ベリー||レモン
- `statistics_centered`: Top10 Jaccard=0.538, new=オレンジ||レモン | グアバ||ミント | ミント||ライチ, dropped=アイス||ミント | バニラ||ミルク | ベリー||レモン
- `drop_normalized_centrality_mean`: Top10 Jaccard=0.667, new=グレープ||ベリー | ミルク||メロン, dropped=ベリー||レモン | ミント||ライム
- `half_normalized_smoothed_positive_ratio`: Top10 Jaccard=0.667, new=オレンジ||レモン | ミント||ライチ, dropped=バニラ||ミルク | ベリー||レモン
