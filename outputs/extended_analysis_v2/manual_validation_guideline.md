# Manual Validation Guideline

## mix_relation_label

- `explicit_mix`
  実際に2つのフレーバーを混ぜる、加える、組み合わせる記述が明確にある
- `likely_mix`
  配合は明示されていないが、同じミックスとして説明されている可能性が高い
- `co_mention_only`
  同じ文にあるだけで、ミックス関係はない
- `unclear`
  文脈だけでは判定できない

## evaluation_label

- `positive`
  ペアまたはミックス全体への肯定的評価がある
- `neutral`
  配合説明のみで評価はない
- `negative`
  ペアまたはミックス全体への否定的評価がある
- `mixed`
  肯定と否定の両方がある
- `unclear`
  評価対象が単体かペアか分からない

## taste_role_label

- `explained`
  少なくとも一方のフレーバーが、甘さ、清涼感、香り、コク、アクセントなどをどのように与えるか説明されている
- `not_explained`
  組合せの記述のみで役割説明はない
- `unclear`
  役割説明か判断できない

## recommendation_validity

- `valid`
  明示的または可能性の高いミックスで、肯定評価または役割説明がある
- `partially_valid`
  ミックス関係は確認できるが、評価・役割説明が弱い
- `invalid`
  共起のみ、商品名由来、意味的重複、または否定的な候補
- `unclear`
  根拠不足で判断できない

## semantic_overlap_label

- `distinct`
  別のフレーバーとして扱える
- `similar`
  近い意味だが別候補として残しうる
- `duplicate`
  実質的に同じ意味で、候補としての重複が大きい
- `unclear`
  文脈だけでは判断できない

## Reviewer Columns

- 1名評価の場合は、基本列 `mix_relation_label` から `reviewer_comment` までを入力する
- 2名評価の場合は、`reviewer1_*` と `reviewer2_*` の列を使用する
- 未評価の項目は空欄のままにする

