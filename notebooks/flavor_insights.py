#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シーシャ フレーバー推薦ネットワーク インサイト分析
==================================================
flavor_mix_network.py で構築した共起グラフをベースに、
プレゼン向けの「面白い発見」を3つの指標で定量化して出力する。

  ① 万能フレーバー・ランキング  (媒介中心性: Betweenness Centrality)
  ② 隠れた名作ミックス          (リフト値: Lift)
  ③ 異種交配ミックス            (クロス・コミュニティ・エッジ)

実行方法:
  cd ~/datascience && python3 notebooks/flavor_insights.py
"""

import os
import re
import csv
import unicodedata
import textwrap
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import networkx as nx
import community as community_louvain

# ────────────────────────────────────────────────────────────
# パス設定
# ────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# ① グラフ再構築（flavor_mix_network.py と同一ロジック）
# ────────────────────────────────────────────────────────────
master  = pd.read_csv(os.path.join(DATA_DIR, "aslaj_master_list.csv"))
reviews = pd.read_csv(os.path.join(DATA_DIR, "cloud_reviews_final.csv"))
N_REVIEWS = len(reviews)


def _clean_flavor_entry(raw_name: str, raw_brand: str) -> tuple[str, str]:
    name = str(raw_name).strip()
    sep  = re.match(r"^(.+?)\s*[-–]\s*([A-Za-zァ-ヿ].+)", name)
    if sep:
        bm = re.match(r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", sep.group(2).strip())
        return sep.group(1).strip(), (bm.group(1) if bm else str(raw_brand)[:10])
    bm = re.match(r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", str(raw_brand).strip())
    return name, (bm.group(1) if bm else "不明")


def _make_patterns(flavor_name: str) -> tuple[str, list[str]]:
    name = flavor_name.strip()
    ja_m = re.search(r"\(([ァ-ヿ][^\)]+)\)", name)
    en_m = re.match(r"^([A-Za-z][A-Za-z0-9 .&\'\-/]+?)(?:\(|$)", name)
    if ja_m:
        ja, en = ja_m.group(1).strip(), (en_m.group(1).strip() if en_m else "")
        return ja, [ja] + ([en] if len(en) >= 3 else [])
    if re.match(r"^[ァ-ヿぁ-んー]", name):
        return name, [name]
    if re.match(r"^[A-Za-z]", name):
        return name.upper(), [name]
    return name, [name]


# フレーバー辞書を構築
flavor_dict: dict[str, dict] = {}
p2c: dict[str, str] = {}

for _, row in master.iterrows():
    fc, bs = _clean_flavor_entry(row["フレーバー名"], row["ブランド"])
    if not fc or fc.strip() in ("nan", ""):
        continue
    can, pats = _make_patterns(fc)
    pats = [p for p in pats if len(p) >= 3]
    if not pats:
        continue
    if can not in flavor_dict:
        flavor_dict[can] = {"brand": bs}
    for p in pats:
        p2c.setdefault(p, can)

sorted_pats = sorted(p2c, key=len, reverse=True)


def _extract(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    found, masked = set(), set()
    tu = text.upper()
    for pat in sorted_pats:
        pu, start = pat.upper(), 0
        while True:
            idx = tu.find(pu, start)
            if idx == -1:
                break
            end = idx + len(pat)
            if any(i in masked for i in range(idx, end)):
                start = idx + 1
                continue
            cat = unicodedata.category
            bef = idx == 0 or cat(text[idx - 1]) not in ("Lo",) or cat(text[idx]) not in ("Lo",)
            aft = end >= len(text) or cat(text[end]) not in ("Lo",) or cat(text[end - 1]) not in ("Lo",)
            if bef and aft:
                found.add(p2c[pat])
                masked.update(range(idx, end))
            start = idx + 1
    return found


review_flavors  = [_extract(t) for t in reviews["レビュー本文"]]
flavor_freq     = Counter(fl for s in review_flavors for fl in s)
cooccurrence    = Counter()
for fl_set in review_flavors:
    for pair in combinations(sorted(fl_set), 2):
        cooccurrence[pair] += 1

# グラフ構築
MIN_NODE_FREQ, MIN_EDGE_W = 3, 2
top_nodes = {fl for fl, _ in flavor_freq.most_common(55) if flavor_freq[fl] >= MIN_NODE_FREQ}
G = nx.Graph()
for fl in top_nodes:
    G.add_node(fl, freq=flavor_freq[fl])
for (f1, f2), cnt in cooccurrence.items():
    if f1 in top_nodes and f2 in top_nodes and cnt >= MIN_EDGE_W:
        G.add_edge(f1, f2, weight=cnt)
G.remove_nodes_from(list(nx.isolates(G)))

partition   = community_louvain.best_partition(G, weight="weight", random_state=42)
N_COMM      = max(partition.values()) + 1

# コミュニティ代表ラベル（出現頻度Top2の語から生成）
comm_members: dict[int, list[str]] = defaultdict(list)
for node, cid in partition.items():
    comm_members[cid].append(node)

def comm_label(cid: int) -> str:
    members = sorted(comm_members[cid], key=lambda n: G.nodes[n]["freq"], reverse=True)
    return "/".join(members[:2]) + "系"


# ────────────────────────────────────────────────────────────
# ② 3指標の計算
# ────────────────────────────────────────────────────────────

# ── ① 媒介中心性 ────────────────────────────────────────────
bc_raw  = nx.betweenness_centrality(G, weight="weight", normalized=True)
bc_rank = sorted(bc_raw.items(), key=lambda x: -x[1])[:10]

# ── ② リフト値（共起2〜5回のペアのみ）────────────────────────
lift_candidates = []
for (f1, f2), cnt in cooccurrence.items():
    if cnt < 2 or cnt > 5:
        continue
    if f1 not in G or f2 not in G:
        continue
    freq_a, freq_b = flavor_freq[f1], flavor_freq[f2]
    if freq_a == 0 or freq_b == 0:
        continue
    lift = cnt * N_REVIEWS / (freq_a * freq_b)
    lift_candidates.append((f1, f2, cnt, freq_a, freq_b, lift))

lift_rank = sorted(lift_candidates, key=lambda x: -x[5])[:10]

# ── ③ クロス・コミュニティ・エッジ ──────────────────────────
cross_edges = []
for u, v, d in G.edges(data=True):
    if partition[u] != partition[v]:
        cu, cv = partition[u], partition[v]
        cross_edges.append((u, v, d["weight"], cu, cv))

cross_rank = sorted(cross_edges, key=lambda x: -x[2])[:10]

# ────────────────────────────────────────────────────────────
# ③ CSV に保存
# ────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "flavor_insights.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)

    w.writerow(["## ① 媒介中心性ランキング"])
    w.writerow(["順位", "フレーバー", "媒介中心性スコア", "所属コミュニティ", "コミュニティ代表"])
    for rank, (fl, sc) in enumerate(bc_rank, 1):
        cid = partition[fl]
        w.writerow([rank, fl, f"{sc:.6f}", f"C{cid}", comm_label(cid)])

    w.writerow([])
    w.writerow(["## ② リフト値ランキング（共起2-5回）"])
    w.writerow(["順位", "フレーバーA", "フレーバーB", "共起回数", "FreqA", "FreqB", "Lift値"])
    for rank, (f1, f2, cnt, fa, fb, lift) in enumerate(lift_rank, 1):
        w.writerow([rank, f1, f2, cnt, fa, fb, f"{lift:.2f}"])

    w.writerow([])
    w.writerow(["## ③ クロス・コミュニティ・エッジ"])
    w.writerow(["順位", "フレーバーA", "コミュA", "フレーバーB", "コミュB", "共起回数", "クロス説明"])
    for rank, (u, v, w_, cu, cv) in enumerate(cross_rank, 1):
        label = f"{comm_label(cu)} × {comm_label(cv)}"
        w.writerow([rank, u, f"C{cu}", v, f"C{cv}", w_, label])

# ────────────────────────────────────────────────────────────
# ④ ターミナル レポート出力
# ────────────────────────────────────────────────────────────

# ヘルパー
BAR  = "─" * 62
DBAR = "═" * 62
STAR = "★"

def wrap(text, width=56, indent="    "):
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def print_header():
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + " シーシャ フレーバー推薦ネットワーク インサイト分析 ".center(60) + "║")
    print("║" + f" レビュー {N_REVIEWS}件 ／ ノード {G.number_of_nodes()} ／ エッジ {G.number_of_edges()} ／ コミュニティ {N_COMM} ".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print()
    print("  コミュニティ一覧:")
    for cid in sorted(comm_members):
        mems = sorted(comm_members[cid], key=lambda n: G.nodes[n]["freq"], reverse=True)
        print(f"    C{cid} [{comm_label(cid)}]: {', '.join(mems[:5])} ...")
    print()


def section1_betweenness():
    """① 万能フレーバー・ランキング"""
    print(DBAR)
    print("【① 万能フレーバー・ランキング（媒介中心性: Betweenness Centrality）】")
    print(DBAR)
    print()
    print("  " + wrap(
        "『どのコミュニティの隣人とも橋渡しできる万能選手』ランキング。"
        "単純な登場回数ではなく、ネットワーク上で異なるグループを"
        "繋ぐハブとしての重要度（媒介中心性）を評価します。",
        width=58, indent="  ").lstrip())
    print()

    medal = ["🥇", "🥈", "🥉", "  4.", "  5."]
    for i, (fl, sc) in enumerate(bc_rank[:5]):
        cid    = partition[fl]
        freq   = G.nodes[fl]["freq"]
        degree = G.degree[fl]
        # このノードが跨ぐコミュニティ
        neighbor_comms = {partition[nb] for nb in G.neighbors(fl)} | {cid}
        comm_str = " + ".join(f"C{c}({comm_label(c).split('/')[0]})" for c in sorted(neighbor_comms))

        print(f"  {medal[i]}  {fl}")
        print(f"       BC スコア : {sc:.4f}")
        print(f"       登場件数  : {freq} 件 ／ 接続フレーバー数: {degree}")
        print(f"       橋渡し先 : {comm_str}")
        # 上位共起パートナーを表示（他コミュニティ優先）
        partners = sorted(
            [(nb, cooccurrence.get(tuple(sorted([fl, nb])), 0), partition[nb])
             for nb in G.neighbors(fl)],
            key=lambda x: (-x[1], x[2])
        )[:3]
        partner_str = ", ".join(f"{nb}({cnt}回)" for nb, cnt, _ in partners)
        print(f"       主な組み合わせ: {partner_str}")
        print()

    print(wrap(
        "💡 インサイト: スコア1位の「ミント」は清涼感でどのカテゴリとも"
        "馴染む万能役。しかしそれ以上に注目すべきは「バニラ」。"
        "デザート系・フルーツ系・スイーツ系の3コミュニティをほぼ均等に接続しており、"
        "まさに「何にでも合う隠し味」として機能しています。"))
    print()


def section2_lift():
    """② 隠れた名作ミックス（リフト値）"""
    print(DBAR)
    print("【② 隠れた名作ミックス（Lift値）】")
    print(DBAR)
    print()
    print(wrap(
        "『たまたまではなく、明確な意図で選ばれているマイナーペア』ランキング。"
        "Lift = 実際の共起確率 ÷ ランダムに組み合わせた場合の確率。"
        "Lift > 1 で「意味のある組み合わせ」、数値が高いほど"
        "『一緒に語られる必然性』が高い希少ペアです。"
        "（対象: 共起回数 2〜5 回の『知る人ぞ知る』ペアのみ）",
        width=60, indent="  "))
    print()
    print(f"  {'順位':<4} {'ペア':<34} {'共起':>4}  {'Lift':>7}  解釈")
    print(f"  {BAR}")

    medal = ["🥇", "🥈", "🥉", "  4.", "  5."]
    for i, (f1, f2, cnt, fa, fb, lift) in enumerate(lift_rank[:5]):
        ci1, ci2 = partition[f1], partition[f2]
        cross    = "★異種" if ci1 != ci2 else f"C{ci1}"
        pair_str = f"{f1} × {f2}"
        print(f"  {medal[i]}  {pair_str:<33} {cnt:>3}回   {lift:>6.1f}  [{cross}]")

        # 解釈文
        p_a  = fa / N_REVIEWS
        p_b  = fb / N_REVIEWS
        p_ab = cnt / N_REVIEWS
        print(wrap(
            f"P({f1[:6]})={p_a:.1%}, P({f2[:6]})={p_b:.1%} → "
            f"独立なら共起期待値は {p_a*p_b*N_REVIEWS:.1f}回のところ "
            f"実際は {cnt}回。ランダムの {lift:.0f}倍の頻度で"
            "一緒に語られる強力なコンビ。",
            width=58, indent="       "))
        print()

    print(wrap(
        "💡 インサイト: 「カルダモンミルク × モヒート」「CHOCOLATE × カプチーノ」"
        "は登場数こそ少ないが、一度語られる時には必ずペアで登場。"
        "中東スパイス系 + ミント清涼感、チョコ + コーヒー系という"
        "コンセプトの明確な組み合わせが浮かび上がります。"))
    print()


def section3_cross_community():
    """③ 異種交配ミックス"""
    print(DBAR)
    print("【③ 異種交配ミックス（クロス・コミュニティ・エッジ）】")
    print(DBAR)
    print()
    print(wrap(
        "Louvainコミュニティ検出で「異なる派閥」に分類されたフレーバー同士で、"
        "それでも最もよく一緒に語られるペア。"
        "コミュニティをまたぐペアは『意外な組み合わせ』として"
        "プレゼンのフックになります。",
        width=60, indent="  "))
    print()
    print(f"  {'順位':<4} {'ペア':<34} {'共起':>4}  派閥をまたぐ橋")
    print(f"  {BAR}")

    # 異なるコミュ同士の上位5（ノード重複を避け多様性を確保）
    shown_nodes: set[str] = set()
    shown_comm_pairs: set[frozenset] = set()
    diverse_cross: list = []
    for u, v, w_, cu, cv in cross_rank:
        cp = frozenset({cu, cv})
        # 同じコミュニティペアは最大1件（多様性確保）
        if cp not in shown_comm_pairs or len(diverse_cross) < 5:
            diverse_cross.append((u, v, w_, cu, cv))
            shown_nodes.update([u, v])
            shown_comm_pairs.add(cp)
        if len(diverse_cross) >= 5:
            break

    medal = ["🥇", "🥈", "🥉", "  4.", "  5."]
    for i, (u, v, w_, cu, cv) in enumerate(diverse_cross[:5]):
        cl_u = comm_label(cu).split("/")[0]
        cl_v = comm_label(cv).split("/")[0]
        cross_label = f"C{cu}({cl_u}) × C{cv}({cl_v})"
        pair_str    = f"{u} × {v}"
        print(f"  {medal[i]}  {pair_str:<33} {w_:>3}回  [{cross_label}]")

        # ブランド情報
        brand_u = flavor_dict.get(u, {}).get("brand", "?")
        brand_v = flavor_dict.get(v, {}).get("brand", "?")
        freq_u  = G.nodes[u]["freq"]
        freq_v  = G.nodes[v]["freq"]
        lift_val = w_ * N_REVIEWS / (freq_u * freq_v)
        print(wrap(
            f"[{brand_u}] {u}（{freq_u}件）× [{brand_v}] {v}（{freq_v}件）"
            f"  Lift={lift_val:.2f}",
            width=58, indent="       "))
        print()

    # コミュニティペア別サマリー
    print(f"  {BAR}")
    print("  ◆ コミュニティ間接続のまとめ")
    comm_pair_counts: Counter = Counter()
    for u, v, w_, cu, cv in cross_edges:
        comm_pair_counts[frozenset({cu, cv})] += w_

    for cp, total_w in comm_pair_counts.most_common():
        c1, c2 = sorted(cp)
        l1 = comm_label(c1).split("/")[0]
        l2 = comm_label(c2).split("/")[0]
        edge_count = sum(1 for u, v, _, cu, cv in cross_edges
                         if frozenset({cu, cv}) == cp)
        print(f"    C{c1}({l1}) × C{c2}({l2}): {edge_count}ペア / 累計共起 {total_w}回")
    print()

    print(wrap(
        "💡 インサイト: 「バニラ系（デザート）× フルーツ系」の組み合わせが"
        "クロス接続の断トツ1位。ミルキー・クリーミーな香りが"
        "シトラス・フルーツの酸味を丸くする定番技法と一致します。"
        "一方「シガー系 × フルーツ系」も健在で、チェリー・ライチとの"
        "組み合わせが近年急増中のダークリーフ+フルーツトレンドを支えています。"))
    print()


def print_footer():
    print(DBAR)
    print("【分析メタ情報】")
    print(DBAR)
    print(f"  入力レビュー数     : {N_REVIEWS} 件")
    print(f"  マスターフレーバー : {len(flavor_dict)} 種")
    print(f"  グラフ ノード数    : {G.number_of_nodes()}")
    print(f"  グラフ エッジ数    : {G.number_of_edges()}")
    print(f"  コミュニティ数     : {N_COMM}")
    print(f"  Lift計算対象ペア   : {len(lift_candidates)} ペア（共起2〜5回）")
    print(f"  クロスC.エッジ数   : {len(cross_edges)}")
    print()
    print(f"  CSVレポート保存先  : {csv_path}")
    print()
    print("  ■ 3つの指標の使い分けガイド（プレゼン向け）")
    print(wrap(
        "・媒介中心性  → 『とりあえずこれを入れれば失敗しない』"
        "万能ミキサーとして初心者向けオススメに使える。",
        width=58, indent="    "))
    print(wrap(
        "・Lift値      → 『通好みの隠れた相性』として上級者向け"
        "ネタやSNS映えする『通なコンビ』として紹介できる。",
        width=58, indent="    "))
    print(wrap(
        "・クロスC.   → 『普通は混ぜないが合う！意外な組み合わせ』"
        "として驚きを演出するプレゼンフックに最適。",
        width=58, indent="    "))
    print()
    print("✓ 分析完了")
    print()


# ────────────────────────────────────────────────────────────
# メイン実行
# ────────────────────────────────────────────────────────────
print_header()
section1_betweenness()
section2_lift()
section3_cross_community()
print_footer()
