#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シーシャ フレーバー推薦共起ネットワーク
=========================================
aslaj_master_list.csv をホワイトリスト辞書として活用し、
cloud_reviews_final.csv のレビュー本文から「同じ記事に登場したフレーバーペア」を
共起ネットワークとして可視化する。

実行方法:
  cd ~/datascience && python3 notebooks/flavor_mix_network.py
"""

import os
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# パス設定
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

REVIEWS_CSV = os.path.join(DATA_DIR, "cloud_reviews_final.csv")
MASTER_CSV  = os.path.join(DATA_DIR, "aslaj_master_list.csv")

# ============================================================
# ステップ 1: マスターリストからフレーバー辞書を構築
# ============================================================
print("[1/6] マスターリストからフレーバー辞書を構築中...")

master = pd.read_csv(MASTER_CSV)
reviews = pd.read_csv(REVIEWS_CSV)


def clean_flavor_entry(raw_name: str, raw_brand: str) -> tuple[str, str]:
    """
    フレーバー名列から「実際のフレーバー名」と「ブランド短縮名」を分離する。

    パターン:
      (A) "スペクトラ- MALAKI(マラキ)　　シーシャフレーバー 50g"
          → flavor="スペクトラ", brand="MALAKI"
      (B) "KASHMIR GOA JAVA(カシミールゴアジャバ)"
          → flavor="KASHMIR GOA JAVA(カシミールゴアジャバ)", brand=raw_brand
      (C) "グリーンティー"
          → flavor="グリーンティー", brand=raw_brand
    """
    name = str(raw_name).strip()

    # パターン A: "フレーバー名 - BRAND..." で区切られている
    sep_match = re.match(r'^(.+?)\s*[-–]\s*([A-Za-zァ-ヿ].+)', name)
    if sep_match:
        flavor_part = sep_match.group(1).strip()
        brand_part  = sep_match.group(2).strip()
        # brand_part から ブランド短縮名を抽出
        bm = re.match(r'([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})', brand_part)
        brand_short = bm.group(1) if bm else raw_brand[:10]
        return flavor_part, brand_short

    # パターン B / C: ブランドはそのまま raw_brand
    bm = re.match(r'([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})',
                  str(raw_brand).strip())
    brand_short = bm.group(1) if bm else "不明"
    return name, brand_short


def build_canonical_and_patterns(flavor_name: str, brand_short: str
                                  ) -> tuple[str, list[str]]:
    """
    クリーン済みフレーバー名から:
      - canonical : ネットワークノード名（短くて読みやすい）
      - patterns  : レビュー本文中で検索するキーワードリスト
    を返す。

    フレーバー名のパターン:
      (i)  "KASHMIR GOA JAVA(カシミールゴアジャバ)" → canonical=カシミールゴアジャバ
      (ii) "BARBERRY"（英語のみ）                  → canonical=BARBERRY
      (iii)"グリーンティー"（日本語のみ）            → canonical=グリーンティー
      (iv) "POMEGRANTE JUICE"（英語のみ）           → canonical=POMEGRANTE JUICE
    """
    name = flavor_name.strip()

    # カッコ内にカタカナ
    ja_in_paren = re.search(r'\(([ァ-ヿ][^\)]+)\)', name)
    # カッコ前の英語部分
    en_part = re.match(r'^([A-Za-z][A-Za-z0-9 .&\'\-/]+?)(?:\(|$)', name)

    if ja_in_paren:
        ja_str = ja_in_paren.group(1).strip()
        en_str = en_part.group(1).strip() if en_part else ""
        canonical = ja_str
        patterns = [ja_str]
        if en_str and len(en_str) >= 3:
            patterns.append(en_str)
    elif re.match(r'^[ァ-ヿぁ-んー]', name):
        # 純カタカナ/ひらがな
        canonical = name
        patterns = [name]
    elif re.match(r'^[A-Za-z]', name):
        # 純英語
        canonical = name.upper()
        patterns = [name]
    else:
        # 漢字まじりや混合
        canonical = name
        patterns = [name]

    return canonical, patterns


# ---- 辞書の組み立て ----
# canonical_name → {patterns: set, brand: str, raw_names: list}
flavor_dict: dict[str, dict] = {}
# pattern → canonical_name (検索用逆引き)
pattern_to_canonical: dict[str, str] = {}

for _, row in master.iterrows():
    flavor_clean, brand_short = clean_flavor_entry(row["フレーバー名"], row["ブランド"])
    if not flavor_clean or flavor_clean.strip() in ("nan", ""):
        continue

    canonical, patterns = build_canonical_and_patterns(flavor_clean, brand_short)

    # 短すぎる・意味のないパターンは除外（ノイズ防止）
    MIN_LEN = 3  # 3文字未満はスキップ
    patterns = [p for p in patterns if len(p) >= MIN_LEN]
    if not patterns:
        continue

    if canonical not in flavor_dict:
        flavor_dict[canonical] = {
            "patterns": set(),
            "brand": brand_short,
            "raw_names": [],
        }
    flavor_dict[canonical]["patterns"].update(patterns)
    flavor_dict[canonical]["raw_names"].append(flavor_clean)

    for p in patterns:
        if p not in pattern_to_canonical:
            pattern_to_canonical[p] = canonical

# 検索パターンを長い順にソート（貪欲最長マッチ用）
sorted_patterns = sorted(pattern_to_canonical.keys(), key=len, reverse=True)

print(f"  フレーバーcanonical数: {len(flavor_dict)}")
print(f"  検索パターン総数: {len(sorted_patterns)}")
print("  上位パターン例:", sorted_patterns[:5])

# ============================================================
# ステップ 2: 各レビューからフレーバー名を抽出
# ============================================================
print("[2/6] レビュー本文からフレーバーを抽出中...")


def extract_flavors(text: str, sorted_pats: list[str],
                    pat_to_canon: dict[str, str]) -> set[str]:
    """
    レビュー本文からマスターリスト登録フレーバーを貪欲最長マッチで抽出。
    一度マッチした文字範囲は再マッチしない（重複カウント防止）。
    """
    if not isinstance(text, str):
        return set()

    found: set[str] = set()
    # マッチ済み範囲をマスクするため位置セットを管理
    masked: set[int] = set()

    for pat in sorted_pats:
        # 大文字小文字を無視して検索
        search_text = text
        pat_upper = pat.upper()
        text_upper = text.upper()

        start = 0
        while True:
            idx = text_upper.find(pat_upper, start)
            if idx == -1:
                break
            end = idx + len(pat)

            # すでにマスク済みの位置を含む場合はスキップ
            if any(i in masked for i in range(idx, end)):
                start = idx + 1
                continue

            # マッチ確認：日本語（カタカナ）の場合、前後が同じカタカナ連続なら除外
            # （例：「ミント」が「ペパーミント」の一部にマッチするのを防ぐ）
            before_ok = True
            after_ok  = True
            if idx > 0:
                prev_char = text[idx - 1]
                if unicodedata.category(prev_char) in ("Lo", "Ll", "Lu", "Nd") \
                        and unicodedata.category(text[idx]) in ("Lo",):
                    before_ok = False
            if end < len(text):
                next_char = text[end]
                if unicodedata.category(next_char) in ("Lo",) \
                        and unicodedata.category(text[end - 1]) in ("Lo",):
                    after_ok = False

            if before_ok and after_ok:
                canonical = pat_to_canon[pat]
                found.add(canonical)
                masked.update(range(idx, end))

            start = idx + 1

    return found


# 各レビューのフレーバーリストを作成
review_flavors: list[set[str]] = []
for text in reviews["レビュー本文"]:
    flavors = extract_flavors(text, sorted_patterns, pattern_to_canonical)
    review_flavors.append(flavors)

# フレーバー出現回数集計（何件のレビューに登場したか）
flavor_doc_freq: Counter = Counter()
for fl_set in review_flavors:
    for fl in fl_set:
        flavor_doc_freq[fl] += 1

print(f"  フレーバーが1件以上マッチしたレビュー: "
      f"{sum(1 for s in review_flavors if s)} / {len(reviews)}")
print(f"  2フレーバー以上マッチしたレビュー: "
      f"{sum(1 for s in review_flavors if len(s) >= 2)}")
print(f"  出現フレーバー総ユニーク数: {len(flavor_doc_freq)}")
print("  頻出フレーバーTop20:")
for fl, cnt in flavor_doc_freq.most_common(20):
    print(f"    {fl:<30} {cnt} 件")

# ============================================================
# ステップ 3: 共起行列の構築
# ============================================================
print("[3/6] 共起行列を構築中...")

# 同一レビュー内のフレーバーペアを集計（各レビューの寄与は最大1）
cooccurrence: Counter = Counter()
for fl_set in review_flavors:
    fl_list = sorted(fl_set)  # ソートして (A,B) = (B,A) を統一
    for pair in combinations(fl_list, 2):
        cooccurrence[pair] += 1

print(f"  フレーバーペア総数: {len(cooccurrence)}")
print("  頻出ペアTop15:")
for (f1, f2), cnt in cooccurrence.most_common(15):
    print(f"    {f1} × {f2}: {cnt}")

# ============================================================
# ステップ 4: ネットワーク構築とフィルタリング
# ============================================================
print("[4/6] NetworkXグラフを構築中...")

# 足切り閾値の設定
# ノード: レビュー登場件数 >= MIN_NODE_FREQ
# エッジ: 共起回数         >= MIN_EDGE_WEIGHT
MIN_NODE_FREQ   = 3   # 最低3件のレビューに登場したフレーバーのみ
MIN_EDGE_WEIGHT = 2   # 最低2回共起したペアのみ
MAX_NODES       = 55  # 視認性のため最大ノード数を制限

# 閾値を満たすノード集合
valid_nodes = {fl for fl, cnt in flavor_doc_freq.items()
               if cnt >= MIN_NODE_FREQ}

# 出現頻度上位 MAX_NODES に制限
top_nodes = {fl for fl, _ in flavor_doc_freq.most_common(MAX_NODES)
             if fl in valid_nodes}

G = nx.Graph()

# ノードを追加（出現頻度を属性として付与）
for fl in top_nodes:
    G.add_node(fl, freq=flavor_doc_freq[fl])

# エッジを追加（両ノードが存在し、共起回数が閾値以上）
for (f1, f2), cnt in cooccurrence.items():
    if f1 in top_nodes and f2 in top_nodes and cnt >= MIN_EDGE_WEIGHT:
        G.add_edge(f1, f2, weight=cnt)

# 孤立ノード（エッジを持たないノード）を除去
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

print(f"  グラフ: {G.number_of_nodes()} ノード, {G.number_of_edges()} エッジ")
print(f"  削除した孤立ノード: {len(isolates)} 個")

# ============================================================
# ステップ 5: Louvainコミュニティ検出
# ============================================================
print("[5/6] Louvainコミュニティ検出中...")

partition = community_louvain.best_partition(G, weight="weight", random_state=42)
n_communities = max(partition.values()) + 1
print(f"  検出コミュニティ数: {n_communities}")

# コミュニティごとの代表フレーバー（頻度Top3）
COMMUNITY_COLORS = [
    "#E74C3C",  # 赤
    "#3498DB",  # 青
    "#2ECC71",  # 緑
    "#F39C12",  # オレンジ
    "#9B59B6",  # 紫
    "#1ABC9C",  # ティール
    "#E67E22",  # ダークオレンジ
    "#E91E63",  # ピンク
    "#00BCD4",  # シアン
    "#795548",  # ブラウン
    "#607D8B",  # ブルーグレー
    "#CDDC39",  # ライムグリーン
]

comm_members: dict[int, list] = defaultdict(list)
for node, comm_id in partition.items():
    comm_members[comm_id].append(node)


COMMUNITY_PROFILES = [
    ("シガーリーフ系 × スパイス・ウッド系", {
        "シガー", "チェリー", "マンゴー", "ライチ", "パッションフルーツ",
        "ラズベリー", "ザクロ", "LYCHEE", "MANGO",
    }),
    ("南国フルーツ系 × 甘味系", {
        "ミルク", "ハニー", "バナナ", "キャラメル", "ストロベリー",
        "メロン", "ココナッツ", "モヒート", "カルダモンミルク",
        "ウォーターメロン/スイカ",
    }),
    ("ミント・清涼系 × フルーツ系", {
        "ミント", "レモン", "オレンジ", "ブルーベリー", "ベリー",
        "GRAPE", "グアバ", "ライム", "グレープフルーツ", "ピーチ",
        "パイナップル", "ジャスミン", "キウイ", "アイス", "ツーアップル",
        "COLA", "アサイー", "チョコレート", "タンジェリン", "ブルーヘブン",
        "グレープ",
    }),
    ("スイーツ系 × フローラル系", {
        "バニラ", "アールグレイ", "ゴールデンデリシャスアップル", "ローズ",
        "ジンジャー", "バター", "コニャック", "ミントクリーム",
        "EARL GREY", "カプチーノ", "CHOCOLATE", "ペア",
    }),
]


def relabel_partition_stably(
    partition_map: dict[str, int],
    graph: nx.Graph,
) -> tuple[dict[str, int], dict[int, str]]:
    """コミュニティIDを意味ベースで安定化し、再実行で色が変わりにくくする。"""
    raw_members: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition_map.items():
        raw_members[cid].append(node)

    scored = []
    for cid, members in raw_members.items():
        member_set = set(members)
        top_member = sorted(
            members, key=lambda n: graph.nodes[n]["freq"], reverse=True
        )[0]
        best_idx = len(COMMUNITY_PROFILES)
        best_score = -1
        best_label = None
        for idx, (label, keywords) in enumerate(COMMUNITY_PROFILES):
            score = len(member_set & keywords)
            if score > best_score:
                best_idx = idx
                best_score = score
                best_label = label
        scored.append((best_idx, -best_score, top_member, cid, best_label))

    scored.sort()
    old_to_new = {old_cid: new_cid for new_cid, (_, _, _, old_cid, _) in enumerate(scored)}
    stable_labels = {
        new_cid: label if label is not None else f"C{new_cid}"
        for new_cid, (_, _, _, _, label) in enumerate(scored)
    }
    remapped = {node: old_to_new[cid] for node, cid in partition_map.items()}
    return remapped, stable_labels


partition, community_labels = relabel_partition_stably(partition, G)
comm_members = defaultdict(list)
for node, comm_id in partition.items():
    comm_members[comm_id].append(node)

for comm_id in sorted(comm_members.keys()):
    members = sorted(comm_members[comm_id],
                     key=lambda n: G.nodes[n]["freq"], reverse=True)
    top3 = " / ".join(members[:3])
    label = community_labels.get(comm_id, top3)
    color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
    print(f"  C{comm_id} [{color}] ({len(members)}ノード): {label} :: {top3}")

# ============================================================
# ステップ 6: ネットワーク可視化
# ============================================================
print("[6/6] ネットワーク図を描画中...")

fig, ax = plt.subplots(figsize=(22, 18))
fig.patch.set_facecolor("#1C1C2E")
ax.set_facecolor("#1C1C2E")

# レイアウト計算（spring layout with tuned parameters）
pos = nx.spring_layout(
    G,
    weight="weight",
    k=3.5,
    iterations=200,
    seed=2025,
)

# ---- ノードサイズ（出現頻度に比例）----
freq_values = [G.nodes[n]["freq"] for n in G.nodes()]
max_freq = max(freq_values) if freq_values else 1
min_freq = min(freq_values) if freq_values else 1

node_sizes = []
node_colors = []
for node in G.nodes():
    freq = G.nodes[node]["freq"]
    # サイズ: 最小300〜最大2500
    size = 300 + 2200 * ((freq - min_freq) / max(max_freq - min_freq, 1)) ** 0.6
    node_sizes.append(size)
    comm_id = partition[node]
    node_colors.append(COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)])

# ---- エッジ（共起頻度に比例した太さ・透明度）----
edges     = list(G.edges(data=True))
weights   = [d["weight"] for _, _, d in edges]
max_w     = max(weights) if weights else 1

for (u, v, d) in edges:
    w         = d["weight"]
    thickness = 0.5 + 8.0 * (w / max_w) ** 0.7
    alpha     = 0.25 + 0.55 * (w / max_w) ** 0.5
    # エッジの色: 両ノードが同コミュニティなら濃い色、跨るなら薄いグレー
    if partition[u] == partition[v]:
        comm_id = partition[u]
        ec = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        ea = alpha
    else:
        ec = "#AAAAAA"
        ea = 0.15

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v)],
        width=thickness,
        alpha=ea,
        edge_color=ec,
    )

# ---- ノード描画 ----
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.88,
    linewidths=1.5,
    edgecolors="white",
)

# ---- ノードラベル（アウトライン付きで可読性向上）----
for node, (x, y) in pos.items():
    freq = G.nodes[node]["freq"]
    # 高頻度ノードは大きめのフォント
    font_size = 7 + min(5, freq // 10)
    txt = ax.text(
        x, y + 0.025, node,
        ha="center", va="bottom",
        fontsize=font_size,
        fontfamily="TakaoGothic",
        color="white",
        fontweight="bold",
        zorder=10,
    )
    txt.set_path_effects([
        pe.withStroke(linewidth=2.5, foreground="#1C1C2E"),
    ])

# ---- 凡例（コミュニティ）----
legend_handles = []
for comm_id in sorted(comm_members.keys()):
    members = sorted(comm_members[comm_id],
                     key=lambda n: G.nodes[n]["freq"], reverse=True)
    label  = community_labels.get(comm_id, " / ".join(members[:2]))
    color  = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
    n_mem  = len(members)
    patch  = mpatches.Patch(
        color=color,
        label=f"C{comm_id} ({n_mem}): {label}",
    )
    legend_handles.append(patch)

ax.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=12,
    framealpha=0.75,
    facecolor="#2C2C3E",
    edgecolor="#555555",
    labelcolor="white",
    title="コミュニティ（Louvain法）",
    title_fontsize=13,
)

# ---- ノードサイズ凡例（右上）----
size_legend_items = []
for label, freq in [("低頻度 (3-5件)", 4), ("中頻度 (10件)", 10), ("高頻度 (30件+)", 35)]:
    sz = 300 + 2200 * ((freq - min_freq) / max(max_freq - min_freq, 1)) ** 0.6
    h = ax.scatter([], [], s=sz, color="#888888", alpha=0.7, label=label, edgecolors="white")
    size_legend_items.append(h)

size_legend = ax.legend(
    handles=size_legend_items,
    loc="upper right",
    fontsize=10,
    framealpha=0.75,
    facecolor="#2C2C3E",
    edgecolor="#555555",
    labelcolor="white",
    title="ノードサイズ（登場件数）",
    title_fontsize=11,
    scatterpoints=1,
)
ax.add_artist(size_legend)
# コミュニティ凡例を再描画（add_artist後に必要）
ax.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=12,
    framealpha=0.75,
    facecolor="#2C2C3E",
    edgecolor="#555555",
    labelcolor="white",
    title="コミュニティ（Louvain法）",
    title_fontsize=13,
)

# ---- タイトル・注釈 ----
ax.set_title(
    "シーシャ フレーバー推薦 共起ネットワーク\n"
    "（aslaj_master_list.csv ホワイトリスト準拠 ／ Louvain法コミュニティ検出）",
    fontsize=16,
    fontweight="bold",
    color="white",
    pad=14,
)
ax.text(
    0.01, 0.01,
    f"ノード数: {G.number_of_nodes()}  ／  エッジ数: {G.number_of_edges()}  ／  "
    f"足切り: 登場>={MIN_NODE_FREQ}件, 共起>={MIN_EDGE_WEIGHT}件",
    transform=ax.transAxes,
    fontsize=8,
    color="#AAAAAA",
    va="bottom",
)
ax.axis("off")

plt.tight_layout(pad=1.5)

out_path = os.path.join(OUT_DIR, "flavor_mix_network.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  保存完了: {out_path}")

# ============================================================
# サマリーレポート
# ============================================================
print("\n" + "=" * 60)
print("■ 分析サマリー")
print("=" * 60)
print(f"\n◆ 入力データ")
print(f"  レビュー件数      : {len(reviews)}")
print(f"  マスターフレーバー : {len(flavor_dict)} 種")
print(f"  検索パターン       : {len(sorted_patterns)} 種")

print(f"\n◆ 抽出結果")
print(f"  ヒットしたフレーバー: {len(flavor_doc_freq)} 種")
print(f"  有効フレーバーペア  : {len(cooccurrence)} ペア")

print(f"\n◆ グラフ統計")
print(f"  ノード数   : {G.number_of_nodes()}")
print(f"  エッジ数   : {G.number_of_edges()}")
print(f"  平均次数   : {np.mean([d for _, d in G.degree()]):.2f}")
print(f"  密度       : {nx.density(G):.4f}")

print(f"\n◆ 頻出フレーバー Top10")
for rank, (fl, cnt) in enumerate(flavor_doc_freq.most_common(10), 1):
    comm_id = partition.get(fl, -1)
    brand   = flavor_dict.get(fl, {}).get("brand", "?")
    print(f"  {rank:2d}. {fl:<25} {cnt:3d}件  C{comm_id}  [{brand}]")

print(f"\n◆ 最強ミックスペア Top10")
for rank, ((f1, f2), cnt) in enumerate(cooccurrence.most_common(10), 1):
    print(f"  {rank:2d}. {f1} × {f2}: {cnt}件")

print(f"\n◆ コミュニティ詳細")
for comm_id in sorted(comm_members.keys()):
    members = sorted(comm_members[comm_id],
                     key=lambda n: G.nodes[n]["freq"], reverse=True)
    print(f"  C{comm_id} ({len(members)}ノード): {', '.join(members)}")

print(f"\n◆ 出力ファイル")
fsize = os.path.getsize(out_path) / 1024
print(f"  {out_path} ({fsize:.0f} KB)")
print("\n✓ 完了")
