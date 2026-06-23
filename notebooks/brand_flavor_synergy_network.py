#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シーシャ ブランド×フレーバー シナジーネットワーク
================================================
「BRAND_フレーバー」混成ノードによる多種ミックス相関可視化。

データソース:
  [S1] aslaj_master_list.csv  ── 単独ブランドフレーバー（272種）
  [S2] ranking table parser    ── レビュー本文のランキング表から
                                  非マスターリストブランド（MUSTHAVE, BONCHE, etc.）を抽出

実行方法:
  cd ~/datascience && python3 notebooks/brand_flavor_synergy_network.py
"""

import os, re, csv, unicodedata, textwrap
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

# ─── パス ────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

REVIEWS_CSV = os.path.join(DATA, "cloud_reviews_final.csv")
MASTER_CSV  = os.path.join(DATA, "aslaj_master_list.csv")
OUT_PNG     = os.path.join(OUT, "brand_flavor_synergy_network.png")
OUT_TXT     = os.path.join(OUT, "brand_mix_insight.txt")

# ─── ブランド分類（ライトリーフ/ダークリーフ/ブースター）──────────
DARK_BRANDS = {
    "MUSTHAVE", "BONCHE", "Dogma", "Tangiers", "Severniy", "Deus",
    "Satyr", "Trofimoff", "DARKSIDE",
}
BOOSTER_KEYWORDS = {"スーパーノヴァ", "SUPERNOVA", "Supernova", "ケーンミント"}

# ─── ① マスターリストからブランド_フレーバー辞書を構築 ──────────
print("[1/7] マスターリストから辞書構築中...")

master  = pd.read_csv(MASTER_CSV)
reviews = pd.read_csv(REVIEWS_CSV)
N = len(reviews)


def clean_brand(raw: str) -> str:
    """ブランド列から純粋なブランド名を抽出。"""
    b = str(raw).strip()
    # (カタカナ/漢字/ひらがな) 読み仮名を除去（前後のスペースごと）
    b = re.sub(r"\s*[\(（][ァ-ヿぁ-ん一-龥\s　]+[)）]\s*", " ", b).strip()
    # 「シーシャフレーバー」「50g」等の商品説明サフィックスで分割
    b = re.split(r"[\s　]*(シーシャ|フレーバー|\d+[gｇ])", b)[0].strip()
    return b


def parse_flavor(raw_name: str) -> tuple[str, list[str]]:
    """
    フレーバー名列から (canonical, search_patterns) を返す。
    複合フォーマット "フレーバー名 - BRAND(...)" は flavor 部のみ取り出す。
    """
    name = str(raw_name).strip()
    # "フレーバー - BRAND(カナ)" パターン
    sep = re.match(r"^(.+?)\s*[-–]\s*([A-Za-zァ-ヿ].+)", name)
    if sep:
        name = sep.group(1).strip()

    ja = re.search(r"\(([ァ-ヿ][^\)]+)\)", name)
    en = re.match(r"^([A-Za-z][A-Za-z0-9 .&'\-/]+?)(?:\(|$)", name)

    if ja:
        canonical = ja.group(1).strip()
        pats = [canonical]
        if en:
            pats.append(en.group(1).strip())
    elif re.match(r"^[ァ-ヿぁ-んー]", name):
        canonical = name
        pats = [name]
    elif re.match(r"^[A-Za-z]", name):
        canonical = name.upper()
        pats = [name]
    else:
        canonical = name
        pats = [name]

    return canonical, [p for p in pats if len(p) >= 3]


# brand_flavor → {patterns, brand, leaf_type, is_booster}
bf_dict: dict[str, dict] = {}
# pattern → brand_flavor node name
p2bf: dict[str, str] = {}
# canonical flavor → list of brand_flavor nodes（曖昧性管理）
flavor_to_nodes: dict[str, list[str]] = defaultdict(list)

for _, row in master.iterrows():
    brand = clean_brand(row["ブランド"])
    if brand in ("不明", ""):
        # 埋め込みブランドを再取得
        sep = re.match(r"^(.+?)\s*[-–]\s*([A-Za-z][A-Za-z0-9 ]+?(?:[\(（]|$))", str(row["フレーバー名"]))
        if sep:
            embedded = re.match(r"([A-Za-z][A-Za-z0-9 ]+)", sep.group(2))
            brand = embedded.group(1).strip() if embedded else "UNKNOWN"
        else:
            brand = "UNKNOWN"

    canonical, pats = parse_flavor(row["フレーバー名"])
    if not pats:
        continue

    # ブランド_フレーバー ノード名（長すぎる場合は短縮）
    brand_short = brand[:14].rstrip()
    node = f"{brand_short}_{canonical}"
    if len(node) > 34:
        node = f"{brand_short}_{canonical[:18]}"

    leaf_type = "dark" if any(b in brand for b in DARK_BRANDS) else "light"
    is_booster = any(k in canonical for k in BOOSTER_KEYWORDS)

    if node not in bf_dict:
        bf_dict[node] = {
            "patterns": set(), "brand": brand_short,
            "canonical_flavor": canonical,
            "leaf_type": leaf_type, "is_booster": is_booster,
        }
    bf_dict[node]["patterns"].update(pats)
    flavor_to_nodes[canonical].append(node)

    for p in pats:
        p2bf.setdefault(p, node)

# ─── ② ランキングテーブルから非マスターリストブランドを追加 ─────
print("[2/7] ランキングテーブルから追加ブランドを抽出中...")

KNOWN_EXTRA_BRANDS = {
    "MUSTHAVE", "Dogma", "Tangiers", "Severniy", "Deus",
    "Satyr", "Trofimoff", "BONCHE", "Sarma", "Fumari",
    "Must", "Darkside",
}


def extract_table_pairs(text: str) -> list[tuple[str, str]]:
    """
    レビュー本文のランキングテーブルから (brand, japanese_flavor) ペアを抽出。
    テーブルフォーマット:
      [数字]\\n[ブランド名のみ]\\n[英語フレーバー (カタカナ読み)]\\n...
    ※ l1 が説明文（「を使った」等含む）の場合は記事本文と判断してスキップ。
    ※ l2 の括弧内がブランド読み（「とは」「の特徴」等後続）の場合もスキップ。
    """
    if not isinstance(text, str):
        return []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    pairs = []
    for i in range(len(lines) - 2):
        l0, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        # 数字行
        if not re.match(r"^\d{1,2}$", l0):
            continue
        # ブランド行：英字で始まる、説明文でない（「を使った」「のおすすめ」等を含まない）
        if not re.match(r"^[A-Za-z]", l1):
            continue
        if re.search(r"を使った|のおすすめ|のMIX|とは|の特徴|–|—|-\s", l1):
            continue
        # ブランド行が単語1〜3個（長い説明文は除外）
        if len(l1.split()) > 3:
            continue
        # フレーバー行：括弧内にカタカナ読みがある
        ja_match = re.search(r"[（(]([ァ-ヿ]{2,})[)）]", l2)
        if not ja_match:
            continue
        # フレーバー行が記事見出しでないことを確認
        if re.search(r"とは|の特徴|を使った|のおすすめ", l2):
            continue

        # 括弧内カタカナをカノニカルフレーバー名として使用（マスターリストと統一）
        flavor_part = ja_match.group(1)

        # ブランド行がブランド名以外の単語（フレーバー名など）でないか確認
        # → 既知ブランドリストに一致するか確認
        brand_raw = l1.strip()
        brand_lower = brand_raw.lower()
        is_known = any(k.lower() in brand_lower or brand_lower in k.lower()
                       for k in KNOWN_EXTRA_BRANDS | {"DARKSIDE", "DOZAJ", "Dozaj"})
        if not is_known:
            # 全大文字 2文字以上（例: BONCHE, SATYR）も許可
            if not re.match(r"^[A-Z]{3,}", brand_raw):
                continue

        pairs.append((brand_raw, flavor_part))

    return pairs


# ランキングから抽出した品を辞書に追加
extra_added = 0
for _, row in reviews.iterrows():
    pairs = extract_table_pairs(row["レビュー本文"])
    for brand_raw, flavor_ja in pairs:
        brand_short = brand_raw[:14].rstrip()
        node = f"{brand_short}_{flavor_ja}"
        if len(node) > 34:
            node = f"{brand_short}_{flavor_ja[:18]}"

        leaf_type = "dark" if any(b.lower() in brand_short.lower() for b in DARK_BRANDS) else "light"
        is_booster = any(k in flavor_ja for k in BOOSTER_KEYWORDS)

        if node not in bf_dict:
            bf_dict[node] = {
                "patterns": set(), "brand": brand_short,
                "canonical_flavor": flavor_ja,
                "leaf_type": leaf_type, "is_booster": is_booster,
            }
        bf_dict[node]["patterns"].add(flavor_ja)
        bf_dict[node]["is_booster"] = is_booster
        p2bf.setdefault(flavor_ja, node)
        extra_added += 1

print(f"  テーブル抽出ペア追加数: {extra_added}")
print(f"  総ブランド_フレーバーノード候補数: {len(bf_dict)}")

# 検索パターンを長い順でソート（貪欲最長マッチ）
sorted_pats = sorted(p2bf, key=len, reverse=True)

# ─── ③ 各レビューからブランド_フレーバー抽出 ────────────────
print("[3/7] レビュー本文からブランド_フレーバーを抽出中...")


def extract_bf_tokens(text: str) -> frozenset[str]:
    """テキストからブランド_フレーバートークンを抽出（貪欲最長マッチ）。"""
    if not isinstance(text, str):
        return frozenset()
    found: set[str] = set()
    masked: set[int] = set()
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
            bef = (idx == 0 or cat(text[idx - 1]) not in ("Lo",)
                   or cat(text[idx]) not in ("Lo",))
            aft = (end >= len(text) or cat(text[end]) not in ("Lo",)
                   or cat(text[end - 1]) not in ("Lo",))
            if bef and aft:
                found.add(p2bf[pat])
                masked.update(range(idx, end))
            start = idx + 1

    # 同一フレーバーが複数ブランドにあった場合、テキスト内のブランド名で絞り込み
    refined: set[str] = set()
    text_upper = text.upper()
    for node in found:
        brand = bf_dict[node]["brand"].upper()
        # ブランド名がテキストに明示されているか確認
        if len(brand) >= 4 and brand in text_upper:
            refined.add(node)
        else:
            # ブランド名が不明確でも「ユニークフレーバー」なら採用
            flavor = bf_dict[node]["canonical_flavor"]
            if len(flavor_to_nodes.get(flavor, [])) <= 1:
                refined.add(node)

    return frozenset(refined)


# テーブルパーサーも活用してトークンを強化
transactions: list[frozenset[str]] = []
titles_list: list[str] = []

for _, row in reviews.iterrows():
    text  = row["レビュー本文"]
    title = str(row.get("レビュータイトル", ""))

    # パターン1: テキスト全体から抽出
    tokens = set(extract_bf_tokens(text))

    # パターン2: テーブルパーサーで補強（ブランド名が明示されているケース）
    tbl_pairs = extract_table_pairs(text)
    for brand_raw, flavor_ja in tbl_pairs:
        brand_short = brand_raw[:12]
        node = f"{brand_short}_{flavor_ja}"
        if len(node) > 32:
            node = f"{brand_short}_{flavor_ja[:18]}"
        if node in bf_dict:
            tokens.add(node)

    transactions.append(frozenset(tokens))
    titles_list.append(title)

# 頻度集計
bf_freq = Counter(nd for tx in transactions for nd in tx)
# 3種以上のトランザクションのみ
tx3plus = [tx for tx in transactions if len(tx) >= 3]

print(f"  ユニークブランド_フレーバーノード（出現>0）: {len(bf_freq)}")
print(f"  3種以上のトランザクション: {len(tx3plus)} / {N} 件")
print("  頻出ノードTop10:")
for nd, cnt in bf_freq.most_common(10):
    lt = bf_dict[nd]["leaf_type"]
    print(f"    {nd:<35} {cnt:>3}件 [{lt}]")

# ─── ④ 共起グラフ構築 ──────────────────────────────────────
print("[4/7] 共起グラフ構築中...")

MIN_FREQ   = 2   # 最低2件のレビューに登場したノードのみ
MIN_COOC   = 2   # 最低2回共起したペアのみ
MAX_NODES  = 60  # 視認性確保のための最大ノード数

valid_nodes = {nd for nd, cnt in bf_freq.items() if cnt >= MIN_FREQ}
top_nodes   = {nd for nd, _ in bf_freq.most_common(MAX_NODES) if nd in valid_nodes}

G = nx.Graph()
for nd in top_nodes:
    G.add_node(nd, freq=bf_freq[nd], **bf_dict[nd])

cooc: Counter = Counter()
for tx in transactions:
    filtered = sorted(tx & top_nodes)
    for pair in combinations(filtered, 2):
        cooc[pair] += 1

for (a, b), cnt in cooc.items():
    if cnt >= MIN_COOC:
        G.add_edge(a, b, weight=cnt)

G.remove_nodes_from(list(nx.isolates(G)))

print(f"  グラフ: {G.number_of_nodes()} ノード, {G.number_of_edges()} エッジ")

# ─── ⑤ Louvainコミュニティ検出 ──────────────────────────────
print("[5/7] Louvainコミュニティ検出中...")

partition = community_louvain.best_partition(G, weight="weight", random_state=42)
N_COMM = max(partition.values()) + 1
comm_members: dict[int, list] = defaultdict(list)
for nd, cid in partition.items():
    comm_members[cid].append(nd)

COMM_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#E91E63",
    "#00BCD4", "#795548", "#607D8B", "#FF5722",
]

print(f"  コミュニティ数: {N_COMM}")
for cid in sorted(comm_members):
    mems = sorted(comm_members[cid], key=lambda n: bf_freq[n], reverse=True)
    dark_cnt = sum(1 for m in mems if bf_dict[m]["leaf_type"] == "dark")
    print(f"  C{cid} ({len(mems)}ノード, dark={dark_cnt}): "
          f"{', '.join(m for m in mems[:3])} ...")

# ─── ⑥ ネットワーク可視化 ──────────────────────────────────
print("[6/7] ネットワーク図を描画中...")

fig, ax = plt.subplots(figsize=(24, 20))
fig.patch.set_facecolor("#12121E")
ax.set_facecolor("#12121E")

# レイアウト
pos = nx.spring_layout(G, weight="weight", k=3.2, iterations=200, seed=42)

# ── ノードサイズ
max_freq = max(bf_freq.get(n, 1) for n in G.nodes())
min_freq = min(bf_freq.get(n, 1) for n in G.nodes())

def node_size(nd):
    f = bf_freq.get(nd, 1)
    return 280 + 2400 * ((f - min_freq) / max(max_freq - min_freq, 1)) ** 0.6

# ── エッジ描画（コミュニティ内/クロスで色分け）
for (u, v, d) in G.edges(data=True):
    w = d["weight"]
    thickness = 0.4 + 9.0 * (w / max(cooc.values())) ** 0.6
    if partition[u] == partition[v]:
        cid = partition[u]
        ec  = COMM_COLORS[cid % len(COMM_COLORS)]
        ea  = 0.25 + 0.55 * (w / max(cooc.values())) ** 0.4
    else:
        ec = "#CCCCCC"
        ea = 0.12
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                           width=thickness, alpha=ea, edge_color=ec)

# ── ノード描画（リーフタイプでボーダー色を変える）
for nd in G.nodes():
    lt  = G.nodes[nd]["leaf_type"]
    bst = G.nodes[nd]["is_booster"]
    cid = partition[nd]
    fill_color   = COMM_COLORS[cid % len(COMM_COLORS)]
    border_color = ("#FFD700" if bst else     # ブースター = 金
                    "#FF6B35" if lt == "dark"  # ダークリーフ = オレンジ赤
                    else "#87CEEB")            # ライトリーフ = 空色

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[nd],
                           node_size=node_size(nd),
                           node_color=fill_color,
                           alpha=0.88,
                           linewidths=3.5 if (lt == "dark" or bst) else 1.5,
                           edgecolors=border_color)

# ── ラベル（ノードの上に2行で表示）
for nd, (x, y) in pos.items():
    brand = bf_dict[nd]["brand"]
    flavor = bf_dict[nd]["canonical_flavor"]
    # ラベルを折り返し
    b_short = brand[:10]
    f_short = flavor[:12]
    label = f"{b_short}\n{f_short}"

    freq = bf_freq.get(nd, 1)
    fsize = 6.5 + min(3.5, freq / 10)
    is_b = bf_dict[nd]["is_booster"]
    txt = ax.text(x, y + 0.028, label,
                  ha="center", va="bottom",
                  fontsize=fsize, fontfamily="TakaoGothic",
                  color="#FFD700" if is_b else "white",
                  fontweight="bold" if is_b else "normal",
                  linespacing=1.1, zorder=10)
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#12121E")])

# ── コミュニティ凡例
comm_handles = []
for cid in sorted(comm_members):
    mems = sorted(comm_members[cid], key=lambda n: bf_freq[n], reverse=True)
    dark_cnt = sum(1 for m in mems if bf_dict[m]["leaf_type"] == "dark")
    tag = "🔴ダーク" if dark_cnt > len(mems) / 2 else "🔵ライト"
    top2 = " / ".join(m.split("_")[1][:6] for m in mems[:2])
    comm_handles.append(mpatches.Patch(
        color=COMM_COLORS[cid % len(COMM_COLORS)],
        label=f"C{cid} {tag}({len(mems)}) {top2}…"
    ))

ax.legend(handles=comm_handles, loc="lower left",
          fontsize=8.5, framealpha=0.75, facecolor="#1E1E30",
          edgecolor="#555555", labelcolor="white",
          title="コミュニティ（Louvain法）", title_fontsize=9)

# ── ボーダー凡例（リーフタイプ）
border_handles = [
    mpatches.Patch(color="#FFD700", label="★ ブースター（Supernova等）"),
    mpatches.Patch(color="#FF6B35", label="■ ダークリーフ/シガー系ブランド"),
    mpatches.Patch(color="#87CEEB", label="■ ライトリーフ/フルーツ系ブランド"),
]
border_legend = ax.legend(handles=border_handles, loc="lower right",
                           fontsize=8.5, framealpha=0.75, facecolor="#1E1E30",
                           edgecolor="#555555", labelcolor="white",
                           title="ノードボーダー（葉タイプ）", title_fontsize=9)
ax.add_artist(comm_handles and ax.get_legend() or border_legend)
ax.legend(handles=comm_handles, loc="lower left",
          fontsize=8.5, framealpha=0.75, facecolor="#1E1E30",
          edgecolor="#555555", labelcolor="white",
          title="コミュニティ（Louvain法）", title_fontsize=9)
ax.add_artist(border_legend)

ax.set_title(
    "シーシャ ブランド×フレーバー シナジーネットワーク\n"
    "（ノードボーダー: 橙=ダークリーフ / 水色=ライトリーフ / 金=ブースター）",
    fontsize=15, fontweight="bold", color="white", pad=14
)
ax.text(0.01, 0.01,
        f"ノード:{G.number_of_nodes()} / エッジ:{G.number_of_edges()} / "
        f"3種以上トランザクション:{len(tx3plus)}件 / min共起:{MIN_COOC}回",
        transform=ax.transAxes, fontsize=8, color="#AAAAAA", va="bottom")
ax.axis("off")

plt.tight_layout(pad=1.5)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  保存: {OUT_PNG}")

# ─── ⑦ インサイトレポート ────────────────────────────────────
print("[7/7] インサイトレポートを生成中...")

DBAR = "═" * 66
BAR  = "─" * 66

report_lines: list[str] = []

def rp(*args):
    t = " ".join(str(a) for a in args)
    print(t); report_lines.append(t)

def rpw(text, w=64, ind="  "):
    t = textwrap.fill(text, width=w, initial_indent=ind, subsequent_indent=ind)
    print(t); report_lines.append(t)

# ── 3種以上ミックスのLift計算
def compute_multi_lift(itemset: frozenset, txns: list, freq: Counter, n: int) -> float:
    support = sum(1 for tx in txns if itemset <= tx)
    if support == 0:
        return 0.0
    p_joint = support / n
    p_indep = 1.0
    for item in itemset:
        p_indep *= freq[item] / n
    return p_joint / p_indep if p_indep > 0 else 0.0


# 3種以上の全組み合わせを列挙しTop5を抽出
print("  3-item頻出セット計算中...")
k3_counter: Counter = Counter()
for tx in tx3plus:
    filtered = sorted(tx & top_nodes)
    for combo in combinations(filtered, 3):
        k3_counter[frozenset(combo)] += 1

k3_with_lift = []
for itemset, support in k3_counter.items():
    if support < 2:
        continue
    lift = compute_multi_lift(itemset, transactions, bf_freq, N)
    k3_with_lift.append((itemset, support, lift))

k3_with_lift.sort(key=lambda x: -x[2])

# 4種以上
k4_counter: Counter = Counter()
for tx in tx3plus:
    filtered = sorted(tx & top_nodes)
    if len(filtered) >= 4:
        for combo in combinations(filtered, 4):
            k4_counter[frozenset(combo)] += 1

k4_with_lift = []
for itemset, support in k4_counter.items():
    if support < 2:
        continue
    lift = compute_multi_lift(itemset, transactions, bf_freq, N)
    k4_with_lift.append((itemset, support, lift))

k4_with_lift.sort(key=lambda x: -x[2])

# ── レポート出力
rp()
rp("╔" + "═" * 64 + "╗")
rp("║" + " シーシャ ブランド×フレーバー シナジー インサイト ".center(64) + "║")
rp("║" + f" レビュー {N}件 ／ ノード {G.number_of_nodes()} ／ 3種以上TXN {len(tx3plus)}件 ".center(64) + "║")
rp("╚" + "═" * 64 + "╝")
rp()

# コミュニティ構成
rp("  ■ 検出コミュニティ（ライトリーフ/ダークリーフ 派閥）")
rp("  " + BAR[:62])
for cid in sorted(comm_members):
    mems = sorted(comm_members[cid], key=lambda n: bf_freq[n], reverse=True)
    dark_cnt  = sum(1 for m in mems if bf_dict[m]["leaf_type"] == "dark")
    light_cnt = len(mems) - dark_cnt
    tag = "ダーク系優勢🔴" if dark_cnt > light_cnt else "ライト系優勢🔵"
    rp(f"  C{cid} [{tag}] ({len(mems)}ノード):")
    for m in mems[:5]:
        lt = bf_dict[m]["leaf_type"]
        bst = "★" if bf_dict[m]["is_booster"] else " "
        rp(f"    {bst} {m:<38} {bf_freq[m]:>3}件")
    if len(mems) > 5:
        rp(f"    ... 他 {len(mems)-5} ノード")
rp()

MEDAL5 = ["🥇", "🥈", "🥉", "  4.", "  5."]

# ── Top5 3種ミックスレシピ（Lift順）
rp(DBAR)
rp("【Top 5 職人レシピ ― 3種ミックス（Lift値順）】")
rp(DBAR)
rp()
rpw("Lift = 実際共起確率 / ランダム期待値。Lift が高いほど「偶然ではない"
    "意図的な組み合わせ」。同一コミュニティ内より異コミュニティ横断レシピに"
    "高いLiftが出る傾向があり、ブランドの壁を越えたシナジーを示す。")
rp()

for rank, (itemset, support, lift) in enumerate(k3_with_lift[:5], 1):
    items = sorted(itemset, key=lambda n: bf_freq[n], reverse=True)
    comms = {partition.get(n) for n in items if n in partition}
    cross  = "★異コミュニティ" if len(comms) > 1 else f"C{list(comms)[0]}内"
    lt_mix = " + ".join(
        ("ダーク" if bf_dict[n]["leaf_type"]=="dark" else "ライト") for n in items
    )
    rp(f"  {MEDAL5[rank-1]}  Lift={lift:.2f}  Support={support}件/{N}件"
       f"  [{cross}]")
    for nd in items:
        b  = bf_dict[nd]["brand"]
        fl = bf_dict[nd]["canonical_flavor"]
        lt = "ダーク" if bf_dict[nd]["leaf_type"]=="dark" else "ライト"
        bst= "（ブースター）" if bf_dict[nd]["is_booster"] else ""
        rp(f"       ・{nd:<38} [{lt}]{bst}  {bf_freq[nd]}件")
    rpw(f"葉タイプ構成: {lt_mix}", ind="       ")

    # 各アイテムの個別確率とジョイント確率
    p_ind = 1.0
    for nd in items:
        p_ind *= bf_freq[nd] / N
    p_actual = support / N
    rp(f"       独立仮定期待共起: {p_ind*N:.2f}件 → 実際: {support}件 → "
       f"Lift={p_actual/p_ind:.2f}x")

    # 代表レビュー
    for i, (tx, title) in enumerate(zip(transactions, titles_list)):
        if itemset <= tx and title:
            rp(f"       代表: 「{title[:50]}」")
            break
    rp()

# ── Top 3 4種以上ミックス
rp(DBAR)
rp("【Top 3 職人レシピ ― 4種ミックス（Lift値順）】")
rp(DBAR)
rp()

if k4_with_lift:
    for rank, (itemset, support, lift) in enumerate(k4_with_lift[:3], 1):
        items = sorted(itemset, key=lambda n: bf_freq[n], reverse=True)
        comms = {partition.get(n) for n in items if n in partition}
        cross = "★異コミュニティ" if len(comms) > 1 else f"C{list(comms)[0]}内"
        rp(f"  {MEDAL5[rank-1]}  Lift={lift:.2f}  Support={support}件  [{cross}]")
        for nd in items:
            lt  = "ダーク" if bf_dict[nd]["leaf_type"] == "dark" else "ライト"
            bst = "（ブースター）" if bf_dict[nd]["is_booster"] else ""
            rp(f"       ・{nd:<38} [{lt}]{bst}  {bf_freq[nd]}件")
        for i, (tx, title) in enumerate(zip(transactions, titles_list)):
            if itemset <= tx and title:
                rp(f"       代表: 「{title[:50]}」")
                break
        rp()
else:
    rp("  （4種ミックスで出現2回以上のセットは検出されませんでした）")
    rp()

# ── ブランドシナジーサマリー
rp(DBAR)
rp("【ブランド間シナジーマトリクス（クロスコミュニティ共起Top10）】")
rp(DBAR)
rp()
rpw("異なるコミュニティに属するブランド_フレーバーが同一レビューに登場した"
    "ペアを抽出。ライトリーフ×ダークリーフの融合レシピを明示する。")
rp()

cross_pairs = [
    (u, v, d["weight"])
    for u, v, d in G.edges(data=True)
    if partition.get(u) != partition.get(v)
]
cross_pairs.sort(key=lambda x: -x[2])

rp(f"  {'ペア':<60} {'共起':>4}  {'タイプ'}")
rp(f"  {BAR[:64]}")
for u, v, w in cross_pairs[:10]:
    lt_u = "ダーク" if bf_dict[u]["leaf_type"] == "dark" else "ライト"
    lt_v = "ダーク" if bf_dict[v]["leaf_type"] == "dark" else "ライト"
    cross_type = f"{lt_u}×{lt_v}"
    pair_label = f"{u} × {v}"[:58]
    rp(f"  {pair_label:<60} {w:>4}回  [{cross_type}]")
rp()

# ── 統計
rp(DBAR)
rp("【分析統計】")
rp(DBAR)
rp()
rp(f"  入力レビュー数              : {N} 件")
rp(f"  ブランド_フレーバーノード候補: {len(bf_dict)} 種")
rp(f"  グラフノード数 (足切り後)    : {G.number_of_nodes()}")
rp(f"  グラフエッジ数              : {G.number_of_edges()}")
rp(f"  3種以上のトランザクション   : {len(tx3plus)} 件")
rp(f"  3種 Lift>1 のセット         : {len(k3_with_lift)}")
rp(f"  4種 Lift>1 のセット         : {len(k4_with_lift)}")
rp()
rp(f"  出力画像: {OUT_PNG}")
rp(f"  出力レポート: {OUT_TXT}")
rp()
rp("✓ 完了")

# ── ファイル保存
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
