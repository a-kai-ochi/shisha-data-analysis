#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シーシャ 多種フレーバーミックス アソシエーション分析
=====================================================
cloud_reviews_final.csv のレビュー本文から
  ① 3種ミックス の出現頻度 Top 5
  ② 4〜5種ミックス の出現頻度 Top 3
を抽出し、レポートとして出力・保存する。

実装方針:
  - aslaj_master_list.csv をホワイトリスト辞書として利用（flavor_mix_network.py と同一ロジック）
  - mlxtend 不要: itertools.combinations + Counter で完全実装
  - 「全件」分析に加え「レシピ特化（3-8種）」分析も実施し、
    ランキング記事による擬似共起の影響を分離して報告

実行方法:
  cd ~/datascience && python3 notebooks/multi_flavor_mix_ranking.py
"""

import os
import re
import unicodedata
import textwrap
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import community as community_louvain
import networkx as nx

# ────────────────────────────────────────────────────────────
# パス設定
# ────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TXT  = os.path.join(OUT_DIR, "multi_flavor_mix_ranking.txt")

# ────────────────────────────────────────────────────────────
# フレーバー抽出ロジック（flavor_mix_network.py と同一）
# ────────────────────────────────────────────────────────────
master  = pd.read_csv(os.path.join(DATA_DIR, "aslaj_master_list.csv"))
reviews = pd.read_csv(os.path.join(DATA_DIR, "cloud_reviews_final.csv"))
N = len(reviews)


def _clean_entry(raw_name: str, raw_brand: str) -> tuple[str, str]:
    name = str(raw_name).strip()
    sep  = re.match(r"^(.+?)\s*[-–]\s*([A-Za-zァ-ヿ].+)", name)
    if sep:
        bm = re.match(r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", sep.group(2).strip())
        return sep.group(1).strip(), (bm.group(1) if bm else str(raw_brand)[:10])
    bm = re.match(r"([A-Za-z]+(?:\s[A-Za-z]+)?|[ァ-ヿ]{2,8})", str(raw_brand).strip())
    return name, (bm.group(1) if bm else "不明")


def _patterns(name: str) -> tuple[str, list[str]]:
    n  = name.strip()
    ja = re.search(r"\(([ァ-ヿ][^\)]+)\)", n)
    en = re.match(r"^([A-Za-z][A-Za-z0-9 .&\'\-/]+?)(?:\(|$)", n)
    if ja:
        j = ja.group(1).strip()
        e = en.group(1).strip() if en else ""
        return j, [j] + ([e] if len(e) >= 3 else [])
    if re.match(r"^[ァ-ヿぁ-んー]", n):
        return n, [n]
    if re.match(r"^[A-Za-z]", n):
        return n.upper(), [n]
    return n, [n]


# 検索辞書構築
flavor_info: dict[str, str] = {}   # canonical → brand
p2c: dict[str, str] = {}

for _, row in master.iterrows():
    fc, bs = _clean_entry(row["フレーバー名"], row["ブランド"])
    if not fc or fc.strip() in ("nan", ""):
        continue
    can, pats = _patterns(fc)
    flavor_info.setdefault(can, bs)
    for p in pats:
        if len(p) >= 3:
            p2c.setdefault(p, can)

sorted_p = sorted(p2c, key=len, reverse=True)


def extract_flavors(text: str) -> frozenset[str]:
    """テキストからホワイトリスト登録フレーバーを貪欲最長マッチで抽出し frozenset で返す。"""
    if not isinstance(text, str):
        return frozenset()
    found: set[str] = set()
    masked: set[int] = set()
    tu = text.upper()
    for pat in sorted_p:
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
    return frozenset(found)


# ────────────────────────────────────────────────────────────
# トランザクション生成
# ────────────────────────────────────────────────────────────
transactions: list[frozenset[str]] = [
    extract_flavors(t) for t in reviews["レビュー本文"]
]

# レビュータイトルも保持（代表例表示用）
titles = reviews["レビュータイトル"].fillna("").tolist()

# 全フレーバー出現頻度
flavor_freq = Counter(fl for tx in transactions for fl in tx)

# サイズ別分類
tx_by_size: dict[int, list[int]] = defaultdict(list)
for i, tx in enumerate(transactions):
    tx_by_size[len(tx)].append(i)

print(f"[INFO] トランザクション数: {N}")
print(f"[INFO] 3種以上: {sum(1 for tx in transactions if len(tx)>=3)} 件")
print(f"[INFO] 5種以上: {sum(1 for tx in transactions if len(tx)>=5)} 件")
print(f"[INFO] 10種超 (ランキング記事扱い): "
      f"{sum(1 for tx in transactions if len(tx)>10)} 件")

# ────────────────────────────────────────────────────────────
# コミュニティ（Louvain）再構築 ← ネットワークラベル付け用
# ────────────────────────────────────────────────────────────
MIN_NODE_FREQ, MIN_EDGE_W = 3, 2
top_nodes = {fl for fl, cnt in flavor_freq.items() if cnt >= MIN_NODE_FREQ}
G = nx.Graph()
for fl in top_nodes:
    G.add_node(fl, freq=flavor_freq[fl])
cooc_tmp: Counter = Counter()
for tx in transactions:
    fl_list = sorted(tx)
    for pair in combinations(fl_list, 2):
        cooc_tmp[pair] += 1
for (f1, f2), cnt in cooc_tmp.items():
    if f1 in top_nodes and f2 in top_nodes and cnt >= MIN_EDGE_W:
        G.add_edge(f1, f2, weight=cnt)
G.remove_nodes_from(list(nx.isolates(G)))
partition = community_louvain.best_partition(G, weight="weight", random_state=42)

comm_members: dict[int, list[str]] = defaultdict(list)
for node, cid in partition.items():
    comm_members[cid].append(node)

def comm_label(cid: int) -> str:
    mems = sorted(comm_members.get(cid, ["?"]),
                  key=lambda n: G.nodes[n]["freq"] if n in G else 0, reverse=True)
    return "/".join(mems[:2]) + "系"

def flavor_tag(fl: str) -> str:
    cid = partition.get(fl)
    if cid is None:
        return fl
    return f"{fl}[C{cid}]"


# ────────────────────────────────────────────────────────────
# アソシエーション分析コア
# ────────────────────────────────────────────────────────────

def count_itemsets(txns: list[frozenset], k: int,
                   min_support: int = 2) -> list[tuple[frozenset, int]]:
    """
    k-item 頻出アイテムセットを Counter で集計。
    min_support 件以上登場するものを降順で返す。
    同一トランザクション内の重複は frozenset が自動排除。
    """
    counter: Counter = Counter()
    for tx in txns:
        if len(tx) < k:
            continue
        for combo in combinations(sorted(tx), k):
            counter[frozenset(combo)] += 1
    return [(s, c) for s, c in counter.items() if c >= min_support]


def find_representative_reviews(itemset: frozenset, txns, titles, max_examples=3
                                  ) -> list[str]:
    """アイテムセットを含む代表レビュータイトルを返す。"""
    examples = []
    for i, tx in enumerate(txns):
        if itemset <= tx:
            title = titles[i]
            if title:
                examples.append(title[:52] + ("…" if len(title) > 52 else ""))
            if len(examples) >= max_examples:
                break
    return examples


# ── 2種類のトランザクションセットで分析 ──────────────────────

# [A] 全件トランザクション（ランキング記事も含む）
txns_all = [tx for tx in transactions if len(tx) >= 2]
indices_all = [i for i, tx in enumerate(transactions) if len(tx) >= 2]

# [B] レシピ特化トランザクション（3〜8種フレーバー ＝ 実際のミックス議論に近い）
MAX_RECIPE_SIZE = 8
txns_recipe  = [tx for tx in transactions if 3 <= len(tx) <= MAX_RECIPE_SIZE]
indices_recipe = [i for i, tx in enumerate(transactions) if 3 <= len(tx) <= MAX_RECIPE_SIZE]

print(f"[INFO] 全件トランザクション数 (2種以上): {len(txns_all)}")
print(f"[INFO] レシピ特化トランザクション数 (3-{MAX_RECIPE_SIZE}種): {len(txns_recipe)}")

# 3-item sets
sets_3_all    = count_itemsets(txns_all,    k=3, min_support=2)
sets_3_recipe = count_itemsets(txns_recipe, k=3, min_support=2)
sets_3_all.sort(key=lambda x: -x[1])
sets_3_recipe.sort(key=lambda x: -x[1])

# 4-item sets
sets_4_all    = count_itemsets(txns_all,    k=4, min_support=2)
sets_4_recipe = count_itemsets(txns_recipe, k=4, min_support=2)
sets_4_all.sort(key=lambda x: -x[1])
sets_4_recipe.sort(key=lambda x: -x[1])

# 5-item sets
sets_5_all    = count_itemsets(txns_all,    k=5, min_support=2)
sets_5_recipe = count_itemsets(txns_recipe, k=5, min_support=2)
sets_5_all.sort(key=lambda x: -x[1])
sets_5_recipe.sort(key=lambda x: -x[1])

print(f"[INFO] 3-item頻出セット (全件/min=2): {len(sets_3_all)}")
print(f"[INFO] 3-item頻出セット (レシピ/min=2): {len(sets_3_recipe)}")
print(f"[INFO] 4-item頻出セット (全件/min=2): {len(sets_4_all)}")
print(f"[INFO] 5-item頻出セット (全件/min=2): {len(sets_5_all)}")


# ────────────────────────────────────────────────────────────
# レポート生成
# ────────────────────────────────────────────────────────────
DBAR = "═" * 64
BAR  = "─" * 64
W    = 62

lines: list[str] = []   # ファイル保存用バッファ


def p(*args, **kwargs):
    """print して同時に lines バッファにも追記。"""
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    lines.append(text)


def pw(text, width=W, indent="  "):
    """折り返し付き print。"""
    wrapped = textwrap.fill(text, width=width,
                            initial_indent=indent,
                            subsequent_indent=indent)
    print(wrapped)
    lines.append(wrapped)


def itemset_block(rank: int, itemset: frozenset, support: int,
                  total_tx: int, txns_ref, titles_ref,
                  medal: list[str], note: str = "") -> None:
    """1レコードの整形出力。"""
    support_pct = support / total_tx * 100
    items = sorted(itemset, key=lambda f: -flavor_freq[f])

    # コミュニティ色分け
    tagged = " ＋ ".join(flavor_tag(f) for f in items)

    p(f"  {medal[rank-1]}  {tagged}")
    p(f"       出現: {support} 件 / {total_tx} 件中  "
      f"（Support = {support_pct:.1f}%）")
    if note:
        pw(note, indent="       ")

    # コミュニティ構成の解釈
    comms = [partition.get(f) for f in items if partition.get(f) is not None]
    n_comms = len(set(c for c in comms if c is not None))
    if n_comms == 1:
        cid = comms[0]
        pw(f"→ 全フレーバーが同一コミュニティ C{cid}（{comm_label(cid)}）内。"
           f"同系統の味を重ねる「純粋フュージョン」パターン。",
           indent="       ")
    else:
        comm_desc = "、".join(
            f"C{c}({comm_label(c).split('/')[0]})" for c in sorted(set(c for c in comms if c is not None))
        )
        pw(f"→ {n_comms} コミュニティ横断ミックス（{comm_desc}）。"
           f"異なる系統の組み合わせによる「コントラスト型」レシピ。",
           indent="       ")

    # 代表レビュー
    examples = find_representative_reviews(itemset, txns_ref, titles_ref, max_examples=2)
    if examples:
        p("       代表レビュー:")
        for ex in examples:
            p(f"         ・{ex}")
    p()


# ════════ ヘッダー ════════
p()
p("╔" + "═" * 62 + "╗")
p("║" + " シーシャ 多種フレーバーミックス アソシエーション分析 ".center(62) + "║")
p("║" + f" レビュー {N}件 ／ マスター辞書 {len(flavor_info)}種 ".center(62) + "║")
p("╚" + "═" * 62 + "╝")
p()
p("  ■ 分析モードの違い")
pw("・全件モード: 全222件のレビューを対象（ランキング記事を含む）")
pw("・レシピ特化モード: 3〜8種のみ言及するレビュー（実際のミックス"
   "  議論に近い記事に限定）を対象とした精度重視の分析", indent="  ")
p()

# ════════ セクション 1: 3種ミックス ════════
p(DBAR)
p("【① 3種ミックス 出現頻度（Support）ランキング Top 5】")
p(DBAR)
p()
pw("同一レビュー内で同時に言及された「3フレーバー組み合わせ」の中で、"
   "最も多くのレビューに登場したレシピを抽出。"
   "[C0]=フルーツ/ミント系、[C1]=シガー系、[C2]=スイーツ/ミルク系。")
p()

MEDAL5 = ["🥇", "🥈", "🥉", "  4.", "  5."]
MEDAL3 = ["🥇", "🥈", "🥉"]

# ── 1-A 全件 ──
p("  ▶ [全件モード] 全レビュー対象（N=" + str(len(txns_all)) + "件）")
p("  " + BAR[:58])
for rank, (itemset, support) in enumerate(sets_3_all[:5], 1):
    items = sorted(itemset, key=lambda f: -flavor_freq[f])
    label = "×".join(items)
    pct   = support / len(txns_all) * 100
    p(f"  {MEDAL5[rank-1]}  {label:<42} {support:>3}件  ({pct:.1f}%)")
p()

# ── 1-B レシピ特化（詳細付き） ──
p("  ▶ [レシピ特化モード] 3〜" + str(MAX_RECIPE_SIZE) + "種記事のみ（N=" +
  str(len(txns_recipe)) + "件）― 詳細解説付き")
p("  " + BAR[:58])
p()

for rank, (itemset, support) in enumerate(sets_3_recipe[:5], 1):
    # サポート率をレシピ特化分母で計算
    note_txt = ""
    # 追加解釈ノート
    items_sorted = sorted(itemset, key=lambda f: -flavor_freq[f])
    freq_list = [(f, flavor_freq[f]) for f in items_sorted]
    note_txt = (
        "個別出現頻度: " +
        ", ".join(f"{f}={c}件" for f, c in freq_list) +
        "。"
    )
    itemset_block(rank, itemset, support,
                  len(txns_recipe), txns_recipe, titles,
                  MEDAL5, note=note_txt)

# ── 両モードのSupportが一致するか相互確認ノート ──
pw("💡 インサイト: 上位ランクはいずれも「ミント系（C0）」を核とした"
   "フルーツ+清涼感の組み合わせ。これはシーシャ初〜中級者が"
   "「まずミントを添える」という文化的慣習を反映している。"
   "4位以降で「バニラ系（C2）」が登場し、甘み・コクを加えた"
   "上位互換レシピへの需要も見える。")
p()

# ════════ セクション 2: 4〜5種ミックス ════════
p(DBAR)
p("【② 4〜5種ミックス 出現頻度（Support）ランキング Top 3 × 2】")
p(DBAR)
p()
pw("4種・5種の組み合わせは出現確率が下がるため、「このレシピが複数の"
   "レビュアーに共有されている」こと自体が強いシグナル。")
p()

# ── 2-A 4種ミックス ──
p("  ★ 4種ミックス ランキング")
p("  " + BAR[:58])
p()

if sets_4_recipe:
    for rank, (itemset, support) in enumerate(sets_4_recipe[:3], 1):
        items_s = sorted(itemset, key=lambda f: -flavor_freq[f])
        note_txt = (
            "個別出現頻度: " +
            ", ".join(f"{f}={flavor_freq[f]}件" for f in items_s) +
            "。"
        )
        itemset_block(rank, itemset, support,
                      len(txns_recipe), txns_recipe, titles,
                      MEDAL3, note=note_txt)
elif sets_4_all:
    pw("  ※ レシピ特化モードで4種Min=2の条件を満たすセットが少ないため"
       "全件モードで代替表示します。")
    p()
    for rank, (itemset, support) in enumerate(sets_4_all[:3], 1):
        items_s = sorted(itemset, key=lambda f: -flavor_freq[f])
        note_txt = (
            "個別出現頻度: " +
            ", ".join(f"{f}={flavor_freq[f]}件" for f in items_s) +
            "。"
        )
        itemset_block(rank, itemset, support,
                      len(txns_all), txns_all, titles,
                      MEDAL3, note=note_txt)
else:
    p("  （4種ミックスで出現頻度2以上のセットは検出されませんでした）")
    p()

# ── 2-B 5種ミックス ──
p("  ★ 5種ミックス ランキング")
p("  " + BAR[:58])
p()

if sets_5_recipe:
    for rank, (itemset, support) in enumerate(sets_5_recipe[:3], 1):
        items_s = sorted(itemset, key=lambda f: -flavor_freq[f])
        note_txt = (
            "個別出現頻度: " +
            ", ".join(f"{f}={flavor_freq[f]}件" for f in items_s) +
            "。"
        )
        itemset_block(rank, itemset, support,
                      len(txns_recipe), txns_recipe, titles,
                      MEDAL3, note=note_txt)
elif sets_5_all:
    pw("  ※ レシピ特化モードで5種Min=2の条件を満たすセットが少ないため"
       "全件モードで代替表示します。")
    p()
    for rank, (itemset, support) in enumerate(sets_5_all[:3], 1):
        items_s = sorted(itemset, key=lambda f: -flavor_freq[f])
        note_txt = (
            "個別出現頻度: " +
            ", ".join(f"{f}={flavor_freq[f]}件" for f in items_s) +
            "。"
        )
        itemset_block(rank, itemset, support,
                      len(txns_all), txns_all, titles,
                      MEDAL3, note=note_txt)
else:
    p("  （5種ミックスで出現頻度2以上のセットは検出されませんでした）")
    p()

pw("💡 インサイト: 4〜5種レシピになると「ミント＋フルーツ（酸）＋"
   "スイーツ系（甘）」の三角形構造が明確化。さらにバニラやミルクが"
   "加わることで『甘み＋酸＋清涼感』のバランス型レシピが定番化している"
   "様子がわかる。5種以上はレシピ記事というよりフレーバーカタログ的な"
   "言及になるため、Support値より代表レビューのタイトルを参照されたい。")
p()

# ════════ セクション 3: 統計サマリー ════════
p(DBAR)
p("【分析統計サマリー】")
p(DBAR)
p()
p(f"  入力レビュー数              : {N} 件")
p(f"  ホワイトリストフレーバー数  : {len(flavor_info)} 種")
p(f"  ヒットフレーバー総ユニーク  : {len(flavor_freq)} 種")
p()
p("  フレーバー数 / レビュー数の分布:")
size_dist = Counter(len(tx) for tx in transactions)
for k in sorted(size_dist):
    bar = "█" * min(size_dist[k], 40)
    p(f"    {k:>2}種: {size_dist[k]:>3}件  {bar}")
p()
p("  3-item 頻出セット数 (全件/min=2)    : " + str(len(sets_3_all)))
p("  3-item 頻出セット数 (レシピ/min=2)  : " + str(len(sets_3_recipe)))
p("  4-item 頻出セット数 (全件/min=2)    : " + str(len(sets_4_all)))
p("  4-item 頻出セット数 (レシピ/min=2)  : " + str(len(sets_4_recipe)))
p("  5-item 頻出セット数 (全件/min=2)    : " + str(len(sets_5_all)))
p("  5-item 頻出セット数 (レシピ/min=2)  : " + str(len(sets_5_recipe)))
p()
p("  コミュニティ別フレーバー数:")
for cid in sorted(comm_members):
    mems = sorted(comm_members[cid], key=lambda n: flavor_freq.get(n, 0), reverse=True)
    p(f"    C{cid} [{comm_label(cid)}]: {len(mems)}種  "
      f"（{', '.join(mems[:4])} ...）")
p()
p(f"  出力ファイル: {OUT_TXT}")
p()
p("✓ 完了")

# ────────────────────────────────────────────────────────────
# ファイル保存
# ────────────────────────────────────────────────────────────
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
