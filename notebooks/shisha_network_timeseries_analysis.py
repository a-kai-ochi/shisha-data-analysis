#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シーシャデータ 共起ネットワーク分析 & 時系列テキスト分析
=============================================================
目的:
  1. レビュー本文からフレーバーの共起ネットワークを構築し、コミュニティを自動検出
  2. 2021-2022 vs 2025-2026 の2時代に分けて味の好みの変遷を比較・可視化

実行方法:
  cd ~/datascience && python3 notebooks/shisha_network_timeseries_analysis.py
"""

# ============================================================
# 1. ライブラリのインポート
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain
import MeCab
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 日本語フォント設定
plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# ============================================================
# 2. パス設定
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

REVIEWS_CSV = os.path.join(DATA_DIR, "cloud_reviews_final.csv")
MASTER_CSV  = os.path.join(DATA_DIR, "aslaj_master_list.csv")

# ============================================================
# 3. データの読み込みと前処理
# ============================================================
print("[1/8] データ読み込み中...")

df = pd.read_csv(REVIEWS_CSV)
master = pd.read_csv(MASTER_CSV)

# 日本語月名 → 日付型変換（10月・11月・12月を先に処理して部分一致を防ぐ）
MONTH_MAP = {
    "12月": "Dec", "11月": "Nov", "10月": "Oct",
    "9月":  "Sep", "8月":  "Aug", "7月":  "Jul",
    "6月":  "Jun", "5月":  "May", "4月":  "Apr",
    "3月":  "Mar", "2月":  "Feb", "1月":  "Jan",
}

def parse_ja_date(s: str) -> pd.Timestamp:
    if pd.isna(s):
        return pd.NaT
    for ja, en in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        s = s.replace(ja, en)
    return pd.to_datetime(s, format="%b %d, %Y", errors="coerce")

df["日付"] = df["更新日"].apply(parse_ja_date)
df["年"]   = df["日付"].dt.year

# 2時代に分割
df_early  = df[df["年"].isin([2021, 2022])].copy().reset_index(drop=True)
df_recent = df[df["年"].isin([2025, 2026])].copy().reset_index(drop=True)

print(f"  2021-2022: {len(df_early)} 件")
print(f"  2025-2026: {len(df_recent)} 件")

# ============================================================
# 4. 形態素解析の設定
# ============================================================
print("[2/8] 形態素解析の準備...")

# シーシャ特有の複合語：MeCabが分割してしまう複合語を後処理でマージする
COMPOUND_TERMS = {
    # (前トークン, 後トークン) → 結合後の表記
    ("ダーク",     "リーフ"):   "ダークリーフ",
    ("シガー",     "リーフ"):   "シガーリーフ",
    ("ライト",     "リーフ"):   "ライトリーフ",
    ("バージニア", "リーフ"):   "バージニアリーフ",
    ("ブロンド",   "リーフ"):   "ブロンドリーフ",
    ("ダーク",     "ブレンド"): "ダークブレンド",
    ("シガー",     "ブレンド"): "シガーブレンド",
    ("クリーム",   "ソーダ"):   "クリームソーダ",
    ("ブルー",     "ベリー"):   "ブルーベリー",
    ("ストロベリー","ミルク"):  "ストロベリーミルク",
    ("アイス",     "クリーム"): "アイスクリーム",
    ("グレープ",   "フルーツ"): "グレープフルーツ",
    ("パッション", "フルーツ"): "パッションフルーツ",
    ("ドラゴン",   "フルーツ"): "ドラゴンフルーツ",
    ("メロン",     "ソーダ"):   "メロンソーダ",
}

# 重要キーワードの正規化マッピング（表記ゆれ統一）
NORMALIZE_MAP = {
    "ダーク葉":   "ダークリーフ",
    "シガー葉":   "シガーリーフ",
    "煙草":       "タバコ",
    "タバコ":     "タバコ",
    "甘み":       "甘さ",
    "甘味":       "甘さ",
    "冷感":       "清涼感",
    "涼感":       "清涼感",
    "清涼":       "清涼感",
}

# ストップワード：汎用すぎる語、助詞相当の名詞、etc.
STOP_WORDS = {
    "シーシャ", "フレーバー", "レビュー", "こと", "もの", "ため", "方",
    "感じ", "感", "系", "種", "型", "品", "物", "時", "点", "時代",
    "日本", "世界", "初心者", "上級者", "ユーザー", "ファン", "方々",
    "今回", "今", "最", "次", "各", "多く", "全て", "全",
    "特徴", "評価", "ランキング", "人気", "注目", "紹介", "ご紹介",
    "紹介し", "説明", "解説", "おすすめ", "スタンダード", "定番",
    "本", "記事", "サイト", "データ", "情報", "番", "位",
    "年", "月", "日", "週", "回", "個", "本", "枚", "袋",
    "ブランド", "ショップ", "店", "販売", "購入", "値段", "価格",
    "の", "が", "は", "を", "に", "へ", "と", "で", "から", "まで",
}

# 品詞フィルタ：名詞（一般/固有/サ変接続）と形容詞のみ
ALLOWED_POS = {"名詞-一般", "名詞-固有名詞", "名詞-サ変接続", "形容詞-自立"}

tagger = MeCab.Tagger()

def tokenize(text: str) -> list[str]:
    """MeCabで形態素解析し、名詞・形容詞のみ抽出 → 複合語マージ → 正規化"""
    if not isinstance(text, str) or not text.strip():
        return []

    raw_tokens = []
    node = tagger.parseToNode(text)
    while node:
        surface  = node.surface
        feature  = node.feature.split(",")
        pos_main = feature[0] if len(feature) > 0 else ""
        pos_sub  = feature[1] if len(feature) > 1 else ""
        pos_key  = f"{pos_main}-{pos_sub}"

        if (pos_main == "名詞" and pos_sub in ("一般", "固有名詞", "サ変接続", "接尾")) \
           or (pos_main == "形容詞" and pos_sub == "自立"):
            if len(surface) >= 2:  # 1文字語は除外
                raw_tokens.append(surface)
        node = node.next

    # --- 複合語マージ（2-gram パターンマッチ）---
    merged = []
    i = 0
    while i < len(raw_tokens):
        if i + 1 < len(raw_tokens):
            pair = (raw_tokens[i], raw_tokens[i + 1])
            if pair in COMPOUND_TERMS:
                merged.append(COMPOUND_TERMS[pair])
                i += 2
                continue
        merged.append(raw_tokens[i])
        i += 1

    # --- 正規化・ストップワード除去 ---
    result = []
    for tok in merged:
        tok = NORMALIZE_MAP.get(tok, tok)
        if tok not in STOP_WORDS and len(tok) >= 2:
            result.append(tok)
    return result


# ============================================================
# 5. 共起ネットワーク構築
# ============================================================
print("[3/8] トークン抽出中...")

def extract_tokens_from_df(df_subset: pd.DataFrame) -> tuple[list[list[str]], Counter]:
    """全レビューからトークンリストを抽出。文単位で共起を計算するため文ごとにリスト化。"""
    all_doc_tokens = []   # ドキュメントごとのトークンリスト（共起計算用）
    flat_tokens   = []    # 全トークンの平坦リスト（頻度集計用）

    for text in df_subset["レビュー本文"].dropna():
        # 文分割（句点・改行・読点で区切り）
        sentences = re.split(r"[。！？\n]", text)
        doc_tokens = []
        for sent in sentences:
            toks = tokenize(sent)
            if toks:
                doc_tokens.append(toks)
                flat_tokens.extend(toks)
        if doc_tokens:
            all_doc_tokens.append(doc_tokens)
    return all_doc_tokens, Counter(flat_tokens)


doc_tokens_early,  freq_early  = extract_tokens_from_df(df_early)
doc_tokens_recent, freq_recent = extract_tokens_from_df(df_recent)
print(f"  2021-2022 ユニーク語数: {len(freq_early)}")
print(f"  2025-2026 ユニーク語数: {len(freq_recent)}")

print("[4/8] 共起ネットワーク構築中...")

def build_cooccurrence_graph(doc_tokens: list[list[list[str]]],
                              freq: Counter,
                              top_n: int = 80,
                              min_freq: int = 2,
                              min_cooc: int = 2) -> nx.Graph:
    """
    上位top_n語に絞り、文単位の共起グラフを構築。
    doc_tokens: [document → [sentence → [token]]]
    """
    # 出現頻度 >= min_freq の上位 top_n 語を対象語彙とする
    vocab = {w for w, c in freq.most_common(top_n) if c >= min_freq}

    cooc_counter: Counter = Counter()
    for doc in doc_tokens:
        for sent_toks in doc:
            filtered = [t for t in sent_toks if t in vocab]
            for pair in combinations(sorted(set(filtered)), 2):
                cooc_counter[pair] += 1

    G = nx.Graph()
    for (w1, w2), cnt in cooc_counter.items():
        if cnt >= min_cooc:
            G.add_edge(w1, w2, weight=cnt)

    # 孤立ノードを追加（頻出語でもエッジがない場合）
    for w, c in freq.most_common(top_n):
        if c >= min_freq and w not in G:
            G.add_node(w)

    # ノード属性：出現頻度
    for node in G.nodes():
        G.nodes[node]["freq"] = freq.get(node, 0)

    return G


# 2021-2022は件数が多いのでしきい値を高め、2025-2026は低めに設定
G_early  = build_cooccurrence_graph(doc_tokens_early,  freq_early,  top_n=80, min_freq=3, min_cooc=3)
G_recent = build_cooccurrence_graph(doc_tokens_recent, freq_recent, top_n=60, min_freq=2, min_cooc=2)

print(f"  2021-2022 graph: {G_early.number_of_nodes()} nodes, {G_early.number_of_edges()} edges")
print(f"  2025-2026 graph: {G_recent.number_of_nodes()} nodes, {G_recent.number_of_edges()} edges")

# ============================================================
# 6. コミュニティ検出（Louvainアルゴリズム）
# ============================================================
print("[5/8] コミュニティ検出中...")

# Louvainアルゴリズムはエッジのある連結成分にのみ適用
def detect_communities(G: nx.Graph) -> dict:
    """Louvainアルゴリズムでコミュニティを検出。孤立ノードは別クラスタ扱い。"""
    # エッジを持つサブグラフで実行
    G_connected = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    if G_connected.number_of_nodes() == 0:
        return {n: 0 for n in G.nodes()}
    partition = community_louvain.best_partition(G_connected, weight="weight", random_state=42)
    # 孤立ノードはコミュニティ -1
    for node in G.nodes():
        if node not in partition:
            partition[node] = -1
    return partition


partition_early  = detect_communities(G_early)
partition_recent = detect_communities(G_recent)

# コミュニティ番号をクラスタラベルに変換（頻出語の内容でクラスタに名前を付ける）
def label_communities(partition: dict, freq: Counter, top_k: int = 3) -> dict[int, str]:
    """各コミュニティを代表語でラベル付け"""
    comm_words: dict[int, list] = defaultdict(list)
    for word, comm_id in partition.items():
        comm_words[comm_id].append((freq.get(word, 0), word))
    labels = {}
    for comm_id, words in comm_words.items():
        top = sorted(words, reverse=True)[:top_k]
        labels[comm_id] = " / ".join(w for _, w in top)
    return labels


labels_early  = label_communities(partition_early,  freq_early)
labels_recent = label_communities(partition_recent, freq_recent)

print(f"  2021-2022 コミュニティ数: {max(partition_early.values()) + 1}")
print(f"  2025-2026 コミュニティ数: {max(v for v in partition_recent.values() if v >= 0) + 1}")

# ============================================================
# 7. 可視化
# ============================================================
print("[6/8] 共起ネットワーク可視化中...")

# カラーパレット（コミュニティ数に対応できるよう多めに定義）
COMMUNITY_COLORS = [
    "#E74C3C",  # 赤
    "#3498DB",  # 青
    "#2ECC71",  # 緑
    "#F39C12",  # オレンジ
    "#9B59B6",  # 紫
    "#1ABC9C",  # ティール
    "#E67E22",  # 濃いオレンジ
    "#34495E",  # グレー
    "#E91E63",  # ピンク
    "#00BCD4",  # シアン
]

def draw_network(ax, G: nx.Graph, partition: dict, title: str,
                 freq: Counter, node_size_scale: float = 300):
    """共起ネットワークを描画"""
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center", fontsize=14)
        ax.set_title(title, fontsize=14, fontweight="bold")
        return

    # レイアウト：spring layout（シード固定で再現性確保）
    pos = nx.spring_layout(G, weight="weight", k=2.5, iterations=100, seed=42)

    # ノードサイズ: 出現頻度に比例
    max_freq = max(freq.values()) if freq else 1
    node_sizes = []
    node_colors = []
    for node in G.nodes():
        f = freq.get(node, 1)
        node_sizes.append(node_size_scale * (f / max_freq) ** 0.5 + 100)
        comm_id = partition.get(node, -1)
        color_idx = comm_id % len(COMMUNITY_COLORS) if comm_id >= 0 else len(COMMUNITY_COLORS) - 1
        node_colors.append(COMMUNITY_COLORS[color_idx])

    # エッジ太さ: 共起頻度に比例
    edges = G.edges(data=True)
    edge_weights = [d.get("weight", 1) for _, _, d in edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1.5 * w / max_w for w in edge_weights]

    # 描画
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.35,
        edge_color="#888888",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
        linewidths=0.5,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7,
        font_family="TakaoGothic",
        font_color="black",
        verticalalignment="bottom",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

    # 凡例（コミュニティ）
    unique_comms = sorted({v for v in partition.values() if v >= 0})
    patches = []
    for comm_id in unique_comms[:8]:  # 最大8コミュニティまで凡例表示
        from collections import defaultdict
        words_in_comm = [w for w, c in partition.items() if c == comm_id]
        top_word = sorted(words_in_comm, key=lambda w: freq.get(w, 0), reverse=True)[0] if words_in_comm else "?"
        c = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        patches.append(mpatches.Patch(color=c, label=f"C{comm_id}: {top_word}"))
    ax.legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.8)


# ---- 図1: 2時代の共起ネットワーク並列表示 ----
fig_net, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig_net.patch.set_facecolor("#F8F9FA")

draw_network(ax1, G_early,  partition_early,  "共起ネットワーク 2021-2022\n（ライト・フルーツ・ミント時代）",
             freq_early,  node_size_scale=500)
draw_network(ax2, G_recent, partition_recent, "共起ネットワーク 2025-2026\n（ダークリーフ・シガー台頭期）",
             freq_recent, node_size_scale=600)

fig_net.suptitle("シーシャ フレーバー共起ネットワーク分析\n2時代のフレーバートレンド比較",
                 fontsize=16, fontweight="bold", y=1.01)
fig_net.tight_layout(pad=2.0)

out_path_net = os.path.join(OUT_DIR, "network_comparison.png")
fig_net.savefig(out_path_net, bbox_inches="tight", facecolor=fig_net.get_facecolor())
print(f"  保存: {out_path_net}")
plt.close(fig_net)


# ---- 図2: 単語頻度比較（上位30語） ----
print("[7/8] 単語頻度比較グラフ作成中...")

def get_top_words(freq: Counter, n: int = 30, exclude: set = None) -> list[tuple[str, int]]:
    exclude = exclude or set()
    return [(w, c) for w, c in freq.most_common(n + len(exclude))
            if w not in exclude][:n]


# 2時代の正規化頻度（ドキュメント数で割ってスケールを合わせる）
n_early  = max(len(df_early),  1)
n_recent = max(len(df_recent), 1)

freq_early_norm  = Counter({w: c / n_early  for w, c in freq_early.items()})
freq_recent_norm = Counter({w: c / n_recent for w, c in freq_recent.items()})

# 共通上位語と各期間特有語
all_top_words = set(w for w, _ in freq_early_norm.most_common(40)) | \
                set(w for w, _ in freq_recent_norm.most_common(40))

# 各語のスコアを結合してDataFrame化
records = []
for w in all_top_words:
    records.append({
        "語":        w,
        "2021-2022": freq_early_norm.get(w, 0),
        "2025-2026": freq_recent_norm.get(w, 0),
    })
compare_df = pd.DataFrame(records).sort_values("2025-2026", ascending=False).head(35)

# 棒グラフ（横）
fig_freq, ax = plt.subplots(figsize=(14, 12))
fig_freq.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#FAFAFA")

y = np.arange(len(compare_df))
height = 0.38

bars1 = ax.barh(y + height / 2, compare_df["2021-2022"], height=height,
                color="#3498DB", alpha=0.8, label="2021-2022")
bars2 = ax.barh(y - height / 2, compare_df["2025-2026"], height=height,
                color="#E74C3C", alpha=0.8, label="2025-2026")

ax.set_yticks(y)
ax.set_yticklabels(compare_df["語"], fontsize=10)
ax.set_xlabel("レビュー1件あたりの平均出現頻度", fontsize=11)
ax.set_title("シーシャ関連語 頻度比較（2021-22 vs 2025-26）\n正規化済み（ドキュメント数で除算）",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.4, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig_freq.tight_layout()
out_path_freq = os.path.join(OUT_DIR, "wordfreq_comparison.png")
fig_freq.savefig(out_path_freq, bbox_inches="tight", facecolor=fig_freq.get_facecolor())
print(f"  保存: {out_path_freq}")
plt.close(fig_freq)


# ---- 図3: キーワード時系列トレンド（注目語の時代別頻度変化） ----
print("[8/8] キーワードトレンド分析グラフ作成中...")

# 着目キーワード：ダークリーフ系・シガー系・ライト系・フルーツ系・ミント系
FOCUS_KEYWORDS = {
    "ダークリーフ系": ["ダークリーフ", "ダーク", "ブラックリーフ", "ダークブレンド"],
    "シガーリーフ系": ["シガーリーフ", "シガー", "葉巻"],
    "コク・重厚感":   ["コク", "重み", "スモーキー", "濃厚", "深み", "旨味", "リッチ"],
    "フルーツ系":     ["フルーツ", "果物", "ベリー", "マンゴー", "ストロベリー", "ライチ"],
    "ミント・清涼系": ["ミント", "清涼感", "冷感", "クール", "メンソール"],
    "スイーツ系":     ["スイーツ", "甘さ", "バニラ", "チョコ", "キャラメル", "クリーム"],
    "フローラル系":   ["フローラル", "ローズ", "ジャスミン", "花"],
}

# 年ごとの各キーワードグループの出現頻度を集計
# 全件で年次時系列データを作る
all_years_data = []
for _, row in df.iterrows():
    year = row["年"]
    if pd.isna(year) or int(year) not in [2021, 2022, 2023, 2025, 2026]:
        continue
    text = row["レビュー本文"]
    if not isinstance(text, str):
        continue
    toks = set(tokenize(text))
    record = {"年": int(year)}
    for group, keywords in FOCUS_KEYWORDS.items():
        record[group] = sum(1 for k in keywords if k in toks)
    all_years_data.append(record)

trend_df = pd.DataFrame(all_years_data)
trend_agg = trend_df.groupby("年").agg(
    {grp: ["sum", "count"] for grp in FOCUS_KEYWORDS.keys()}
)

# 正規化（年ごとの記事数で割る）
yearly_counts = trend_df.groupby("年").size()
trend_norm = {}
for grp in FOCUS_KEYWORDS.keys():
    trend_norm[grp] = {}
    for year in sorted(trend_df["年"].unique()):
        n_docs = yearly_counts.get(year, 1)
        raw = trend_df[trend_df["年"] == year][grp].sum()
        trend_norm[grp][year] = raw / n_docs

trend_norm_df = pd.DataFrame(trend_norm).T  # rows=グループ, cols=年

# レーダーチャート風の比較 + 折れ線グラフ
fig_trend = plt.figure(figsize=(18, 6))
fig_trend.patch.set_facecolor("#F8F9FA")
gs = GridSpec(1, 2, figure=fig_trend, width_ratios=[1.2, 1])

# 左: 折れ線グラフ（年次トレンド）
ax_line = fig_trend.add_subplot(gs[0])
ax_line.set_facecolor("#FAFAFA")

colors_trend = ["#8E44AD", "#C0392B", "#D35400", "#27AE60", "#2980B9", "#F39C12", "#16A085"]
years_available = sorted(trend_norm_df.columns.tolist())

for i, (grp, color) in enumerate(zip(FOCUS_KEYWORDS.keys(), colors_trend)):
    vals = [trend_norm_df.loc[grp, y] if y in trend_norm_df.columns else 0
            for y in years_available]
    ax_line.plot(years_available, vals, "o-", label=grp, color=color, linewidth=2,
                 markersize=7, alpha=0.85)

ax_line.set_xticks(years_available)
ax_line.set_xticklabels([str(y) for y in years_available], fontsize=10)
ax_line.set_xlabel("年", fontsize=11)
ax_line.set_ylabel("レビュー1件あたりのキーワード出現数", fontsize=10)
ax_line.set_title("フレーバーカテゴリ別 年次トレンド\n(2021→2026)", fontsize=12, fontweight="bold")
ax_line.legend(fontsize=8, loc="upper left", framealpha=0.85)
ax_line.grid(alpha=0.3, linestyle="--")
ax_line.spines["top"].set_visible(False)
ax_line.spines["right"].set_visible(False)

# 右: 2時代比較棒グラフ（ダークリーフ・シガー系特化）
ax_bar = fig_trend.add_subplot(gs[1])
ax_bar.set_facecolor("#FAFAFA")

focus_groups = list(FOCUS_KEYWORDS.keys())
early_vals  = [trend_norm_df.loc[g, 2022] if 2022 in trend_norm_df.columns else 0
               for g in focus_groups]
recent_vals = [trend_norm_df.loc[g, 2026] if 2026 in trend_norm_df.columns else 0
               for g in focus_groups]

x = np.arange(len(focus_groups))
w = 0.35
ax_bar.bar(x - w/2, early_vals,  w, label="2022年", color="#3498DB", alpha=0.8)
ax_bar.bar(x + w/2, recent_vals, w, label="2026年", color="#E74C3C", alpha=0.8)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(focus_groups, rotation=30, ha="right", fontsize=9)
ax_bar.set_ylabel("1件あたりキーワード出現数", fontsize=10)
ax_bar.set_title("2022年 vs 2026年\nカテゴリ別シフト比較", fontsize=12, fontweight="bold")
ax_bar.legend(fontsize=10)
ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

fig_trend.suptitle("シーシャフレーバー 嗜好トレンド分析（2021-2026）",
                   fontsize=14, fontweight="bold", y=1.02)
fig_trend.tight_layout(pad=2.5)

out_path_trend = os.path.join(OUT_DIR, "keyword_trend.png")
fig_trend.savefig(out_path_trend, bbox_inches="tight", facecolor=fig_trend.get_facecolor())
print(f"  保存: {out_path_trend}")
plt.close(fig_trend)


# ---- 図4: ネットワーク単体（高解像度） ----
for period_name, G, partition, freq, subtitle in [
    ("2021_2022", G_early,  partition_early,  freq_early,
     "2021-2022：ライトフルーツ・ミント全盛期"),
    ("2025_2026", G_recent, partition_recent, freq_recent,
     "2025-2026：ダークリーフ・シガーリーフ台頭期"),
]:
    fig_single, ax_s = plt.subplots(figsize=(16, 14))
    fig_single.patch.set_facecolor("#F0F3F4")
    draw_network(ax_s, G, partition, subtitle, freq, node_size_scale=800)
    fig_single.tight_layout()
    out_path_s = os.path.join(OUT_DIR, f"network_{period_name}.png")
    fig_single.savefig(out_path_s, bbox_inches="tight",
                       facecolor=fig_single.get_facecolor())
    print(f"  保存: {out_path_s}")
    plt.close(fig_single)


# ============================================================
# 8. テキストサマリー出力
# ============================================================
print("\n" + "=" * 60)
print("■ 分析サマリー")
print("=" * 60)

print(f"\n◆ データ概要")
print(f"  総レビュー数: {len(df)}")
print(f"  2021-2022: {len(df_early)} 件  │  2025-2026: {len(df_recent)} 件")

print(f"\n◆ 2021-2022 Top20語（正規化）")
for rank, (w, c) in enumerate(freq_early_norm.most_common(20), 1):
    print(f"  {rank:2d}. {w} ({c:.3f})")

print(f"\n◆ 2025-2026 Top20語（正規化）")
for rank, (w, c) in enumerate(freq_recent_norm.most_common(20), 1):
    print(f"  {rank:2d}. {w} ({c:.3f})")

print(f"\n◆ 2021-2022 コミュニティ構成")
for comm_id, label in sorted(labels_early.items()):
    members = [w for w, c in partition_early.items() if c == comm_id]
    print(f"  C{comm_id}: {label}  （{len(members)}語）")

print(f"\n◆ 2025-2026 コミュニティ構成")
for comm_id, label in sorted(labels_recent.items()):
    members = [w for w, c in partition_recent.items() if c == comm_id]
    print(f"  C{comm_id}: {label}  （{len(members)}語）")

print(f"\n◆ パラダイムシフト指標")
dark_leaf_early  = freq_early_norm.get("ダークリーフ",  0) + freq_early_norm.get("ダーク", 0)
dark_leaf_recent = freq_recent_norm.get("ダークリーフ", 0) + freq_recent_norm.get("ダーク", 0)
cigar_early  = freq_early_norm.get("シガーリーフ",  0) + freq_early_norm.get("シガー", 0)
cigar_recent = freq_recent_norm.get("シガーリーフ", 0) + freq_recent_norm.get("シガー", 0)
mint_early   = freq_early_norm.get("ミント", 0) + freq_early_norm.get("清涼感", 0)
mint_recent  = freq_recent_norm.get("ミント", 0) + freq_recent_norm.get("清涼感", 0)
fruit_early  = freq_early_norm.get("フルーツ", 0)
fruit_recent = freq_recent_norm.get("フルーツ", 0)

def change_pct(early, recent):
    if early == 0:
        return "+∞" if recent > 0 else "±0"
    pct = (recent - early) / early * 100
    return f"{pct:+.0f}%"

print(f"  ダークリーフ系: {dark_leaf_early:.3f} → {dark_leaf_recent:.3f}  ({change_pct(dark_leaf_early, dark_leaf_recent)})")
print(f"  シガーリーフ系: {cigar_early:.3f}  → {cigar_recent:.3f}   ({change_pct(cigar_early, cigar_recent)})")
print(f"  ミント・清涼系: {mint_early:.3f}  → {mint_recent:.3f}   ({change_pct(mint_early, mint_recent)})")
print(f"  フルーツ系:     {fruit_early:.3f}  → {fruit_recent:.3f}   ({change_pct(fruit_early, fruit_recent)})")

print(f"\n◆ 出力ファイル（{OUT_DIR}/）")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname} ({size_kb:.0f} KB)")

print("\n✓ 全処理完了")
