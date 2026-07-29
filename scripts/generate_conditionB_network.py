#!/usr/bin/env python3
"""Generate a poster-ready cooccurrence network for Condition B."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from adjustText import adjust_text
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

plt.rcParams["font.family"] = "TakaoGothic"
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
POSTER_DIR = ROOT / "poster_analysis"
SUMMARY_MD = POSTER_DIR / "summary.md"
REVIEW_EXTRACTION_CSV = POSTER_DIR / "review_extraction_summary.csv"

FIGURE_PNG = POSTER_DIR / "figure_network_conditionB.png"
FIGURE_PDF = POSTER_DIR / "figure_network_conditionB.pdf"
ALL_LABELS_PNG = POSTER_DIR / "figure_network_conditionB_all_labels.png"
ALL_LABELS_PDF = POSTER_DIR / "figure_network_conditionB_all_labels.pdf"
MAJOR_LABELS_PNG = POSTER_DIR / "figure_network_conditionB_major_labels.png"
MAJOR_LABELS_PDF = POSTER_DIR / "figure_network_conditionB_major_labels.pdf"
NODE_METRICS_CSV = POSTER_DIR / "network_node_metrics.csv"
EDGE_METRICS_CSV = POSTER_DIR / "network_edge_metrics.csv"

MIN_PAIR_COUNT = 2
MIN_FLAVORS = 2
MAX_FLAVORS = 5
TOP_LABEL_COUNT = 15
TOP_MAJOR_FONT_COUNT = 10
SUMMARY_START = "<!-- conditionB_network:start -->"
SUMMARY_END = "<!-- conditionB_network:end -->"

COMMUNITY_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


def parse_flavor_list(serialized: object) -> list[str]:
    """Parse the stored pipe-separated flavor list."""
    if not isinstance(serialized, str) or not serialized:
        return []
    return [item for item in serialized.split("|") if item]


def load_condition_b_reviews() -> pd.DataFrame:
    """Load normalized review extraction rows and filter to Condition B."""
    df = pd.read_csv(REVIEW_EXTRACTION_CSV)
    filtered = df[
        (df["flavor_count"] >= MIN_FLAVORS)
        & (df["flavor_count"] <= MAX_FLAVORS)
    ].copy()
    return filtered.reset_index(drop=True)


def build_counters(filtered_df: pd.DataFrame) -> tuple[Counter, Counter]:
    """Build document-frequency and pair-cooccurrence counters."""
    node_counter: Counter = Counter()
    pair_counter: Counter = Counter()

    for serialized in filtered_df["extracted_flavors"]:
        flavors = sorted(set(parse_flavor_list(serialized)))
        for flavor in flavors:
            node_counter[flavor] += 1
        for pair in combinations(flavors, 2):
            pair_counter[pair] += 1

    return node_counter, pair_counter


def build_graph(
    *,
    node_counter: Counter,
    pair_counter: Counter,
    n_reviews: int,
) -> nx.Graph:
    """Construct the thresholded cooccurrence graph."""
    graph = nx.Graph()
    for pair, count in pair_counter.items():
        if count < MIN_PAIR_COUNT:
            continue
        left, right = pair
        graph.add_node(left, review_frequency=node_counter[left])
        graph.add_node(right, review_frequency=node_counter[right])
        graph.add_edge(
            left,
            right,
            weight=count,
            support=count / n_reviews if n_reviews else 0.0,
        )
    return graph


def detect_communities(graph: nx.Graph) -> dict[str, int]:
    """Detect node communities with Louvain when available."""
    if graph.number_of_nodes() == 0:
        return {}

    try:
        raw_communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            seed=42,
        )
    except Exception:
        raw_communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))

    ordered = sorted(
        raw_communities,
        key=lambda members: (
            -sum(graph.degree(node, weight="weight") for node in members),
            min(members),
        ),
    )
    partition: dict[str, int] = {}
    for community_id, members in enumerate(ordered):
        for node in sorted(members):
            partition[node] = community_id
    return partition


def build_distance_graph(graph: nx.Graph) -> nx.Graph:
    """Create a graph whose edge weight behaves like distance."""
    distance_graph = nx.Graph()
    for left, right, data in graph.edges(data=True):
        weight = float(data["weight"])
        distance_graph.add_edge(left, right, distance=1.0 / weight if weight else 1.0)
    return distance_graph


def build_node_metrics(
    *,
    graph: nx.Graph,
    partition: dict[str, int],
) -> pd.DataFrame:
    """Build node-level network metrics."""
    degree_dict = dict(graph.degree())
    weighted_degree_dict = dict(graph.degree(weight="weight"))
    distance_graph = build_distance_graph(graph)
    betweenness = nx.betweenness_centrality(
        distance_graph,
        weight="distance",
        normalized=True,
    )

    rows = []
    for node in sorted(graph.nodes()):
        rows.append(
            {
                "flavor": node,
                "review_frequency": int(graph.nodes[node]["review_frequency"]),
                "degree": int(degree_dict[node]),
                "weighted_degree": float(weighted_degree_dict[node]),
                "betweenness_centrality": float(betweenness[node]),
                "community_id": int(partition[node]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["weighted_degree", "degree", "review_frequency", "flavor"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_edge_metrics(
    *,
    graph: nx.Graph,
    partition: dict[str, int],
) -> pd.DataFrame:
    """Build edge-level metrics for exported CSV."""
    rows = []
    for left, right, data in graph.edges(data=True):
        rows.append(
            {
                "flavor_a": left,
                "flavor_b": right,
                "pair_key": f"{left}||{right}",
                "cooccurrence_count": int(data["weight"]),
                "support": float(data["support"]),
                "same_community": partition[left] == partition[right],
                "community_a": int(partition[left]),
                "community_b": int(partition[right]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cooccurrence_count", "flavor_a", "flavor_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def label_positions(
    *,
    pos: dict[str, tuple[float, float]],
    label_nodes: list[str],
    extra_scale: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Offset label positions slightly away from the graph center."""
    if not label_nodes:
        return {}

    center_x = sum(pos[node][0] for node in label_nodes) / len(label_nodes)
    center_y = sum(pos[node][1] for node in label_nodes) / len(label_nodes)
    adjusted = {}

    for idx, node in enumerate(label_nodes):
        x, y = pos[node]
        dx = x - center_x
        dy = y - center_y
        norm = math.hypot(dx, dy) or 1.0
        radial_x = dx / norm
        radial_y = dy / norm
        offset = extra_scale * (0.035 + 0.006 * (idx % 4))
        adjusted[node] = (x + radial_x * offset, y + radial_y * offset)
    return adjusted


def draw_labels(
    *,
    ax: plt.Axes,
    pos: dict[str, tuple[float, float]],
    node_metrics_df: pd.DataFrame,
    label_nodes: list[str],
    all_labels: bool,
) -> None:
    """Draw labels with adjustText and thin leader lines."""
    weighted_degree_rank = {
        flavor: rank
        for rank, flavor in enumerate(node_metrics_df["flavor"].tolist(), start=1)
    }
    initial_positions = label_positions(
        pos=pos,
        label_nodes=label_nodes,
        extra_scale=1.35 if all_labels else 1.15,
    )

    texts = []
    x_points = []
    y_points = []
    for node in label_nodes:
        x, y = initial_positions[node]
        rank = weighted_degree_rank[node]
        font_size = 10.5 if rank <= TOP_MAJOR_FONT_COUNT else 7.6
        if not all_labels and rank > TOP_MAJOR_FONT_COUNT:
            font_size = 8.0
        text = ax.text(
            x,
            y,
            node,
            fontsize=font_size,
            fontweight="bold" if rank <= TOP_MAJOR_FONT_COUNT else "normal",
            ha="center",
            va="center",
            color="#1E1E1E",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.84,
            },
            zorder=12,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        texts.append(text)
        x_points.append(pos[node][0])
        y_points.append(pos[node][1])

    adjust_text(
        texts,
        x=x_points,
        y=y_points,
        ax=ax,
        avoid_points=True,
        avoid_self=True,
        expand=(1.12, 1.22) if all_labels else (1.08, 1.16),
        force_text=(0.45, 0.75) if all_labels else (0.3, 0.55),
        force_static=(0.25, 0.35),
        force_pull=(0.015, 0.03),
        max_move=(30, 30),
        ensure_inside_axes=False,
        arrowprops={
            "arrowstyle": "-",
            "color": "#7A7A7A",
            "lw": 0.45,
            "alpha": 0.65,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )


def render_network_figure(
    *,
    graph: nx.Graph,
    node_metrics_df: pd.DataFrame,
    partition: dict[str, int],
    output_png: Path,
    output_pdf: Path,
    all_labels: bool,
    pos: dict[str, tuple[float, float]],
) -> None:
    """Render one network figure variant."""
    fig, ax = plt.subplots(figsize=(19, 15), dpi=320)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    weighted_degrees = node_metrics_df.set_index("flavor")["weighted_degree"].to_dict()
    max_weighted_degree = max(weighted_degrees.values()) if weighted_degrees else 1.0
    min_weighted_degree = min(weighted_degrees.values()) if weighted_degrees else 0.0

    node_sizes = []
    node_colors = []
    for node in graph.nodes():
        base = weighted_degrees[node]
        scaled = (base - min_weighted_degree) / max(max_weighted_degree - min_weighted_degree, 1.0)
        node_sizes.append(280 + 3000 * (scaled ** 0.7))
        node_colors.append(COMMUNITY_COLORS[partition[node] % len(COMMUNITY_COLORS)])

    edge_weights = [float(data["weight"]) for _, _, data in graph.edges(data=True)]
    max_edge_weight = max(edge_weights) if edge_weights else 1.0

    same_edges = []
    same_widths = []
    same_colors = []
    cross_edges = []
    cross_widths = []
    for left, right, data in graph.edges(data=True):
        width = 0.8 + 5.0 * (float(data["weight"]) / max_edge_weight) ** 0.8
        if partition[left] == partition[right]:
            same_edges.append((left, right))
            same_widths.append(width)
            same_colors.append(COMMUNITY_COLORS[partition[left] % len(COMMUNITY_COLORS)])
        else:
            cross_edges.append((left, right))
            cross_widths.append(width)

    if cross_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=cross_edges,
            width=cross_widths,
            edge_color="#C7CCD1",
            alpha=0.35,
        )
    if same_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=same_edges,
            width=same_widths,
            edge_color=same_colors,
            alpha=0.55,
        )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=1.3,
        alpha=0.92,
    )

    label_nodes = (
        node_metrics_df["flavor"].tolist()
        if all_labels
        else node_metrics_df.head(TOP_LABEL_COUNT)["flavor"].tolist()
    )
    draw_labels(
        ax=ax,
        pos=pos,
        node_metrics_df=node_metrics_df,
        label_nodes=label_nodes,
        all_labels=all_labels,
    )

    ax.set_title(
        "レビュー内フレーバー共起ネットワーク（Condition B）",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        -0.06,
        "注記: エッジは同一レビュー内での共起を表し、実際のミックスを直接意味しない",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

    ax.set_axis_off()
    ax.margins(0.24 if all_labels else 0.18)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.97))
    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def update_summary(
    *,
    node_metrics_df: pd.DataFrame,
    edge_metrics_df: pd.DataFrame,
    community_count: int,
) -> None:
    """Append or replace the Condition B network summary in summary.md."""
    if SUMMARY_MD.exists():
        original = SUMMARY_MD.read_text(encoding="utf-8")
    else:
        original = "# poster_analysis summary\n"

    top5 = node_metrics_df.head(5)
    lines = [
        SUMMARY_START,
        "## 17. Condition B 共起ネットワーク",
        f"- ノード数: {len(node_metrics_df)}",
        f"- エッジ数: {len(edge_metrics_df)}",
        f"- 使用した閾値: Condition B（抽出フレーバー数 2〜5）、名称正規化後、min_pair_count={MIN_PAIR_COUNT}",
        f"- コミュニティ数: {community_count}",
        "- 上位5ノード（degree / weighted degree / betweenness centrality）:",
    ]
    for _, row in top5.iterrows():
        lines.append(
            "  - "
            f"{row['flavor']}: degree={int(row['degree'])}, "
            f"weighted_degree={float(row['weighted_degree']):.1f}, "
            f"betweenness={float(row['betweenness_centrality']):.4f}"
        )
    lines.extend(
        [
            "- 既存図との差分:",
            "  - 既存の `notebooks/flavor_mix_network.py` は全レビューを対象にし、出現頻度 `>=3` と `MAX_NODES=55` を併用した旧ネットワークだった。",
            "  - 今回の図は `poster_analysis/review_extraction_summary.csv` の正規化後データから Condition B のみを抽出し、エッジ閾値を `min_pair_count=2` に統一している。",
            "  - 背景を白基調に変更し、ノード色はコミュニティ、ノードサイズは weighted degree、エッジ幅は共起回数に比例させてポスター用に再設計した。",
        ]
    )
    lines.append(SUMMARY_END)
    block = "\n".join(lines)

    import re

    pattern = re.compile(
        rf"{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}",
        flags=re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(block, original)
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    SUMMARY_MD.write_text(updated, encoding="utf-8")


def run_self_checks() -> list[str]:
    """Run basic checks on the graph-building logic."""
    toy = pd.DataFrame(
        [
            {"extracted_flavors": "A|B|C", "flavor_count": 3},
            {"extracted_flavors": "A|B", "flavor_count": 2},
            {"extracted_flavors": "B|C", "flavor_count": 2},
        ]
    )
    node_counter, pair_counter = build_counters(toy)
    messages = []

    assert node_counter["A"] == 2
    assert node_counter["B"] == 3
    messages.append("PASS: Condition B レビューからノード出現回数を数えられる")

    assert pair_counter[("A", "B")] == 2
    assert pair_counter[("B", "C")] == 2
    messages.append("PASS: レビュー単位の共起回数を数えられる")

    graph = build_graph(node_counter=node_counter, pair_counter=pair_counter, n_reviews=len(toy))
    assert graph.number_of_edges() == 2
    messages.append("PASS: min_pair_count=2 の閾値でエッジを抽出できる")

    partition = detect_communities(graph)
    assert set(partition) == set(graph.nodes())
    messages.append("PASS: 各ノードにコミュニティIDを付与できる")

    node_metrics_df = build_node_metrics(graph=graph, partition=partition)
    assert {"degree", "weighted_degree", "betweenness_centrality"} <= set(node_metrics_df.columns)
    messages.append("PASS: ノード指標を出力できる")

    return messages


def main() -> None:
    filtered_df = load_condition_b_reviews()
    node_counter, pair_counter = build_counters(filtered_df)
    graph = build_graph(
        node_counter=node_counter,
        pair_counter=pair_counter,
        n_reviews=len(filtered_df),
    )
    partition = detect_communities(graph)
    node_metrics_df = build_node_metrics(graph=graph, partition=partition)
    edge_metrics_df = build_edge_metrics(graph=graph, partition=partition)
    pos = nx.spring_layout(
        graph,
        weight="weight",
        k=1.9 / math.sqrt(max(graph.number_of_nodes(), 1)),
        iterations=400,
        seed=42,
    )

    node_metrics_df.to_csv(NODE_METRICS_CSV, index=False, encoding="utf-8-sig")
    edge_metrics_df.to_csv(EDGE_METRICS_CSV, index=False, encoding="utf-8-sig")
    render_network_figure(
        graph=graph,
        node_metrics_df=node_metrics_df,
        partition=partition,
        output_png=ALL_LABELS_PNG,
        output_pdf=ALL_LABELS_PDF,
        all_labels=True,
        pos=pos,
    )
    render_network_figure(
        graph=graph,
        node_metrics_df=node_metrics_df,
        partition=partition,
        output_png=MAJOR_LABELS_PNG,
        output_pdf=MAJOR_LABELS_PDF,
        all_labels=False,
        pos=pos,
    )
    MAJOR_LABELS_PNG.replace(FIGURE_PNG)
    MAJOR_LABELS_PDF.replace(FIGURE_PDF)
    render_network_figure(
        graph=graph,
        node_metrics_df=node_metrics_df,
        partition=partition,
        output_png=MAJOR_LABELS_PNG,
        output_pdf=MAJOR_LABELS_PDF,
        all_labels=False,
        pos=pos,
    )
    update_summary(
        node_metrics_df=node_metrics_df,
        edge_metrics_df=edge_metrics_df,
        community_count=(max(partition.values()) + 1) if partition else 0,
    )

    test_messages = run_self_checks()

    print("conditionB network outputs generated")
    print(f"- reviews used: {len(filtered_df)}")
    print(f"- nodes: {graph.number_of_nodes()}")
    print(f"- edges: {graph.number_of_edges()}")
    print(f"- min_pair_count: {MIN_PAIR_COUNT}")
    print(f"- communities: {(max(partition.values()) + 1) if partition else 0}")
    print("- output files:")
    print(f"  - {ALL_LABELS_PNG.relative_to(ROOT)}")
    print(f"  - {ALL_LABELS_PDF.relative_to(ROOT)}")
    print(f"  - {MAJOR_LABELS_PNG.relative_to(ROOT)}")
    print(f"  - {MAJOR_LABELS_PDF.relative_to(ROOT)}")
    print(f"  - {FIGURE_PNG.relative_to(ROOT)}")
    print(f"  - {FIGURE_PDF.relative_to(ROOT)}")
    print(f"  - {NODE_METRICS_CSV.relative_to(ROOT)}")
    print(f"  - {EDGE_METRICS_CSV.relative_to(ROOT)}")
    print(f"  - {SUMMARY_MD.relative_to(ROOT)}")
    print("- tests:")
    for message in test_messages:
        print(f"  - {message}")


if __name__ == "__main__":
    main()
