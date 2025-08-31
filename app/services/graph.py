from pyvis.network import Network
import networkx as nx
from typing import List, Dict, Tuple
import tempfile
import math
from .vector_store import get_embeddings_by_ids


def _cosine_sim(u: List[float], v: List[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0 or nv <= 0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def _pairwise_sims(ids: List[str], embs: Dict[str, List[float]]) -> Dict[Tuple[int, int], float]:
    sims: Dict[Tuple[int, int], float] = {}
    n = len(ids)
    for i in range(n):
        ei = embs.get(ids[i])
        if ei is None:
            continue
        for j in range(i + 1, n):
            ej = embs.get(ids[j])
            if ej is None:
                continue
            s = _cosine_sim(ei, ej)
            sims[(i, j)] = s
    return sims


def _build_mutual_knn_edges(ids: List[str], sims: Dict[Tuple[int, int], float], k: int, thr: float) -> List[Tuple[str, str, float]]:
    n = len(ids)
    # Collect neighbor lists from sims
    nbrs: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for (i, j), s in sims.items():
        if s >= thr:
            nbrs[i].append((j, s))
            nbrs[j].append((i, s))
    for i in range(n):
        nbrs[i].sort(key=lambda x: x[1], reverse=True)
        if k > 0:
            del nbrs[i][k:]
    # Keep mutual only
    edges: List[Tuple[str, str, float]] = []
    for i in range(n):
        picked = {j for j, _ in nbrs[i]}
        for j, s in nbrs[i]:
            if i in {ii for ii, _ in nbrs[j]}:
                a, b = ids[i], ids[j]
                if a < b:
                    edges.append((a, b, s))
    return edges


def build_graph(
    results: List[Dict],
    score_key: str = "score",
    sim_threshold: float = 0.75,
    knn: int = 5,
    include_tooltips: bool = True,
    min_visual_sim: float = 0.40,
):
    # Collect node ids and fetch embeddings from the vector store
    ids = [r["id"] for r in results]
    embs = get_embeddings_by_ids(ids)

    # Fallback: if embeddings missing for many nodes, degrade gracefully to distance-difference edges
    use_pairwise = len(embs) >= max(3, int(0.6 * len(ids)))

    G = nx.Graph()
    for r in results:
        nid = r["id"]
        title = r.get("meta", {}).get("title", nid)
        tooltip = None
        if include_tooltips:
            snippet = (r.get("text", "") or "")[:160].replace("\n", " ")
            tooltip = f"<b>{title}</b><br>{snippet}"
        if tooltip:
            G.add_node(nid, label=title, title=tooltip)
        else:
            G.add_node(nid, label=title)

    if use_pairwise and len(ids) >= 2:
        # Compute all pairwise similarities once
        sims = _pairwise_sims(ids, embs)
        # Strong edges: mutual kNN above threshold
        strong_edges = _build_mutual_knn_edges(ids, sims, k=max(1, int(knn)), thr=float(sim_threshold))
        strong_set = {(a, b) if a < b else (b, a) for a, b, _ in strong_edges}
        for a, b, s in strong_edges:
            # width/value scaled by similarity among strong range
            value = 3 + 5 * max(0.0, (s - sim_threshold) / max(1e-6, 1.0 - sim_threshold))
            G.add_edge(a, b, weight=s, value=value, color="rgba(60,60,60,0.85)", title=f"sim: {s:.2f}")
        # Weak edges: show faint links above min_visual_sim (not necessarily mutual), avoid duplicates
        for (i, j), s in sims.items():
            if s < min_visual_sim:
                continue
            a, b = ids[i], ids[j]
            key = (a, b) if a < b else (b, a)
            if key in strong_set:
                continue
            # Thin, dashed, transparent
            t = (s - min_visual_sim) / max(1e-6, (sim_threshold - min_visual_sim))
            t = max(0.0, min(1.0, t))
            alpha = 0.15 + 0.35 * t  # 0.15–0.50
            value = 1 + 3 * t        # 1–4
            G.add_edge(a, b, weight=s, value=value, color=f"rgba(120,120,120,{alpha:.2f})", dashes=True, title=f"sim: {s:.2f}")
    else:
        # Degrade to similarity based on distance difference (original behavior)
        for i, a in enumerate(results):
            for j in range(i + 1, len(results)):
                b = results[j]
                sa = a.get(score_key) or 0.5
                sb = b.get(score_key) or 0.5
                s = 1.0 - abs(sa - sb)
                if s >= 0.7:
                    value = 1 + 4 * (s - 0.7) / 0.3
                    G.add_edge(a["id"], b["id"], weight=s, value=value, color="rgba(120,120,120,0.5)", title=f"sim*: {s:.2f}")

    # Color by communities (greedy modularity); fallback to connected components
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    except Exception:
        comms = [set(c) for c in nx.connected_components(G)]
    for ci, comm in enumerate(comms):
        color = colors[ci % len(colors)]
        for n in comm:
            if n in G.nodes:
                G.nodes[n]["color"] = color

    # Node size by degree
    for n in G.nodes:
        deg = G.degree[n]
        G.nodes[n]["value"] = max(5, 10 + 2 * deg)

    net = Network(height="600px", width="100%", directed=False, notebook=False)
    net.from_nx(G)

    # Physics/layout tuning
    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.2,
              "springLength": 120,
              "springConstant": 0.03,
              "damping": 0.4,
              "avoidOverlap": 0.6
            }
          },
          "edges": {
            "smooth": { "type": "dynamic" },
            "color": { "inherit": false },
            "scaling": { "min": 1, "max": 6 }
          },
          "nodes": {
            "shape": "dot",
            "scaling": { "min": 5, "max": 30 },
            "font": { "size": 14 }
          },
          "layout": { "improvedLayout": true }
        }
        """
    )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp.name, open_browser=False, notebook=False)
    return tmp.name
