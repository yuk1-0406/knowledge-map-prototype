from pyvis.network import Network
import networkx as nx
from typing import List, Dict
import tempfile


def build_graph(results: List[Dict], score_key: str = "score", threshold: float = 0.7):
    G = nx.Graph()
    for r in results:
        nid = r["id"]
        title = r.get("meta", {}).get("title", nid)
        G.add_node(nid, label=title)

    for i, a in enumerate(results):
        for j in range(i + 1, len(results)):
            b = results[j]
            sa = a.get(score_key) or 0.5
            sb = b.get(score_key) or 0.5
            s = 1.0 - abs(sa - sb)
            if s >= threshold:
                G.add_edge(a["id"], b["id"], weight=s)

    net = Network(height="600px", width="100%", directed=False, notebook=False)
    net.from_nx(G)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    # Write HTML without notebook integration to avoid Jinja2-related issues
    net.write_html(tmp.name, open_browser=False, notebook=False)
    return tmp.name
