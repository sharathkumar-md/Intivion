"""
Graph-based feature extraction using NetworkX.

The motivation here is Section 8.5 — collusion rings and IP hub behaviour
can't be caught by looking at users individually. You need to look at the
network structure. So I build a bipartite graph: users connected to IPs
and devices, weighted by how often they appear together.

Then I run:
- Connected components to find clusters
- Louvain community detection to catch fraud rings
- PageRank to score hub importance
- Degree centrality as a simpler connectivity measure

The key feature that actually fires on collusion rings is `shared_ip_user_count`
— how many other users share any of your IPs. Ring users score 4.0 on average
vs 0.0 for normals.
"""

import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community as nx_community

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / "features.log", mode="a", encoding="utf-8"),
    ],
)
_sh = logging.root.handlers[0]
if hasattr(_sh, "stream"):
    _sh.stream.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger("forexguard.features.graph")


def build_graph(events_df: pd.DataFrame) -> nx.Graph:
    """
    Construct a weighted bipartite graph:
      - Nodes: user_ids, ip_addresses, device_fingerprints
      - Edges: user–IP (weight = login count), user–device (weight = event count)
    """
    log.info("Building user-IP-device graph...")

    G = nx.Graph()

    # --- Nodes ---
    users   = events_df["user_id"].unique()
    ips     = events_df["ip_address"].unique()
    devices = events_df["device_fingerprint"].unique()

    for u in users:
        G.add_node(u, node_type="user")
    for ip in ips:
        G.add_node(ip, node_type="ip")
    for d in devices:
        G.add_node(d, node_type="device")

    # --- User–IP edges ---
    user_ip = (
        events_df.groupby(["user_id", "ip_address"])
        .size()
        .reset_index(name="weight")
    )
    for _, row in user_ip.iterrows():
        G.add_edge(row["user_id"], row["ip_address"],
                   weight=int(row["weight"]), edge_type="user_ip")

    # --- User–Device edges ---
    user_dev = (
        events_df.groupby(["user_id", "device_fingerprint"])
        .size()
        .reset_index(name="weight")
    )
    for _, row in user_dev.iterrows():
        G.add_edge(row["user_id"], row["device_fingerprint"],
                   weight=int(row["weight"]), edge_type="user_device")

    log.info(
        "Graph built: %d nodes (%d users, %d IPs, %d devices), %d edges",
        G.number_of_nodes(), len(users), len(ips), len(devices),
        G.number_of_edges(),
    )
    return G


def extract_graph_features(
    G: nx.Graph,
    user_ids: list[str],
) -> pd.DataFrame:
    """
    Extract per-user graph features for ML input.

    Features
    --------
    shared_ip_user_count   : # distinct users sharing any IP with this user (8.5)
    degree_centrality      : normalised degree in the full graph
    pagerank_score         : PageRank over the full graph
    component_size         : size of the connected component this user belongs to
    community_id           : Louvain community id
    community_size         : size of the Louvain community
    is_hub_neighbor        : 1 if any of the user's IPs has >3 users connected
    """
    log.info("Extracting graph features for %d users...", len(user_ids))

    # --- Connected components ---
    log.info("  Computing connected components...")
    component_map: dict[str, int]  = {}   # node -> component_id
    component_size: dict[str, int] = {}   # node -> size
    for i, comp in enumerate(nx.connected_components(G)):
        sz = len(comp)
        for node in comp:
            component_map[node]  = i
            component_size[node] = sz

    # --- PageRank ---
    log.info("  Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200)

    # --- Degree centrality ---
    log.info("  Computing degree centrality...")
    deg_centrality = nx.degree_centrality(G)

    # --- Louvain community detection ---
    log.info("  Running Louvain community detection...")
    louvain_comms = nx_community.louvain_communities(G, seed=42, weight="weight")
    louvain_map: dict[str, int]  = {}
    louvain_size: dict[str, int] = {}
    for i, comm in enumerate(louvain_comms):
        sz = len(comm)
        for node in comm:
            louvain_map[node]  = i
            louvain_size[node] = sz

    # --- IP-hub neighbors ---
    # Count distinct users reachable through each IP node
    log.info("  Computing IP-hub neighbours...")
    ip_user_count: dict[str, int] = {}   # ip -> count of user neighbours
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "ip":
            ip_user_count[node] = sum(
                1 for n in G.neighbors(node)
                if G.nodes[n].get("node_type") == "user"
            )

    records = []
    for uid in user_ids:
        if uid not in G:
            records.append({
                "user_id": uid,
                "shared_ip_user_count": 0,
                "degree_centrality": 0.0,
                "pagerank_score": 0.0,
                "component_size": 1,
                "community_id": -1,
                "community_size": 1,
                "is_hub_neighbor": 0,
            })
            continue

        # Users reachable via shared IPs (exclude self)
        shared_users: set[str] = set()
        hub_flag = 0
        for neighbor in G.neighbors(uid):
            if G.nodes[neighbor].get("node_type") == "ip":
                ip_users = ip_user_count.get(neighbor, 0)
                if ip_users > 3:
                    hub_flag = 1
                for n2 in G.neighbors(neighbor):
                    if G.nodes[n2].get("node_type") == "user" and n2 != uid:
                        shared_users.add(n2)

        records.append({
            "user_id":             uid,
            "shared_ip_user_count": len(shared_users),
            "degree_centrality":    deg_centrality.get(uid, 0.0),
            "pagerank_score":       pagerank.get(uid, 0.0),
            "component_size":       component_size.get(uid, 1),
            "community_id":         louvain_map.get(uid, -1),
            "community_size":       louvain_size.get(uid, 1),
            "is_hub_neighbor":      hub_flag,
        })

    result = pd.DataFrame(records).set_index("user_id")
    log.info("Graph features shape: %s", result.shape)
    return result


def build_graph_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: raw events -> per-user graph features DataFrame."""
    G        = build_graph(events_df)
    user_ids = sorted(events_df["user_id"].unique())
    features = extract_graph_features(G, user_ids)
    return features, G


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    raw_dir   = Path(__file__).parent.parent / "data" / "raw"
    events_df = pd.read_parquet(raw_dir / "events.parquet")

    graph_features, G = build_graph_features(events_df)

    out_path = raw_dir / "features_graph.parquet"
    graph_features.to_parquet(out_path)

    # Save graph for later reuse
    graph_path = raw_dir / "user_graph.gpickle"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    log.info("Saved graph features -> %s", out_path)
    log.info("Saved graph object   -> %s", graph_path)

    # Sanity: collusion ring users should have high shared_ip_user_count
    labels_df = pd.read_parquet(raw_dir / "labels.parquet")
    ring_users = labels_df[labels_df["anomaly_type"] == "collusion_ring"]["user_id"]
    if len(ring_users):
        ring_scores = graph_features.loc[
            graph_features.index.isin(ring_users), "shared_ip_user_count"
        ]
        log.info(
            "Collusion ring users: mean shared_ip_user_count = %.2f (normal users mean = %.2f)",
            ring_scores.mean(),
            graph_features.loc[
                ~graph_features.index.isin(ring_users), "shared_ip_user_count"
            ].mean(),
        )
    log.info("Graph feature extraction complete [OK]")
    print(graph_features.describe().to_string())
