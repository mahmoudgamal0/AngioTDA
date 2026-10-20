"""Augment graph_stats from 10D → 26D by computing missing NetworkX centrality/geometry stats.

Reads existing npz files, reconstructs NetworkX graph, computes all 26 stats from scratch.
Uses node_feat and edge_index/edge_attr (not old graph_stats) so it's idempotent.

Usage:
  conda run -n mg python scripts/augment_graph_stats.py
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np

from angiotda import PROJECT_ROOT
GRAPHS_DIR = PROJECT_ROOT / "data/graphs"


def compute_extended_stats(npz_path):
    """Compute 26D graph_stats for one graph file from node_feat + edge data."""
    data = dict(np.load(str(npz_path), allow_pickle=True))

    nf_key = "node_feat" if "node_feat" in data else "node_feat_slim"
    node_feat = data[nf_key]
    edge_index = data["edge_index"]
    edge_attr = data["edge_attr"]
    N = int(data["n_nodes"])

    if N == 0:
        data["graph_stats"] = np.zeros(26, dtype=np.float32)
        np.savez_compressed(str(npz_path), **data)
        return str(npz_path)

    # Reconstruct NetworkX graph from edge_index (bidirectional, take unique)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    seg_dists = []    # euclidean distances
    seg_lengths = []  # path lengths
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i < j:
            dist = float(edge_attr[k, 0]) if edge_attr.shape[1] > 0 else 1.0
            G.add_edge(i, j, weight=dist)
            seg_dists.append(dist)
            if edge_attr.shape[1] > 1:
                seg_lengths.append(float(edge_attr[k, 1]))
    seg_dists = np.array(seg_dists, dtype=np.float32) if seg_dists else np.array([], dtype=np.float32)
    seg_lengths = np.array(seg_lengths, dtype=np.float32) if seg_lengths else seg_dists.copy()

    n_unique_edges = len(seg_dists)

    # Node type counts from node_feat (col 3 = is_junction, col 4 = is_tip)
    is_junction = node_feat[:, 3] > 0.5
    is_tip = node_feat[:, 4] > 0.5
    n_junctions = float(is_junction.sum())
    n_tips = float(is_tip.sum())

    # Degree stats
    degrees = np.array([G.degree(i) for i in range(N)], dtype=np.float32)
    n_branch = float((degrees >= 3).sum())
    n_endpt = float((degrees == 1).sum())
    n_isol = float((degrees == 0).sum())
    max_e = N * (N - 1) / 2 if N > 1 else 1.0
    density = n_unique_edges / max_e

    # Components
    comps = list(nx.connected_components(G))
    n_comp = float(len(comps))
    largest_frac = float(max(len(c) for c in comps) / N) if comps else 0.0

    # Euler mesh count: E - N + C
    mesh_count = max(0.0, float(n_unique_edges - N + n_comp))

    # Short cycles (3-5) — count from NetworkX
    try:
        short_cycles = [c for c in nx.cycle_basis(G) if len(c) <= 5]
        n_short_cyc = float(len(short_cycles))
    except Exception:
        n_short_cyc = 0.0

    # Tube coverage — fraction of image covered; approximate from total edge length
    # Normalised edge lengths sum / area (1.0 for normalised coords)
    tube_coverage = float(seg_dists.sum()) if len(seg_dists) > 0 else 0.0

    # Edge length stats
    mean_el = float(seg_dists.mean()) if len(seg_dists) > 0 else 0.0
    std_el = float(seg_dists.std()) if len(seg_dists) > 1 else 0.0
    tot_el = float(seg_dists.sum()) / max(N, 1)

    # Branch length stats
    mean_bl = float(seg_lengths.mean()) if len(seg_lengths) > 0 else 0.0
    std_bl = float(seg_lengths.std()) if len(seg_lengths) > 1 else 0.0
    mean_tort = float(seg_lengths.mean() / max(seg_dists.mean(), 1e-8)) if len(seg_dists) > 0 else 0.0
    branch_index = n_branch / max(tot_el * N, 1e-8)

    # Centrality measures
    try:
        betweenness = np.array(list(nx.betweenness_centrality(G).values()), dtype=np.float32)
    except Exception:
        betweenness = np.zeros(N, dtype=np.float32)

    try:
        closeness = np.array(list(nx.closeness_centrality(G).values()), dtype=np.float32)
    except Exception:
        closeness = np.zeros(N, dtype=np.float32)

    # Eccentricity per connected component
    try:
        ecc_vals = np.zeros(N, dtype=np.float32)
        for comp in nx.connected_components(G):
            if len(comp) >= 2:
                sg = G.subgraph(comp)
                ecc_dict = nx.eccentricity(sg)
                for node, ecc in ecc_dict.items():
                    ecc_vals[node] = ecc
        eccentricity = ecc_vals
    except Exception:
        eccentricity = np.zeros(N, dtype=np.float32)

    try:
        clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
    except Exception:
        clustering = np.zeros(N, dtype=np.float32)

    def _ms(arr):
        return float(arr.mean()), float(arr.std() if len(arr) > 1 else 0.0)

    m_bc, s_bc = _ms(betweenness)
    m_cc, s_cc = _ms(closeness)
    m_ec = float(eccentricity.mean())
    m_cl = float(clustering.mean())

    # Build 26D stats (matching v1 schema)
    new_stats = np.array([
        float(N), float(n_unique_edges), n_branch, n_endpt, n_isol,       # 0-4
        float(degrees.mean()), float(degrees.std() if N > 1 else 0.0),    # 5-6
        density, n_comp, largest_frac,                                      # 7-9
        mean_el, std_el, tot_el,                                            # 10-12
        mesh_count, n_short_cyc,                                            # 13-14
        tube_coverage,                                                      # 15
        mean_bl, std_bl, mean_tort,                                         # 16-18
        m_bc, s_bc, m_cc, s_cc, m_ec, m_cl,                               # 19-24
        branch_index,                                                       # 25
    ], dtype=np.float32)

    data["graph_stats"] = new_stats
    np.savez_compressed(str(npz_path), **data)
    return str(npz_path)


def main():
    npz_files = sorted(GRAPHS_DIR.glob("*.npz"))
    print(f"Found {len(npz_files)} graph files")

    d = np.load(str(npz_files[0]), allow_pickle=True)
    print(f"Current graph_stats dim: {d['graph_stats'].shape[0]}")

    n_updated = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(compute_extended_stats, npz_files)):
            if result is not None:
                n_updated += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(npz_files)} processed, {n_updated} updated")

    print(f"\nDone. Updated {n_updated}/{len(npz_files)} files.")

    # Verify
    d = np.load(str(npz_files[0]), allow_pickle=True)
    print(f"New graph_stats dim: {d['graph_stats'].shape[0]}")
    print(f"Stats: {d['graph_stats']}")


if __name__ == "__main__":
    main()
