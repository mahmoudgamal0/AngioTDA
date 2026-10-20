"""
Build skeleton graphs with morphology-aware featurization.

Node features (7D): x, y, degree, is_junction, is_tip, mean_branch_length, mean_branch_angle
Edge features (8D): euclidean_dist, path_length, tortuosity, angle,
                    mean_tube_width, std_tube_width, mean_intensity, std_intensity
Graph stats  (10D): n_nodes, n_edges, n_junctions, n_tips, n_components,
                    mesh_count_euler, n_short_cycles, tube_coverage,
                    mean_edge_length, total_edge_length_per_node
"""

import math
import warnings
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label
from skimage.morphology import remove_small_objects, skeletonize

import skan

warnings.filterwarnings("ignore")

NODE_FEAT_DIM = 7
EDGE_FEAT_DIM = 8
GRAPH_STATS_DIM = 10
MIN_TUBE_AREA = 50

NODE_FEAT_NAMES = [
    "x", "y", "degree", "is_junction", "is_tip",
    "mean_branch_length", "mean_branch_angle",
]
EDGE_FEAT_NAMES = [
    "euclidean_distance", "path_length", "tortuosity", "angle",
    "mean_tube_width", "std_tube_width", "mean_intensity", "std_intensity",
]
GRAPH_STATS_NAMES = [
    "n_nodes", "n_edges", "n_junctions", "n_tips", "n_components",
    "mesh_count_euler", "n_short_cycles", "tube_coverage",
    "mean_edge_length", "total_edge_length_per_node",
]


def _branching_angle(G: nx.Graph, node: int, pos: np.ndarray) -> float:
    """Mean pairwise angle between incident edges at a node."""
    nbrs = list(G.neighbors(node))
    if len(nbrs) < 2:
        return 0.0
    p = pos[node]
    vecs = [pos[n] - p for n in nbrs]
    angles = []
    for v1, v2 in combinations(vecs, 2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angles.append(math.acos(cos_a))
    return float(np.mean(angles)) if angles else 0.0


def image_to_graph(img_norm: np.ndarray, img_raw_norm: np.ndarray, *,
                   threshold_scale: float = 1.0,
                   min_tube_area: int = MIN_TUBE_AREA) -> dict | None:
    """Extract graph with morphology-aware features from a normalized well image.

    Args:
        img_norm: normalized [0,1] image for thresholding/skeletonization
        img_raw_norm: same normalized image, used for intensity sampling along branches

    Returns dict with node_feat, edge_index, edge_attr, graph_stats, or None if empty.
    """
    if not np.isfinite(threshold_scale) or threshold_scale <= 0:
        raise ValueError("threshold_scale must be finite and positive")
    if min_tube_area < 1:
        raise ValueError("min_tube_area must be positive")
    H, W = img_norm.shape
    max_dim = max(H, W)

    # --- Segmentation and skeletonization ---
    thresh = threshold_otsu(img_norm) * threshold_scale
    mask = img_norm > thresh
    mask = remove_small_objects(mask, min_size=min_tube_area)
    if mask.sum() == 0:
        return None

    skel = skeletonize(mask)
    sk = skan.Skeleton(skel)
    bd = skan.summarize(sk, separator="-")

    if len(bd) == 0:
        return None

    # --- Distance transform for tube width ---
    dist_transform = ndimage.distance_transform_edt(mask)

    # --- Node extraction ---
    src_ids = bd["node-id-src"].values
    dst_ids = bd["node-id-dst"].values
    src_coords = bd[["coord-src-0", "coord-src-1"]].values
    dst_coords = bd[["coord-dst-0", "coord-dst-1"]].values

    all_ids = np.concatenate([src_ids, dst_ids])
    all_coords = np.vstack([src_coords, dst_coords])
    _, uniq_idx = np.unique(all_ids, return_index=True)
    unique_ids = all_ids[uniq_idx]
    unique_coords = all_coords[uniq_idx]

    N = len(unique_ids)
    if N == 0:
        return None

    id_to_idx = {nid: i for i, nid in enumerate(unique_ids)}

    # --- Normalized coordinates ---
    rows_norm = unique_coords[:, 0].astype(np.float32) / max(H - 1, 1)
    cols_norm = unique_coords[:, 1].astype(np.float32) / max(W - 1, 1)
    pos_norm = np.stack([cols_norm, rows_norm], axis=1)  # (N, 2) — x, y

    # --- Per-node degree and branch stats ---
    from collections import Counter
    deg_counter = Counter(all_ids.tolist())
    degrees = np.array([deg_counter[nid] for nid in unique_ids], dtype=np.float32)

    sum_len = np.zeros(N, dtype=np.float32)
    cnt_br = np.zeros(N, dtype=np.float32)
    seg_lengths_px = bd["branch-distance"].values.astype(np.float32)

    for k, (s, d) in enumerate(zip(src_ids, dst_ids)):
        si, di = id_to_idx[s], id_to_idx[d]
        bl = seg_lengths_px[k]
        sum_len[si] += bl; cnt_br[si] += 1
        sum_len[di] += bl; cnt_br[di] += 1

    mean_bl_norm = (sum_len / np.maximum(cnt_br, 1)) / max_dim

    # --- Build NetworkX graph for topology ---
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for s, d in zip(src_ids, dst_ids):
        G.add_edge(id_to_idx[s], id_to_idx[d])

    # --- Branch angle per node ---
    branch_angle = np.array([_branching_angle(G, i, pos_norm) for i in range(N)],
                            dtype=np.float32)

    # --- Node features (7D) ---
    node_feat = np.stack([
        cols_norm,                                    # 0: x
        rows_norm,                                    # 1: y
        degrees,                                      # 2: degree
        (degrees >= 3).astype(np.float32),            # 3: is_junction
        (degrees == 1).astype(np.float32),            # 4: is_tip
        mean_bl_norm,                                 # 5: mean_branch_length
        branch_angle,                                 # 6: mean_branch_angle
    ], axis=1).astype(np.float32)  # (N, 7)

    # --- Edge features (8D) with tube morphology ---
    E = len(bd)
    src_u = np.array([id_to_idx[s] for s in src_ids], dtype=np.int32)
    dst_u = np.array([id_to_idx[d] for d in dst_ids], dtype=np.int32)

    # Upper triangle for undirected edges
    upper = src_u < dst_u
    lower = ~upper

    # Collect unique undirected edges
    e_src = np.concatenate([src_u[upper], dst_u[lower]])
    e_dst = np.concatenate([dst_u[upper], src_u[lower]])
    e_branch_idx = np.concatenate([np.where(upper)[0], np.where(lower)[0]])
    n_unique_edges = len(e_src)

    edge_feats = np.zeros((n_unique_edges, EDGE_FEAT_DIM), dtype=np.float32)

    for i in range(n_unique_edges):
        bi = e_branch_idx[i]
        si, di = e_src[i], e_dst[i]

        # Euclidean distance (normalized)
        diff = pos_norm[di] - pos_norm[si]
        euclid = float(np.linalg.norm(diff))

        # Path length from skan (normalized)
        path_len_px = seg_lengths_px[bi]
        path_len_norm = path_len_px / max_dim

        # Tortuosity
        tortuosity = path_len_px / max(euclid * max_dim, 1.0)

        # Angle (bearing)
        angle = abs(float(np.arctan2(diff[1], diff[0])))

        # Tube width and intensity along branch path
        try:
            coords = sk.path_coordinates(bi)
            rc = coords.astype(int)
            # Clamp to image bounds
            rc[:, 0] = np.clip(rc[:, 0], 0, H - 1)
            rc[:, 1] = np.clip(rc[:, 1], 0, W - 1)

            widths = dist_transform[rc[:, 0], rc[:, 1]]
            intensities = img_raw_norm[rc[:, 0], rc[:, 1]]

            mean_width = float(widths.mean()) / max_dim
            std_width = float(widths.std()) / max_dim if len(widths) > 1 else 0.0
            mean_int = float(intensities.mean())
            std_int = float(intensities.std()) if len(intensities) > 1 else 0.0
        except Exception:
            mean_width, std_width = 0.0, 0.0
            mean_int, std_int = 0.0, 0.0

        edge_feats[i] = [euclid, path_len_norm, tortuosity, angle,
                         mean_width, std_width, mean_int, std_int]

    # Both directions for message passing
    edge_index = np.stack([
        np.concatenate([e_src, e_dst]),
        np.concatenate([e_dst, e_src]),
    ], axis=0).astype(np.int32)  # (2, 2*n_unique_edges)
    edge_attr = np.tile(edge_feats, (2, 1)).astype(np.float32)  # (2*n_unique_edges, 8)

    # --- Graph stats (10D) ---
    tube_area = float(mask.sum())
    tube_coverage = tube_area / (H * W)
    n_junctions = float((degrees >= 3).sum())
    n_tips = float((degrees == 1).sum())
    comps = list(nx.connected_components(G))
    n_comp = float(len(comps))

    n_topo = len(np.unique(np.vstack([src_coords, dst_coords]), axis=0))
    mesh_count = float(max(0, E - n_topo + len(comps)))

    # Short cycles
    n_short_cyc = 0.0
    try:
        DG = G.to_directed()
        seen = set()
        for cyc in nx.simple_cycles(DG, length_bound=5):
            if len(cyc) >= 3:
                key = tuple(sorted(cyc))
                if key not in seen:
                    seen.add(key)
                    n_short_cyc += 1.0
    except Exception:
        pass

    seg_dists = np.array([np.linalg.norm(pos_norm[e_dst[i]] - pos_norm[e_src[i]])
                          for i in range(n_unique_edges)], dtype=np.float32)
    mean_el = float(seg_dists.mean()) if len(seg_dists) > 0 else 0.0
    tot_el_per_node = float(seg_dists.sum()) / max(N, 1)

    graph_stats = np.array([
        float(N),           # 0: n_nodes
        float(E),           # 1: n_edges
        n_junctions,        # 2: n_junctions
        n_tips,             # 3: n_tips
        n_comp,             # 4: n_components
        mesh_count,         # 5: mesh_count_euler
        n_short_cyc,        # 6: n_short_cycles
        tube_coverage,      # 7: tube_coverage
        mean_el,            # 8: mean_edge_length
        tot_el_per_node,    # 9: total_edge_length_per_node
    ], dtype=np.float32)

    return {
        "node_feat": node_feat,        # (N, 7)
        "edge_index": edge_index,      # (2, 2E)
        "edge_attr": edge_attr,        # (2E, 8)
        "graph_stats": graph_stats,    # (10,)
        "n_nodes": N,
        "n_edges": n_unique_edges,
    }


def load_and_extract(tif_path: Path) -> dict | None:
    """Load a TIFF image and extract graph."""
    arr = np.array(Image.open(tif_path), dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    img_norm = (arr - lo) / (hi - lo + 1e-8)
    return image_to_graph(img_norm, img_norm)


def empty_graph() -> dict:
    """Return an all-zero graph for empty wells."""
    return {
        "node_feat": np.zeros((0, NODE_FEAT_DIM), dtype=np.float32),
        "edge_index": np.zeros((2, 0), dtype=np.int32),
        "edge_attr": np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
        "graph_stats": np.zeros(GRAPH_STATS_DIM, dtype=np.float32),
        "n_nodes": 0,
        "n_edges": 0,
    }
