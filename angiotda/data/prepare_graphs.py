"""Extract a graph cache and a verifiable manifest from externally supplied TIFFs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from angiotda import PROJECT_ROOT, CONTROLS
from angiotda.data.paths import DATASET
from angiotda.data.build_graphs import image_to_graph
from angiotda.data.augment_graph_stats import compute_extended_stats
from angiotda.training.splits import atomic_write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DATASET.sample_index)
    parser.add_argument("--output", type=Path, default=DATASET.graphs_dir)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-component", type=int, default=50)
    args = parser.parse_args()
    if args.threshold_scale <= 0 or not np.isfinite(args.threshold_scale) or args.min_component < 1:
        parser.error("threshold-scale must be finite and positive; min-component must be positive")
    index = args.index if args.index.is_absolute() else PROJECT_ROOT/args.index
    output = args.output if args.output.is_absolute() else PROJECT_ROOT/args.output
    samples = sorted([s for s in json.loads(index.read_text())["samples"]
                      if s["compound"] not in CONTROLS], key=lambda s: s["image_name"])
    if not samples: raise ValueError("index contains no non-control wells")
    manifest_path = output/"manifest.json"
    config = dict(threshold_scale=args.threshold_scale, min_component_pixels=args.min_component,
                  index_sha256=sha(index))
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None and previous.get("configuration") != config:
        raise RuntimeError("cache settings changed; choose a new output directory")
    if previous is None and output.exists() and any(output.iterdir()):
        raise RuntimeError("output is not an empty cache; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="partial", scope="all_noncontrol_wells", configuration=config, wells=[])
    cached = {r["image_name"]:r for r in (previous or {}).get("wells",[])}
    for sample in samples:
        name = sample["image_name"]
        if Path(name).name != name: raise ValueError("image_name must be a file stem")
        source = Path(sample["image_path"])
        if not source.is_absolute(): source = PROJECT_ROOT/source
        target = output/f"{name}.npz"
        source_hash = sha(source)
        old = cached.get(name)
        if old and old.get("source_sha256") == source_hash and target.exists() and old.get("graph_sha256") == sha(target):
            manifest["wells"].append(old)
            continue
        raw = np.asarray(Image.open(source),dtype=np.float32)
        if raw.ndim != 2 or not np.isfinite(raw).all(): raise ValueError(f"invalid grayscale image: {name}")
        image = (raw-raw.min())/(raw.max()-raw.min()+1e-8)
        graph = image_to_graph(image,image,threshold_scale=args.threshold_scale,min_tube_area=args.min_component)
        if graph is None or graph["n_nodes"] < 1: raise ValueError(f"empty graph: {name}")
        if not all(np.isfinite(graph[k]).all() for k in ("node_feat","edge_attr","graph_stats")):
            raise ValueError(f"nonfinite graph: {name}")
        np.savez_compressed(target,**graph)
        compute_extended_stats(target)
        manifest["wells"].append(dict(image_name=name,source_sha256=source_hash,
                                     graph_sha256=sha(target),nodes=int(graph["n_nodes"]),edges=int(graph["n_edges"])))
        atomic_write_json(manifest_path,manifest)
        print(f"{len(manifest['wells'])}/{len(samples)} {name}",flush=True)
    manifest["status"]="complete"
    atomic_write_json(manifest_path,manifest)
    print(manifest_path)

if __name__ == "__main__": main()
