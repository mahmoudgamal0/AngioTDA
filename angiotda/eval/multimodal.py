"""Shared evaluation for the multimodal encoders (late fusion and CLIP).

Both eval scripts embed every well through the encoder's `embed()` method and
report replicate consistency + per-MoA permutation enrichment, so the two models
are scored by identical code.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from angiotda.eval.metrics import load_label_map, load_moa, replicate_consistency, per_label_enrichment, summarize_enrichment
from angiotda.data.multimodal import build_dataset, build_eval_loader, batch_to_device


def embed_wells(encoder, dataset, device, batch_size=16):
    """Returns (well_embs, compound_embs), both L2-normalized."""
    loader = build_eval_loader(dataset, batch_size=batch_size)
    emb_accum = defaultdict(list)

    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            images, x, edge_index, edge_attr, node_mask = batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                z = encoder.embed(images, x, edge_index, edge_attr, node_mask)
            z = z.float().cpu().numpy()
            for i, c in enumerate(batch["compound"]):
                emb_accum[c].append(z[i])

    well_embs, compound_embs = {}, {}
    for c, vecs in emb_accum.items():
        for wi, v in enumerate(vecs):
            well_embs[(c, wi)] = v / max(np.linalg.norm(v), 1e-8)
        avg = np.mean(vecs, axis=0)
        compound_embs[c] = avg / max(np.linalg.norm(avg), 1e-8)
    return well_embs, compound_embs


def report(well_embs, compound_embs, label_map, label_name="Label", min_compounds=5, top_k=8):
    print(f"Compounds: {len(compound_embs)}, Wells: {len(well_embs)}")

    rc = replicate_consistency(well_embs)
    print("\nReplicate Consistency:")
    print(f"  Δ = {rc['delta']:+.4f}, AUROC = {rc['auroc']:.4f}")

    results = per_label_enrichment(compound_embs, label_map, min_compounds=min_compounds)
    summary = summarize_enrichment(results)
    print(f"\nCompound-label permutation {label_name} enrichment:")
    print(f"  {summary['n_bonferroni']} Bonferroni, {summary['n_nominal']} nominal / {summary['n_tested']} tested")

    for r in sorted(results, key=lambda x: x["p"])[:top_k]:
        star = "*" if r["p"] * max(summary["n_tested"], 1) < 0.05 else ""
        print(f"  {r['term']:<40} Δ={r['delta']:+.4f}  p={r['p']:.4f} {star}")

    return {"replicate_consistency": rc, "enrichment": results}


def load_eval_labels(project_root, label_json=None, label_key=None):
    if label_json is not None:
        label_path = Path(label_json)
        if not label_path.is_absolute():
            label_path = Path(project_root) / label_path
        return load_label_map(label_path, key=label_key)
    return load_moa("sailem57", project_root=Path(project_root))


def run_eval(encoder, ckpt, project_root, device, batch_size=16,
 label_json=None, label_key=None,
             label_name="Label", min_compounds=5):
    """Embed all wells with `encoder` and print the standard report."""
    ds = build_dataset(project_root, img_size=ckpt.get("img_size", 224))
    well_embs, compound_embs = embed_wells(encoder, ds, device, batch_size=batch_size)
    label_map = load_eval_labels(project_root, label_json=label_json, label_key=label_key)
    return report(
        well_embs,
        compound_embs,
        label_map,
        label_name=label_name,
        min_compounds=min_compounds,
    )
