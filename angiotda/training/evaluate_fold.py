"""Re-embed and evaluate one trained model only on its held-out fold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from angiotda import PROJECT_ROOT

from angiotda.data.dataset import GraphDatasetV2, collate_fn
from angiotda.data.paths import DATASET
from angiotda.eval.metrics import load_moa
from angiotda.training.common import (
    CompoundSubsetDataset,
    build_crossfit_encoder,
    collect_embeddings,
    evaluate_and_save,
    save_embedding_artifact,
)
from angiotda.training.train_fold import validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    split_path = args.split if args.split.is_absolute() else PROJECT_ROOT / args.split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("arch") != "biological_crossfit_graph_ssl":
        raise ValueError(f"not a biological cross-fit checkpoint: {checkpoint_path}")
    split = json.loads(split_path.read_text())
    fold = validate_split(split, checkpoint["fold"])
    if checkpoint["split_sha256"] != split["compound_partition_sha256"]:
        raise RuntimeError("checkpoint/split digest mismatch")

    dataset = CompoundSubsetDataset(
        GraphDatasetV2(DATASET.sample_index, PROJECT_ROOT / checkpoint["config"].get(
            "graphs_dir", "data/graphs")),
        fold["heldout_compounds"],
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    encoder = build_crossfit_encoder(
        checkpoint["architecture"], checkpoint["config"]).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    graph_ablation = checkpoint.get("config", {}).get("graph_ablation", "none")
    compounds, image_names, embeddings = collect_embeddings(
        encoder, loader, device, graph_ablation, checkpoint["seed"])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT / "reports" / "biological_crossfit"
            / (checkpoint["config"].get("run_name") or
               f"{checkpoint['architecture']}_seed{checkpoint['seed']}_fold{checkpoint['fold']}")
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    metadata = {
        "architecture": checkpoint["architecture"],
        "seed": checkpoint["seed"], "fold": checkpoint["fold"],
        "split_sha256": checkpoint["split_sha256"],
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "embedding_space": "fold_specific_do_not_pool_across_folds",
        "graph_ablation": graph_ablation,
        "graph_ablation_scope": "training_and_heldout_evaluation",
    }
    suffix = checkpoint_path.stem
    save_embedding_artifact(
        output_dir / f"heldout_embeddings_{suffix}.npz",
        compounds, image_names, embeddings, metadata)
    payload = evaluate_and_save(
        output_dir / f"heldout_evaluation_{suffix}.json",
        compounds, embeddings, load_moa("sailem57"),
        args.n_permutations, metadata)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
