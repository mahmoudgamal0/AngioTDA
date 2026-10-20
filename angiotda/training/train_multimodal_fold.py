"""Train one held-out DINO or four-view fold without compound leakage.

The checkpoint is selected by training loss only. Held-out target/pathway
retrieval is reported periodically for observability and is never used for
optimization, early stopping, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from angiotda import PROJECT_ROOT
from angiotda.training.common import format_optional

from angiotda.data.dataset import CompoundPairSampler  # noqa: E402
from angiotda.data.multimodal import (  # noqa: E402
    batch_to_device,
    build_dataset,
    collate_graph_image,
    seed_worker,
)
from angiotda.data.paths import DATASET  # noqa: E402
from angiotda.eval.metrics import load_offtargets  # noqa: E402
from angiotda.eval.retrieval import (  # noqa: E402
    cross_replicate_retrieval,
    profile_cosine,
    summarize_retrieval,
    tfidf_profile_vectors,
)
from angiotda.losses import NTXentLoss  # noqa: E402
from angiotda.models.image_encoder import ImageEncoder  # noqa: E402
from angiotda.models.fusion import LateFusionEncoder  # noqa: E402
from angiotda.training.common import (  # noqa: E402
    CompoundSubsetDataset,
    audit_physical_pairs,
    audit_training_batch,
    save_embedding_artifact,
)
from angiotda.training.train_fold import (  # noqa: E402
    atomic_torch_save,
    upsert_history,
    validate_split,
)
from angiotda.models.four_view_loss import FourViewObjective  # noqa: E402
from angiotda.models.four_view_model import FourViewEncoder  # noqa: E402
from angiotda.data.photometric import (  # noqa: E402
    PhotometricAugmentConfig,
    apply_photometric_augment,
)
from angiotda.eval.biology import (  # noqa: E402
    load_chembl37_human_reactome,
    neighborhood_rows,
    replicate_identity_rows,
    reconstruct_replicate_views,
)


ARCHITECTURES = ("dino_repeat", "four_view_repeat", "late_fusion_repeat")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_encoder(args):
    if args.architecture == "dino_repeat":
        return ImageEncoder(
            output_dim=args.embed_dim,
            model_type=args.model_type,
            freeze_backbone=False,
            input_mode="repeat",
        )
    if args.architecture == "late_fusion_repeat":
        return LateFusionEncoder(
            embed_dim=args.embed_dim,
            model_type=args.model_type,
            node_feat_dim=7,
            edge_feat_dim=8,
            n_layers=args.graph_layers,
            hidden_dim=args.graph_hidden,
            graph_weight=0.5,
            input_mode="repeat",
        )
    return FourViewEncoder(
        embed_dim=args.embed_dim,
        model_type=args.model_type,
        node_feat_dim=7,
        edge_feat_dim=8,
        n_layers=args.graph_layers,
        hidden_dim=args.graph_hidden,
        init_temperature=args.temperature,
        input_mode="repeat",
    )


def forward_embeddings(encoder, architecture: str, batch: dict, device):
    images, x, edge_index, edge_attr, node_mask = batch_to_device(batch, device)
    if architecture == "dino_repeat":
        z = encoder(images)
        return z, None, None
    if architecture == "late_fusion_repeat":
        return encoder.forward_components(
            images, x, edge_index, edge_attr, node_mask)
    z_image, z_graph = encoder(images, x, edge_index, edge_attr, node_mask)
    z_fused = torch.nn.functional.normalize(z_image + z_graph, dim=-1)
    return z_fused, z_image, z_graph


def collect_embeddings(encoder, architecture: str, loader, device):
    was_training = encoder.training
    encoder.eval()
    compounds, image_names, fused, image, graph = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z, z_image, z_graph = forward_embeddings(
                    encoder, architecture, batch, device)
            compounds.extend(batch["compound"])
            image_names.extend(batch["image_name"])
            fused.extend(z.float().cpu().numpy())
            if z_image is not None:
                image.extend(z_image.float().cpu().numpy())
                graph.extend(z_graph.float().cpu().numpy())
    encoder.train(was_training)
    return {
        "compounds": compounds,
        "image_names": image_names,
        "embeddings": np.asarray(fused, dtype=np.float32),
        "image_embeddings": (np.asarray(image, dtype=np.float32) if image else None),
        "graph_embeddings": (np.asarray(graph, dtype=np.float32) if graph else None),
    }


def quick_endpoint(first, second, raw_profiles):
    compounds = set(first) & set(second)
    profiles = tfidf_profile_vectors(raw_profiles, compounds=compounds)
    rows = cross_replicate_retrieval(
        first, second, profiles, (10,), profile_cosine)
    summary = summarize_retrieval(rows, n_bootstrap=0)
    return {
        "n_query_compounds": summary["n_query_compounds"],
        "ndcg_at_10": summary["metrics"].get("ndcg_at_10", {}).get("mean"),
    }


def periodic_biology(embeddings: dict, fold: int) -> dict:
    first, second, pair_audit = reconstruct_replicate_views(
        embeddings["compounds"], embeddings["image_names"],
        embeddings["embeddings"])
    stability = neighborhood_rows(first, second, fold)
    correlations = [row["rank_correlation"] for row in stability.values()]
    identity = replicate_identity_rows(first, second, fold)
    return {
        "heldout_compounds": len(first),
        "pair_audit": pair_audit,
        "neighbor_rank_correlation": (
            float(np.mean(correlations)) if correlations else None),
        "replicate_consistency": (
            float(np.mean([row["replicate_consistency"] for row in identity.values()]))
            if identity else None),
        "replicate_auroc": (
            float(np.mean([row["replicate_auroc"] for row in identity.values()]))
            if identity else None),
        "chembl_targets": quick_endpoint(
            first, second,
            load_offtargets(path=DATASET.index_dir / "offtargets_full291.json")),
        "chembl37_human_reactome": quick_endpoint(
            first, second, load_chembl37_human_reactome()),
    }


def render_periodic(path: Path, run_name: str, history: list[dict]) -> None:
    lines = [
        f"# Periodic held-out biology: {run_name}", "",
        "These metrics are monitoring only. They never select a checkpoint.", "",
        "| Epoch | Train loss | Replicate cosine | Replicate AUROC | Neighbour rho | Target queries | Target nDCG@10 | Reactome queries | Reactome nDCG@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda x: "—" if x is None else f"{x:.4f}"
    for row in history:
        biology = row["biology"]
        target = biology["chembl_targets"]
        reactome = biology["chembl37_human_reactome"]
        lines.append(
            f"| {row['epoch']} | {row['training_loss']:.4f} | "
            f"{fmt(biology.get('replicate_consistency'))} | "
            f"{fmt(biology.get('replicate_auroc'))} | "
            f"{fmt(biology['neighbor_rank_correlation'])} | "
            f"{target['n_query_compounds']} | {fmt(target['ndcg_at_10'])} | "
            f"{reactome['n_query_compounds']} | {fmt(reactome['ndcg_at_10'])} |")
    path.write_text("\n".join(lines) + "\n")


def checkpoint_payload(args, split, epoch, loss, best_loss, global_step,
                       encoder, optimizer, scheduler, scaler, sampler):
    return {
        "arch": "biological_crossfit_multimodal_ssl",
        "architecture": args.architecture,
        "fold": args.fold,
        "seed": args.seed,
        "split_seed": split["seed"],
        "split_sha256": split["compound_partition_sha256"],
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "global_step": global_step,
        "encoder_state": encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "sampler_rng_state": sampler._rng.getstate(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        "config": vars(args),
        "objective": {
            "dino_repeat": "replicate_nt_xent",
            "four_view_repeat": (
                "image_replicate_nt_xent + graph_replicate_nt_xent + "
                "four_view_multi_positive_infonce"),
            "late_fusion_repeat": (
                "replicate_nt_xent(normalize(0.5*normalized_dino + "
                "0.5*normalized_gine))"),
        }[args.architecture],
        "model_selection": "minimum_training_loss_only",
        "heldout_biology_used_for_model_selection": False,
    }


def train(args):
    set_seed(args.seed)
    if args.photometric_aug and args.architecture != "dino_repeat":
        raise ValueError("photometric augmentation ablation is DINO-only")
    if args.lr_image is None:
        args.lr_image = (1e-4 if args.architecture == "dino_repeat" else 1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_path = args.split if args.split.is_absolute() else PROJECT_ROOT / args.split
    split = json.loads(split_path.read_text())
    fold = validate_split(split, args.fold)

    photometric_cfg = PhotometricAugmentConfig(
        enabled=args.photometric_aug,
        gaussian_noise_std=args.gaussian_noise_std,
        shot_noise_scale=args.shot_noise_scale,
        intensity_scale_min=args.intensity_scale_min,
        intensity_scale_max=args.intensity_scale_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gradient_strength_min=args.gradient_strength_min,
        gradient_strength_max=args.gradient_strength_max,
    )
    image_transform = (functools.partial(
        apply_photometric_augment, cfg=photometric_cfg)
        if args.photometric_aug else None)
    base_dataset = build_dataset(
        PROJECT_ROOT, img_size=args.img_size, image_transform=image_transform)
    evaluation_dataset = (build_dataset(PROJECT_ROOT, img_size=args.img_size)
                          if args.photometric_aug else base_dataset)
    expected_universe = {
        compound for record in split["folds"]
        for compound in record["heldout_compounds"]
    }
    if set(base_dataset.compounds) != expected_universe:
        raise RuntimeError("split compound universe differs from multimodal dataset")
    train_dataset = CompoundSubsetDataset(base_dataset, fold["train_compounds"])
    heldout_dataset = CompoundSubsetDataset(
        evaluation_dataset, fold["heldout_compounds"])
    if set(train_dataset.compounds) & set(heldout_dataset.compounds):
        raise AssertionError("compound leakage after multimodal dataset construction")
    pair_audit = audit_physical_pairs(train_dataset)
    if pair_audit["unpairable_compounds"]:
        raise RuntimeError("unpairable training compounds")

    sampler = CompoundPairSampler(
        train_dataset, batch_compounds=args.batch_compounds, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=sampler, collate_fn=collate_graph_image,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        generator=generator, worker_init_fn=seed_worker)
    heldout_loader = torch.utils.data.DataLoader(
        heldout_dataset, batch_size=args.eval_batch_size, shuffle=False,
        collate_fn=collate_graph_image, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker)

    encoder = build_encoder(args).to(device)
    dino_objective = NTXentLoss(args.temperature)
    four_view_objective = FourViewObjective(
        temperature=args.temperature,
        image_weight=args.image_weight,
        graph_weight=args.graph_weight,
        cross_modal_weight=args.cross_modal_weight)
    parameters = [parameter for parameter in encoder.parameters()
                  if parameter.requires_grad]
    if args.architecture == "dino_repeat":
        parameter_groups = [{"params": parameters, "lr": args.lr_image}]
    else:
        parameter_groups = [
            {"params": encoder.image_encoder.parameters(), "lr": args.lr_image},
            {"params": encoder.graph_encoder.parameters(), "lr": args.lr_graph},
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(sampler)
    warmup = min(args.warmup_steps, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    run_name = args.run_name or (
        f"{args.architecture}_seed{args.seed}_fold{args.fold}")
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "biological_crossfit" / run_name
    report_dir = PROJECT_ROOT / "reports" / "biological_crossfit" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "split": str(split_path),
        "split_sha256": split["compound_partition_sha256"],
        "run_name": run_name,
        "training_pair_audit": pair_audit,
        "heldout_labels_used_for_optimization": False,
        "heldout_labels_used_for_model_selection": False,
        "photometric_augmentation": vars(photometric_cfg),
    }
    (report_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n")

    start_epoch, global_step, best_loss = 1, 0, float("inf")
    latest = checkpoint_dir / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        identity = (state["architecture"], state["seed"], state["fold"])
        if identity != (args.architecture, args.seed, args.fold):
            raise RuntimeError(f"checkpoint identity mismatch: {identity}")
        saved_aug = state.get("config", {}).get("photometric_aug", False)
        if saved_aug != args.photometric_aug:
            raise RuntimeError("checkpoint photometric-augmentation mismatch")
        if state["split_sha256"] != split["compound_partition_sha256"]:
            raise RuntimeError("checkpoint split mismatch")
        encoder.load_state_dict(state["encoder_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state") is not None:
            scaler.load_state_dict(state["scaler_state"])
        sampler._rng.setstate(state["sampler_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        # ``map_location=device`` also moves saved RNG tensors to CUDA, but
        # PyTorch's RNG restoration APIs require CPU ByteTensors.
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([
                rng_state.cpu() for rng_state in state["cuda_rng_state"]
            ])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        best_loss = state["best_loss"]
    elif latest.exists():
        raise FileExistsError(f"existing run at {latest}; pass --resume")

    periodic_path = report_dir / "periodic_biology.json"
    periodic_history = (json.loads(periodic_path.read_text())
                        if periodic_path.exists() else [])
    state = None
    for epoch in range(start_epoch, args.epochs + 1):
        encoder.train()
        loss_total, n_batches = 0.0, 0
        for batch in train_loader:
            audit_training_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=scaler is not None):
                fused, image, graph = forward_embeddings(
                    encoder, args.architecture, batch, device)
                if args.architecture in {"dino_repeat", "late_fusion_repeat"}:
                    result = dino_objective(fused, batch["compound"])
                else:
                    result = four_view_objective(image, graph, batch["compound"])
                loss = result["loss"]
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                optimizer.step()
            scheduler.step()
            global_step += 1
            loss_total += float(loss.detach())
            n_batches += 1
        average_loss = loss_total / max(n_batches, 1)
        upsert_history(report_dir / "train_history.jsonl", {
            "epoch": epoch,
            "loss": average_loss,
            "lr": scheduler.get_last_lr()[0],
            "n_batches": n_batches,
        })
        best_loss = min(best_loss, average_loss)
        state = checkpoint_payload(
            args, split, epoch, average_loss, best_loss, global_step,
            encoder, optimizer, scheduler, scaler, sampler)
        atomic_torch_save(state, latest)
        if average_loss <= best_loss:
            atomic_torch_save(state, checkpoint_dir / "best.pt")

        should_evaluate = (epoch == 1 or epoch == args.epochs or
                           (args.eval_every > 0 and epoch % args.eval_every == 0))
        if should_evaluate:
            heldout = collect_embeddings(
                encoder, args.architecture, heldout_loader, device)
            biology = periodic_biology(heldout, args.fold)
            row = {"epoch": epoch, "training_loss": average_loss,
                   "biology": biology}
            periodic_history = [item for item in periodic_history
                                if item["epoch"] != epoch]
            periodic_history.append(row)
            periodic_history.sort(key=lambda item: item["epoch"])
            periodic_path.write_text(json.dumps(periodic_history, indent=2) + "\n")
            render_periodic(
                report_dir / "periodic_biology.md", run_name, periodic_history)
            print(
                f"{run_name} epoch={epoch} loss={average_loss:.5f} "
                f"rho={format_optional(biology['neighbor_rank_correlation'])} "
                f"target_ndcg10={format_optional(biology['chembl_targets']['ndcg_at_10'])} "
                f"reactome_ndcg10={format_optional(biology['chembl37_human_reactome']['ndcg_at_10'])}",
                flush=True)

    if state is None:
        state = torch.load(latest, map_location=device, weights_only=False)
        encoder.load_state_dict(state["encoder_state"])
    atomic_torch_save(state, checkpoint_dir / "final.pt")
    heldout = collect_embeddings(encoder, args.architecture, heldout_loader, device)
    metadata = {
        "architecture": args.architecture,
        "seed": args.seed,
        "fold": args.fold,
        "split_sha256": split["compound_partition_sha256"],
        "checkpoint": str((checkpoint_dir / "final.pt").relative_to(PROJECT_ROOT)),
        "embedding_space": "fold_specific_do_not_pool_across_folds",
        "input_mode": "repeat_grayscale_to_rgb_no_channel_adapter",
        "model_selection": "minimum_training_loss_only",
        "heldout_biology_used_for_model_selection": False,
        "photometric_augmentation": vars(photometric_cfg),
    }
    save_embedding_artifact(
        report_dir / "heldout_embeddings_final.npz",
        heldout["compounds"], heldout["image_names"],
        heldout["embeddings"], metadata)
    if heldout["image_embeddings"] is not None:
        save_embedding_artifact(
            report_dir / "heldout_image_embeddings_final.npz",
            heldout["compounds"], heldout["image_names"],
            heldout["image_embeddings"], {**metadata, "view": "image"})
        save_embedding_artifact(
            report_dir / "heldout_graph_embeddings_final.npz",
            heldout["compounds"], heldout["image_names"],
            heldout["graph_embeddings"], {**metadata, "view": "graph"})
    return report_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=Path, default=Path(
        "splits/compound_folds_seed42.json"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-compounds", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--model-type", default="dinov3_vits16")
    parser.add_argument("--graph-layers", type=int, default=4)
    parser.add_argument("--graph-hidden", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--graph-weight", type=float, default=1.0)
    parser.add_argument("--cross-modal-weight", type=float, default=1.0)
    parser.add_argument(
        "--lr-image", type=float,
        help="Defaults to 1e-4 for DINO-only and 1e-5 for four-view training.")
    parser.add_argument("--lr-graph", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--photometric-aug", action="store_true")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.03)
    parser.add_argument("--shot-noise-scale", type=float, default=48.0)
    parser.add_argument("--intensity-scale-min", type=float, default=0.8)
    parser.add_argument("--intensity-scale-max", type=float, default=1.2)
    parser.add_argument("--gamma-min", type=float, default=0.8)
    parser.add_argument("--gamma-max", type=float, default=1.2)
    parser.add_argument("--gradient-strength-min", type=float, default=0.0)
    parser.add_argument("--gradient-strength-max", type=float, default=0.35)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
